"""Autonomous Neural Engineer (ANE).

ANE turns an engineering objective into an executable plan, generates candidates,
searches low-dimensional transformation parameters with derivative-free methods,
evaluates candidates against a stated metric, diagnoses failures and retries with
alternative strategies, and returns the verified best artifact plus limitations.

Determinism: given the same request, inputs and seed, ANE produces the same plan
and the same selected candidate. LLM-assisted planning is available through
:class:`uzen.core.ane.llm.LLMPlanner`, whose proposals are validated by the same
registry/policy gates — an LLM can propose, but it cannot authorize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from uzen.core.ane.search import SEARCH_ALGORITHMS
from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.core.nwpc.registry import OperationRecord, SearchRequest, default_registry
from uzen.core.vel.verify import VerificationReport, verify
from uzen.errors import (
    BudgetExhausted,
    InvalidOperationParameters,
    NoValidAlignment,
    NoValidStrategy,
    UzenError,
)
from uzen.policies.strict_plus.effects import EngineEffect, TaskScope
from uzen.policies.strict_plus.engine import StrictPlusEngine, default_engine
from uzen.util.canonical import utc_now_iso
from uzen.util.rng import parse_seed

KEYWORDS: dict[str, tuple[str, ...]] = {
    "merge": ("merge", "blend", "combine", "soup", "fuse", "ties", "dare"),
    "align": ("align", "permut", "procrustes", "basis", "match"),
    "compress": ("compress", "low-rank", "low rank", "svd", "factor", "sparse", "prune"),
    "generate": ("generate", "initialize", "init", "synthesize weights", "new weights"),
    "reconstruct": ("reconstruct", "fill", "complete", "repair", "rebuild weights"),
    "modify": ("scale", "offset", "interpolate", "extrapolate", "perturb", "transform"),
    "architecture": ("architecture", "topology", "graph", "operator", "layer", "depth", "width"),
}


@dataclass
class JobRequest:
    task_scope: TaskScope
    objective: str
    inputs: list[dict[str, Any]] = field(default_factory=list)
    operations: list[dict[str, Any]] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    policy: str = "STRICT_PLUS"
    evaluation_plan: dict[str, Any] = field(default_factory=dict)
    output_format: str = "safetensors"
    compute_requirements: dict[str, Any] = field(default_factory=dict)
    seed: int | str | None = 0
    budget: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    def input_names(self) -> list[str]:
        return [str(i.get("name", f"input{idx}")) for idx, i in enumerate(self.inputs)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_scope": self.task_scope.value,
            "objective": self.objective,
            "inputs": self.inputs,
            "operations": self.operations,
            "constraints": self.constraints,
            "policy": self.policy,
            "evaluation_plan": self.evaluation_plan,
            "output_format": self.output_format,
            "compute_requirements": self.compute_requirements,
            "seed": self.seed,
            "budget": self.budget,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobRequest":
        scope_value = str(data.get("task_scope", "inspection"))
        try:
            scope = TaskScope(scope_value)
        except ValueError as exc:
            raise InvalidOperationParameters(
                f"unknown task_scope '{scope_value}'; valid: {[s.value for s in TaskScope]}"
            ) from exc
        policy = str(data.get("policy", "STRICT_PLUS"))
        if policy != "STRICT_PLUS":
            raise InvalidOperationParameters(
                f"unsupported policy '{policy}': UZEN only executes under STRICT_PLUS"
            )
        return cls(
            task_scope=scope,
            objective=str(data.get("objective", "")),
            inputs=[dict(i) for i in data.get("inputs", [])],
            operations=[dict(o) for o in data.get("operations", [])],
            constraints=dict(data.get("constraints") or {}),
            policy=policy,
            evaluation_plan=dict(data.get("evaluation_plan") or {}),
            output_format=str(data.get("output_format", "safetensors")),
            compute_requirements=dict(data.get("compute_requirements") or {}),
            seed=data.get("seed", 0),
            budget=dict(data.get("budget") or {}),
            options=dict(data.get("options") or {}),
        )


@dataclass
class PlanStage:
    name: str
    op_code: str
    input_names: list[str]
    fixed_params: dict[str, Any] = field(default_factory=dict)
    search_space: list[tuple[str, float, float]] = field(default_factory=list)
    description: str = ""
    effect: EngineEffect = EngineEffect.PARAM_TRANSFORM
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op_code": self.op_code,
            "input_names": self.input_names,
            "fixed_params": self.fixed_params,
            "search_space": [
                {"param": p, "low": lo, "high": hi} for p, lo, hi in self.search_space
            ],
            "description": self.description,
            "effect": self.effect.value,
            "alternatives": self.alternatives,
        }


@dataclass
class EngineeringPlan:
    request: JobRequest
    stages: list[PlanStage]
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    search_algorithm: str = "coordinate_descent"
    evaluation: dict[str, Any] = field(default_factory=dict)
    candidate_budget: int = 12
    evaluations_per_stage: int = 24

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.request.objective,
            "scope": self.request.task_scope.value,
            "stages": [s.to_dict() for s in self.stages],
            "rationale": self.rationale,
            "warnings": self.warnings,
            "search_algorithm": self.search_algorithm,
            "evaluation": self.evaluation,
            "candidate_budget": self.candidate_budget,
            "evaluations_per_stage": self.evaluations_per_stage,
            "policy": "STRICT_PLUS",
        }


@dataclass
class Candidate:
    name: str
    op_code: str
    params: dict[str, Any]
    metric: float | None
    ok: bool
    package: WeightPackage | None = None
    record: OperationRecord | None = None
    problems: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op_code": self.op_code,
            "params": self.params,
            "metric": self.metric,
            "ok": self.ok,
            "problems": self.problems,
            "details": self.details,
        }


@dataclass
class ANEResult:
    request: JobRequest
    plan: EngineeringPlan
    candidates: list[Candidate]
    best: Candidate | None
    package: WeightPackage | None
    verification: VerificationReport | None
    trace: list[dict[str, Any]]
    policy: list[dict[str, Any]]
    seeds: dict[str, int]
    notes: list[str]
    warnings: list[str]
    diagnosis: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.request.objective,
            "scope": self.request.task_scope.value,
            "plan": self.plan.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "best_candidate": self.best.to_dict() if self.best else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "trace": self.trace,
            "policy": self.policy,
            "seeds": self.seeds,
            "notes": self.notes,
            "warnings": self.warnings,
            "diagnosis": self.diagnosis,
            "created_at": self.created_at,
            "produced_package": {
                "family": self.package.family,
                "num_tensors": len(self.package),
                "numel": self.package.numel(),
                "fingerprint": self.package.fingerprint(),
            }
            if self.package
            else None,
        }


# --------------------------------------------------------------------------- #
# objectives
# --------------------------------------------------------------------------- #


class Objective:
    """Evaluation metric for candidates. Higher is always better."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        *,
        base: WeightPackage | None = None,
        target: WeightPackage | None = None,
        graph: Graph | None = None,
        inputs: Mapping[str, np.ndarray] | None = None,
        reference: WeightPackage | None = None,
    ) -> None:
        self.name = str(spec.get("metric", "distance_to_target"))
        self.weight = float(spec.get("weight", 1.0))
        self.base = base
        self.target = target
        self.graph = graph
        self.inputs = inputs
        self.reference = reference
        self.tolerance = float(spec.get("tolerance", 1e-6))
        if self.name == "distance_to_target" and target is None:
            raise InvalidOperationParameters(
                "evaluation metric 'distance_to_target' requires a target package"
            )
        if self.name == "activation_agreement" and (graph is None or inputs is None or reference is None):
            raise InvalidOperationParameters(
                "evaluation metric 'activation_agreement' requires graph, inputs and a reference package"
            )

    def evaluate(self, package: WeightPackage) -> float:
        if self.name == "distance_to_target":
            assert self.target is not None
            shared = [n for n in package.names() if n in self.target and package.tensors[n].shape == self.target.tensors[n].shape]
            if not shared:
                return -1e9
            num = sum(float(np.sum((package.tensors[n].astype(np.float64) - self.target.tensors[n].astype(np.float64)) ** 2)) for n in shared)
            den = sum(float(np.sum(self.target.tensors[n].astype(np.float64) ** 2)) for n in shared) or 1.0
            return float(1.0 - np.sqrt(num / den))
        if self.name == "activation_agreement":
            assert self.graph is not None and self.inputs is not None and self.reference is not None
            try:
                a = execute(self.graph, self.inputs, package, check_finite=False)
                b = execute(self.graph, self.inputs, self.reference, check_finite=False)
            except Exception:
                return -1e9
            scores = []
            for name, arr in a.outputs.items():
                if name not in b.outputs or b.outputs[name].shape != arr.shape:
                    continue
                ref = b.outputs[name].astype(np.float64)
                cur = arr.astype(np.float64)
                den = float(np.linalg.norm(ref)) or 1.0
                scores.append(1.0 - float(np.linalg.norm(cur - ref) / den))
            return float(np.mean(scores)) if scores else -1e9
        if self.name == "sparsity":
            total = 0
            zeros = 0
            for name in package.names():
                arr = package.tensors[name]
                total += arr.size
                zeros += int(np.count_nonzero(np.abs(arr) < 1e-12))
            return float(zeros / total) if total else 0.0
        if self.name == "parameter_efficiency":
            return -float(package.numel())
        if self.name == "distance_from_base":
            if self.base is None:
                raise InvalidOperationParameters("distance_from_base requires a base package")
            shared = [n for n in package.names() if n in self.base]
            num = sum(float(np.sum((package.tensors[n].astype(np.float64) - self.base.tensors[n].astype(np.float64)) ** 2)) for n in shared)
            den = sum(float(np.sum(self.base.tensors[n].astype(np.float64) ** 2)) for n in shared) or 1.0
            return float(np.sqrt(num / den))
        if self.name == "finite":
            for name in package.names():
                arr = package.tensors[name]
                if arr.dtype.kind in "fc" and not np.isfinite(arr).all():
                    return -1e9
            return 1.0
        raise InvalidOperationParameters(
            f"unknown evaluation metric '{self.name}'; available: distance_to_target, "
            "activation_agreement, sparsity, parameter_efficiency, distance_from_base, finite"
        )


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #


class AutonomousNeuralEngineer:
    def __init__(self, *, engine: StrictPlusEngine | None = None) -> None:
        self.engine = engine or default_engine()
        self.registry = default_registry()

    # ------------------------------------------------------------------ #
    # planning
    # ------------------------------------------------------------------ #

    def _keywords(self, text: str) -> set[str]:
        low = text.lower()
        found: set[str] = set()
        for key, words in KEYWORDS.items():
            if any(w in low for w in words):
                found.add(key)
        return found

    def plan(
        self,
        request: JobRequest,
        *,
        input_names: Sequence[str] | None = None,
    ) -> EngineeringPlan:
        names = list(input_names or request.input_names())
        if not names and request.task_scope not in (
            TaskScope.MODEL_CREATION,
            TaskScope.ARCHITECTURE_SYNTHESIS,
            TaskScope.MODEL_ARCHITECTURE_RESEARCH,
        ):
            raise InvalidOperationParameters(
                "the request supplies no inputs; weight engineering needs at least one package"
            )
        budget = request.budget or {}
        algorithm = str(budget.get("search_algorithm", "coordinate_descent"))
        if algorithm not in SEARCH_ALGORITHMS:
            raise InvalidOperationParameters(
                f"unknown search_algorithm '{algorithm}'; available: {sorted(SEARCH_ALGORITHMS)}"
            )
        candidate_budget = int(budget.get("candidates", 12))
        evaluations = int(budget.get("evaluations", 24))
        if candidate_budget < 1 or evaluations < 1:
            raise InvalidOperationParameters("budget values must be positive")

        stages: list[PlanStage] = []
        rationale: list[str] = []
        warnings: list[str] = []

        if request.operations:
            for idx, op in enumerate(request.operations):
                code = str(op.get("code") or op.get("op") or "")
                if not code:
                    raise InvalidOperationParameters(f"operation #{idx} has no 'code'")
                spec = self.registry.get(code)
                args = [str(a) for a in op.get("args", [])]
                if not args:
                    # Bind the operation's declared input arity to the request inputs in
                    # order, so binary operations (alignment, merging) receive both
                    # packages. A later stage can reference an earlier stage output by
                    # name through an explicit "args" list.
                    wanted = max(spec.min_inputs, min(spec.max_inputs, len(names)))
                    args = names[: max(0, wanted)]
                stages.append(
                    PlanStage(
                        name=f"op{idx}",
                        op_code=code,
                        input_names=args,
                        fixed_params=dict(op.get("params") or {}),
                        search_space=[],
                        description=f"explicitly requested operation {code}",
                        effect=spec.effect,
                    )
                )
            rationale.append("plan uses the explicitly requested operations in order")
        else:
            kws = self._keywords(request.objective)
            scope = request.task_scope
            if scope.is_architecture_scope or "architecture" in kws:
                rationale.append(
                    "the objective targets architecture engineering; weight-only transforms are "
                    "not substituted for structural work"
                )
                stages = [
                    PlanStage(
                        name="architecture",
                        op_code="<nase>",
                        input_names=names,
                        description="architecture synthesis/research (handled by NASE/MAR)",
                        effect=EngineEffect.ARCH_MUTATION,
                    )
                ]
            elif "generate" in kws or scope == TaskScope.WEIGHT_GENERATION:
                method = str(request.options.get("method", "constraint_matched" if len(names) > 1 else "orthogonal"))
                code = {
                    "orthogonal": "w.gen.structured",
                    "structured_random": "w.gen.structured",
                    "constraint_matched": "w.gen.constraint_matched",
                    "low_rank": "w.gen.low_rank_regeneration",
                    "from_factors": "w.gen.from_factors",
                }.get(method, "w.gen.structured")
                stages = [
                    PlanStage(
                        name="generate",
                        op_code=code,
                        input_names=names[:2],
                        fixed_params=dict(request.options.get("params") or {}),
                        description=f"generate parameters with method '{method}'",
                    )
                ]
                rationale.append(
                    "no pretrained transformation can produce the requested parameters; "
                    "generation will be used and labelled as non-learned"
                )
            elif "reconstruct" in kws or scope == TaskScope.WEIGHT_RECONSTRUCTION:
                stages = [
                    PlanStage(
                        name="reconstruct",
                        op_code="w.recon.fill_missing",
                        input_names=names,
                        fixed_params={"allow_random_fill": bool(request.options.get("allow_random_fill", False))},
                        description="fill missing parameters from donors",
                    )
                ]
            elif "merge" in kws and len(names) >= 2:
                stages = [
                    PlanStage(
                        name="merge",
                        op_code="w.merge.task_arithmetic",
                        input_names=names,
                        fixed_params={},
                        search_space=[(f"weight_{i}", -1.0, 1.0) for i in range(len(names) - 1)],
                        description="search a task-arithmetic mixture over the supplied models",
                        alternatives=["w.merge.soup", "w.merge.ties", "w.merge.dare", "w.merge.disjoint"],
                    ),
                    PlanStage(
                        name="align",
                        op_code="w.align.auto",
                        input_names=[names[0]],
                        fixed_params={"threshold": 0.5},
                        description="verify/establish parameter correspondence before arithmetic",
                    ),
                ]
                rationale.append(
                    "merging across checkpoints is only defensible after alignment; NPA runs first "
                    "when structures or lineages differ"
                )
            elif "compress" in kws:
                stages = [
                    PlanStage(
                        name="compress",
                        op_code="w.decomp.svd_truncate",
                        input_names=names[:1],
                        fixed_params={},
                        search_space=[("energy", 0.5, 1.0)],
                        description="search the low-rank energy target",
                        alternatives=["w.decomp.sparsify", "w.decomp.factorize"],
                    )
                ]
            elif "align" in kws and len(names) >= 2:
                stages = [
                    PlanStage(
                        name="align",
                        op_code="w.align.auto",
                        input_names=names[:2],
                        fixed_params={"threshold": float(request.options.get("threshold", 0.5))},
                        description="establish parameter correspondence",
                        alternatives=["w.align.procrustes", "w.align.permutation", "w.align.basis"],
                    )
                ]
            else:
                stages = [
                    PlanStage(
                        name="modify",
                        op_code="w.arithmetic.linear_interpolate",
                        input_names=names[:2],
                        fixed_params={},
                        search_space=[("t", 0.0, 1.0)],
                        description="search an interpolation path between the supplied packages",
                        alternatives=["w.arithmetic.slerp", "w.arithmetic.extrapolate"],
                    )
                ]
                rationale.append(
                    "the objective is open-ended; a parameterized interpolation search is the "
                    "most conservative executable interpretation"
                )
            if not rationale:
                rationale.append("strategy selected from objective keywords and task scope")

        evaluation = {
            "metric": request.evaluation_plan.get("metric", "finite"),
            "weight": request.evaluation_plan.get("weight", 1.0),
            **{k: v for k, v in request.evaluation_plan.items() if k not in ("metric", "weight")},
        }
        if len(names) >= 2 and evaluation.get("metric") == "finite":
            evaluation["metric"] = "distance_from_base"
            warnings.append(
                "no evaluation metric was supplied; using distance_from_base (a structural "
                "proxy, not a task-quality measure)"
            )
        return EngineeringPlan(
            request=request,
            stages=stages,
            rationale=rationale,
            warnings=warnings,
            search_algorithm=algorithm,
            evaluation=evaluation,
            candidate_budget=candidate_budget,
            evaluations_per_stage=evaluations,
        )

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #

    def execute(
        self,
        request: JobRequest,
        inputs: Mapping[str, WeightPackage],
        *,
        target: WeightPackage | None = None,
        graph: Graph | None = None,
        graph_inputs: Mapping[str, np.ndarray] | None = None,
        reference: WeightPackage | None = None,
        job_id: str | None = None,
    ) -> ANEResult:
        if request.task_scope.is_architecture_scope:
            raise NoValidStrategy(
                "architecture-scoped jobs are executed by NASE/MAR, not by the weight "
                "transformation planner",
                hint="use /v1/architectures/synthesize or /v1/architectures/research",
            )
        plan = self.plan(request, input_names=list(inputs))
        base = inputs.get(plan.stages[0].input_names[0]) if plan.stages else None
        objective = Objective(
            plan.evaluation,
            base=base,
            target=target,
            graph=graph,
            inputs=graph_inputs,
            reference=reference,
        )
        seed = parse_seed(request.seed)
        env: dict[str, WeightPackage] = dict(inputs)
        candidates: list[Candidate] = []
        trace: list[dict[str, Any]] = []
        policy: list[dict[str, Any]] = []
        seeds: dict[str, int] = {}
        notes: list[str] = []
        warnings: list[str] = list(plan.warnings)
        diagnosis: list[str] = []

        current_name = plan.stages[0].input_names[0] if plan.stages else None
        best_overall: Candidate | None = None
        stage_index = 0
        while stage_index < len(plan.stages):
            stage = plan.stages[stage_index]
            try:
                package, record, stage_candidates, stage_diag = self._run_stage(
                    stage,
                    stage_index,
                    env,
                    objective,
                    scope=request.task_scope,
                    seed=seed,
                    algorithm=plan.search_algorithm,
                    budget=plan.evaluations_per_stage,
                    job_id=job_id,
                )
                diagnosis.extend(stage_diag)
            except NoValidAlignment as exc:
                alternative = self._next_alignment_alternative(stage, exc, diagnosis)
                if alternative is None:
                    # The domain-level answer is that no correspondence exists; do not
                    # degrade it into a generic planning failure, and never fabricate
                    # aligned weights to make the job succeed.
                    raise NoValidAlignment(
                        "every alignment strategy failed; no defensible parameter "
                        "correspondence exists for these inputs",
                        stage=stage.op_code,
                        attempts=diagnosis,
                    ) from exc
                plan.stages.insert(
                    stage_index,
                    PlanStage(
                        name=f"{stage.name}_retry",
                        op_code=alternative,
                        input_names=stage.input_names,
                        fixed_params={**stage.fixed_params, "threshold": 0.3, "allow_partial": False},
                        description=f"retry with alternative alignment '{alternative}' after failure",
                    ),
                )
                diagnosis.append(f"retrying alignment with '{alternative}'")
                continue
            except UzenError as exc:
                diagnosis.append(f"stage '{stage.name}' failed: {exc}")
                raise
            candidates.extend(stage_candidates)
            assert package is not None and record is not None
            env[stage.name] = package
            current_name = stage.name
            if best_overall is None or (stage_candidates and stage_candidates[-1].metric is not None):
                metric_value = stage_candidates[-1].metric if stage_candidates else None
                best_overall = Candidate(
                    name=stage.name,
                    op_code=stage.op_code,
                    params=record.params_used,
                    metric=metric_value,
                    ok=True,
                    package=package,
                    record=record,
                )
            trace.append(record.to_dict())
            policy.extend(record.policy)
            seeds[stage.name] = record.seed
            stage_index += 1
        if best_overall is None or best_overall.package is None:
            raise NoValidStrategy("the plan produced no candidate artifact", diagnosis=diagnosis)

        verification = None
        if graph is not None and graph_inputs is not None:
            verification = verify(
                best_overall.package,
                reference=reference,
                graph=graph,
                inputs=graph_inputs,
                policy_records=policy,
                operations=trace,
            )
            if not verification.ok:
                warnings.append(
                    "verification reported problems: " + "; ".join(verification.problems[:5])
                )
        else:
            verification = verify(
                best_overall.package,
                reference=reference,
                policy_records=policy,
                operations=trace,
                measure_performance=False,
            )
            notes.append(
                "functional behaviour was not evaluated because no executable graph/inputs "
                "were supplied"
            )
        notes.append(
            "all transformations are closed-form or derivative-free; no gradient was computed"
        )
        return ANEResult(
            request=request,
            plan=plan,
            candidates=candidates,
            best=best_overall,
            package=best_overall.package,
            verification=verification,
            trace=trace,
            policy=policy,
            seeds=seeds,
            notes=notes,
            warnings=warnings,
            diagnosis=diagnosis,
        )

    # ------------------------------------------------------------------ #

    def _next_alignment_alternative(self, stage: PlanStage, exc: Exception, diagnosis: list[str]) -> str | None:
        """Next alignment strategy that has not already been attempted."""
        order = [
            "w.align.procrustes",
            "w.align.basis",
            "w.align.permutation",
            "w.align.linear_map",
            "w.align.identity",
        ]
        tried: set[str] = set(stage.alternatives)
        for message in diagnosis:
            for code in order:
                if code in message:
                    tried.add(code)
        tried.add(stage.op_code)
        for code in order:
            if code not in tried:
                return code
        return None

    def _run_stage(
        self,
        stage: PlanStage,
        index: int,
        env: Mapping[str, WeightPackage],
        objective: Objective,
        *,
        scope: TaskScope,
        seed: int,
        algorithm: str,
        budget: int,
        job_id: str | None,
    ) -> tuple[WeightPackage, OperationRecord, list[Candidate], list[str]]:
        spec = self.registry.get(stage.op_code)
        args = []
        for name in stage.input_names:
            if name not in env:
                raise InvalidOperationParameters(
                    f"stage '{stage.name}' requires package '{name}' which is not available"
                )
            args.append(env[name])
        candidates: list[Candidate] = []
        diagnosis: list[str] = []

        if not stage.search_space:
            params = self.registry.get(stage.op_code).validate_params(stage.fixed_params)
            metric = None
            try:
                metric = objective.evaluate(args[0]) if args else None
            except Exception as exc:
                diagnosis.append(f"objective evaluation failed before transform: {exc}")
            record = self.registry.invoke(
                stage.op_code, args, params, scope=scope, seed=seed + index, engine=self.engine, job_id=job_id
            )
            try:
                metric = objective.evaluate(record.output)
            except Exception as exc:
                diagnosis.append(f"objective evaluation failed on candidate: {exc}")
            candidates.append(
                Candidate(
                    name=stage.name,
                    op_code=stage.op_code,
                    params=record.params_used,
                    metric=metric,
                    ok=True,
                    package=record.output,
                    record=record,
                    details=record.details,
                )
            )
            return record.output, record, candidates, diagnosis

        # Parameterized stage: derivative-free search over the declared space.
        names_in_space = [p for p, _, _ in stage.search_space]
        bounds = [(lo, hi) for _, lo, hi in stage.search_space]

        def params_for(values: np.ndarray) -> dict[str, Any]:
            params = dict(stage.fixed_params)
            for name, value in zip(names_in_space, values):
                pspec = spec.params.get(name)
                if pspec is not None and pspec.type == "int":
                    params[name] = int(round(float(value)))
                else:
                    params[name] = float(value)
            if stage.op_code.startswith("w.merge.") and names_in_space and names_in_space[0].startswith("weight_"):
                weights = [float(v) for v in values]
                params.pop("weight_0", None)
                params["weights"] = weights
                for n in names_in_space:
                    params.pop(n, None)
            return params

        def objective_fn(values: np.ndarray) -> float:
            try:
                params = params_for(values)
            except Exception:
                return -1e9
            try:
                record = self.registry.invoke(
                    stage.op_code, args, params, scope=scope, seed=seed + index, engine=self.engine, job_id=job_id
                )
            except Exception as exc:
                diagnosis.append(f"candidate rejected: {exc}")
                return -1e9
            try:
                metric = objective.evaluate(record.output)
            except Exception as exc:
                diagnosis.append(f"objective failure: {exc}")
                return -1e9
            candidates.append(
                Candidate(
                    name=f"{stage.name}_{len(candidates)}",
                    op_code=stage.op_code,
                    params=params,
                    metric=metric,
                    ok=True,
                    package=record.output,
                    record=record,
                )
            )
            return metric

        search_fn = SEARCH_ALGORITHMS[algorithm]
        request = SearchRequest(
            objective=objective_fn,
            bounds=bounds,
            budget=budget,
            seed=seed + index * 104729,
            maximize=True,
            options=dict(self.plan_options(stage)),
        )
        result, _decision = self.registry.invoke_search(
            f"w.search.{algorithm}", request, scope=scope, engine=self.engine
        )
        best_params = params_for(result.best_params)
        record = self.registry.invoke(
            stage.op_code, args, best_params, scope=scope, seed=seed + index, engine=self.engine, job_id=job_id
        )
        ok_candidates = [c for c in candidates if c.metric is not None]
        if ok_candidates:
            best = max(ok_candidates, key=lambda c: c.metric or -1e9)
            if best.metric is not None and (best.metric < -1e8 or best.metric < float(self.plan_options(stage).get("min_metric", -1e8))):
                diagnosis.append(
                    f"best candidate metric {best.metric:.4g} is degenerate; stage may have failed"
                )
        if not candidates:
            raise NoValidStrategy(
                f"search for stage '{stage.name}' produced no evaluable candidate",
                diagnosis=diagnosis,
            )
        diagnosis.append(
            f"searched {result.evaluations} candidates with {algorithm}; best metric "
            f"{result.best_value:.6g}"
        )
        return record.output, record, candidates, diagnosis

    def plan_options(self, stage: PlanStage) -> dict[str, Any]:
        return {"steps": 5, "beam_width": 3, "min_metric": -1e8}
