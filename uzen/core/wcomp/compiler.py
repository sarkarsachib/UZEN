"""Weight Compiler (WComp).

Turns a declarative engineering program (a list of stages) into a validated,
policy-checked execution plan, then executes it through the operation registry.
Also produces runtime compatibility reports for NIR graphs bound to parameter
packages, which is what tells a caller *exactly* why a graph cannot run.

Nothing here simulates execution: ``CompiledPipeline.execute`` performs the real
tensor math via registered operations, or raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.core.nwpc.registry import (
    OperationRecord,
    OperationRegistry,
    default_registry,
)
from uzen.errors import (
    GraphValidationError,
    OperationNotRegistered,
    OperationUnavailable,
    ProgramInvalid,
)
from uzen.policies.strict_plus.effects import EngineEffect, TaskScope
from uzen.policies.strict_plus.engine import StrictPlusEngine, default_engine
from uzen.util.rng import parse_seed

SCHEMA = "uzen.execution_plan/1"


@dataclass
class Stage:
    name: str
    code: str
    args: list[str]
    params: dict[str, Any]
    title: str = ""


@dataclass
class CompiledStage:
    name: str
    code: str
    args: list[str]
    params: dict[str, Any]
    effect: EngineEffect
    title: str = ""
    maturity: str = "implemented"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "code": self.code,
            "args": list(self.args),
            "params": self.params,
            "effect": self.effect.value,
            "maturity": self.maturity,
            "title": self.title,
        }


@dataclass
class CompilationReport:
    ok: bool
    stages: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)

    def require(self) -> None:
        if not self.ok:
            raise ProgramInvalid("compilation failed", problems=self.problems[:20])

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stages": self.stages,
            "problems": self.problems,
            "required_inputs": self.required_inputs,
            "outputs": self.outputs,
            "effects": self.effects,
            "schema": SCHEMA,
        }


@dataclass
class PipelineResult:
    outputs: dict[str, WeightPackage]
    trace: list[OperationRecord]
    policy: list[dict[str, Any]]
    seeds: dict[str, int]

    def package(self, name: str | None = None) -> WeightPackage:
        if name is None:
            if len(self.outputs) == 1:
                return next(iter(self.outputs.values()))
            if "weights" in self.outputs:
                return self.outputs["weights"]
            raise ProgramInvalid(
                f"pipeline produced multiple outputs ({sorted(self.outputs)}); name one explicitly"
            )
        try:
            return self.outputs[name]
        except KeyError as exc:
            raise ProgramInvalid(
                f"no pipeline output named '{name}'; available: {sorted(self.outputs)}"
            ) from exc

    def operations(self) -> list[dict[str, Any]]:
        return [rec.to_dict() for rec in self.trace]


class CompiledPipeline:
    def __init__(
        self,
        stages: Sequence[CompiledStage],
        *,
        required_inputs: Sequence[str],
        scope: TaskScope,
        engine: StrictPlusEngine,
        registry: OperationRegistry,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.stages = list(stages)
        self.required_inputs = list(required_inputs)
        self.scope = scope
        self.engine = engine
        self.registry = registry
        self.metadata = dict(metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "scope": self.scope.value,
            "required_inputs": self.required_inputs,
            "stages": [s.to_dict() for s in self.stages],
            "metadata": self.metadata,
        }

    def effects(self) -> list[str]:
        return sorted({s.effect.value for s in self.stages})

    # ------------------------------------------------------------------ #

    def execute(
        self,
        inputs: Mapping[str, WeightPackage],
        *,
        seed: int | str | None = None,
        job_id: str | None = None,
        scope: TaskScope | None = None,
    ) -> PipelineResult:
        scope = scope or self.scope
        if scope != self.scope:
            raise OperationUnavailable(
                f"pipeline was compiled for scope '{self.scope.value}' but is being executed "
                f"under '{scope.value}'; recompile for the authorized scope"
            )
        missing = [name for name in self.required_inputs if name not in inputs]
        if missing:
            raise ProgramInvalid(f"missing pipeline inputs: {missing}")
        env: dict[str, WeightPackage] = {name: inputs[name] for name in self.required_inputs}
        trace: list[OperationRecord] = []
        policy: list[dict[str, Any]] = []
        seeds: dict[str, int] = {}
        base_seed = parse_seed(seed)
        for index, stage in enumerate(self.stages):
            args = []
            for ref in stage.args:
                if ref not in env:
                    raise ProgramInvalid(
                        f"stage '{stage.name}' references unknown value '{ref}'"
                    )
                args.append(env[ref])
            stage_seed = (base_seed + index * 7919) % (2**63 - 1)
            record = self.registry.invoke(
                stage.code,
                args,
                stage.params,
                scope=scope,
                seed=stage_seed,
                engine=self.engine,
                job_id=job_id,
            )
            env[stage.name] = record.output
            trace.append(record)
            policy.extend(record.policy)
            seeds[stage.name] = stage_seed
        outputs = {name: env[name] for name in self.metadata.get("outputs", []) if name in env}
        if not outputs:
            last = self.stages[-1].name if self.stages else None
            if last is not None:
                outputs = {last: env[last]}
        return PipelineResult(outputs=outputs, trace=trace, policy=policy, seeds=seeds)


def compile_stages(
    stages: Sequence[Stage],
    *,
    scope: TaskScope,
    registry: OperationRegistry | None = None,
    engine: StrictPlusEngine | None = None,
    available_inputs: Sequence[str] = (),
    outputs: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> tuple[CompiledPipeline, CompilationReport]:
    registry = registry or default_registry()
    engine = engine or default_engine()
    problems: list[str] = []
    compiled: list[CompiledStage] = []
    known: set[str] = set(available_inputs)
    required_inputs: list[str] = []

    for index, stage in enumerate(stages):
        if not stage.name:
            problems.append(f"stage {index} has no name")
            continue
        if stage.name in known:
            problems.append(f"stage name '{stage.name}' is used twice")
            continue
        try:
            spec = registry.get(stage.code)
        except OperationNotRegistered as exc:
            problems.append(str(exc))
            continue
        if not spec.maturity.executable:
            problems.append(
                f"stage '{stage.name}': operation '{spec.code}' is {spec.maturity.value}, "
                "not executable"
            )
            continue
        if scope not in spec.scopes:
            problems.append(
                f"stage '{stage.name}': operation '{spec.code}' is not authorized in scope "
                f"'{scope.value}'"
            )
            continue
        try:
            validated = spec.validate_params(stage.params)
        except Exception as exc:
            problems.append(f"stage '{stage.name}': {exc}")
            continue
        stage_args: list[str] = []
        for ref in stage.args:
            if ref not in known:
                problems.append(
                    f"stage '{stage.name}' references '{ref}' before it is defined "
                    f"(known: {sorted(known) or 'none'})"
                )
            else:
                stage_args.append(ref)
        if len(stage_args) < spec.min_inputs or len(stage_args) > spec.max_inputs:
            problems.append(
                f"stage '{stage.name}': operation '{spec.code}' requires "
                f"{spec.min_inputs}..{spec.max_inputs} package inputs, got {len(stage_args)}"
            )
        # Policy dry-run: reject prohibited/unavailable effects at compile time.
        decision = engine.evaluate(spec.effect, spec.code, scope=scope, maturity=spec.maturity)
        if not decision.allowed:
            problems.append(f"stage '{stage.name}': {decision.reason}")
        compiled.append(
            CompiledStage(
                name=stage.name,
                code=spec.code,
                args=stage_args,
                params=validated,
                effect=spec.effect,
                title=spec.title,
                maturity=spec.maturity.value,
            )
        )
        known.add(stage.name)

    for name in available_inputs:
        required_inputs.append(name)
    for name in outputs:
        if name not in known:
            problems.append(f"declared output '{name}' is never produced")

    report = CompilationReport(
        ok=not problems,
        stages=[s.to_dict() for s in compiled],
        problems=problems,
        required_inputs=required_inputs,
        outputs=list(outputs) or ([compiled[-1].name] if compiled else []),
        effects=sorted({s.effect.value for s in compiled}),
    )
    pipeline = CompiledPipeline(
        compiled,
        required_inputs=required_inputs,
        scope=scope,
        engine=engine,
        registry=registry,
        metadata={**dict(metadata or {}), "outputs": report.outputs},
    )
    return pipeline, report


# --------------------------------------------------------------------------- #
# execution-plan documents (JSON) -> compiled pipeline
# --------------------------------------------------------------------------- #


def compile_plan(
    plan: Mapping[str, Any],
    *,
    scope: TaskScope | None = None,
    registry: OperationRegistry | None = None,
    engine: StrictPlusEngine | None = None,
) -> tuple[CompiledPipeline, CompilationReport]:
    """Compile a JSON execution plan.

    Shape::

        {
          "scope": "weight_modification",
          "inputs": [{"name": "base", "artifact_id": "..."}],
          "stages": [{"name": "a", "op": "w.arithmetic.delta", "args": ["base", "donor"],
                      "params": {}}],
          "outputs": ["a"]
        }
    """
    if not isinstance(plan, Mapping):
        raise ProgramInvalid("execution plan must be an object")
    stage_dicts = plan.get("stages")
    if not isinstance(stage_dicts, list) or not stage_dicts:
        raise ProgramInvalid("execution plan requires a non-empty 'stages' list")
    input_names = [str(i["name"]) for i in plan.get("inputs", []) if "name" in i]
    scope_value = scope or TaskScope(str(plan.get("scope", TaskScope.WEIGHT_MODIFICATION.value)))
    stages: list[Stage] = []
    for entry in stage_dicts:
        if not isinstance(entry, Mapping):
            raise ProgramInvalid("each stage must be an object")
        stages.append(
            Stage(
                name=str(entry.get("name", "")),
                code=str(entry.get("op", entry.get("code", ""))),
                args=[str(a) for a in entry.get("args", [])],
                params=dict(entry.get("params") or {}),
                title=str(entry.get("title", "")),
            )
        )
    return compile_stages(
        stages,
        scope=scope_value,
        registry=registry,
        engine=engine,
        available_inputs=input_names,
        outputs=[str(o) for o in plan.get("outputs", [])],
        metadata={"plan_name": plan.get("name", "execution_plan")},
    )


# --------------------------------------------------------------------------- #
# graph compatibility (runtime-facing)
# --------------------------------------------------------------------------- #


def check_graph_compatibility(graph: Graph, package: WeightPackage) -> dict[str, Any]:
    """Report exactly what would prevent this graph from executing."""
    report = graph.validate(package)
    problems = list(report.errors)
    unsupported_ops: list[str] = []
    from uzen.core.nir.ops import OPERATORS

    for nid, node in graph.nodes.items():
        spec = OPERATORS.get(node.op)
        if spec is None:
            unsupported_ops.append(f"{nid}:{node.op}")
        elif not spec.maturity.executable:
            unsupported_ops.append(f"{nid}:{node.op}({spec.maturity.value})")
    if unsupported_ops:
        problems.append(f"unsupported operators: {unsupported_ops}")
    missing = [p for p in graph.parameter_names() if p not in package]
    return {
        "ok": not problems,
        "executable": not problems,
        "problems": problems,
        "warnings": report.warnings,
        "unsupported_operators": unsupported_ops,
        "missing_parameters": missing,
        "num_nodes": len(graph.nodes),
        "parameter_names": graph.parameter_names(),
        "inferred": report.inferred,
        "bound_tensors": len(package),
        "runtime": "numpy",
    }
