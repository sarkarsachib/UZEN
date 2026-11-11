"""Job orchestration service.

Owns the full job lifecycle described in the specification::

    CREATED -> QUEUED -> DISPATCHED -> RUNNING -> VALIDATING -> SUCCEEDED/FAILED
    (cancellation: CANCEL_REQUESTED -> CANCELLED)

The service is the seam between the headless control plane (REST/MCP/CLI/workers) and
the computational plane (ANE + NWPC + VEL). It resolves input artifacts, builds graph
context when the request supplies an architecture spec, runs the engineering job,
persists the result as an immutable content-addressed artifact, and records the full
provenance and verification evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.ane.planner import ANEResult, AutonomousNeuralEngineer, JobRequest
from uzen.core.nase.synth import ArchitectureSynthesizer
from uzen.core.nir.executor import execute
from uzen.core.nws.families import build_architecture
from uzen.core.nws.store import ArtifactRecord, ArtifactStore
from uzen.core.nws.tensors import WeightPackage
from uzen.core.vel.verify import verify
from uzen.core.wie.inspector import WeightIntelligenceEngine
from uzen.engines.mar.engine import ModelArchitectureResearch
from uzen.engines.war.engine import WeightArchitectureResearch
from uzen.engines.wge.engine import WeightGenerationEngine
from uzen.engines.wme.engine import WeightModificationEngine
from uzen.errors import (
    ArtifactNotFound,
    InvalidOperationParameters,
    JobNotFound,
    JobStateError,
    UzenError,
)
from uzen.infrastructure.storage.db import Database, JobRow, TERMINAL_STATES
from uzen.policies.strict_plus.engine import StrictPlusEngine, default_engine
from uzen.util.canonical import utc_now_iso


class JobService:
    def __init__(
        self,
        root: str | Path,
        *,
        engine: StrictPlusEngine | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = engine or default_engine()
        self.store = ArtifactStore(self.root / "artifacts")
        self.db = Database(db_path or (self.root / "uzen.sqlite3"))
        self.ane = AutonomousNeuralEngineer(engine=self.engine)
        self.wie = WeightIntelligenceEngine()
        self.wme = WeightModificationEngine(engine=self.engine)
        self.wge = WeightGenerationEngine(engine=self.engine)
        self.war = WeightArchitectureResearch(engine=self.engine)
        self.mar = ModelArchitectureResearch(engine=self.engine)
        self.nase = ArchitectureSynthesizer()

    # ------------------------------------------------------------------ #
    # artifacts
    # ------------------------------------------------------------------ #

    def import_artifact(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        family: str | None = None,
    ) -> ArtifactRecord:
        p = Path(path)
        report = self.wie.inspect_file(p)
        if report.format == "safetensors":
            from uzen.infrastructure.storage.safetensors_io import read_safetensors

            arrays, header = read_safetensors(p)
        elif report.format == "npz":
            data = np.load(p, allow_pickle=False)
            arrays = {k: np.asarray(data[k]) for k in data.files}
            header = None
        elif report.format == "gguf":
            from uzen.infrastructure.storage.gguf_io import read_gguf

            arrays, _gguf_metadata = read_gguf(p)
            header = None
        else:
            raise InvalidOperationParameters(f"unsupported import format '{report.format}'")
        guessed = report.family_guess.get("family", "generic")
        package = WeightPackage.from_arrays(
            arrays,
            name=name or p.stem,
            family=family or (guessed if guessed != "unknown" else "generic"),
            metadata={
                "imported_from": str(p),
                "source_sha256": report.metadata.get("sha256"),
                "family_inference": report.family_guess,
            },
        )
        record = self.store.put(
            package,
            operation="import",
            reuse_existing=True,
            metadata={"source_path": str(p), "family_guess": report.family_guess},
            note="imported source artifact (immutable origin)",
        )
        self.db.index_artifact(
            artifact_id=record.artifact_id,
            path=str(record.path),
            name=package.name,
            family=package.family,
            operation="import",
            numel=package.numel(),
            num_tensors=len(package),
        )
        if report.family_guess.get("family") not in (None, "unknown"):
            self.db.register_model(
                artifact_id=record.artifact_id,
                family=package.family,
                spec={"inferred": report.family_guess},
            )
        return record

    def get_package(self, artifact_id: str) -> WeightPackage:
        if not self.store.exists(artifact_id):
            raise ArtifactNotFound(f"artifact '{artifact_id}' is not in the store")
        return self.store.get(artifact_id)

    # ------------------------------------------------------------------ #
    # jobs
    # ------------------------------------------------------------------ #

    def submit(self, request: Mapping[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        job_request = JobRequest.from_dict(request)
        if job_request.policy != "STRICT_PLUS":
            raise InvalidOperationParameters("only STRICT_PLUS jobs are accepted")
        # Reject unregistered / out-of-scope / malformed operations at submission, so a
        # job never enters the queue carrying work the engine cannot legally perform.
        from uzen.core.nwpc.registry import default_registry
        from uzen.errors import OperationUnavailable

        registry = default_registry()
        for index, entry in enumerate(job_request.operations):
            if not isinstance(entry, Mapping):
                raise InvalidOperationParameters(f"operation #{index} must be an object")
            code = str(entry.get("op") or entry.get("code") or "")
            if not code:
                raise InvalidOperationParameters(f"operation #{index} has no 'op' code")
            spec = registry.get(code)
            if not spec.maturity.executable:
                raise OperationUnavailable(
                    f"operation '{code}' is '{spec.maturity.value}' and cannot be executed",
                    operation=code,
                )
            if job_request.task_scope not in spec.scopes:
                raise OperationUnavailable(
                    f"operation '{code}' is not authorized in scope '{job_request.task_scope.value}'",
                    operation=code,
                    scope=job_request.task_scope.value,
                )
            spec.validate_params(dict(entry.get("params") or {}))
        row = self.db.create_job(request, idempotency_key=idempotency_key)
        self.db.append_event(row.id, "queued", {"scope": job_request.task_scope.value})
        return row.to_dict()

    def get(self, job_id: str) -> dict[str, Any]:
        return self.db.get_job(job_id).to_dict()

    def list(self, *, state: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.db.list_jobs(state=state, limit=limit)]

    def events(self, job_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        return self.db.events(job_id, after=after)

    def cancel(self, job_id: str) -> dict[str, Any]:
        row = self.db.get_job(job_id)
        if row.state in TERMINAL_STATES:
            return row.to_dict()
        self.db.request_cancel(job_id)
        if row.state in ("QUEUED", "CREATED"):
            self.db.transition(job_id, "CANCELLED", payload={"reason": "cancelled before execution"})
        return self.db.get_job(job_id).to_dict()

    # ------------------------------------------------------------------ #

    def run(self, job_id: str) -> dict[str, Any]:
        row = self.db.get_job(job_id)
        if row.state in TERMINAL_STATES:
            raise JobStateError(f"job '{job_id}' already finished ({row.state})")
        if self.db.is_cancel_requested(job_id):
            self.db.transition(job_id, "CANCELLED", payload={"reason": "cancel requested"})
            return self.db.get_job(job_id).to_dict()
        self.db.transition(job_id, "RUNNING", progress=0.05)
        request = JobRequest.from_dict(row.request)
        try:
            result = self._execute(job_id, request)
            if isinstance(result, dict) and "artifact_id" in result:
                result.setdefault("output_artifact_id", result["artifact_id"])
        except UzenError as exc:
            self.db.set_error(job_id, {"error": exc.to_dict(), "timestamp": utc_now_iso()})
            self.db.release_lease(job_id)
            return self.db.get_job(job_id).to_dict()
        except Exception as exc:  # unexpected: recorded with type for diagnosis
            self.db.set_error(
                job_id,
                {
                    "error": {
                        "code": "internal.error",
                        "message": str(exc),
                        "details": {"type": type(exc).__name__},
                    },
                    "timestamp": utc_now_iso(),
                },
            )
            self.db.release_lease(job_id)
            return self.db.get_job(job_id).to_dict()
        self.db.transition(job_id, "VALIDATING", progress=0.9)
        self.db.set_result(job_id, result)
        self.db.release_lease(job_id)
        return self.db.get_job(job_id).to_dict()

    def run_pending(self, limit: int = 1) -> list[dict[str, Any]]:
        out = []
        for row in self.db.list_jobs(state="QUEUED", limit=limit):
            out.append(self.run(row.id))
        return out

    # ------------------------------------------------------------------ #

    def _execute(self, job_id: str, request: JobRequest) -> dict[str, Any]:
        scope = request.task_scope
        if scope.value in ("model_architecture_research", "architecture_synthesis", "model_creation"):
            return self._execute_architecture(job_id, request)
        if scope.value == "weight_architecture_research" and not request.operations:
            return self._execute_weight_research(job_id, request)

        packages = self._resolve_inputs(request)
        target, reference, graph, graph_inputs = self._resolve_context(request, packages)
        self.db.append_event(
            job_id,
            "planning",
            {"objective": request.objective, "inputs": list(packages)},
        )
        result: ANEResult = self.ane.execute(
            request,
            packages,
            target=target,
            graph=graph,
            graph_inputs=graph_inputs,
            reference=reference,
            job_id=job_id,
        )
        self.db.set_plan(job_id, result.plan.to_dict())
        self.db.append_event(
            job_id,
            "executed",
            {
                "candidates": len(result.candidates),
                "best_metric": result.best.metric if result.best else None,
                "diagnosis": result.diagnosis[-5:],
            },
        )
        if self.db.is_cancel_requested(job_id):
            self.db.transition(job_id, "CANCELLED", payload={"reason": "cancel observed before persist"})
            return {"cancelled": True}

        package = result.package
        assert package is not None
        validation = result.verification.to_dict() if result.verification else {}
        parent_ids = [p for p in (package.source_artifact_id,) if p]
        for name, pkg in packages.items():
            if pkg.source_artifact_id and pkg.source_artifact_id not in parent_ids:
                parent_ids.append(pkg.source_artifact_id)
        record = self.store.put(
            package,
            operation=result.best.op_code if result.best else "engineering_job",
            reuse_existing=True,
            parents=parent_ids,
            operations=result.trace,
            rng_seeds=result.seeds,
            policy=result.policy,
            validation=validation,
            metadata={
                "job_id": job_id,
                "objective": request.objective,
                "scope": scope.value,
                "notes": result.notes,
                "warnings": result.warnings,
                "diagnosis": result.diagnosis,
            },
            job_id=job_id,
        )
        self.db.index_artifact(
            artifact_id=record.artifact_id,
            path=str(record.path),
            name=package.name,
            family=package.family,
            operation=record.manifest.lineage.operation,
            numel=package.numel(),
            num_tensors=len(package),
            job_id=job_id,
        )
        return {
            "artifact_id": record.artifact_id,
            "summary": record.manifest.summary,
            "plan": result.plan.to_dict(),
            "candidates": [c.to_dict() for c in result.candidates],
            "best_candidate": result.best.to_dict() if result.best else None,
            "verification": validation,
            "notes": result.notes,
            "warnings": result.warnings,
            "diagnosis": result.diagnosis,
            "policy": result.policy,
            "trace": result.trace,
            "output_format": request.output_format,
        }

    # ------------------------------------------------------------------ #

    def _resolve_inputs(self, request: JobRequest) -> dict[str, WeightPackage]:
        packages: dict[str, WeightPackage] = {}
        for index, entry in enumerate(request.inputs):
            name = str(entry.get("name", f"input{index}"))
            if "artifact_id" in entry:
                packages[name] = self.get_package(str(entry["artifact_id"]))
            elif "path" in entry:
                record = self.import_artifact(str(entry["path"]))
                packages[name] = self.store.get(record.artifact_id)
            elif "inline" in entry:
                arrays = {k: np.asarray(v) for k, v in entry["inline"].items()}
                packages[name] = WeightPackage.from_arrays(arrays, name=name)
            else:
                raise InvalidOperationParameters(
                    f"input '{name}' must specify artifact_id, path, or inline tensors"
                )
        if not packages and request.task_scope.value not in ("weight_generation",):
            raise InvalidOperationParameters(
                "the job supplies no inputs; weight engineering requires at least one package"
            )
        if not packages:
            raise InvalidOperationParameters(
                "weight generation requires a template artifact or an architecture spec"
            )
        return packages

    def _resolve_context(
        self, request: JobRequest, packages: Mapping[str, WeightPackage]
    ) -> tuple[WeightPackage | None, WeightPackage | None, Any, dict[str, np.ndarray] | None]:
        plan = request.evaluation_plan or {}
        target = None
        reference = None
        if plan.get("target_artifact_id"):
            target = self.get_package(str(plan["target_artifact_id"]))
        if plan.get("reference_artifact_id"):
            reference = self.get_package(str(plan["reference_artifact_id"]))
        graph = None
        graph_inputs: dict[str, np.ndarray] | None = None
        graph_spec = plan.get("graph") or request.options.get("graph")
        if graph_spec:
            definition = build_architecture(
                str(graph_spec.get("family", "mlp")),
                {k: v for k, v in graph_spec.items() if k != "family"},
            )
            basis = reference or next(iter(packages.values()))
            compat = definition.graph.validate(basis)
            if compat.ok:
                graph = definition.graph
                batch = int(plan.get("batch", 2))
                seq = int(plan.get("seq", 8))
                inputs = definition.sample_inputs(batch=batch, seq=seq, seed=0)
                # ensure shapes match the supplied package family
                try:
                    execute(graph, inputs, basis)
                    graph_inputs = inputs
                except Exception:
                    graph = None
                    graph_inputs = None
        if graph is None and plan.get("metric") == "activation_agreement":
            raise InvalidOperationParameters(
                "activation_agreement requires evaluation_plan.graph (a family spec) and a "
                "compatible reference artifact"
            )
        return target, reference, graph, graph_inputs

    # ------------------------------------------------------------------ #

    def _execute_weight_research(self, job_id: str, request: JobRequest) -> dict[str, Any]:
        packages = self._resolve_inputs(request)
        primary = next(iter(packages.values()))
        report = self.war.run(
            primary,
            experiments=request.operations or None,
            seed=request.seed,
            scope=request.task_scope,
        )
        self.db.append_event(job_id, "research_complete", {"experiments": len(report["experiments"])})
        # Research reports are measurements, not modified parameter sets: no artifact is
        # fabricated here. Any concrete candidate must be produced by an explicit
        # subsequent engineering job with its own provenance.
        return {
            "research": report,
            "artifacts": [],
            "notes": [
                "weight architecture research reports representation trade-offs; it does not "
                "claim downstream quality improvements",
            ],
        }

    def _execute_architecture(self, job_id: str, request: JobRequest) -> dict[str, Any]:
        options = request.options
        if "family" in options:
            spec = {k: v for k, v in options.items() if k != "family"}
            definition = self.nase.synthesize({"family": options["family"], **spec})
            package = definition.build_package(seed=request.seed)
            definition.graph.require_valid(package)
            inputs = definition.sample_inputs(seed=0)
            result = execute(definition.graph, inputs, package)
            verification = verify(
                package,
                graph=definition.graph,
                inputs=inputs,
                policy_records=[],
                operations=[],
            )
            record = self.store.put(
                package,
                operation="architecture_synthesis",
                reuse_existing=True,
                metadata={
                    "job_id": job_id,
                    "architecture": definition.summary(),
                    "graph": definition.graph.to_dict(),
                },
                validation=verification.to_dict(),
                job_id=job_id,
            )
            self.db.index_artifact(
                artifact_id=record.artifact_id,
                path=str(record.path),
                name=package.name,
                family=package.family,
                operation="architecture_synthesis",
                numel=package.numel(),
                num_tensors=len(package),
                job_id=job_id,
            )
            self.db.register_model(
                artifact_id=record.artifact_id,
                family=package.family,
                spec={**definition.spec, "graph": definition.graph.to_dict()},
            )
            return {
                "artifact_id": record.artifact_id,
                "architecture": definition.summary(),
                "forward_outputs": {k: list(v.shape) for k, v in result.outputs.items()},
                "verification": verification.to_dict(),
                "notes": [
                    "weights are structured random initializations with per-tensor origin labels",
                    "the NIR graph is stored in the artifact metadata for reconstruction",
                ],
            }
        if "research" in options or request.task_scope.value in ("model_architecture_research", "architecture_synthesis"):
            family = str(options.get("research_family", options.get("family", "transformer_decoder")))
            base_spec = dict(options.get("base_spec") or options.get("spec") or {})
            if not base_spec:
                raise InvalidOperationParameters(
                    "architecture research requires options.base_spec (e.g. {d_model, n_heads, n_layers, ...})"
                )
            report = self.mar.explore(
                family=family,
                base_spec=base_spec,
                variants=options.get("variants"),
                seed=request.seed,
                scope=request.task_scope,
            )
            return {
                "research": report,
                "notes": [
                    "architecture research measures cost/shape properties of random-weight "
                    "candidates; it does not establish task quality"
                ],
            }
        raise InvalidOperationParameters(
            "architecture job requires options.family (synthesis) or options.base_spec (research)"
        )

    # ------------------------------------------------------------------ #
    # read APIs
    # ------------------------------------------------------------------ #

    def artifacts(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.artifacts(limit=limit)

    def artifact_manifest(self, artifact_id: str) -> dict[str, Any]:
        return self.store.manifest(artifact_id).to_dict()

    def verify_artifact(self, artifact_id: str) -> dict[str, Any]:
        return self.store.verify(artifact_id)

    def models(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.models(limit=limit)

    def stats(self) -> dict[str, Any]:
        return {"store": self.store.stats(), "database": self.db.stats()}
