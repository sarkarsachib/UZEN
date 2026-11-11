"""REST control plane for UZEN.

A versioned, OpenAPI-documented HTTP surface over the same engines the SDK and CLI
use. It is strictly a *control plane*: metadata, orchestration and artifact
management. All tensor math happens in the job service / worker execution plane.

Run::

    uzen serve --root ./uzen-workspace --host 0.0.0.0 --port 8080

Authentication: set ``UZEN_API_TOKEN`` to require ``Authorization: Bearer <token>``
on every ``/v1`` route. Worker routes additionally verify the worker's registered
secret hash.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

try:  # FastAPI is an optional extra: the engine never requires it.
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

    FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    FastAPI = None  # type: ignore
    FASTAPI_AVAILABLE = False

from uzen.core.nwpc.registry import default_registry
from uzen.core.nws.families import family_catalogue
from uzen.errors import UzenError
from uzen.infrastructure.observability.metrics import METRICS
from uzen.infrastructure.security.tokens import hash_secret
from uzen.infrastructure.storage.db import JOB_STATES
from uzen.jobs.service import JobService
from uzen.policies.strict_plus.engine import default_engine
from uzen.runtimes.pytorch_runtime import PyTorchRuntime, torch_available
from uzen.workers.protocol.messages import detect_capabilities

API_VERSION = "v1"


def create_app(
    *,
    root: str | Path | None = None,
    service: JobService | None = None,
    api_token: str | None = None,
) -> Any:
    if not FASTAPI_AVAILABLE:
        from uzen.errors import RuntimeUnavailable

        raise RuntimeUnavailable(
            "FastAPI is not installed; install the API extra with "
            "`pip install 'uzen[api]'` or run the engine via CLI/SDK instead"
        )
    service = service or JobService(root or os.environ.get("UZEN_ROOT", "./uzen-workspace"))
    token = api_token if api_token is not None else os.environ.get("UZEN_API_TOKEN")
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="uzen-job")
    lock = threading.Lock()

    app = FastAPI(
        title="UZEN",
        version="0.1.0",
        description=(
            "Universal Neural Weight Engineering & Architecture Research System — "
            "headless control plane. STRICT_PLUS: no training, fine-tuning, "
            "backpropagation or gradient-based parameter updates."
        ),
    )
    app.state.service = service

    @app.middleware("http")
    async def observe_http(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        METRICS.inc(
            "uzen_http_requests_total",
            method=request.method,
            path=path,
            status=str(response.status_code),
        )
        METRICS.observe(
            "uzen_http_request_duration_seconds",
            time.perf_counter() - started,
            method=request.method,
            path=path,
        )
        return response

    # ------------------------------------------------------------------ #
    # auth / errors
    # ------------------------------------------------------------------ #

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not token:
            return
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail={"code": "auth.missing", "message": "bearer token required"})
        presented = authorization.split(" ", 1)[1].strip()
        if not hmac.compare_digest(hash_secret(presented), hash_secret(token)):
            raise HTTPException(status_code=401, detail={"code": "auth.invalid", "message": "invalid token"})

    @app.exception_handler(UzenError)
    async def uzen_error_handler(request: Request, exc: UzenError) -> JSONResponse:
        detail = exc.to_dict()
        METRICS.inc("uzen_errors_total", code=str(detail.get("code", type(exc).__name__)))
        return JSONResponse(status_code=exc.http_status, content={"detail": detail})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": "internal.error",
                    "message": str(exc),
                    "details": {"type": type(exc).__name__, "path": str(request.url)},
                }
            },
        )

    # ------------------------------------------------------------------ #
    # meta
    # ------------------------------------------------------------------ #

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": "0.1.0", "policy": "STRICT_PLUS"}

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        snapshot = service.stats()
        counts = snapshot.get("jobs_by_state", {})
        for state in JOB_STATES:
            METRICS.set("uzen_jobs", counts.get(state, 0), state=state)
        METRICS.set("uzen_artifacts_indexed", snapshot.get("artifacts_indexed", 0))
        METRICS.set("uzen_workers", snapshot.get("workers", 0))
        return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4")

    @app.get(f"/{API_VERSION}/stats")
    def stats(_: None = Depends(require_auth)) -> dict[str, Any]:
        capabilities = detect_capabilities()
        return {
            **service.stats(),
            "engine": capabilities,
            "pytorch_available": torch_available(),
            "strict_plus": default_engine().describe()["statements"],
        }

    @app.get(f"/{API_VERSION}/operations")
    def operations(family: str | None = None, executable_only: bool = False, _: None = Depends(require_auth)):
        registry = default_registry()
        return {
            "operations": registry.catalogue(family=family, executable_only=executable_only),
            "families": registry.families(),
            "count": len(registry.all()),
        }

    @app.get(f"/{API_VERSION}/operations/{{code}}")
    def operation_detail(code: str, _: None = Depends(require_auth)):
        return default_registry().get(code).to_dict()

    @app.get(f"/{API_VERSION}/policies/strict-plus")
    def policy(_: None = Depends(require_auth)):
        return default_engine().describe()

    @app.get(f"/{API_VERSION}/architectures/families")
    def families(_: None = Depends(require_auth)):
        return {"families": family_catalogue()}

    # ------------------------------------------------------------------ #
    # jobs
    # ------------------------------------------------------------------ #

    @app.post(f"/{API_VERSION}/jobs", status_code=201)
    def submit_job(
        payload: dict[str, Any],
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        _: None = Depends(require_auth),
    ):
        job = service.submit(payload, idempotency_key=idempotency_key)
        executor.submit(_safe_run, service, job["id"])
        return job

    def _safe_run(svc: JobService, job_id: str) -> None:
        with lock:
            try:
                svc.run(job_id)
            except Exception:
                # run() records failures internally; this guards the thread pool.
                pass

    @app.get(f"/{API_VERSION}/jobs")
    def list_jobs(state: str | None = None, limit: int = 50, _: None = Depends(require_auth)):
        return {"jobs": service.list(state=state, limit=limit)}

    @app.get(f"/{API_VERSION}/jobs/{{job_id}}")
    def get_job(job_id: str, _: None = Depends(require_auth)):
        return service.get(job_id)

    @app.post(f"/{API_VERSION}/jobs/{{job_id}}/run")
    def run_job(job_id: str, _: None = Depends(require_auth)):
        return service.run(job_id)

    @app.post(f"/{API_VERSION}/jobs/{{job_id}}/cancel")
    def cancel_job(job_id: str, _: None = Depends(require_auth)):
        return service.cancel(job_id)

    @app.get(f"/{API_VERSION}/jobs/{{job_id}}/events")
    def job_events(job_id: str, after: int = 0, _: None = Depends(require_auth)):
        return {"events": service.events(job_id, after=after)}

    # ------------------------------------------------------------------ #
    # artifacts / models / weights
    # ------------------------------------------------------------------ #

    @app.post(f"/{API_VERSION}/models/import")
    def import_model(payload: dict[str, Any], _: None = Depends(require_auth)):
        record = service.import_artifact(
            payload["path"], name=payload.get("name"), family=payload.get("family")
        )
        return {
            "artifact_id": record.artifact_id,
            "name": record.manifest.name,
            "family": record.manifest.family,
            "summary": record.manifest.summary,
            "checksum": record.manifest.checksum,
        }

    @app.get(f"/{API_VERSION}/models")
    def list_models(limit: int = 100, _: None = Depends(require_auth)):
        return {"models": service.models(limit=limit)}

    @app.get(f"/{API_VERSION}/weights")
    def list_weights(limit: int = 100, _: None = Depends(require_auth)):
        return {"weights": service.artifacts(limit=limit)}

    @app.post(f"/{API_VERSION}/weights/analyze")
    def analyze_weights(payload: dict[str, Any], _: None = Depends(require_auth)):
        if "artifact_id" in payload:
            package = service.get_package(payload["artifact_id"])
            return service.wie.inspect_package(package).to_dict()
        if "path" in payload:
            return service.wie.inspect_file(payload["path"]).to_dict()
        raise HTTPException(400, detail={"code": "request.invalid", "message": "provide artifact_id or path"})

    @app.post(f"/{API_VERSION}/weights/generate")
    def generate_weights(payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.interfaces.python_sdk.client import Uzen

        client = Uzen(root=service.root)
        return client.generate(
            family=payload.get("family"),
            spec=payload.get("spec"),
            template_artifact_id=payload.get("template_artifact_id"),
            method=str(payload.get("method", "orthogonal")),
            params=payload.get("params"),
            seed=payload.get("seed", 0),
        )

    @app.post(f"/{API_VERSION}/weights/modify")
    def modify_weights(payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.interfaces.python_sdk.client import Uzen

        client = Uzen(root=service.root)
        return client.modify(
            payload["artifact_id"],
            payload["method"],
            params=payload.get("params"),
            others=payload.get("others", []),
            seed=payload.get("seed", 0),
        )

    @app.post(f"/{API_VERSION}/weights/reconstruct")
    def reconstruct_weights(payload: dict[str, Any], _: None = Depends(require_auth)):
        job = service.submit(
            {
                "task_scope": "weight_reconstruction",
                "objective": payload.get("objective", "reconstruct weights from supplied components"),
                "inputs": payload["inputs"],
                "operations": payload.get("operations", []),
                "evaluation_plan": payload.get("evaluation_plan", {}),
                "seed": payload.get("seed", 0),
                "budget": payload.get("budget", {}),
                "options": payload.get("options", {}),
            }
        )
        return service.run(job["id"])

    @app.post(f"/{API_VERSION}/weights/evaluate")
    def evaluate_weights(payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.interfaces.python_sdk.client import Uzen

        client = Uzen(root=service.root)
        return client.evaluate(
            payload["artifact_id"],
            reference_artifact_id=payload.get("reference_artifact_id"),
            family=payload.get("family"),
            spec=payload.get("spec"),
        )

    @app.get(f"/{API_VERSION}/artifacts/{{artifact_id}}")
    def artifact_manifest(artifact_id: str, _: None = Depends(require_auth)):
        return service.artifact_manifest(artifact_id)

    @app.get(f"/{API_VERSION}/artifacts/{{artifact_id}}/verify")
    def verify_artifact(artifact_id: str, _: None = Depends(require_auth)):
        return service.verify_artifact(artifact_id)

    @app.get(f"/{API_VERSION}/artifacts/{{artifact_id}}/lineage")
    def artifact_lineage(artifact_id: str, _: None = Depends(require_auth)):
        return service.store.lineage_graph().describe(artifact_id)

    @app.get(f"/{API_VERSION}/artifacts/{{artifact_id}}/export")
    def artifact_export(artifact_id: str, fmt: str = "safetensors", _: None = Depends(require_auth)):
        target = Path(service.root) / "exports" / f"{artifact_id}.{ 'safetensors' if fmt == 'safetensors' else fmt }"
        path = service.store.export(artifact_id, target, fmt=fmt)
        if fmt == "safetensors":
            return FileResponse(path, media_type="application/octet-stream", filename=f"{artifact_id}.safetensors")
        return JSONResponse({"path": str(path)})

    # ------------------------------------------------------------------ #
    # architectures
    # ------------------------------------------------------------------ #

    @app.post(f"/{API_VERSION}/architectures/synthesize")
    def synthesize_architecture(payload: dict[str, Any], _: None = Depends(require_auth)):
        job = service.submit(
            {
                "task_scope": "architecture_synthesis",
                "objective": payload.get("objective", "synthesize an architecture"),
                "inputs": payload.get("inputs", []),
                "options": {"family": payload["family"], **payload.get("spec", {})},
                "seed": payload.get("seed", 0),
            }
        )
        return service.run(job["id"])

    @app.post(f"/{API_VERSION}/architectures/research")
    def research_architecture(payload: dict[str, Any], _: None = Depends(require_auth)):
        job = service.submit(
            {
                "task_scope": "model_architecture_research",
                "objective": payload.get("objective", "research architecture variants"),
                "options": {
                    "research_family": payload.get("family", "transformer_decoder"),
                    "base_spec": payload.get("base_spec", {}),
                    "variants": payload.get("variants"),
                },
                "seed": payload.get("seed", 0),
            }
        )
        return service.run(job["id"])

    # ------------------------------------------------------------------ #
    # workers
    # ------------------------------------------------------------------ #

    @app.get(f"/{API_VERSION}/workers")
    def list_workers(_: None = Depends(require_auth)):
        return {"workers": service.db.workers()}

    @app.post(f"/{API_VERSION}/workers/register", status_code=201)
    def register_worker(payload: dict[str, Any]):
        import secrets

        worker_id = str(payload.get("worker_id", "")).strip()
        if not worker_id:
            raise HTTPException(400, detail={"code": "worker.invalid", "message": "worker_id required"})
        secret = secrets.token_urlsafe(48)
        service.db.register_worker(
            worker_id=worker_id,
            protocol_version=str(payload.get("protocol_version", "uzen.worker/1")),
            capabilities=dict(payload.get("capabilities") or {}),
            token_hash=hash_secret(secret),
        )
        return {
            "worker_id": worker_id,
            "secret": secret,
            "protocol_version": "uzen.worker/1",
            "note": "store this secret on the worker; it is not retrievable again",
        }

    def _authenticate_worker(payload: Mapping[str, Any]) -> str:
        worker_id = str(payload.get("worker_id", ""))
        secret = str(payload.get("secret", ""))
        stored = service.db.worker_token_hash(worker_id)
        if not stored or not secret or not hmac.compare_digest(hash_secret(secret), stored):
            raise HTTPException(401, detail={"code": "worker.auth_failed", "message": "invalid worker credentials"})
        return worker_id

    @app.post(f"/{API_VERSION}/workers/heartbeat")
    def worker_heartbeat(payload: dict[str, Any]):
        worker_id = _authenticate_worker(payload)
        service.db.heartbeat(worker_id, status=str(payload.get("status", "online")))
        return {"ok": True, "worker_id": worker_id}

    @app.post(f"/{API_VERSION}/workers/claim")
    def worker_claim(payload: dict[str, Any]):
        worker_id = _authenticate_worker(payload)
        job = service.db.claim_job(worker_id, lease_seconds=float(payload.get("lease_seconds", 120.0)))
        return {"job": job.to_dict() if job else None}

    @app.post(f"/{API_VERSION}/workers/renew")
    def worker_renew(payload: dict[str, Any]):
        worker_id = _authenticate_worker(payload)
        ok = service.db.renew_lease(str(payload["job_id"]), worker_id, lease_seconds=float(payload.get("lease_seconds", 120.0)))
        return {"ok": ok}

    @app.post(f"/{API_VERSION}/workers/report")
    def worker_report(payload: dict[str, Any]):
        worker_id = _authenticate_worker(payload)
        job_id = str(payload.get("job_id", ""))
        if not job_id:
            raise HTTPException(400, detail={"code": "worker.invalid", "message": "job_id required"})
        service.db.append_event(job_id, "worker_report", {"worker_id": worker_id, "result": payload.get("result")})
        service.db.release_lease(job_id)
        return {"ok": True}

    return app
