"""Worker daemon.

Two real modes:

**Local mode** (default): the daemon owns a local workspace and executes queued jobs
from its own job database. This is the single-node execution plane.

**Remote mode** (``control_plane=...``): the daemon registers with a UZEN REST control
plane, heartbeats, claims jobs under a lease, executes them locally, and reports the
outcome. Input artifacts are exchanged through a shared filesystem path (the
``exchange_dir``), which is the transport implemented in this build; object storage is
a planned adapter, not a simulated one.

A worker never executes arbitrary user scripts: it only runs jobs through
:class:`uzen.jobs.service.JobService`, i.e. registered operations against artifacts.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from uzen.errors import UzenError, WorkerAuthError
from uzen.jobs.service import JobService
from uzen.util.canonical import utc_now_iso
from uzen.workers.protocol.messages import (
    PROTOCOL_VERSION,
    WorkerRegistration,
    detect_capabilities,
    make_message,
)


@dataclass
class WorkerDaemon:
    root: str | Path = "./uzen-worker"
    worker_id: str | None = None
    control_plane: str | None = None
    secret: str | None = None
    exchange_dir: str | Path | None = None
    poll_interval: float = 1.0
    lease_seconds: float = 120.0
    max_jobs: int | None = None
    _service: JobService | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.worker_id = self.worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self.exchange_dir = Path(self.exchange_dir) if self.exchange_dir else self.root / "exchange"
        self.exchange_dir.mkdir(parents=True, exist_ok=True)
        self._service = JobService(self.root / "workspace")

    @property
    def service(self) -> JobService:
        assert self._service is not None
        return self._service

    # ------------------------------------------------------------------ #
    # local mode
    # ------------------------------------------------------------------ #

    def run_once(self) -> dict[str, Any]:
        if self.control_plane:
            return self._remote_cycle()
        pending = self.service.db.list_jobs(state="QUEUED", limit=1)
        if not pending:
            return {"worker_id": self.worker_id, "executed": None, "note": "queue empty"}
        job = self.service.run(pending[0].id)
        return {"worker_id": self.worker_id, "executed": job["id"], "state": job["state"], "result": job.get("result") or job.get("error")}

    def run_forever(self, *, poll_interval: float | None = None, max_jobs: int | None = None, max_iterations: int | None = None) -> None:
        interval = poll_interval if poll_interval is not None else self.poll_interval
        executed = 0
        iterations = 0
        if self.control_plane:
            self.register()
        while True:
            iterations += 1
            outcome = self.run_once()
            if outcome.get("executed"):
                executed += 1
            if max_jobs is not None and executed >= max_jobs:
                return
            if max_iterations is not None and iterations >= max_iterations:
                return
            if not outcome.get("executed"):
                time.sleep(interval)

    # ------------------------------------------------------------------ #
    # remote mode
    # ------------------------------------------------------------------ #

    def register(self) -> dict[str, Any]:
        if not self.control_plane:
            raise UzenError("remote registration requires control_plane=...")
        body = WorkerRegistration(worker_id=self.worker_id or "worker").to_dict()
        response = self._post("/v1/workers/register", body, authenticated=False)
        secret = response.get("secret")
        if not secret:
            raise WorkerAuthError("control plane did not return a worker secret")
        self.secret = str(secret)
        assert self.worker_id is not None
        self._write_credentials(self.worker_id, self.secret)
        return response

    def _write_credentials(self, worker_id: str, secret: str) -> None:
        path = self.root / "worker-credentials.json"
        path.write_text(json.dumps({"worker_id": worker_id, "secret": secret}), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def heartbeat(self, status: str = "online") -> dict[str, Any]:
        assert self.worker_id is not None
        return self._post(
            "/v1/workers/heartbeat",
            {"worker_id": self.worker_id, "secret": self.secret, "status": status},
        )

    def claim(self) -> dict[str, Any] | None:
        assert self.worker_id is not None
        response = self._post(
            "/v1/workers/claim",
            {
                "worker_id": self.worker_id,
                "secret": self.secret,
                "lease_seconds": self.lease_seconds,
                "capabilities": detect_capabilities(),
            },
        )
        return response.get("job")

    def report(self, job_id: str, state: str, *, result: Any = None, error: Any = None) -> dict[str, Any]:
        assert self.worker_id is not None
        return self._post(
            "/v1/workers/report",
            {
                "worker_id": self.worker_id,
                "secret": self.secret,
                "job_id": job_id,
                "result": {"state": state, "result": result, "error": error},
            },
        )

    def _remote_cycle(self) -> dict[str, Any]:
        if not self.secret:
            self.register()
        self.heartbeat()
        job = self.claim()
        if not job:
            return {"worker_id": self.worker_id, "executed": None, "note": "no job available"}
        job_id = str(job["id"])
        self.service.db.append_event(job_id, "claimed_by_worker", {"worker_id": self.worker_id})
        start = time.perf_counter()
        self.service.db.claim_job(str(self.worker_id), lease_seconds=self.lease_seconds)
        # Execute the claimed request locally (artifacts must be reachable via the
        # exchange directory or inline tensors).
        submitted = self.service.submit(job["request"])
        executed = self.service.run(submitted["id"])
        duration_ms = (time.perf_counter() - start) * 1000.0
        state = executed["state"]
        self.report(job_id, state, result=executed.get("result"), error=executed.get("error"))
        return {
            "worker_id": self.worker_id,
            "executed": submitted["id"],
            "control_plane_job": job_id,
            "state": state,
            "duration_ms": round(duration_ms, 3),
        }

    # ------------------------------------------------------------------ #

    def _post(self, path: str, body: Mapping[str, Any], *, authenticated: bool = True) -> dict[str, Any]:
        assert self.control_plane is not None
        if authenticated and not self.secret:
            raise WorkerAuthError("worker is not registered (no secret); call register() first")
        url = f"{self.control_plane.rstrip('/')}{path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(dict(body)).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "X-UZEN-Protocol": PROTOCOL_VERSION},
        )
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            raise WorkerAuthError(
                f"worker request failed with HTTP {exc.code}", body=payload[:500], path=path
            ) from exc
        except urllib.error.URLError as exc:
            raise UzenError(f"cannot reach control plane: {exc.reason}", path=path) from exc

    def describe(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "root": str(self.root),
            "mode": "remote" if self.control_plane else "local",
            "control_plane": self.control_plane,
            "capabilities": detect_capabilities(),
            "credentials_file": str(self.root / "worker-credentials.json"),
        }
