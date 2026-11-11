"""SQLite metadata store for the control plane.

Artifacts live on disk (content addressed). This database holds the *control* data:
jobs and their state transitions, progress events with monotonic sequence numbers,
model/artifact indexes, worker registrations and leases, and idempotency keys.

SQLite is used for the single-node control plane; the schema is written so it can be
ported to PostgreSQL without redesign (text ids, explicit timestamps, no SQLite-only
types).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from uzen.errors import IdempotencyConflict, JobNotFound, JobStateError
from uzen.util.canonical import canonical_json, sha256_text, utc_now_iso

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    task_scope TEXT NOT NULL,
    objective TEXT NOT NULL DEFAULT '',
    request_json TEXT NOT NULL,
    plan_json TEXT,
    result_json TEXT,
    error_json TEXT,
    idempotency_key TEXT UNIQUE,
    request_hash TEXT,
    worker_id TEXT,
    progress REAL NOT NULL DEFAULT 0.0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);

CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    UNIQUE(job_id, seq)
);

CREATE TABLE IF NOT EXISTS artifacts_index (
    artifact_id TEXT PRIMARY KEY,
    name TEXT,
    family TEXT,
    operation TEXT,
    path TEXT NOT NULL,
    numel INTEGER,
    num_tensors INTEGER,
    job_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    family TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(artifact_id)
);

CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,
    protocol_version TEXT NOT NULL,
    capabilities_json TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    registered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leases (
    job_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    expires_at REAL NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

JOB_STATES = (
    "CREATED",
    "QUEUED",
    "DISPATCHED",
    "RUNNING",
    "VALIDATING",
    "SUCCEEDED",
    "FAILED",
    "CANCEL_REQUESTED",
    "CANCELLED",
)

TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "CREATED": {"QUEUED", "CANCELLED", "FAILED"},
    "QUEUED": {"DISPATCHED", "RUNNING", "CANCEL_REQUESTED", "CANCELLED", "FAILED"},
    "DISPATCHED": {"RUNNING", "CANCEL_REQUESTED", "FAILED", "CANCELLED"},
    "RUNNING": {"VALIDATING", "FAILED", "CANCEL_REQUESTED", "CANCELLED", "SUCCEEDED"},
    "VALIDATING": {"SUCCEEDED", "FAILED"},
    "CANCEL_REQUESTED": {"CANCELLED", "FAILED"},
    "SUCCEEDED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}


@dataclass
class JobRow:
    id: str
    state: str
    task_scope: str
    objective: str
    request: dict[str, Any]
    plan: dict[str, Any] | None
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    progress: float
    cancel_requested: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "task_scope": self.task_scope,
            "objective": self.objective,
            "request": self.request,
            "plan": self.plan,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "cancel_requested": self.cancel_requested,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    # jobs
    # ------------------------------------------------------------------ #

    def create_job(
        self,
        request: Mapping[str, Any],
        *,
        job_id: str | None = None,
        idempotency_key: str | None = None,
        state: str = "QUEUED",
    ) -> JobRow:
        job_id = job_id or str(uuid.uuid4())
        now = utc_now_iso()
        request_json = json.dumps(request, separators=(",", ":"), sort_keys=True)
        request_hash = sha256_text(request_json)
        with self._lock:
            if idempotency_key:
                existing = self._conn.execute(
                    "SELECT * FROM idempotency WHERE key = ?", (idempotency_key,)
                ).fetchone()
                if existing:
                    if existing["request_hash"] != request_hash:
                        raise IdempotencyConflict(
                            "idempotency key was reused with a different request body",
                            key=idempotency_key,
                            existing_job=existing["job_id"],
                        )
                    row = self._conn.execute(
                        "SELECT * FROM jobs WHERE id = ?", (existing["job_id"],)
                    ).fetchone()
                    return _row_to_job(row)
                self._conn.execute(
                    "INSERT INTO idempotency(key, job_id, request_hash, created_at) VALUES (?,?,?,?)",
                    (idempotency_key, job_id, request_hash, now),
                )
            self._conn.execute(
                "INSERT INTO jobs(id, state, task_scope, objective, request_json, idempotency_key, "
                "request_hash, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    state,
                    str(request.get("task_scope", "inspection")),
                    str(request.get("objective", "")),
                    request_json,
                    idempotency_key,
                    request_hash,
                    now,
                    now,
                ),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        self.append_event(job_id, "created", {"state": state, "task_scope": request.get("task_scope")})
        return _row_to_job(row)

    def get_job(self, job_id: str) -> JobRow:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(f"job '{job_id}' not found")
        return _row_to_job(row)

    def list_jobs(self, *, state: str | None = None, limit: int = 50) -> list[JobRow]:
        query = "SELECT * FROM jobs"
        params: list[Any] = []
        if state:
            query += " WHERE state = ?"
            params.append(state)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_row_to_job(r) for r in rows]

    def transition(
        self,
        job_id: str,
        new_state: str,
        *,
        payload: Mapping[str, Any] | None = None,
        progress: float | None = None,
        force: bool = False,
    ) -> JobRow:
        if new_state not in JOB_STATES:
            raise JobStateError(f"unknown job state '{new_state}'")
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(f"job '{job_id}' not found")
            current = row["state"]
            if not force and new_state != current and new_state not in ALLOWED_TRANSITIONS.get(current, set()):
                raise JobStateError(
                    f"illegal job transition {current} -> {new_state}", job_id=job_id
                )
            now = utc_now_iso()
            updates = ["state = ?", "updated_at = ?"]
            params: list[Any] = [new_state, now]
            if progress is not None:
                updates.append("progress = ?")
                params.append(float(progress))
            params.append(job_id)
            self._conn.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?", params)
            self._conn.commit()
        self.append_event(job_id, "state_changed", {"from": current, "to": new_state, **(payload or {})})
        if new_state != current:
            # State-named event so subscribers can stream lifecycle progress without
            # decoding every state_changed payload (CREATED -> queued -> running ...).
            self.append_event(
                job_id,
                new_state.lower(),
                {"from": current, "to": new_state, **(payload or {})},
            )
        return self.get_job(job_id)

    def set_plan(self, job_id: str, plan: Mapping[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET plan_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(plan, separators=(",", ":"), sort_keys=True), utc_now_iso(), job_id),
            )
            self._conn.commit()

    def set_result(self, job_id: str, result: Mapping[str, Any], *, state: str = "SUCCEEDED") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET result_json = ?, state = ?, progress = 1.0, updated_at = ? WHERE id = ?",
                (json.dumps(result, separators=(",", ":"), sort_keys=True), state, utc_now_iso(), job_id),
            )
            self._conn.commit()
        self.append_event(job_id, "completed", {"state": state})

    def set_error(self, job_id: str, error: Mapping[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET error_json = ?, state = 'FAILED', updated_at = ? WHERE id = ?",
                (json.dumps(error, separators=(",", ":"), sort_keys=True), utc_now_iso(), job_id),
            )
            self._conn.commit()
        self.append_event(job_id, "failed", dict(error))

    def request_cancel(self, job_id: str) -> JobRow:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(f"job '{job_id}' not found")
            if row["state"] in TERMINAL_STATES:
                return _row_to_job(row)
            self._conn.execute(
                "UPDATE jobs SET cancel_requested = 1, state = ?, updated_at = ? WHERE id = ?",
                ("CANCEL_REQUESTED", utc_now_iso(), job_id),
            )
            self._conn.commit()
        self.append_event(job_id, "cancel_requested", {})
        return self.get_job(job_id)

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #

    def append_event(self, job_id: str, kind: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS m FROM job_events WHERE job_id = ?", (job_id,)
            ).fetchone()
            seq = int(row["m"]) + 1
            self._conn.execute(
                "INSERT INTO job_events(job_id, seq, kind, payload_json, created_at) VALUES (?,?,?,?,?)",
                (job_id, seq, kind, json.dumps(payload, separators=(",", ":"), sort_keys=True), utc_now_iso()),
            )
            self._conn.commit()

    def events(self, job_id: str, *, after: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        self.get_job(job_id)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
                (job_id, int(after), int(limit)),
            ).fetchall()
        return [
            {
                "seq": r["seq"],
                "kind": r["kind"],
                "payload": json.loads(r["payload_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # artifacts / models
    # ------------------------------------------------------------------ #

    def index_artifact(
        self,
        *,
        artifact_id: str,
        path: str,
        name: str | None = None,
        family: str | None = None,
        operation: str | None = None,
        numel: int | None = None,
        num_tensors: int | None = None,
        job_id: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO artifacts_index(artifact_id, name, family, operation, path, "
                "numel, num_tensors, job_id, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (artifact_id, name, family, operation, path, numel, num_tensors, job_id, utc_now_iso()),
            )
            self._conn.commit()

    def artifacts(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM artifacts_index ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    def register_model(self, *, artifact_id: str, family: str, spec: Mapping[str, Any]) -> str:
        model_id = f"model_{sha256_text(artifact_id)[:16]}"
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO models(model_id, artifact_id, family, spec_json, created_at) "
                "VALUES (?,?,?,?,?)",
                (model_id, artifact_id, family, canonical_json(dict(spec)), utc_now_iso()),
            )
            self._conn.commit()
        return model_id

    def models(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM models ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [
            {
                "model_id": r["model_id"],
                "artifact_id": r["artifact_id"],
                "family": r["family"],
                "spec": json.loads(r["spec_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # workers
    # ------------------------------------------------------------------ #

    def register_worker(
        self,
        *,
        worker_id: str,
        protocol_version: str,
        capabilities: Mapping[str, Any],
        token_hash: str,
    ) -> None:
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                "INSERT INTO workers(worker_id, protocol_version, capabilities_json, token_hash, status, "
                "last_seen, registered_at) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(worker_id) DO UPDATE SET capabilities_json=excluded.capabilities_json, "
                "token_hash=excluded.token_hash, last_seen=excluded.last_seen, status='online'",
                (worker_id, protocol_version, canonical_json(dict(capabilities)), token_hash, "online", now, now),
            )
            self._conn.commit()

    def worker_token_hash(self, worker_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT token_hash FROM workers WHERE worker_id = ?", (worker_id,)
            ).fetchone()
        return None if row is None else str(row["token_hash"])

    def heartbeat(self, worker_id: str, status: str = "online") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE workers SET last_seen = ?, status = ? WHERE worker_id = ?",
                (utc_now_iso(), status, worker_id),
            )
            self._conn.commit()

    def workers(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM workers ORDER BY registered_at DESC").fetchall()
        return [
            {
                "worker_id": r["worker_id"],
                "protocol_version": r["protocol_version"],
                "capabilities": json.loads(r["capabilities_json"]),
                "status": r["status"],
                "last_seen": r["last_seen"],
                "registered_at": r["registered_at"],
            }
            for r in rows
        ]

    def claim_job(self, worker_id: str, *, lease_seconds: float = 120.0, now: float | None = None) -> JobRow | None:
        import time as _time

        now = now if now is not None else _time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE state IN ('QUEUED','DISPATCHED') "
                "AND (id NOT IN (SELECT job_id FROM leases WHERE expires_at > ?)) "
                "ORDER BY created_at ASC LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "INSERT OR REPLACE INTO leases(job_id, worker_id, expires_at) VALUES (?,?,?)",
                (row["id"], worker_id, now + lease_seconds),
            )
            self._conn.execute(
                "UPDATE jobs SET state = 'DISPATCHED', worker_id = ?, updated_at = ? WHERE id = ?",
                (worker_id, utc_now_iso(), row["id"]),
            )
            self._conn.commit()
            job_id = row["id"]
        self.append_event(job_id, "dispatched", {"worker_id": worker_id, "lease_seconds": lease_seconds})
        return self.get_job(job_id)

    def renew_lease(self, job_id: str, worker_id: str, *, lease_seconds: float = 120.0) -> bool:
        import time as _time

        with self._lock:
            row = self._conn.execute("SELECT * FROM leases WHERE job_id = ?", (job_id,)).fetchone()
            if row is None or row["worker_id"] != worker_id:
                return False
            self._conn.execute(
                "UPDATE leases SET expires_at = ? WHERE job_id = ?",
                (_time.time() + lease_seconds, job_id),
            )
            self._conn.commit()
        return True

    def release_lease(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM leases WHERE job_id = ?", (job_id,))
            self._conn.commit()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            jobs = {row["state"]: row["n"] for row in self._conn.execute(
                "SELECT state, COUNT(*) AS n FROM jobs GROUP BY state"
            ).fetchall()}
            assets = self._conn.execute("SELECT COUNT(*) AS n FROM artifacts_index").fetchone()["n"]
            workers = self._conn.execute("SELECT COUNT(*) AS n FROM workers").fetchone()["n"]
        return {"jobs_by_state": jobs, "artifacts_indexed": assets, "workers": workers, "db": str(self.path)}


def _row_to_job(row: sqlite3.Row) -> JobRow:
    return JobRow(
        id=row["id"],
        state=row["state"],
        task_scope=row["task_scope"],
        objective=row["objective"],
        request=json.loads(row["request_json"]),
        plan=json.loads(row["plan_json"]) if row["plan_json"] else None,
        result=json.loads(row["result_json"]) if row["result_json"] else None,
        error=json.loads(row["error_json"]) if row["error_json"] else None,
        progress=float(row["progress"]),
        cancel_requested=bool(row["cancel_requested"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
