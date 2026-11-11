"""Worker authentication and request signing.

Workers authenticate with a shared secret that is *never stored in plaintext* on the
control plane (only its SHA-256 hash is persisted). Every control-plane request is
HMAC-signed over canonical JSON of the envelope, with a timestamp and a nonce:

    signature = HMAC-SHA256(secret, canonical_json({...envelope...}))

Verification enforces:
  * signature match (constant-time comparison),
  * timestamp freshness (replay window),
  * nonce uniqueness within the window (replay guard),
  * worker identity presence.

This is deliberately simple, auditable transport security for worker↔control-plane
traffic; it is not a replacement for TLS, which should be terminated in front of the
service.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from uzen.errors import WorkerAuthError
from uzen.util.canonical import canonical_bytes, sha256_text

DEFAULT_WINDOW_SECONDS = 300


def generate_worker_secret() -> str:
    return secrets.token_urlsafe(48)


def hash_secret(secret: str) -> str:
    return sha256_text(secret)


def sign(secret: str, payload: Mapping[str, Any]) -> str:
    return hmac.new(secret.encode("utf-8"), canonical_bytes(dict(payload)), hashlib.sha256).hexdigest()


def verify_signature(secret: str, payload: Mapping[str, Any], signature: str) -> bool:
    expected = sign(secret, payload)
    return hmac.compare_digest(expected, signature)


class ReplayGuard:
    """In-memory nonce cache with expiry (single-node control plane)."""

    def __init__(self, window_seconds: int = DEFAULT_WINDOW_SECONDS) -> None:
        self.window = window_seconds
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def check_and_store(self, nonce: str, *, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        with self._lock:
            expired = [k for k, ts in self._seen.items() if now - ts > self.window]
            for k in expired:
                self._seen.pop(k, None)
            if nonce in self._seen:
                return False
            self._seen[nonce] = now
            return True


@dataclass
class SignedEnvelope:
    worker_id: str
    timestamp: float
    nonce: str
    body: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "body": self.body,
        }


def build_envelope(worker_id: str, body: Mapping[str, Any], *, secret: str) -> dict[str, Any]:
    envelope = SignedEnvelope(
        worker_id=worker_id,
        timestamp=time.time(),
        nonce=secrets.token_hex(16),
        body=dict(body),
    ).to_payload()
    envelope["signature"] = sign(secret, envelope)
    return envelope


def verify_envelope(
    envelope: Mapping[str, Any],
    *,
    secret_resolver,
    replay_guard: ReplayGuard | None = None,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    now: float | None = None,
) -> dict[str, Any]:
    """Verify a signed envelope; returns the body. Raises WorkerAuthError."""
    now = now if now is not None else time.time()
    for key in ("worker_id", "timestamp", "nonce", "signature"):
        if key not in envelope:
            raise WorkerAuthError(f"signed envelope is missing '{key}'")
    worker_id = str(envelope["worker_id"])
    secret = secret_resolver(worker_id)
    if not secret:
        raise WorkerAuthError(f"unknown worker '{worker_id}'", worker_id=worker_id)
    body = {k: v for k, v in envelope.items() if k != "signature"}
    if not verify_signature(secret, body, str(envelope["signature"])):
        raise WorkerAuthError("invalid worker signature", worker_id=worker_id)
    try:
        ts = float(envelope["timestamp"])
    except (TypeError, ValueError) as exc:
        raise WorkerAuthError("invalid envelope timestamp") from exc
    if abs(now - ts) > window_seconds:
        raise WorkerAuthError(
            f"envelope timestamp outside the {window_seconds}s replay window",
            worker_id=worker_id,
        )
    if replay_guard is not None and not replay_guard.check_and_store(str(envelope["nonce"]), now=now):
        raise WorkerAuthError("replayed nonce", worker_id=worker_id)
    return dict(envelope.get("body") or {})


class WorkerCredentialStore:
    """Holds worker secrets for the control plane (hash only) and signing side."""

    def __init__(self) -> None:
        self._hashes: dict[str, str] = {}
        self._lock = threading.Lock()

    def register(self, worker_id: str, secret: str) -> str:
        with self._lock:
            self._hashes[worker_id] = hash_secret(secret)
        return self._hashes[worker_id]

    def token_hash(self, worker_id: str) -> str | None:
        with self._lock:
            return self._hashes.get(worker_id)

    def __contains__(self, worker_id: object) -> bool:
        with self._lock:
            return worker_id in self._hashes
