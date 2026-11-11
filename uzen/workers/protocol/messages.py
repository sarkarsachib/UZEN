"""Worker protocol messages and capability advertisement.

A worker advertises what it can actually do; the control plane dispatches only
compatible jobs. Capabilities are detected from the real environment (CPU count,
memory, Python/PyTorch versions, registered operation codes) rather than hardcoded.
"""

from __future__ import annotations

import os
import platform
import socket
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from uzen.util.canonical import canonical_bytes, sha256_bytes, utc_now_iso

PROTOCOL_VERSION = "uzen.worker/1"


def detect_capabilities() -> dict[str, Any]:
    from uzen.core.nir.ops import OPERATORS
    from uzen.core.nwpc.registry import default_registry
    from uzen.runtimes.pytorch_runtime import torch_available

    memory_bytes: int | None = None
    try:
        memory_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        memory_bytes = None
    cuda = False
    torch_version = None
    if torch_available():
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
            torch_version = torch.__version__
        except Exception:
            cuda = False
    registry = default_registry()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "device": "cuda" if cuda else "cpu",
        "cuda_available": cuda,
        "torch_version": torch_version,
        "pytorch_available": torch_available(),
        "safetensors": True,
        "supported_operations": sorted(spec.code for spec in registry.all()),
        "supported_architectures": ["transformer_decoder", "mlp", "cnn", "recurrent", "autoencoder"],
        "supported_operators": sorted(OPERATORS),
        "max_tensor_bytes": None,
        "artifact_transport": ["shared_filesystem", "inline_tensors"],
        "strict_plus": True,
        "detected_at": utc_now_iso(),
    }


@dataclass
class WorkerRegistration:
    worker_id: str
    protocol_version: str = PROTOCOL_VERSION
    capabilities: dict[str, Any] = field(default_factory=detect_capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "protocol_version": self.protocol_version,
            "capabilities": self.capabilities,
        }


@dataclass
class JobLease:
    job_id: str
    worker_id: str
    lease_seconds: float = 120.0
    acquired_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "worker_id": self.worker_id,
            "lease_seconds": self.lease_seconds,
            "acquired_at": self.acquired_at,
        }


@dataclass
class JobReport:
    job_id: str
    worker_id: str
    state: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "worker_id": self.worker_id,
            "state": self.state,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "reported_at": utc_now_iso(),
        }


@dataclass
class MessageEnvelope:
    kind: str
    payload: dict[str, Any]
    message_id: str
    sent_at: str = field(default_factory=utc_now_iso)

    def digest(self) -> str:
        return sha256_bytes(canonical_bytes({"kind": self.kind, "payload": self.payload}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": self.payload,
            "message_id": self.message_id,
            "sent_at": self.sent_at,
            "digest": self.digest(),
            "protocol_version": PROTOCOL_VERSION,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MessageEnvelope":
        envelope = cls(
            kind=str(data["kind"]),
            payload=dict(data.get("payload") or {}),
            message_id=str(data.get("message_id", "")),
            sent_at=str(data.get("sent_at", utc_now_iso())),
        )
        if data.get("digest") and data["digest"] != envelope.digest():
            raise ValueError("worker message digest mismatch")
        return envelope


def make_message(kind: str, payload: Mapping[str, Any]) -> MessageEnvelope:
    import uuid

    return MessageEnvelope(kind=kind, payload=dict(payload), message_id=str(uuid.uuid4()))
