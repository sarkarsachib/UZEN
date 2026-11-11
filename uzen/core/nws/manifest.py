"""Artifact and lineage records.

An artifact is the durable result of an engineering job. Its manifest carries
everything needed to reproduce and audit it: tensor inventory, origin
classification, the operation sequence, the exact parameters used, RNG seeds, the
policy decisions that authorised each stage, validation evidence and the parent
artifacts it was derived from.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from uzen.core.nws.tensors import WeightPackage
from uzen.util.canonical import ContentHasher, canonical_bytes, sha256_bytes, utc_now_iso

SCHEMA_VERSION = "uzen.artifact/1"
UZEN_VERSION = "0.1.0"


def toolchain_info() -> dict[str, Any]:
    import numpy as np

    info: dict[str, Any] = {
        "uzen_version": UZEN_VERSION,
        "artifact_schema": SCHEMA_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:  # optional, recorded only when actually present
        import torch  # noqa: F401

        info["torch"] = torch.__version__
    except Exception:  # pragma: no cover - absence is the common case
        info["torch"] = None
    return info


@dataclass
class LineageRecord:
    """Directed lineage for one artifact."""

    operation: str
    parents: list[str] = field(default_factory=list)
    operations: list[dict[str, Any]] = field(default_factory=list)
    rng_seeds: dict[str, int] = field(default_factory=dict)
    engine: str = "nwpc.numpy"
    job_id: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "operation": self.operation,
            "parents": list(self.parents),
            "operations": list(self.operations),
            "rng_seeds": {str(k): int(v) for k, v in self.rng_seeds.items()},
            "engine": self.engine,
        }
        if self.job_id:
            out["job_id"] = self.job_id
        if self.note:
            out["note"] = self.note
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LineageRecord":
        return cls(
            operation=str(data.get("operation", "unknown")),
            parents=list(data.get("parents", [])),
            operations=list(data.get("operations", [])),
            rng_seeds={str(k): int(v) for k, v in (data.get("rng_seeds") or {}).items()},
            engine=str(data.get("engine", "nwpc.numpy")),
            job_id=data.get("job_id"),
            note=str(data.get("note", "")),
        )


def compute_artifact_id(
    *,
    tensor_fingerprint: str,
    family: str,
    operation: str,
    parents: Sequence[str],
    operations: Sequence[Mapping[str, Any]],
    seeds: Mapping[str, int] | None = None,
) -> str:
    """Deterministic content address for an artifact.

    Two artifacts with identical tensor content produced by the same operation
    chain from the same parents are the same artifact, and therefore get the same
    id — this is what makes storage idempotent and retries safe.
    """
    h = ContentHasher()
    h.field("schema", SCHEMA_VERSION)
    h.field("tensors", tensor_fingerprint)
    h.field("family", family)
    h.field("operation", operation)
    for p in parents:
        h.field("parent", p)
    h.field("ops", canonical_bytes(list(operations)))
    if seeds:
        h.field("seeds", canonical_bytes({str(k): int(v) for k, v in sorted(seeds.items())}))
    return h.hexdigest()


@dataclass
class PackageManifest:
    artifact_id: str
    name: str
    family: str
    family_spec: dict[str, Any]
    tensors: list[dict[str, Any]]
    lineage: LineageRecord
    policy: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    toolchain: dict[str, Any] = field(default_factory=toolchain_info)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "artifact_id": self.artifact_id,
            "name": self.name,
            "family": self.family,
            "family_spec": self.family_spec,
            "tensors": self.tensors,
            "lineage": self.lineage.to_dict(),
            "policy": self.policy,
            "validation": self.validation,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "toolchain": self.toolchain,
            "summary": self.summary,
        }
        if include_checksum:
            out["checksum"] = self.checksum or self.compute_checksum(out)
        return out

    @staticmethod
    def compute_checksum(body: Mapping[str, Any]) -> str:
        payload = {k: v for k, v in body.items() if k not in {"checksum"}}
        return sha256_bytes(canonical_bytes(payload))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PackageManifest":
        return cls(
            artifact_id=str(data["artifact_id"]),
            name=str(data.get("name", "weights")),
            family=str(data.get("family", "generic")),
            family_spec=dict(data.get("family_spec") or {}),
            tensors=list(data.get("tensors") or []),
            lineage=LineageRecord.from_dict(data.get("lineage") or {}),
            policy=list(data.get("policy") or []),
            validation=dict(data.get("validation") or {}),
            metadata=dict(data.get("metadata") or {}),
            checksum=str(data.get("checksum", "")),
            created_at=str(data.get("created_at", utc_now_iso())),
            toolchain=dict(data.get("toolchain") or {}),
            summary=dict(data.get("summary") or {}),
        )

    def verify_checksum(self) -> tuple[bool, str | None]:
        body = self.to_dict(include_checksum=False)
        expected = self.checksum or self.compute_checksum(body)
        actual = self.compute_checksum(body)
        if actual != expected:
            return False, f"manifest checksum mismatch: stored {expected}, recomputed {actual}"
        return True, None


def build_manifest(
    package: WeightPackage,
    *,
    artifact_id: str,
    lineage: LineageRecord,
    policy: Sequence[Mapping[str, Any]] = (),
    validation: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PackageManifest:
    body = {
        "schema": SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "name": package.name,
        "family": package.family,
        "family_spec": package.family_spec,
        "tensors": package.tensors_meta_list(),
        "lineage": lineage.to_dict(),
        "policy": list(policy),
        "validation": dict(validation or {}),
        "metadata": dict(metadata or {}),
        "created_at": utc_now_iso(),
        "toolchain": toolchain_info(),
        "summary": {
            "num_tensors": len(package),
            "numel": package.numel(),
            "nbytes": package.nbytes(),
            "fingerprint": package.fingerprint(),
            "structure_digest": package.structure_digest(),
        },
    }
    return PackageManifest(
        **{k: body[k] for k in (
            "artifact_id", "name", "family", "family_spec", "tensors", "policy",
            "validation", "metadata", "created_at", "toolchain", "summary",
        )},
        lineage=lineage,
        checksum=PackageManifest.compute_checksum(body),
    )
