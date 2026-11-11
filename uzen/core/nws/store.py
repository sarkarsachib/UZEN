"""Content-addressed, immutable artifact store.

Layout::

    <root>/
      artifacts/<aa>/<artifact_id>/
        manifest.json         # canonical, checksummed
        tensors.safetensors   # real safetensors payload
        lineage.json          # parent id + operation chain
      audit/audit.jsonl       # hash-chained policy audit (optional)

Guarantees
----------
* **Immutability**: an existing artifact directory is never rewritten. If new
  content maps onto an existing id, the stored content is verified; a mismatch
  raises :class:`ArtifactImmutable` instead of clobbering history.
* **Atomicity**: content is staged in a temp directory and renamed into place,
  so a crash cannot produce a half-written artifact.
* **Idempotency**: identical content produced twice yields one artifact, which
  makes worker retries safe.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from uzen.core.nws.manifest import (
    LineageRecord,
    PackageManifest,
    build_manifest,
    compute_artifact_id,
)
from uzen.core.nws.tensors import TensorMeta, WeightPackage
from uzen.errors import (
    ArtifactImmutable,
    ArtifactIntegrityError,
    ArtifactNotFound,
    UnsupportedFormat,
)
from uzen.infrastructure.provenance.lineage import LineageGraph, LineageNode
from uzen.infrastructure.storage.safetensors_io import (
    read_header,
    read_safetensors,
    safetensors_tensor_digest,
    write_safetensors,
)
from uzen.util.canonical import (
    atomic_publish_dir,
    atomic_write_json,
    ensure_dir,
    read_json,
    sha256_file,
    utc_now_iso,
)

MANIFEST_NAME = "manifest.json"
TENSORS_NAME = "tensors.safetensors"
LINEAGE_NAME = "lineage.json"


@dataclass
class ArtifactRecord:
    artifact_id: str
    path: Path
    manifest: PackageManifest

    @property
    def num_tensors(self) -> int:
        return int(self.manifest.summary.get("num_tensors", len(self.manifest.tensors)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": str(self.path),
            "name": self.manifest.name,
            "family": self.manifest.family,
            "num_tensors": self.num_tensors,
            "numel": self.manifest.summary.get("numel"),
            "created_at": self.manifest.created_at,
            "operation": self.manifest.lineage.operation,
            "parents": self.manifest.lineage.parents,
            "checksum": self.manifest.checksum,
        }


class ArtifactStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        ensure_dir(self.root / "artifacts")

    # ------------------------------------------------------------------ #
    # paths
    # ------------------------------------------------------------------ #

    def artifact_dir(self, artifact_id: str) -> Path:
        if not artifact_id or len(artifact_id) < 8 or not all(
            c in "0123456789abcdef" for c in artifact_id
        ):
            raise ArtifactNotFound(f"invalid artifact id: {artifact_id!r}")
        return self.root / "artifacts" / artifact_id[:2] / artifact_id

    def exists(self, artifact_id: str) -> bool:
        return (self.artifact_dir(artifact_id) / MANIFEST_NAME).exists()

    # ------------------------------------------------------------------ #
    # write
    # ------------------------------------------------------------------ #

    def put(
        self,
        package: WeightPackage,
        *,
        operation: str,
        parents: list[str] | None = None,
        operations: list[Mapping[str, Any]] | None = None,
        rng_seeds: Mapping[str, int] | None = None,
        policy: list[Mapping[str, Any]] | None = None,
        validation: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        engine: str = "nwpc.numpy",
        job_id: str | None = None,
        note: str = "",
        expected_artifact_id: str | None = None,
        reuse_existing: bool = False,
    ) -> ArtifactRecord:
        package.validate()
        parents = list(parents or [])
        operations = [dict(o) for o in (operations or [])]
        tensor_fp = package.fingerprint()
        artifact_id = compute_artifact_id(
            tensor_fingerprint=tensor_fp,
            family=package.family,
            operation=operation,
            parents=parents,
            operations=operations,
            seeds=rng_seeds,
        )
        if expected_artifact_id and expected_artifact_id != artifact_id:
            raise ArtifactIntegrityError(
                "artifact id does not match content",
                expected=expected_artifact_id,
                actual=artifact_id,
            )
        lineage = LineageRecord(
            operation=operation,
            parents=parents,
            operations=operations,
            rng_seeds=dict(rng_seeds or {}),
            engine=engine,
            job_id=job_id,
            note=note,
        )
        bf16_names = {
            n for n, m in package.metas.items() if m.source_dtype == "bfloat16"
        }
        # bfloat16 artifacts are stored widened (float32) with the true dtype
        # recorded in the manifest, so export re-encodes only when asked.
        manifest = build_manifest(
            package,
            artifact_id=artifact_id,
            lineage=lineage,
            policy=[* (policy or [])],
            validation=dict(validation or {}),
            metadata={
                **dict(metadata or {}),
                "stored_dtype_note": (
                    "bfloat16 tensors stored as float32 (lossless widening)"
                    if bf16_names
                    else "native dtypes"
                ),
            },
        )
        target = self.artifact_dir(artifact_id)

        if target.exists():
            stored = self._load_manifest(artifact_id)
            # The id is derived from tensor content plus the operation chain, parents and
            # seeds, so an existing artifact with this id is the same computation. The
            # stored bytes are the authority: verify them and reuse the record. Callers
            # persisting retried jobs pass ``reuse_existing=True`` so per-run bookkeeping
            # (such as the job id) may differ; the manifest keeps the first derivation
            # and each job record keeps its own evidence.
            tensors, _ = read_safetensors(target / TENSORS_NAME)
            if safetensors_tensor_digest(tensors) != tensor_fp:
                raise ArtifactImmutable(
                    f"artifact {artifact_id} exists but its stored payload does not match "
                    "the expected content; refusing to overwrite"
                )
            if stored.family != package.family:
                raise ArtifactImmutable(
                    f"artifact {artifact_id} exists with family '{stored.family}', but this "
                    f"write declares '{package.family}'; refusing to overwrite"
                )
            if not reuse_existing:

                def semantic(m) -> dict[str, Any]:
                    return {
                        "name": m.name,
                        "family": m.family,
                        "family_spec": m.family_spec,
                        "metadata": m.metadata,
                        "lineage": {
                            "operation": m.lineage.operation,
                            "parents": list(m.lineage.parents),
                            "operations": list(m.lineage.operations),
                        },
                    }

                if semantic(stored) != semantic(manifest):
                    raise ArtifactImmutable(
                        f"artifact {artifact_id} already exists with different provenance; "
                        "content-addressed artifacts are immutable"
                    )
            return ArtifactRecord(artifact_id=artifact_id, path=target, manifest=stored)

        def writer(staging: Path) -> None:
            write_safetensors(
                staging / TENSORS_NAME,
                package.tensors,
                metadata={
                    "artifact_id": artifact_id,
                    "family": package.family,
                    "name": package.name,
                    "format": "uzen",
                },
                bf16_tensors=set(),
                atomic=False,
            )
            atomic_write_json(staging / MANIFEST_NAME, manifest.to_dict())
            atomic_write_json(
                staging / LINEAGE_NAME,
                {
                    "artifact_id": artifact_id,
                    "parents": parents,
                    "operation": operation,
                    "operations": operations,
                    "created_at": manifest.created_at,
                },
            )

        atomic_publish_dir(target, writer)
        return ArtifactRecord(artifact_id=artifact_id, path=target, manifest=manifest)

    # ------------------------------------------------------------------ #
    # read
    # ------------------------------------------------------------------ #

    def _load_manifest(self, artifact_id: str) -> PackageManifest:
        d = self.artifact_dir(artifact_id)
        path = d / MANIFEST_NAME
        if not path.exists():
            raise ArtifactNotFound(f"artifact {artifact_id} not found in store {self.root}")
        try:
            return PackageManifest.from_dict(read_json(path))
        except (KeyError, ValueError) as exc:
            raise ArtifactIntegrityError(f"artifact {artifact_id} manifest is corrupt: {exc}") from exc

    def manifest(self, artifact_id: str) -> PackageManifest:
        return self._load_manifest(artifact_id)

    def get(self, artifact_id: str, *, tensors: list[str] | None = None) -> WeightPackage:
        manifest = self._load_manifest(artifact_id)
        d = self.artifact_dir(artifact_id)
        arrays, _ = read_safetensors(d / TENSORS_NAME)
        metas: dict[str, TensorMeta] = {}
        for entry in manifest.tensors:
            meta = TensorMeta.from_dict(entry)
            if meta.name in arrays:
                metas[meta.name] = meta
        package = WeightPackage(
            tensors=arrays,
            metas=metas,
            name=manifest.name,
            family=manifest.family,
            family_spec=dict(manifest.family_spec),
            metadata={**manifest.metadata, "artifact_id": artifact_id,
                      "provenance_operation": manifest.lineage.operation},
            source_artifact_id=artifact_id,
        )
        if tensors is not None:
            package = package.select(*tensors)
        return package

    def header(self, artifact_id: str) -> dict[str, Any]:
        d = self.artifact_dir(artifact_id)
        header = read_header(d / TENSORS_NAME)
        return header.summary()

    # ------------------------------------------------------------------ #
    # verification
    # ------------------------------------------------------------------ #

    def verify(self, artifact_id: str) -> dict[str, Any]:
        """Full integrity check: manifest checksum, payload digest, lineage, policy."""
        problems: list[str] = []
        d = self.artifact_dir(artifact_id)
        if not d.exists():
            raise ArtifactNotFound(f"artifact {artifact_id} not found")
        manifest = self._load_manifest(artifact_id)

        ok, reason = manifest.verify_checksum()
        if not ok:
            problems.append(reason or "manifest checksum failed")

        arrays, header = read_safetensors(d / TENSORS_NAME)
        fingerprint = safetensors_tensor_digest(arrays)
        if manifest.summary.get("fingerprint") and manifest.summary["fingerprint"] != fingerprint:
            problems.append("tensor payload digest does not match manifest")

        if header.metadata.get("artifact_id") != artifact_id:
            problems.append("embedded safetensors metadata artifact_id mismatch")

        declared = {t["name"]: tuple(t["shape"]) for t in manifest.tensors}
        for name, shape in declared.items():
            if name not in arrays:
                problems.append(f"tensor '{name}' declared but missing from payload")
            elif tuple(arrays[name].shape) != shape:
                problems.append(f"tensor '{name}' shape {arrays[name].shape} != declared {shape}")
        for name in arrays:
            if name not in declared:
                problems.append(f"tensor '{name}' present in payload but not declared")

        for name, arr in arrays.items():
            if arr.dtype.kind in "fc" and not np.isfinite(arr).all():
                problems.append(f"tensor '{name}' contains non-finite values")

        lineage_ok = bool(manifest.lineage.operation)
        if not lineage_ok:
            problems.append("lineage has no operation description")
        for parent in manifest.lineage.parents:
            if not self.exists(parent):
                problems.append(f"missing parent artifact {parent}")

        policy_ok = True
        prohibited = {"gradient_update", "optimizer_state", "data_training"}
        for decision in manifest.policy:
            if not decision.get("allowed", False):
                problems.append(f"policy record denies a stage: {decision.get('subject')}")
                policy_ok = False
            if decision.get("effect") in prohibited:
                problems.append(f"prohibited effect recorded: {decision.get('effect')}")
                policy_ok = False

        return {
            "artifact_id": artifact_id,
            "ok": not problems,
            "problems": problems,
            "tensor_digest": fingerprint,
            "declared_tensor_digest": manifest.summary.get("fingerprint"),
            "manifest_checksum_ok": ok,
            "lineage_ok": lineage_ok,
            "policy_ok": policy_ok,
            "num_tensors": len(arrays),
            "numel": int(sum(a.size for a in arrays.values())),
            "header": header.summary(),
            "checked_at": utc_now_iso(),
        }

    # ------------------------------------------------------------------ #
    # listing / export
    # ------------------------------------------------------------------ #

    def list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        base = self.root / "artifacts"
        if not base.exists():
            return out
        for prefix in sorted(base.iterdir()):
            if not prefix.is_dir():
                continue
            for art in sorted(prefix.iterdir()):
                mpath = art / MANIFEST_NAME
                if not mpath.exists():
                    continue
                try:
                    manifest = PackageManifest.from_dict(read_json(mpath))
                except Exception:  # pragma: no cover - corrupt dirs are reported, not raised
                    out.append({"artifact_id": art.name, "corrupt": True})
                    continue
                out.append(
                    {
                        "artifact_id": art.name,
                        "name": manifest.name,
                        "family": manifest.family,
                        "operation": manifest.lineage.operation,
                        "created_at": manifest.created_at,
                        "num_tensors": manifest.summary.get("num_tensors"),
                        "numel": manifest.summary.get("numel"),
                        "parents": manifest.lineage.parents,
                    }
                )
        return out

    def lineage_graph(self) -> LineageGraph:
        graph = LineageGraph()
        for item in self.list():
            if item.get("corrupt"):
                continue
            manifest = self._load_manifest(item["artifact_id"])
            graph.add(
                LineageNode(
                    artifact_id=item["artifact_id"],
                    parents=list(manifest.lineage.parents),
                    operation=manifest.lineage.operation,
                    operations=list(manifest.lineage.operations),
                    created_at=manifest.created_at,
                    family=manifest.family,
                )
            )
        return graph

    def export(
        self,
        artifact_id: str,
        destination: str | os.PathLike[str],
        *,
        fmt: str = "safetensors",
    ) -> Path:
        manifest = self._load_manifest(artifact_id)
        d = self.artifact_dir(artifact_id)
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "safetensors":
            shutil.copyfile(d / TENSORS_NAME, dest)
            return dest
        if fmt in ("npz", "numpy"):
            arrays, _ = read_safetensors(d / TENSORS_NAME)
            np.savez(dest, **arrays)
            return dest
        if fmt == "gguf":
            from uzen.infrastructure.storage.gguf_io import write_gguf

            arrays, _ = read_safetensors(d / TENSORS_NAME)
            write_gguf(
                dest,
                arrays,
                {
                    "general.architecture": manifest.family,
                    "uzen.artifact_id": artifact_id,
                    "uzen.operation": manifest.lineage.operation,
                },
            )
            return dest
        if fmt == "manifest":
            atomic_write_json(dest, manifest.to_dict())
            return dest
        if fmt == "artifact":
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(d, dest)
            return dest
        raise UnsupportedFormat(
            f"export format '{fmt}' is not implemented; supported: safetensors, npz, gguf, "
            "manifest, artifact"
        )

    def stats(self) -> dict[str, Any]:
        items = [i for i in self.list() if not i.get("corrupt")]
        total_numel = sum(int(i.get("numel") or 0) for i in items)
        return {
            "root": str(self.root),
            "artifacts": len(items),
            "total_numel": total_numel,
            "families": sorted({i["family"] for i in items}),
            "corrupt": sum(1 for i in self.list() if i.get("corrupt")),
        }
