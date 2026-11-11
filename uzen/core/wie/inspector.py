"""Weight Intelligence Engine (WIE).

Understands supplied neural artifacts before anything is transformed:

* safe format detection and loading (never unpickles untrusted checkpoints),
* header-only inspection of large safetensors files,
* tensor inventory, dtypes, shapes, roles, statistics and anomalies,
* architecture-family inference with explicit evidence (never a bare guess),
* parameter correspondence/compatibility analysis between two checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nws.roles import ParameterRole, canonical_stem, infer_role
from uzen.core.nws.tensors import WeightPackage
from uzen.core.nws.dtypes import dtype_name
from uzen.errors import UnsupportedFormat
from uzen.infrastructure.storage.safetensors_io import read_header, read_safetensors
from uzen.util.canonical import sha256_file, utc_now_iso
from uzen.util.linalg import cosine_similarity_matrix, gram_spectrum, tensor_statistics

SAFE_EXTENSIONS = {".safetensors": "safetensors", ".npz": "npz", ".gguf": "gguf"}
UNSAFE_EXTENSIONS = {
    ".pt": "PyTorch pickle container",
    ".pth": "PyTorch pickle container",
    ".bin": "PyTorch/legacy pickle container",
    ".ckpt": "checkpoint container (usually pickle)",
    ".pkl": "Python pickle",
    ".pickle": "Python pickle",
    ".h5": "HDF5 container",
    ".onnx": "ONNX protobuf model",
}


@dataclass
class InspectionReport:
    source: str
    format: str
    ok: bool
    tensors: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)
    family_guess: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    inspected_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "format": self.format,
            "ok": self.ok,
            "tensors": self.tensors,
            "summary": self.summary,
            "anomalies": self.anomalies,
            "family_guess": self.family_guess,
            "problems": self.problems,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "inspected_at": self.inspected_at,
        }


_FAMILY_SIGNATURES: dict[str, list[tuple[str, str]]] = {
    "transformer_decoder": [
        ("qkv|q_proj|query", "attention projections"),
        ("mlp|ffn|gate_proj|up_proj|down_proj|fc1|fc2", "feed-forward blocks"),
        ("embed_tokens|wte|tok_emb|embed", "token embeddings"),
        ("norm|ln_", "normalization layers"),
    ],
    "cnn": [
        ("conv", "convolutional kernels"),
        ("bn|batch_norm", "batch normalization"),
        ("pool", "pooling"),
    ],
    "recurrent": [
        ("lstm|gru|rnn|recurrent", "recurrent cells"),
    ],
    "mlp": [
        ("fc1|fc2|linear|dense", "linear stack"),
    ],
    "autoencoder": [
        ("encoder|decoder|enc\\.|dec\\.", "encoder/decoder halves"),
    ],
}


class WeightIntelligenceEngine:
    """Inspection and analysis front-end for neural artifacts."""

    # ------------------------------------------------------------------ #
    # files
    # ------------------------------------------------------------------ #

    def detect_format(self, path: str | Path) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        suffix = p.suffix.lower()
        if suffix in SAFE_EXTENSIONS:
            return SAFE_EXTENSIONS[suffix]
        if suffix in UNSAFE_EXTENSIONS:
            raise UnsupportedFormat(
                f"'{p.name}' looks like a {UNSAFE_EXTENSIONS[suffix]}. UZEN never unpickles "
                "untrusted checkpoints. Convert it to safetensors (e.g. with the PyTorch "
                "runtime adapter or `safetensors.torch.save_file`) and import the result.",
                path=str(p),
                detected=UNSAFE_EXTENSIONS[suffix],
            )
        # Sniff container magics for extensionless files.
        try:
            with open(p, "rb") as handle:
                if handle.read(4) == b"GGUF":
                    return "gguf"
        except OSError:
            pass
        # Sniff the safetensors header length field.
        try:
            header = read_header(p)
            if header.num_tensors >= 0:
                return "safetensors"
        except Exception:
            pass
        raise UnsupportedFormat(
            f"unrecognised checkpoint format for '{p.name}'; supported: .safetensors, .npz",
            path=str(p),
        )

    def inspect_file(
        self,
        path: str | Path,
        *,
        full: bool = True,
        load_tensors: bool = True,
    ) -> InspectionReport:
        fmt = self.detect_format(path)
        p = Path(path)
        if fmt == "safetensors":
            header = read_header(p, compute_sha=True)
            summary = header.summary()
            tensors: list[dict[str, Any]] = []
            if not load_tensors:
                for name, entry in sorted(header.entries.items()):
                    tensors.append(
                        {
                            "name": name,
                            "shape": list(entry.shape),
                            "dtype": entry.dtype_code,
                            "nbytes": entry.nbytes,
                            "role": infer_role(name, entry.shape).value,
                        }
                    )
                return InspectionReport(
                    source=str(p),
                    format=fmt,
                    ok=True,
                    tensors=tensors,
                    summary=summary,
                    metadata=dict(header.metadata),
                    family_guess=self._guess_family_from_names([t["name"] for t in tensors]),
                )
            arrays, _ = read_safetensors(p)
            package = WeightPackage.from_arrays(arrays, name=p.stem, family="imported")
            report = self.inspect_package(package, stats=full)
            report.source = str(p)
            report.format = fmt
            report.summary.update(summary)
            report.metadata = dict(header.metadata)
            return report
        if fmt == "npz":
            data = np.load(p, allow_pickle=False)
            arrays = {k: np.asarray(data[k]) for k in data.files}
            package = WeightPackage.from_arrays(arrays, name=p.stem, family="imported")
            report = self.inspect_package(package, stats=full)
            report.source = str(p)
            report.format = fmt
            report.metadata = {"sha256": sha256_file(p)}
            return report
        if fmt == "gguf":
            from uzen.infrastructure.storage.gguf_io import read_gguf

            arrays, gguf_metadata = read_gguf(p)
            package = WeightPackage.from_arrays(arrays, name=p.stem, family="imported")
            report = self.inspect_package(package, stats=full)
            report.source = str(p)
            report.format = fmt
            report.metadata = {**gguf_metadata, "sha256": sha256_file(p)}
            return report
        raise UnsupportedFormat(f"unsupported format '{fmt}'")

    # ------------------------------------------------------------------ #
    # packages
    # ------------------------------------------------------------------ #

    def inspect_package(self, package: WeightPackage, *, stats: bool = True) -> InspectionReport:
        tensors: list[dict[str, Any]] = []
        anomalies: list[str] = []
        warnings: list[str] = []
        for name in package.names():
            arr = package.tensors[name]
            entry: dict[str, Any] = {
                "name": name,
                "shape": list(arr.shape),
                "dtype": dtype_name(arr.dtype),
                "role": package.metas[name].role.value,
                "origin": package.metas[name].origin.value,
                "nbytes": int(arr.nbytes),
            }
            if stats:
                entry["statistics"] = tensor_statistics(arr)
                if arr.ndim == 2 and min(arr.shape) >= 2:
                    entry["spectrum"] = gram_spectrum(arr, top_k=4)
            tensors.append(entry)
            if arr.dtype.kind in "fc":
                if not np.isfinite(arr).all():
                    nan = int(np.isnan(arr).sum())
                    inf = int(np.isinf(arr).sum())
                    anomalies.append(f"{name}: {nan} NaN and {inf} infinite values")
                elif arr.size and float(np.std(arr)) == 0.0:
                    warnings.append(f"{name}: constant tensor (std=0)")
                elif arr.size and float(np.std(arr)) < 1e-9:
                    warnings.append(f"{name}: near-constant tensor")
        summary = package.summary(stats=False)
        summary["total_numel"] = package.numel()
        summary["total_nbytes"] = package.nbytes()
        return InspectionReport(
            source=package.source_artifact_id or package.name,
            format="package",
            ok=not anomalies,
            tensors=tensors,
            summary=summary,
            anomalies=anomalies,
            warnings=warnings,
            family_guess=self._guess_family(package),
            problems=list(anomalies),
        )

    def analyze_tensors(self, package: WeightPackage) -> dict[str, Any]:
        per_tensor: dict[str, Any] = {}
        by_role: dict[str, dict[str, float]] = {}
        for name in package.names():
            arr = package.tensors[name]
            stats = tensor_statistics(arr)
            per_tensor[name] = stats
            role = package.metas[name].role.value
            bucket = by_role.setdefault(role, {"numel": 0.0, "l2": 0.0, "abs_mean": 0.0, "count": 0.0})
            bucket["numel"] += float(stats.get("numel", 0))
            bucket["l2"] += float(stats.get("l2_norm", 0.0))
            bucket["abs_mean"] += float(stats.get("abs_mean", 0.0))
            bucket["count"] += 1.0
        for role, bucket in by_role.items():
            if bucket["count"]:
                bucket["mean_abs_mean"] = bucket["abs_mean"] / bucket["count"]
        return {
            "per_tensor": per_tensor,
            "by_role": by_role,
            "num_tensors": len(per_tensor),
            "numel": package.numel(),
            "note": "statistics are computed on the stored (possibly widened) arrays",
        }

    # ------------------------------------------------------------------ #
    # family inference
    # ------------------------------------------------------------------ #

    def _guess_family(self, package: WeightPackage) -> dict[str, Any]:
        return self._guess_family_from_names(package.names())

    def _guess_family_from_names(self, names: Sequence[str]) -> dict[str, Any]:
        import re

        scores: dict[str, list[str]] = {}
        for family, signatures in _FAMILY_SIGNATURES.items():
            evidence: list[str] = []
            for pattern, description in signatures:
                hits = [n for n in names if re.search(pattern, n.lower())]
                if hits:
                    evidence.append(f"{description}: {len(hits)} tensors (e.g. {hits[0]})")
            if evidence:
                scores[family] = evidence
        if not scores:
            return {
                "family": "unknown",
                "confidence": 0.0,
                "evidence": [],
                "note": "no known family signature matched; treat as a custom architecture",
            }
        ranked = sorted(scores.items(), key=lambda kv: len(kv[1]), reverse=True)
        family, evidence = ranked[0]
        coverage = len(evidence) / len(_FAMILY_SIGNATURES[family])
        return {
            "family": family,
            "confidence": round(min(0.95, 0.4 + 0.55 * coverage), 3),
            "evidence": evidence,
            "alternatives": [f for f, _ in ranked[1:]],
            "note": "inferred from tensor naming only; it is a hypothesis, not a guarantee",
        }

    # ------------------------------------------------------------------ #
    # compatibility
    # ------------------------------------------------------------------ #

    def compare(self, a: WeightPackage, b: WeightPackage) -> dict[str, Any]:
        names_a, names_b = set(a.names()), set(b.names())
        common = sorted(names_a & names_b)
        shape_mismatch = {
            n: {"a": list(a.tensors[n].shape), "b": list(b.tensors[n].shape)}
            for n in common
            if a.tensors[n].shape != b.tensors[n].shape
        }
        dtype_mismatch = {
            n: {"a": dtype_name(a.tensors[n].dtype), "b": dtype_name(b.tensors[n].dtype)}
            for n in common
            if a.tensors[n].dtype != b.tensors[n].dtype
        }
        role_conflicts = {
            n: {"a": a.metas[n].role.value, "b": b.metas[n].role.value}
            for n in common
            if a.metas[n].role != b.metas[n].role
        }
        comparable = [n for n in common if a.tensors[n].shape == b.tensors[n].shape and a.tensors[n].ndim >= 2]
        similarity: dict[str, float] = {}
        for name in comparable[:32]:
            x = a.tensors[name].reshape(a.tensors[name].shape[0], -1).astype(np.float64)
            y = b.tensors[name].reshape(b.tensors[name].shape[0], -1).astype(np.float64)
            if x.shape != y.shape:
                continue
            sim = cosine_similarity_matrix(x, y)
            similarity[name] = float(np.mean(np.diag(sim)))
        mean_similarity = float(np.mean(list(similarity.values()))) if similarity else 0.0
        same_structure = a.structure_digest() == b.structure_digest()
        return {
            "a": a.name,
            "b": b.name,
            "same_structure_digest": same_structure,
            "common_tensors": len(common),
            "only_in_a": sorted(names_a - names_b)[:50],
            "only_in_b": sorted(names_b - names_a)[:50],
            "shape_mismatches": shape_mismatch,
            "dtype_mismatches": dtype_mismatch,
            "role_conflicts": role_conflicts,
            "mean_diagonal_cosine": mean_similarity,
            "direct_arithmetic_applicable": bool(
                same_structure and common and not shape_mismatch and not dtype_mismatch
            ),
            "note": (
                "matching shapes are necessary but not sufficient for meaningful arithmetic: "
                "run NPA alignment before combining parameters from independent lineages"
            ),
            "recommended_next_step": (
                "none — structures match; still consider alignment if lineages differ"
                if same_structure and not shape_mismatch
                else "run w.align.auto (NPA) to establish parameter correspondence"
            ),
        }

    # ------------------------------------------------------------------ #
    # naming / correspondence helpers
    # ------------------------------------------------------------------ #

    def correspondence_map(self, source: WeightPackage, target: WeightPackage) -> dict[str, str]:
        """Map source tensor names onto target names using canonical stems + shapes."""
        target_by_stem: dict[str, list[str]] = {}
        for name in target.names():
            target_by_stem.setdefault(canonical_stem(name), []).append(name)
        mapping: dict[str, str] = {}
        for name in source.names():
            if name in target:
                mapping[name] = name
                continue
            candidates = target_by_stem.get(canonical_stem(name), [])
            exact = [c for c in candidates if target.tensors[c].shape == source.tensors[name].shape]
            if len(exact) == 1:
                mapping[name] = exact[0]
        return mapping

    def parameter_roles(self, package: WeightPackage) -> dict[str, str]:
        return {name: package.metas[name].role.value for name in package.names()}

    def find_tensors(
        self,
        package: WeightPackage,
        *,
        role: ParameterRole | None = None,
        pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        out = []
        for name in package.names():
            meta = package.metas[name]
            if role is not None and meta.role != role:
                continue
            if pattern is not None and pattern not in name:
                continue
            out.append(
                {
                    "name": name,
                    "shape": list(meta.shape),
                    "dtype": meta.dtype,
                    "role": meta.role.value,
                    "origin": meta.origin.value,
                }
            )
        return out
