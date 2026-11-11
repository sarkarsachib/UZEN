"""Verification and Evaluation Lab (VEL).

VEL separates four kinds of evidence that must never be conflated:

* **structural** — the parameter set is internally consistent and (when a reference
  is supplied) has the same architecture,
* **numerical** — values are finite, in range, and stable,
* **functional** — a real forward pass produces sane, comparable outputs,
* **performance** — measured latency, parameter count and activation footprint.

A passing report never asserts semantic equivalence between two models; it asserts
that the specific measured properties hold.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import ShapeMismatch, TensorInvalid
from uzen.util.canonical import utc_now_iso
from uzen.util.linalg import cosine_similarity_matrix, linear_cka, tensor_statistics

MAX_LATENCY_REPEATS = 10


@dataclass
class VerificationReport:
    name: str
    structural: dict[str, Any] = field(default_factory=dict)
    numerical: dict[str, Any] = field(default_factory=dict)
    functional: dict[str, Any] = field(default_factory=dict)
    regression: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_at: str = field(default_factory=utc_now_iso)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "problems": self.problems,
            "warnings": self.warnings,
            "structural": self.structural,
            "numerical": self.numerical,
            "functional": self.functional,
            "regression": self.regression,
            "performance": self.performance,
            "provenance": self.provenance,
            "policy": self.policy,
            "checked_at": self.checked_at,
        }

    def require(self) -> None:
        if self.problems:
            raise TensorInvalid(
                f"verification failed for '{self.name}'", problems=self.problems[:20]
            )


# --------------------------------------------------------------------------- #
# structural
# --------------------------------------------------------------------------- #


def structural_checks(
    package: WeightPackage,
    reference: WeightPackage | None = None,
    *,
    expect_same_architecture: bool = False,
) -> dict[str, Any]:
    problems: list[str] = []
    issues = package.issues(require_finite=True)
    problems.extend(issues)
    names = package.names()
    if not names:
        problems.append("package contains no tensors")
    shapes = {n: list(package.tensors[n].shape) for n in names}
    dtypes = {n: str(package.tensors[n].dtype) for n in names}
    result: dict[str, Any] = {
        "num_tensors": len(names),
        "numel": package.numel(),
        "nbytes": package.nbytes(),
        "dtypes": package.dtypes(),
        "structure_digest": package.structure_digest(),
        "shapes": shapes,
        "dtype_map": dtypes,
        "problems": problems,
    }
    if reference is not None:
        ref_names = set(reference.names())
        missing = sorted(ref_names - set(names))
        extra = sorted(set(names) - ref_names)
        shape_diff = {
            n: {"reference": list(reference.tensors[n].shape), "candidate": list(package.tensors[n].shape)}
            for n in sorted(ref_names & set(names))
            if reference.tensors[n].shape != package.tensors[n].shape
        }
        architecture_unchanged = not missing and not extra and not shape_diff
        result.update(
            {
                "reference_num_tensors": len(ref_names),
                "missing_tensors": missing,
                "extra_tensors": extra,
                "shape_mismatches": shape_diff,
                "architecture_unchanged": architecture_unchanged,
                "structure_digest_match": package.structure_digest()
                == reference.structure_digest(),
            }
        )
        if missing:
            problems.append(f"reference tensors missing from candidate: {missing[:10]}")
        if shape_diff:
            problems.append(f"tensor shapes changed for: {sorted(shape_diff)[:10]}")
        if expect_same_architecture and not architecture_unchanged:
            problems.append(
                "the task scope required an architecture-preserving weight modification, "
                "but the parameter structure changed"
            )
    result["ok"] = not problems
    return result


# --------------------------------------------------------------------------- #
# numerical
# --------------------------------------------------------------------------- #


def numerical_checks(
    package: WeightPackage,
    reference: WeightPackage | None = None,
) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []
    per_tensor: dict[str, Any] = {}
    for name in package.names():
        arr = package.tensors[name]
        stats = tensor_statistics(arr)
        if not stats.get("finite", True):
            problems.append(f"{name}: non-finite values (nan={stats.get('nan_count')}, inf={stats.get('inf_count')})")
        if arr.dtype.kind == "f" and arr.size and stats.get("finite", True):
            if stats["std"] == 0.0 and stats["mean"] != 0.0:
                warnings.append(f"{name}: constant tensor with non-zero mean (degenerate)")
            if stats.get("abs_mean", 0.0) > 1e6:
                warnings.append(f"{name}: very large magnitudes (abs_mean={stats['abs_mean']:.3g})")
        if reference is not None and name in reference and reference.tensors[name].shape == arr.shape:
            ref = reference.tensors[name].astype(np.float64)
            cur = arr.astype(np.float64)
            denom = float(np.linalg.norm(ref)) or 1.0
            rel = float(np.linalg.norm(cur - ref) / denom)
            fl_ref = ref.reshape(-1)
            fl_cur = cur.reshape(-1)
            n1, n2 = float(np.linalg.norm(fl_ref)), float(np.linalg.norm(fl_cur))
            cos = float(np.dot(fl_ref, fl_cur) / (n1 * n2)) if n1 > 1e-12 and n2 > 1e-12 else 0.0
            stats["relative_l2_vs_reference"] = rel
            stats["cosine_vs_reference"] = cos
            stats["changed"] = not np.array_equal(cur, ref, equal_nan=True)
        per_tensor[name] = stats
    result = {
        "per_tensor": per_tensor,
        "num_non_finite": sum(1 for s in per_tensor.values() if not s.get("finite", True)),
        "problems": problems,
        "warnings": warnings,
        "ok": not problems,
    }
    return result


# --------------------------------------------------------------------------- #
# functional
# --------------------------------------------------------------------------- #


def functional_checks(
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
    package: WeightPackage,
    reference: WeightPackage | None = None,
    *,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    problems: list[str] = []
    validation = graph.validate(package)
    if not validation.ok:
        problems.extend(f"graph invalid: {e}" for e in validation.errors[:10])
    try:
        result = execute(graph, inputs, package)
    except Exception as exc:
        return {
            "ok": False,
            "problems": problems + [f"forward pass failed: {exc}"],
            "outputs": {},
        }
    outputs: dict[str, Any] = {}
    for name, arr in result.outputs.items():
        stats = tensor_statistics(arr)
        outputs[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite": stats.get("finite", False),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "l2_norm": stats.get("l2_norm"),
        }
        if not stats.get("finite", False):
            problems.append(f"output '{name}' contains non-finite values")
    comparison: dict[str, Any] = {}
    if reference is not None:
        try:
            ref_result = execute(graph, inputs, reference)
        except Exception as exc:
            problems.append(f"reference forward pass failed: {exc}")
            ref_result = None
        if ref_result is not None:
            for name, arr in result.outputs.items():
                if name not in ref_result.outputs:
                    continue
                ref = ref_result.outputs[name]
                if ref.shape != arr.shape:
                    problems.append(f"output '{name}' shape {arr.shape} != reference {ref.shape}")
                    continue
                diff = np.abs(arr.astype(np.float64) - ref.astype(np.float64))
                denom = float(np.linalg.norm(ref.astype(np.float64))) or 1.0
                comparison[name] = {
                    "max_abs_diff": float(diff.max()) if diff.size else 0.0,
                    "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
                    "relative_l2_diff": float(np.linalg.norm(diff) / denom),
                    "allclose": bool(np.allclose(arr, ref, atol=tolerance, rtol=tolerance)),
                }
                if arr.ndim >= 2 and arr.shape[-1] > 1:
                    a2 = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
                    b2 = ref.reshape(-1, ref.shape[-1]).astype(np.float64)
                    comparison[name]["linear_cka"] = linear_cka(a2, b2)
                    if arr.shape[0] and arr.shape[-1] > 1:
                        # top-1 agreement is meaningful for logits-like outputs
                        agree = float(np.mean(np.argmax(a2, axis=1) == np.argmax(b2, axis=1)))
                        comparison[name]["argmax_agreement"] = agree
    return {
        "ok": not problems,
        "problems": problems,
        "outputs": outputs,
        "comparison": comparison,
        "num_ops_executed": result.num_ops,
        "evaluation_note": "functional checks use inference-mode forward passes only; no gradients",
    }


# --------------------------------------------------------------------------- #
# performance
# --------------------------------------------------------------------------- #


def performance_checks(
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
    package: WeightPackage,
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    repeats = max(1, min(repeats, MAX_LATENCY_REPEATS))
    latencies: list[float] = []
    activation_bytes = 0
    try:
        tracemalloc.start()
        for _ in range(repeats):
            start = time.perf_counter()
            execute(graph, inputs, package, check_finite=False)
            latencies.append(time.perf_counter() - start)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        activation_bytes = int(peak)
    except Exception as exc:  # pragma: no cover - reported, not swallowed
        return {"ok": False, "problems": [f"performance measurement failed: {exc}"]}
    latencies.sort()
    return {
        "ok": True,
        "latency_ms_median": round(latencies[len(latencies) // 2] * 1000.0, 3),
        "latency_ms_min": round(latencies[0] * 1000.0, 3),
        "latency_ms_max": round(latencies[-1] * 1000.0, 3),
        "parameter_numel": package.numel(),
        "parameter_bytes": package.nbytes(),
        "peak_traced_bytes": activation_bytes,
        "device": "cpu",
        "backend": "numpy",
        "repeats": repeats,
        "note": "measured on CPU with the NumPy reference runtime",
    }


# --------------------------------------------------------------------------- #
# policy / provenance
# --------------------------------------------------------------------------- #


def policy_checks(policy_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    problems: list[str] = []
    prohibited = {"gradient_update", "optimizer_state", "data_training"}
    for rec in policy_records:
        if rec.get("effect") in prohibited:
            problems.append(f"prohibited effect in execution history: {rec.get('effect')}")
        if rec.get("allowed") is False:
            problems.append(f"denied policy decision in execution history: {rec.get('subject')}")
    return {
        "ok": not problems,
        "problems": problems,
        "num_decisions": len(policy_records),
        "policy": "STRICT_PLUS",
        "effects": sorted({str(r.get("effect")) for r in policy_records}),
    }


def provenance_checks(
    package: WeightPackage,
    *,
    expected_parents: Sequence[str] = (),
    operations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    problems: list[str] = []
    origins: dict[str, str] = {}
    for name, meta in package.metas.items():
        origins[name] = meta.origin.value
        if meta.origin.value == "randomly_initialized" and "note" not in package.metadata:
            pass
    random_tensors = [n for n, o in origins.items() if o == "randomly_initialized"]
    transformed = [n for n, o in origins.items() if o in ("transformed", "analytically_derived", "extracted")]
    return {
        "ok": not problems,
        "problems": problems,
        "origins": {
            key: sum(1 for v in origins.values() if v == key)
            for key in sorted(set(origins.values()))
        },
        "randomly_initialized_tensors": random_tensors[:20],
        "parents": list(expected_parents),
        "num_operations": len(operations),
        "note": (
            "randomly_initialized tensors carry no learned information and are labelled as such"
            if random_tensors
            else "no randomly initialized tensors in this artifact"
        ),
    }


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #


def verify(
    package: WeightPackage,
    *,
    reference: WeightPackage | None = None,
    graph: Graph | None = None,
    inputs: Mapping[str, np.ndarray] | None = None,
    expect_same_architecture: bool = False,
    policy_records: Sequence[Mapping[str, Any]] = (),
    parents: Sequence[str] = (),
    operations: Sequence[Mapping[str, Any]] = (),
    measure_performance: bool = True,
) -> VerificationReport:
    report = VerificationReport(name=package.name)
    report.structural = structural_checks(
        package, reference, expect_same_architecture=expect_same_architecture
    )
    report.numerical = numerical_checks(package, reference)
    report.policy = policy_checks(policy_records) if policy_records else {"ok": True, "num_decisions": 0}
    report.provenance = provenance_checks(package, expected_parents=parents, operations=operations)

    if graph is not None and inputs is not None:
        report.functional = functional_checks(graph, inputs, package, reference)
        if measure_performance:
            report.performance = performance_checks(graph, inputs, package)
    else:
        report.functional = {
            "ok": True,
            "skipped": True,
            "reason": "no graph/inputs supplied; functional behaviour was not evaluated",
        }

    if reference is not None:
        report.regression = _regression(report.functional, report.structural)

    for section in ("structural", "numerical", "functional", "policy", "provenance"):
        part = getattr(report, section)
        report.problems.extend(part.get("problems", []) or [])
    report.problems.extend(report.performance.get("problems", []) or [])
    for section in ("numerical", "structural"):
        part = getattr(report, section)
        report.warnings.extend(part.get("warnings", []) or [])
    return report


def _regression(functional: Mapping[str, Any], structural: Mapping[str, Any]) -> dict[str, Any]:
    comparison = functional.get("comparison") or {}
    worst = 0.0
    for stats in comparison.values():
        worst = max(worst, float(stats.get("relative_l2_diff", 0.0)))
    return {
        "compared_outputs": list(comparison),
        "worst_relative_l2_diff": worst,
        "architecture_unchanged": structural.get("architecture_unchanged"),
        "all_outputs_allclose": all(
            bool(s.get("allclose")) for s in comparison.values()
        ) if comparison else None,
        "note": "regression compares forward outputs against a reference parameter set",
    }
