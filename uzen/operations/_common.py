"""Shared helpers for weight operations.

These utilities keep tensor selection, shape agreement, dtype policy and outcome
construction identical across every operation family, so that an operation cannot
quietly diverge in how it treats mismatched structures.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from uzen.core.nwpc.registry import OperationOutcome
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import DTypeMismatch, ShapeMismatch
from uzen.util.linalg import is_finite, safe_norm


def selected_names(pkg: WeightPackage, select: Sequence[str] | None) -> list[str]:
    if not select:
        return pkg.names()
    names = pkg.select(*select).names()
    if not names:
        raise ShapeMismatch(
            f"selection {list(select)} matched no tensors in package '{pkg.name}'"
        )
    return names


def intersect_names(
    packages: Sequence[WeightPackage],
    select: Sequence[str] | None = None,
    *,
    allow_partial: bool = False,
) -> list[str]:
    """Names common to every package, validated for shape agreement."""
    if not packages:
        raise ShapeMismatch("at least one package is required")
    base = set(selected_names(packages[0], select))
    common = set(base)
    for pkg in packages[1:]:
        common &= set(selected_names(pkg, select))
    names = sorted(common)
    missing = base - common
    if missing and not allow_partial:
        raise ShapeMismatch(
            "packages do not share the same parameter set; missing in some package: "
            f"{sorted(missing)[:10]} (pass allow_partial=true to operate on the intersection)"
        )
    if not names:
        raise ShapeMismatch("packages share no tensors to operate on")
    for name in names:
        shapes = {tuple(pkg.tensors[name].shape) for pkg in packages}
        if len(shapes) > 1:
            raise ShapeMismatch(
                f"tensor '{name}' has inconsistent shapes across inputs: "
                + ", ".join(str(s) for s in sorted(shapes))
            )
    _check_dtypes(packages, names)
    return names


def _check_dtypes(packages: Sequence[WeightPackage], names: Sequence[str]) -> None:
    for name in names:
        kinds = {pkg.tensors[name].dtype.kind for pkg in packages}
        if len(kinds) > 1:
            raise DTypeMismatch(
                f"tensor '{name}' mixes dtype kinds {sorted(kinds)} across inputs"
            )
        if "c" in kinds:
            raise DTypeMismatch(f"tensor '{name}' is complex; complex tensors are unsupported")


def compute_dtype(*arrays: np.ndarray) -> np.dtype:
    dtypes = [np.dtype(a.dtype) for a in arrays]
    if all(d.kind in "iu" for d in dtypes):
        return dtypes[0]
    out = dtypes[0]
    for d in dtypes[1:]:
        out = np.promote_types(out, d)
    if out.kind in "iu":
        out = np.dtype(np.float32)
    return out


def float_view(arr: np.ndarray) -> np.ndarray:
    """Work in float64 internally; cast back only at the end."""
    if arr.dtype.kind in "fc":
        return arr.astype(np.float64, copy=False)
    return arr.astype(np.float64)


def stats_delta(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    b = float_view(before).reshape(-1)
    a = float_view(after).reshape(-1)
    delta = a - b
    bn, an = safe_norm(b), safe_norm(a)
    cos = float(np.dot(a, b) / (an * bn)) if an > 1e-12 and bn > 1e-12 else 0.0
    return {
        "shape": [int(d) for d in before.shape],
        "relative_l2_change": float(safe_norm(delta) / (bn or 1.0)),
        "cosine_to_source": cos,
        "mean_abs_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "finite": is_finite(after),
    }


def make_outcome(
    *,
    op: str,
    tensors: Mapping[str, np.ndarray],
    replaced: Iterable[str],
    details: Mapping[str, Any] | None = None,
    origin=None,
    alignment: Mapping[str, Any] | None = None,
    dropped: Iterable[str] = (),
    added: Iterable[str] = (),
    family_spec: Mapping[str, Any] | None = None,
) -> OperationOutcome:
    from uzen.policies.strict_plus.effects import Origin

    payload = dict(details or {})
    payload.setdefault("op", op)
    for name, arr in tensors.items():
        if np.asarray(arr).dtype.kind in "fc" and not np.isfinite(arr).all():
            raise ShapeMismatch(f"operation '{op}' produced non-finite values for '{name}'")
    return OperationOutcome(
        tensors={k: np.ascontiguousarray(v) for k, v in tensors.items()},
        replaced=set(replaced),
        dropped=set(dropped),
        added=set(added),
        details=payload,
        origin=origin or Origin.TRANSFORMED,
        alignment=dict(alignment) if alignment is not None else None,
        family_spec=dict(family_spec) if family_spec is not None else None,
    )


def require_float_tensors(package: WeightPackage, names: Sequence[str], *, op: str) -> None:
    for name in names:
        arr = package.tensors[name]
        if arr.dtype.kind not in "fc":
            raise DTypeMismatch(
                f"operation '{op}' requires floating-point tensors; '{name}' has {arr.dtype}"
            )


def flatten_2d(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(1, -1)
