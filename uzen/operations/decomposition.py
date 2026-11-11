"""Representation engineering: SVD, low-rank projection, orthogonal transforms,
permutation, basis changes and structured sparsification.

These operations manipulate the *parameterization* of existing weights without
changing the model's architecture. When a transformation changes tensor shapes
(e.g. genuine low-rank factorization into separate factors), the operation says so
and the caller must rebuild the graph — UZEN never silently alters architecture.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from uzen.core.nwpc.registry import OperationSpec, ParamSpec, default_registry
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import InvalidOperationParameters, ShapeMismatch
from uzen.operations._common import (
    float_view,
    make_outcome,
    require_float_tensors,
    selected_names,
    stats_delta,
)
from uzen.policies.strict_plus.effects import EngineEffect, Origin, TaskScope
from uzen.util.linalg import low_rank_approximation, orthogonal_matrix, topk_mask

SELECT = ParamSpec("list[str]", default=None)
_SCOPES = frozenset(
    {
        TaskScope.WEIGHT_MODIFICATION,
        TaskScope.WEIGHT_OPTIMIZATION,
        TaskScope.WEIGHT_RECONSTRUCTION,
        TaskScope.WEIGHT_ARCHITECTURE_RESEARCH,
    }
)


def _svd_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    pkg = inputs[0]
    names = [n for n in selected_names(pkg, p.get("select")) if pkg.tensors[n].ndim == 2]
    if not names:
        raise InvalidOperationParameters("SVD truncation requires at least one 2-D tensor")
    require_float_tensors(pkg, names, op="w.decomp.svd_truncate")
    rank = p.get("rank")
    energy = p.get("energy")
    if rank is None and energy is None:
        raise InvalidOperationParameters("provide either 'rank' or 'energy'")
    out: dict[str, np.ndarray] = {}
    per_tensor: dict[str, Any] = {}
    for name in names:
        arr = float_view(pkg.tensors[name])
        if rank is not None:
            k = int(rank)
        else:
            s = np.linalg.svd(arr, compute_uv=False)
            cum = np.cumsum(s**2) / max(1e-12, float(np.sum(s**2)))
            k = int(np.searchsorted(cum, float(energy)) + 1)
        approx, diag = low_rank_approximation(arr, k)
        per_tensor[name] = diag
        out[name] = ctx.emit(approx.astype(pkg.tensors[name].dtype, copy=False), name=name)
    captured = float(np.mean([v["spectral_energy_captured"] for v in per_tensor.values()]))
    return make_outcome(
        op="w.decomp.svd_truncate",
        tensors=out,
        replaced=names,
        origin=Origin.ANALYTICALLY_DERIVED,
        details={
            "num_tensors": len(names),
            "rank": rank,
            "energy": energy,
            "mean_energy_captured": captured,
            "per_tensor": per_tensor,
            "note": "Eckart-Young optimal rank-k reconstruction (shapes preserved)",
        },
    )


def _factor_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    """Produce separate low-rank factors (U, V) as new tensors.

    This intentionally changes the tensor inventory: it is a *weight architecture*
    operation, and consumers must rebuild the graph to use the factors.
    """
    pkg = inputs[0]
    names = [n for n in selected_names(pkg, p.get("select")) if pkg.tensors[n].ndim == 2]
    if not names:
        raise InvalidOperationParameters("factorization requires at least one 2-D tensor")
    rank = int(p["rank"])
    if rank <= 0:
        raise InvalidOperationParameters("rank must be positive")
    new_tensors: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    for name in names:
        arr = float_view(pkg.tensors[name])
        u, s, vt = np.linalg.svd(arr, full_matrices=False)
        k = min(rank, s.size)
        left = u[:, :k] * np.sqrt(s[:k])
        right = np.sqrt(s[:k])[:, None] * vt[:k, :]
        base = name.rsplit(".", 1)[0] if "." in name else name
        left_name = f"{base}.factor_a" if "." in name else f"{name}.factor_a"
        right_name = f"{base}.factor_b" if "." in name else f"{name}.factor_b"
        new_tensors[left_name] = ctx.emit(left.astype(pkg.tensors[name].dtype, copy=False), name=left_name)
        new_tensors[right_name] = ctx.emit(right.astype(pkg.tensors[name].dtype, copy=False), name=right_name)
        diag = low_rank_approximation(arr, k)[1]
        details[name] = {"rank": k, "factors": [left_name, right_name], **diag}
    return make_outcome(
        op="w.decomp.factorize",
        tensors=new_tensors,
        replaced=set(),
        added=set(new_tensors),
        dropped=set(names),
        origin=Origin.ANALYTICALLY_DERIVED,
        details={
            "num_factorized": len(names),
            "rank": rank,
            "per_tensor": details,
            "note": (
                "tensor shapes changed: weights were replaced by factor pairs. "
                "The computational graph must be rebuilt to consume factors; "
                "the architecture was NOT modified automatically"
            ),
        },
    )


def _orthogonal_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    pkg = inputs[0]
    names = [n for n in selected_names(pkg, p.get("select")) if pkg.tensors[n].ndim >= 2]
    if not names:
        raise InvalidOperationParameters("orthogonal transform requires rank>=2 tensors")
    method = str(p.get("method", "qr"))
    side = str(p.get("side", "right"))
    scale = float(p.get("scale", 1.0))
    out: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    for index, name in enumerate(names):
        arr = float_view(pkg.tensors[name])
        rows = int(arr.shape[0])
        cols = int(np.prod(arr.shape[1:]))
        seed = ctx.child_seed("orth", name, index)
        if side == "right":
            q = orthogonal_matrix(cols, cols, seed=seed, method=method)
            result = (arr.reshape(rows, cols) @ q).reshape(arr.shape) * scale
            diag = {"orthogonal_side": "right", "size": cols}
        elif side == "left":
            q = orthogonal_matrix(rows, rows, seed=seed, method=method)
            result = (q.T @ arr.reshape(rows, cols)).reshape(arr.shape) * scale
            diag = {"orthogonal_side": "left", "size": rows}
        else:
            raise InvalidOperationParameters("side must be 'left' or 'right'")
        # Verify orthogonality preservation numerically.
        q_check = orthogonal_matrix(diag["size"], diag["size"], seed=seed, method=method)
        diag["orthogonality_error"] = float(
            np.max(np.abs(q_check.T @ q_check - np.eye(diag["size"])))
        )
        diag["singular_value_drift"] = _sv_drift(arr, result)
        details[name] = diag
        out[name] = ctx.emit(result.astype(pkg.tensors[name].dtype, copy=False), name=name)
    return make_outcome(
        op="w.decomp.orthogonal_transform",
        tensors=out,
        replaced=names,
        origin=Origin.TRANSFORMED,
        details={
            "num_tensors": len(names),
            "method": method,
            "side": side,
            "scale": scale,
            "per_tensor": details,
            "note": "orthogonal basis change: singular values are preserved up to scale",
        },
    )


def _sv_drift(before: np.ndarray, after: np.ndarray) -> float:
    if before.ndim != 2:
        return 0.0
    a = np.linalg.svd(before, compute_uv=False)
    b = np.linalg.svd(after, compute_uv=False)
    n = min(a.size, b.size)
    denom = float(np.linalg.norm(a[:n])) or 1.0
    return float(np.linalg.norm(a[:n] - b[:n]) / denom)


def _permute_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    pkg = inputs[0]
    names = [n for n in selected_names(pkg, p.get("select")) if pkg.tensors[n].ndim >= 2]
    axis = int(p.get("axis", 0))
    explicit = p.get("permutation")
    out: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    for index, name in enumerate(names):
        arr = pkg.tensors[name]
        size = arr.shape[axis]
        if explicit is not None:
            perm = np.asarray(explicit, dtype=np.int64)
            if perm.size != size:
                raise InvalidOperationParameters(
                    f"permutation length {perm.size} != axis size {size} for '{name}'"
                )
            if sorted(perm.tolist()) != list(range(size)):
                raise InvalidOperationParameters("permutation must be a bijection of 0..n-1")
        else:
            perm = ctx.rng("permute", name, index).permutation(size)
        out[name] = ctx.emit(np.take(arr, perm, axis=axis), name=name)
        details[name] = {"axis": axis, "size": int(size), "permutation_digest": _perm_digest(perm)}
    return make_outcome(
        op="w.decomp.permute",
        tensors=out,
        replaced=names,
        origin=Origin.TRANSFORMED,
        details={
            "num_tensors": len(names),
            "axis": axis,
            "seeded": explicit is None,
            "per_tensor": details,
            "note": "axis permutation; a permutation alone does not change the represented function "
                    "when applied consistently across the whole model",
        },
    )


def _perm_digest(perm: np.ndarray) -> str:
    from uzen.util.canonical import sha256_bytes

    return sha256_bytes(perm.astype("<i8").tobytes())[:16]


def _sparsify_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    pkg = inputs[0]
    names = selected_names(pkg, p.get("select"))
    require_float_tensors(pkg, names, op="w.decomp.sparsify")
    fraction = float(p["fraction"])
    if fraction >= 1.0:
        raise InvalidOperationParameters("fraction must be < 1.0 to actually sparsify")
    mode = str(p.get("mode", "magnitude"))
    out: dict[str, np.ndarray] = {}
    kept: dict[str, float] = {}
    for index, name in enumerate(names):
        arr = float_view(pkg.tensors[name])
        if mode == "magnitude":
            mask = topk_mask(arr, fraction)
        elif mode == "random":
            mask = ctx.rng("sparsify", name, index).random(arr.shape) < fraction
        elif mode == "row":
            if arr.ndim < 2:
                out[name] = ctx.emit(arr.astype(pkg.tensors[name].dtype, copy=False), name=name)
                kept[name] = 1.0
                continue
            norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
            k = max(1, int(round(fraction * norms.size)))
            idx = np.argpartition(norms, -k)[-k:]
            rowwise = np.zeros(arr.shape[0], dtype=bool)
            rowwise[idx] = True
            mask = np.broadcast_to(rowwise.reshape((arr.shape[0],) + (1,) * (arr.ndim - 1)), arr.shape)
        else:
            raise InvalidOperationParameters("mode must be magnitude|random|row")
        masked = np.where(mask, arr, 0.0)
        kept[name] = float(np.mean(mask))
        out[name] = ctx.emit(masked.astype(pkg.tensors[name].dtype, copy=False), name=name)
    return make_outcome(
        op="w.decomp.sparsify",
        tensors=out,
        replaced=names,
        origin=Origin.TRANSFORMED,
        details={
            "num_tensors": len(names),
            "fraction": fraction,
            "mode": mode,
            "mean_kept_fraction": float(np.mean(list(kept.values()))) if kept else 0.0,
            "note": "structured/unstructured sparsification by zeroing; shapes are preserved "
                    "so the architecture is unchanged (storage savings require a sparse codec)",
        },
    )


def _register(spec: OperationSpec) -> None:
    default_registry().register(spec)


_register(OperationSpec(
    code="w.decomp.svd_truncate",
    title="Truncated SVD reconstruction",
    family="decomposition",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_svd_impl,
    description="Rank-k (or energy-targeted) SVD reconstruction of 2-D tensors.",
    scopes=_SCOPES,
    params={
        "rank": ParamSpec("int", default=None, minimum=1),
        "energy": ParamSpec("float", default=None, minimum=0.0, maximum=1.0),
        "select": SELECT,
    },
    assumptions=("a low-rank approximation is an acceptable representation for these weights",),
))

_register(OperationSpec(
    code="w.decomp.factorize",
    title="Low-rank factorization",
    family="decomposition",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_factor_impl,
    description="Replace 2-D weights by (factor_a, factor_b) pairs of the requested rank.",
    scopes=_SCOPES,
    params={"rank": ParamSpec("int", required=True, minimum=1), "select": SELECT},
    assumptions=(
        "the downstream graph will be rebuilt to consume factor pairs",
        "tensor shapes change, so this is a weight-architecture operation",
    ),
))

_register(OperationSpec(
    code="w.decomp.orthogonal_transform",
    title="Orthogonal basis transform",
    family="representation",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_orthogonal_impl,
    description="Apply a seeded orthogonal transform on the left or right of each tensor.",
    scopes=_SCOPES,
    params={
        "method": ParamSpec("str", default="qr", choices=("qr", "svd", "hadamard")),
        "side": ParamSpec("str", default="right", choices=("left", "right")),
        "scale": ParamSpec("float", default=1.0, minimum=-1e6, maximum=1e6),
        "select": SELECT,
    },
    assumptions=(
        "a change of basis that preserves singular values preserves representational capacity",
        "hadamard requires power-of-two dimensions",
    ),
))

_register(OperationSpec(
    code="w.decomp.permute",
    title="Axis permutation",
    family="representation",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_permute_impl,
    description="Permute an axis of each tensor (seeded or explicit bijection).",
    scopes=_SCOPES,
    params={
        "axis": ParamSpec("int", default=0, minimum=-8, maximum=8),
        "permutation": ParamSpec("list[int]", default=None),
        "select": SELECT,
    },
    assumptions=("permutation correspondence must be applied consistently elsewhere in the model",),
))

_register(OperationSpec(
    code="w.decomp.sparsify",
    title="Structured sparsification",
    family="structural",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_sparsify_impl,
    description="Zero low-magnitude entries (or whole rows) to a target density.",
    scopes=_SCOPES,
    params={
        "fraction": ParamSpec("float", required=True, minimum=0.0, maximum=1.0),
        "mode": ParamSpec("str", default="magnitude", choices=("magnitude", "random", "row")),
        "select": SELECT,
    },
    assumptions=("zeroing parameters is an acceptable approximation for this model",),
))
