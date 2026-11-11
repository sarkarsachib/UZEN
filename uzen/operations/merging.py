"""Weight merging operations.

Merging is mathematically meaningful only under conditions this module states
explicitly and checks where it can: identical parameter shapes (enforced), and —
for methods that combine *deltas* — a shared parameterization/initialization
lineage (stated as an assumption; not automatically provable).

Implements: linear soups, task arithmetic, TIES, DARE, magnitude-weighted merges
and a disjoint-sign merge. All randomness is seeded and recorded.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from uzen.core.nws.tensors import WeightPackage
from uzen.core.nwpc.registry import OperationSpec, ParamSpec, default_registry
from uzen.errors import InvalidOperationParameters, ShapeMismatch
from uzen.operations._common import (
    compute_dtype,
    float_view,
    intersect_names,
    make_outcome,
    require_float_tensors,
)
from uzen.policies.strict_plus.effects import EngineEffect, TaskScope
from uzen.util.linalg import safe_norm

WEIGHTS = ParamSpec("list[float]", default=None, description="per-input weights")
DENSITY = ParamSpec("float", default=0.2, minimum=0.0, maximum=1.0,
                    description="fraction of parameters kept (1.0 keeps everything)")
SELECT = ParamSpec("list[str]", default=None)
PARTIAL = ParamSpec("bool", default=False)


def _weights_for(p: dict[str, Any], n_models: int) -> np.ndarray:
    raw = p.get("weights")
    if raw is None:
        return np.ones(n_models, dtype=np.float64)
    if len(raw) != n_models:
        raise InvalidOperationParameters(
            f"expected {n_models} weights (one per non-base input), got {len(raw)}"
        )
    w = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(w)):
        raise InvalidOperationParameters("weights must be finite")
    return w


def _soup_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    names = intersect_names(inputs, p.get("select"), allow_partial=bool(p.get("partial")))
    require_float_tensors(inputs[0], names, op="w.merge.soup")
    w = _weights_for(p, len(inputs))
    w_sum = float(np.sum(w))
    if abs(w_sum) < 1e-12:
        raise InvalidOperationParameters("weights sum to zero; the soup would be undefined")
    out: dict[str, np.ndarray] = {}
    per_tensor: dict[str, Any] = {}
    for name in names:
        stacked = np.stack([float_view(pkg.tensors[name]) for pkg in inputs])
        dtype = compute_dtype(*[pkg.tensors[name] for pkg in inputs])
        merged = np.tensordot(w / w_sum, stacked, axes=(0, 0))
        out[name] = ctx.emit(np.asarray(merged).astype(dtype, copy=False), name=name)
        per_tensor[name] = {
            "shape": list(out[name].shape),
            "l2_norm": safe_norm(out[name]),
        }
    return make_outcome(
        op="w.merge.soup",
        tensors=out,
        replaced=names,
        details={
            "num_inputs": len(inputs),
            "weights": [float(v) for v in w],
            "normalized": True,
            "num_tensors": len(names),
            "per_tensor": per_tensor if p.get("report_per_tensor") else {},
            "note": "weighted average of parameter tensors (model soup)",
        },
    )


def _delta_stack(base: WeightPackage, models: Sequence[WeightPackage], name: str) -> np.ndarray:
    return np.stack([float_view(m.tensors[name]) - float_view(base.tensors[name]) for m in models])


def _ties_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    base, models = inputs[0], inputs[1:]
    if not models:
        raise InvalidOperationParameters("TIES requires a base package and at least one model")
    names = intersect_names([base, *models], p.get("select"), allow_partial=bool(p.get("partial")))
    require_float_tensors(base, names, op="w.merge.ties")
    w = _weights_for(p, len(models))
    density = float(p["density"])
    out: dict[str, np.ndarray] = {}
    stats: dict[str, Any] = {"kept_fraction": [], "agree_fraction": [], "elected_positive": [], "elected_negative": []}
    for name in names:
        deltas = _delta_stack(base, models, name)
        k = len(models)
        trim_masks = np.zeros_like(deltas)
        for i in range(k):
            flat = np.abs(deltas[i]).reshape(-1)
            keep = max(1, int(round(density * flat.size)))
            idx = np.argpartition(flat, -keep)[-keep:] if keep < flat.size else np.arange(flat.size)
            mask = np.zeros(flat.size, dtype=bool)
            mask[idx] = True
            trim_masks[i] = mask.reshape(deltas.shape[1:])
        trimmed = deltas * trim_masks
        signs = np.sign(trimmed)
        weighted_sign_sum = np.tensordot(w, signs, axes=(0, 0))
        elect = np.sign(weighted_sign_sum)
        agree = (signs == elect[None]) & (trimmed != 0)
        wgt = w.reshape((k,) + (1,) * (deltas.ndim - 1))
        numer = np.sum(trimmed * agree * wgt, axis=0)
        denom = np.sum(agree * wgt, axis=0)
        merged_delta = np.where(denom > 1e-12, numer / np.maximum(denom, 1e-12), 0.0)
        result = float_view(base.tensors[name]) + merged_delta
        dtype = compute_dtype(base.tensors[name], *[m.tensors[name] for m in models])
        out[name] = ctx.emit(np.asarray(result).astype(dtype, copy=False), name=name)
        stats["kept_fraction"].append(float(np.mean(trim_masks)))
        stats["agree_fraction"].append(float(np.mean(agree)))
        stats["elected_positive"].append(float(np.mean(elect > 0)))
        stats["elected_negative"].append(float(np.mean(elect < 0)))
    return make_outcome(
        op="w.merge.ties",
        tensors=out,
        replaced=names,
        details={
            "num_models": len(models),
            "weights": [float(v) for v in w],
            "density": density,
            "num_tensors": len(names),
            "statistics": stats,
            "note": (
                "TIES: magnitude trimming, majority sign election, then weighted mean of "
                "sign-consistent deltas relative to the base checkpoint"
            ),
        },
    )


def _dare_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    base, models = inputs[0], inputs[1:]
    if not models:
        raise InvalidOperationParameters("DARE requires a base package and at least one model")
    names = intersect_names([base, *models], p.get("select"), allow_partial=bool(p.get("partial")))
    require_float_tensors(base, names, op="w.merge.dare")
    w = _weights_for(p, len(models))
    density = float(p["density"])
    if density <= 0:
        raise InvalidOperationParameters("DARE density must be > 0")
    out: dict[str, np.ndarray] = {}
    kept: list[float] = []
    for index, name in enumerate(names):
        deltas = _delta_stack(base, models, name)
        rng = ctx.rng("dare", name, index)
        mask = rng.random(deltas.shape) < density
        kept.append(float(mask.mean()))
        dare = np.where(mask, deltas / density, 0.0)
        wgt = w.reshape((len(models),) + (1,) * (deltas.ndim - 1))
        merged = np.sum(dare * wgt, axis=0) / max(float(np.sum(w)), 1e-12)
        result = float_view(base.tensors[name]) + merged
        dtype = compute_dtype(base.tensors[name], *[m.tensors[name] for m in models])
        out[name] = ctx.emit(np.asarray(result).astype(dtype, copy=False), name=name)
    return make_outcome(
        op="w.merge.dare",
        tensors=out,
        replaced=names,
        details={
            "num_models": len(models),
            "weights": [float(v) for v in w],
            "density": density,
            "seed": ctx.seed,
            "mean_kept_fraction": float(np.mean(kept)),
            "num_tensors": len(names),
            "note": "DARE: seeded random drop of delta entries with 1/density rescaling",
        },
    )


def _task_arithmetic_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    base, models = inputs[0], inputs[1:]
    if not models:
        raise InvalidOperationParameters("task arithmetic requires a base and at least one task model")
    names = intersect_names([base, *models], p.get("select"), allow_partial=bool(p.get("partial")))
    require_float_tensors(base, names, op="w.merge.task_arithmetic")
    w = _weights_for(p, len(models))
    out: dict[str, np.ndarray] = {}
    magnitudes: dict[str, float] = {}
    for name in names:
        base_f = float_view(base.tensors[name])
        acc = np.zeros_like(base_f)
        for i, model in enumerate(models):
            acc += w[i] * (float_view(model.tensors[name]) - base_f)
        result = base_f + acc
        dtype = compute_dtype(base.tensors[name], *[m.tensors[name] for m in models])
        out[name] = ctx.emit(np.asarray(result).astype(dtype, copy=False), name=name)
        magnitudes[name] = float(safe_norm(acc) / (safe_norm(base_f) or 1.0))
    return make_outcome(
        op="w.merge.task_arithmetic",
        tensors=out,
        replaced=names,
        details={
            "num_models": len(models),
            "weights": [float(v) for v in w],
            "relative_delta_norms": magnitudes,
            "num_tensors": len(names),
            "note": "base + sum_i w_i * (model_i - base)",
        },
    )


def _disjoint_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    base, models = inputs[0], inputs[1:]
    names = intersect_names([base, *models], p.get("select"), allow_partial=bool(p.get("partial")))
    w = _weights_for(p, len(models))
    out: dict[str, np.ndarray] = {}
    conflict_rates: dict[str, float] = {}
    for name in names:
        deltas = _delta_stack(base, models, name)
        signs = np.sign(deltas)
        agree = np.all(signs == signs[0:1], axis=0)
        wgt = w.reshape((len(models),) + (1,) * (deltas.ndim - 1))
        merged = np.sum(deltas * agree[None] * wgt, axis=0) / max(float(np.sum(w)), 1e-12)
        result = float_view(base.tensors[name]) + merged
        dtype = compute_dtype(base.tensors[name], *[m.tensors[name] for m in models])
        out[name] = ctx.emit(np.asarray(result).astype(dtype, copy=False), name=name)
        conflict_rates[name] = float(1.0 - np.mean(agree))
    return make_outcome(
        op="w.merge.disjoint",
        tensors=out,
        replaced=names,
        details={
            "num_models": len(models),
            "weights": [float(v) for v in w],
            "conflict_rates": conflict_rates,
            "num_tensors": len(names),
            "note": "averages only sign-disjoint deltas; conflicting coordinates keep the base value",
        },
    )


def _register(spec: OperationSpec) -> None:
    default_registry().register(spec)


_COMMON_PARAMS = {
    "weights": WEIGHTS,
    "select": SELECT,
    "partial": PARTIAL,
    "report_per_tensor": ParamSpec("bool", default=False),
}
_MERGE_SCOPES = frozenset(
    {
        TaskScope.WEIGHT_MODIFICATION,
        TaskScope.WEIGHT_GENERATION,
        TaskScope.WEIGHT_RECONSTRUCTION,
    }
)

_register(OperationSpec(
    code="w.merge.soup",
    title="Model soup",
    family="merging",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_soup_impl,
    description="Weighted average of all input parameter sets.",
    scopes=_MERGE_SCOPES,
    params=_COMMON_PARAMS,
    assumptions=("all inputs share the same architecture and parameter shapes",),
    min_inputs=1,
    max_inputs=32,
))

_register(OperationSpec(
    code="w.merge.task_arithmetic",
    title="Task arithmetic",
    family="merging",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_task_arithmetic_impl,
    description="base + sum_i w_i * (model_i - base).",
    scopes=_MERGE_SCOPES,
    params=_COMMON_PARAMS,
    assumptions=(
        "inputs[0] is the common base checkpoint",
        "task models are fine-tuned from that base (shared parameterization)",
        "the deltas are expressed in the same basis; otherwise align parameters first",
    ),
    min_inputs=2,
    max_inputs=32,
))

_register(OperationSpec(
    code="w.merge.ties",
    title="TIES merging",
    family="merging",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_ties_impl,
    description="Trim, elect sign, then merge sign-consistent deltas (TIES).",
    scopes=_MERGE_SCOPES,
    params={**_COMMON_PARAMS, "density": DENSITY},
    assumptions=(
        "inputs[0] is the shared base checkpoint",
        "delta magnitudes are comparable across models",
    ),
    min_inputs=2,
    max_inputs=32,
))

_register(OperationSpec(
    code="w.merge.dare",
    title="DARE merging",
    family="merging",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_dare_impl,
    description="Seeded random delta dropping with rescaling, then weighted merge (DARE).",
    scopes=_MERGE_SCOPES,
    params={**_COMMON_PARAMS, "density": DENSITY},
    assumptions=(
        "inputs[0] is the shared base checkpoint",
        "delta sparsity is high enough that random drop + rescale is an unbiased estimator",
    ),
    min_inputs=2,
    max_inputs=32,
))

_register(OperationSpec(
    code="w.merge.disjoint",
    title="Disjoint sign merge",
    family="merging",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_disjoint_impl,
    description="Merge only coordinates whose deltas agree in sign across models.",
    scopes=_MERGE_SCOPES,
    params=_COMMON_PARAMS,
    assumptions=("inputs[0] is the shared base checkpoint",),
    min_inputs=2,
    max_inputs=32,
))
