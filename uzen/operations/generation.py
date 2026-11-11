"""Weight generation: analytic construction, structured random initialization,
constraint-matched synthesis, inherited composition and factor reconstruction.

Honesty rules enforced here:

* structured/random initialization is labelled ``randomly_initialized`` and records
  the seed — it is never presented as learned intelligence;
* construction from donor statistics is labelled ``analytically_derived`` and says
  which statistics were matched;
* factor reconstruction is labelled ``analytically_derived`` and reports the
  spectral energy actually captured.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from uzen.core.nwpc.registry import OperationSpec, ParamSpec, default_registry
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import InvalidOperationParameters, ShapeMismatch
from uzen.operations._common import float_view, make_outcome, selected_names
from uzen.policies.strict_plus.effects import EngineEffect, Origin, TaskScope
from uzen.util.linalg import low_rank_approximation, orthogonal_matrix

_SCOPES = frozenset(
    {
        TaskScope.WEIGHT_GENERATION,
        TaskScope.WEIGHT_RECONSTRUCTION,
        TaskScope.WEIGHT_MODIFICATION,
        TaskScope.WEIGHT_ARCHITECTURE_RESEARCH,
        TaskScope.MODEL_CREATION,
    }
)


def _structured_random(
    ctx,
    name: str,
    shape: tuple[int, ...],
    distribution: str,
    std: float | None,
    seed_path: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = ctx.rng(seed_path, name)
    if distribution in ("orthogonal", "semi_orthogonal"):
        if len(shape) < 2:
            # Vectors (biases/norms/gains) have no column space to orthogonalize; use a
            # scaled isotropic vector and say so in the metadata.
            scale = std if std is not None else 1.0 / np.sqrt(max(1, int(np.prod(shape))))
            return rng.standard_normal(shape) * scale, {
                "distribution": "isotropic_vector(rank-1 fallback)",
                "scale": float(scale),
                "seed": ctx.child_seed(seed_path, name),
            }
        rows = shape[0]
        cols = int(np.prod(shape[1:]))
        if rows >= cols:
            q = orthogonal_matrix(rows, cols, seed=ctx.child_seed(seed_path, name), method="qr")
        else:
            q = orthogonal_matrix(cols, rows, seed=ctx.child_seed(seed_path, name), method="qr").T
        fan = std if std is not None else 1.0 / np.sqrt(max(1, cols))
        return q * fan, {"distribution": "semi_orthogonal(qr)", "gain": float(fan), "seed": ctx.child_seed(seed_path, name)}
    if distribution == "xavier_uniform":
        fan_in, fan_out = _fans(shape)
        limit = np.sqrt(6.0 / max(1, fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape), {"distribution": "xavier_uniform", "limit": float(limit)}
    if distribution == "he_normal":
        fan_in, _ = _fans(shape)
        scale = std if std is not None else np.sqrt(2.0 / max(1, fan_in))
        return rng.standard_normal(shape) * scale, {"distribution": "he_normal", "scale": float(scale)}
    if distribution == "normal":
        scale = std if std is not None else 0.02
        return rng.standard_normal(shape) * scale, {"distribution": "normal", "scale": float(scale)}
    if distribution == "uniform":
        scale = std if std is not None else 0.02
        return rng.uniform(-scale, scale, size=shape), {"distribution": "uniform", "scale": float(scale)}
    if distribution == "zeros":
        return np.zeros(shape), {"distribution": "zeros"}
    raise InvalidOperationParameters(
        f"unknown distribution '{distribution}'; "
        "choose orthogonal|xavier_uniform|he_normal|normal|uniform|zeros"
    )


def _fans(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 1:
        return shape[0], shape[0]
    fan_in = int(np.prod(shape[1:]))
    fan_out = int(shape[0])
    return fan_in, fan_out


def _gen_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    template = inputs[0]
    names = selected_names(template, p.get("select"))
    distribution = str(p.get("distribution", "orthogonal"))
    std = p.get("std")
    out: dict[str, np.ndarray] = {}
    reports: dict[str, Any] = {}
    for name in names:
        shape = tuple(int(d) for d in template.tensors[name].shape)
        arr, diag = _structured_random(ctx, name, shape, distribution, std, "generate")
        out[name] = ctx.emit(arr.astype(template.tensors[name].dtype, copy=False), name=name)
        reports[name] = {**diag, "shape": list(shape)}
    return make_outcome(
        op="w.gen.structured",
        tensors=out,
        replaced=names,
        origin=Origin.RANDOMLY_INITIALIZED,
        details={
            "distribution": distribution,
            "seed": ctx.seed,
            "num_tensors": len(names),
            "per_tensor": reports,
            "note": (
                "STRUCTURED RANDOM INITIALIZATION — these parameters are not learned. "
                "No claim is made that they encode useful intelligence."
            ),
        },
    )


def _constraint_matched_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    template, statistics_source = inputs[0], inputs[1]
    names = selected_names(template, p.get("select"))
    out: dict[str, np.ndarray] = {}
    reports: dict[str, Any] = {}
    for name in names:
        shape = tuple(int(d) for d in template.tensors[name].shape)
        if name in statistics_source and statistics_source.tensors[name].shape == shape:
            donor = float_view(statistics_source.tensors[name])
            mean, std = float(donor.mean()), float(donor.std())
            rng = ctx.rng("constraint_matched", name)
            sample = rng.standard_normal(shape) * std + mean
            source = "donor_statistics"
        else:
            donor = None
            sample, _ = _structured_random(ctx, name, shape, "normal", None, "constraint_matched_fallback")
            mean, std = float(sample.mean()), float(sample.std())
            source = "fallback_random"
        out[name] = ctx.emit(sample.astype(template.tensors[name].dtype, copy=False), name=name)
        reports[name] = {"source": source, "target_mean": mean, "target_std": std}
    return make_outcome(
        op="w.gen.constraint_matched",
        tensors=out,
        replaced=names,
        origin=Origin.ANALYTICALLY_DERIVED,
        details={
            "seed": ctx.seed,
            "num_tensors": len(names),
            "per_tensor": reports,
            "note": (
                "tensors constructed to match per-tensor first/second moments of a donor; "
                "distributional matching does not reproduce learned function"
            ),
        },
    )


def _from_factors_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    left_pkg, right_pkg = inputs[0], inputs[1]
    suffix_a = str(p.get("suffix_a", ".factor_a"))
    suffix_b = str(p.get("suffix_b", ".factor_b"))
    out: dict[str, np.ndarray] = {}
    reports: dict[str, Any] = {}
    for name in left_pkg.names():
        if not name.endswith(suffix_a):
            continue
        base = name[: -len(suffix_a)]
        partner = f"{base}{suffix_b}"
        if partner not in right_pkg:
            raise InvalidOperationParameters(f"missing factor '{partner}' for '{name}'")
        a = float_view(left_pkg.tensors[name])
        b = float_view(right_pkg.tensors[partner])
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ShapeMismatch(
                f"factor shapes incompatible for '{base}': {a.shape} @ {b.shape}"
            )
        product = a @ b
        target = f"{base}.weight"
        out[target] = ctx.emit(product.astype(left_pkg.tensors[name].dtype, copy=False), name=target)
        reports[base] = {
            "rank": int(a.shape[1]),
            "shape": list(product.shape),
            "reconstruction_error": 0.0,
        }
    if not out:
        raise InvalidOperationParameters(
            f"no '{suffix_a}' tensors found to reconstruct from"
        )
    return make_outcome(
        op="w.gen.from_factors",
        tensors=out,
        replaced=set(),
        added=set(out),
        origin=Origin.ANALYTICALLY_DERIVED,
        details={
            "num_reconstructed": len(out),
            "per_tensor": reports,
            "note": "exact algebraic reconstruction from low-rank factor pairs",
        },
    )


def _low_rank_regeneration_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    template, donor = inputs[0], inputs[1]
    names = selected_names(template, p.get("select"))
    rank = int(p["rank"])
    out: dict[str, np.ndarray] = {}
    captured: list[float] = []
    skipped: list[str] = []
    for name in names:
        if name not in donor or donor.tensors[name].shape != template.tensors[name].shape:
            skipped.append(name)
            continue
        arr = float_view(donor.tensors[name])
        if arr.ndim != 2:
            skipped.append(name)
            continue
        approx, diag = low_rank_approximation(arr, rank)
        captured.append(float(diag["spectral_energy_captured"]))
        out[name] = ctx.emit(approx.astype(template.tensors[name].dtype, copy=False), name=name)
    if not out:
        raise InvalidOperationParameters("no compatible 2-D tensors were found in the donor")
    return make_outcome(
        op="w.gen.low_rank_regeneration",
        tensors=out,
        replaced=set(out),
        origin=Origin.ANALYTICALLY_DERIVED,
        details={
            "rank": rank,
            "num_regenerated": len(out),
            "skipped": skipped,
            "mean_energy_captured": float(np.mean(captured)) if captured else 0.0,
            "note": "rank-truncated regeneration of donor parameters (lossy, quantified)",
        },
    )


def _compose_donors_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    conflict = str(p.get("conflict", "error"))
    if conflict not in ("error", "first", "last", "average"):
        raise InvalidOperationParameters("conflict must be error|first|last|average")
    merged_tensors: dict[str, np.ndarray] = {}
    provenance: dict[str, str] = {}
    for index, pkg in enumerate(inputs):
        for name in pkg.names():
            if name not in merged_tensors:
                merged_tensors[name] = np.array(pkg.tensors[name], copy=True)
                provenance[name] = f"input[{index}]"
                continue
            existing = merged_tensors[name]
            incoming = pkg.tensors[name]
            if existing.shape != incoming.shape:
                raise ShapeMismatch(
                    f"tensor '{name}' has incompatible shapes across donors: "
                    f"{existing.shape} vs {incoming.shape}"
                )
            if conflict == "error":
                raise InvalidOperationParameters(
                    f"tensor '{name}' appears in more than one donor; "
                    "set conflict=first|last|average to resolve"
                )
            if conflict == "last":
                merged_tensors[name] = np.array(incoming, copy=True)
                provenance[name] = f"input[{index}]"
            elif conflict == "average":
                merged_tensors[name] = (
                    (float_view(existing) + float_view(incoming)) / 2.0
                ).astype(existing.dtype)
                provenance[name] = provenance[name] + f"+input[{index}]"
    out = {name: ctx.emit(arr, name=name) for name, arr in merged_tensors.items()}
    return make_outcome(
        op="w.gen.compose_donors",
        tensors=out,
        replaced=set(),
        added=set(out),
        origin=Origin.INHERITED,
        details={
            "num_donors": len(inputs),
            "conflict": conflict,
            "provenance": provenance,
            "note": "tensor sets composed from multiple donors; every tensor keeps its source",
        },
    )


def _register(spec: OperationSpec) -> None:
    default_registry().register(spec)


_register(OperationSpec(
    code="w.gen.structured",
    title="Structured random initialization",
    family="generation",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_gen_impl,
    description="Generate parameters matching a template's shapes. Not learned.",
    scopes=_SCOPES,
    params={
        "distribution": ParamSpec("str", default="orthogonal",
                                  choices=("orthogonal", "xavier_uniform", "he_normal", "normal", "uniform", "zeros")),
        "std": ParamSpec("float", default=None, minimum=0.0, maximum=1e6),
        "select": ParamSpec("list[str]", default=None),
    },
    assumptions=(
        "random initialization produces parameters without learned information",
        "orthogonal initialization requires rank>=2 tensors",
    ),
    origin=Origin.RANDOMLY_INITIALIZED,
))

_register(OperationSpec(
    code="w.gen.constraint_matched",
    title="Constraint-matched generation",
    family="generation",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_constraint_matched_impl,
    description="Construct parameters whose per-tensor moments match a donor.",
    scopes=_SCOPES,
    params={"select": ParamSpec("list[str]", default=None)},
    assumptions=("matching moments preserves scale behaviour, not learned function",),
    min_inputs=2,
    max_inputs=2,
    origin=Origin.ANALYTICALLY_DERIVED,
))

_register(OperationSpec(
    code="w.gen.from_factors",
    title="Reconstruct from factors",
    family="generation",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_from_factors_impl,
    description="Multiply factor pairs back into dense weight tensors.",
    scopes=_SCOPES,
    params={
        "suffix_a": ParamSpec("str", default=".factor_a"),
        "suffix_b": ParamSpec("str", default=".factor_b"),
    },
    assumptions=("factor_a (n x r) and factor_b (r x m) multiply into the target weight",),
    min_inputs=2,
    max_inputs=2,
    origin=Origin.ANALYTICALLY_DERIVED,
))

_register(OperationSpec(
    code="w.gen.low_rank_regeneration",
    title="Low-rank regeneration",
    family="generation",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_low_rank_regeneration_impl,
    description="Regenerate template-shaped weights from rank-truncated donor weights.",
    scopes=_SCOPES,
    params={"rank": ParamSpec("int", required=True, minimum=1), "select": ParamSpec("list[str]", default=None)},
    assumptions=("the donor tensor is a meaningful source for this parameter slot",),
    min_inputs=2,
    max_inputs=2,
    origin=Origin.ANALYTICALLY_DERIVED,
))

_register(OperationSpec(
    code="w.gen.compose_donors",
    title="Donor composition",
    family="generation",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_compose_donors_impl,
    description="Compose a parameter set out of several donors with explicit conflict policy.",
    scopes=_SCOPES,
    params={"conflict": ParamSpec("str", default="error", choices=("error", "first", "last", "average"))},
    assumptions=("tensors with the same name are semantically the same slot",),
    min_inputs=1,
    max_inputs=16,
    origin=Origin.INHERITED,
))
