"""Weight reconstruction: filling missing parameters, renaming/rebinding tensors and
reconstructing a complete parameter set from partial components.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from uzen.core.nwpc.registry import OperationSpec, ParamSpec, default_registry
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import InvalidOperationParameters, ShapeMismatch
from uzen.operations._common import float_view, make_outcome
from uzen.policies.strict_plus.effects import EngineEffect, Origin, TaskScope
from uzen.util.linalg import low_rank_approximation

_SCOPES = frozenset(
    {
        TaskScope.WEIGHT_RECONSTRUCTION,
        TaskScope.WEIGHT_GENERATION,
        TaskScope.MODEL_RECONSTRUCTION,
        TaskScope.WEIGHT_MODIFICATION,
    }
)


def _fill_missing_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    template = inputs[0]
    donors = inputs[1:]
    allow_random = bool(p.get("allow_random_fill", False))
    distribution = str(p.get("distribution", "orthogonal"))
    filled: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    for name in template.names():
        shape = tuple(int(d) for d in template.tensors[name].shape)
        donor_hit = None
        for index, donor in enumerate(donors):
            if name in donor and tuple(donor.tensors[name].shape) == shape:
                donor_hit = (index, donor.tensors[name])
                break
        if donor_hit is not None:
            index, arr = donor_hit
            filled[name] = ctx.emit(np.array(arr, copy=True), name=name)
            sources[name] = f"donor[{index}]"
            continue
        if not allow_random:
            missing.append(name)
            continue
        rng = ctx.rng("fill", name)
        if len(shape) >= 2 and distribution.startswith("orthogonal"):
            from uzen.util.linalg import orthogonal_matrix

            rows, cols = shape[0], int(np.prod(shape[1:]))
            q = orthogonal_matrix(rows, cols, seed=ctx.child_seed("fill", name), method="qr") if rows >= cols else orthogonal_matrix(cols, rows, seed=ctx.child_seed("fill", name), method="qr").T
            arr = q * (1.0 / np.sqrt(max(1, cols)))
            arr = arr.reshape(shape)
        else:
            arr = rng.standard_normal(shape) * 0.02
        filled[name] = ctx.emit(arr.astype(template.tensors[name].dtype, copy=False), name=name)
        sources[name] = f"structured_random({distribution})"
    if missing:
        raise InvalidOperationParameters(
            "cannot reconstruct: no donor provides these parameters and random fill is disabled: "
            f"{missing[:10]}",
            missing=missing,
        )
    random_tensors = [n for n, s in sources.items() if s.startswith("structured_random")]
    return make_outcome(
        op="w.recon.fill_missing",
        tensors=filled,
        replaced=set(filled),
        origin=Origin.RECONSTRUCTED,
        details={
            "num_tensors": len(filled),
            "sources": sources,
            "randomly_filled": random_tensors,
            "seed": ctx.seed,
            "note": (
                "missing tensors were filled from donors where possible; tensors listed in "
                "randomly_filled are STRUCTURED RANDOM and carry no learned information"
            ),
        },
    )


def _rename_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    pkg = inputs[0]
    mapping: dict[str, str] = dict(p.get("mapping") or {})
    if not mapping:
        raise InvalidOperationParameters("mapping must be a non-empty {old: new} object")
    out: dict[str, np.ndarray] = {}
    renamed: dict[str, str] = {}
    drop: set[str] = set()
    for name in pkg.names():
        new_name = mapping.get(name, name)
        if new_name in out:
            raise ShapeMismatch(f"rename collision: multiple tensors map to '{new_name}'")
        out[new_name] = ctx.emit(np.array(pkg.tensors[name], copy=True), name=new_name)
        if new_name != name:
            renamed[name] = new_name
            drop.add(name)
    unknown = set(mapping) - set(pkg.names())
    if unknown:
        raise InvalidOperationParameters(f"mapping references unknown tensors {sorted(unknown)}")
    return make_outcome(
        op="w.recon.rename",
        tensors=out,
        replaced=set(renamed.values()),
        added=set(renamed.values()),
        dropped=drop,
        origin=Origin.RECONSTRUCTED,
        details={
            "renamed": renamed,
            "num_renamed": len(renamed),
            "note": "parameter names rebound to a different naming convention; values untouched",
        },
    )


def _stage_then_fill_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    """Reconstruct a template from a partial set using low-rank structure.

    For each missing template tensor, if a donor provides a tensor with the same
    trailing dimension, factorize the donor and re-expand to the required shape.
    Otherwise the operation fails explicitly (no guessing).
    """
    template, donor = inputs[0], inputs[1]
    rank = int(p.get("rank", 8))
    out: dict[str, np.ndarray] = {}
    methods: dict[str, str] = {}
    unresolved: list[str] = []
    for name in template.names():
        shape = tuple(int(d) for d in template.tensors[name].shape)
        if name in donor and tuple(donor.tensors[name].shape) == shape:
            out[name] = ctx.emit(np.array(donor.tensors[name], copy=True), name=name)
            methods[name] = "donor_direct"
            continue
        candidates = [
            n for n in donor.names()
            if donor.tensors[n].ndim == 2 and template.tensors[name].ndim == 2
            and donor.tensors[n].shape[1] == shape[1]
        ]
        if not candidates:
            unresolved.append(name)
            continue
        source = donor.tensors[sorted(candidates)[0]]
        approx, _ = low_rank_approximation(float_view(source), rank)
        # resize rows deterministically to the required output dimension
        if approx.shape[0] != shape[0]:
            reps = int(np.ceil(shape[0] / approx.shape[0]))
            approx = np.tile(approx, (reps, 1))[: shape[0]]
        out[name] = ctx.emit(approx.astype(template.tensors[name].dtype, copy=False), name=name)
        methods[name] = f"low_rank_from:{sorted(candidates)[0]}"
    if unresolved:
        raise InvalidOperationParameters(
            "reconstruction could not resolve these tensors from the supplied donor: "
            f"{unresolved[:10]}",
            unresolved=unresolved,
        )
    return make_outcome(
        op="w.recon.low_rank_fill",
        tensors=out,
        replaced=set(out),
        origin=Origin.RECONSTRUCTED,
        details={
            "methods": methods,
            "rank": rank,
            "num_tensors": len(out),
            "note": (
                "reconstruction used donor factors; row resizing is a deterministic structural "
                "guess and is reported so it can be audited"
            ),
        },
    )


def _register(spec: OperationSpec) -> None:
    default_registry().register(spec)


_register(OperationSpec(
    code="w.recon.fill_missing",
    title="Fill missing parameters",
    family="reconstruction",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_fill_missing_impl,
    description="Complete a template parameter set from donor packages, optionally random-filling.",
    scopes=_SCOPES,
    params={
        "allow_random_fill": ParamSpec("bool", default=False),
        "distribution": ParamSpec("str", default="orthogonal",
                                  choices=("orthogonal", "normal", "xavier_uniform", "he_normal")),
    },
    assumptions=(
        "a donor with a same-named, same-shaped tensor is semantically compatible",
        "random fill produces non-learned parameters and is labelled as such",
    ),
    min_inputs=1,
    max_inputs=16,
    origin=Origin.RECONSTRUCTED,
))

_register(OperationSpec(
    code="w.recon.rename",
    title="Rebind tensor names",
    family="reconstruction",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_rename_impl,
    description="Rename tensors (e.g. to a target framework's naming convention).",
    scopes=_SCOPES,
    params={"mapping": ParamSpec("dict", required=True)},
    assumptions=("renaming does not change values; the target graph must use the new names",),
    origin=Origin.RECONSTRUCTED,
))

_register(OperationSpec(
    code="w.recon.low_rank_fill",
    title="Low-rank reconstruction",
    family="reconstruction",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_stage_then_fill_impl,
    description="Reconstruct a full parameter set from a partial donor via low-rank structure.",
    scopes=_SCOPES,
    params={"rank": ParamSpec("int", default=8, minimum=1)},
    assumptions=("missing parameters can be approximated from donor factors with matching width",),
    min_inputs=2,
    max_inputs=2,
    origin=Origin.RECONSTRUCTED,
))
