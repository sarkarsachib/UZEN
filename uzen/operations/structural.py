"""Structural engineering of parameter sets: extraction, transplantation, tying,
head pruning and tensor slicing.

Structural operations that change tensor *shapes* are reported as such, because a
shape change is a model-architecture concern: the NIR graph must be updated before
the weights can execute. UZEN never rebinds a model silently.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any

import numpy as np

from uzen.core.nwpc.registry import OperationSpec, ParamSpec, default_registry
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import InvalidOperationParameters, ShapeMismatch
from uzen.operations._common import float_view, make_outcome, selected_names
from uzen.policies.strict_plus.effects import EngineEffect, Origin, TaskScope

_SCOPES = frozenset(
    {
        TaskScope.WEIGHT_MODIFICATION,
        TaskScope.WEIGHT_RECONSTRUCTION,
        TaskScope.WEIGHT_ARCHITECTURE_RESEARCH,
        TaskScope.MODEL_RECONSTRUCTION,
        TaskScope.MODEL_MODIFICATION,
    }
)


def _extract_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    pkg = inputs[0]
    layers = [int(i) for i in (p.get("layers") or [])]
    if not layers:
        raise InvalidOperationParameters("layers must be a non-empty list of layer indices")
    keep_extra = [str(x) for x in (p.get("keep") or [])]
    kept: list[str] = []
    for name in pkg.names():
        m = re.search(r"(?:layers?|h|blocks?)\.(\d+)\.", name)
        is_layer = m is not None and int(m.group(1)) in layers
        if is_layer or any(fnmatch.fnmatch(name, pat) for pat in keep_extra):
            kept.append(name)
    if not kept:
        raise InvalidOperationParameters(
            f"no tensors matched layers {layers} or keep patterns {keep_extra}"
        )
    dropped = [n for n in pkg.names() if n not in kept]
    return make_outcome(
        op="w.structural.extract",
        tensors={n: pkg.tensors[n] for n in kept},
        replaced=set(),
        dropped=dropped,
        origin=Origin.EXTRACTED,
        details={
            "layers": layers,
            "num_kept": len(kept),
            "num_dropped": len(dropped),
            "note": "subset of an existing parameter set; the graph must be rebuilt to match",
        },
        family_spec={**pkg.family_spec, "extracted_layers": layers},
    )


def _transplant_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    target, donor = inputs[0], inputs[1]
    mapping: dict[str, str] = dict(p.get("mapping") or {})
    if not mapping:
        raise InvalidOperationParameters("mapping must map target tensor name -> donor tensor name")
    out: dict[str, np.ndarray] = {}
    added: list[str] = []
    for target_name, donor_name in mapping.items():
        if donor_name not in donor:
            raise InvalidOperationParameters(
                f"donor lacks tensor '{donor_name}' (requested for '{target_name}')"
            )
        arr = donor.tensors[donor_name]
        if target_name in target and target.tensors[target_name].shape != arr.shape:
            raise ShapeMismatch(
                f"transplant shape mismatch for '{target_name}': "
                f"target {target.tensors[target_name].shape} vs donor {arr.shape}"
            )
        if target_name not in target:
            added.append(target_name)
        out[target_name] = ctx.emit(np.array(arr, copy=True), name=target_name)
    return make_outcome(
        op="w.structural.transplant",
        tensors=out,
        replaced=set(mapping) - set(added),
        added=set(added),
        origin=Origin.EXTRACTED,
        details={
            "mapping": mapping,
            "num_transplanted": len(mapping),
            "note": "components copied verbatim from a donor parameter set",
        },
    )


def _tie_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    pkg = inputs[0]
    names = [str(n) for n in (p.get("names") or [])]
    if len(names) < 2:
        raise InvalidOperationParameters("parameter tying requires at least two tensor names")
    for name in names:
        if name not in pkg:
            raise InvalidOperationParameters(f"tensor '{name}' not present")
        if pkg.tensors[name].shape != pkg.tensors[names[0]].shape:
            raise ShapeMismatch(
                f"cannot tie '{name}' {pkg.tensors[name].shape} to '{names[0]}' "
                f"{pkg.tensors[names[0]].shape}"
            )
    strategy = str(p.get("strategy", "mean"))
    stack = np.stack([float_view(pkg.tensors[n]) for n in names])
    if strategy == "mean":
        tied = stack.mean(axis=0)
    elif strategy == "first":
        tied = stack[0]
    elif strategy == "second":
        tied = stack[1] if stack.shape[0] > 1 else stack[0]
    else:
        raise InvalidOperationParameters("strategy must be mean|first|second")
    tied = tied.astype(pkg.tensors[names[0]].dtype)
    out = {name: ctx.emit(np.array(tied, copy=True), name=name) for name in names}
    return make_outcome(
        op="w.structural.tie",
        tensors=out,
        replaced=set(names),
        origin=Origin.TRANSFORMED,
        details={
            "names": names,
            "strategy": strategy,
            "note": "parameter sharing: all listed tensors now hold identical content",
        },
    )


def _prune_heads_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    pkg = inputs[0]
    name = str(p["name"])
    if name not in pkg:
        raise InvalidOperationParameters(f"tensor '{name}' not present")
    arr = float_view(pkg.tensors[name])
    heads = int(p["n_heads"])
    remove = [int(h) for h in p["heads"]]
    if arr.ndim != 2:
        raise InvalidOperationParameters("head pruning expects a 2-D projection tensor")
    rows, cols = arr.shape
    if rows % heads != 0:
        raise InvalidOperationParameters(f"rows {rows} not divisible by n_heads {heads}")
    if cols % heads != 0:
        raise InvalidOperationParameters(f"cols {cols} not divisible by n_heads {heads}")
    if any(h < 0 or h >= heads for h in remove):
        raise InvalidOperationParameters(f"head indices must be in [0, {heads})")
    keep = [h for h in range(heads) if h not in set(remove)]
    if not keep:
        raise InvalidOperationParameters("cannot remove every head")
    rd, cd = rows // heads, cols // heads
    out_rows = np.concatenate([arr[h * rd:(h + 1) * rd] for h in keep], axis=0)
    out_arr = np.concatenate([out_rows[:, h * cd:(h + 1) * cd] for h in keep], axis=1)
    out = {name: ctx.emit(out_arr.astype(pkg.tensors[name].dtype, copy=False), name=name)}
    return make_outcome(
        op="w.structural.prune_heads",
        tensors=out,
        replaced={name},
        origin=Origin.EXTRACTED,
        details={
            "removed_heads": remove,
            "remaining_heads": keep,
            "shape_before": list(arr.shape),
            "shape_after": list(out_arr.shape),
            "architecture_change": True,
            "note": (
                "attention heads were structurally removed: tensor shapes changed, so the "
                "model definition (n_heads and adjacent projections) MUST be updated to match"
            ),
        },
    )


def _slice_impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
    pkg = inputs[0]
    name = str(p["name"])
    if name not in pkg:
        raise InvalidOperationParameters(f"tensor '{name}' not present")
    arr = pkg.tensors[name]
    axis = int(p.get("axis", 0)) % max(1, arr.ndim)
    idx = [slice(None)] * arr.ndim
    idx[axis] = slice(p.get("start"), p.get("stop"), p.get("step"))
    sliced = arr[tuple(idx)]
    out = {name: ctx.emit(np.array(sliced, copy=True), name=name)}
    return make_outcome(
        op="w.structural.slice",
        tensors=out,
        replaced={name},
        origin=Origin.EXTRACTED,
        details={
            "axis": axis,
            "start": p.get("start"),
            "stop": p.get("stop"),
            "shape_before": list(arr.shape),
            "shape_after": list(sliced.shape),
            "architecture_change": True,
            "note": "tensor was sliced; the graph binding must be updated for execution",
        },
    )


def _register(spec: OperationSpec) -> None:
    default_registry().register(spec)


_register(OperationSpec(
    code="w.structural.extract",
    title="Layer extraction",
    family="structural",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_extract_impl,
    description="Keep the tensors of selected layers (plus glob-matched extras), drop the rest.",
    scopes=_SCOPES,
    params={
        "layers": ParamSpec("list[int]", required=True),
        "keep": ParamSpec("list[str]", default=None, description="glob patterns always retained"),
    },
    assumptions=("layer indices follow a 'layers.<n>.' style naming convention",),
    origin=Origin.EXTRACTED,
))

_register(OperationSpec(
    code="w.structural.transplant",
    title="Component transplantation",
    family="structural",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_transplant_impl,
    description="Copy parameter tensors from a donor package into the subject package.",
    scopes=_SCOPES,
    params={"mapping": ParamSpec("dict", required=True)},
    assumptions=("transplanted components are semantically compatible with the recipient",),
    min_inputs=2,
    max_inputs=2,
    origin=Origin.EXTRACTED,
))

_register(OperationSpec(
    code="w.structural.tie",
    title="Parameter tying",
    family="structural",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_tie_impl,
    description="Make several tensors share one value (mean, first or second).",
    scopes=_SCOPES,
    params={
        "names": ParamSpec("list[str]", required=True),
        "strategy": ParamSpec("str", default="mean", choices=("mean", "first", "second")),
    },
    assumptions=("the tied tensors have identical shapes and compatible roles",),
    origin=Origin.TRANSFORMED,
))

_register(OperationSpec(
    code="w.structural.prune_heads",
    title="Attention-head pruning",
    family="structural",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_prune_heads_impl,
    description="Remove attention heads from a projection tensor (shape-changing).",
    scopes=_SCOPES,
    params={
        "name": ParamSpec("str", required=True),
        "n_heads": ParamSpec("int", required=True, minimum=1),
        "heads": ParamSpec("list[int]", required=True),
    },
    assumptions=(
        "the tensor is laid out as [n_heads * dim, n_heads * dim] (or fused QKV split externally)",
        "removing heads is an architecture change that must be mirrored in the model definition",
    ),
    requires_alignment=False,
    mutates_architecture=True,
    origin=Origin.EXTRACTED,
))

_register(OperationSpec(
    code="w.structural.slice",
    title="Tensor slicing",
    family="structural",
    effect=EngineEffect.PARAM_TRANSFORM,
    fn=_slice_impl,
    description="Slice a single tensor along one axis (shape-changing).",
    scopes=_SCOPES,
    params={
        "name": ParamSpec("str", required=True),
        "axis": ParamSpec("int", default=0, minimum=-8, maximum=8),
        "start": ParamSpec("int", default=None),
        "stop": ParamSpec("int", default=None),
        "step": ParamSpec("int", default=None),
    },
    assumptions=("the downstream graph will be rebound to the new shape",),
    mutates_architecture=True,
    origin=Origin.EXTRACTED,
))
