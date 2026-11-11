"""Weight arithmetic: interpolation, extrapolation, deltas and elementwise algebra.

Every operation here is a closed-form mathematical transformation. None of them
computes a gradient, and none mutates its inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from uzen.core.nwpc.registry import OperationSpec, ParamSpec, default_registry
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import InvalidOperationParameters, ShapeMismatch
from uzen.operations._common import (
    compute_dtype,
    float_view,
    intersect_names,
    make_outcome,
    require_float_tensors,
    selected_names,
    stats_delta,
)
from uzen.policies.strict_plus.effects import EngineEffect, Origin, TaskScope
from uzen.util.linalg import linear_interpolate, spherical_interpolate

SELECT = ParamSpec("list[str]", default=None, description="glob patterns restricting the tensors touched")
PARTIAL = ParamSpec("bool", default=False, description="operate on the intersection when structures differ")


def _register(spec: OperationSpec) -> None:
    default_registry().register(spec)


def _binary(
    *,
    code: str,
    title: str,
    fn,
    params: dict[str, ParamSpec],
    description: str,
    origin: Origin = Origin.TRANSFORMED,
    assumptions: tuple[str, ...] = (),
) -> None:
    def impl(ctx, inputs: list[WeightPackage], p: dict[str, Any]):
        a, b = inputs[0], inputs[1]
        names = intersect_names([a, b], p.get("select"), allow_partial=bool(p.get("partial")))
        require_float_tensors(a, names, op=code)
        out: dict[str, np.ndarray] = {}
        per_tensor: dict[str, Any] = {}
        for name in names:
            x = a.tensors[name]
            y = b.tensors[name]
            dtype = compute_dtype(x, y)
            result = fn(float_view(x), float_view(y), p)
            if np.asarray(result).shape != x.shape:
                raise ShapeMismatch(
                    f"operation '{code}' changed the shape of '{name}' "
                    f"from {x.shape} to {np.asarray(result).shape}; weight arithmetic must preserve shapes"
                )
            out[name] = ctx.emit(np.asarray(result).astype(dtype, copy=False), name=name)
            if p.get("report_per_tensor", False):
                per_tensor[name] = stats_delta(x, out[name])
        details: dict[str, Any] = {
            "num_tensors": len(names),
            "params": {k: v for k, v in p.items() if k not in ("select",)},
            "note": "closed-form parameter transformation; no gradient was computed",
        }
        if per_tensor:
            details["per_tensor"] = per_tensor
        return make_outcome(
            op=code,
            tensors=out,
            replaced=names,
            details=details,
            origin=origin,
            alignment=None,
        )

    _register(
        OperationSpec(
            code=code,
            title=title,
            family="arithmetic",
            effect=EngineEffect.PARAM_TRANSFORM,
            fn=impl,
            description=description,
            scopes=frozenset(
                {
                    TaskScope.WEIGHT_MODIFICATION,
                    TaskScope.WEIGHT_GENERATION,
                    TaskScope.WEIGHT_RECONSTRUCTION,
                    TaskScope.WEIGHT_OPTIMIZATION,
                }
            ),
            params={
                "select": SELECT,
                "partial": PARTIAL,
                "report_per_tensor": ParamSpec("bool", default=False),
                **params,
            },
            assumptions=assumptions,
            min_inputs=2,
            max_inputs=2,
            origin=origin,
        )
    )


_binary(
    code="w.arithmetic.linear_interpolate",
    title="Linear interpolation",
    fn=lambda x, y, p: linear_interpolate(x, y, float(p["t"])),
    params={"t": ParamSpec("float", required=True, minimum=0.0, maximum=1.0, description="interpolation weight")},
    description="(1 - t) * a + t * b, elementwise.",
    assumptions=(
        "both tensors describe the same parameter slot (same shape and semantics)",
        "interpolation of independently trained models is only meaningful when they share lineage",
    ),
)

_binary(
    code="w.arithmetic.slerp",
    title="Spherical interpolation",
    fn=lambda x, y, p: spherical_interpolate(x, y, float(p["t"])),
    params={
        "t": ParamSpec("float", required=True, minimum=0.0, maximum=1.0),
        "preserve_norm": ParamSpec("bool", default=False, description="rescale to the lerp of input norms"),
    },
    description="Spherical linear interpolation of two tensors treated as vectors.",
    assumptions=("both tensors share the same shape and parameter semantics",),
)


def _slerp_norm_impl():
    from uzen.util.linalg import slerp_preserve_norm

    return slerp_preserve_norm


def _slerp_dispatch(ctx, inputs, p):
    a, b = inputs[0], inputs[1]
    names = intersect_names([a, b], p.get("select"), allow_partial=bool(p.get("partial")))
    require_float_tensors(a, names, op="w.arithmetic.slerp")
    out: dict[str, np.ndarray] = {}
    for name in names:
        x, y = a.tensors[name], b.tensors[name]
        dtype = compute_dtype(x, y)
        if p.get("preserve_norm"):
            result = _slerp_norm_impl()(x, y, float(p["t"]))
        else:
            result = spherical_interpolate(float_view(x), float_view(y), float(p["t"]))
        out[name] = ctx.emit(np.asarray(result).astype(dtype, copy=False), name=name)
    return make_outcome(
        op="w.arithmetic.slerp",
        tensors=out,
        replaced=names,
        details={
            "num_tensors": len(names),
            "t": float(p["t"]),
            "preserve_norm": bool(p.get("preserve_norm")),
            "note": "spherical interpolation preserves direction structure better than lerp for large angles",
        },
    )


# Replace the auto-registered slerp with the norm-aware implementation.
default_registry()._ops["w.arithmetic.slerp"].fn = _slerp_dispatch


_binary(
    code="w.arithmetic.extrapolate",
    title="Extrapolation",
    fn=lambda x, y, p: x + float(p["factor"]) * (y - x),
    params={
        "factor": ParamSpec("float", required=True, minimum=-10.0, maximum=10.0,
                            description="1 = b, >1 beyond b, <0 behind a"),
    },
    description="a + factor * (b - a).",
    assumptions=("a and b share lineage; extrapolation amplifies differences",),
)

_binary(
    code="w.arithmetic.delta",
    title="Delta extraction",
    fn=lambda x, y, p: y - x,
    params={"sign": ParamSpec("float", default=1.0, choices=("1.0", "-1.0"), description="unused; kept for clarity")}
    if False else {},
    description="target - reference (a task vector when target is a fine-tuned sibling).",
    origin=Origin.EXTRACTED,
    assumptions=("target and reference are the same architecture",),
)

_binary(
    code="w.arithmetic.hadamard",
    title="Hadamard product",
    fn=lambda x, y, p: x * y,
    params={},
    description="Elementwise product of two parameter tensors.",
    assumptions=("elementwise correspondence is meaningful for these parameters",),
)

_binary(
    code="w.arithmetic.scaled_add",
    title="Scaled addition",
    fn=lambda x, y, p: x + float(p["scale"]) * y,
    params={"scale": ParamSpec("float", required=True, minimum=-100.0, maximum=100.0)},
    description="base + scale * delta.",
    assumptions=("the second tensor is a delta expressed in the same parameter basis",),
)


def _scale_impl(ctx, inputs, p):
    pkg = inputs[0]
    names = selected_names(pkg, p.get("select"))
    require_float_tensors(pkg, names, op="w.arithmetic.scale")
    out = {}
    for name in names:
        arr = pkg.tensors[name]
        out[name] = ctx.emit((float_view(arr) * float(p["factor"])).astype(arr.dtype, copy=False), name=name)
    return make_outcome(
        op="w.arithmetic.scale",
        tensors=out,
        replaced=names,
        details={"num_tensors": len(names), "factor": float(p["factor"])},
    )


_register(
    OperationSpec(
        code="w.arithmetic.scale",
        title="Tensor scaling",
        family="arithmetic",
        effect=EngineEffect.PARAM_TRANSFORM,
        fn=_scale_impl,
        description="Multiply selected tensors by a scalar.",
        scopes=frozenset(
            {
                TaskScope.WEIGHT_MODIFICATION,
                TaskScope.WEIGHT_GENERATION,
                TaskScope.WEIGHT_OPTIMIZATION,
                TaskScope.WEIGHT_RECONSTRUCTION,
            }
        ),
        params={
            "factor": ParamSpec("float", required=True, minimum=-1e6, maximum=1e6),
            "select": SELECT,
        },
        assumptions=("uniform scaling of a parameter block preserves its shape and role",),
    )
)


def _negate_impl(ctx, inputs, p):
    pkg = inputs[0]
    names = selected_names(pkg, p.get("select"))
    out = {name: ctx.emit((-float_view(pkg.tensors[name])).astype(pkg.tensors[name].dtype, copy=False), name=name) for name in names}
    return make_outcome(op="w.arithmetic.negate", tensors=out, replaced=names, details={"num_tensors": len(names)})


_register(
    OperationSpec(
        code="w.arithmetic.negate",
        title="Negation",
        family="arithmetic",
        effect=EngineEffect.PARAM_TRANSFORM,
        fn=_negate_impl,
        description="Negate selected tensors (useful for task-vector subtraction).",
        params={"select": SELECT},
        scopes=frozenset({TaskScope.WEIGHT_MODIFICATION, TaskScope.WEIGHT_RECONSTRUCTION}),
    )
)


def _add_scalar_impl(ctx, inputs, p):
    pkg = inputs[0]
    names = selected_names(pkg, p.get("select"))
    out = {
        name: ctx.emit((float_view(pkg.tensors[name]) + float(p["value"])).astype(pkg.tensors[name].dtype, copy=False), name=name)
        for name in names
    }
    return make_outcome(
        op="w.arithmetic.add_scalar",
        tensors=out,
        replaced=names,
        details={"num_tensors": len(names), "value": float(p["value"])},
    )


_register(
    OperationSpec(
        code="w.arithmetic.add_scalar",
        title="Scalar offset",
        family="arithmetic",
        effect=EngineEffect.PARAM_TRANSFORM,
        fn=_add_scalar_impl,
        description="Add a constant to selected tensors (e.g. bias offsetting).",
        params={
            "value": ParamSpec("float", required=True, minimum=-1e6, maximum=1e6),
            "select": SELECT,
        },
        scopes=frozenset({TaskScope.WEIGHT_MODIFICATION, TaskScope.WEIGHT_OPTIMIZATION}),
        assumptions=("meaningful only when a constant offset is a valid perturbation for the parameter role",),
    )
)
