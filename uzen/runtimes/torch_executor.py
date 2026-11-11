"""Torch-CPU execution of NIR graphs.

This is a second, real implementation of the NIR operator kernels on ``torch.Tensor``.
It mirrors the NumPy reference executor's validation (symbolic dims, declared shapes,
finite outputs) and its numerics: normalization, attention, activations and the routed
kernels compute in float64 and cast back exactly where the reference does.

Every forward pass runs under ``torch.inference_mode()``; parameters are constants and
no autograd graph is built. Sampling operators reuse the reference's seeded NumPy
generator so seeded outputs stay identical across backends.

Operators without a torch kernel raise :class:`UnsupportedOperator`; the runtime's
``backend="auto"`` then falls back to NumPy and records why, while ``backend="torch"``
refuses to fall back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from uzen.core.nir.graph import Graph
from uzen.core.nir.ops import get_operator, param_ref_names, port_order
from uzen.core.nir.types import TensorType, replace_symbols
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import (
    InvalidOperationParameters,
    RuntimeUnavailable,
    ShapeMismatch,
    TensorInvalid,
    UnsupportedOperator,
)

try:  # optional dependency
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

GRAPH_INPUT = "input:"


@dataclass
class TorchExecution:
    outputs: dict[str, np.ndarray]
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    num_ops: int = 0


def is_available() -> bool:
    return TORCH_AVAILABLE


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeUnavailable(
            "the torch executor needs the optional 'torch' package; install the CPU wheel "
            "with `pip install torch --index-url https://download.pytorch.org/whl/cpu`",
            import_error=TORCH_IMPORT_ERROR,
        )


def _dtype_name(tensor: Any) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _pn(params: Mapping[str, Any], key: str, default: str) -> str:
    return str(params.get(key, default)).replace("param:", "")


def _pname(params: Mapping[str, Any], key: str) -> str:
    return str(params[key]).replace("param:", "")


def _pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return (int(value[0]), int(value[1]))
    raise InvalidOperationParameters(f"conv2d '{name}' must be an int or [h, w]")


def _axes(value: Any, rank: int) -> tuple[int, ...]:
    axes = (value,) if isinstance(value, int) else tuple(value)
    out = []
    for axis in axes:
        if axis < -rank or axis >= rank:
            raise InvalidOperationParameters(f"axis {axis} out of range for rank {rank}")
        out.append(axis % rank)
    return tuple(out)


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #


def _linear(inputs, params, bound):
    x = inputs[0]
    w = bound[_pn(params, "weight", "weight")]
    if x.shape[-1] != w.shape[1]:
        raise ShapeMismatch(f"linear: input last dim {x.shape[-1]} != weight in {w.shape[1]}")
    dtype = torch.promote_types(x.dtype, w.dtype)
    y = x.to(dtype) @ w.T.to(dtype)
    if params.get("bias"):
        b = bound[_pname(params, "bias")]
        if tuple(b.shape) != (w.shape[0],):
            raise ShapeMismatch(f"linear bias shape {tuple(b.shape)} != ({w.shape[0]},)")
        y = y + b.to(dtype)
    return [y]


def _embedding(inputs, params, bound):
    idx = inputs[0]
    w = bound[_pn(params, "weight", "weight")]
    if idx.dtype not in (torch.int32, torch.int64):
        raise InvalidOperationParameters(f"embedding indices must be integer, got {_dtype_name(idx)}")
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= w.shape[0]):
        raise InvalidOperationParameters(f"embedding index out of range [0, {w.shape[0]})")
    return [w[idx]]


def _pos_embedding(inputs, params, bound):
    x = inputs[0]
    w = bound[_pn(params, "weight", "weight")]
    if x.dim() < 3:
        raise InvalidOperationParameters(f"pos_embedding expects (batch, seq, dim), got {tuple(x.shape)}")
    t = x.shape[1]
    if t > w.shape[0]:
        raise ShapeMismatch(
            f"sequence length {t} exceeds positional embedding table size {w.shape[0]}"
        )
    return [x + w[:t].reshape((1, t) + tuple(x.shape[2:]))]


def _layernorm(inputs, params, bound):
    x = inputs[0]
    eps = float(params.get("eps", 1e-5))
    x64 = x.to(torch.float64)
    mean = x64.mean(dim=-1, keepdim=True)
    var = x64.var(dim=-1, keepdim=True, unbiased=False)
    y = (x64 - mean) / torch.sqrt(var + eps)
    if params.get("weight"):
        y = y * bound[_pname(params, "weight")]
    if params.get("bias"):
        y = y + bound[_pname(params, "bias")]
    return [y.to(x.dtype)]


def _rmsnorm(inputs, params, bound):
    x = inputs[0]
    eps = float(params.get("eps", 1e-6))
    x64 = x.to(torch.float64)
    scale = 1.0 / torch.sqrt((x64 * x64).mean(dim=-1, keepdim=True) + eps)
    y = x64 * scale
    if params.get("weight"):
        y = y * bound[_pname(params, "weight")]
    return [y.to(x.dtype)]


def _gelu64(x):
    x64 = x.to(torch.float64)
    c = float(np.sqrt(2.0 / np.pi))
    return 0.5 * x64 * (1.0 + torch.tanh(c * (x64 + 0.044715 * x64**3)))


def _silu64(x):
    x64 = x.to(torch.float64)
    return x64 / (1.0 + torch.exp(-x64))


_ACTIVATIONS: dict[str, Callable[[Any], Any]] = {
    "gelu": _gelu64,
    "silu": _silu64,
    "relu": lambda x: torch.clamp_min(x.to(torch.float64), 0.0),
    "tanh": lambda x: torch.tanh(x.to(torch.float64)),
}


def _attention(inputs, params, bound):
    q, k, v = inputs
    if not (q.dim() == k.dim() == v.dim() == 4):
        raise InvalidOperationParameters("attention expects (batch, heads, seq, head_dim) tensors")
    q64, k64, v64 = (t.to(torch.float64) for t in (q, k, v))
    scale = params.get("scale", "auto")
    scale = 1.0 / float(np.sqrt(q.shape[-1])) if scale == "auto" else float(scale)
    scores = torch.einsum("bhtd,bhsd->bhts", q64, k64) * scale
    if bool(params.get("causal", True)):
        t, s = scores.shape[-2], scores.shape[-1]
        if t != s:
            raise InvalidOperationParameters(f"causal attention requires square scores, got {t} x {s}")
        mask = torch.triu(torch.ones((t, s), dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    m = scores.max(dim=-1, keepdim=True).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(scores - m)
    denom = e.sum(dim=-1, keepdim=True)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    ctx = torch.einsum("bhts,bhsd->bhtd", e / denom, v64)
    return [ctx.to(q.dtype)]


def _split_qkv(inputs, params, bound):
    x = inputs[0]
    if x.dim() != 3:
        raise InvalidOperationParameters(f"split_qkv expects (batch, seq, 3*dim), got {tuple(x.shape)}")
    heads = int(params["n_heads"])
    bsz, seq, last = x.shape
    dh = last // (3 * heads)
    r = x.reshape(bsz, seq, 3, heads, dh).permute(0, 2, 3, 1, 4)
    return [r[:, 0], r[:, 1], r[:, 2]]


def _merge_heads(inputs, params, bound):
    x = inputs[0]
    b, h, t, dh = x.shape
    return [x.permute(0, 2, 1, 3).reshape(b, t, h * dh)]


def _reshape(inputs, params, bound):
    x = inputs[0]
    target = list(params["shape"])
    shape: list[int] = []
    for dim in target:
        if isinstance(dim, int):
            shape.append(dim)
        elif isinstance(dim, str):
            shape.append(int(x.shape[list(x.shape).index(dim)]))
        else:
            raise InvalidOperationParameters(f"cannot resolve reshape dim {dim!r} at runtime")
    return [x.reshape(tuple(shape))]


def _transpose(inputs, params, bound):
    x = inputs[0]
    perm = params.get("perm")
    if perm is None:
        perm = list(reversed(range(x.dim())))
    return [x.permute(tuple(int(p) for p in perm))]


def _slice(inputs, params, bound):
    x = inputs[0]
    axis = int(params.get("axis", 0)) % x.dim()
    idx = [slice(None)] * x.dim()
    idx[axis] = slice(params.get("start"), params.get("stop"), params.get("step"))
    return [x[tuple(idx)]]


def _flatten(inputs, params, bound):
    x = inputs[0]
    start = int(params.get("start_dim", 1))
    prefix = tuple(x.shape[:start])
    return [x.reshape(prefix + (-1,))]


def _reduce(kind: str) -> Callable[..., list[Any]]:
    def kernel(inputs, params, bound):
        x = inputs[0]
        axes = _axes(params.get("axis", -1), x.dim())
        keep = bool(params.get("keepdims", False))
        if kind == "mean" and not x.is_floating_point():
            x = x.to(torch.float64)
        fn = torch.mean if kind == "mean" else torch.sum
        return [fn(x, dim=axes if len(axes) > 1 else axes[0], keepdim=keep)]

    return kernel


def _conv2d(inputs, params, bound):
    x = inputs[0]
    w = bound[_pn(params, "weight", "weight")]
    stride = _pair(params.get("stride", 1), "stride")
    padding = _pair(params.get("padding", 0), "padding")
    dtype = torch.promote_types(x.dtype, w.dtype)
    bias = bound[_pname(params, "bias")].to(dtype) if params.get("bias") else None
    y = F.conv2d(x.to(dtype), w.to(dtype), bias, stride=stride, padding=padding)
    return [y]


def _maxpool2d(inputs, params, bound):
    x = inputs[0]
    kernel = _pair(params.get("kernel", 2), "kernel")
    stride = _pair(params.get("stride", params.get("kernel", 2)), "stride")
    padding = _pair(params.get("padding", 0), "padding")
    if padding != (0, 0):
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]), value=float("-inf"))
    return [F.max_pool2d(x, kernel_size=kernel, stride=stride)]


def _rnn(inputs, params, bound):
    x = inputs[0]
    wx = bound[_pn(params, "weight_ih", "weight_ih")].to(torch.float64)
    wh = bound[_pn(params, "weight_hh", "weight_hh")].to(torch.float64)
    b = bound[_pname(params, "bias")].to(torch.float64) if params.get("bias") else None
    bsz, steps, _ = x.shape
    hidden = wh.shape[0]
    h = torch.zeros((bsz, hidden), dtype=torch.float64, device=x.device)
    outs = []
    for step in range(steps):
        h = torch.tanh(x[:, step].to(torch.float64) @ wx.T + h @ wh.T + (b if b is not None else 0.0))
        outs.append(h)
    return [torch.stack(outs, dim=1)]


def _lstm(inputs, params, bound):
    x = inputs[0]
    wx = bound[_pn(params, "weight_ih", "weight_ih")].to(torch.float64)
    wh = bound[_pn(params, "weight_hh", "weight_hh")].to(torch.float64)
    b = bound[_pname(params, "bias")].to(torch.float64) if params.get("bias") else None
    bsz, steps, _ = x.shape
    hidden = wh.shape[0] // 4
    h = torch.zeros((bsz, hidden), dtype=torch.float64, device=x.device)
    c = torch.zeros((bsz, hidden), dtype=torch.float64, device=x.device)
    outs = []
    for step in range(steps):
        gates = x[:, step].to(torch.float64) @ wx.T + h @ wh.T + (b if b is not None else 0.0)
        i = torch.sigmoid(gates[:, :hidden])
        f = torch.sigmoid(gates[:, hidden : 2 * hidden])
        g = torch.tanh(gates[:, 2 * hidden : 3 * hidden])
        o = torch.sigmoid(gates[:, 3 * hidden :])
        c = f * c + i * g
        h = o * torch.tanh(c)
        outs.append(h)
    return [torch.stack(outs, dim=1)]


def _gru(inputs, params, bound):
    x = inputs[0]
    wx = bound[_pn(params, "weight_ih", "weight_ih")].to(torch.float64)
    wh = bound[_pn(params, "weight_hh", "weight_hh")].to(torch.float64)
    b = bound[_pname(params, "bias")].to(torch.float64) if params.get("bias") else None
    bsz, steps, _ = x.shape
    hidden = wh.shape[0] // 3
    h = torch.zeros((bsz, hidden), dtype=torch.float64, device=x.device)
    outs = []
    for step in range(steps):
        xg = x[:, step].to(torch.float64) @ wx.T + (b if b is not None else 0.0)
        hg = h @ wh.T
        r = torch.sigmoid(xg[:, :hidden] + hg[:, :hidden])
        z = torch.sigmoid(xg[:, hidden : 2 * hidden] + hg[:, hidden : 2 * hidden])
        n = torch.tanh(xg[:, 2 * hidden :] + r * hg[:, 2 * hidden :])
        h = (1.0 - z) * n + z * h
        outs.append(h)
    return [torch.stack(outs, dim=1)]


def _moe(inputs, params, bound):
    x = inputs[0]
    router = bound[_pn(params, "router", "router")]
    up_w = bound[_pn(params, "up_weight", "up_weight")]
    down_w = bound[_pn(params, "down_weight", "down_weight")]
    up_b = bound[_pname(params, "up_bias")] if params.get("up_bias") else None
    down_b = bound[_pname(params, "down_bias")] if params.get("down_bias") else None
    activation = _ACTIVATIONS[str(params.get("activation", "gelu"))]
    x64 = x.to(torch.float64)
    probs = torch.softmax(x64 @ router.T.to(torch.float64), dim=-1)
    n_experts = probs.shape[-1]
    top_k = min(max(1, int(params.get("top_k", 2))), n_experts)
    gates, selected = torch.topk(probs, top_k, dim=-1)
    gates = gates / torch.clamp_min(gates.sum(dim=-1, keepdim=True), 1e-12)

    flat = x64.reshape(-1, x64.shape[-1])
    sel = selected.reshape(-1, top_k)
    gts = gates.reshape(-1, top_k)
    out = torch.zeros_like(flat)
    for expert in range(n_experts):
        mask = sel == expert
        if not bool(mask.any()):
            continue
        rows = mask.any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
        gate = (gts[rows] * mask[rows].to(gts.dtype)).sum(dim=-1, keepdim=True)
        hidden = flat[rows] @ up_w[expert].T.to(torch.float64)
        if up_b is not None:
            hidden = hidden + up_b[expert].to(torch.float64)
        hidden = activation(hidden)
        y = hidden @ down_w[expert].T.to(torch.float64)
        if down_b is not None:
            y = y + down_b[expert].to(torch.float64)
        out.index_add_(0, rows, y * gate)
    return [out.reshape(x64.shape).to(x.dtype)]


def _ssm(inputs, params, bound):
    x = inputs[0].to(torch.float64)
    a = bound[_pn(params, "A", "A")].to(torch.float64)
    b = bound[_pn(params, "B", "B")].to(torch.float64)
    c = bound[_pn(params, "C", "C")].to(torch.float64)
    d = bound[_pname(params, "D")].to(torch.float64) if params.get("D") else None
    bsz, steps, _ = x.shape
    state = torch.zeros((bsz, a.shape[0], a.shape[1]), dtype=torch.float64, device=x.device)
    outs = []
    for step in range(steps):
        state = a[None] * state + b[None] * x[:, step, :, None]
        y = (state * c[None]).sum(dim=-1)
        if d is not None:
            y = y + d[None] * x[:, step]
        outs.append(y)
    return [torch.stack(outs, dim=1).to(inputs[0].dtype)]


def _gcn(inputs, params, bound):
    x, adj = inputs
    w = bound[_pn(params, "weight", "weight")]
    x64 = x.to(torch.float64)
    a = adj.to(torch.float64) + torch.eye(adj.shape[-1], dtype=torch.float64, device=adj.device)[None]
    deg = a.sum(dim=-1, keepdim=True)
    inv = 1.0 / torch.sqrt(torch.clamp(deg, min=1e-12))
    a_norm = a * inv * inv.transpose(-1, -2)
    y = torch.einsum("bij,bjf->bif", a_norm, x64) @ w.T.to(torch.float64)
    if params.get("bias"):
        y = y + bound[_pname(params, "bias")].to(torch.float64)
    return [y.to(torch.promote_types(x.dtype, w.dtype))]


def _sample_normal(inputs, params, bound):
    mu, logvar = inputs
    eps = np.random.default_rng(int(params.get("seed", 0))).standard_normal(tuple(mu.shape))
    eps_t = torch.from_numpy(eps).to(mu.device)
    z = mu.to(torch.float64) + torch.exp(0.5 * logvar.to(torch.float64)) * eps_t
    return [z.to(mu.dtype)]


def _add_noise(inputs, params, bound):
    x = inputs[0]
    level = float(params.get("alpha_bar", 0.5))
    eps = np.random.default_rng(int(params.get("seed", 0))).standard_normal(tuple(x.shape))
    eps_t = torch.from_numpy(eps).to(x.device)
    out = float(np.sqrt(level)) * x.to(torch.float64) + float(np.sqrt(1.0 - level)) * eps_t
    return [out.to(x.dtype)]


def _matmul(inputs, params, bound):
    a, b = inputs
    dtype = torch.promote_types(a.dtype, b.dtype)
    return [torch.matmul(a.to(dtype), b.to(dtype))]


def _concat(inputs, params, bound):
    dtype = inputs[0].dtype
    for tensor in inputs[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return [torch.cat([tensor.to(dtype) for tensor in inputs], dim=int(params.get("axis", -1)))]


_KERNELS: dict[str, Callable[..., list[Any]]] = {
    "linear": _linear,
    "embedding": _embedding,
    "pos_embedding": _pos_embedding,
    "layernorm": _layernorm,
    "rmsnorm": _rmsnorm,
    "add": lambda i, p, b: [i[0] + i[1]],
    "sub": lambda i, p, b: [i[0] - i[1]],
    "mul": lambda i, p, b: [i[0] * i[1]],
    "scale": lambda i, p, b: [(i[0] * float(p.get("factor", 1.0))).to(i[0].dtype)],
    "matmul": _matmul,
    "relu": lambda i, p, b: [torch.clamp_min(i[0], 0).to(i[0].dtype)],
    "gelu": lambda i, p, b: [_gelu64(i[0]).to(i[0].dtype)],
    "silu": lambda i, p, b: [_silu64(i[0]).to(i[0].dtype)],
    "tanh": lambda i, p, b: [torch.tanh(i[0]).to(i[0].dtype)],
    "sigmoid": lambda i, p, b: [torch.sigmoid(i[0]).to(i[0].dtype)],
    "softmax": lambda i, p, b: [
        torch.softmax(i[0].to(torch.float64), dim=_axes(p.get("axis", -1), i[0].dim())[0]).to(i[0].dtype)
    ],
    "split_qkv": _split_qkv,
    "merge_heads": _merge_heads,
    "attention": _attention,
    "reshape": _reshape,
    "transpose": _transpose,
    "concat": _concat,
    "slice": _slice,
    "flatten": _flatten,
    "mean": _reduce("mean"),
    "sum": _reduce("sum"),
    "conv2d": _conv2d,
    "maxpool2d": _maxpool2d,
    "rnn": _rnn,
    "lstm": _lstm,
    "gru": _gru,
    "identity": lambda i, p, b: [i[0]],
    "dropout": lambda i, p, b: [i[0]],
    "select": lambda i, p, b: [torch.where(i[0] != 0, i[1], i[2])],
    "moe": _moe,
    "ssm": _ssm,
    "gcn": _gcn,
    "sample_normal": _sample_normal,
    "add_noise": _add_noise,
}


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


def _key(ref: Any) -> str:
    if ref.node.startswith(GRAPH_INPUT):
        return ref.node
    return f"{ref.node}:{ref.port}"


def _resolve_params(params: Mapping[str, Any], symbols: Mapping[str, int]) -> dict[str, Any]:
    out = dict(params)
    shape = out.get("shape")
    if isinstance(shape, (list, tuple)):
        out["shape"] = [int(symbols[d]) if isinstance(d, str) and d in symbols else d for d in shape]
    return out


def _bindings(
    graph: Graph, package: WeightPackage, device: Any
) -> dict[str, dict[str, Any]]:
    cache: dict[str, Any] = {}
    resolved: dict[str, dict[str, Any]] = {}
    for nid, node in graph.nodes.items():
        per_node: dict[str, Any] = {}
        for value in node.params.values():
            for name in param_ref_names(value):
                if name not in cache:
                    array = np.ascontiguousarray(package.get(name))
                    cache[name] = torch.from_numpy(array).to(device)
                per_node[name] = cache[name]
        resolved[nid] = per_node
    return resolved


def execute_torch(
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
    package: WeightPackage,
    *,
    capture: Sequence[str] | None = None,
    check_finite: bool = True,
    validate_types: bool = True,
    device: str = "cpu",
) -> TorchExecution:
    """Run a graph forward on torch-CPU tensors (inference mode, no autograd)."""
    _require_torch()
    device_obj = torch.device(device)

    missing = [name for name in graph.inputs if name not in inputs]
    if missing:
        raise ShapeMismatch(f"missing graph inputs: {missing}")
    extra = [name for name in inputs if name not in graph.inputs]
    if extra:
        raise ShapeMismatch(f"unexpected graph inputs: {extra}")

    env: dict[str, Any] = {}
    symbols: dict[str, int] = {}
    for name, spec in graph.inputs.items():
        array = np.asarray(inputs[name])
        if array.ndim != len(spec.shape):
            raise ShapeMismatch(
                f"input '{name}' has rank {array.ndim}, declared rank {len(spec.shape)}"
            )
        for dim_spec, dim_val in zip(spec.shape, array.shape):
            if isinstance(dim_spec, int):
                if int(dim_spec) != int(dim_val):
                    raise ShapeMismatch(
                        f"input '{name}' dimension {dim_val} != declared {dim_spec}"
                    )
            elif isinstance(dim_spec, str):
                prior = symbols.get(dim_spec)
                if prior is not None and prior != int(dim_val):
                    raise ShapeMismatch(
                        f"symbol '{dim_spec}' bound inconsistently: {prior} vs {dim_val}"
                    )
                symbols[dim_spec] = int(dim_val)
            else:
                raise ShapeMismatch(
                    f"input '{name}' declares an unknown (None) dimension; use concrete dims or symbols"
                )
        env[f"{GRAPH_INPUT}{name}"] = torch.from_numpy(
            np.ascontiguousarray(array)
        ).to(device_obj)

    bound = _bindings(graph, package, device_obj)
    order = graph.topological_order()
    captures = set(capture or ())
    captured: dict[str, np.ndarray] = {}

    with torch.inference_mode():
        for nid in order:
            node = graph.nodes[nid]
            op = get_operator(node.op)
            kernel = _KERNELS.get(node.op)
            if kernel is None:
                raise UnsupportedOperator(
                    f"torch executor does not implement NIR operator '{node.op}' (node '{nid}'); "
                    f"supported operators: {', '.join(sorted(_KERNELS))}",
                    node=nid,
                    operator=node.op,
                )
            in_tensors = []
            for port in port_order(op, node.inputs):
                key = _key(node.inputs[port])
                if key not in env:
                    raise ShapeMismatch(f"node '{nid}': missing value for input port '{port}'")
                in_tensors.append(env[key])
            params = _resolve_params(node.params, symbols)
            outs = kernel(in_tensors, params, bound.get(nid, {}))
            tensors = [
                out if isinstance(out, torch.Tensor) else torch.as_tensor(out, device=device_obj)
                for out in outs
            ]
            if len(tensors) != max(1, len(node.outputs)):
                in_types = [
                    TensorType.of(tuple(env[_key(node.inputs[p])].shape), _dtype_name(env[_key(node.inputs[p])]))
                    for p in port_order(op, node.inputs)
                ]
                node.outputs = tuple(op.infer_types(in_types, node.params, {}))
            for index, tensor in enumerate(tensors):
                if validate_types and index < len(node.outputs):
                    declared = node.outputs[index]
                    try:
                        expected = replace_symbols(declared.shape, symbols)
                    except KeyError:
                        expected = None
                    if expected is not None and tuple(expected) != tuple(tensor.shape):
                        raise ShapeMismatch(
                            f"node '{nid}' ({node.op}) produced shape {tuple(tensor.shape)}, "
                            f"declared {declared.shape}"
                        )
                if check_finite and tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                    raise TensorInvalid(f"node '{nid}' ({node.op}) produced non-finite values")
                env[f"{nid}:{index}"] = tensor
            if nid in captures:
                captured[nid] = tensors[0].detach().cpu().numpy().copy()

    outputs: dict[str, np.ndarray] = {}
    for name, ref in graph.outputs.items():
        key = _key(ref)
        if key not in env:
            raise ShapeMismatch(f"graph output '{name}' references missing value {ref}")
        outputs[name] = env[key].detach().cpu().numpy()
    return TorchExecution(
        outputs=outputs, intermediates=captured, order=order, num_ops=len(order)
    )
