"""NIR operator library: typed signatures plus real NumPy kernels.

Every operator declares
  * how output shapes/dtypes are inferred from inputs, params and bound tensors,
  * a kernel that performs the actual computation on NumPy arrays,
  * a maturity level, so unsupported mathematical territory is reported honestly.

Parameter binding convention: any node param whose value is the string
``"param:<name>"`` (or a list of such strings) refers to a tensor in the bound
weight package. The executor resolves these and the validator checks shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from uzen.core.nir.types import Dim, TensorType, broadcast_shapes, dims_product
from uzen.policies.strict_plus.effects import Maturity
from uzen.errors import InvalidOperationParameters, ShapeMismatch, UnsupportedOperator

InferFn = Callable[[list[TensorType], Mapping[str, Any], Mapping[str, TensorType]], list[TensorType]]
RunFn = Callable[[list[np.ndarray], Mapping[str, Any], Mapping[str, np.ndarray]], list[np.ndarray]]


def is_param_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("param:")


def param_ref_names(value: Any) -> list[str]:
    if is_param_ref(value):
        return [value[len("param:"):]]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for v in value:
            out.extend(param_ref_names(v))
        return out
    return []


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    min_inputs: int
    max_inputs: int
    infer: InferFn
    run: RunFn
    description: str = ""
    maturity: Maturity = Maturity.IMPLEMENTED
    family: str = "core"
    differentiable_free: bool = True
    # Declared input-port order for kernels with positional semantics (q/k/v, x/adj).
    # Without this the executor would feed inputs alphabetically.
    ports: tuple[str, ...] = ()

    def infer_types(
        self,
        inputs: list[TensorType],
        params: Mapping[str, Any],
        bound: Mapping[str, TensorType],
    ) -> list[TensorType]:
        return self.infer(inputs, params, bound)

    def __str__(self) -> str:
        return f"op:{self.name}"


def port_order(op: OperatorSpec, inputs: Mapping[str, Any]) -> list[str]:
    """Deterministic input ordering: operator-declared ports first, else sorted."""
    if op.ports and all(port in inputs for port in op.ports):
        return list(op.ports)
    return sorted(inputs)


# --------------------------------------------------------------------------- #
# helpers used by inference functions
# --------------------------------------------------------------------------- #


def _require_bound(bound: Mapping[str, TensorType], name: str, op: str) -> TensorType:
    try:
        return bound[name]
    except KeyError as exc:
        raise InvalidOperationParameters(
            f"operator '{op}' needs bound tensor '{name}' but it was not provided"
        ) from exc


def _float_dtype(*types: TensorType) -> str:
    for t in types:
        if t.dtype.startswith("float") or t.dtype in ("float16", "float32", "float64", "bfloat16"):
            return t.dtype
    return "float32"


def _same(shape: Sequence[Dim], dtype: str) -> TensorType:
    return TensorType(tuple(shape), dtype)


def _param_float(params: Mapping[str, Any], key: str, default: float) -> float:
    v = params.get(key, default)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise InvalidOperationParameters(f"param '{key}' must be a number, got {v!r}")
    return float(v)


def _param_int(params: Mapping[str, Any], key: str, default: int) -> int:
    v = params.get(key, default)
    if not isinstance(v, int) or isinstance(v, bool):
        raise InvalidOperationParameters(f"param '{key}' must be an integer, got {v!r}")
    return int(v)


def _axis(value: Any, rank: int, *, name: str = "axis") -> tuple[int, ...]:
    if isinstance(value, int):
        axes = (value,)
    elif isinstance(value, (list, tuple)) and all(isinstance(v, int) for v in value):
        axes = tuple(int(v) for v in value)
    else:
        raise InvalidOperationParameters(f"param '{name}' must be an int or list of ints")
    out = []
    for a in axes:
        if a < -rank or a >= rank:
            raise InvalidOperationParameters(f"axis {a} out of range for rank {rank}")
        out.append(a % rank)
    return tuple(out)


# --------------------------------------------------------------------------- #
# elementary kernels
# --------------------------------------------------------------------------- #


def _linear_infer(inputs, params, bound):
    x = inputs[0]
    w = _require_bound(bound, params.get("weight", "weight").replace("param:", ""), "linear")
    if w.rank != 2:
        raise InvalidOperationParameters("linear weight must be 2-D (out_features, in_features)")
    out_features = w.shape[0]
    in_features = w.shape[1]
    if isinstance(x.shape[-1], int) and isinstance(in_features, int) and x.shape[-1] != in_features:
        raise ShapeMismatch(
            f"linear input feature dim {x.shape[-1]} != weight in_features {in_features}"
        )
    return [TensorType(tuple(x.shape[:-1]) + (out_features,), _float_dtype(x))]


def _linear_run(inputs, params, bound):
    x = inputs[0]
    name = str(params.get("weight", "weight")).replace("param:", "")
    w = bound[name]
    if x.shape[-1] != w.shape[1]:
        raise ShapeMismatch(f"linear: input last dim {x.shape[-1]} != weight in {w.shape[1]}")
    y = x @ w.T
    bname = params.get("bias")
    if bname:
        b = bound[str(bname).replace("param:", "")]
        if b.shape != (w.shape[0],):
            raise ShapeMismatch(f"linear bias shape {b.shape} != ({w.shape[0]},)")
        y = y + b
    return [y.astype(_float_dtype_name(x, w), copy=False)]


def _float_dtype_name(a: np.ndarray, b: np.ndarray) -> str:
    return str(np.result_type(a.dtype, b.dtype))


def _embedding_infer(inputs, params, bound):
    idx = inputs[0]
    w = _require_bound(bound, str(params.get("weight", "weight")).replace("param:", ""), "embedding")
    if w.rank != 2:
        raise InvalidOperationParameters("embedding weight must be 2-D (num_embeddings, dim)")
    return [TensorType(tuple(idx.shape) + (w.shape[1],), _float_dtype(w))]


def _embedding_run(inputs, params, bound):
    idx = inputs[0]
    w = bound[str(params.get("weight", "weight")).replace("param:", "")]
    if idx.dtype.kind not in "iu":
        raise InvalidOperationParameters(f"embedding indices must be integer, got {idx.dtype}")
    if idx.size and (idx.min() < 0 or idx.max() >= w.shape[0]):
        raise InvalidOperationParameters(
            f"embedding index out of range [0, {w.shape[0]}): "
            f"min={int(idx.min())}, max={int(idx.max())}"
        )
    return [w[idx]]


def _pos_embedding_infer(inputs, params, bound):
    x = inputs[0]
    w = _require_bound(bound, str(params.get("weight", "weight")).replace("param:", ""), "pos_embedding")
    return [_same(x.shape, _float_dtype(x, w))]


def _pos_embedding_run(inputs, params, bound):
    x = inputs[0]
    w = bound[str(params.get("weight", "weight")).replace("param:", "")]
    if x.ndim < 3:
        raise InvalidOperationParameters(f"pos_embedding expects (batch, seq, dim), got {x.shape}")
    t = x.shape[1]
    if t > w.shape[0]:
        raise ShapeMismatch(
            f"sequence length {t} exceeds positional embedding table size {w.shape[0]}"
        )
    return [x + w[:t].reshape((1, t) + tuple(x.shape[2:]))]


def _layernorm_infer(inputs, params, bound):
    x = inputs[0]
    eps = _param_float(params, "eps", 1e-5)
    if eps <= 0:
        raise InvalidOperationParameters("layernorm eps must be positive")
    if params.get("weight"):
        w = _require_bound(bound, str(params["weight"]).replace("param:", ""), "layernorm")
        if isinstance(x.shape[-1], int) and isinstance(w.shape[0], int) and x.shape[-1] != w.shape[0]:
            raise ShapeMismatch(f"layernorm weight size {w.shape[0]} != feature dim {x.shape[-1]}")
    return [_same(x.shape, _float_dtype(x))]


def _layernorm_run(inputs, params, bound):
    x = inputs[0].astype(np.float64, copy=False)
    eps = float(params.get("eps", 1e-5))
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    if params.get("weight"):
        g = bound[str(params["weight"]).replace("param:", "")]
        y = y * g
    if params.get("bias"):
        b = bound[str(params["bias"]).replace("param:", "")]
        y = y + b
    return [y.astype(inputs[0].dtype, copy=False)]


def _rmsnorm_infer(inputs, params, bound):
    x = inputs[0]
    _param_float(params, "eps", 1e-6)
    return [_same(x.shape, _float_dtype(x))]


def _rmsnorm_run(inputs, params, bound):
    x = inputs[0].astype(np.float64, copy=False)
    eps = float(params.get("eps", 1e-6))
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    y = x * scale
    if params.get("weight"):
        y = y * bound[str(params["weight"]).replace("param:", "")]
    return [y.astype(inputs[0].dtype, copy=False)]


def _binary_infer(op_name: str):
    def infer(inputs, params, bound):
        if len(inputs) != 2:
            raise InvalidOperationParameters(f"{op_name} requires exactly 2 inputs")
        shape = broadcast_shapes(inputs[0].shape, inputs[1].shape)
        return [TensorType(shape, _float_dtype(*inputs))]

    return infer


def _binary_run(fn):
    def run(inputs, params, bound):
        a, b = inputs
        return [np.asarray(fn(a, b))]

    return run


def _unary_infer(inputs, params, bound):
    if len(inputs) != 1:
        raise InvalidOperationParameters("unary operator requires exactly 1 input")
    return [_same(inputs[0].shape, _float_dtype(inputs[0]))]


def _activate(fn):
    def run(inputs, params, bound):
        return [fn(inputs[0]).astype(inputs[0].dtype, copy=False)]

    return run


def _gelu(x):
    x64 = x.astype(np.float64, copy=False)
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x64 * (1.0 + np.tanh(c * (x64 + 0.044715 * x64**3)))


def _silu(x):
    x64 = x.astype(np.float64, copy=False)
    return x64 / (1.0 + np.exp(-x64))


def _softmax_infer(inputs, params, bound):
    x = inputs[0]
    _axis(params.get("axis", -1), x.rank)
    return [_same(x.shape, _float_dtype(x))]


def _softmax_run(inputs, params, bound):
    x = inputs[0].astype(np.float64, copy=False)
    axis = _axis(params.get("axis", -1), x.ndim)[0]
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return [(e / e.sum(axis=axis, keepdims=True)).astype(inputs[0].dtype, copy=False)]


def _scale_infer(inputs, params, bound):
    x = inputs[0]
    _param_float(params, "factor", 1.0)
    return [_same(x.shape, _float_dtype(x))]


def _scale_run(inputs, params, bound):
    return [(inputs[0] * float(params.get("factor", 1.0))).astype(inputs[0].dtype, copy=False)]


def _matmul_infer(inputs, params, bound):
    a, b = inputs[0], inputs[1]
    if a.rank < 2 or b.rank < 2:
        raise InvalidOperationParameters("matmul requires rank >= 2 inputs")
    if isinstance(a.shape[-1], int) and isinstance(b.shape[-2], int) and a.shape[-1] != b.shape[-2]:
        raise ShapeMismatch(f"matmul inner dims disagree: {a.shape[-1]} vs {b.shape[-2]}")
    batch = broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return [TensorType(batch + (a.shape[-2], b.shape[-1]), _float_dtype(a, b))]


def _matmul_run(inputs, params, bound):
    return [np.matmul(inputs[0], inputs[1])]


def _split_qkv_infer(inputs, params, bound):
    x = inputs[0]
    heads = _param_int(params, "n_heads", 0)
    if heads <= 0:
        raise InvalidOperationParameters("split_qkv requires n_heads > 0")
    last = x.shape[-1]
    if not isinstance(last, int):
        raise InvalidOperationParameters("split_qkv requires a concrete last dimension")
    if last % (3 * heads) != 0:
        raise InvalidOperationParameters(
            f"split_qkv: last dim {last} is not divisible by 3*n_heads ({3*heads})"
        )
    dh = last // (3 * heads)
    seq = x.shape[:-2]
    out = TensorType(tuple(seq) + (heads, x.shape[-2] if x.rank >= 2 else None, dh), _float_dtype(x))
    return [out, out, out]


def _split_qkv_run(inputs, params, bound):
    x = inputs[0]
    if x.ndim != 3:
        raise InvalidOperationParameters(
            f"split_qkv expects (batch, seq, 3*dim), got shape {x.shape}"
        )
    heads = int(params["n_heads"])
    bsz, seq, last = x.shape
    dh = last // (3 * heads)
    # (B, T, 3, H, Dh) -> (B, 3, H, T, Dh)
    r = np.transpose(x.reshape(bsz, seq, 3, heads, dh), (0, 2, 3, 1, 4))
    return [r[:, 0], r[:, 1], r[:, 2]]


def _merge_heads_infer(inputs, params, bound):
    x = inputs[0]
    heads = _param_int(params, "n_heads", 0)
    if heads <= 0:
        raise InvalidOperationParameters("merge_heads requires n_heads > 0")
    if x.rank != 4:
        raise InvalidOperationParameters("merge_heads expects (batch, heads, seq, head_dim)")
    if isinstance(x.shape[1], int) and x.shape[1] != heads:
        raise ShapeMismatch(f"merge_heads: head dim {x.shape[1]} != n_heads {heads}")
    b, _, t, dh = x.shape
    total = dh * heads if isinstance(dh, int) else None
    return [TensorType((b, t, total), _float_dtype(x))]


def _merge_heads_run(inputs, params, bound):
    x = inputs[0]
    b, h, t, dh = x.shape
    return [np.transpose(x, (0, 2, 1, 3)).reshape(b, t, h * dh)]


def _attention_infer(inputs, params, bound):
    if len(inputs) != 3:
        raise InvalidOperationParameters("attention requires (q, k, v)")
    q, k, v = inputs
    if not (q.rank == k.rank == v.rank == 4):
        raise InvalidOperationParameters(
            "attention expects q,k,v of shape (batch, heads, seq, head_dim)"
        )
    if q.shape[-1] != k.shape[-1]:
        raise ShapeMismatch("attention: q/k head_dim mismatch")
    if k.shape[-2] != v.shape[-2]:
        raise ShapeMismatch("attention: k/v sequence length mismatch")
    return [TensorType(tuple(q.shape), _float_dtype(q, k, v))]


def _attention_run(inputs, params, bound):
    q, k, v = (a.astype(np.float64, copy=False) for a in inputs)
    scale = params.get("scale", "auto")
    scale = 1.0 / np.sqrt(q.shape[-1]) if scale == "auto" else float(scale)
    scores = np.einsum("bhtd,bhsd->bhts", q, k) * scale
    causal = bool(params.get("causal", True))
    if causal:
        t, s = scores.shape[-2], scores.shape[-1]
        if t != s:
            raise InvalidOperationParameters(
                f"causal attention requires square scores, got {t} x {s}"
            )
        mask = np.triu(np.ones((t, s), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    m = np.max(scores, axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(scores - m)
    denom = e.sum(axis=-1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    ctx = np.einsum("bhts,bhsd->bhtd", e / denom, v)
    return [ctx.astype(inputs[0].dtype, copy=False)]


def _reshape_infer(inputs, params, bound):
    x = inputs[0]
    target = params.get("shape")
    if not isinstance(target, (list, tuple)):
        raise InvalidOperationParameters("reshape requires a 'shape' list")
    dims: list[Dim] = []
    for d in target:
        if isinstance(d, int):
            dims.append(d)
        elif isinstance(d, str):
            if d not in x.shape:
                raise InvalidOperationParameters(
                    f"reshape symbol '{d}' not present in input shape {x.shape}"
                )
            dims.append(d)
        elif d is None:
            dims.append(None)
        else:
            raise InvalidOperationParameters(f"invalid reshape dimension {d!r}")
    # resolve a single -1 when the input is fully concrete
    if dims.count(-1) > 1:
        raise InvalidOperationParameters("reshape allows at most one -1")
    if -1 in dims and all(isinstance(v, int) for v in x.shape):
        known = dims_product([d for d in dims if d != -1])
        assert isinstance(known, int)
        total = int(np.prod(x.shape))
        if known == 0 or total % known != 0:
            raise ShapeMismatch(f"cannot reshape {x.shape} into {tuple(dims)}")
        dims[dims.index(-1)] = total // known
    if any(d == -1 for d in dims):
        raise InvalidOperationParameters("reshape with -1 requires concrete input dims")
    return [TensorType(tuple(dims), _float_dtype(x))]


def _reshape_run(inputs, params, bound):
    x = inputs[0]
    target = list(params["shape"])
    shape: list[int] = []
    for i, d in enumerate(target):
        if isinstance(d, int):
            shape.append(d)
        elif isinstance(d, str):
            shape.append(int(x.shape[list(x.shape).index(d)]))
        else:
            raise InvalidOperationParameters(f"cannot resolve reshape dim {d!r} at runtime")
    return [x.reshape(tuple(shape))]


def _transpose_infer(inputs, params, bound):
    x = inputs[0]
    perm = params.get("perm")
    if perm is None:
        perm = list(reversed(range(x.rank)))
    if not isinstance(perm, (list, tuple)) or sorted(int(p) for p in perm) != list(range(x.rank)):
        raise InvalidOperationParameters(f"invalid transpose perm {perm!r} for rank {x.rank}")
    return [TensorType(tuple(x.shape[int(p)] for p in perm), _float_dtype(x))]


def _transpose_run(inputs, params, bound):
    x = inputs[0]
    perm = params.get("perm")
    if perm is None:
        perm = list(reversed(range(x.ndim)))
    return [np.transpose(x, tuple(int(p) for p in perm))]


def _concat_infer(inputs, params, bound):
    if not inputs:
        raise InvalidOperationParameters("concat requires at least one input")
    axis = _axis(params.get("axis", -1), inputs[0].rank)[0]
    shape = list(inputs[0].shape)
    total: int | None = 0
    for t in inputs:
        if t.rank != inputs[0].rank:
            raise ShapeMismatch("concat inputs must have equal rank")
        for i, (a, b) in enumerate(zip(shape, t.shape)):
            if i == axis:
                continue
            if isinstance(a, int) and isinstance(b, int) and a != b:
                raise ShapeMismatch(f"concat dim {i} mismatch: {a} vs {b}")
        d = t.shape[axis]
        total = None if (total is None or not isinstance(d, int)) else total + d
    shape[axis] = total
    return [TensorType(tuple(shape), _float_dtype(*inputs))]


def _concat_run(inputs, params, bound):
    axis = int(params.get("axis", -1))
    return [np.concatenate(inputs, axis=axis)]


def _slice_infer(inputs, params, bound):
    x = inputs[0]
    axis = _axis(params.get("axis", 0), x.rank)[0]
    start = params.get("start")
    stop = params.get("stop")
    if start is not None and not isinstance(start, int):
        raise InvalidOperationParameters("slice start must be an int")
    if stop is not None and not isinstance(stop, int):
        raise InvalidOperationParameters("slice stop must be an int")
    shape = list(x.shape)
    if isinstance(x.shape[axis], int):
        s, e = slice(start, stop).indices(x.shape[axis])[:2]
        shape[axis] = max(0, e - s)
    else:
        shape[axis] = None
    return [TensorType(tuple(shape), _float_dtype(x))]


def _slice_run(inputs, params, bound):
    x = inputs[0]
    axis = int(params.get("axis", 0)) % x.ndim
    idx = [slice(None)] * x.ndim
    idx[axis] = slice(params.get("start"), params.get("stop"), params.get("step"))
    return [x[tuple(idx)]]


def _flatten_infer(inputs, params, bound):
    x = inputs[0]
    start = _param_int(params, "start_dim", 1)
    if not 0 <= start < max(1, x.rank):
        raise InvalidOperationParameters(f"flatten start_dim {start} out of range")
    prefix = tuple(x.shape[:start])
    rest = dims_product(x.shape[start:])
    return [TensorType(prefix + (rest,), _float_dtype(x))]


def _flatten_run(inputs, params, bound):
    x = inputs[0]
    start = int(params.get("start_dim", 1))
    prefix = x.shape[:start]
    return [x.reshape(prefix + (-1,))]


def _reduction_infer(inputs, params, bound):
    x = inputs[0]
    axes = _axis(params.get("axis", -1), x.rank)
    keep = bool(params.get("keepdims", False))
    shape = [d for i, d in enumerate(x.shape) if keep or i not in axes]
    return [TensorType(tuple(shape), _float_dtype(x))]


def _reduction_run(fn):
    def run(inputs, params, bound):
        x = inputs[0]
        axes = params.get("axis", -1)
        keep = bool(params.get("keepdims", False))
        return [np.asarray(fn(x, axis=axes, keepdims=keep))]

    return run


def _conv2d_infer(inputs, params, bound):
    x = inputs[0]
    w = _require_bound(bound, str(params.get("weight", "weight")).replace("param:", ""), "conv2d")
    if x.rank != 4 or w.rank != 4:
        raise InvalidOperationParameters("conv2d expects input (B,C,H,W) and weight (O,C,kh,kw)")
    if isinstance(x.shape[1], int) and isinstance(w.shape[1], int) and x.shape[1] != w.shape[1]:
        raise ShapeMismatch(f"conv2d channel mismatch: input {x.shape[1]} vs weight {w.shape[1]}")
    stride = _pair(params.get("stride", 1), "stride")
    padding = _pair(params.get("padding", 0), "padding")
    kh, kw = w.shape[2], w.shape[3]
    out = [x.shape[0], w.shape[0]]
    for i, (size, k, s, p) in enumerate(((x.shape[2], kh, stride[0], padding[0]), (x.shape[3], kw, stride[1], padding[1]))):
        if isinstance(size, int) and isinstance(k, int):
            out.append((size + 2 * p - k) // s + 1)
        else:
            out.append(None)
    return [TensorType(tuple(out), _float_dtype(x, w))]


def _pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return (int(value[0]), int(value[1]))
    raise InvalidOperationParameters(f"conv2d '{name}' must be an int or [h, w]")


def _conv2d_run(inputs, params, bound):
    x = inputs[0]
    w = bound[str(params.get("weight", "weight")).replace("param:", "")]
    stride = _pair(params.get("stride", 1), "stride")
    padding = _pair(params.get("padding", 0), "padding")
    if padding != (0, 0):
        x = np.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
    windows = np.lib.stride_tricks.sliding_window_view(x, (w.shape[2], w.shape[3]), axis=(2, 3))
    windows = windows[:, :, :: stride[0], :: stride[1]]
    y = np.einsum("bchwkl,ockl->bohw", windows, w)
    bname = params.get("bias")
    if bname:
        b = bound[str(bname).replace("param:", "")]
        y = y + b.reshape((1, -1, 1, 1))
    return [y.astype(np.result_type(x.dtype, w.dtype), copy=False)]


def _pool2d_infer(inputs, params, bound):
    x = inputs[0]
    if x.rank != 4:
        raise InvalidOperationParameters("pool2d expects (B,C,H,W)")
    kernel = _pair(params.get("kernel", 2), "kernel")
    stride = _pair(params.get("stride", params.get("kernel", 2)), "stride")
    padding = _pair(params.get("padding", 0), "padding")
    out = [x.shape[0], x.shape[1]]
    for size, k, s, p in ((x.shape[2], kernel[0], stride[0], padding[0]), (x.shape[3], kernel[1], stride[1], padding[1])):
        out.append((size + 2 * p - k) // s + 1 if isinstance(size, int) else None)
    return [TensorType(tuple(out), _float_dtype(x))]


def _pool2d_run(inputs, params, bound):
    x = inputs[0]
    kernel = _pair(params.get("kernel", 2), "kernel")
    stride = _pair(params.get("stride", params.get("kernel", 2)), "stride")
    padding = _pair(params.get("padding", 0), "padding")
    if padding != (0, 0):
        x = np.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(x, kernel, axis=(2, 3))
    windows = windows[:, :, :: stride[0], :: stride[1]]
    return [windows.max(axis=(-1, -2))]


def _recurrent_infer(kind: str):
    def infer(inputs, params, bound):
        x = inputs[0]
        if x.rank != 3:
            raise InvalidOperationParameters(f"{kind} expects input (batch, seq, features)")
        wh = _require_bound(bound, str(params.get("weight_hh", "weight_hh")).replace("param:", ""), kind)
        hidden = wh.shape[0] // (3 if kind == "gru" else 4 if kind == "lstm" else 1)
        return [TensorType((x.shape[0], x.shape[1], hidden), _float_dtype(x, wh))]

    return infer


def _rnn_run(inputs, params, bound):
    x = inputs[0]
    wx = bound[str(params.get("weight_ih", "weight_ih")).replace("param:", "")]
    wh = bound[str(params.get("weight_hh", "weight_hh")).replace("param:", "")]
    b = bound.get(str(params.get("bias", "bias")).replace("param:", "")) if params.get("bias") else None
    bsz, t, _ = x.shape
    hidden = wh.shape[0]
    h = np.zeros((bsz, hidden), dtype=np.float64)
    outs = []
    for step in range(t):
        h = np.tanh(x[:, step] @ wx.T + h @ wh.T + (b if b is not None else 0.0))
        outs.append(h)
    return [np.stack(outs, axis=1)]


def _lstm_run(inputs, params, bound):
    x = inputs[0]
    wx = bound[str(params.get("weight_ih", "weight_ih")).replace("param:", "")]
    wh = bound[str(params.get("weight_hh", "weight_hh")).replace("param:", "")]
    b = bound.get(str(params.get("bias", "bias")).replace("param:", "")) if params.get("bias") else None
    bsz, t, _ = x.shape
    hidden = wh.shape[0] // 4
    h = np.zeros((bsz, hidden), dtype=np.float64)
    c = np.zeros((bsz, hidden), dtype=np.float64)
    outs = []
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
    for step in range(t):
        gates = x[:, step] @ wx.T + h @ wh.T + (b if b is not None else 0.0)
        i = sig(gates[:, :hidden])
        f = sig(gates[:, hidden:2 * hidden])
        g = np.tanh(gates[:, 2 * hidden:3 * hidden])
        o = sig(gates[:, 3 * hidden:])
        c = f * c + i * g
        h = o * np.tanh(c)
        outs.append(h)
    return [np.stack(outs, axis=1)]


def _gru_run(inputs, params, bound):
    x = inputs[0]
    wx = bound[str(params.get("weight_ih", "weight_ih")).replace("param:", "")]
    wh = bound[str(params.get("weight_hh", "weight_hh")).replace("param:", "")]
    b = bound.get(str(params.get("bias", "bias")).replace("param:", "")) if params.get("bias") else None
    bsz, t, _ = x.shape
    hidden = wh.shape[0] // 3
    h = np.zeros((bsz, hidden), dtype=np.float64)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
    outs = []
    for step in range(t):
        xg = x[:, step] @ wx.T + (b if b is not None else 0.0)
        hg = h @ wh.T
        r = sig(xg[:, :hidden] + hg[:, :hidden])
        z = sig(xg[:, hidden:2 * hidden] + hg[:, hidden:2 * hidden])
        n = np.tanh(xg[:, 2 * hidden:] + r * hg[:, 2 * hidden:])
        h = (1.0 - z) * n + z * h
        outs.append(h)
    return [np.stack(outs, axis=1)]


def _identity_infer(inputs, params, bound):
    return [_same(inputs[0].shape, inputs[0].dtype)]


def _identity_run(inputs, params, bound):
    return [inputs[0]]


def _dropout_infer(inputs, params, bound):
    p = _param_float(params, "p", 0.0)
    if not 0.0 <= p < 1.0:
        raise InvalidOperationParameters("dropout p must be in [0, 1)")
    return [_same(inputs[0].shape, inputs[0].dtype)]


def _dropout_run(inputs, params, bound):
    # Inference semantics only: UZEN never executes a training forward pass.
    return [inputs[0]]


# --------------------------------------------------------------------------- #
# routed / stateful / sampling operators (families beyond transformers)
# --------------------------------------------------------------------------- #

_ACTIVATIONS = {
    "gelu": _gelu,
    "silu": _silu,
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
}


def _select_infer(inputs, params, bound):
    if len(inputs) != 3:
        raise InvalidOperationParameters("select requires (condition, a, b)")
    cond, a, b = inputs
    shape = broadcast_shapes(a.shape, b.shape)
    # A predicate must be broadcastable to the selected values; it may not enlarge them.
    broadcast_shapes(cond.shape, shape)
    return [TensorType(shape, _float_dtype(a, b))]


def _select_run(inputs, params, bound):
    cond, a, b = inputs
    return [np.where(np.asarray(cond) != 0, a, b)]


def _moe_infer(inputs, params, bound):
    x = inputs[0]
    if x.rank < 2:
        raise InvalidOperationParameters("moe expects (..., features)")
    router = _require_bound(bound, str(params.get("router", "router")).replace("param:", ""), "moe")
    up = _require_bound(bound, str(params.get("up_weight", "up_weight")).replace("param:", ""), "moe")
    down = _require_bound(
        bound, str(params.get("down_weight", "down_weight")).replace("param:", ""), "moe"
    )
    if router.rank != 2:
        raise InvalidOperationParameters("moe router weight must be 2-D (experts, features)")
    if up.rank != 3 or down.rank != 3:
        raise InvalidOperationParameters("moe expert stacks must be 3-D (experts, out, in)")
    n_experts, features = router.shape
    if isinstance(x.shape[-1], int) and isinstance(features, int) and x.shape[-1] != features:
        raise ShapeMismatch(f"moe feature dim {x.shape[-1]} != router features {features}")
    if isinstance(up.shape[0], int) and isinstance(n_experts, int) and up.shape[0] != n_experts:
        raise ShapeMismatch("moe: router and expert stacks disagree on expert count")
    if isinstance(up.shape[2], int) and isinstance(features, int) and up.shape[2] != features:
        raise ShapeMismatch("moe: expert up-projection input dim != model width")
    if isinstance(down.shape[1], int) and isinstance(features, int) and down.shape[1] != features:
        raise ShapeMismatch("moe: expert down-projection output dim != model width")
    if isinstance(up.shape[1], int) and isinstance(down.shape[2], int) and up.shape[1] != down.shape[2]:
        raise ShapeMismatch("moe: expert hidden dims disagree")
    top_k = _param_int(params, "top_k", 2)
    if top_k < 1 or (isinstance(n_experts, int) and top_k > n_experts):
        raise InvalidOperationParameters(
            f"moe top_k must be in [1, {n_experts}], got {top_k}"
        )
    if str(params.get("activation", "gelu")) not in _ACTIVATIONS:
        raise InvalidOperationParameters(
            f"moe activation must be one of {sorted(_ACTIVATIONS)}"
        )
    return [TensorType(tuple(x.shape), _float_dtype(x))]


def _moe_run(inputs, params, bound):
    x = inputs[0]
    router = bound[str(params.get("router", "router")).replace("param:", "")]
    up_w = bound[str(params.get("up_weight", "up_weight")).replace("param:", "")]
    down_w = bound[str(params.get("down_weight", "down_weight")).replace("param:", "")]
    up_b = (
        bound[str(params["up_bias"]).replace("param:", "")] if params.get("up_bias") else None
    )
    down_b = (
        bound[str(params["down_bias"]).replace("param:", "")]
        if params.get("down_bias")
        else None
    )
    activation = _ACTIVATIONS[str(params.get("activation", "gelu"))]
    x64 = x.astype(np.float64, copy=False)
    logits = x64 @ router.T.astype(np.float64)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    n_experts = probs.shape[-1]
    top_k = min(max(1, int(params.get("top_k", 2))), n_experts)
    selected = np.argpartition(-probs, top_k - 1, axis=-1)[..., :top_k]
    gates = np.take_along_axis(probs, selected, axis=-1)
    gates = gates / np.maximum(gates.sum(axis=-1, keepdims=True), 1e-12)
    # Sparse dispatch: an expert only computes the tokens routed to it, so unused
    # experts are never evaluated (and their parameters are never read).
    flat = x64.reshape(-1, x64.shape[-1])
    flat_selected = selected.reshape(-1, top_k)
    flat_gates = gates.reshape(-1, top_k)
    out = np.zeros_like(flat)
    for expert in range(n_experts):
        rows, slots = np.nonzero(flat_selected == expert)
        if rows.size == 0:
            continue
        gate = flat_gates[rows, slots][:, None]
        routed = flat[rows]
        hidden = routed @ up_w[expert].T.astype(np.float64)
        if up_b is not None:
            hidden = hidden + up_b[expert].astype(np.float64)
        hidden = activation(hidden)
        y = hidden @ down_w[expert].T.astype(np.float64)
        if down_b is not None:
            y = y + down_b[expert].astype(np.float64)
        np.add.at(out, rows, y * gate)
    return [out.reshape(x64.shape).astype(inputs[0].dtype, copy=False)]


def _ssm_infer(inputs, params, bound):
    x = inputs[0]
    if x.rank != 3:
        raise InvalidOperationParameters("ssm expects (batch, seq, channels)")
    a = _require_bound(bound, str(params.get("A", "A")).replace("param:", ""), "ssm")
    b = _require_bound(bound, str(params.get("B", "B")).replace("param:", ""), "ssm")
    c = _require_bound(bound, str(params.get("C", "C")).replace("param:", ""), "ssm")
    if a.rank != 2 or b.rank != 2 or c.rank != 2:
        raise InvalidOperationParameters("ssm expects A, B, C of shape (channels, state_dim)")
    channels = x.shape[-1]
    for name, tensor in (("A", a), ("B", b), ("C", c)):
        if isinstance(channels, int) and isinstance(tensor.shape[0], int) and tensor.shape[0] != channels:
            raise ShapeMismatch(f"ssm {name} channels {tensor.shape[0]} != input channels {channels}")
    if isinstance(a.shape[1], int) and isinstance(b.shape[1], int) and a.shape[1] != b.shape[1]:
        raise ShapeMismatch("ssm state dimensions disagree between A and B")
    if isinstance(b.shape[1], int) and isinstance(c.shape[1], int) and b.shape[1] != c.shape[1]:
        raise ShapeMismatch("ssm state dimensions disagree between B and C")
    return [TensorType(tuple(x.shape), _float_dtype(x, a))]


def _ssm_run(inputs, params, bound):
    x = inputs[0].astype(np.float64, copy=False)
    a = bound[str(params.get("A", "A")).replace("param:", "")].astype(np.float64)
    b = bound[str(params.get("B", "B")).replace("param:", "")].astype(np.float64)
    c = bound[str(params.get("C", "C")).replace("param:", "")].astype(np.float64)
    d = bound[str(params["D"]).replace("param:", "")].astype(np.float64) if params.get("D") else None
    bsz, steps, _ = x.shape
    state = np.zeros((bsz, a.shape[0], a.shape[1]), dtype=np.float64)
    outs = []
    for t in range(steps):
        # Diagonal state-space recurrence: h_t = A*h_{t-1} + B*x_t ; y_t = <C, h_t> + D*x_t
        state = a[None] * state + b[None] * x[:, t, :, None]
        y = np.einsum("bcn,cn->bc", state, c)
        if d is not None:
            y = y + d[None] * x[:, t]
        outs.append(y)
    return [np.stack(outs, axis=1).astype(inputs[0].dtype, copy=False)]


def _gcn_infer(inputs, params, bound):
    if len(inputs) != 2:
        raise InvalidOperationParameters("gcn requires (node_features, adjacency)")
    x, adj = inputs
    if x.rank != 3:
        raise InvalidOperationParameters("gcn node features must be (batch, nodes, features)")
    if adj.rank != 3:
        raise InvalidOperationParameters("gcn adjacency must be (batch, nodes, nodes)")
    if adj.shape[0] != x.shape[0] or adj.shape[1] != x.shape[1] or adj.shape[2] != x.shape[1]:
        raise ShapeMismatch(f"gcn adjacency {adj.shape} != (batch, {x.shape[1]}, {x.shape[1]})")
    w = _require_bound(bound, str(params.get("weight", "weight")).replace("param:", ""), "gcn")
    if w.rank != 2:
        raise InvalidOperationParameters("gcn weight must be 2-D (out_features, in_features)")
    if isinstance(x.shape[-1], int) and isinstance(w.shape[1], int) and x.shape[-1] != w.shape[1]:
        raise ShapeMismatch(f"gcn input features {x.shape[-1]} != weight in_features {w.shape[1]}")
    return [TensorType((x.shape[0], x.shape[1], w.shape[0]), _float_dtype(x, w))]


def _gcn_run(inputs, params, bound):
    x, adj = inputs
    w = bound[str(params.get("weight", "weight")).replace("param:", "")]
    a = adj.astype(np.float64, copy=False) + np.eye(adj.shape[-1], dtype=np.float64)[None]
    deg = a.sum(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    a_norm = a * inv * np.swapaxes(inv, -1, -2)
    y = np.einsum("bij,bjf->bif", a_norm, x.astype(np.float64)) @ w.T.astype(np.float64)
    if params.get("bias"):
        y = y + bound[str(params["bias"]).replace("param:", "")].astype(np.float64)
    return [y.astype(np.result_type(x.dtype, w.dtype), copy=False)]


def _sample_normal_infer(inputs, params, bound):
    if len(inputs) != 2:
        raise InvalidOperationParameters("sample_normal requires (mean, log_variance)")
    mu, logvar = inputs
    if mu.shape != logvar.shape:
        raise ShapeMismatch(f"sample_normal mean {mu.shape} != log_variance {logvar.shape}")
    _param_int(params, "seed", 0)
    return [_same(mu.shape, _float_dtype(mu, logvar))]


def _sample_normal_run(inputs, params, bound):
    mu, logvar = inputs
    # Reparameterization with an explicit seed: deterministic, reproducible, and never
    # gradient-carrying (consistent with STRICT_PLUS).
    eps = np.random.default_rng(int(params.get("seed", 0))).standard_normal(mu.shape)
    z = mu.astype(np.float64) + np.exp(0.5 * logvar.astype(np.float64)) * eps
    return [z.astype(mu.dtype, copy=False)]


def _add_noise_infer(inputs, params, bound):
    x = inputs[0]
    level = _param_float(params, "alpha_bar", 0.5)
    if not 0.0 < level <= 1.0:
        raise InvalidOperationParameters("add_noise alpha_bar must be in (0, 1]")
    _param_int(params, "seed", 0)
    return [_same(x.shape, _float_dtype(x))]


def _add_noise_run(inputs, params, bound):
    x = inputs[0]
    level = float(params.get("alpha_bar", 0.5))
    eps = np.random.default_rng(int(params.get("seed", 0))).standard_normal(x.shape)
    out = np.sqrt(level) * x.astype(np.float64) + np.sqrt(1.0 - level) * eps
    return [out.astype(x.dtype, copy=False)]


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


OPERATORS: dict[str, OperatorSpec] = {}


def register(op: OperatorSpec) -> OperatorSpec:
    if op.name in OPERATORS:
        raise ValueError(f"operator '{op.name}' already registered")
    OPERATORS[op.name] = op
    return op


def _reg(
    name,
    min_in,
    max_in,
    infer,
    run,
    description,
    family="core",
    maturity=Maturity.IMPLEMENTED,
    ports=(),
):
    register(
        OperatorSpec(
            name,
            min_in,
            max_in,
            infer,
            run,
            description,
            maturity=maturity,
            family=family,
            ports=tuple(ports),
        )
    )


_reg("linear", 1, 1, _linear_infer, _linear_run, "y = x @ W^T + b", "nn")
_reg("embedding", 1, 1, _embedding_infer, _embedding_run, "token-id lookup into a weight matrix", "nn")
_reg("pos_embedding", 1, 1, _pos_embedding_infer, _pos_embedding_run, "add a learned positional table", "nn")
_reg("layernorm", 1, 1, _layernorm_infer, _layernorm_run, "layer normalization over the last axis", "norm")
_reg("rmsnorm", 1, 1, _rmsnorm_infer, _rmsnorm_run, "root-mean-square normalization", "norm")
_reg("add", 2, 2, _binary_infer("add"), _binary_run(lambda a, b: a + b), "elementwise addition", "math")
_reg("sub", 2, 2, _binary_infer("sub"), _binary_run(lambda a, b: a - b), "elementwise subtraction", "math")
_reg("mul", 2, 2, _binary_infer("mul"), _binary_run(lambda a, b: a * b), "elementwise product", "math")
_reg("scale", 1, 1, _scale_infer, _scale_run, "scalar multiplication", "math")
_reg("matmul", 2, 2, _matmul_infer, _matmul_run, "batched matrix multiplication", "math")
_reg("relu", 1, 1, _unary_infer, _activate(lambda x: np.maximum(x, 0)), "rectified linear unit", "activation")
_reg("gelu", 1, 1, _unary_infer, _activate(_gelu), "Gaussian error linear unit (tanh approximation)", "activation")
_reg("silu", 1, 1, _unary_infer, _activate(_silu), "sigmoid linear unit (swish)", "activation")
_reg("tanh", 1, 1, _unary_infer, _activate(np.tanh), "hyperbolic tangent", "activation")
_reg("sigmoid", 1, 1, _unary_infer, _activate(lambda x: 1.0 / (1.0 + np.exp(-x))), "logistic sigmoid", "activation")
_reg("softmax", 1, 1, _softmax_infer, _softmax_run, "numerically stable softmax", "activation")
_reg("split_qkv", 1, 1, _split_qkv_infer, _split_qkv_run, "fused QKV projection -> three head tensors", "attention")
_reg("merge_heads", 1, 1, _merge_heads_infer, _merge_heads_run, "merge attention heads back to model width", "attention")
_reg("attention", 3, 3, _attention_infer, _attention_run, "scaled dot-product attention (optional causal mask)", "attention", ports=("q", "k", "v"))
_reg("reshape", 1, 1, _reshape_infer, _reshape_run, "shape transformation (symbolic dims allowed)", "structure")
_reg("transpose", 1, 1, _transpose_infer, _transpose_run, "axis permutation", "structure")
_reg("concat", 1, 32, _concat_infer, _concat_run, "concatenation along an axis", "structure")
_reg("slice", 1, 1, _slice_infer, _slice_run, "axis slice", "structure")
_reg("flatten", 1, 1, _flatten_infer, _flatten_run, "flatten trailing dimensions", "structure")
_reg("mean", 1, 1, _reduction_infer, _reduction_run(np.mean), "mean reduction", "math")
_reg("sum", 1, 1, _reduction_infer, _reduction_run(np.sum), "sum reduction", "math")
_reg("conv2d", 1, 1, _conv2d_infer, _conv2d_run, "2-D convolution (optional padding/stride)", "conv")
_reg("maxpool2d", 1, 1, _pool2d_infer, _pool2d_run, "2-D max pooling", "conv")
_reg("rnn", 1, 1, _recurrent_infer("rnn"), _rnn_run, "vanilla tanh recurrent layer over time", "recurrent")
_reg("lstm", 1, 1, _recurrent_infer("lstm"), _lstm_run, "LSTM layer over time", "recurrent")
_reg("gru", 1, 1, _recurrent_infer("gru"), _gru_run, "GRU layer over time", "recurrent")
_reg("identity", 1, 1, _identity_infer, _identity_run, "pass-through", "structure")
_reg("dropout", 1, 1, _dropout_infer, _dropout_run, "inference-mode dropout (identity)", "regularization")
_reg("select", 3, 3, _select_infer, _select_run, "conditional select: condition ? a : b", "control", ports=("cond", "a", "b"))
_reg("moe", 1, 1, _moe_infer, _moe_run, "top-k routed mixture-of-experts block (dense kernels)", "moe")
_reg("ssm", 1, 1, _ssm_infer, _ssm_run, "diagonal state-space layer over time (S4D-style recurrence)", "ssm")
_reg("gcn", 2, 2, _gcn_infer, _gcn_run, "graph convolution with symmetric degree-normalized adjacency", "gnn", ports=("x", "adj"))
_reg("sample_normal", 2, 2, _sample_normal_infer, _sample_normal_run, "reparameterized Gaussian sample (seeded)", "sampling")
_reg("add_noise", 1, 1, _add_noise_infer, _add_noise_run, "diffusion forward noising step (seeded)", "diffusion")


def get_operator(name: str) -> OperatorSpec:
    try:
        op = OPERATORS[name]
    except KeyError as exc:
        raise UnsupportedOperator(
            f"operator '{name}' is not implemented by this runtime; "
            f"known operators: {', '.join(sorted(OPERATORS))}"
        ) from exc
    if not op.maturity.executable:
        raise UnsupportedOperator(
            f"operator '{name}' has maturity '{op.maturity.value}' and cannot execute"
        )
    return op


def operator_catalogue() -> list[dict[str, Any]]:
    return [
        {
            "name": op.name,
            "family": op.family,
            "min_inputs": op.min_inputs,
            "max_inputs": op.max_inputs,
            "description": op.description,
            "maturity": op.maturity.value,
        }
        for op in sorted(OPERATORS.values(), key=lambda o: (o.family, o.name))
    ]
