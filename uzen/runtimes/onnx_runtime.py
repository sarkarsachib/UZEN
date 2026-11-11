"""ONNX interchange for the supported NIR subset.

This is a real converter, not a placeholder: it composes ONNX primitive operators so the
exported model matches the NumPy kernels numerically (for example GELU is emitted as its
tanh approximation, exactly like ``uzen.core.nir.ops._gelu``), and it can lift a matching
ONNX graph back into NIR with a bound :class:`WeightPackage`.

Coverage is explicitly incremental. Operators without a faithful mapping (attention,
recurrent cells, MoE, SSM, GCN, seeded sampling, split/merge heads) raise
``UnsupportedOperator`` naming the operator, instead of exporting something that would
compute a different function.

The optional ``onnx`` package is required for both directions. UZEN does not embed an
ONNX *executor*: execution stays in the NumPy NIR runtime or in an external runtime such
as onnxruntime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from uzen.core.nir.graph import Graph
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import InvalidOperationParameters, RuntimeUnavailable, UnsupportedOperator

try:  # optional dependency, never imported at module load in a way that breaks the core
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    ONNX_AVAILABLE = True
    ONNX_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised only without the extra
    onnx = None  # type: ignore[assignment]
    helper = None  # type: ignore[assignment]
    numpy_helper = None  # type: ignore[assignment]
    TensorProto = None  # type: ignore[assignment]
    ONNX_AVAILABLE = False
    ONNX_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

ONNX_OPSET = 17
ONNX_IR_VERSION = 8

_DTYPE_TO_ONNX = {
    "float32": "FLOAT",
    "float16": "FLOAT16",
    "float64": "DOUBLE",
    "int64": "INT64",
    "int32": "INT32",
    "bool": "BOOL",
}

_ONNX_TO_DTYPE = {
    "FLOAT": "float32",
    "FLOAT16": "float16",
    "DOUBLE": "float64",
    "INT64": "int64",
    "INT32": "int32",
    "BOOL": "bool",
}

# NIR operators with a faithful ONNX mapping (implemented in _emit_*).
_EXPORTABLE = {
    "linear",
    "embedding",
    "add",
    "sub",
    "mul",
    "scale",
    "matmul",
    "relu",
    "gelu",
    "silu",
    "tanh",
    "sigmoid",
    "softmax",
    "layernorm",
    "rmsnorm",
    "conv2d",
    "maxpool2d",
    "reshape",
    "transpose",
    "concat",
    "slice",
    "flatten",
    "mean",
    "sum",
    "select",
    "identity",
    "dropout",
}


def is_available() -> bool:
    return ONNX_AVAILABLE


def _require_onnx() -> None:
    if not ONNX_AVAILABLE:
        raise RuntimeUnavailable(
            "the ONNX adapter needs the optional 'onnx' package; install the 'onnx' extra "
            "(pip install -e '.[onnx]')",
            reason=ONNX_IMPORT_ERROR,
        )


def _onnx_dtype(dtype: str) -> int:
    name = _DTYPE_TO_ONNX.get(str(dtype))
    if name is None:
        raise UnsupportedOperator(f"ONNX export does not support dtype '{dtype}'")
    return int(getattr(TensorProto, name))


def _np_dtype(np_dtype: Any) -> str:
    dt = np.dtype(np_dtype)
    mapping = {
        np.dtype(np.float32): "float32",
        np.dtype(np.float16): "float16",
        np.dtype(np.float64): "float64",
        np.dtype(np.int64): "int64",
        np.dtype(np.int32): "int32",
        np.dtype(np.bool_): "bool",
    }
    if dt not in mapping:
        raise UnsupportedOperator(f"ONNX import does not support dtype '{dt}'")
    return mapping[dt]


def _dim_list(shape: Iterable[Any]) -> list[Any]:
    dims: list[Any] = []
    for d in shape:
        dims.append(int(d) if isinstance(d, int) else str(d))
    return dims


class _Exporter:
    def __init__(self, graph: Graph, package: WeightPackage, *, model_name: str) -> None:
        self.graph = graph
        self.package = package
        self.model_name = model_name
        self.nodes: list[Any] = []
        self.initializers: list[Any] = []
        self._initializer_names: set[str] = set()
        self.env: dict[str, str] = {f"input:{name}": name for name in graph.inputs}
        self.counter = 0

    # ------------------------------------------------------------------ #

    def initializer(self, name: str, array: np.ndarray) -> str:
        if name not in self._initializer_names:
            arr = np.ascontiguousarray(array)
            self.initializers.append(numpy_helper.from_array(arr, name=name))
            self._initializer_names.add(name)
        return name

    def constant(self, prefix: str, value: Any, dtype: str = "float32") -> str:
        self.counter += 1
        name = f"{prefix}__const{self.counter}"
        return self.initializer(name, np.asarray(value, dtype=dtype))

    def tensor(self, ref: Ref) -> str:
        key = f"{ref.node}:{ref.port}"
        if ref.node.startswith("input:"):
            return self.env.get(f"input:{ref.node[len('input:'):]}")
        try:
            return self.env[key]
        except KeyError as exc:
            raise InvalidOperationParameters(f"ONNX export: unresolved input '{ref}'") from exc

    def emit(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], name=output, **attrs))
        return output

    def param_name(self, value: Any) -> str:
        if not isinstance(value, str) or not value.startswith("param:"):
            raise InvalidOperationParameters(f"expected a parameter reference, got {value!r}")
        return value[len("param:") :]

    # ------------------------------------------------------------------ #

    def emit_node(self, node_id: str, op: str, inputs: Mapping[str, Ref], params: Mapping[str, Any]) -> None:
        def out(port: int = 0) -> str:
            return f"{node_id}:{port}"

        def arg(port: str) -> str:
            return self.tensor(inputs[port])

        if op == "linear":
            weight_name = self.param_name(params["weight"])
            weight = self.package.tensors[weight_name]
            wt = self.initializer(f"{node_id}__weight_t", np.asarray(weight).T)
            current = self.emit("MatMul", [arg("x"), wt], f"{node_id}__matmul")
            if params.get("bias"):
                bias = self.package.tensors[self.param_name(params["bias"])]
                b = self.initializer(f"{node_id}__bias", np.asarray(bias))
                current = self.emit("Add", [current, b], out())
            else:
                current = self.emit("Identity", [current], out())
        elif op == "embedding":
            weight = self.package.tensors[self.param_name(params["weight"])]
            w = self.initializer(f"{node_id}__embedding", np.asarray(weight))
            self.emit("Gather", [w, arg("x")], out(), axis=0)
        elif op in ("add", "sub", "mul"):
            self.emit({"add": "Add", "sub": "Sub", "mul": "Mul"}[op], [arg("a"), arg("b")], out())
        elif op == "scale":
            factor = float(params.get("factor", 1.0))
            c = self.constant(node_id, np.asarray(factor, dtype=np.float32))
            self.emit("Mul", [arg("x"), c], out())
        elif op == "matmul":
            self.emit("MatMul", [arg("a"), arg("b")], out())
        elif op in ("relu", "tanh", "sigmoid"):
            self.emit({"relu": "Relu", "tanh": "Tanh", "sigmoid": "Sigmoid"}[op], [arg("x")], out())
        elif op == "softmax":
            self.emit("Softmax", [arg("x")], out(), axis=int(params.get("axis", -1)))
        elif op == "gelu":
            x = arg("x")
            three = self.constant(node_id, np.asarray(3.0, dtype=np.float32))
            c044 = self.constant(node_id, np.asarray(0.044715, dtype=np.float32))
            c_sqrt = self.constant(node_id, np.asarray(np.sqrt(2.0 / np.pi), dtype=np.float32))
            half = self.constant(node_id, np.asarray(0.5, dtype=np.float32))
            one = self.constant(node_id, np.asarray(1.0, dtype=np.float32))
            x3 = self.emit("Pow", [x, three], f"{node_id}__x3")
            scaled = self.emit("Mul", [x3, c044], f"{node_id}__scaled")
            inner = self.emit("Add", [x, scaled], f"{node_id}__inner")
            inner = self.emit("Mul", [inner, c_sqrt], f"{node_id}__sqrt2pi")
            tanh = self.emit("Tanh", [inner], f"{node_id}__tanh")
            plus1 = self.emit("Add", [tanh, one], f"{node_id}__plus1")
            halfx = self.emit("Mul", [x, half], f"{node_id}__halfx")
            self.emit("Mul", [halfx, plus1], out())
        elif op == "silu":
            x = arg("x")
            sig = self.emit("Sigmoid", [x], f"{node_id}__sigmoid")
            self.emit("Mul", [x, sig], out())
        elif op == "layernorm":
            scale = self.package.tensors[self.param_name(params["weight"])]
            scale_name = self.initializer(f"{node_id}__ln_scale", np.asarray(scale))
            ln_inputs = [arg("x"), scale_name]
            if params.get("bias"):
                bias = self.package.tensors[self.param_name(params["bias"])]
                ln_inputs.append(self.initializer(f"{node_id}__ln_bias", np.asarray(bias)))
            self.emit(
                "LayerNormalization",
                ln_inputs,
                out(),
                axis=-1,
                epsilon=float(params.get("eps", 1e-5)),
            )
        elif op == "rmsnorm":
            x = arg("x")
            scale = self.package.tensors[self.param_name(params["weight"])]
            s = self.initializer(f"{node_id}__rms_scale", np.asarray(scale))
            eps = self.constant(node_id, np.asarray(float(params.get("eps", 1e-6)), dtype=np.float32))
            sq = self.emit("Mul", [x, x], f"{node_id}__sq")
            mean = self.emit("ReduceMean", [sq], f"{node_id}__mean", axes=[-1], keepdims=1)
            var = self.emit("Add", [mean, eps], f"{node_id}__var")
            root = self.emit("Sqrt", [var], f"{node_id}__root")
            norm = self.emit("Div", [x, root], f"{node_id}__norm")
            self.emit("Mul", [norm, s], out())
        elif op == "conv2d":
            weight = self.package.tensors[self.param_name(params["weight"])]
            w = self.initializer(f"{node_id}__conv", np.asarray(weight))
            conv_inputs = [arg("x"), w]
            if params.get("bias"):
                bias = self.package.tensors[self.param_name(params["bias"])]
                conv_inputs.append(self.initializer(f"{node_id}__conv_bias", np.asarray(bias)))
            pad = int(params.get("padding", 0))
            self.emit(
                "Conv",
                conv_inputs,
                out(),
                strides=[int(params.get("stride", 1))] * 2,
                pads=[pad, pad, pad, pad],
            )
        elif op == "maxpool2d":
            kernel = int(params.get("kernel", 2))
            stride = int(params.get("stride", kernel))
            pad = int(params.get("padding", 0))
            self.emit(
                "MaxPool",
                [arg("x")],
                out(),
                kernel_shape=[kernel, kernel],
                strides=[stride, stride],
                pads=[pad, pad, pad, pad],
            )
        elif op == "reshape":
            target = list(params["shape"])
            resolved = [0 if isinstance(d, str) else int(d) for d in target]
            shape_name = self.initializer(f"{node_id}__shape", np.asarray(resolved, dtype=np.int64))
            self.emit("Reshape", [arg("x"), shape_name], out())
        elif op == "transpose":
            self.emit("Transpose", [arg("x")], out(), perm=[int(p) for p in params["perm"]])
        elif op == "concat":
            ports = [p for p in inputs]
            self.emit("Concat", [self.tensor(inputs[p]) for p in ports], out(), axis=int(params.get("axis", -1)))
        elif op == "slice":
            x = arg("x")
            axis = int(params.get("axis", 0))
            start = params.get("start")
            stop = params.get("stop")
            step = params.get("step")
            starts = self.initializer(f"{node_id}__starts", np.asarray([0 if start is None else int(start)], dtype=np.int64))
            ends = self.initializer(
                f"{node_id}__ends",
                np.asarray([(2**63 - 1) if stop is None else int(stop)], dtype=np.int64),
            )
            axes = self.initializer(f"{node_id}__axes", np.asarray([axis], dtype=np.int64))
            slice_inputs = [x, starts, ends, axes]
            if step is not None:
                steps = self.initializer(f"{node_id}__steps", np.asarray([int(step)], dtype=np.int64))
                slice_inputs.append(steps)
            self.emit("Slice", slice_inputs, out())
        elif op == "flatten":
            self.emit("Flatten", [arg("x")], out(), axis=int(params.get("start_dim", 1)))
        elif op in ("mean", "sum"):
            axis = params.get("axis", -1)
            axes = [int(a) for a in (axis if isinstance(axis, (list, tuple)) else [axis])]
            onnx_op = "ReduceMean" if op == "mean" else "ReduceSum"
            self.emit(onnx_op, [arg("x")], out(), axes=axes, keepdims=int(bool(params.get("keepdims", False))))
        elif op == "select":
            self.emit("Where", [arg("cond"), arg("a"), arg("b")], out())
        elif op in ("identity", "dropout"):
            self.emit("Identity", [arg("x")], out())
        else:  # pragma: no cover - guarded by _EXPORTABLE
            raise UnsupportedOperator(f"no ONNX mapping for NIR operator '{op}'")

        self.env[f"{node_id}:0"] = out()

    # ------------------------------------------------------------------ #

    def build(self):
        report = self.graph.validate(self.package)
        report.require()
        for nid in self.graph.topological_order():
            node = self.graph.nodes[nid]
            if node.op not in _EXPORTABLE:
                raise UnsupportedOperator(
                    f"ONNX export does not implement NIR operator '{node.op}' (node '{nid}'); "
                    f"exportable operators: {', '.join(sorted(_EXPORTABLE))}",
                    node=nid,
                    operator=node.op,
                )
            self.emit_node(nid, node.op, node.inputs, node.params)

        inputs = [
            helper.make_tensor_value_info(
                name, _onnx_dtype(tt.dtype), _dim_list(tt.shape)
            )
            for name, tt in self.graph.inputs.items()
        ]
        outputs = []
        for name, ref in self.graph.outputs.items():
            node = self.graph.nodes[ref.node]
            declared = node.outputs[ref.port]
            outputs.append(
                helper.make_tensor_value_info(
                    self.tensor(ref), _onnx_dtype(declared.dtype), _dim_list(declared.shape)
                )
            )
        graph_proto = helper.make_graph(
            self.nodes,
            self.model_name,
            inputs,
            outputs,
            initializer=self.initializers,
        )
        model = helper.make_model(
            graph_proto,
            producer_name="uzen",
            opset_imports=[helper.make_opsetid("", ONNX_OPSET)],
        )
        model.ir_version = ONNX_IR_VERSION
        onnx.checker.check_model(model)
        return model


def export_onnx(
    graph: Graph,
    package: WeightPackage,
    *,
    model_name: str | None = None,
    save_path: str | Path | None = None,
) -> Any:
    """Convert a validated NIR graph + package into an ONNX ModelProto."""
    _require_onnx()
    exporter = _Exporter(graph, package, model_name=model_name or graph.name or "uzen_graph")
    model = exporter.build()
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(destination))
    return model


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #


def _value_info(name: str, info: Any) -> tuple[str, TensorType]:
    tensor_type = info.type.tensor_type
    dtype = _ONNX_TO_DTYPE.get(TensorProto.DataType.Name(tensor_type.elem_type))
    if dtype is None:
        raise UnsupportedOperator(
            f"ONNX input '{name}' uses unsupported element type {tensor_type.elem_type}"
        )
    dims: list[Any] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(dim.dim_param or "N")
    return name, TensorType.of(dims, dtype)


def _const_input(node: Any, index: int, arrays: Mapping[str, np.ndarray]) -> np.ndarray | None:
    if index >= len(node.input):
        return None
    name = node.input[index]
    return arrays.get(name)


def import_onnx(source: str | Path | Any) -> tuple[Graph, WeightPackage, dict[str, Any]]:
    """Lift a supported ONNX graph into NIR + a bound weight package.

    Supports the same subset the exporter emits, including the common
    ``MatMul(constant) [+ Add(constant)]`` linear pattern. Unsupported node types are
    reported together, never skipped.
    """
    _require_onnx()
    model = source if isinstance(source, onnx.ModelProto) else onnx.load(str(source))
    onnx.checker.check_model(model)

    arrays: dict[str, np.ndarray] = {
        init.name: np.asarray(numpy_helper.to_array(init)) for init in model.graph.initializer
    }
    lifted: dict[str, np.ndarray] = {}
    graph = Graph(name=model.graph.name or "onnx_import")

    refs: dict[str, Ref] = {}
    for info in model.graph.input:
        if info.name in arrays:
            continue
        name, tt = _value_info(info.name, info)
        graph.add_input(name, tt)
        refs[name] = Ref(f"input:{name}")

    produced: dict[str, str] = {}
    unsupported: list[str] = []

    def lifted_name(prefix: str, array: np.ndarray) -> str:
        base = f"onnx.{prefix}"
        name = base
        counter = 1
        while name in lifted or name in arrays:
            counter += 1
            name = f"{base}.{counter}"
        lifted[name] = np.asarray(array)
        return name

    def node_id(node: Any, index: int) -> str:
        raw = node.name or f"{node.op_type.lower()}_{index}"
        cleaned = "".join(c if (c.isalnum() or c in "._-") else "_" for c in raw)
        return cleaned or f"node_{index}"

    def ref_of(name: str) -> Ref:
        try:
            return refs[name]
        except KeyError as exc:
            raise UnsupportedOperator(
                f"ONNX import: value '{name}' is neither a graph input nor a supported node output"
            ) from exc

    def single(nid: str, op: str, node: Any, **params: Any) -> None:
        src = ref_of(node.input[0])
        ref = graph.add_node(nid, op, {"x": src}, params or None)
        refs[node.output[0]] = ref
        produced[node.output[0]] = nid

    for index, node in enumerate(model.graph.node):
        nid = node_id(node, index)
        op_type = node.op_type
        try:
            if op_type == "MatMul":
                const = _const_input(node, 1, arrays)
                if const is not None and const.ndim == 2:
                    name = lifted_name(nid, np.asarray(const).T)
                    ref = graph.add_node(
                        nid, "linear", {"x": ref_of(node.input[0])}, {"weight": f"param:{name}"}
                    )
                    refs[node.output[0]] = ref
                    produced[node.output[0]] = nid
                else:
                    a, b = ref_of(node.input[0]), ref_of(node.input[1])
                    refs[node.output[0]] = graph.add_node(nid, "matmul", {"a": a, "b": b})
                    produced[node.output[0]] = nid
            elif op_type == "Gemm":
                trans_b = int(next((a.i for a in node.attribute if a.name == "transB"), 0))
                weight = _const_input(node, 1, arrays)
                if weight is None:
                    raise UnsupportedOperator("ONNX Gemm with a non-constant weight is unsupported")
                weight = np.asarray(weight)
                if trans_b:
                    weight = weight.T
                name = lifted_name(f"{nid}.weight", weight)
                params: dict[str, Any] = {"weight": f"param:{name}"}
                bias = _const_input(node, 2, arrays)
                if bias is not None:
                    bias_name = lifted_name(f"{nid}.bias", np.asarray(bias))
                    params["bias"] = f"param:{bias_name}"
                ref = graph.add_node(nid, "linear", {"x": ref_of(node.input[0])}, params)
                refs[node.output[0]] = ref
                produced[node.output[0]] = nid
            elif op_type == "Add":
                const = _const_input(node, 1, arrays)
                src = node.input[0]
                if const is not None and const.ndim == 1 and src in produced:
                    target = graph.nodes[produced[src]]
                    if target.op == "linear" and "bias" not in target.params:
                        name = lifted_name(f"{nid}.bias", np.asarray(const))
                        target.params["bias"] = f"param:{name}"
                        refs[node.output[0]] = refs[src]
                        continue
                a, b = ref_of(node.input[0]), ref_of(node.input[1])
                refs[node.output[0]] = graph.add_node(nid, "add", {"a": a, "b": b})
                produced[node.output[0]] = nid
            elif op_type in ("Sub", "Mul"):
                a, b = ref_of(node.input[0]), ref_of(node.input[1])
                refs[node.output[0]] = graph.add_node(nid, op_type.lower(), {"a": a, "b": b})
                produced[node.output[0]] = nid
            elif op_type == "Relu":
                single(nid, "relu", node)
            elif op_type == "Tanh":
                single(nid, "tanh", node)
            elif op_type == "Sigmoid":
                single(nid, "sigmoid", node)
            elif op_type == "Softmax":
                axis = int(next((a.i for a in node.attribute if a.name == "axis"), -1))
                single(nid, "softmax", node, axis=axis)
            elif op_type == "Identity":
                single(nid, "identity", node)
            elif op_type == "Where":
                refs[node.output[0]] = graph.add_node(
                    nid,
                    "select",
                    {
                        "cond": ref_of(node.input[0]),
                        "a": ref_of(node.input[1]),
                        "b": ref_of(node.input[2]),
                    },
                )
                produced[node.output[0]] = nid
            elif op_type == "Reshape":
                shape = _const_input(node, 1, arrays)
                if shape is None:
                    raise UnsupportedOperator("ONNX Reshape with a dynamic shape is unsupported")
                target = [int(d) for d in shape.tolist()]
                if 0 in target:
                    raise UnsupportedOperator(
                        "ONNX Reshape uses 0 (copy the corresponding input dim), which has no "
                        "NIR equivalent and is not silently rewritten"
                    )
                single(nid, "reshape", node, shape=target)
            elif op_type == "Transpose":
                perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
                single(nid, "transpose", node, perm=perm) if perm else single(nid, "transpose", node)
            elif op_type == "Concat":
                axis = int(next((a.i for a in node.attribute if a.name == "axis"), -1))
                ports = {chr(ord("a") + i): ref_of(name) for i, name in enumerate(node.input)}
                refs[node.output[0]] = graph.add_node(nid, "concat", ports, {"axis": axis})
                produced[node.output[0]] = nid
            elif op_type == "Flatten":
                axis = int(next((a.i for a in node.attribute if a.name == "axis"), 1))
                single(nid, "flatten", node, start_dim=axis)
            elif op_type in ("ReduceMean", "ReduceSum"):
                axes = next((list(a.ints) for a in node.attribute if a.name == "axes"), None)
                if axes is None:
                    axes_initializer = _const_input(node, 1, arrays)
                    axes = list(np.asarray(axes_initializer).tolist()) if axes_initializer is not None else [-1]
                keepdims = bool(next((a.i for a in node.attribute if a.name == "keepdims"), 1))
                axis = axes[0] if len(axes) == 1 else axes
                single(
                    nid,
                    "mean" if op_type == "ReduceMean" else "sum",
                    node,
                    axis=axis,
                    keepdims=keepdims,
                )
            elif op_type == "Conv":
                weight = _const_input(node, 1, arrays)
                if weight is None:
                    raise UnsupportedOperator("ONNX Conv with a non-constant weight is unsupported")
                params = {"weight": f"param:{lifted_name(f'{nid}.weight', np.asarray(weight))}"}
                bias = _const_input(node, 2, arrays)
                if bias is not None:
                    params["bias"] = f"param:{lifted_name(f'{nid}.bias', np.asarray(bias))}"
                for attr in node.attribute:
                    if attr.name == "strides":
                        params["stride"] = int(attr.ints[0])
                    if attr.name == "pads":
                        params["padding"] = int(attr.ints[0])
                single(nid, "conv2d", node, **params)
            elif op_type == "MaxPool":
                params: dict[str, Any] = {}
                for attr in node.attribute:
                    if attr.name == "kernel_shape":
                        params["kernel"] = int(attr.ints[0])
                    if attr.name == "strides":
                        params["stride"] = int(attr.ints[0])
                    if attr.name == "pads":
                        params["padding"] = int(attr.ints[0])
                single(nid, "maxpool2d", node, **params)
            else:
                unsupported.append(f"{nid}:{op_type}")
        except UnsupportedOperator as exc:
            unsupported.append(f"{nid}:{op_type} ({exc.message})")

    if unsupported:
        raise UnsupportedOperator(
            "ONNX import encountered nodes without a faithful NIR mapping: "
            + ", ".join(unsupported[:12]),
            nodes=unsupported,
        )

    for output in model.graph.output:
        if output.name not in refs:
            raise UnsupportedOperator(
                f"ONNX output '{output.name}' is not produced by a supported node"
            )
        graph.set_output(output.name, refs[output.name])

    package = WeightPackage.from_arrays(
        lifted,
        name=f"{graph.name}_weights",
        family="onnx_import",
        metadata={"source": "onnx", "opset": ONNX_OPSET},
    )
    report = graph.validate(package)
    if not report.ok:
        raise InvalidOperationParameters(
            "imported ONNX graph does not validate as NIR", errors=report.errors[:10]
        )
    metadata = {
        "ir_version": int(model.ir_version),
        "producer": model.producer_name,
        "opset": [f"{o.domain or 'ai.onnx'}:{o.version}" for o in model.opset_import],
        "num_nodes": len(model.graph.node),
        "num_initializers": len(arrays),
    }
    return graph, package, metadata
