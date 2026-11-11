"""NumPy execution of NIR graphs (forward-only, inference semantics).

The executor is the only place where a graph becomes numerical reality. It is
strict: an unknown operator, a missing bound tensor, a shape disagreement or a
non-finite output is reported as a typed error rather than being silently
tolerated. STRICT_PLUS is enforced at the boundary: execution never enables
autograd and parameters are treated as constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from uzen.core.nir.graph import Graph
from uzen.core.nir.ops import get_operator, port_order
from uzen.core.nir.types import Ref, TensorType, replace_symbols
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import ShapeMismatch, TensorInvalid

GRAPH_INPUT = "input:"


@dataclass
class ExecutionResult:
    outputs: dict[str, np.ndarray]
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    num_ops: int = 0

    def output(self, name: str) -> np.ndarray:
        try:
            return self.outputs[name]
        except KeyError as exc:
            raise TensorInvalid(
                f"graph produced no output named '{name}'; available: {sorted(self.outputs)}"
            ) from exc

    def summary(self) -> dict[str, Any]:
        return {
            "outputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in self.outputs.items()},
            "num_ops": self.num_ops,
            "captured_intermediates": sorted(self.intermediates),
        }


def resolve_bindings(
    graph: Graph, package: WeightPackage
) -> dict[str, dict[str, np.ndarray]]:
    """Resolve ``param:`` references into concrete arrays per node."""
    bound: dict[str, dict[str, np.ndarray]] = {}
    for nid, node in graph.nodes.items():
        # Bound tensors are keyed by their *tensor name* (the kernels and shape
        # inference functions address parameters by name, e.g. "embed.weight").
        resolved: dict[str, np.ndarray] = {}
        for value in node.params.values():
            for name, _ in _collect_refs(value):
                resolved[name] = package.get(name)
        bound[nid] = resolved
    return bound


def _collect_refs(value: Any) -> list[tuple[str, Any]]:
    from uzen.core.nir.ops import param_ref_names

    if isinstance(value, str) and value.startswith("param:"):
        return [(value[len("param:"):], None)]
    if isinstance(value, (list, tuple)):
        out: list[tuple[str, Any]] = []
        for v in value:
            out.extend(_collect_refs(v))
        return out
    return []


def bind_tensors(graph: Graph, params: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Resolve param references from an explicit name->array mapping."""
    bound: dict[str, dict[str, np.ndarray]] = {}
    missing: list[str] = []
    for nid, node in graph.nodes.items():
        resolved: dict[str, np.ndarray] = {}
        for value in node.params.values():
            for name, _ in _collect_refs(value):
                if name not in params:
                    missing.append(name)
                    continue
                resolved[name] = params[name]
        bound[nid] = resolved
    if missing:
        raise ShapeMismatch(f"missing bound tensors: {sorted(set(missing))[:10]}")
    return bound


def execute(
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
    params: WeightPackage | Mapping[str, np.ndarray] | None = None,
    *,
    capture: Sequence[str] | None = None,
    check_finite: bool = True,
    validate_types: bool = True,
) -> ExecutionResult:
    """Run a graph forward on concrete inputs."""
    package: WeightPackage | None
    if isinstance(params, WeightPackage):
        package = params
        bound = resolve_bindings(graph, package)
    elif params is None:
        bound = {}
    else:
        package = None
        bound = bind_tensors(graph, params)

    missing_inputs = [name for name in graph.inputs if name not in inputs]
    if missing_inputs:
        raise ShapeMismatch(f"missing graph inputs: {missing_inputs}")
    extra = [name for name in inputs if name not in graph.inputs]
    if extra:
        raise ShapeMismatch(f"unexpected graph inputs: {extra}")

    env: dict[str, np.ndarray] = {}
    symbol_env: dict[str, int] = {}
    for name, spec in graph.inputs.items():
        arr = np.asarray(inputs[name])
        if arr.ndim != len(spec.shape):
            raise ShapeMismatch(
                f"input '{name}' has rank {arr.ndim}, declared rank {len(spec.shape)} "
                f"({spec.shape})"
            )
        for dim_spec, dim_val in zip(spec.shape, arr.shape):
            if isinstance(dim_spec, int):
                if int(dim_spec) != int(dim_val):
                    raise ShapeMismatch(
                        f"input '{name}' dimension {dim_val} != declared {dim_spec} "
                        f"(declared shape {spec.shape})"
                    )
            elif isinstance(dim_spec, str):
                prior = symbol_env.get(dim_spec)
                if prior is not None and prior != dim_val:
                    raise ShapeMismatch(
                        f"symbol '{dim_spec}' bound inconsistently: {prior} vs {dim_val}"
                    )
                symbol_env[dim_spec] = int(dim_val)
            else:
                raise ShapeMismatch(
                    f"input '{name}' declares an unknown (None) dimension; "
                    "graph inputs must use concrete dims or symbols"
                )
        env[f"{GRAPH_INPUT}{name}"] = arr

    order = graph.topological_order()
    captures = set(capture or ())
    captured: dict[str, np.ndarray] = {}

    for nid in order:
        node = graph.nodes[nid]
        op = get_operator(node.op)
        in_arrays: list[np.ndarray] = []
        for port in port_order(op, node.inputs):
            ref = node.inputs[port]
            key = _key(ref)
            if key not in env:
                raise ShapeMismatch(
                    f"node '{nid}': missing value for input port '{port}' ({ref})"
                )
            in_arrays.append(env[key])
        kwargs = bound.get(nid, {})
        params = _resolve_params(node.params, symbol_env)
        try:
            outs = op.run(in_arrays, params, kwargs)
        except (ShapeMismatch, TensorInvalid):
            raise
        except Exception as exc:
            raise ShapeMismatch(
                f"node '{nid}' ({node.op}) failed: {exc}; "
                f"input shapes={[list(a.shape) for a in in_arrays]}"
            ) from exc
        if len(outs) != max(1, len(node.outputs)):
            # Node outputs may be unknown until validation ran; recompute types.
            in_types = [
                TensorType.of(env[_key(node.inputs[p])].shape, str(env[_key(node.inputs[p])].dtype))
                for p in port_order(op, node.inputs)
            ]
            node.outputs = tuple(op.infer_types(in_types, node.params, {}))
        for i, arr in enumerate(outs):
            arr = np.asarray(arr)
            if validate_types and i < len(node.outputs):
                declared = node.outputs[i]
                expected = None
                try:
                    expected = replace_symbols(declared.shape, symbol_env)
                except KeyError:
                    expected = None
                if expected is not None and tuple(expected) != tuple(arr.shape):
                    raise ShapeMismatch(
                        f"node '{nid}' ({node.op}) produced shape {arr.shape}, "
                        f"declared {declared.shape}"
                    )
            if check_finite and arr.dtype.kind in "fc" and not np.isfinite(arr).all():
                raise TensorInvalid(
                    f"node '{nid}' ({node.op}) produced non-finite values"
                )
            env[f"{nid}:{i}"] = arr
        if nid in captures:
            captured[nid] = np.array(outs[0], copy=True)

    outputs: dict[str, np.ndarray] = {}
    for name, ref in graph.outputs.items():
        key = _key(ref)
        if key not in env:
            raise ShapeMismatch(f"graph output '{name}' references missing value {ref}")
        outputs[name] = env[key]
    return ExecutionResult(
        outputs=outputs,
        intermediates=captured,
        order=order,
        num_ops=len(order),
    )


def _resolve_params(params: Mapping[str, Any], symbol_env: Mapping[str, int]) -> dict[str, Any]:
    """Resolve symbolic dimensions inside a node's ``shape`` parameter."""
    out = dict(params)
    shape = out.get("shape")
    if isinstance(shape, (list, tuple)):
        resolved: list[Any] = []
        for d in shape:
            if isinstance(d, str) and d in symbol_env:
                resolved.append(int(symbol_env[d]))
            else:
                resolved.append(d)
        out["shape"] = resolved
    return out


def _key(ref: Ref) -> str:
    if ref.node.startswith(GRAPH_INPUT):
        return ref.node
    return f"{ref.node}:{ref.port}"


def run_layers(
    graph: Graph, inputs: Mapping[str, np.ndarray], params: WeightPackage, *, capture: Sequence[str]
) -> dict[str, np.ndarray]:
    return execute(graph, inputs, params, capture=capture).intermediates


class GraphWalker:
    """Iterate nodes in topological order (used by inspection/analysis tools)."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.order = graph.topological_order()

    def __iter__(self) -> Iterator[tuple[int, str]]:
        return enumerate(self.order)

    def depth(self) -> dict[str, int]:
        depth: dict[str, int] = {}
        for nid in self.order:
            node = self.graph.nodes[nid]
            preds = [
                depth.get(r.node, 0)
                for r in node.inputs.values()
                if not r.node.startswith(GRAPH_INPUT)
            ]
            depth[nid] = (max(preds) + 1) if preds else 0
        return depth
