"""Typed computational graph for NIR with validation and serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from uzen.core.nir.ops import OPERATORS, get_operator, param_ref_names, port_order
from uzen.core.nir.types import Dim, Ref, TensorType
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import GraphValidationError, ShapeMismatch, UnsupportedOperator

GRAPH_SCHEMA = "uzen.nir/1"


@dataclass
class Node:
    id: str
    op: str
    inputs: dict[str, Ref] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    outputs: tuple[TensorType, ...] = ()
    note: str = ""

    def param_names(self) -> list[str]:
        out: list[str] = []
        for value in self.params.values():
            out.extend(param_ref_names(value))
        return sorted(set(out))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
            "params": self.params,
            "outputs": [t.to_dict() for t in self.outputs],
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Node":
        return cls(
            id=str(data["id"]),
            op=str(data["op"]),
            inputs={str(k): Ref.parse(str(v)) for k, v in (data.get("inputs") or {}).items()},
            params=dict(data.get("params") or {}),
            outputs=tuple(TensorType.from_dict(t) for t in (data.get("outputs") or [])),
            note=str(data.get("note", "")),
        )


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    num_nodes: int = 0
    num_params: int = 0
    param_names: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    inferred: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def require(self) -> None:
        if not self.ok:
            raise GraphValidationError(
                "NIR graph validation failed", errors=self.errors[:20]
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "num_nodes": self.num_nodes,
            "num_params": self.num_params,
            "param_names": self.param_names,
            "order": self.order,
            "inferred": self.inferred,
        }


class Graph:
    """A directed, typed computational graph.

    Nodes consume named input ports (each bound to an upstream ``Ref``) and emit
    one or more typed outputs. Graph-level inputs are declared explicitly, so
    cycles are detected rather than silently interpreted, and stateful recurrent
    structure lives inside operators (``rnn``/``lstm``/``gru``) rather than as
    implicit graph cycles.
    """

    def __init__(
        self,
        *,
        name: str = "graph",
        inputs: Mapping[str, TensorType] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.inputs: dict[str, TensorType] = dict(inputs or {})
        self.nodes: dict[str, Node] = {}
        self.outputs: dict[str, Ref] = {}
        self.metadata: dict[str, Any] = dict(metadata or {})

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    def add_input(self, name: str, type_: TensorType) -> Ref:
        if name in self.inputs:
            raise GraphValidationError(f"input '{name}' already declared")
        self.inputs[name] = type_
        return Ref(f"input:{name}", 0)

    def add_node(
        self,
        node_id: str,
        op: str,
        inputs: Mapping[str, Ref | str | tuple[str, int]] | None = None,
        params: Mapping[str, Any] | None = None,
        *,
        note: str = "",
    ) -> Ref:
        if node_id in self.nodes:
            raise GraphValidationError(f"duplicate node id '{node_id}'")
        if not node_id or ":" in node_id:
            raise GraphValidationError(f"invalid node id '{node_id}'")
        resolved_inputs: dict[str, Ref] = {}
        for port, ref in (inputs or {}).items():
            if isinstance(ref, Ref):
                resolved_inputs[str(port)] = ref
            elif isinstance(ref, tuple):
                resolved_inputs[str(port)] = Ref(str(ref[0]), int(ref[1]))
            else:
                resolved_inputs[str(port)] = Ref.parse(str(ref))
        node = Node(id=node_id, op=op, inputs=resolved_inputs, params=dict(params or {}), note=note)
        # Fail fast on unknown operators so synthesis cannot build unrunnable graphs.
        get_operator(op)
        self.nodes[node_id] = node
        return Ref(node_id, 0)

    def set_output(self, name: str, ref: Ref | str) -> None:
        self.outputs[name] = ref if isinstance(ref, Ref) else Ref.parse(ref)

    def connect(self, node_id: str, port: str, ref: Ref | str) -> None:
        if node_id not in self.nodes:
            raise GraphValidationError(f"unknown node '{node_id}'")
        self.nodes[node_id].inputs[port] = ref if isinstance(ref, Ref) else Ref.parse(ref)

    def copy(self) -> "Graph":
        g = Graph(name=self.name, inputs=dict(self.inputs), metadata=dict(self.metadata))
        for nid, node in self.nodes.items():
            g.nodes[nid] = Node(
                id=node.id,
                op=node.op,
                inputs=dict(node.inputs),
                params=dict(node.params),
                outputs=tuple(node.outputs),
                note=node.note,
            )
        g.outputs = dict(self.outputs)
        return g

    # ------------------------------------------------------------------ #
    # structure
    # ------------------------------------------------------------------ #

    def node(self, node_id: str) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise GraphValidationError(f"unknown node '{node_id}'") from exc

    def source_of(self, ref: Ref) -> tuple[str, str] | None:
        if ref.node.startswith("input:"):
            return ("input", ref.node[len("input:"):])
        if ref.node in self.nodes:
            return ("node", ref.node)
        return None

    def consumers(self, node_id: str) -> list[str]:
        return [n.id for n in self.nodes.values() if any(r.node == node_id for r in n.inputs.values())]

    def parameter_names(self) -> list[str]:
        names: set[str] = set()
        for node in self.nodes.values():
            names.update(node.param_names())
        return sorted(names)

    def topological_order(self) -> list[str]:
        """Deterministic Kahn topological sort; raises on cycles/unknown refs."""
        indegree: dict[str, int] = {nid: 0 for nid in self.nodes}
        dependents: dict[str, set[str]] = {nid: set() for nid in self.nodes}
        for nid, node in self.nodes.items():
            for ref in node.inputs.values():
                src = self.source_of(ref)
                if src is None:
                    raise GraphValidationError(
                        f"node '{nid}' references unknown value '{ref}'"
                    )
                kind, name = src
                if kind == "input":
                    if name not in self.inputs:
                        raise GraphValidationError(
                            f"node '{nid}' references undeclared graph input '{name}'"
                        )
                    continue
                if name == nid:
                    raise GraphValidationError(f"node '{nid}' depends on itself")
                if nid not in dependents[name]:
                    dependents[name].add(nid)
                    indegree[nid] += 1
        ready = sorted([nid for nid, deg in indegree.items() if deg == 0])
        order: list[str] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for child in sorted(dependents[nid]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready.sort()
        if len(order) != len(self.nodes):
            cyclic = sorted(set(self.nodes) - set(order))
            raise GraphValidationError(
                "computational graph contains a cycle; recurrent structure must live inside "
                "a recurrent operator (rnn/lstm/gru), not as a graph cycle",
                cyclic_nodes=cyclic,
            )
        return order

    # ------------------------------------------------------------------ #
    # typing / validation
    # ------------------------------------------------------------------ #

    def infer_types(
        self,
        order: Sequence[str] | None = None,
        *,
        bound: Mapping[str, TensorType | Sequence[Dim]] | None = None,
    ) -> tuple[dict[str, list[TensorType]], list[str]]:
        """Infer output types node by node; returns (types_by_node, errors).

        ``bound`` supplies parameter tensor types (a package manifest, or a family's
        ``parameter_shapes``) so shapes can be inferred for structural editing without
        loading weights.
        """
        order = list(order or self.topological_order())
        bound_types: dict[str, TensorType] = {
            name: value if isinstance(value, TensorType) else TensorType.of(tuple(value))
            for name, value in (bound or {}).items()
        }
        types: dict[str, list[TensorType]] = {}
        errors: list[str] = []

        def resolve(ref: Ref) -> TensorType | None:
            if ref.node.startswith("input:"):
                return self.inputs.get(ref.node[len("input:"):])
            outs = types.get(ref.node)
            if outs is None or ref.port >= len(outs):
                return None
            return outs[ref.port]

        for nid in order:
            node = self.nodes[nid]
            op = get_operator(node.op)
            in_types: list[TensorType] = []
            missing = False
            for port in port_order(op, node.inputs):
                t = resolve(node.inputs[port])
                if t is None:
                    errors.append(f"node '{nid}': unresolved input port '{port}'")
                    missing = True
                else:
                    in_types.append(t)
            if missing:
                continue
            if not (op.min_inputs <= len(in_types) <= op.max_inputs):
                errors.append(
                    f"node '{nid}' ({node.op}) got {len(in_types)} inputs, "
                    f"expected {op.min_inputs}..{op.max_inputs}"
                )
                continue
            try:
                outs = op.infer_types(in_types, node.params, bound_types)
            except (ShapeMismatch, Exception) as exc:  # shape inference must never crash validation
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                errors.append(f"node '{nid}' ({node.op}): {exc}")
                continue
            node.outputs = tuple(outs)
            types[nid] = list(outs)
        return types, errors

    def validate(self, package: WeightPackage | None = None) -> ValidationReport:
        report = ValidationReport(ok=False, num_nodes=len(self.nodes))
        try:
            order = self.topological_order()
        except GraphValidationError as exc:
            report.errors.append(str(exc))
            return report
        report.order = order

        bound_types: dict[str, TensorType] = {}
        if package is not None:
            for name, meta in package.metas.items():
                bound_types[name] = TensorType.of(meta.shape, meta.dtype)

        # Operator + parameter binding checks, then shape inference with bindings.
        for nid in order:
            node = self.nodes[nid]
            if node.op not in OPERATORS:
                report.errors.append(f"node '{nid}': operator '{node.op}' is unknown")
                continue
            for pname in node.param_names():
                if package is None:
                    report.warnings.append(
                        f"node '{nid}': parameter '{pname}' is not bound (no package supplied)"
                    )
                elif pname not in package:
                    report.errors.append(
                        f"node '{nid}': bound parameter '{pname}' missing from package '{package.name}'"
                    )
                elif not node.params:
                    report.errors.append(f"node '{nid}': parameter references without params")
            if not node.inputs:
                report.errors.append(f"node '{nid}' ({node.op}): no input ports connected")
            if node.op not in ("embedding",) and node.outputs and all(
                not t.shape for t in node.outputs
            ):
                report.warnings.append(f"node '{nid}': operator produced a scalar output")

        if report.errors:
            return report

        # Shape inference needs bound parameter shapes. Without a package we validate
        # structure only (known operators, refs, cycles, connected inputs) and record
        # that inference was skipped, rather than failing a legitimate graph edit.
        if package is None:
            for nid in order:
                for ref in self.nodes[nid].inputs.values():
                    if ref.node.startswith("input:"):
                        if ref.node[len("input:") :] not in self.inputs:
                            report.errors.append(
                                f"node '{nid}': undeclared graph input '{ref.node[len('input:'):]}'"
                            )
                    elif ref.node not in self.nodes:
                        report.errors.append(f"node '{nid}': unresolved ref '{ref}'")
            for out_name, ref in self.outputs.items():
                if ref.node.startswith("input:"):
                    if ref.node[len("input:") :] not in self.inputs:
                        report.errors.append(f"output '{out_name}' references unknown input")
                elif ref.node not in self.nodes:
                    report.errors.append(
                        f"output '{out_name}' references unknown node '{ref.node}'"
                    )
            report.param_names = self.parameter_names()
            report.num_params = len(report.param_names)
            report.warnings.append(
                "shape inference skipped: no parameter package supplied (bind a package to "
                "validate shapes and dtypes)"
            )
            report.ok = not report.errors
            return report

        # Shape inference with resolved parameter types.
        types: dict[str, list[TensorType]] = {}
        for nid in order:
            node = self.nodes[nid]
            op = get_operator(node.op)
            in_types: list[TensorType] = []
            failed = False
            for port in port_order(op, node.inputs):
                ref = node.inputs[port]
                if ref.node.startswith("input:"):
                    name = ref.node[len("input:"):]
                    if name not in self.inputs:
                        report.errors.append(f"node '{nid}': undeclared graph input '{name}'")
                        failed = True
                        break
                    in_types.append(self.inputs[name])
                else:
                    outs = types.get(ref.node)
                    if not outs or ref.port >= len(outs):
                        report.errors.append(f"node '{nid}': unresolved ref '{ref}'")
                        failed = True
                        break
                    in_types.append(outs[ref.port])
            if failed:
                continue
            try:
                outs = op.infer_types(in_types, node.params, bound_types)
            except Exception as exc:
                report.errors.append(f"node '{nid}' ({node.op}): {exc}")
                continue
            node.outputs = tuple(outs)
            types[nid] = list(outs)

        for out_name, ref in self.outputs.items():
            if ref.node.startswith("input:"):
                if ref.node[len("input:"):] not in self.inputs:
                    report.errors.append(f"output '{out_name}' references unknown input")
                continue
            if ref.node not in self.nodes:
                report.errors.append(f"output '{out_name}' references unknown node '{ref.node}'")
                continue
            outs = types.get(ref.node, [])
            if ref.port >= len(outs):
                report.errors.append(f"output '{out_name}' references missing port {ref.port}")

        report.param_names = self.parameter_names()
        report.num_params = len(report.param_names)
        report.inferred = {nid: [t.to_dict() for t in outs] for nid, outs in types.items()}
        if package is not None:
            missing = [p for p in report.param_names if p not in package]
            if missing:
                report.errors.append(f"package is missing parameters: {missing[:10]}")

        report.ok = not report.errors
        return report

    def require_valid(self, package: WeightPackage | None = None) -> ValidationReport:
        report = self.validate(package)
        report.require()
        return report

    # ------------------------------------------------------------------ #
    # structural editing (used by NASE)
    # ------------------------------------------------------------------ #

    def replace_operator(self, node_id: str, new_op: str, new_params: Mapping[str, Any] | None = None) -> "Graph":
        """Operator substitution with signature compatibility enforced by validation."""
        g = self.copy()
        node = g.node(node_id)
        node.op = new_op
        if new_params is not None:
            node.params = dict(new_params)
        node.outputs = ()
        return g

    def substitute_subgraph(self, mapping: Mapping[str, str]) -> "Graph":
        """Rewrite references from node ids to other node ids (graph composition)."""
        g = self.copy()
        for target in g.nodes.values():
            for port, ref in list(target.inputs.items()):
                if ref.node in mapping:
                    target.inputs[port] = Ref(mapping[ref.node], ref.port)
        for name, ref in list(g.outputs.items()):
            if ref.node in mapping:
                g.outputs[name] = Ref(mapping[ref.node], ref.port)
        return g

    def without_nodes(self, node_ids: Iterable[str]) -> "Graph":
        drop = set(node_ids)
        g = Graph(name=self.name, inputs=dict(self.inputs), metadata=dict(self.metadata))
        for nid, node in self.nodes.items():
            if nid not in drop:
                g.nodes[nid] = node
        g.outputs = dict(self.outputs)
        return g

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GRAPH_SCHEMA,
            "name": self.name,
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "nodes": [self.nodes[nid].to_dict() for nid in self.topological_order()]
            if self.nodes
            else [],
            "outputs": {k: str(v) for k, v in self.outputs.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Graph":
        g = cls(
            name=str(data.get("name", "graph")),
            inputs={k: TensorType.from_dict(v) for k, v in (data.get("inputs") or {}).items()},
            metadata=dict(data.get("metadata") or {}),
        )
        for node_data in data.get("nodes") or []:
            node = Node.from_dict(node_data)
            if node.op not in OPERATORS:
                raise UnsupportedOperator(f"serialized graph uses unknown operator '{node.op}'")
            g.nodes[node.id] = node
        g.outputs = {k: Ref.parse(str(v)) for k, v in (data.get("outputs") or {}).items()}
        return g

    def summary(self) -> dict[str, Any]:
        families: dict[str, int] = {}
        for node in self.nodes.values():
            try:
                fam = get_operator(node.op).family
            except Exception:
                fam = "unknown"
            families[fam] = families.get(fam, 0) + 1
        params = 0
        for node in self.nodes.values():
            for out in node.outputs:
                prod = 1
                for d in out.shape:
                    if not isinstance(d, int):
                        prod = 0
                        break
                    prod *= d
                params += prod
        return {
            "name": self.name,
            "num_nodes": len(self.nodes),
            "num_edges": sum(len(n.inputs) for n in self.nodes.values()),
            "operator_families": families,
            "num_parameters": len(self.parameter_names()),
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "outputs": [str(r) for r in self.outputs.values()],
        }
