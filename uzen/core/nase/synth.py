"""Neural Architecture Synthesis Engine (NASE).

Real graph construction and surgery over NIR:

* **synthesis** from a family spec (transformer / mlp / cnn / recurrent / autoencoder),
* **composition** of architectures (sequential, parallel, residual),
* **operator substitution** with signature re-validation,
* **node bypass** (drop a sub-block and reconnect consumers),
* **layer truncation** (drop trailing blocks and rewire),
* **validation** against a parameter package.

Every mutation returns a new graph plus an explicit report of what changed. Weight
shapes are never silently regenerated: a structural change is reported together with
the parameters it invalidates, so weight engineering and architecture engineering
stay separate concerns.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from uzen.core.nir.graph import Graph, Node
from uzen.core.nir.ops import get_operator
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.families import ArchitectureDefinition, build_architecture, family_catalogue
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import GraphValidationError, InvalidOperationParameters, ShapeMismatch


@dataclass
class SynthesisReport:
    definition: ArchitectureDefinition
    changes: list[str] = field(default_factory=list)
    invalidated_parameters: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.definition.family,
            "spec": self.definition.spec,
            "changes": self.changes,
            "invalidated_parameters": self.invalidated_parameters,
            "warnings": self.warnings,
            "summary": self.definition.summary(),
        }


def _prefix_graph(graph: Graph, prefix: str) -> Graph:
    g = Graph(
        name=f"{prefix}{graph.name}",
        inputs={f"{prefix}{k}": v for k, v in graph.inputs.items()},
        metadata={**graph.metadata, "prefixed_with": prefix},
    )
    for nid, node in graph.nodes.items():
        new_inputs = {}
        for port, ref in node.inputs.items():
            if ref.node.startswith("input:"):
                new_inputs[port] = Ref("input:" + prefix + ref.node[len("input:"):], ref.port)
            else:
                new_inputs[port] = Ref(prefix + ref.node, ref.port)
        params = {}
        for key, value in node.params.items():
            params[key] = _prefix_param_value(value, prefix)
        g.nodes[prefix + nid] = Node(
            id=prefix + nid,
            op=node.op,
            inputs=new_inputs,
            params=params,
            outputs=tuple(node.outputs),
            note=node.note,
        )
    g.outputs = {name: Ref(prefix + ref.node, ref.port) for name, ref in graph.outputs.items()}
    return g


def _prefix_param_value(value: Any, prefix: str) -> Any:
    if isinstance(value, str) and value.startswith("param:"):
        return "param:" + prefix + value[len("param:"):]
    if isinstance(value, list):
        return [_prefix_param_value(v, prefix) for v in value]
    return value


def _rewire_input(graph: Graph, input_name: str, ref: Ref) -> None:
    target = f"input:{input_name}"
    for node in graph.nodes.values():
        for port, existing in list(node.inputs.items()):
            if existing.node == target:
                node.inputs[port] = ref


def _merge_graphs(graphs: Sequence[Graph], name: str, metadata: Mapping[str, Any] | None = None) -> Graph:
    merged = Graph(name=name, inputs={}, metadata=dict(metadata or {}))
    for g in graphs:
        merged.inputs.update(g.inputs)
        for nid, node in g.nodes.items():
            merged.nodes[nid] = copy.deepcopy(node)
        merged.outputs.update(g.outputs)
    return merged


def _concrete_shape_differs(a: TensorType, b: TensorType) -> bool:
    if len(a.shape) != len(b.shape):
        return True
    for da, db in zip(a.shape, b.shape):
        if isinstance(da, int) and isinstance(db, int) and da != db:
            return True
    return False


def _definition_output_type(
    definition: ArchitectureDefinition, name: str | None = None
) -> TensorType | None:
    """Infer one declared output type from the family's parameter shapes.

    Composition needs to know whether consecutive stages agree before any weights
exist. Inference failures return ``None``; the full validation performed once a
package is bound still rejects genuine mismatches.
    """
    try:
        bound = {
            pname: TensorType.of(tuple(shape))
            for pname, shape in definition.parameter_shapes.items()
        }
        types, errors = definition.graph.copy().infer_types(bound=bound)
        if errors:
            return None
        outlets = definition.graph.outputs
        ref = outlets.get(name) if name else next(iter(outlets.values()), None)
        if ref is None:
            return None
        if ref.node.startswith("input:"):
            return definition.graph.inputs.get(ref.node[len("input:") :])
        outs = types.get(ref.node) or []
        return outs[ref.port] if ref.port < len(outs) else None
    except Exception:
        return None


def compose_definitions(
    definitions: Sequence[ArchitectureDefinition],
    *,
    mode: str = "sequential",
    name: str | None = None,
) -> ArchitectureDefinition:
    """Compose two or more architectures into one graph."""
    if not definitions:
        raise InvalidOperationParameters("composition requires at least one architecture")
    if mode not in ("sequential", "parallel", "residual"):
        raise InvalidOperationParameters("mode must be sequential|parallel|residual")
    prefixes = [f"c{i}_" for i in range(len(definitions))]
    graphs = [_prefix_graph(d.graph, p) for d, p in zip(definitions, prefixes)]
    params: dict[str, tuple[int, ...]] = {}
    for d, p in zip(definitions, prefixes):
        params.update({f"{p}{k}": v for k, v in d.parameter_shapes.items()})

    if mode == "parallel":
        merged = _merge_graphs(graphs, name or "parallel_composition")
        spec = {"mode": mode, "components": [d.family for d in definitions]}
        definition = ArchitectureDefinition(
            family="composed",
            spec=spec,
            graph=merged,
            parameter_shapes=params,
            default_inputs={},
            input_dtypes={},
        )
        return definition

    merged = _merge_graphs(graphs, name or f"{mode}_composition")
    changes: list[str] = []
    prev_out_ref: Ref | None = None
    for index in range(len(graphs)):
        if index == 0:
            prev_out_ref = list(graphs[0].outputs.values())[0]
            continue
        g = graphs[index]
        # feed the previous stage's first output into this stage's first input
        stage_input = next(iter(g.inputs))
        # ``merged`` holds deep copies of the stage nodes, so the rewire must target it
        # rather than the per-stage graph it was copied from.
        _rewire_input(merged, stage_input, prev_out_ref)
        # ``stage_input`` is prefixed (c1_x); the definition's own input is unprefixed.
        original_input = (
            stage_input[len(prefixes[index]) :]
            if stage_input.startswith(prefixes[index])
            else stage_input
        )
        prev_type = _definition_output_type(definitions[index - 1])
        stage_input_type = definitions[index].graph.inputs.get(original_input)
        if (
            prev_type is not None
            and stage_input_type is not None
            and _concrete_shape_differs(prev_type, stage_input_type)
        ):
            changes.append(
                f"shape adapter implicitly required: {tuple(prev_type.shape)} -> "
                f"{tuple(stage_input_type.shape)} (validation will reject mismatches)"
            )
        # The consumed stage input is no longer a graph input, so callers must not
        # supply it (the executor rejects unexpected inputs).
        merged.inputs.pop(stage_input, None)
        first_out = list(g.outputs.values())[0]
        if mode == "residual":
            add_id = f"c{index}_residual_add"
            merged.add_node(add_id, "add", {"a": prev_out_ref, "b": first_out})
            prev_out_ref = Ref(add_id, 0)
        else:
            prev_out_ref = first_out

    assert prev_out_ref is not None
    merged.outputs = {"out": prev_out_ref}
    merged.metadata["composition"] = mode
    spec = {
        "mode": mode,
        "components": [d.family for d in definitions],
        "component_specs": [d.spec for d in definitions],
        "composition_changes": changes,
    }
    return ArchitectureDefinition(
        family="composed",
        spec=spec,
        graph=merged,
        parameter_shapes=params,
        default_inputs={},
        input_dtypes={},
    )


class ArchitectureSynthesizer:
    """NASE facade."""

    # ------------------------------------------------------------------ #

    def synthesize(self, spec: Mapping[str, Any]) -> ArchitectureDefinition:
        family = str(spec.get("family", ""))
        if not family:
            raise InvalidOperationParameters("architecture spec requires a 'family'")
        clean = {k: v for k, v in spec.items() if k != "family"}
        return build_architecture(family, clean)

    def compose(
        self,
        definitions: Sequence[ArchitectureDefinition],
        *,
        mode: str = "sequential",
        name: str | None = None,
    ) -> ArchitectureDefinition:
        return compose_definitions(definitions, mode=mode, name=name)

    def substitute(
        self,
        definition: ArchitectureDefinition,
        node_id: str,
        new_op: str,
        params: Mapping[str, Any] | None = None,
    ) -> SynthesisReport:
        get_operator(new_op)  # fail fast on unknown operators
        before = definition.parameter_shapes.copy()
        graph = definition.graph.replace_operator(node_id, new_op, params)
        validation = graph.validate()
        if not validation.ok:
            raise GraphValidationError(
                f"operator substitution '{node_id}' -> '{new_op}' produced an invalid graph",
                errors=validation.errors[:10],
            )
        # Signature check: infer the whole graph with the family's parameter shapes so
        # an operator whose arity/ports/dtypes do not fit the substitution site is
        # rejected here rather than at execution time.
        bound = {
            pname: TensorType.of(tuple(shape))
            for pname, shape in definition.parameter_shapes.items()
        }
        _types, type_errors = graph.copy().infer_types(bound=bound)
        if type_errors:
            raise GraphValidationError(
                f"operator substitution '{node_id}' -> '{new_op}' is not type-compatible",
                errors=type_errors[:10],
            )
        after_params = set(graph.parameter_names())
        invalidated = sorted(set(before) - after_params)
        new_definition = ArchitectureDefinition(
            family=definition.family,
            spec={**definition.spec, "operator_substitutions": {
                **dict(definition.spec.get("operator_substitutions", {})),
                node_id: new_op,
            }},
            graph=graph,
            parameter_shapes={k: v for k, v in before.items() if k in after_params},
            default_inputs=dict(definition.default_inputs),
            input_dtypes=dict(definition.input_dtypes),
        )
        return SynthesisReport(
            definition=new_definition,
            changes=[f"replaced operator of node '{node_id}' with '{new_op}'"],
            invalidated_parameters=invalidated,
            warnings=(
                [f"node '{node_id}' parameter references changed; re-bind before execution"]
                if invalidated
                else []
            ),
        )

    def bypass(
        self,
        definition: ArchitectureDefinition,
        node_id: str,
        *,
        port: str = "a",
    ) -> SynthesisReport:
        graph = definition.graph
        if node_id not in graph.nodes:
            raise GraphValidationError(f"unknown node '{node_id}'")
        node = graph.nodes[node_id]
        if port not in node.inputs:
            raise GraphValidationError(
                f"node '{node_id}' has no input port '{port}' (ports: {sorted(node.inputs)})"
            )
        replacement = node.inputs[port]
        new_graph = graph.copy()
        for other in new_graph.nodes.values():
            for p, ref in list(other.inputs.items()):
                if ref.node == node_id:
                    other.inputs[p] = replacement
        for name, ref in list(new_graph.outputs.items()):
            if ref.node == node_id:
                new_graph.outputs[name] = replacement
        del new_graph.nodes[node_id]
        validation = new_graph.validate()
        if not validation.ok:
            raise GraphValidationError(
                f"bypassing '{node_id}' left an invalid graph", errors=validation.errors[:10]
            )
        invalidated = sorted(set(definition.graph.parameter_names()) - set(new_graph.parameter_names()))
        return SynthesisReport(
            definition=ArchitectureDefinition(
                family=definition.family,
                spec={**definition.spec, "bypassed_nodes": list(definition.spec.get("bypassed_nodes", [])) + [node_id]},
                graph=new_graph,
                parameter_shapes={k: v for k, v in definition.parameter_shapes.items() if k not in invalidated},
                default_inputs=dict(definition.default_inputs),
                input_dtypes=dict(definition.input_dtypes),
            ),
            changes=[f"bypassed node '{node_id}' through port '{port}' ({replacement})"],
            invalidated_parameters=invalidated,
            warnings=[f"parameters no longer used: {invalidated}"] if invalidated else [],
        )

    def truncate_layers(
        self,
        definition: ArchitectureDefinition,
        keep_layers: int,
        *,
        layer_prefix: str = "l",
    ) -> SynthesisReport:
        graph = definition.graph
        total_layers = int(definition.spec.get("n_layers", 0))
        if not total_layers:
            raise InvalidOperationParameters(
                f"truncation is only supported for layered families (got '{definition.family}')"
            )
        if keep_layers < 1 or keep_layers > total_layers:
            raise InvalidOperationParameters(
                f"keep_layers must be in [1, {total_layers}], got {keep_layers}"
            )
        if keep_layers == total_layers:
            return SynthesisReport(definition=definition, changes=["no truncation needed"])
        new_graph = graph.copy()
        removed_params: set[str] = set()
        removed_nodes: list[str] = []
        for layer in range(keep_layers, total_layers):
            prefix = f"{layer_prefix}{layer}_"
            for nid in [n for n in new_graph.nodes if n.startswith(prefix)]:
                removed_params.update(new_graph.nodes[nid].param_names())
                del new_graph.nodes[nid]
                removed_nodes.append(nid)
        # rewire input: the residual output of the last kept layer.
        last_kept = f"{layer_prefix}{keep_layers - 1}_res2"
        if last_kept not in new_graph.nodes:
            raise GraphValidationError(
                f"cannot locate the residual output of the last kept layer ('{last_kept}')"
            )
        replacement = Ref(last_kept, 0)
        for node in new_graph.nodes.values():
            for port, ref in list(node.inputs.items()):
                if ref.node.startswith(f"{layer_prefix}{keep_layers}_") or ref.node.startswith(
                    f"{layer_prefix}{keep_layers + 1}_"
                ):
                    node.inputs[port] = replacement
        validation = new_graph.validate()
        if not validation.ok:
            raise GraphValidationError(
                "layer truncation produced an invalid graph", errors=validation.errors[:10]
            )
        spec = {**definition.spec, "n_layers": keep_layers, "original_n_layers": total_layers}
        invalidated = sorted(removed_params)
        return SynthesisReport(
            definition=ArchitectureDefinition(
                family=definition.family,
                spec=spec,
                graph=new_graph,
                parameter_shapes={k: v for k, v in definition.parameter_shapes.items() if k not in set(invalidated)},
                default_inputs=dict(definition.default_inputs),
                input_dtypes=dict(definition.input_dtypes),
            ),
            changes=[
                f"removed layers {list(range(keep_layers, total_layers))} ({len(removed_nodes)} nodes)",
                f"rewired consumers to '{last_kept}'",
            ],
            invalidated_parameters=invalidated,
            warnings=[
                "architecture changed: regenerate or transplant parameters for the surviving layers",
                "positional embeddings and head shapes are unchanged",
            ],
        )

    def validate(
        self,
        definition: ArchitectureDefinition,
        package: WeightPackage | None = None,
    ) -> dict[str, Any]:
        report = definition.graph.validate(package)
        return {
            "ok": report.ok,
            "errors": report.errors,
            "warnings": report.warnings,
            "num_nodes": report.num_nodes,
            "parameters": report.param_names,
            "inferred": report.inferred,
        }

    def catalogue(self) -> list[dict[str, Any]]:
        return family_catalogue()

    def graph_of(self, definition: ArchitectureDefinition) -> dict[str, Any]:
        return definition.graph.to_dict()

    def from_graph(self, graph: Graph, *, parameter_shapes: Mapping[str, Sequence[int]] | None = None, family: str = "custom") -> ArchitectureDefinition:
        shapes = {name: tuple(int(d) for d in shape) for name, shape in (parameter_shapes or {}).items()}
        if not shapes:
            shapes = {name: (1,) for name in graph.parameter_names()}
        return ArchitectureDefinition(
            family=family,
            spec={"synthesized": True},
            graph=graph,
            parameter_shapes=shapes,
            default_inputs={k: tuple(v.shape) for k, v in graph.inputs.items()},
            input_dtypes={k: v.dtype for k, v in graph.inputs.items()},
        )
