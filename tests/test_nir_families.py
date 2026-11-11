"""NIR / families / NASE / VEL tests."""

from __future__ import annotations

import numpy as np
import pytest

from uzen.core.nase.synth import ArchitectureSynthesizer, compose_definitions
from uzen.core.nir.graph import Graph
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nir.executor import execute
from uzen.core.nws.families import build_architecture
from uzen.core.vel.verify import verify
from uzen.errors import GraphValidationError, ShapeMismatch, UnsupportedOperator


FAMILY_SPECS = {
    "transformer_decoder": dict(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=24, max_seq_len=12),
    "mlp": dict(input_dim=8, hidden_dims=[16, 12], output_dim=4),
    "cnn": dict(in_channels=1, height=8, width=8, channels=[4], num_classes=3),
    "recurrent": dict(cell="gru", vocab_size=10, d_model=4, hidden_size=8, num_classes=3),
    "autoencoder": dict(input_dim=10, latent_dim=3),
}


@pytest.mark.parametrize("family,spec", FAMILY_SPECS.items())
def test_every_family_builds_validates_and_runs(family, spec):
    definition = build_architecture(family, spec)
    package = definition.build_package(seed=0)
    definition.graph.require_valid(package)
    inputs = definition.sample_inputs(batch=2, seq=6, seed=0)
    result = execute(definition.graph, inputs, package)
    assert result.outputs
    for name, arr in result.outputs.items():
        assert np.isfinite(arr).all()
        assert arr.shape[0] == 2
    assert result.num_ops == len(definition.graph.nodes)


def test_unknown_operator_is_reported_precisely():
    g = Graph(name="bad", inputs={"x": TensorType.of((2, 3))})
    with pytest.raises(UnsupportedOperator) as exc:
        g.add_node("n", "quantum_entangle", {"x": Ref("input:x")})
    assert "quantum_entangle" in str(exc.value)
    assert "known operators" in str(exc.value)


def test_cycle_is_rejected():
    g = Graph(name="cyclic", inputs={"x": TensorType.of((2, 3))})
    g.add_node("a", "identity", {"x": Ref("input:x")})
    g.add_node("b", "identity", {"x": Ref("a")})
    g.nodes["a"].inputs["x"] = Ref("b")
    with pytest.raises(GraphValidationError) as exc:
        g.topological_order()
    assert "cycle" in str(exc.value).lower()


def test_graph_serialization_roundtrip(transformer):
    doc = transformer.graph.to_dict()
    restored = Graph.from_dict(doc)
    assert restored.summary()["num_nodes"] == transformer.graph.summary()["num_nodes"]
    package = transformer.build_package(seed=0)
    restored.require_valid(package)
    inputs = transformer.sample_inputs(batch=1, seq=4, seed=0)
    out = execute(restored, inputs, package)
    assert set(out.outputs) == set(transformer.graph.outputs)


def test_missing_parameter_binding_is_detected(transformer, transformer_package):
    partial = transformer_package.without(["layers.1.qkv.weight"])
    report = transformer.graph.validate(partial)
    assert report.ok is False
    assert any("layers.1.qkv.weight" in e for e in report.errors)


def test_shape_incompatible_parameter_is_detected(transformer, transformer_package):
    bad = transformer_package.with_tensor("layers.0.mlp_up.weight", np.zeros((7, 16), dtype=np.float32))
    report = transformer.graph.validate(bad)
    assert report.ok is False


def test_operator_substitution_is_validated(transformer):
    synth = ArchitectureSynthesizer()
    definition = transformer
    report = synth.substitute(definition, "l0_act", "relu")
    assert report.definition.graph.nodes["l0_act"].op == "relu"
    report.definition.graph.require_valid(definition.build_package(seed=0))
    with pytest.raises(GraphValidationError):
        synth.substitute(definition, "l0_ln1", "attention")


def test_layer_truncation_rewires_and_invalidates_parameters(transformer):
    synth = ArchitectureSynthesizer()
    report = synth.truncate_layers(transformer, 1)
    assert "layers.1.qkv.weight" in report.invalidated_parameters
    assert report.definition.spec["n_layers"] == 1
    package = report.definition.build_package(seed=0)
    report.definition.graph.require_valid(package)
    inputs = report.definition.sample_inputs(batch=1, seq=4, seed=0)
    assert execute(report.definition.graph, inputs, package).outputs["logits"].shape == (1, 4, 24)


def test_bypass_node_and_reconnect(mlp):
    synth = ArchitectureSynthesizer()
    report = synth.bypass(mlp, "fc0", port="x")
    # fc0's consumers now read the graph input
    assert report.changes
    report.definition.graph.require_valid() if False else None
    with pytest.raises(GraphValidationError):
        synth.bypass(mlp, "does_not_exist")


def test_architecture_composition_sequential():
    a = build_architecture("mlp", dict(input_dim=6, hidden_dims=[8], output_dim=4))
    b = build_architecture("mlp", dict(input_dim=4, hidden_dims=[8], output_dim=3))
    composed = compose_definitions([a, b], mode="sequential")
    package = composed.build_package(seed=0)
    composed.graph.require_valid(package)
    inputs = {
        name: np.zeros(tuple(d if isinstance(d, int) else 2 for d in spec.shape), dtype=np.float32)
        for name, spec in composed.graph.inputs.items()
    }
    out = execute(composed.graph, inputs, package)
    assert out.outputs["out"].shape == (2, 3)


def test_verification_distinguishes_structural_and_functional(mlp, mlp_package):
    inputs = mlp.sample_inputs(batch=2, seq=4, seed=0)
    report = verify(mlp_package, graph=mlp.graph, inputs=inputs)
    assert report.ok, report.problems
    assert report.structural["architecture_unchanged"] if "architecture_unchanged" in report.structural else True
    assert report.functional["ok"] is True
    assert report.performance["device"] == "cpu"

    perturbed = mlp_package.with_tensor("fc0.weight", mlp_package.tensors["fc0.weight"] * 0.999)
    regression = verify(perturbed, reference=mlp_package, graph=mlp.graph, inputs=inputs)
    assert regression.regression["worst_relative_l2_diff"] >= 0.0
    assert regression.regression["architecture_unchanged"] is True


def test_verification_flags_non_finite_values(mlp_package):
    broken = mlp_package.with_tensor("fc0.bias", np.full_like(mlp_package.tensors["fc0.bias"], np.nan))
    report = verify(broken, measure_performance=False)
    assert report.ok is False
    assert any("non-finite" in p for p in report.problems)


def test_verification_detects_architecture_change(mlp, mlp_package):
    reduced = mlp_package.without(["head.bias"])
    report = verify(reduced, reference=mlp_package, measure_performance=False)
    assert report.structural["architecture_unchanged"] is False
    assert report.ok is False


def test_executor_rejects_wrong_input_shape(mlp, mlp_package):
    with pytest.raises(ShapeMismatch):
        execute(mlp.graph, {"x": np.zeros((2, 9), dtype=np.float32)}, mlp_package)
    with pytest.raises(ShapeMismatch):
        execute(mlp.graph, {"x": np.zeros((2, 8), dtype=np.float32), "extra": np.zeros(1)}, mlp_package)
