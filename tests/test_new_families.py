"""Non-transformer architecture families and their NIR kernels.

Every assertion here executes the real NumPy kernels: no mocked forward passes and no
tolerance so loose that a wrong implementation would pass. The numeric tests compare
each new operator against an independently written reference computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nir.ops import OPERATORS, _gelu
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.families import build_architecture, family_catalogue
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import InvalidOperationParameters, ShapeMismatch

FAMILY_SPECS = {
    "moe": dict(vocab_size=16, d_model=8, n_experts=4, top_k=2, d_ff=16, num_classes=3),
    "state_space": dict(vocab_size=16, d_model=8, state_dim=4, num_classes=3),
    "gnn": dict(in_features=5, hidden_dim=6, num_classes=3, n_layers=2),
    "vae": dict(input_dim=6, latent_dim=3, hidden_dim=8),
    "diffusion": dict(input_dim=6, hidden_dim=12, alpha_bar=0.6, noise_seed=4),
    "multimodal": dict(in_channels=1, height=8, width=8, vocab_size=12, d_model=6, num_classes=3),
    "hybrid": dict(in_channels=1, height=8, width=8, channels=[3], hidden_size=5, num_classes=3),
}


@pytest.mark.parametrize("family,spec", FAMILY_SPECS.items())
def test_family_builds_validates_and_runs(family, spec):
    definition = build_architecture(family, spec)
    package = definition.build_package(seed=0)
    definition.graph.require_valid(package)
    inputs = definition.sample_inputs(batch=2, seq=6, seed=0)
    result = execute(definition.graph, inputs, package)
    assert result.num_ops == len(definition.graph.nodes)
    assert result.outputs
    for name, arr in result.outputs.items():
        assert np.isfinite(arr).all(), name
        assert arr.shape[0] == 2, (name, arr.shape)


def test_catalogue_lists_the_new_families():
    names = {entry["family"] for entry in family_catalogue()}
    assert set(FAMILY_SPECS) <= names


def test_new_kernels_are_registered_as_implemented():
    for name in ("select", "moe", "ssm", "gcn", "sample_normal", "add_noise"):
        assert OPERATORS[name].maturity.value == "implemented"


def test_moe_rejects_invalid_top_k():
    with pytest.raises(InvalidOperationParameters):
        build_architecture(
            "moe",
            dict(vocab_size=8, d_model=4, n_experts=2, top_k=3, num_classes=2),
        )


def test_select_is_conditional_execution():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((2, 3)).astype(np.float32)
    cond = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    g = Graph(name="select", inputs={"c": TensorType.of((2, 3)), "a": TensorType.of((2, 3)), "b": TensorType.of((2, 3))})
    g.add_node("sel", "select", {"a": Ref("input:a"), "b": Ref("input:b"), "cond": Ref("input:c")})
    g.set_output("y", Ref("sel"))
    package = WeightPackage.from_arrays({})
    out = execute(g, {"c": cond, "a": a, "b": b}, package).outputs["y"]
    np.testing.assert_array_equal(out, np.where(cond != 0, a, b))


def test_ssm_matches_explicit_recurrence():
    rng = np.random.default_rng(1)
    a = (-np.exp(rng.uniform(np.log(1e-2), np.log(0.5), size=(3, 2)))).astype(np.float32)
    b = rng.standard_normal((3, 2)).astype(np.float32)
    c = rng.standard_normal((3, 2)).astype(np.float32)
    d = rng.standard_normal(3).astype(np.float32)
    x = rng.standard_normal((2, 4, 3)).astype(np.float32)
    package = WeightPackage.from_arrays({"A": a, "B": b, "C": c, "D": d})
    g = Graph(name="ssm", inputs={"x": TensorType.of((2, 4, 3))})
    g.add_node(
        "ssm",
        "ssm",
        {"x": Ref("input:x")},
        {"A": "param:A", "B": "param:B", "C": "param:C", "D": "param:D"},
    )
    g.set_output("y", Ref("ssm"))
    got = execute(g, {"x": x}, package).outputs["y"]

    state = np.zeros((2, 3, 2), dtype=np.float64)
    expected = []
    for t in range(4):
        state = a[None].astype(np.float64) * state + b[None].astype(np.float64) * x[:, t, :, None]
        expected.append(np.einsum("bcn,cn->bc", state, c.astype(np.float64)) + d[None] * x[:, t])
    np.testing.assert_allclose(got, np.stack(expected, axis=1), rtol=1e-5, atol=1e-6)


def test_gcn_matches_manual_normalized_aggregation():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, 3, 2)).astype(np.float32)
    adj = np.array([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]], dtype=np.float32)
    w = rng.standard_normal((4, 2)).astype(np.float32)
    bias = rng.standard_normal(4).astype(np.float32)
    package = WeightPackage.from_arrays({"w": w, "b": bias})
    g = Graph(name="gcn", inputs={"x": TensorType.of((1, 3, 2)), "adj": TensorType.of((1, 3, 3))})
    g.add_node(
        "gcn",
        "gcn",
        {"x": Ref("input:x"), "adj": Ref("input:adj")},
        {"weight": "param:w", "bias": "param:b"},
    )
    g.set_output("y", Ref("gcn"))
    got = execute(g, {"x": x, "adj": adj}, package).outputs["y"]

    a_hat = adj.astype(np.float64) + np.eye(3, dtype=np.float64)[None]
    deg = a_hat.sum(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(deg)
    a_norm = a_hat * inv * np.swapaxes(inv, -1, -2)
    expected = np.einsum("bij,bjf->bif", a_norm, x.astype(np.float64)) @ w.T.astype(np.float64) + bias
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_moe_top1_matches_the_selected_expert():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 3)).astype(np.float32)
    router = rng.standard_normal((2, 3)).astype(np.float32)
    up = rng.standard_normal((2, 4, 3)).astype(np.float32)
    up_b = rng.standard_normal((2, 4)).astype(np.float32)
    down = rng.standard_normal((2, 3, 4)).astype(np.float32)
    down_b = rng.standard_normal((2, 3)).astype(np.float32)
    package = WeightPackage.from_arrays(
        {"router": router, "up": up, "up_b": up_b, "down": down, "down_b": down_b}
    )
    g = Graph(name="moe", inputs={"x": TensorType.of((2, 3, 3))})
    g.add_node(
        "moe",
        "moe",
        {"x": Ref("input:x")},
        {
            "router": "param:router",
            "up_weight": "param:up",
            "up_bias": "param:up_b",
            "down_weight": "param:down",
            "down_bias": "param:down_b",
            "top_k": 1,
            "activation": "gelu",
        },
    )
    g.set_output("y", Ref("moe"))
    got = execute(g, {"x": x}, package).outputs["y"]

    logits = x.astype(np.float64) @ router.T.astype(np.float64)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    selected = probs.argmax(axis=-1)
    expected = np.zeros_like(x, dtype=np.float64)
    for e in range(2):
        hidden = _gelu(x.astype(np.float64) @ up[e].T.astype(np.float64) + up_b[e])
        y_e = hidden @ down[e].T.astype(np.float64) + down_b[e]
        expected += np.where((selected == e)[..., None], y_e, 0.0)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_vae_sampling_is_seeded_and_seed_sensitive():
    definition = build_architecture("vae", dict(input_dim=6, latent_dim=3, hidden_dim=8, sample_seed=0))
    package = definition.build_package(seed=0)
    inputs = definition.sample_inputs(batch=2, seed=0)
    first = execute(definition.graph, inputs, package).outputs["latent"]
    second = execute(definition.graph, inputs, package).outputs["latent"]
    np.testing.assert_array_equal(first, second)

    resampled = definition.graph.copy()
    resampled.nodes["z"].params["seed"] = 7
    other = execute(resampled, inputs, package).outputs["latent"]
    assert not np.allclose(first, other)


def test_diffusion_noising_matches_the_closed_form():
    definition = build_architecture(
        "diffusion", dict(input_dim=4, hidden_dim=6, alpha_bar=0.3, noise_seed=9)
    )
    package = definition.build_package(seed=0)
    inputs = definition.sample_inputs(batch=2, seq=5, seed=0)
    out = execute(definition.graph, inputs, package).outputs["x_t"]
    eps = np.random.default_rng(9).standard_normal(inputs["x"].shape)
    expected = np.sqrt(0.3) * inputs["x"].astype(np.float64) + np.sqrt(0.7) * eps
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_sample_normal_rejects_mismatched_shapes():
    g = Graph(name="bad", inputs={"mu": TensorType.of((2, 3)), "logvar": TensorType.of((2, 4))})
    g.add_node("z", "sample_normal", {"mu": Ref("input:mu"), "logvar": Ref("input:logvar")}, {"seed": 0})
    g.set_output("z", Ref("z"))
    with pytest.raises(ShapeMismatch):
        execute(g, {"mu": np.zeros((2, 3)), "logvar": np.zeros((2, 4))}, WeightPackage.from_arrays({}))
