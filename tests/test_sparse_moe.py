"""Sparse-dispatch MoE: equivalence with the dense reference, and true sparsity."""

from __future__ import annotations

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nir.ops import _ACTIVATIONS
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.tensors import WeightPackage


def _graph(shape, params):
    graph = Graph(name="moe_sparse_test")
    graph.add_input("x", TensorType.of(shape))
    ref = graph.add_node("moe0", "moe", {"x": Ref("input:x")}, params)
    graph.set_output("y", ref)
    return graph


def _reference(x, router, up, down, up_b, down_b, *, top_k, activation):
    x64 = x.astype(np.float64)
    logits = x64 @ router.T.astype(np.float64)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    selected = np.argpartition(-probs, top_k - 1, axis=-1)[..., :top_k]
    gates = np.take_along_axis(probs, selected, axis=-1)
    gates = gates / np.maximum(gates.sum(axis=-1, keepdims=True), 1e-12)
    out = np.zeros_like(x64)
    for expert in range(router.shape[0]):
        mask = selected == expert
        if not mask.any():
            continue
        gate = np.sum(gates * mask, axis=-1, keepdims=True)
        hidden = _ACTIVATIONS[activation](x64 @ up[expert].T + up_b[expert])
        out = out + (hidden @ down[expert].T + down_b[expert]) * gate
    return out


def test_sparse_moe_matches_dense_routing():
    rng = np.random.default_rng(7)
    batch, tokens, dim, experts, hidden = 3, 4, 8, 4, 6
    arrays = {
        "router": rng.normal(size=(experts, dim)).astype(np.float32),
        "up": rng.normal(size=(experts, hidden, dim)).astype(np.float32),
        "down": rng.normal(size=(experts, dim, hidden)).astype(np.float32),
        "up_b": rng.normal(size=(experts, hidden)).astype(np.float32),
        "down_b": rng.normal(size=(experts, dim)).astype(np.float32),
    }
    package = WeightPackage.from_arrays(arrays)
    params = {
        "router": "param:router",
        "up_weight": "param:up",
        "down_weight": "param:down",
        "up_bias": "param:up_b",
        "down_bias": "param:down_b",
        "top_k": 2,
        "activation": "gelu",
    }
    x = rng.normal(size=(batch, tokens, dim)).astype(np.float32)
    graph = _graph(["B", "T", dim], params)
    graph.require_valid(package)

    got = execute(graph, {"x": x}, package).outputs["y"]
    want = _reference(
        x,
        arrays["router"],
        arrays["up"],
        arrays["down"],
        arrays["up_b"],
        arrays["down_b"],
        top_k=2,
        activation="gelu",
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert np.abs(got).sum() > 0


def test_unrouted_experts_are_never_evaluated():
    """Experts that receive no tokens must not be touched at all."""
    dim, hidden, experts = 4, 4, 3
    router = np.zeros((experts, dim), dtype=np.float32)
    router[0, 0] = 50.0  # every token routes deterministically to expert 0
    router[1:, :] = -50.0
    up = np.full((experts, hidden, dim), np.nan, dtype=np.float32)
    down = np.full((experts, dim, hidden), np.nan, dtype=np.float32)
    up[0] = 0.1
    down[0] = 0.2
    up_b = np.zeros((experts, hidden), dtype=np.float32)
    down_b = np.zeros((experts, dim), dtype=np.float32)
    arrays = {"router": router, "up": up, "down": down, "up_b": up_b, "down_b": down_b}
    package = WeightPackage.from_arrays(arrays)
    x = np.ones((2, 3, dim), dtype=np.float32)

    graph = _graph(
        ["B", "T", dim],
        {
            "router": "param:router",
            "up_weight": "param:up",
            "down_weight": "param:down",
            "up_bias": "param:up_b",
            "down_bias": "param:down_b",
            "top_k": 1,
            "activation": "relu",
        },
    )
    graph.require_valid(package)
    result = execute(graph, {"x": x}, package).outputs["y"]

    # Dense dispatch would multiply by a zero gate and still evaluate the NaN experts.
    assert np.isfinite(result).all()
    assert np.abs(result).sum() > 0
