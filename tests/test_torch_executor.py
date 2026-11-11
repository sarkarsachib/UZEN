"""Torch-CPU NIR execution: parity with the NumPy reference across every family."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uzen.core.nir.executor import execute  # noqa: E402
from uzen.core.nws.families import build_architecture  # noqa: E402
from uzen.errors import InvalidOperationParameters  # noqa: E402
from uzen.runtimes.pytorch_runtime import PyTorchRuntime  # noqa: E402
from uzen.runtimes.torch_executor import execute_torch, is_available  # noqa: E402

FAMILY_SPECS = {
    "transformer_decoder": dict(
        d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=24, max_seq_len=12
    ),
    "mlp": dict(input_dim=8, hidden_dims=[16, 12], output_dim=4),
    "cnn": dict(in_channels=1, height=8, width=8, channels=[4], num_classes=3),
    "recurrent": dict(cell="gru", vocab_size=10, d_model=4, hidden_size=8, num_classes=3),
    "autoencoder": dict(input_dim=10, latent_dim=3),
    "moe": dict(vocab_size=16, d_model=8, n_experts=4, top_k=2, d_ff=16, num_classes=3),
    "state_space": dict(vocab_size=16, d_model=8, state_dim=4, num_classes=3),
    "gnn": dict(in_features=5, hidden_dim=6, num_classes=3, n_layers=2),
    "vae": dict(input_dim=6, latent_dim=3, hidden_dim=8),
    "diffusion": dict(input_dim=6, hidden_dim=12, alpha_bar=0.6, noise_seed=4),
    "multimodal": dict(
        in_channels=1, height=8, width=8, vocab_size=12, d_model=6, num_classes=3
    ),
    "hybrid": dict(
        in_channels=1, height=8, width=8, channels=[3], hidden_size=5, num_classes=3
    ),
}


@pytest.mark.parametrize("family,spec", FAMILY_SPECS.items())
def test_torch_cpu_matches_numpy_for_every_family(family, spec):
    definition = build_architecture(family, spec)
    package = definition.build_package(seed=7)
    inputs = definition.sample_inputs(batch=2, seq=5, seed=7)
    expected = execute(definition.graph, inputs, package).outputs

    got = execute_torch(definition.graph, inputs, package)
    assert got.num_ops == len(definition.graph.nodes)
    assert set(got.outputs) == set(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(
            got.outputs[name], value, rtol=1e-5, atol=1e-6, err_msg=f"{family}:{name}"
        )


def test_runtime_prefers_torch_and_can_force_the_reference():
    definition = build_architecture("mlp", dict(input_dim=4, hidden_dims=[6], output_dim=2))
    package = definition.build_package(seed=1)
    inputs = definition.sample_inputs(batch=2, seed=1)
    runtime = PyTorchRuntime(device="cpu")

    torch_outputs = runtime.execute(definition.graph, inputs, package)
    assert runtime.last_backend == "torch"
    assert runtime.last_fallback_reason is None

    numpy_outputs = runtime.execute(definition.graph, inputs, package, backend="numpy")
    assert runtime.last_backend == "numpy"
    for name in torch_outputs:
        np.testing.assert_allclose(torch_outputs[name], numpy_outputs[name], rtol=1e-6, atol=1e-7)


def test_torch_backend_can_be_requested_explicitly():
    definition = build_architecture("transformer_decoder", FAMILY_SPECS["transformer_decoder"])
    package = definition.build_package(seed=2)
    inputs = definition.sample_inputs(batch=2, seq=5, seed=2)
    runtime = PyTorchRuntime(device="cpu")

    got = runtime.execute(definition.graph, inputs, package, backend="torch")
    assert runtime.last_backend == "torch"
    expected = execute(definition.graph, inputs, package).outputs
    for name in expected:
        np.testing.assert_allclose(got[name], expected[name], rtol=1e-5, atol=1e-6)


def test_capture_returns_intermediates():
    definition = build_architecture("mlp", dict(input_dim=4, hidden_dims=[6], output_dim=2))
    package = definition.build_package(seed=3)
    inputs = definition.sample_inputs(batch=3, seed=3)
    first = definition.graph.topological_order()[0]

    result = execute_torch(definition.graph, inputs, package, capture=[first])
    assert set(result.intermediates) == {first}
    assert result.intermediates[first].shape[0] == 3


def test_invalid_backend_is_rejected():
    definition = build_architecture("mlp", dict(input_dim=3, hidden_dims=[4], output_dim=2))
    package = definition.build_package(seed=4)
    inputs = definition.sample_inputs(batch=2, seed=4)
    with pytest.raises(InvalidOperationParameters):
        PyTorchRuntime(device="cpu").execute(
            definition.graph, inputs, package, backend="metal"
        )


def test_executor_reports_availability():
    assert is_available() is True
