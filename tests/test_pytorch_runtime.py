"""PyTorch runtime adapter, exercised against the CPU-only build."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uzen.core.nir.executor import execute as numpy_execute  # noqa: E402
from uzen.core.nws.families import build_architecture  # noqa: E402
from uzen.errors import DTypeMismatch, RuntimeUnavailable, UnsupportedFormat  # noqa: E402
from uzen.runtimes.pytorch_runtime import PyTorchRuntime, torch_available  # noqa: E402


@pytest.fixture()
def runtime() -> PyTorchRuntime:
    return PyTorchRuntime(device="cpu")


def test_cpu_capabilities_are_honest(runtime):
    capabilities = runtime.capabilities()
    assert capabilities.backend == "pytorch"
    assert capabilities.device == "cpu"
    assert capabilities.supports_autograd is False
    assert capabilities.supports_torch_checkpoints is True
    assert capabilities.extra["available"] is True
    # This environment uses the CPU-only wheel: no CUDA runtime to claim.
    assert torch.version.cuda is None
    assert capabilities.extra["cuda_available"] is False
    assert capabilities.extra["cuda_device_count"] == 0
    assert capabilities.to_dict()["strict_plus_compatible"] is True


def test_numpy_tensor_roundtrip(runtime):
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    tensor = runtime.numpy_to_tensor(array)
    assert tensor.device.type == "cpu"
    assert tensor.requires_grad is False
    np.testing.assert_array_equal(runtime.tensor_to_numpy(tensor), array)


def test_non_tensor_inputs_are_rejected(runtime):
    with pytest.raises(DTypeMismatch):
        runtime.tensor_to_numpy([1, 2, 3])


def test_grad_tracked_tensors_are_refused(runtime):
    tensor = torch.ones(2, requires_grad=True)
    with pytest.raises(RuntimeUnavailable):
        runtime.tensor_to_numpy(tensor)


def test_bf16_state_dict_is_widened_and_restored(runtime):
    source = torch.tensor([[1.5, -2.0], [0.25, 3.0]], dtype=torch.bfloat16)
    state = {
        "w": source,
        "b": torch.tensor([0, 1], dtype=torch.int64),
        "flag": torch.tensor([True, False]),
    }
    package = runtime.from_state_dict(state)

    assert package.tensors["w"].dtype == np.float32
    assert package.metas["w"].source_dtype == "bfloat16"
    assert "bfloat16" in package.metas["w"].note
    np.testing.assert_array_equal(package.tensors["w"], source.float().numpy())
    np.testing.assert_array_equal(package.tensors["b"], np.array([0, 1], dtype=np.int64))
    assert package.tensors["flag"].dtype == np.bool_

    restored = runtime.to_state_dict(package)
    assert restored["w"].dtype is torch.bfloat16
    np.testing.assert_array_equal(restored["w"].float().numpy(), package.tensors["w"])


def test_pt_checkpoint_roundtrip_preserves_tensors(tmp_path, runtime):
    definition = build_architecture("mlp", dict(input_dim=4, hidden_dims=[6], output_dim=2))
    package = definition.build_package(seed=3)
    path = tmp_path / "weights.pt"
    runtime.save(package, path)

    loaded = runtime.load(path)
    assert set(loaded.names()) == set(package.names())
    for name in package.names():
        np.testing.assert_array_equal(loaded.tensors[name], package.tensors[name])


def test_safetensors_roundtrip_uses_the_codec(tmp_path, runtime):
    definition = build_architecture("mlp", dict(input_dim=3, hidden_dims=[4], output_dim=2))
    package = definition.build_package(seed=4)
    path = tmp_path / "weights.safetensors"
    runtime.save(package, path)

    loaded = runtime.load(path)
    assert set(loaded.names()) == set(package.names())
    for name in package.names():
        np.testing.assert_array_equal(loaded.tensors[name], package.tensors[name])


class _NotAStateDict:
    pass


def test_arbitrary_pickled_objects_are_refused(tmp_path, runtime):
    path = tmp_path / "bad.pt"
    torch.save(_NotAStateDict(), str(path))
    with pytest.raises(UnsupportedFormat):
        runtime.load(path)


def test_cuda_is_refused_when_unavailable():
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA; the CPU guard is not applicable")
    with pytest.raises(RuntimeUnavailable):
        PyTorchRuntime(device="cuda")


def test_forward_matches_numpy_runtime_and_tracks_no_gradients(runtime):
    definition = build_architecture("mlp", dict(input_dim=5, hidden_dims=[7], output_dim=3))
    package = definition.build_package(seed=1)
    inputs = definition.sample_inputs(batch=2, seed=1)
    expected = numpy_execute(definition.graph, inputs, package).outputs

    got = runtime.execute(definition.graph, inputs, package)
    assert set(got) == set(expected)
    for name in expected:
        np.testing.assert_allclose(got[name], expected[name], rtol=1e-6, atol=1e-7)
    assert all(not getattr(value, "requires_grad", False) for value in package.tensors.values())


def test_availability_flag_matches_the_environment():
    assert torch_available() is True
