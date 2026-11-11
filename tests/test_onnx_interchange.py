"""ONNX interchange: real export, official validation, and import back to NIR."""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

from uzen.core.nir.executor import execute  # noqa: E402
from uzen.core.nws.families import build_architecture  # noqa: E402
from uzen.errors import UnsupportedOperator  # noqa: E402
from uzen.runtimes.onnx_runtime import export_onnx, import_onnx, is_available  # noqa: E402


def _mlp():
    return build_architecture("mlp", dict(input_dim=5, hidden_dims=[7], output_dim=3))


def test_export_passes_the_official_checker_and_executes_identically(tmp_path):
    definition = _mlp()
    package = definition.build_package(seed=0)
    inputs = definition.sample_inputs(batch=2, seed=0)
    expected = execute(definition.graph, inputs, package).outputs

    path = tmp_path / "mlp.onnx"
    model = export_onnx(definition.graph, package, save_path=path)
    onnx.checker.check_model(model)

    loaded = onnx.load(str(path))
    assert {i.name for i in loaded.graph.input} == set(inputs)
    assert len(loaded.graph.output) == len(definition.graph.outputs)
    assert all(o.name for o in loaded.graph.output)

    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
    if ort is not None:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        values = session.run(None, {name: arr for name, arr in inputs.items()})
        expected_in_order = [expected[name] for name in definition.graph.outputs]
        assert len(values) == len(expected_in_order)
        for got, want in zip(values, expected_in_order):
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_import_lifts_the_exported_model_back_into_valid_nir(tmp_path):
    definition = _mlp()
    package = definition.build_package(seed=0)
    inputs = definition.sample_inputs(batch=2, seed=0)
    expected = execute(definition.graph, inputs, package).outputs
    path = tmp_path / "roundtrip.onnx"
    export_onnx(definition.graph, package, save_path=path)

    graph, weights, metadata = import_onnx(path)
    graph.require_valid(weights)
    restored = execute(graph, inputs, weights).outputs

    expected_values = [expected[name] for name in definition.graph.outputs]
    restored_values = [restored[name] for name in graph.outputs]
    assert len(restored_values) == len(expected_values)
    for got, want in zip(restored_values, expected_values):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert metadata["num_nodes"] >= 4
    assert metadata["num_initializers"] >= 1


def test_export_refuses_operators_it_cannot_map_faithfully():
    definition = build_architecture(
        "transformer_decoder",
        dict(d_model=8, n_heads=2, n_layers=1, d_ff=16, vocab_size=12, max_seq_len=6),
    )
    with pytest.raises(UnsupportedOperator) as exc:
        export_onnx(definition.graph, definition.build_package(seed=0))
    assert "does not implement NIR operator" in str(exc.value)


def test_import_reports_unsupported_onnx_operators_instead_of_dropping_them():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    node = helper.make_node("Sin", ["x"], ["y"], name="sine")
    graph = helper.make_graph([node], "unsupported", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    with pytest.raises(UnsupportedOperator) as exc:
        import_onnx(model)
    assert "Sin" in str(exc.value)


def test_adapter_reports_availability():
    assert is_available() is True
