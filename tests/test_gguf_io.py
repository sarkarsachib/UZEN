"""GGUF v3 adapter: real binary round-trips and pipeline integration."""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from uzen.core.nws.families import build_architecture
from uzen.core.nws.store import ArtifactStore
from uzen.core.wie.inspector import WeightIntelligenceEngine
from uzen.errors import ArtifactIntegrityError, UnsupportedFormat
from uzen.infrastructure.storage.ggml_quant import (
    GGML_Q4_K,
    GGML_Q5_0,
    GGML_Q5_1,
    GGML_Q6_K,
    decode_q4_k,
    decode_q6_k,
    get_type,
    type_info,
)
from uzen.infrastructure.storage.gguf_io import (
    GGML_Q4_0,
    dequantize_ggml,
    inspect_gguf,
    read_gguf,
    requantize_gguf,
    write_gguf,
)
from uzen.jobs.service import JobService


def test_gguf_roundtrip_preserves_f32_tensors_and_metadata(tmp_path):
    rng = np.random.default_rng(0)
    arrays = {
        "embed.weight": rng.standard_normal((8, 4)).astype(np.float32),
        "layer.weight": rng.standard_normal((3, 4, 2)).astype(np.float32),
        "norm.weight": np.ones(4, dtype=np.float32),
    }
    path = write_gguf(
        tmp_path / "model.gguf",
        arrays,
        {
            "general.architecture": "mlp",
            "uzen.note": "roundtrip",
            "uzen.tags": ["a", "b"],
            "uzen.verified": True,
            "uzen.layers": 2,
            "uzen.scale": 0.5,
        },
    )
    assert path.read_bytes()[:4] == b"GGUF"

    back, metadata = read_gguf(path)
    assert set(back) == set(arrays)
    for name, arr in arrays.items():
        assert back[name].dtype == np.float32
        np.testing.assert_array_equal(back[name], arr)
    assert metadata["general.architecture"] == "mlp"
    assert metadata["uzen.tags"] == ["a", "b"]
    assert metadata["uzen.verified"] is True
    assert metadata["uzen.layers"] == 2
    assert metadata["uzen.scale"] == pytest.approx(0.5)
    assert metadata["general.alignment"] == 32


def test_gguf_f16_roundtrip_with_custom_alignment(tmp_path):
    values = np.linspace(-2.0, 2.0, 12, dtype=np.float16).reshape(3, 4)
    path = write_gguf(
        tmp_path / "f16.gguf", {"w": values}, {"general.architecture": "cnn"}, alignment=64
    )
    back, metadata = read_gguf(path)
    assert back["w"].dtype == np.float16
    np.testing.assert_array_equal(back["w"], values)
    assert metadata["general.alignment"] == 64


def test_gguf_rejects_bad_magic(tmp_path):
    path = tmp_path / "broken.gguf"
    path.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ArtifactIntegrityError):
        read_gguf(path)


def test_gguf_refuses_dtypes_it_cannot_represent(tmp_path):
    with pytest.raises(UnsupportedFormat):
        write_gguf(tmp_path / "bad.gguf", {"w": np.zeros((2, 2), dtype=np.float64)})


def test_wie_detects_gguf_and_the_service_imports_it(tmp_path):
    arrays = {"a.weight": np.arange(12, dtype=np.float32).reshape(3, 4)}
    path = write_gguf(tmp_path / "tiny.gguf", arrays, {"general.architecture": "mlp"})

    report = WeightIntelligenceEngine().inspect_file(path)
    assert report.format == "gguf"
    assert report.ok is True

    service = JobService(tmp_path / "ws")
    record = service.import_artifact(path, family="mlp")
    assert record.num_tensors == 1
    loaded = service.get_package(record.artifact_id)
    np.testing.assert_array_equal(loaded.tensors["a.weight"], arrays["a.weight"])


def test_artifact_store_exports_gguf(tmp_path):
    definition = build_architecture("mlp", dict(input_dim=4, hidden_dims=[8], output_dim=2))
    package = definition.build_package(seed=0)
    store = ArtifactStore(tmp_path / "artifacts")
    record = store.put(package, operation="init")

    out = store.export(record.artifact_id, tmp_path / "exported.gguf", fmt="gguf")
    back, metadata = read_gguf(out)
    assert metadata["uzen.artifact_id"] == record.artifact_id
    for name in package.names():
        np.testing.assert_array_equal(back[name], package.tensors[name])


def test_dequantize_q4_0_uses_the_ggml_nibble_layout():
    qs = np.arange(16, dtype=np.uint8)
    raw = np.float16(0.5).tobytes() + qs.tobytes()
    got = dequantize_ggml(raw, GGML_Q4_0, 32)
    low = (qs & 0x0F).astype(np.float32)
    high = (qs >> 4).astype(np.float32)
    expected = np.concatenate([(low - 8.0) * 0.5, (high - 8.0) * 0.5])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-7)


def test_q8_0_export_roundtrips_within_quantization_error(tmp_path):
    rng = np.random.default_rng(5)
    values = rng.standard_normal((8, 4)).astype(np.float32)
    path = write_gguf(tmp_path / "q8.gguf", {"w": values}, quantize="q8_0")

    back, metadata = read_gguf(path)
    assert back["w"].shape == values.shape
    assert back["w"].dtype == np.float32
    tolerance = float(np.abs(values).max()) / 127.0 * 0.51 + 1e-6
    assert np.abs(back["w"] - values).max() <= tolerance
    assert metadata["general.alignment"] == 32


def test_q8_0_export_requires_a_block_multiple(tmp_path):
    with pytest.raises(UnsupportedFormat):
        write_gguf(tmp_path / "bad.gguf", {"w": np.zeros((3, 5), np.float32)}, quantize="q8_0")


def test_unknown_quantized_type_is_refused_with_its_code():
    with pytest.raises(UnsupportedFormat) as exc:
        dequantize_ggml(b"\x00" * 64, 12, 32)
    assert "12" in str(exc.value)


def test_dequantize_q5_0_reconstructs_the_fifth_bit():
    # d=1.0; qh sets bit 0 (low half) and bit 16 (high half); qs[0] has low nibble 1.
    raw = (
        np.float16(1.0).tobytes()
        + np.uint32(0x00010001).tobytes()
        + bytes([0x01] + [0x00] * 15)
    )
    got = dequantize_ggml(raw, GGML_Q5_0, 32)
    assert got[0] == pytest.approx(1.0)
    assert got[16] == pytest.approx(0.0)
    assert got[1] == pytest.approx(-16.0)


def test_dequantize_q5_1_adds_the_block_minimum():
    raw = (
        np.float16(1.0).tobytes()
        + np.float16(0.5).tobytes()
        + np.uint32(0x00010001).tobytes()
        + bytes([0x01] + [0x00] * 15)
    )
    got = dequantize_ggml(raw, GGML_Q5_1, 32)
    assert got[0] == pytest.approx(17.5)
    assert got[16] == pytest.approx(16.5)


def test_registry_reports_geometry_and_codec_availability():
    q2k = type_info(10)
    assert (q2k["name"], q2k["block_size"], q2k["block_bytes"]) == ("q2_k", 256, 84)
    assert q2k["effective_bits"] == pytest.approx(2.625)
    assert q2k["decoder_available"] is False

    q4k = type_info(12)
    assert (q4k["name"], q4k["block_size"], q4k["block_bytes"]) == ("q4_k", 256, 144)
    assert q4k["effective_bits"] == pytest.approx(4.5)
    assert q4k["decoder_available"] is True
    assert type_info(14)["decoder_available"] is True

    q5 = type_info(6)
    assert q5["name"] == "q5_0"
    assert q5["decoder_available"] is True
    assert get_type(8).name == "q8_0"


def _synthetic_gguf(tmp_path, dims, type_id, payload):
    header = bytearray(b"GGUF")
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", 1) + struct.pack("<Q", 0)
    name = b"w"
    header += struct.pack("<Q", len(name)) + name
    header += struct.pack("<I", len(dims))
    header += b"".join(struct.pack("<Q", d) for d in dims)
    header += struct.pack("<I", type_id) + struct.pack("<Q", 0)
    header += b"\x00" * ((-len(header)) % 32)
    path = tmp_path / f"type{type_id}.gguf"
    path.write_bytes(bytes(header) + payload)
    return path


def test_inspect_reports_metadata_only_k_quant_tensors(tmp_path):
    path = _synthetic_gguf(tmp_path, (256,), 10, b"\x00" * 84)
    report = inspect_gguf(path)
    tensor = report["tensors"][0]
    assert tensor["type_name"] == "q2_k"
    assert tensor["byte_length"] == 84
    assert tensor["decoder_available"] is False
    assert tensor["in_registry"] is True

    with pytest.raises(UnsupportedFormat) as exc:
        read_gguf(path)
    assert "q2_k" in str(exc.value)


def test_requantize_reports_quantization_error(tmp_path):
    values = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    source = write_gguf(tmp_path / "source.gguf", {"w": values})

    report = requantize_gguf(source, tmp_path / "q8.gguf", target="q8_0")
    assert report["target"] == "q8_0"
    stats = report["tensors"]["w"]
    assert stats["max_abs_error"] >= 0.0
    assert stats["max_abs_error"] <= 0.02
    back, _ = read_gguf(tmp_path / "q8.gguf")
    assert back["w"].shape == values.shape


def _q4_k_block(rng) -> bytes:
    return (
        np.float16(0.5).tobytes()
        + np.float16(0.25).tobytes()
        + rng.integers(0, 256, 12, dtype=np.uint8).tobytes()
        + rng.integers(0, 256, 128, dtype=np.uint8).tobytes()
    )


def _q6_k_block(rng) -> bytes:
    return (
        rng.integers(0, 256, 128, dtype=np.uint8).tobytes()
        + rng.integers(0, 256, 64, dtype=np.uint8).tobytes()
        + rng.integers(-128, 128, 16, dtype=np.int8).tobytes()
        + np.float16(0.5).tobytes()
    )


REFERENCE_C = Path(__file__).parent / "reference" / "ggml_dequant_ref.c"


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc is unavailable")
def test_q4_k_and_q6_k_match_compiled_upstream_reference(tmp_path):
    binary = tmp_path / "ggml_dequant_ref"
    subprocess.run(
        ["gcc", "-O2", "-std=c11", str(REFERENCE_C), "-o", str(binary)],
        check=True,
    )
    rng = np.random.default_rng(11)
    cases = (
        ("q4k", _q4_k_block(rng), decode_q4_k, GGML_Q4_K),
        ("q6k", _q6_k_block(rng), decode_q6_k, GGML_Q6_K),
    )
    for kind, raw, decoder, type_id in cases:
        block_path = tmp_path / f"{kind}.bin"
        block_path.write_bytes(raw)
        result = subprocess.run(
            [str(binary), kind, str(block_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        expected = np.array([float(x) for x in result.stdout.split()], dtype=np.float32)
        np.testing.assert_allclose(decoder(raw, 256), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(dequantize_ggml(raw, type_id, 256), expected, rtol=1e-6, atol=1e-6)


def test_read_gguf_dequantizes_q4_k_and_q6_k(tmp_path):
    rng = np.random.default_rng(12)
    q4 = _q4_k_block(rng)
    q6 = _q6_k_block(rng)
    q4_path = _synthetic_gguf(tmp_path, (256,), GGML_Q4_K, q4)
    q6_path = _synthetic_gguf(tmp_path, (256,), GGML_Q6_K, q6)

    q4_arrays, _ = read_gguf(q4_path)
    q6_arrays, _ = read_gguf(q6_path)
    assert q4_arrays["w"].shape == (256,)
    assert q6_arrays["w"].shape == (256,)
    np.testing.assert_allclose(q4_arrays["w"], decode_q4_k(q4, 256))
    np.testing.assert_allclose(q6_arrays["w"], decode_q6_k(q6, 256))
