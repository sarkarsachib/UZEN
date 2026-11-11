"""GGUF v3 container adapter.

GGUF is a self-describing binary container: a versioned header, typed key/value
metadata, a tensor descriptor table, and an alignment-padded data section. This module
reads and writes that container itself.

Tensor types are described by the :mod:`ggml_quant` registry. F32 (0) and F16 (1) are
stored exactly; Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 are dequantized to float32 on read;
Q8_0 can be written on request. K-quant and IQ types are registered with their real
block geometry but have no codec, so they are inspectable and refused rather than
guessed.

Dimension convention: ggml stores dimensions fastest-varying first, so an array of
shape ``(a, b, c)`` is written with dims ``(c, b, a)`` and reconstructed by reversing
them. All tensor payloads are C-contiguous little-endian.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from uzen.errors import ArtifactIntegrityError, UnsupportedFormat
from uzen.infrastructure.storage.ggml_quant import (
    GGML_F16,
    GGML_F32,
    GGML_Q4_0,
    GGML_Q4_1,
    GGML_Q5_0,
    GGML_Q5_1,
    GGML_Q8_0,
    decoder_available,
    dequantize as dequantize_type,
    get_type,
    quantize as quantize_type,
)

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

_GGML_TO_DTYPE = {GGML_F32: np.dtype(np.float32), GGML_F16: np.dtype(np.float16)}
_DTYPE_TO_GGML = {np.dtype(np.float32): GGML_F32, np.dtype(np.float16): GGML_F16}

# gguf metadata value type ids (gguf spec v3)
_GV_UINT8, _GV_INT8, _GV_UINT16, _GV_INT16, _GV_UINT32, _GV_INT32 = 0, 1, 2, 3, 4, 5
_GV_FLOAT32, _GV_BOOL, _GV_STRING, _GV_ARRAY = 6, 7, 8, 9
_GV_UINT64, _GV_INT64, _GV_FLOAT64 = 10, 11, 12
_KV_STRUCT = {
    _GV_UINT8: "<B",
    _GV_INT8: "<b",
    _GV_UINT16: "<H",
    _GV_INT16: "<h",
    _GV_UINT32: "<I",
    _GV_INT32: "<i",
    _GV_FLOAT32: "<f",
    _GV_BOOL: "<?",
    _GV_UINT64: "<Q",
    _GV_INT64: "<q",
    _GV_FLOAT64: "<d",
}

_MAX_METADATA_ENTRIES = 100_000
_MAX_TENSOR_COUNT = 1_000_000
_MAX_STRING = 1 << 24
_MAX_DIMS = 8


class _Cursor:
    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            raise ArtifactIntegrityError("GGUF file is truncated")
        out = self.buf[self.pos : self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def u32(self) -> int:
        return int(self.unpack("<I"))

    def u64(self) -> int:
        return int(self.unpack("<Q"))

    def string(self) -> str:
        n = self.u64()
        if n > _MAX_STRING:
            raise ArtifactIntegrityError(
                f"GGUF string length {n} exceeds the {_MAX_STRING}-byte safety limit"
            )
        try:
            return self.take(n).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactIntegrityError("GGUF string is not valid UTF-8") from exc


def _read_value(cur: _Cursor, vtype: int):
    fmt = _KV_STRUCT.get(vtype)
    if fmt is not None:
        return cur.unpack(fmt)
    if vtype == _GV_STRING:
        return cur.string()
    if vtype == _GV_ARRAY:
        elem = cur.u32()
        count = cur.u64()
        if count > _MAX_METADATA_ENTRIES:
            raise ArtifactIntegrityError(
                f"GGUF metadata array length {count} exceeds the safety limit"
            )
        return [_read_value(cur, elem) for _ in range(count)]
    raise ArtifactIntegrityError(f"GGUF metadata value type {vtype} is not a known gguf type")


def dequantize_ggml(raw: bytes, ggml_type: int, element_count: int) -> np.ndarray:
    """Backwards-compatible wrapper over the GGML codec registry."""
    return dequantize_type(raw, ggml_type, element_count)


def read_gguf(path: str | os.PathLike[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load F32/F16 tensors and typed metadata from a GGUF v3 file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    raw = p.read_bytes()
    if not raw:
        raise ArtifactIntegrityError("GGUF file is empty")
    if raw[:4] != GGUF_MAGIC:
        raise ArtifactIntegrityError("not a GGUF file (bad magic)")
    cur = _Cursor(raw)
    cur.take(4)
    version = cur.u32()
    if version != GGUF_VERSION:
        raise UnsupportedFormat(
            f"GGUF version {version} is not supported; this adapter reads v3",
            path=str(p),
        )
    tensor_count = cur.u64()
    kv_count = cur.u64()
    if tensor_count > _MAX_TENSOR_COUNT or kv_count > _MAX_METADATA_ENTRIES:
        raise ArtifactIntegrityError("GGUF header declares an implausible number of entries")

    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = cur.string()
        metadata[key] = _read_value(cur, cur.u32())

    infos: list[tuple[str, tuple[int, ...], int, int]] = []
    for _ in range(tensor_count):
        name = cur.string()
        n_dims = cur.u32()
        if n_dims > _MAX_DIMS:
            raise ArtifactIntegrityError(f"GGUF tensor '{name}' declares {n_dims} dimensions")
        dims = tuple(cur.u64() for _ in range(n_dims))
        ggml_type = cur.u32()
        offset = cur.u64()
        infos.append((name, dims, ggml_type, offset))

    alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT) or DEFAULT_ALIGNMENT)
    if alignment <= 0 or alignment & (alignment - 1):
        raise ArtifactIntegrityError(f"GGUF alignment {alignment} is not a positive power of two")
    data_start = (cur.pos + alignment - 1) // alignment * alignment
    if data_start > len(raw):
        raise ArtifactIntegrityError("GGUF data section is missing")

    arrays: dict[str, np.ndarray] = {}
    for name, dims, ggml_type, offset in infos:
        shape = tuple(int(d) for d in reversed(dims))
        count = int(np.prod(shape)) if shape else 1
        start = data_start + int(offset)
        dtype = _GGML_TO_DTYPE.get(ggml_type)
        if dtype is not None:
            nbytes = count * dtype.itemsize
            end = start + nbytes
            if start < data_start or end > len(raw):
                raise ArtifactIntegrityError(f"GGUF tensor '{name}' payload is out of bounds")
            arrays[name] = np.frombuffer(raw[start:end], dtype=dtype).reshape(shape).copy()
            continue
        entry = get_type(ggml_type)
        if count % entry.block_size:
            raise UnsupportedFormat(
                f"GGUF tensor '{name}' has {count} elements, which is not a multiple of the "
                f"{entry.block_size}-element {entry.name} block",
                path=str(p),
                tensor=name,
            )
        nbytes = count // entry.block_size * entry.block_bytes
        end = start + nbytes
        if start < data_start or end > len(raw):
            raise ArtifactIntegrityError(f"GGUF tensor '{name}' payload is out of bounds")
        arrays[name] = dequantize_ggml(raw[start:end], ggml_type, count).reshape(shape).copy()
    return arrays, metadata


def inspect_gguf(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Report GGUF metadata and tensor descriptors without decoding payloads.

    This works for every registered type, including K-quant and IQ tensors that have no
    decoder, and is what inspection tools should use before attempting a full read.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    raw = p.read_bytes()
    if not raw:
        raise ArtifactIntegrityError("GGUF file is empty")
    if raw[:4] != GGUF_MAGIC:
        raise ArtifactIntegrityError("not a GGUF file (bad magic)")
    cur = _Cursor(raw)
    cur.take(4)
    version = cur.u32()
    if version != GGUF_VERSION:
        raise UnsupportedFormat(
            f"GGUF version {version} is not supported; this adapter reads v3", path=str(p)
        )
    tensor_count = cur.u64()
    kv_count = cur.u64()
    if tensor_count > _MAX_TENSOR_COUNT or kv_count > _MAX_METADATA_ENTRIES:
        raise ArtifactIntegrityError("GGUF header declares an implausible number of entries")

    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = cur.string()
        metadata[key] = _read_value(cur, cur.u32())

    infos: list[tuple[str, tuple[int, ...], int, int]] = []
    for _ in range(tensor_count):
        name = cur.string()
        n_dims = cur.u32()
        if n_dims > _MAX_DIMS:
            raise ArtifactIntegrityError(f"GGUF tensor '{name}' declares {n_dims} dimensions")
        dims = tuple(cur.u64() for _ in range(n_dims))
        ggml_type = cur.u32()
        offset = cur.u64()
        infos.append((name, dims, ggml_type, offset))

    alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT) or DEFAULT_ALIGNMENT)
    if alignment <= 0 or alignment & (alignment - 1):
        raise ArtifactIntegrityError(f"GGUF alignment {alignment} is not a positive power of two")
    data_start = (cur.pos + alignment - 1) // alignment * alignment
    if data_start > len(raw):
        raise ArtifactIntegrityError("GGUF data section is missing")

    tensors: list[dict[str, Any]] = []
    for name, dims, ggml_type, offset in infos:
        shape = tuple(int(d) for d in reversed(dims))
        count = int(np.prod(shape)) if shape else 1
        descriptor: dict[str, Any] = {
            "name": name,
            "shape": list(shape),
            "elements": count,
            "ggml_type": ggml_type,
            "offset": int(offset),
        }
        try:
            entry = get_type(ggml_type)
        except UnsupportedFormat:
            descriptor.update(
                {
                    "type_name": f"unknown_{ggml_type}",
                    "in_registry": False,
                    "decoder_available": False,
                    "block_size": None,
                    "block_bytes": None,
                    "byte_length": None,
                }
            )
        else:
            aligned = count % entry.block_size == 0
            descriptor.update(
                {
                    "type_name": entry.name,
                    "in_registry": True,
                    "decoder_available": decoder_available(ggml_type),
                    "block_size": entry.block_size,
                    "block_bytes": entry.block_bytes,
                    "byte_length": (
                        count // entry.block_size * entry.block_bytes if aligned else None
                    ),
                }
            )
        tensors.append(descriptor)
    return {
        "path": str(p),
        "file_bytes": len(raw),
        "metadata": metadata,
        "tensors": tensors,
        "data_start": data_start,
    }


def requantize_gguf(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    target: str = "q8_0",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode a GGUF file, re-encode it at ``target`` and report quantization error."""
    arrays, source_metadata = read_gguf(source)
    merged = dict(source_metadata)
    merged.update(dict(metadata or {}))
    path = write_gguf(destination, arrays, merged, quantize=target)
    restored, _ = read_gguf(path)

    report: dict[str, Any] = {
        "source": str(source),
        "destination": str(path),
        "target": target,
        "tensors": {},
    }
    for name, original in arrays.items():
        original64 = original.astype(np.float64)
        decoded64 = restored[name].astype(np.float64)
        difference = np.abs(decoded64 - original64)
        denominator = np.maximum(np.abs(original64), 1e-12)
        report["tensors"][name] = {
            "shape": list(original.shape),
            "max_abs_error": float(difference.max()) if difference.size else 0.0,
            "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
            "relative_max_error": float((difference / denominator).max())
            if difference.size
            else 0.0,
        }
    return report


def _encode_value(value: Any) -> tuple[int, bytes]:
    if isinstance(value, bool):
        return _GV_BOOL, struct.pack("<?", value)
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return _GV_STRING, struct.pack("<Q", len(encoded)) + encoded
    if isinstance(value, (int, np.integer)):
        number = int(value)
        if -(2**31) <= number < 2**31:
            return _GV_INT32, struct.pack("<i", number)
        return _GV_INT64, struct.pack("<q", number)
    if isinstance(value, (float, np.floating)):
        return _GV_FLOAT32, struct.pack("<f", float(value))
    if isinstance(value, (list, tuple)):
        if not value:
            return _GV_ARRAY, struct.pack("<IQ", _GV_STRING, 0)
        if all(isinstance(v, str) for v in value):
            elem = _GV_STRING
            payload = b"".join(_encode_value(v)[1] for v in value)
        elif all(isinstance(v, bool) for v in value):
            elem = _GV_BOOL
            payload = b"".join(struct.pack("<?", v) for v in value)
        elif all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in value):
            elem = _GV_INT32
            payload = b"".join(struct.pack("<i", int(v)) for v in value)
        elif all(isinstance(v, (float, np.floating)) for v in value):
            elem = _GV_FLOAT32
            payload = b"".join(struct.pack("<f", float(v)) for v in value)
        else:
            raise UnsupportedFormat(
                "GGUF metadata arrays must be homogeneous str/int/float/bool",
                value_type=type(value).__name__,
            )
        return _GV_ARRAY, struct.pack("<IQ", elem, len(value)) + payload
    raise UnsupportedFormat(
        f"cannot encode metadata value of type {type(value).__name__} into GGUF"
    )


def _quantize_q8_0(arr: np.ndarray) -> bytes:
    return quantize_type(np.asarray(arr), GGML_Q8_0)


def write_gguf(
    path: str | os.PathLike[str],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any] | None = None,
    *,
    alignment: int = DEFAULT_ALIGNMENT,
    quantize: str | None = None,
) -> Path:
    """Write a GGUF v3 file.

    ``quantize=None`` stores float32/float16 tensors exactly. ``quantize="q8_0"``
    stores block-quantized Q8_0 tensors (lossy; read back as float32).
    """
    if alignment <= 0 or alignment & (alignment - 1):
        raise UnsupportedFormat("GGUF alignment must be a positive power of two")
    quantize_name = (quantize or "").strip().lower() or None
    if quantize_name not in (None, "q8_0"):
        raise UnsupportedFormat(f"unknown GGUF quantization '{quantize}'; supported: q8_0")

    entries: list[tuple[str, tuple[int, ...], int, bytes]] = []
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        if quantize_name == "q8_0":
            entries.append(
                (
                    str(name),
                    tuple(int(d) for d in reversed(arr.shape)),
                    GGML_Q8_0,
                    _quantize_q8_0(arr),
                )
            )
            continue
        ggml_type = _DTYPE_TO_GGML.get(arr.dtype)
        if ggml_type is None:
            raise UnsupportedFormat(
                f"GGUF export supports float32 and float16 tensors; '{name}' is {arr.dtype}",
                tensor=str(name),
                dtype=str(arr.dtype),
            )
        entries.append(
            (str(name), tuple(int(d) for d in reversed(arr.shape)), ggml_type, arr.tobytes(order="C"))
        )

    meta: dict[str, Any] = {"general.architecture": "uzen"}
    meta.update(dict(metadata or {}))
    meta["general.alignment"] = int(alignment)

    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(entries))
    header += struct.pack("<Q", len(meta))
    for key in sorted(meta):
        value_type, payload = _encode_value(meta[key])
        key_bytes = key.encode("utf-8")
        header += struct.pack("<Q", len(key_bytes)) + key_bytes
        header += struct.pack("<I", value_type) + payload

    infos = bytearray()
    data = bytearray()
    for name, dims, ggml_type, payload in entries:
        data += b"\x00" * ((-len(data)) % alignment)
        offset = len(data)
        name_bytes = name.encode("utf-8")
        infos += struct.pack("<Q", len(name_bytes)) + name_bytes
        infos += struct.pack("<I", len(dims))
        infos += b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<I", ggml_type)
        infos += struct.pack("<Q", offset)
        data += payload

    prefix = bytes(header + infos)
    prefix += b"\x00" * ((-len(prefix)) % alignment)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(prefix + bytes(data))
    os.replace(temporary, destination)
    return destination
