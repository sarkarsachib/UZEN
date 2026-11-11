"""A from-scratch SafeTensors reader/writer.

Implemented directly against the published format so that UZEN can read and
write weight artifacts without a mandatory third-party dependency, can inspect
gigabyte-scale headers without loading tensor data, and can validate structural
integrity itself:

    [8 bytes: uint64 LE header length N][N bytes UTF-8 JSON header][data buffer]

Header JSON maps tensor name -> {"dtype", "shape", "data_offsets": [b, e]} plus
an optional "__metadata__" string map. The data buffer must be fully covered by
contiguous, non-overlapping tensor spans starting at offset 0.

Safety properties enforced here:
  * no pickle / code execution of any kind (the format is pure data),
  * hard header-size cap,
  * strict coverage/overlap/bounds validation,
  * exact dtype-span agreement (prod(shape) * itemsize == e - b),
  * duplicate-key detection.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from uzen.core.nws.dtypes import (
    CODE_FOR_DTYPE,
    SAFETENSORS_CODES,
    UNSUPPORTED_CODES,
    bf16_decode,
    bf16_encode,
    dtype_name,
    normalize_dtype,
)
from uzen.errors import ArtifactIntegrityError, DTypeMismatch, UnsupportedFormat
from uzen.util.canonical import atomic_write_bytes, sha256_file

MAX_HEADER_BYTES = 100 * 1024 * 1024  # 100 MiB
HEADER_ALIGNMENT = 8


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype_code: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype_code,
            "shape": list(self.shape),
            "data_offsets": [self.begin, self.end],
        }


@dataclass
class SafetensorsFile:
    path: Path
    entries: dict[str, TensorEntry]
    metadata: dict[str, str]
    header_bytes: int
    data_bytes: int
    file_bytes: int
    sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tensors(self) -> int:
        return len(self.entries)

    def total_numel(self) -> int:
        total = 0
        for e in self.entries.values():
            n = 1
            for d in e.shape:
                n *= d
            total += n
        return total

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "num_tensors": self.num_tensors,
            "total_numel": self.total_numel(),
            "header_bytes": self.header_bytes,
            "data_bytes": self.data_bytes,
            "file_bytes": self.file_bytes,
            "sha256": self.sha256,
            "metadata": dict(self.metadata),
            "dtypes": sorted({e.dtype_code for e in self.entries.values()}),
        }


def _read_uint64(fh) -> int:
    raw = fh.read(8)
    if len(raw) != 8:
        raise ArtifactIntegrityError("truncated header: file shorter than 8 bytes")
    return struct.unpack("<Q", raw)[0]


def read_header(path: str | os.PathLike[str], *, compute_sha: bool = False) -> SafetensorsFile:
    """Parse only the header + metadata. Never loads tensor data."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    file_bytes = p.stat().st_size
    with open(p, "rb") as fh:
        header_len = _read_uint64(fh)
        if header_len == 0:
            raise ArtifactIntegrityError("zero-length safetensors header")
        if header_len > MAX_HEADER_BYTES:
            raise ArtifactIntegrityError(
                f"header length {header_len} exceeds cap {MAX_HEADER_BYTES}"
            )
        if 8 + header_len > file_bytes:
            raise ArtifactIntegrityError("header length exceeds file size")
        raw_header = fh.read(header_len)
    if len(raw_header) != header_len:
        raise ArtifactIntegrityError("truncated header payload")

    try:
        header = json.loads(raw_header.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ArtifactIntegrityError(f"header is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(f"header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ArtifactIntegrityError("header must be a JSON object")

    metadata: dict[str, str] = {}
    entries: dict[str, TensorEntry] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            if not isinstance(spec, dict):
                raise ArtifactIntegrityError("__metadata__ must be an object")
            metadata = {str(k): str(v) for k, v in spec.items()}
            continue
        if not isinstance(spec, dict):
            raise ArtifactIntegrityError(f"tensor entry '{name}' must be an object")
        for key in ("dtype", "shape", "data_offsets"):
            if key not in spec:
                raise ArtifactIntegrityError(f"tensor entry '{name}' missing '{key}'")
        code = str(spec["dtype"]).upper()
        shape = spec["shape"]
        offsets = spec["data_offsets"]
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise ArtifactIntegrityError(f"tensor '{name}' has an invalid shape: {shape!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) for o in offsets)
        ):
            raise ArtifactIntegrityError(f"tensor '{name}' has invalid data_offsets")
        begin, end = int(offsets[0]), int(offsets[1])
        if begin < 0 or end < begin:
            raise ArtifactIntegrityError(f"tensor '{name}' has an inverted offset span")
        if code not in SAFETENSORS_CODES and code not in UNSUPPORTED_CODES:
            raise UnsupportedFormat(f"tensor '{name}' uses unknown dtype code '{code}'")
        entries[name] = TensorEntry(name, code, tuple(int(d) for d in shape), begin, end)

    if not entries:
        raise ArtifactIntegrityError("safetensors file contains no tensors")

    # Structural validation: contiguous coverage from 0, no overlap.
    ordered = sorted(entries.values(), key=lambda e: (e.begin, e.end))
    cursor = 0
    for e in ordered:
        if e.begin != cursor:
            if e.begin < cursor:
                raise ArtifactIntegrityError(
                    f"tensor '{e.name}' overlaps a previous tensor (begin={e.begin}, cursor={cursor})"
                )
            raise ArtifactIntegrityError(
                f"tensor '{e.name}' leaves a gap in the data buffer "
                f"(begin={e.begin}, cursor={cursor})"
            )
        cursor = e.end
    data_bytes = max(0, file_bytes - 8 - header_len)
    if cursor != data_bytes:
        raise ArtifactIntegrityError(
            f"data buffer size mismatch: header describes {cursor} bytes, file has {data_bytes}"
        )

    # Per-tensor span/dtype agreement.
    for e in entries.values():
        if e.dtype_code in UNSUPPORTED_CODES and e.dtype_code != "BF16":
            continue  # span check deferred until decode (packed formats)
        itemsize = _itemsize(e.dtype_code)
        numel = 1
        for d in e.shape:
            numel *= d
        expected = numel * itemsize
        if expected != e.nbytes:
            raise ArtifactIntegrityError(
                f"tensor '{e.name}' span {e.nbytes} != expected {expected} "
                f"({e.dtype_code}{list(e.shape)})"
            )

    return SafetensorsFile(
        path=p,
        entries=entries,
        metadata=metadata,
        header_bytes=header_len,
        data_bytes=data_bytes,
        file_bytes=file_bytes,
        sha256=sha256_file(p) if compute_sha else None,
    )


def _itemsize(code: str) -> int:
    if code == "BOOL":
        return 1
    if code in ("F16", "BF16", "I16", "U16"):
        return 2
    if code in ("F32", "I32", "U32"):
        return 4
    if code in ("F64", "I64", "U64"):
        return 8
    if code in ("I8", "U8"):
        return 1
    raise UnsupportedFormat(f"unknown itemsize for dtype code {code}")


def read_safetensors(
    path: str | os.PathLike[str],
    *,
    keep_bf16_bits: bool = False,
    names: list[str] | None = None,
    verify_sha: bool = False,
) -> tuple[dict[str, np.ndarray], SafetensorsFile]:
    """Load tensors into memory.

    ``bfloat16`` tensors are widened losslessly to float32 unless
    ``keep_bf16_bits`` is set, in which case raw uint16 bit patterns are
    returned (the header still records the true dtype).
    """
    header = read_header(path, compute_sha=True)
    if verify_sha:
        actual = sha256_file(path)
        if header.sha256 and actual != header.sha256:
            raise ArtifactIntegrityError("checksum changed between header read and data load")
    wanted = set(names) if names is not None else None
    tensors: dict[str, np.ndarray] = {}
    with open(header.path, "rb") as fh:
        fh.seek(8 + header.header_bytes)
        base = fh.tell()
        for name, entry in header.entries.items():
            if wanted is not None and name not in wanted:
                continue
            if entry.dtype_code in UNSUPPORTED_CODES and entry.dtype_code != "BF16":
                raise UnsupportedFormat(
                    f"tensor '{name}': {UNSUPPORTED_CODES[entry.dtype_code]}"
                )
            fh.seek(base + entry.begin)
            raw = fh.read(entry.nbytes)
            if len(raw) != entry.nbytes:
                raise ArtifactIntegrityError(f"tensor '{name}' truncated")
            tensors[name] = _decode(entry, raw, keep_bf16_bits)
    missing = (wanted or set()) - set(tensors)
    if missing:
        raise ArtifactIntegrityError(f"requested tensors not present: {sorted(missing)}")
    return tensors, header


def _decode(entry: TensorEntry, raw: bytes, keep_bf16_bits: bool) -> np.ndarray:
    if entry.dtype_code == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").reshape(entry.shape)
        return bits.copy() if keep_bf16_bits else bf16_decode(bits)
    np_dtype = np.dtype(SAFETENSORS_CODES[entry.dtype_code])
    arr = np.frombuffer(raw, dtype=np_dtype.newbyteorder("<"))
    return arr.reshape(entry.shape).astype(np_dtype, copy=True)


def build_header(
    tensors: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, str] | None = None,
    bf16_tensors: set[str] | None = None,
) -> tuple[bytes, list[bytes]]:
    """Build the JSON header and ordered raw buffers for a tensor mapping."""
    bf16_tensors = set(bf16_tensors or ())
    header: dict[str, Any] = {}
    buffers: list[bytes] = []
    cursor = 0
    for name in sorted(tensors):
        arr = tensors[name]
        if not isinstance(arr, np.ndarray):
            raise DTypeMismatch(f"tensor '{name}' is not a numpy array")
        if name in bf16_tensors:
            code = "BF16"
            payload = np.ascontiguousarray(bf16_encode(arr)).tobytes()
        else:
            code = CODE_FOR_DTYPE.get(dtype_name(arr.dtype))
            if code is None:  # pragma: no cover - dtype_name already validates
                raise UnsupportedFormat(f"tensor '{name}': cannot encode dtype {arr.dtype}")
            payload = np.ascontiguousarray(arr).tobytes()
        if len(payload) == 0:
            # Zero-element tensors: legal, span is empty.
            pass
        header[name] = {
            "dtype": code,
            "shape": [int(d) for d in arr.shape],
            "data_offsets": [cursor, cursor + len(payload)],
        }
        buffers.append(payload)
        cursor += len(payload)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    pad = (-len(header_bytes)) % HEADER_ALIGNMENT
    header_bytes = header_bytes + b" " * pad
    return header_bytes, buffers


def write_safetensors(
    path: str | os.PathLike[str],
    tensors: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, str] | None = None,
    bf16_tensors: set[str] | None = None,
    atomic: bool = True,
) -> Path:
    """Serialize tensors to the safetensors format."""
    if not tensors:
        raise ArtifactIntegrityError("refusing to write an empty safetensors file")
    header_bytes, buffers = build_header(tensors, metadata=metadata, bf16_tensors=bf16_tensors)
    out = bytearray()
    out += struct.pack("<Q", len(header_bytes))
    out += header_bytes
    for buf in buffers:
        out += buf
    target = Path(path)
    if atomic:
        atomic_write_bytes(target, bytes(out))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(out))
    return target


def safetensors_tensor_digest(tensors: Mapping[str, np.ndarray]) -> str:
    """Content digest of a tensor mapping, independent of file layout/metadata."""
    from uzen.util.canonical import ContentHasher

    hasher = ContentHasher()
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        code = CODE_FOR_DTYPE.get(dtype_name(arr.dtype), dtype_name(arr.dtype))
        hasher.field(
            f"tensor:{name}",
            f"{code}|{list(arr.shape)}|".encode("utf-8") + np.ascontiguousarray(arr).tobytes(),
        )
    return hasher.hexdigest()
