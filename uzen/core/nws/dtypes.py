"""Canonical dtype handling for the neural weight substrate.

UZEN stores tensors as NumPy arrays. ``bfloat16`` has no NumPy dtype, so it is
handled explicitly: on read it is widened *losslessly* to float32 and the source
dtype is recorded; on write the exact bfloat16 bit pattern is reconstructed only
when the values are representable (otherwise an explicit error is raised rather
than silently rounding).
"""

from __future__ import annotations

from typing import Final

import numpy as np

from uzen.errors import DTypeMismatch, UnsupportedFormat
from uzen.util.linalg import bfloat16_bits_to_float32, float32_to_bfloat16_bits

# Canonical name -> numpy dtype
CANONICAL_DTYPES: Final[dict[str, np.dtype]] = {
    "bool": np.dtype(np.bool_),
    "uint8": np.dtype(np.uint8),
    "int8": np.dtype(np.int8),
    "int16": np.dtype(np.int16),
    "uint16": np.dtype(np.uint16),
    "int32": np.dtype(np.int32),
    "uint32": np.dtype(np.uint32),
    "int64": np.dtype(np.int64),
    "uint64": np.dtype(np.uint64),
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}

# safetensors dtype code <-> canonical name
SAFETENSORS_CODES: Final[dict[str, str]] = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "U16": "uint16",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
}
CODE_FOR_DTYPE: Final[dict[str, str]] = {v: k for k, v in SAFETENSORS_CODES.items()}
CODE_FOR_DTYPE["bfloat16"] = "BF16"

# Dtype codes we recognise but deliberately do not decode yet. They are reported
# as unsupported rather than guessed at.
UNSUPPORTED_CODES: Final[dict[str, str]] = {
    "BF16": "bfloat16 is supported through lossless widening to float32; "
            "use the bf16 codec path (see read_safetensors keep_bits)",
    "F8_E4M3": "8-bit float formats are not implemented",
    "F8_E5M2": "8-bit float formats are not implemented",
    "F4": "4-bit packed formats are not implemented",
    "F6_E2M3": "6-bit float formats are not implemented",
    "F6_E3M2": "6-bit float formats are not implemented",
    "I4": "4-bit packed integer formats are not implemented",
    "U4": "4-bit packed integer formats are not implemented",
}

FLOAT_DTYPES: Final[frozenset[np.dtype]] = frozenset(
    {np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)}
)
INTEGER_DTYPES: Final[frozenset[np.dtype]] = frozenset(
    d for name, d in CANONICAL_DTYPES.items() if d.kind in "iu"
)


def normalize_dtype(dtype: str | np.dtype) -> np.dtype:
    """Map a user/format dtype string onto a canonical NumPy dtype."""
    name = str(dtype).lower().strip()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
        "single": "float32",
        "fp64": "float64",
        "double": "float64",
        "long": "int64",
        "int": "int32",
        "boolean": "bool",
    }
    name = aliases.get(name, name)
    if name == "bfloat16":
        # Widened representation.
        return np.dtype(np.float32)
    if name not in CANONICAL_DTYPES:
        raise UnsupportedFormat(f"unsupported tensor dtype '{dtype}'")
    return CANONICAL_DTYPES[name]


def dtype_name(dtype: np.dtype) -> str:
    d = np.dtype(dtype)
    for name, canonical in CANONICAL_DTYPES.items():
        if canonical == d:
            return name
    raise UnsupportedFormat(f"dtype {d} has no canonical UZEN name")


def require_compatible(a: np.ndarray, b: np.ndarray, *, allow_cast: bool = True) -> np.dtype:
    """Return the computation dtype for two tensors or raise a typed error."""
    da, db = np.dtype(a.dtype), np.dtype(b.dtype)
    if da == db:
        return da
    if not allow_cast:
        raise DTypeMismatch(f"dtype mismatch: {da} vs {db}")
    if da.kind in "fc" and db.kind in "fc":
        return np.promote_types(da, db)
    raise DTypeMismatch(
        f"cannot combine floating and non-floating tensors ({da} vs {db}) without an explicit cast"
    )


def cast_within_family(array: np.ndarray, target: np.dtype, *, context: str = "") -> np.ndarray:
    """Cast an array, but refuse silent float->int or int->float narrowing."""
    src, dst = np.dtype(array.dtype), np.dtype(target)
    if src == dst:
        return array
    if src.kind in "fc" and dst.kind in "iu":
        raise DTypeMismatch(
            f"refusing implicit float->integer cast{f' ({context})' if context else ''}: {src} -> {dst}"
        )
    if src.kind in "iu" and dst.kind in "fc":
        raise DTypeMismatch(
            f"refusing implicit integer->float cast{f' ({context})' if context else ''}: {src} -> {dst}"
        )
    return array.astype(dst, copy=False)


def ensure_array(value, *, name: str = "tensor") -> np.ndarray:
    """Validate that a value is a real, C-contiguous-capable NumPy array."""
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple, int, float, bool)):
        arr = np.asarray(value)
    else:
        raise DTypeMismatch(
            f"{name}: expected a numpy array, got {type(value).__name__}; "
            "use a runtime adapter (e.g. uzen.runtimes.pytorch) to convert framework tensors"
        )
    if arr.dtype.kind not in "biufc":
        raise UnsupportedFormat(f"{name}: unsupported dtype {arr.dtype}")
    if arr.dtype.kind == "c":
        raise UnsupportedFormat(f"{name}: complex tensors are not supported by the weight substrate")
    return np.ascontiguousarray(arr)


def bf16_encode(array: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns (uint16), rejecting unrepresentable values."""
    arr = np.asarray(array)
    if np.dtype(arr.dtype) not in FLOAT_DTYPES:
        raise DTypeMismatch(f"bfloat16 encoding requires a float tensor, got {arr.dtype}")
    f32 = arr.astype(np.float32)
    bits = float32_to_bfloat16_bits(f32)
    # Verify the round-trip is exact: if not, the caller asked for precision the
    # format cannot hold and we refuse rather than silently quantize.
    restored = bfloat16_bits_to_float32(bits)
    if not np.array_equal(restored, f32, equal_nan=True):
        mismatch = int(np.count_nonzero(~np.isclose(restored, f32, rtol=0, atol=0, equal_nan=True)))
        raise DTypeMismatch(
            f"{mismatch} values are not exactly representable in bfloat16; "
            "encode explicitly with a round-to-nearest cast if that is intended"
        )
    return bits


def bf16_decode(bits: np.ndarray) -> np.ndarray:
    return bfloat16_bits_to_float32(np.asarray(bits, dtype=np.uint16))
