"""GGML quantization type registry and block codecs.

The registry is deliberately metadata-first. Every known type records its id, name,
block geometry and effective bits per weight; dequantization and quantization are
*separately registered* kernels. A type can therefore be inspected and reported as
metadata-only without any risk of being silently decoded by a guessed layout.

Implemented codecs here: Q4_0, Q4_1 and Q8_0 (decode), Q8_0 (encode). K-quant, IQ and
ternary types are registered with their real geometry but no codec until an exact,
version-matched implementation is added; ``dequantize`` then raises
:class:`UnsupportedFormat` naming the type instead of returning fabricated values.

Third-party codecs can be added with :func:`register_decoder` / :func:`register_encoder`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from uzen.errors import ArtifactIntegrityError, UnsupportedFormat

GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q5_0 = 6
GGML_Q5_1 = 7
GGML_Q8_0 = 8
GGML_Q2_K = 10
GGML_Q3_K = 11
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_Q8_K = 15
GGML_IQ2_XXS = 16
GGML_IQ2_XS = 17
GGML_IQ3_XXS = 18
GGML_IQ1_S = 19
GGML_IQ4_NL = 20
GGML_IQ3_S = 21
GGML_IQ2_S = 22
GGML_IQ4_XS = 23


@dataclass(frozen=True)
class GGMLType:
    type_id: int
    name: str
    block_size: int
    block_bytes: int
    family: str

    @property
    def effective_bits(self) -> float:
        return round(self.block_bytes * 8 / self.block_size, 6)

    def to_dict(self) -> dict:
        return {
            "type_id": self.type_id,
            "name": self.name,
            "block_size": self.block_size,
            "block_bytes": self.block_bytes,
            "effective_bits": self.effective_bits,
            "family": self.family,
            "decoder_available": self.type_id in _DECODERS,
            "encoder_available": self.type_id in _ENCODERS,
        }


# Geometry as published by GGML; IDs must match the targeted GGML version.
_TYPES: tuple[GGMLType, ...] = (
    GGMLType(GGML_F32, "f32", 1, 4, "float"),
    GGMLType(GGML_F16, "f16", 1, 2, "float"),
    GGMLType(GGML_Q4_0, "q4_0", 32, 18, "legacy"),
    GGMLType(GGML_Q4_1, "q4_1", 32, 20, "legacy"),
    GGMLType(GGML_Q5_0, "q5_0", 32, 22, "legacy"),
    GGMLType(GGML_Q5_1, "q5_1", 32, 24, "legacy"),
    GGMLType(GGML_Q8_0, "q8_0", 32, 34, "legacy"),
    GGMLType(GGML_Q2_K, "q2_k", 256, 84, "k_quant"),
    GGMLType(GGML_Q3_K, "q3_k", 256, 110, "k_quant"),
    GGMLType(GGML_Q4_K, "q4_k", 256, 144, "k_quant"),
    GGMLType(GGML_Q5_K, "q5_k", 256, 176, "k_quant"),
    GGMLType(GGML_Q6_K, "q6_k", 256, 210, "k_quant"),
    GGMLType(GGML_Q8_K, "q8_k", 256, 292, "k_quant"),
    GGMLType(GGML_IQ2_XXS, "iq2_xxs", 256, 66, "iq"),
    GGMLType(GGML_IQ2_XS, "iq2_xs", 256, 74, "iq"),
    GGMLType(GGML_IQ3_XXS, "iq3_xxs", 256, 98, "iq"),
    GGMLType(GGML_IQ1_S, "iq1_s", 256, 50, "iq"),
    GGMLType(GGML_IQ4_NL, "iq4_nl", 32, 18, "iq"),
    GGMLType(GGML_IQ3_S, "iq3_s", 256, 110, "iq"),
    GGMLType(GGML_IQ2_S, "iq2_s", 256, 82, "iq"),
    GGMLType(GGML_IQ4_XS, "iq4_xs", 256, 136, "iq"),
)

GGML_TYPES: dict[int, GGMLType] = {entry.type_id: entry for entry in _TYPES}
GGML_TYPES_BY_NAME: dict[str, GGMLType] = {entry.name: entry for entry in _TYPES}

Decoder = Callable[[bytes, int], np.ndarray]
Encoder = Callable[[np.ndarray], bytes]

_DECODERS: dict[int, Decoder] = {}
_ENCODERS: dict[int, Encoder] = {}
_QUANT_BLOCK = 32


def get_type(type_id: int) -> GGMLType:
    try:
        return GGML_TYPES[int(type_id)]
    except KeyError as exc:
        raise UnsupportedFormat(
            f"ggml type id {type_id} is not in this registry; update the registry for the "
            "targeted GGML version rather than guessing its layout",
            ggml_type=int(type_id),
        ) from exc


def type_info(type_id: int) -> dict:
    return get_type(type_id).to_dict()


def catalogue() -> list[dict]:
    return [entry.to_dict() for entry in sorted(GGML_TYPES.values(), key=lambda t: t.type_id)]


def register_decoder(type_id: int, decoder: Decoder, *, replace: bool = False) -> None:
    get_type(type_id)
    if type_id in _DECODERS and not replace:
        raise ValueError(f"a decoder for ggml type {type_id} is already registered")
    _DECODERS[type_id] = decoder


def register_encoder(type_id: int, encoder: Encoder, *, replace: bool = False) -> None:
    get_type(type_id)
    if type_id in _ENCODERS and not replace:
        raise ValueError(f"an encoder for ggml type {type_id} is already registered")
    _ENCODERS[type_id] = encoder


def decoder_available(type_id: int) -> bool:
    return int(type_id) in _DECODERS


def encoder_available(type_id: int) -> bool:
    return int(type_id) in _ENCODERS


# --------------------------------------------------------------------------- #
# implemented codecs
# --------------------------------------------------------------------------- #


def decode_q4_0(raw: bytes, element_count: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    scale = blocks[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    qs = blocks[:, 2:18]
    nibbles = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.float32)
    return (scale[:, None] * (nibbles - 8.0)).reshape(-1)


def decode_q4_1(raw: bytes, element_count: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 20)
    scale = blocks[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    minimum = blocks[:, 2:4].copy().view("<f2").reshape(-1).astype(np.float32)
    qs = blocks[:, 4:20]
    nibbles = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.float32)
    return (scale[:, None] * nibbles + minimum[:, None]).reshape(-1)


def decode_q8_0(raw: bytes, element_count: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    scale = blocks[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    values = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return (scale[:, None] * values).reshape(-1)


def _q5_values(raw: bytes, block_bytes: int, qh_offset: int, qs_offset: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, block_bytes)
    qh = blocks[:, qh_offset : qh_offset + 4].copy().view("<u4").reshape(-1)
    qs = blocks[:, qs_offset : qs_offset + 16]
    j = np.arange(16, dtype=np.uint32)
    high_bit_0 = (((qh[:, None] >> j[None, :]) & 1) << 4).astype(np.uint8)
    high_bit_1 = (((qh[:, None] >> (j[None, :] + 16)) & 1) << 4).astype(np.uint8)
    low = ((qs & 0x0F) | high_bit_0).astype(np.float32)
    high = ((qs >> 4) | high_bit_1).astype(np.float32)
    return np.concatenate([low, high], axis=1)


def decode_q5_0(raw: bytes, element_count: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 22)
    scale = blocks[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    values = _q5_values(raw, 22, 2, 6) - 16.0
    return (scale[:, None] * values).reshape(-1)


def decode_q5_1(raw: bytes, element_count: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 24)
    scale = blocks[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    minimum = blocks[:, 2:4].copy().view("<f2").reshape(-1).astype(np.float32)
    values = _q5_values(raw, 24, 4, 8)
    return (scale[:, None] * values + minimum[:, None]).reshape(-1)


def encode_q8_0(array: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
    if flat.size % _QUANT_BLOCK:
        raise UnsupportedFormat(
            f"Q8_0 requires an element count divisible by {_QUANT_BLOCK}; got {flat.size}"
        )
    blocks = flat.reshape(-1, _QUANT_BLOCK)
    amplitudes = np.abs(blocks).max(axis=1)
    scale = (amplitudes / 127.0).astype(np.float16)
    scale32 = scale.astype(np.float32)
    inverse = np.zeros_like(scale32)
    np.divide(1.0, scale32, out=inverse, where=scale32 != 0)
    q = np.clip(np.rint(blocks * inverse[:, None]), -127, 127).astype(np.int8)
    payload = np.concatenate(
        [scale.view(np.uint8).reshape(-1, 2), q.view(np.uint8).reshape(-1, _QUANT_BLOCK)],
        axis=1,
    )
    return payload.tobytes()


def _scale_min_k4(scales: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Port of ggml's get_scale_min_k4 for arrays of Q4_K scale blocks."""
    if j < 4:
        d = scales[:, j] & 63
        m = scales[:, j + 4] & 63
    else:
        d = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
        m = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return d.astype(np.float32), m.astype(np.float32)


def decode_q4_k(raw: bytes, element_count: int) -> np.ndarray:
    """Q4_K super-blocks: 8 sub-blocks of 32, 6-bit scales/mins, 4-bit quants."""
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 144)
    d = blocks[:, 0:2].copy().view("<f2").reshape(-1).astype(np.float32)
    dmin = blocks[:, 2:4].copy().view("<f2").reshape(-1).astype(np.float32)
    scales = blocks[:, 4:16]
    qs = blocks[:, 16:144]
    out = np.empty((blocks.shape[0], 256), dtype=np.float32)
    for group in range(4):
        sc1, m1 = _scale_min_k4(scales, 2 * group)
        sc2, m2 = _scale_min_k4(scales, 2 * group + 1)
        q = qs[:, 32 * group : 32 * group + 32]
        out[:, 64 * group : 64 * group + 32] = (
            (d * sc1)[:, None] * (q & 0x0F) - (dmin * m1)[:, None]
        )
        out[:, 64 * group + 32 : 64 * group + 64] = (
            (d * sc2)[:, None] * (q >> 4) - (dmin * m2)[:, None]
        )
    return out.reshape(-1)


def decode_q6_k(raw: bytes, element_count: int) -> np.ndarray:
    """Q6_K super-blocks: 16 sub-blocks of 16, int8 scales, signed 6-bit quants."""
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 210)
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = blocks[:, 208:210].copy().view("<f2").reshape(-1).astype(np.float32)
    out = np.empty((blocks.shape[0], 256), dtype=np.float32)
    sub = np.arange(32)
    is_ = sub // 16
    for half in range(2):
        ql_h = ql[:, 64 * half : 64 * half + 64]
        qh_h = qh[:, 32 * half : 32 * half + 32]
        sc = scales[:, 8 * half : 8 * half + 8]
        q1 = ((ql_h[:, :32] & 0x0F) | (((qh_h >> 0) & 3) << 4)).astype(np.int16) - 32
        q2 = ((ql_h[:, 32:] & 0x0F) | (((qh_h >> 2) & 3) << 4)).astype(np.int16) - 32
        q3 = ((ql_h[:, :32] >> 4) | (((qh_h >> 4) & 3) << 4)).astype(np.int16) - 32
        q4 = ((ql_h[:, 32:] >> 4) | (((qh_h >> 6) & 3) << 4)).astype(np.int16) - 32
        base = 128 * half
        out[:, base : base + 32] = d[:, None] * sc[:, is_ + 0] * q1
        out[:, base + 32 : base + 64] = d[:, None] * sc[:, is_ + 2] * q2
        out[:, base + 64 : base + 96] = d[:, None] * sc[:, is_ + 4] * q3
        out[:, base + 96 : base + 128] = d[:, None] * sc[:, is_ + 6] * q4
    return out.reshape(-1)


def dequantize(raw: bytes, type_id: int, element_count: int) -> np.ndarray:
    entry = get_type(type_id)
    if entry.type_id == GGML_F32:
        expected = element_count * 4
        if len(raw) != expected:
            raise ArtifactIntegrityError(f"F32 payload is {len(raw)} bytes, expected {expected}")
        return np.frombuffer(raw, dtype="<f4").copy()
    if entry.type_id == GGML_F16:
        expected = element_count * 2
        if len(raw) != expected:
            raise ArtifactIntegrityError(f"F16 payload is {len(raw)} bytes, expected {expected}")
        return np.frombuffer(raw, dtype="<f2").copy()
    decoder = _DECODERS.get(entry.type_id)
    if decoder is None:
        raise UnsupportedFormat(
            f"ggml type {entry.type_id} ({entry.name}) is registered with block geometry "
            f"({entry.block_size} values / {entry.block_bytes} bytes) but has no decoder; "
            "refusing to guess a dequantization layout",
            ggml_type=entry.type_id,
            type_name=entry.name,
        )
    if element_count % entry.block_size:
        raise UnsupportedFormat(
            f"ggml type {entry.name} requires element counts that are multiples of "
            f"{entry.block_size}; got {element_count}",
            ggml_type=entry.type_id,
        )
    expected = element_count // entry.block_size * entry.block_bytes
    if len(raw) != expected:
        raise ArtifactIntegrityError(
            f"{entry.name} payload is {len(raw)} bytes, expected {expected}"
        )
    return decoder(raw, element_count)


def quantize(array: np.ndarray, type_id: int) -> bytes:
    entry = get_type(type_id)
    if entry.type_id == GGML_F32:
        return np.ascontiguousarray(array, dtype=np.float32).tobytes()
    if entry.type_id == GGML_F16:
        return np.ascontiguousarray(array, dtype=np.float16).tobytes()
    encoder = _ENCODERS.get(entry.type_id)
    if encoder is None:
        raise UnsupportedFormat(
            f"ggml type {entry.type_id} ({entry.name}) has no registered encoder",
            ggml_type=entry.type_id,
            type_name=entry.name,
        )
    return encoder(np.asarray(array))


register_decoder(GGML_Q4_0, decode_q4_0)
register_decoder(GGML_Q4_1, decode_q4_1)
register_decoder(GGML_Q5_0, decode_q5_0)
register_decoder(GGML_Q5_1, decode_q5_1)
register_decoder(GGML_Q8_0, decode_q8_0)
register_decoder(GGML_Q4_K, decode_q4_k)
register_decoder(GGML_Q6_K, decode_q6_k)
register_encoder(GGML_Q8_0, encode_q8_0)


def register_type(entry: GGMLType, *, replace: bool = False) -> None:
    """Register a type from a newer GGML version without editing the base table."""
    if entry.type_id in GGML_TYPES and not replace:
        raise ValueError(f"ggml type {entry.type_id} is already registered")
    GGML_TYPES[entry.type_id] = entry
    GGML_TYPES_BY_NAME[entry.name] = entry
