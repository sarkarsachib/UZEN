"""Deterministic serialization, hashing and atomic filesystem primitives.

Artifacts in UZEN are content addressed: the same mathematical content must
always produce the same identifier, on any machine, in any process. That
requires a canonical encoding (sorted keys, no floats formatted ambiguously, no
NaN) and byte-exact hashing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# time
# --------------------------------------------------------------------------- #


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="microseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# JSON canonicalisation
# --------------------------------------------------------------------------- #


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError("canonical JSON cannot encode NaN/Infinity")
        # repr() round-trips exactly for float64.
        return obj
    if isinstance(obj, (set, frozenset)):
        return sorted(_canonicalize(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_canonical_dict"):
        return _canonicalize(obj.to_canonical_dict())
    if isinstance(obj, (bytes, bytearray)):
        raise TypeError("bytes are not JSON-canonical; hash them explicitly")
    return obj


def canonical_json(obj: Any) -> str:
    """Stable JSON text: sorted keys, compact separators, utf-8 safe."""
    return json.dumps(
        _canonicalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class ContentHasher:
    """Incremental hasher with explicit domain separation between fields."""

    def __init__(self) -> None:
        self._h = hashlib.sha256()

    def field(self, name: str, value: bytes | str) -> "ContentHasher":
        payload = value.encode("utf-8") if isinstance(value, str) else value
        self._h.update(name.encode("utf-8"))
        self._h.update(b"\x1f")
        self._h.update(len(payload).to_bytes(8, "little"))
        self._h.update(payload)
        self._h.update(b"\x1e")
        return self

    def hexdigest(self) -> str:
        return self._h.hexdigest()


def digest_frames(parts: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# atomic filesystem writes
# --------------------------------------------------------------------------- #


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> None:
    atomic_write_bytes(path, canonical_bytes(obj))


def atomic_publish_dir(target: str | os.PathLike[str], writer) -> Path:
    """Build a directory in a temp location, then rename it into place atomically.

    ``writer`` receives the staging :class:`Path`. If the target already exists
    the staged copy is discarded (content addressing means an existing target
    with the same id holds identical content — callers verify this separately).
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=str(target.parent), prefix=".stage-"))
    try:
        writer(staging)
        if target.exists():
            shutil.rmtree(staging, ignore_errors=True)
            return target
        os.replace(staging, target)
        return target
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
