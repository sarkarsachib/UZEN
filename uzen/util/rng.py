"""Deterministic RNG discipline.

Every randomized operation in UZEN (sparse pruning masks, DARE drops, structured
random initialization, evolutionary search) must be reproducible from a seed
recorded in provenance. We therefore use isolated ``numpy.random.Generator``
instances derived from an integer/hex seed — never the global numpy state — and
derive child streams by name so that adding an operation to a program does not
perturb unrelated streams.
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterable

import numpy as np


def parse_seed(seed: int | str | None) -> int:
    """Normalize a user-supplied seed (int, decimal string or hex string)."""
    if seed is None:
        return int.from_bytes(hashlib.sha256(os.urandom(16)).digest()[:8], "little")
    if isinstance(seed, bool):
        raise TypeError("seed must not be a bool")
    if isinstance(seed, int):
        if seed < 0:
            raise ValueError("seed must be non-negative")
        return seed
    text = str(seed).strip()
    if not text:
        raise ValueError("empty seed")
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(text, 10)
    except ValueError as exc:  # deterministic fallback: hash the phrase
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little")


def derive_seed(seed: int, *path: str | int) -> int:
    """Derive a stable child seed from a parent seed and a semantic path."""
    h = hashlib.sha256(str(seed).encode("utf-8"))
    for part in path:
        h.update(b"\x1f")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little")


def generator(seed: int | str | None = None, *path: str | int) -> np.random.Generator:
    base = parse_seed(seed)
    child = derive_seed(base, *path) if path else base
    return np.random.default_rng(child)


def seeded_permutation(n: int, seed: int | str | None, *path: str | int) -> np.ndarray:
    return generator(seed, *path).permutation(n)


def deterministic_bernoulli(shape: Iterable[int], keep_prob: float, seed: int | str | None, *path: str | int) -> np.ndarray:
    if not 0.0 <= keep_prob <= 1.0:
        raise ValueError("keep_prob must be in [0, 1]")
    rng = generator(seed, *path)
    return rng.random(tuple(shape)) < keep_prob
