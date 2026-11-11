"""Typed values for the Neural Intermediate Representation (NIR).

NIR is a typed computational *graph*, not a sequential layer list. Shapes may be
partially symbolic: an ``int`` is a concrete dimension, a ``str`` is a symbol
(``"B"``, ``"T"``, ``"L"``) resolved at execution time, and ``None`` is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

Dim = int | str | None


def dims_product(shape: Sequence[Dim]) -> int | None:
    total = 1
    for d in shape:
        if not isinstance(d, int):
            return None
        total *= d
    return total


def is_concrete(shape: Sequence[Dim]) -> bool:
    return all(isinstance(d, int) for d in shape)


def normalize_shape(shape: Iterable[Any]) -> tuple[Dim, ...]:
    out: list[Dim] = []
    for d in shape:
        if isinstance(d, bool):
            raise TypeError("boolean is not a valid tensor dimension")
        if isinstance(d, int):
            out.append(int(d))
        elif d is None:
            out.append(None)
        elif isinstance(d, str):
            out.append(d)
        else:
            raise TypeError(f"invalid dimension specifier {d!r}")
    return tuple(out)


@dataclass(frozen=True)
class TensorType:
    shape: tuple[Dim, ...]
    dtype: str = "float32"

    @classmethod
    def of(cls, shape: Iterable[Any], dtype: str = "float32") -> "TensorType":
        return cls(normalize_shape(shape), dtype)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def to_dict(self) -> dict[str, Any]:
        return {"shape": [d for d in self.shape], "dtype": self.dtype}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorType":
        return cls(normalize_shape(data.get("shape", [])), str(data.get("dtype", "float32")))

    def __str__(self) -> str:
        dims = ", ".join(str(d) for d in self.shape)
        return f"tensor<{self.dtype}>[{dims}]"


def broadcast_shapes(a: Sequence[Dim], b: Sequence[Dim]) -> tuple[Dim, ...]:
    """Numpy broadcasting over possibly-symbolic shapes."""
    ra, rb = len(a), len(b)
    rank = max(ra, rb)
    left = (1,) * (rank - ra) + tuple(a)
    right = (1,) * (rank - rb) + tuple(b)
    out: list[Dim] = []
    for x, y in zip(left, right):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        elif isinstance(x, int) and isinstance(y, int):
            raise ValueError(f"shapes are not broadcastable: {tuple(a)} vs {tuple(b)}")
        elif x is None:
            out.append(y)
        elif y is None:
            out.append(x)
        elif isinstance(x, str) and isinstance(y, str):
            # Two different symbols in a broadcast position cannot be resolved
            # statically; keep the outer symbol and let runtime validate.
            out.append(x)
        else:
            raise ValueError(f"shapes are not broadcastable: {tuple(a)} vs {tuple(b)}")
    return tuple(out)


def replace_symbols(shape: Sequence[Dim], env: Mapping[str, int]) -> tuple[int, ...]:
    out: list[int] = []
    for d in shape:
        if isinstance(d, int):
            out.append(d)
        elif isinstance(d, str):
            if d not in env:
                raise KeyError(f"unresolved symbolic dimension '{d}'")
            out.append(int(env[d]))
        else:
            raise KeyError("cannot resolve an unknown (None) dimension at runtime")
    return tuple(out)


@dataclass(frozen=True)
class Ref:
    """A reference to a node output port, e.g. ``ln1`` or ``qkv:1``."""

    node: str
    port: int = 0

    @classmethod
    def parse(cls, text: str) -> "Ref":
        # Graph inputs are addressed as "input:<name>"; the suffix is a graph-input
        # name, not a port index, so keep the whole token as the node id.
        if text.startswith("input:"):
            return cls(text, 0)
        if ":" in text:
            node, _, port = text.partition(":")
            try:
                return cls(node, int(port))
            except ValueError as exc:
                raise ValueError(f"invalid ref '{text}': port must be an integer") from exc
        return cls(text, 0)

    def __str__(self) -> str:
        return self.node if self.port == 0 else f"{self.node}:{self.port}"

    def to_dict(self) -> str:
        return str(self)
