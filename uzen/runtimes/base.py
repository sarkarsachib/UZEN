"""Runtime abstraction.

A runtime adapter is what actually turns an artifact into executable arrays. UZEN
never claims a backend exists merely because an interface does: each adapter reports
honest capabilities, and unavailable backends raise
:class:`uzen.errors.RuntimeUnavailable`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import RuntimeUnavailable


@dataclass(frozen=True)
class RuntimeCapabilities:
    name: str
    backend: str
    device: str = "cpu"
    dtypes: tuple[str, ...] = ("float32", "float64", "float16")
    supports_autograd: bool = False
    supports_safetensors: bool = True
    supports_torch_checkpoints: bool = False
    supports_onnx: bool = False
    supports_gguf: bool = False
    max_tensor_bytes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "device": self.device,
            "dtypes": list(self.dtypes),
            "supports_autograd": self.supports_autograd,
            "supports_safetensors": self.supports_safetensors,
            "supports_torch_checkpoints": self.supports_torch_checkpoints,
            "supports_onnx": self.supports_onnx,
            "supports_gguf": self.supports_gguf,
            "max_tensor_bytes": self.max_tensor_bytes,
            "extra": self.extra,
            "strict_plus_compatible": not self.supports_autograd,
        }


class Runtime(Protocol):
    name: str

    def capabilities(self) -> RuntimeCapabilities: ...

    def load(self, path: str | os.PathLike[str]) -> WeightPackage: ...

    def save(self, package: WeightPackage, path: str | os.PathLike[str]) -> str: ...

    def execute(
        self,
        graph: Graph,
        inputs: Mapping[str, np.ndarray],
        package: WeightPackage,
        *,
        capture: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]: ...


def require_available(condition: bool, message: str, **details: Any) -> None:
    if not condition:
        raise RuntimeUnavailable(message, **details)
