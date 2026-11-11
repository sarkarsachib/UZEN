"""PyTorch runtime adapter.

Real PyTorch integration, imported lazily so the core engine stays dependency-light:

* safe checkpoint loading (``weights_only=True``; pickle deserialization of arbitrary
  objects is refused),
* conversion between ``state_dict`` tensors and UZEN :class:`WeightPackage` arrays
  (lossless widening for bfloat16, real dtype tagging),
* export back to ``state_dict`` and ``.pt``,
* execution of NIR graphs (via the NumPy reference kernels after conversion) under
  ``torch.inference_mode()`` so **no autograd graph is ever built**,
* honest capability reporting including CUDA availability when present.

It never performs training, gradient computation, or parameter updates. If the
``torch`` package is absent, instantiation raises
:class:`uzen.errors.RuntimeUnavailable`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import (
    DTypeMismatch,
    InvalidOperationParameters,
    RuntimeUnavailable,
    UnsupportedFormat,
    UnsupportedOperator,
)
from uzen.runtimes.base import RuntimeCapabilities

try:  # pragma: no cover - environment dependent
    import torch  # type: ignore

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)


def torch_available() -> bool:
    return TORCH_AVAILABLE


class PyTorchRuntime:
    name = "pytorch"

    def __init__(self, *, device: str = "cpu") -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeUnavailable(
                "PyTorch is not installed in this environment; install it with "
                "`pip install torch` to use the PyTorch runtime. The NumPy reference "
                "runtime remains fully functional.",
                import_error=TORCH_IMPORT_ERROR,
            )
        self.device = torch.device(device)
        self.last_backend = "numpy"
        self.last_fallback_reason: str | None = None
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeUnavailable(
                "CUDA was requested but this PyTorch build has no CUDA support; UZEN "
                "will not silently fall back to CPU",
                requested_device=str(device),
                torch_version=torch.__version__,
            )

    # ------------------------------------------------------------------ #

    def capabilities(self) -> RuntimeCapabilities:
        if not TORCH_AVAILABLE:
            return RuntimeCapabilities(
                name=self.name,
                backend="pytorch",
                device="unavailable",
                supports_autograd=False,
                extra={"available": False, "import_error": TORCH_IMPORT_ERROR},
            )
        return RuntimeCapabilities(
            name=self.name,
            backend="pytorch",
            device=str(self.device),
            dtypes=("float16", "bfloat16", "float32", "float64", "int8", "int32", "int64", "bool"),
            supports_autograd=False,
            supports_safetensors=True,
            supports_torch_checkpoints=True,
            supports_onnx=False,
            supports_gguf=False,
            extra={
                "available": True,
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "execution": "torch-CPU NIR kernels under inference_mode (NumPy fallback for operators without a torch kernel)",
                "strict_plus_enforcement": "torch.inference_mode() for every forward pass; no autograd",
            },
        )

    # ------------------------------------------------------------------ #
    # conversion
    # ------------------------------------------------------------------ #

    def tensor_to_numpy(self, tensor: Any) -> np.ndarray:
        if not TORCH_AVAILABLE:
            raise RuntimeUnavailable("PyTorch is not installed")
        if not isinstance(tensor, torch.Tensor):
            raise DTypeMismatch(f"expected a torch.Tensor, got {type(tensor).__name__}")
        if tensor.requires_grad:
            raise RuntimeUnavailable(
                "refusing to convert a tensor that requires gradients: STRICT_PLUS forbids "
                "operating on autograd-tracked parameters. Detach and disable grad first."
            )
        detached = tensor.detach().to("cpu")
        if detached.dtype == torch.bfloat16:
            # NumPy has no bfloat16 dtype: widen losslessly; the source dtype is recorded
            # in the package metadata by from_state_dict.
            detached = detached.to(torch.float32)
        return detached.numpy().copy()

    def numpy_to_tensor(self, array: np.ndarray) -> Any:
        if not TORCH_AVAILABLE:
            raise RuntimeUnavailable("PyTorch is not installed")
        return torch.from_numpy(np.ascontiguousarray(array))

    # ------------------------------------------------------------------ #

    def from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        name: str = "torch_state_dict",
        family: str = "imported",
        metadata: Mapping[str, Any] | None = None,
    ) -> WeightPackage:
        arrays: dict[str, np.ndarray] = {}
        source_dtypes: dict[str, str] = {}
        for key, value in state_dict.items():
            arr = self.tensor_to_numpy(value)
            arrays[key] = arr
            if str(value.dtype).replace("torch.", "") == "bfloat16":
                source_dtypes[key] = "bfloat16"
        package = WeightPackage.from_arrays(
            arrays,
            name=name,
            family=family,
            metadata={**dict(metadata or {}), "source_format": "torch_state_dict"},
        )
        for key, dtype in source_dtypes.items():
            meta = package.metas[key]
            package.metas[key] = type(meta)(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                role=meta.role,
                origin=meta.origin,
                source_artifact_id=meta.source_artifact_id,
                source_dtype=dtype,
                ops=meta.ops,
                note="bfloat16 widened losslessly to float32",
            )
        return package

    def to_state_dict(self, package: WeightPackage) -> dict[str, Any]:
        if not TORCH_AVAILABLE:
            raise RuntimeUnavailable("PyTorch is not installed")
        out: dict[str, Any] = {}
        for name in package.names():
            meta = package.metas[name]
            arr = package.tensors[name]
            if meta.source_dtype == "bfloat16":
                out[name] = torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
            else:
                out[name] = torch.from_numpy(np.ascontiguousarray(arr))
        return out

    # ------------------------------------------------------------------ #

    def load(self, path: str | os.PathLike[str]) -> WeightPackage:
        p = Path(path)
        if p.suffix.lower() in (".safetensors",):
            from uzen.infrastructure.storage.safetensors_io import read_safetensors

            arrays, header = read_safetensors(p)
            return WeightPackage.from_arrays(
                arrays, name=p.stem, family=header.metadata.get("family", "imported")
            )
        if p.suffix.lower() not in (".pt", ".pth", ".bin", ".ckpt"):
            raise UnsupportedFormat(f"PyTorch runtime cannot read '{p.name}'")
        try:
            obj = torch.load(str(p), map_location="cpu", weights_only=True)
        except TypeError as exc:  # very old torch without weights_only
            raise RuntimeUnavailable(
                "this PyTorch version cannot load checkpoints with weights_only=True; "
                "upgrade to torch>=1.13 so UZEN can avoid unpickling arbitrary objects",
                torch_version=torch.__version__,
            ) from exc
        except Exception as exc:
            raise UnsupportedFormat(
                f"failed to load '{p.name}' as a weights-only checkpoint: {exc}. "
                "UZEN refuses to unpickle arbitrary objects; convert the checkpoint to "
                "safetensors or re-save it with torch.save(model.state_dict(), ...)."
            ) from exc
        if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return self.from_state_dict(obj, name=p.stem)
        if hasattr(obj, "state_dict"):
            return self.from_state_dict(obj.state_dict(), name=p.stem)
        raise UnsupportedFormat(
            f"'{p.name}' did not contain a state_dict of tensors (got {type(obj).__name__})"
        )

    def save(self, package: WeightPackage, path: str | os.PathLike[str]) -> str:
        p = Path(path)
        if p.suffix.lower() in (".pt", ".pth", ".bin"):
            state = self.to_state_dict(package)
            torch.save(state, str(p))
            return str(p)
        from uzen.infrastructure.storage.safetensors_io import write_safetensors

        bf16 = {n for n, m in package.metas.items() if m.source_dtype == "bfloat16"}
        write_safetensors(p, package.tensors, bf16_tensors=bf16, metadata={"family": package.family})
        return str(p)

    # ------------------------------------------------------------------ #

    def execute(
        self,
        graph: Graph,
        inputs: Mapping[str, np.ndarray],
        package: WeightPackage,
        *,
        capture: Sequence[str] | None = None,
        backend: str = "auto",
    ) -> dict[str, np.ndarray]:
        """Forward-only execution under ``torch.inference_mode()``.

        ``backend="auto"`` (default) uses the torch-CPU NIR kernels when every operator
        has a torch implementation, otherwise the NumPy reference; ``last_backend``
        records the path that actually ran. ``backend="torch"`` refuses to fall back,
        and ``backend="numpy"`` forces the reference implementation.
        """
        if backend not in ("auto", "torch", "numpy"):
            raise InvalidOperationParameters(f"unknown execution backend '{backend}'")
        for name in package.names():
            arr = package.tensors[name]
            if type(arr).__module__.startswith("torch") and getattr(arr, "requires_grad", False):
                raise RuntimeUnavailable(
                    f"parameter '{name}' requires gradients; STRICT_PLUS forbids executing "
                    "a forward pass that builds an autograd graph"
                )
        if backend != "numpy":
            from uzen.runtimes.torch_executor import execute_torch

            try:
                with torch.inference_mode():
                    torch_result = execute_torch(
                        graph, inputs, package, capture=capture, device=str(self.device)
                    )
            except UnsupportedOperator as exc:
                if backend == "torch":
                    raise
                self.last_fallback_reason = str(exc)
            else:
                self.last_backend = "torch"
                self.last_fallback_reason = None
                return dict(torch_result.outputs)

        from uzen.core.nir.executor import execute as numpy_execute

        self.last_backend = "numpy"
        with torch.inference_mode():
            result = numpy_execute(graph, inputs, package, capture=capture)
        return dict(result.outputs)

    def describe(self) -> dict[str, Any]:
        return self.capabilities().to_dict()
