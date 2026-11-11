"""NumPy reference runtime.

This is the always-available execution backend. It loads/saves safetensors and NPZ
artifacts, executes NIR graphs through the typed operator kernels, and never enables
autograd (there is none).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import UnsupportedFormat
from uzen.infrastructure.storage.safetensors_io import read_safetensors, write_safetensors
from uzen.runtimes.base import RuntimeCapabilities


class NumpyRuntime:
    name = "numpy"

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            name=self.name,
            backend="numpy",
            device="cpu",
            dtypes=("float16", "float32", "float64", "int32", "int64", "uint8", "bool"),
            supports_autograd=False,
            supports_safetensors=True,
            supports_torch_checkpoints=False,
            supports_onnx=False,
            supports_gguf=False,
            extra={
                "numpy_version": np.__version__,
                "execution": "typed NIR interpreter",
                "note": "reference runtime; deterministic closed-form kernels only",
            },
        )

    # ------------------------------------------------------------------ #

    def load(self, path: str | os.PathLike[str]) -> WeightPackage:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".safetensors" or suffix == "":
            arrays, header = read_safetensors(p)
            return WeightPackage.from_arrays(
                arrays,
                name=p.stem,
                family=header.metadata.get("family", "imported"),
                metadata={"source_path": str(p), "source_format": "safetensors"},
            )
        if suffix == ".npz":
            data = np.load(p, allow_pickle=False)
            arrays = {k: np.asarray(data[k]) for k in data.files}
            return WeightPackage.from_arrays(
                arrays,
                name=p.stem,
                family="imported",
                metadata={"source_path": str(p), "source_format": "npz"},
            )
        raise UnsupportedFormat(
            f"the NumPy runtime cannot read '{p.name}'; supported: .safetensors, .npz"
        )

    def save(self, package: WeightPackage, path: str | os.PathLike[str]) -> str:
        p = Path(path)
        if p.suffix.lower() == ".npz":
            np.savez(p, **package.tensors)
            return str(p)
        write_safetensors(
            p,
            package.tensors,
            metadata={
                "family": package.family,
                "name": package.name,
                "format": "uzen",
            },
            atomic=True,
        )
        return str(p)

    def execute(
        self,
        graph: Graph,
        inputs: Mapping[str, np.ndarray],
        package: WeightPackage,
        *,
        capture: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        result = execute(graph, inputs, package, capture=capture)
        return dict(result.outputs)

    def execute_with_intermediates(
        self,
        graph: Graph,
        inputs: Mapping[str, np.ndarray],
        package: WeightPackage,
        *,
        capture: Sequence[str],
    ) -> dict[str, np.ndarray]:
        return execute(graph, inputs, package, capture=capture).intermediates

    def describe(self) -> dict[str, Any]:
        return self.capabilities().to_dict()
