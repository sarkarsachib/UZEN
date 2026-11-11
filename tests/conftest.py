"""Shared fixtures for the UZEN test-suite.

Every fixture builds real artifacts with real code paths — no mocks of the
computational layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uzen.core.nws.families import ArchitectureDefinition, build_architecture
from uzen.core.nws.store import ArtifactStore
from uzen.core.nws.tensors import WeightPackage
from uzen.jobs.service import JobService


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "uzen"


@pytest.fixture()
def store(workspace: Path) -> ArtifactStore:
    return ArtifactStore(workspace / "artifacts")


@pytest.fixture()
def service(workspace: Path) -> JobService:
    return JobService(workspace)


@pytest.fixture()
def mlp() -> ArchitectureDefinition:
    return build_architecture(
        "mlp", {"input_dim": 8, "hidden_dims": [16, 12], "output_dim": 4}
    )


@pytest.fixture()
def mlp_package(mlp: ArchitectureDefinition) -> WeightPackage:
    return mlp.build_package(seed=7, name="mlp_base")


@pytest.fixture()
def transformer() -> ArchitectureDefinition:
    return build_architecture(
        "transformer_decoder",
        {
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 32,
            "vocab_size": 24,
            "max_seq_len": 12,
        },
    )


@pytest.fixture()
def transformer_package(transformer: ArchitectureDefinition) -> WeightPackage:
    return transformer.build_package(seed=11, name="tiny_transformer")


@pytest.fixture()
def linear_pair() -> tuple[WeightPackage, WeightPackage, np.ndarray]:
    """Two packages related by a known orthogonal transform (ground truth)."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((6, 5))
    from uzen.util.linalg import orthogonal_matrix

    q = orthogonal_matrix(5, 5, seed=3)
    a = WeightPackage.from_arrays({"layer.weight": w, "layer.bias": np.zeros(6)}, name="a")
    b = WeightPackage.from_arrays({"layer.weight": w @ q, "layer.bias": np.zeros(6)}, name="b")
    return a, b, q
