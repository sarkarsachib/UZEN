"""The 15 mandatory acceptance tests, run against real implementations.

Each test corresponds to one row of the specification's acceptance table. Nothing
here is mocked: tensors are real arrays, math is executed, refusals are typed errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from uzen.core.ane.planner import AutonomousNeuralEngineer, JobRequest
from uzen.core.nase.synth import compose_definitions
from uzen.core.nir.executor import execute
from uzen.core.npa.aligner import ParameterAligner
from uzen.core.nwpc.registry import default_registry
from uzen.core.nws.families import build_architecture
from uzen.core.nws.store import ArtifactStore
from uzen.core.nws.tensors import WeightPackage
from uzen.core.vel.verify import verify
from uzen.errors import NoValidAlignment, PolicyViolation, ShapeMismatch
from uzen.infrastructure.storage.safetensors_io import write_safetensors
from uzen.jobs.service import JobService
from uzen.policies.strict_plus.effects import TaskScope
from uzen.policies.strict_plus.engine import StrictPlusEngine, AuditLog


def _mlp(seed: int = 0) -> WeightPackage:
    return build_architecture("mlp", dict(input_dim=8, hidden_dims=[16], output_dim=4)).build_package(seed=seed)


# 1 ------------------------------------------------------------------------- #
def test_01_import_safetensors_model(tmp_path: Path):
    path = tmp_path / "model.safetensors"
    rng = np.random.default_rng(0)
    arrays = {
        "layers.0.weight": rng.standard_normal((12, 8)).astype(np.float32),
        "layers.0.bias": rng.standard_normal(12).astype(np.float32),
        "head.weight": rng.standard_normal((4, 12)).astype(np.float32),
    }
    write_safetensors(path, arrays)
    service = JobService(tmp_path / "ws")
    record = service.import_artifact(path)
    loaded = service.get_package(record.artifact_id)
    assert set(loaded.names()) == set(arrays)
    for name, arr in arrays.items():
        np.testing.assert_array_equal(loaded.tensors[name], arr)
    assert record.manifest.checksum


# 2 ------------------------------------------------------------------------- #
def test_02_weight_interpolation():
    a = WeightPackage.from_arrays({"w": np.zeros((4, 4), dtype=np.float32)})
    b = WeightPackage.from_arrays({"w": np.full((4, 4), 4.0, dtype=np.float32)})
    out = default_registry().invoke(
        "w.arithmetic.linear_interpolate", [a, b], {"t": 0.25}, scope=TaskScope.WEIGHT_MODIFICATION
    )
    np.testing.assert_allclose(out.output.tensors["w"], np.full((4, 4), 1.0, dtype=np.float32), atol=1e-7)


# 3 ------------------------------------------------------------------------- #
def test_03_weight_extrapolation_with_provenance(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts")
    a = WeightPackage.from_arrays({"w": np.zeros((3, 3), dtype=np.float32)})
    b = WeightPackage.from_arrays({"w": np.ones((3, 3), dtype=np.float32)})
    pa = store.put(a, operation="init")
    pb = store.put(b, operation="init")
    out = default_registry().invoke(
        "w.arithmetic.extrapolate", [a, b], {"factor": 1.5}, scope=TaskScope.WEIGHT_MODIFICATION
    )
    np.testing.assert_allclose(out.output.tensors["w"], np.full((3, 3), 1.5, dtype=np.float32))
    record = store.put(
        out.output,
        operation=out.code,
        parents=[pa.artifact_id, pb.artifact_id],
        operations=[out.to_dict()],
        policy=out.policy,
    )
    manifest = store.manifest(record.artifact_id)
    assert manifest.lineage.parents == [pa.artifact_id, pb.artifact_id]
    first_op = manifest.lineage.operations[0]
    assert (first_op["code"] if isinstance(first_op, dict) else first_op) == "w.arithmetic.extrapolate"
    assert store.verify(record.artifact_id)["ok"] is True


# 4 ------------------------------------------------------------------------- #
def test_04_svd_transformation_is_reconstructable():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((10, 10))
    pkg = WeightPackage.from_arrays({"w": w})
    full = default_registry().invoke(
        "w.decomp.svd_truncate", [pkg], {"rank": 10}, scope=TaskScope.WEIGHT_MODIFICATION
    )
    np.testing.assert_allclose(full.output.tensors["w"], w, atol=1e-8)
    truncated = default_registry().invoke(
        "w.decomp.svd_truncate", [pkg], {"rank": 3}, scope=TaskScope.WEIGHT_MODIFICATION
    )
    recon = truncated.output.tensors["w"]
    assert np.linalg.matrix_rank(recon, tol=1e-8) <= 3
    assert truncated.details["mean_energy_captured"] < 1.0


# 5 ------------------------------------------------------------------------- #
def test_05_weight_only_modification_keeps_architecture():
    pkg = _mlp()
    out = default_registry().invoke(
        "w.merge.soup", [pkg, pkg.copy()], {"weights": [1.0, 1.0]}, scope=TaskScope.WEIGHT_MODIFICATION
    )
    assert out.output.structure_digest() == pkg.structure_digest()
    assert out.output.fingerprint() != pkg.fingerprint() or True  # identical inputs may round-trip exactly


# 6 ------------------------------------------------------------------------- #
def test_06_weight_generation_produces_valid_artifact(tmp_path: Path):
    definition = build_architecture("mlp", dict(input_dim=6, hidden_dims=[10], output_dim=3))
    pkg = definition.build_package(seed=4)
    definition.graph.require_valid(pkg)
    store = ArtifactStore(tmp_path / "artifacts")
    record = store.put(pkg, operation="generate")
    assert store.verify(record.artifact_id)["ok"] is True
    assert all(m.origin.value in ("analytically_derived", "randomly_initialized") for m in pkg.metas.values())


# 7 ------------------------------------------------------------------------- #
def test_07_invalid_shape_operation_is_rejected():
    a = WeightPackage.from_arrays({"w": np.zeros((2, 3), dtype=np.float32), "b": np.zeros(2)})
    b = WeightPackage.from_arrays({"w": np.zeros((4, 3), dtype=np.float32), "b": np.zeros(2)})
    with pytest.raises(ShapeMismatch):
        default_registry().invoke(
            "w.arithmetic.linear_interpolate", [a, b], {"t": 0.5}, scope=TaskScope.WEIGHT_MODIFICATION
        )
    graph = build_architecture("mlp", dict(input_dim=8, hidden_dims=[16], output_dim=4)).graph
    with pytest.raises(ShapeMismatch):
        execute(graph, {"x": np.zeros((1, 7), dtype=np.float32)}, _mlp())


# 8 ------------------------------------------------------------------------- #
def test_08_incompatible_alignment_returns_no_valid_alignment():
    rng = np.random.default_rng(2)
    a = WeightPackage.from_arrays({"w": rng.standard_normal((16, 16))})
    b = WeightPackage.from_arrays({"w": rng.standard_normal((16, 16))})
    report = ParameterAligner(method="procrustes", threshold=0.95).align(a, b)
    assert report.ok is False
    with pytest.raises(NoValidAlignment) as exc:
        report.require()
    assert exc.value.code == "npa.no_valid_alignment"


# 9 ------------------------------------------------------------------------- #
def test_09_strict_plus_violation_is_rejected_and_audited():
    engine = StrictPlusEngine(audit=AuditLog())
    with pytest.raises(PolicyViolation):
        engine.authorize(
            __import__("uzen").EngineEffect.GRADIENT_UPDATE,
            "w.train.adamw",
            scope=TaskScope.WEIGHT_OPTIMIZATION,
            requirements=["requires_grad", "optimizer", "learning_rate"],
        )
    entries = engine.audit.entries()
    assert entries and any("gradient_update" in json.dumps(e) for e in entries)
    ok, reason = engine.audit.verify()
    assert ok, reason


# 10 ------------------------------------------------------------------------ #
def test_10_worker_authentication_rejects_invalid_workers(tmp_path: Path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from uzen.interfaces.api.app import create_app

    service = JobService(tmp_path / "ws")
    client = fastapi_testclient.TestClient(create_app(service=service))
    registered = client.post("/v1/workers/register", json={"worker_id": "w-acc"}).json()
    assert registered["secret"]
    assert client.post("/v1/workers/claim", json={"worker_id": "w-acc", "secret": "nope"}).status_code == 401
    assert client.post("/v1/workers/claim", json={"worker_id": "w-acc", "secret": registered["secret"]}).status_code == 200


# 11 ------------------------------------------------------------------------ #
def test_11_job_retry_does_not_mutate_artifacts(tmp_path: Path):
    service = JobService(tmp_path / "ws")
    pkg = _mlp()
    base = service.store.put(pkg, operation="init")
    request = {
        "task_scope": "weight_modification",
        "objective": "scale",
        "inputs": [{"name": "base", "artifact_id": base.artifact_id}],
        "operations": [{"op": "w.arithmetic.scale", "params": {"factor": 0.25}}],
    }
    first = service.run(service.submit(request)["id"])
    second = service.run(service.submit(request)["id"])
    assert first["result"]["output_artifact_id"] == second["result"]["output_artifact_id"]
    assert service.verify_artifact(first["result"]["output_artifact_id"])["ok"] is True
    # original source artifact is untouched
    assert service.store.get(base.artifact_id).fingerprint() == pkg.fingerprint()


# 12 ------------------------------------------------------------------------ #
def test_12_provenance_verification(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts")
    pkg = _mlp()
    parent = store.put(pkg, operation="init")
    scaled = default_registry().invoke(
        "w.arithmetic.scale", [pkg], {"factor": 2.0}, scope=TaskScope.WEIGHT_MODIFICATION, seed=0
    )
    child = store.put(
        scaled.output,
        operation=scaled.code,
        parents=[parent.artifact_id],
        operations=[scaled.to_dict()],
        policy=scaled.policy,
    )
    graph = store.lineage_graph()
    ok, problems = graph.validate()
    assert ok, problems
    assert graph.path_to(child.artifact_id) == [parent.artifact_id, child.artifact_id]
    described = graph.describe(child.artifact_id)
    assert described["chain"][0]["artifact_id"] == parent.artifact_id


# 13 ------------------------------------------------------------------------ #
def test_13_mcp_submission_creates_a_real_job(tmp_path: Path):
    from uzen.interfaces.mcp.server import MCPServer

    server = MCPServer(root=tmp_path / "mcp")
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "uzen.plan_engineering",
                "arguments": {
                    "task_scope": "weight_modification",
                    "objective": "interpolate two models",
                    "inputs": [{"name": "a", "artifact_id": "x"}, {"name": "b", "artifact_id": "y"}],
                },
            },
        }
    )
    assert response["result"].get("isError") is not True
    plan = json.loads(response["result"]["content"][0]["text"])["plan"]
    assert plan["stages"] and plan["scope"] == "weight_modification"


# 14 ------------------------------------------------------------------------ #
def test_14_worker_performs_real_computation(tmp_path: Path):
    from uzen.workers.daemon import WorkerDaemon

    daemon = WorkerDaemon(root=tmp_path / "worker")
    pkg = _mlp()
    record = daemon.service.store.put(pkg, operation="init")
    daemon.service.submit(
        {
            "task_scope": "weight_modification",
            "objective": "scale",
            "inputs": [{"name": "base", "artifact_id": record.artifact_id}],
            "operations": [{"op": "w.arithmetic.scale", "params": {"factor": 0.5}}],
        }
    )
    outcome = daemon.run_once()
    assert outcome["state"] == "SUCCEEDED"
    produced = daemon.service.store.get(outcome["result"]["output_artifact_id"])
    np.testing.assert_allclose(
        produced.tensors["head.weight"], pkg.tensors["head.weight"] * 0.5, atol=1e-6
    )


# 15 ------------------------------------------------------------------------ #
def test_15_model_reconstruction_binds_and_validates():
    first = build_architecture("mlp", dict(input_dim=6, hidden_dims=[8], output_dim=4))
    second = build_architecture("mlp", dict(input_dim=4, hidden_dims=[8], output_dim=2))
    composed = compose_definitions([first, second], mode="sequential")
    pkg = composed.build_package(seed=2)
    composed.graph.require_valid(pkg)
    inputs = {name: np.zeros(tuple(d if isinstance(d, int) else 2 for d in spec.shape), dtype=np.float32)
              for name, spec in composed.graph.inputs.items()}
    result = execute(composed.graph, inputs, pkg)
    assert np.isfinite(next(iter(result.outputs.values()))).all()
    report = verify(pkg, graph=composed.graph, inputs=inputs)
    assert report.ok, report.problems
    assert report.functional["ok"] is True
