"""Job service, SDK, REST API, MCP and worker tests (real implementations)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from uzen.core.nws.families import build_architecture
from uzen.infrastructure.storage.safetensors_io import write_safetensors
from uzen.interfaces.python_sdk.client import Uzen
from uzen.jobs.service import JobService


@pytest.fixture()
def base_artifact(service: JobService) -> str:
    definition = build_architecture("mlp", dict(input_dim=8, hidden_dims=[16], output_dim=4))
    package = definition.build_package(seed=3)
    record = service.store.put(package, operation="init", metadata={"architecture": definition.summary()})
    return record.artifact_id


# --------------------------------------------------------------------------- #
# job service
# --------------------------------------------------------------------------- #


def test_submit_run_creates_artifact_and_events(service: JobService, base_artifact: str):
    job = service.submit(
        {
            "task_scope": "weight_modification",
            "objective": "scale parameters",
            "inputs": [{"name": "base", "artifact_id": base_artifact}],
            "operations": [{"op": "w.arithmetic.scale", "params": {"factor": 0.5}}],
        }
    )
    assert job["state"] == "QUEUED"
    done = service.run(job["id"])
    assert done["state"] == "SUCCEEDED", done
    result = done["result"]
    assert result["output_artifact_id"]
    assert service.verify_artifact(result["output_artifact_id"])["ok"] is True
    events = service.events(job["id"])
    kinds = [e["kind"] for e in events]
    assert "queued" in kinds and "running" in kinds
    assert any(k in ("artifact_created", "completed", "succeeded") for k in kinds), kinds
    manifest = service.artifact_manifest(result["output_artifact_id"])
    assert manifest["lineage"]["parents"] == [base_artifact]


def test_idempotency_key_returns_same_job(service: JobService, base_artifact: str):
    request = {
        "task_scope": "weight_modification",
        "objective": "scale",
        "inputs": [{"name": "base", "artifact_id": base_artifact}],
        "operations": [{"op": "w.arithmetic.scale", "params": {"factor": 2.0}}],
    }
    first = service.submit(request, idempotency_key="k1")
    second = service.submit(request, idempotency_key="k1")
    assert first["id"] == second["id"]


def test_retry_does_not_duplicate_artifact(service: JobService, base_artifact: str):
    job = service.submit(
        {
            "task_scope": "weight_modification",
            "objective": "scale",
            "inputs": [{"name": "base", "artifact_id": base_artifact}],
            "operations": [{"op": "w.arithmetic.scale", "params": {"factor": 1.5}}],
        }
    )
    first = service.run(job["id"])
    with pytest.raises(Exception):
        service.run(job["id"])  # terminal state cannot be re-run programmatically
    # Re-submitting the identical request is content-addressed to the same artifact.
    twice = service.submit(
        {
            "task_scope": "weight_modification",
            "objective": "scale",
            "inputs": [{"name": "base", "artifact_id": base_artifact}],
            "operations": [{"op": "w.arithmetic.scale", "params": {"factor": 1.5}}],
        }
    )
    second = service.run(twice["id"])
    assert second["result"]["output_artifact_id"] == first["result"]["output_artifact_id"]
    assert len([a for a in service.store.list() if a["artifact_id"] == first["result"]["output_artifact_id"]]) == 1


def test_cancel_before_run(service: JobService, base_artifact: str):
    job = service.submit(
        {
            "task_scope": "weight_modification",
            "objective": "scale",
            "inputs": [{"name": "base", "artifact_id": base_artifact}],
            "operations": [{"op": "w.arithmetic.scale", "params": {"factor": 3.0}}],
        }
    )
    cancelled = service.cancel(job["id"])
    assert cancelled["state"] == "CANCELLED"


def test_job_rejects_prohibited_policy_and_unknown_scope(service: JobService):
    with pytest.raises(Exception):
        service.submit({"task_scope": "weight_modification", "objective": "x", "policy": "GRADIENT_TRAINING"})
    with pytest.raises(Exception):
        service.submit({"task_scope": "fine_tuning", "objective": "x"})


def test_job_rejects_prohibited_operation(service: JobService, base_artifact: str):
    with pytest.raises(Exception) as exc:
        service.submit(
            {
                "task_scope": "weight_modification",
                "objective": "finetune the weights with adam",
                "inputs": [{"name": "base", "artifact_id": base_artifact}],
                "operations": [{"op": "w.train.finetune", "params": {"lr": 0.1}}],
            }
        ).get("id")
    assert "not registered" in str(exc.value) or "prohibit" in str(exc.value).lower()


def test_import_safetensors_and_analyze(service: JobService, tmp_path: Path):
    path = tmp_path / "tiny.safetensors"
    rng = np.random.default_rng(0)
    write_safetensors(
        path,
        {
            "model.embed_tokens.weight": rng.standard_normal((16, 8)).astype(np.float32),
            "model.layers.0.q_proj.weight": rng.standard_normal((8, 8)).astype(np.float32),
            "model.layers.0.mlp_up.weight": rng.standard_normal((16, 8)).astype(np.float32),
            "model.norm.weight": np.ones(8, dtype=np.float32),
        },
        metadata={"family": "transformer_decoder"},
    )
    record = service.import_artifact(path, family="transformer_decoder")
    assert record.num_tensors == 4
    analysis = service.wie.inspect_package(service.get_package(record.artifact_id)).to_dict()
    assert analysis["ok"] is True
    assert analysis["summary"]["numel"] == 16 * 8 + 64 + 128 + 8
    assert analysis["family_guess"]["family"] in ("transformer_decoder", "unknown")


def test_architecture_job_creates_valid_model(service: JobService):
    job = service.submit(
        {
            "task_scope": "architecture_synthesis",
            "objective": "create a small MLP",
            "options": {"family": "mlp", "input_dim": 4, "hidden_dims": [8], "output_dim": 2},
            "seed": 1,
        }
    )
    done = service.run(job["id"])
    assert done["state"] == "SUCCEEDED", done
    artifact_id = done["result"]["output_artifact_id"]
    manifest = service.artifact_manifest(artifact_id)
    assert manifest["metadata"]["graph"]["nodes"]
    assert service.verify_artifact(artifact_id)["ok"] is True


# --------------------------------------------------------------------------- #
# SDK
# --------------------------------------------------------------------------- #


def test_sdk_generate_modify_evaluate(tmp_path: Path):
    client = Uzen(root=tmp_path / "sdk")
    spec = dict(input_dim=4, hidden_dims=[8], output_dim=3)
    generated = client.generate(family="mlp", spec=spec, seed=0)
    artifact_id = generated["artifact_id"]
    assert artifact_id
    assert client.evaluate(artifact_id, family="mlp", spec=spec)["ok"] is True

    modified = client.modify(artifact_id, "svd", params={"rank": 2})
    assert modified["architecture_preserved"] is True
    out = modified["artifact_id"]
    assert out != artifact_id
    report = client.evaluate(out, reference_artifact_id=artifact_id, family="mlp", spec=spec)
    assert report["regression"]["architecture_unchanged"] is True
    assert client.verify_artifact(out)["ok"] is True


def test_sdk_inspect_rejects_pickle(tmp_path: Path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"\x80\x02}q\x00.")
    with pytest.raises(Exception) as exc:
        Uzen(root=tmp_path / "sdk2").inspect(path)
    assert "format" in str(exc.value).lower() or "pickle" in str(exc.value).lower()


# --------------------------------------------------------------------------- #
# REST API
# --------------------------------------------------------------------------- #


@pytest.fixture()
def api_client(service: JobService):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from uzen.interfaces.api.app import create_app

    app = create_app(service=service)
    return fastapi_testclient.TestClient(app)


def test_api_meta_and_catalogue(api_client):
    assert api_client.get("/health").json()["policy"] == "STRICT_PLUS"
    ops = api_client.get("/v1/operations").json()
    assert ops["count"] >= 40
    policy = api_client.get("/v1/policies/strict-plus").json()
    assert any("backpropagation" in s for s in policy["statements"])
    families = api_client.get("/v1/architectures/families").json()["families"]
    assert {f["family"] for f in families} >= {"mlp", "transformer_decoder"}


def test_api_generate_and_modify(api_client):
    generated = api_client.post(
        "/v1/weights/generate",
        json={"family": "mlp", "spec": {"input_dim": 4, "hidden_dims": [8], "output_dim": 2}, "seed": 0},
    )
    assert generated.status_code == 200, generated.text
    artifact_id = generated.json()["artifact_id"]
    modified = api_client.post(
        "/v1/weights/modify",
        json={"artifact_id": artifact_id, "method": "scale", "params": {"factor": 0.5}},
    )
    assert modified.status_code == 200, modified.text
    out = modified.json()["artifact_id"]
    assert modified.json()["architecture_preserved"] is True
    assert api_client.get(f"/v1/artifacts/{out}/verify").json()["ok"] is True
    lineage = api_client.get(f"/v1/artifacts/{out}/lineage").json()
    assert lineage["artifact_id"] == out


def test_api_error_shape_is_typed(api_client):
    response = api_client.post("/v1/weights/modify", json={"artifact_id": "0" * 64, "method": "scale"})
    assert response.status_code in (400, 404, 409)
    body = response.json()["detail"]
    assert "code" in body and "message" in body


def test_api_job_lifecycle(api_client, base_artifact: str):
    created = api_client.post(
        "/v1/jobs",
        json={
            "task_scope": "weight_modification",
            "objective": "scale",
            "inputs": [{"name": "base", "artifact_id": base_artifact}],
            "operations": [{"op": "w.arithmetic.scale", "params": {"factor": 0.5}}],
        },
    )
    assert created.status_code == 201, created.text
    job_id = created.json()["id"]
    events = api_client.get(f"/v1/jobs/{job_id}/events").json()["events"]
    assert events


def test_api_worker_registration_and_bad_secret(api_client):
    registered = api_client.post(
        "/v1/workers/register", json={"worker_id": "w-test", "capabilities": {"cpu_count": 2}}
    )
    assert registered.status_code == 201
    secret = registered.json()["secret"]
    good = api_client.post("/v1/workers/heartbeat", json={"worker_id": "w-test", "secret": secret})
    assert good.status_code == 200
    bad = api_client.post("/v1/workers/heartbeat", json={"worker_id": "w-test", "secret": "wrong"})
    assert bad.status_code == 401
    unknown = api_client.post("/v1/workers/heartbeat", json={"worker_id": "nobody", "secret": secret})
    assert unknown.status_code == 401


# --------------------------------------------------------------------------- #
# MCP
# --------------------------------------------------------------------------- #


def test_mcp_initialize_list_and_call(tmp_path: Path):
    from uzen.interfaces.mcp.server import MCPServer

    client = Uzen(root=tmp_path / "mcp")
    generated = client.generate(family="mlp", spec=dict(input_dim=4, hidden_dims=[8], output_dim=2), seed=0)
    server = MCPServer(root=tmp_path / "mcp")

    init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["serverInfo"]["name"] == "uzen"
    assert init["result"]["protocolVersion"]

    tools = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"uzen.inspect_model", "uzen.modify_weights", "uzen.evaluate_candidate"} <= names

    called = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "uzen.evaluate_candidate", "arguments": {"artifact_id": generated["artifact_id"]}},
        }
    )
    payload = json.loads(called["result"]["content"][0]["text"])
    assert payload["ok"] is True
    assert payload["structural"]["num_tensors"] >= 1

    unknown = server.handle(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "uzen.nope", "arguments": {}}}
    )
    assert "error" in unknown

    bad_method = server.handle({"jsonrpc": "2.0", "id": 5, "method": "does/not/exist"})
    assert bad_method["error"]["code"] == -32601


def test_mcp_tool_error_is_reported_not_hidden(tmp_path: Path):
    from uzen.interfaces.mcp.server import MCPServer

    server = MCPServer(root=tmp_path / "mcp2")
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "uzen.analyze_weights", "arguments": {"artifact_id": "f" * 64}},
        }
    )
    assert response["result"]["isError"] is True
    error = json.loads(response["result"]["content"][0]["text"])
    assert error["code"]


# --------------------------------------------------------------------------- #
# worker daemon
# --------------------------------------------------------------------------- #


def test_local_worker_executes_queued_job(tmp_path: Path):
    from uzen.workers.daemon import WorkerDaemon

    daemon = WorkerDaemon(root=tmp_path / "worker", worker_id="w-local")
    definition = build_architecture("mlp", dict(input_dim=4, hidden_dims=[8], output_dim=2))
    record = daemon.service.store.put(definition.build_package(seed=0), operation="init")
    job = daemon.service.submit(
        {
            "task_scope": "weight_modification",
            "objective": "scale",
            "inputs": [{"name": "base", "artifact_id": record.artifact_id}],
            "operations": [{"op": "w.arithmetic.scale", "params": {"factor": 0.1}}],
        }
    )
    outcome = daemon.run_once()
    assert outcome["executed"] == job["id"]
    assert outcome["state"] == "SUCCEEDED"
    assert daemon.run_once()["executed"] is None


def test_worker_capabilities_are_detected_not_hardcoded():
    from uzen.workers.protocol.messages import detect_capabilities

    caps = detect_capabilities()
    assert caps["python"]
    assert caps["cpu_count"] >= 1
    assert "w.arithmetic.scale" in caps["supported_operations"]
    assert caps["protocol_version"] == "uzen.worker/1"
    assert caps["strict_plus"] is True
