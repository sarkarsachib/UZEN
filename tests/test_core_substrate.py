"""Substrate tests: safetensors codec, artifact store, provenance, STRICT_PLUS."""

from __future__ import annotations

import json

import numpy as np
import pytest

from uzen.core.nws.store import ArtifactStore
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import (
    ArtifactImmutable,
    OperationNotRegistered,
    PolicyViolation,
    ScopeViolation,
    UnsupportedFormat,
)
from uzen.infrastructure.storage.safetensors_io import (
    read_header,
    read_safetensors,
    write_safetensors,
)
from uzen.policies.strict_plus.effects import EngineEffect, Maturity, Origin, TaskScope
from uzen.policies.strict_plus.engine import AuditLog, StrictPlusEngine, default_engine


# --------------------------------------------------------------------------- #
# safetensors
# --------------------------------------------------------------------------- #


def test_safetensors_roundtrip_all_supported_dtypes(tmp_path):
    path = tmp_path / "t.safetensors"
    arrays = {
        "f32": np.arange(6, dtype=np.float32).reshape(2, 3),
        "f64": np.linspace(-1, 1, 5),
        "i64": np.array([1, 2, 3], dtype=np.int64),
        "u8": np.array([255, 0], dtype=np.uint8),
        "bool": np.array([True, False, True]),
        "f16": np.array([0.5, 0.25], dtype=np.float16),
    }
    write_safetensors(path, arrays, metadata={"family": "unit-test"})
    loaded, header = read_safetensors(path)
    assert set(loaded) == set(arrays)
    for name, arr in arrays.items():
        np.testing.assert_array_equal(loaded[name], arr)
        assert loaded[name].dtype == arr.dtype
    assert header.metadata["family"] == "unit-test"


def test_safetensors_header_only_read_does_not_need_data(tmp_path):
    path = tmp_path / "t.safetensors"
    write_safetensors(path, {"a": np.zeros((4, 4), dtype=np.float32)})
    header = read_header(path)
    assert header.num_tensors == 1
    assert header.entries["a"].shape == (4, 4)
    assert header.total_numel() == 16


def test_safetensors_rejects_truncated_and_corrupt_files(tmp_path):
    path = tmp_path / "t.safetensors"
    write_safetensors(path, {"a": np.zeros(8, dtype=np.float32)})
    data = path.read_bytes()
    path.write_bytes(data[:-4])
    with pytest.raises(Exception):
        read_header(path)
    corrupt = tmp_path / "c.safetensors"
    corrupt.write_bytes(b"\xff" * 32)
    with pytest.raises(Exception):
        read_header(corrupt)


def test_bfloat16_roundtrip_is_lossless(tmp_path):
    from uzen.core.nws.dtypes import bf16_decode, bf16_encode
    from uzen.util.linalg import bfloat16_bits_to_float32, float32_to_bfloat16_bits

    values = np.array([0.0, 1.0, -2.5, 1.5], dtype=np.float32)
    bits = float32_to_bfloat16_bits(values)
    restored = bfloat16_bits_to_float32(bits)
    # bfloat16 has 8 mantissa bits: these values are exactly representable
    np.testing.assert_array_equal(restored, values)
    with pytest.raises(Exception):
        bf16_encode(np.array([0.1234567891], dtype=np.float32))


def test_unsafe_checkpoint_format_is_refused(tmp_path):
    from uzen.core.wie.inspector import WeightIntelligenceEngine

    unsafe = tmp_path / "model.pt"
    unsafe.write_bytes(b"\x80\x02}\x00.")
    wie = WeightIntelligenceEngine()
    with pytest.raises(UnsupportedFormat) as exc:
        wie.inspect_file(unsafe)
    assert "unpickle" in str(exc.value)
    assert "safetensors" in str(exc.value)


# --------------------------------------------------------------------------- #
# artifact store / provenance
# --------------------------------------------------------------------------- #


def test_store_is_content_addressed_and_idempotent(store: ArtifactStore, mlp_package):
    first = store.put(mlp_package, operation="init")
    second = store.put(mlp_package.copy(), operation="init")
    assert first.artifact_id == second.artifact_id
    assert len([i for i in store.list() if not i.get("corrupt")]) == 1


def test_store_roundtrip_preserves_tensor_digest(store: ArtifactStore, mlp_package):
    record = store.put(mlp_package, operation="init")
    loaded = store.get(record.artifact_id)
    assert loaded.fingerprint() == mlp_package.fingerprint()
    assert loaded.structure_digest() == mlp_package.structure_digest()


def test_store_verification_detects_payload_tampering(store: ArtifactStore, mlp_package):
    record = store.put(mlp_package, operation="init")
    assert store.verify(record.artifact_id)["ok"] is True
    payload = record.path / "tensors.safetensors"
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 0xFF
    payload.write_bytes(bytes(raw))
    result = store.verify(record.artifact_id)
    assert result["ok"] is False
    assert any("digest" in p for p in result["problems"])


def test_store_is_immutable_once_published(store: ArtifactStore, mlp_package):
    record = store.put(mlp_package, operation="init", metadata={"first": True})
    import json

    manifest_before = (record.path / "manifest.json").read_text()
    # A retry with the same content/operation is idempotent and must not rewrite history.
    again = store.put(mlp_package, operation="init", metadata={"first": True})
    assert again.artifact_id == record.artifact_id
    assert (record.path / "manifest.json").read_text() == manifest_before
    with pytest.raises(ArtifactImmutable):
        store.put(mlp_package, operation="init", metadata={"second": True})


def test_provenance_lineage_records_parents(store: ArtifactStore, mlp_package):
    from uzen.core.nwpc.registry import default_registry

    parent = store.put(mlp_package, operation="init")
    scaled = default_registry().invoke(
        "w.arithmetic.scale",
        [store.get(parent.artifact_id)],
        {"factor": 0.5},
        scope=TaskScope.WEIGHT_MODIFICATION,
        seed=1,
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
    chain = graph.path_to(child.artifact_id)
    assert chain == [parent.artifact_id, child.artifact_id]
    assert store.verify(child.artifact_id)["ok"] is True


# --------------------------------------------------------------------------- #
# STRICT_PLUS
# --------------------------------------------------------------------------- #


def test_policy_denies_gradient_effects_unconditionally():
    engine = StrictPlusEngine(audit=AuditLog())
    for effect in (EngineEffect.GRADIENT_UPDATE, EngineEffect.OPTIMIZER_STATE, EngineEffect.DATA_TRAINING):
        decision = engine.evaluate(effect, "test.op", scope=TaskScope.WEIGHT_OPTIMIZATION)
        assert decision.allowed is False
        assert "STRICT_PLUS prohibits" in decision.reason
        with pytest.raises(PolicyViolation):
            engine.authorize(effect, "test.op", scope=TaskScope.WEIGHT_OPTIMIZATION)


def test_policy_scope_is_not_silently_expanded():
    engine = StrictPlusEngine(audit=AuditLog())
    decision = engine.evaluate(EngineEffect.ARCH_MUTATION, "nase.synthesize", scope=TaskScope.WEIGHT_MODIFICATION)
    assert decision.allowed is False
    assert "outside the authorized scope" in decision.reason
    ok = engine.evaluate(EngineEffect.ARCH_MUTATION, "nase.synthesize", scope=TaskScope.MODEL_MODIFICATION)
    assert ok.allowed is True


def test_policy_rejects_plan_with_prohibited_stage():
    engine = StrictPlusEngine(audit=AuditLog())
    with pytest.raises(PolicyViolation):
        engine.authorize_plan(
            [
                ("align", EngineEffect.PARAM_TRANSFORM, Maturity.IMPLEMENTED),
                ("finetune", EngineEffect.GRADIENT_UPDATE, Maturity.IMPLEMENTED),
            ],
            scope=TaskScope.WEIGHT_MODIFICATION,
        )
    decisions = engine.audit.entries()
    rejected = [e for e in decisions if "gradient_update" in json.dumps(e)]
    assert rejected, decisions


def test_policy_rejects_planned_capability_as_unexecutable():
    engine = StrictPlusEngine(audit=AuditLog())
    decision = engine.evaluate(
        EngineEffect.PARAM_TRANSFORM, "cuda.kernel", scope=TaskScope.WEIGHT_MODIFICATION, maturity=Maturity.PLANNED
    )
    assert decision.allowed is False
    assert "not executable" in decision.reason


def test_audit_log_hash_chain_detects_tampering():
    log = AuditLog()
    log.append("a", {"x": 1})
    log.append("b", {"y": 2})
    ok, reason = log.verify()
    assert ok, reason
    tampered = log.entries()
    tampered[0]["payload"]["x"] = 999
    log._entries[0] = tampered[0]
    ok, reason = log.verify()
    assert not ok


def test_registry_rejects_unregistered_operation(mlp_package):
    from uzen.core.nwpc.registry import default_registry

    registry = default_registry()
    with pytest.raises(OperationNotRegistered):
        registry.invoke("w.train.finetune", [mlp_package], {}, scope=TaskScope.WEIGHT_MODIFICATION)


def test_mutation_guard_blocks_grad_tracking_writes():
    engine = StrictPlusEngine(audit=AuditLog())
    with pytest.raises(PolicyViolation):
        engine.guard_mutation("tensor", gradient_active=True, scope=TaskScope.WEIGHT_MODIFICATION)
    engine.guard_mutation("tensor", gradient_active=False, scope=TaskScope.WEIGHT_MODIFICATION)


def test_default_policy_description_is_explicit():
    description = default_engine().describe()
    statements = " ".join(description["statements"]).lower()
    assert "does not perform conventional model training" in statements
    assert "backpropagation" in statements
    assert "gradient-based parameter updates" in statements
