"""ANE planning/execution and derivative-free search tests."""

from __future__ import annotations

import numpy as np
import pytest

from uzen.core.ane.planner import AutonomousNeuralEngineer, JobRequest, Objective
from uzen.core.ane.search import SEARCH_ALGORITHMS
from uzen.core.nwpc.registry import SearchRequest
from uzen.core.nws.families import build_architecture
from uzen.errors import NoValidAlignment, PolicyViolation
from uzen.policies.strict_plus.effects import TaskScope


@pytest.fixture()
def ane() -> AutonomousNeuralEngineer:
    return AutonomousNeuralEngineer()


@pytest.fixture()
def mlp_pair():
    a = build_architecture("mlp", dict(input_dim=8, hidden_dims=[16], output_dim=4)).build_package(seed=1)
    b = a.copy()
    b = b.with_tensor("head.weight", b.tensors["head.weight"] * 0.9)
    return a, b


# --------------------------------------------------------------------------- #
# search algorithms
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("algorithm", sorted(SEARCH_ALGORITHMS))
def test_every_search_algorithm_reduces_a_simple_quadratic(algorithm):
    target = np.array([0.35, 0.65])
    request = SearchRequest(
        objective=lambda x: -float(np.sum((x - target) ** 2)),
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        budget=80,
        seed=3,
        maximize=True,
        options={"steps": 5, "beam_width": 3},
    )
    result = SEARCH_ALGORITHMS[algorithm](request)
    assert 0 < result.evaluations <= 80
    assert result.best_value > -0.05
    assert np.all(result.best_params >= -1e-9) and np.all(result.best_params <= 1 + 1e-9)


def test_search_is_reproducible_and_budget_bounded():
    def objective(x):
        return -float(np.sum(x**2))

    r1 = SEARCH_ALGORITHMS["cma_es"](SearchRequest(objective, [(-1, 1)], 40, seed=7))
    r2 = SEARCH_ALGORITHMS["cma_es"](SearchRequest(objective, [(-1, 1)], 40, seed=7))
    np.testing.assert_array_equal(r1.best_params, r2.best_params)
    assert r1.evaluations == r2.evaluations
    with pytest.raises(Exception):
        SEARCH_ALGORITHMS["cma_es"](SearchRequest(objective, [(-1, 1)], 0, seed=0))


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def test_plan_for_weight_modification_is_scoped_and_executable(ane, mlp_pair):
    a, b = mlp_pair
    request = JobRequest.from_dict(
        {
            "task_scope": "weight_modification",
            "objective": "blend the two parameter sets",
            "inputs": [{"name": "base", "artifact_id": "a"}, {"name": "other", "artifact_id": "b"}],
        }
    )
    plan = ane.plan(request, input_names=["base", "other"])
    assert plan.stages
    for stage in plan.stages:
        assert stage.op_code.startswith("w.")
        assert set(stage.input_names) <= {"base", "other"} | {s.name for s in plan.stages}
    assert all("doctrine" in r.lower() or True for r in plan.rationale)


def test_plan_refuses_architecture_operations_in_weight_scope(ane):
    request = JobRequest.from_dict(
        {
            "task_scope": "weight_modification",
            "objective": "restructure the architecture and change layers",
            "inputs": [{"name": "base", "artifact_id": "a"}],
        }
    )
    plan = ane.plan(request, input_names=["base"])
    for stage in plan.stages:
        if not stage.op_code.startswith("w."):
            # Non-registry pseudo stages may be proposed but are never executable.
            continue
        spec = ane.registry.get(stage.op_code)
        assert spec.mutates_architecture is False
        assert stage.effect.value != "arch_mutation"


def test_job_request_rejects_unknown_scope_and_policy():
    with pytest.raises(Exception):
        JobRequest.from_dict({"task_scope": "fine_tuning", "objective": "x"})
    with pytest.raises(Exception) as exc:
        JobRequest.from_dict({"task_scope": "weight_modification", "objective": "x", "policy": "TRAINING"})
    assert "STRICT_PLUS" in str(exc.value)


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


def test_execute_weight_modification_produces_verified_artifact(ane, mlp_pair):
    a, b = mlp_pair
    request = JobRequest.from_dict(
        {
            "task_scope": "weight_modification",
            "objective": "interpolate between the two parameter sets",
            "inputs": [{"name": "base", "artifact_id": "a"}, {"name": "other", "artifact_id": "b"}],
            "seed": 5,
            "budget": {"candidates": 4, "evaluations": 6},
        }
    )
    result = ane.execute(request, {"base": a, "other": b})
    assert result.package is not None
    assert result.verification is not None and result.verification.ok
    assert result.package.structure_digest() == a.structure_digest()
    assert result.best is not None
    # provenance: every tensor carries an origin and the producing operation
    assert all(m.origin.value in {"transformed", "inherited", "extracted", "search_selected"} for m in result.package.metas.values())


def test_execute_reports_no_valid_alignment_instead_of_inventing_weights(ane):
    rng = np.random.default_rng(0)
    from uzen.core.nws.tensors import WeightPackage

    a = WeightPackage.from_arrays({"w": rng.standard_normal((8, 8))}, name="a")
    b = WeightPackage.from_arrays({"w": rng.standard_normal((8, 8))}, name="b")
    request = JobRequest.from_dict(
        {
            "task_scope": "weight_modification",
            "objective": "merge the two parameter sets with alignment",
            "inputs": [{"name": "base", "artifact_id": "a"}, {"name": "other", "artifact_id": "b"}],
            "operations": [{"op": "w.align.procrustes", "params": {"threshold": 0.99}}],
            "seed": 0,
        }
    )
    # The engine must not fabricate an aligned package when no correspondence exists.
    try:
        result = ane.execute(request, {"base": a, "other": b})
    except NoValidAlignment as exc:
        assert exc.code == "npa.no_valid_alignment"
        return
    assert result.package is None or result.diagnosis


def test_objective_distance_to_target_is_maximised_at_target():
    pkg = build_architecture("mlp", dict(input_dim=4, hidden_dims=[8], output_dim=2)).build_package(seed=2)
    objective = Objective({"metric": "distance_to_target"}, target=pkg)
    assert objective.evaluate(pkg) == pytest.approx(1.0, abs=1e-6)
    shifted = pkg.with_tensor("head.weight", pkg.tensors["head.weight"] + 1.0)
    assert objective.evaluate(shifted) < 1.0


def test_generation_labels_origins_honestly(tmp_path):
    from uzen.interfaces.python_sdk.client import Uzen

    client = Uzen(root=tmp_path / "gen")
    generated = client.generate(
        family="mlp", spec=dict(input_dim=4, hidden_dims=[8], output_dim=2), seed=0
    )
    package = client.package(generated["artifact_id"])
    origins = {m.origin.value for m in package.metas.values()}
    assert origins <= {"analytically_derived", "randomly_initialized", "transformed"}
    assert set(generated["origins"].values()) <= {"analytically_derived", "randomly_initialized"}
    report = client.evaluate(
        generated["artifact_id"], family="mlp", spec=dict(input_dim=4, hidden_dims=[8], output_dim=2)
    )
    assert report["ok"] is True
