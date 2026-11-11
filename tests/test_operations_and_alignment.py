"""Operation and alignment tests on real tensors."""

from __future__ import annotations

import numpy as np
import pytest

from uzen.core.npa.aligner import ParameterAligner, align_packages
from uzen.core.nwpc.registry import default_registry
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import (
    InvalidOperationParameters,
    NoValidAlignment,
    ShapeMismatch,
    TensorInvalid,
)
from uzen.policies.strict_plus.effects import TaskScope
from uzen.util.linalg import linear_sum_assignment, orthogonal_procrustes


def invoke(code, inputs, params, scope=TaskScope.WEIGHT_MODIFICATION, seed=0):
    return default_registry().invoke(code, inputs, params, scope=scope, seed=seed)


# --------------------------------------------------------------------------- #
# arithmetic
# --------------------------------------------------------------------------- #


def test_linear_interpolation_is_numerically_exact():
    a = WeightPackage.from_arrays({"w": np.full((3, 2), 1.0, dtype=np.float32)})
    b = WeightPackage.from_arrays({"w": np.full((3, 2), 5.0, dtype=np.float32)})
    result = invoke("w.arithmetic.linear_interpolate", [a, b], {"t": 0.25})
    np.testing.assert_allclose(result.output.tensors["w"], np.full((3, 2), 2.0, dtype=np.float32))


def test_extrapolation_is_numerically_exact():
    a = WeightPackage.from_arrays({"w": np.zeros((2, 2), dtype=np.float32)})
    b = WeightPackage.from_arrays({"w": np.ones((2, 2), dtype=np.float32)})
    result = invoke("w.arithmetic.extrapolate", [a, b], {"factor": 2.0})
    np.testing.assert_allclose(result.output.tensors["w"], np.full((2, 2), 2.0, dtype=np.float32))
    assert result.output.metas["w"].origin.value == "transformed"


def test_slerp_preserves_norm_and_direction_midpoint():
    a = WeightPackage.from_arrays({"w": np.array([[1.0, 0.0]], dtype=np.float32)})
    b = WeightPackage.from_arrays({"w": np.array([[0.0, 1.0]], dtype=np.float32)})
    mid = invoke("w.arithmetic.slerp", [a, b], {"t": 0.5, "preserve_norm": True})
    vec = mid.output.tensors["w"].reshape(-1)
    np.testing.assert_allclose(vec, np.array([np.sqrt(0.5), np.sqrt(0.5)]), atol=1e-6)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)


def test_shape_mismatch_is_rejected_explicitly():
    a = WeightPackage.from_arrays({"w": np.zeros((2, 2), dtype=np.float32)})
    b = WeightPackage.from_arrays({"w": np.zeros((3, 2), dtype=np.float32)})
    with pytest.raises(ShapeMismatch) as exc:
        invoke("w.arithmetic.linear_interpolate", [a, b], {"t": 0.5})
    assert "inconsistent shapes" in str(exc.value)


def test_missing_parameter_is_rejected():
    a = WeightPackage.from_arrays({"w": np.zeros((2, 2), dtype=np.float32)})
    b = WeightPackage.from_arrays({"other": np.zeros((2, 2), dtype=np.float32)})
    with pytest.raises(ShapeMismatch):
        invoke("w.arithmetic.linear_interpolate", [a, b], {"t": 0.5})
    # explicit partial mode operates on the shared intersection, and refuses when empty
    shared = WeightPackage.from_arrays({"w": np.zeros((2, 2), dtype=np.float32)})
    disjoint = WeightPackage.from_arrays({"z": np.zeros((2, 2), dtype=np.float32)})
    out = invoke("w.arithmetic.linear_interpolate", [a, shared], {"t": 0.5, "partial": True})
    assert out.output.names() == ["w"]
    with pytest.raises(ShapeMismatch):
        invoke("w.arithmetic.linear_interpolate", [a, disjoint], {"t": 0.5, "partial": True})


def test_unknown_parameter_is_rejected(mlp_package):
    with pytest.raises(InvalidOperationParameters) as exc:
        invoke("w.arithmetic.scale", [mlp_package], {"factor": 0.5, "nonsense": 1})
    assert "does not accept parameters" in str(exc.value)


def test_weight_only_modification_preserves_architecture(mlp_package):
    out = invoke("w.arithmetic.scale", [mlp_package], {"factor": 0.75}).output
    assert out.structure_digest() == mlp_package.structure_digest()
    assert out.fingerprint() != mlp_package.fingerprint()
    assert set(out.names()) == set(mlp_package.names())


def test_source_is_immutable_under_operations(mlp_package):
    before = mlp_package.fingerprint()
    copy = mlp_package.copy()
    invoke("w.arithmetic.scale", [mlp_package], {"factor": 2.0})
    assert mlp_package.fingerprint() == before
    assert copy.fingerprint() == before


# --------------------------------------------------------------------------- #
# decomposition
# --------------------------------------------------------------------------- #


def test_svd_rank_full_rank_reconstructs_exactly():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 6))
    pkg = WeightPackage.from_arrays({"w": w}, name="m")
    rank = np.linalg.matrix_rank(w)
    result = invoke("w.decomp.svd_truncate", [pkg], {"rank": rank})
    np.testing.assert_allclose(result.output.tensors["w"], w, atol=1e-6)
    assert result.details["mean_energy_captured"] > 0.999


def test_svd_truncation_reports_energy_loss():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((16, 16))
    pkg = WeightPackage.from_arrays({"w": w})
    result = invoke("w.decomp.svd_truncate", [pkg], {"rank": 1})
    assert 0.0 < result.details["mean_energy_captured"] < 1.0
    assert np.allclose(np.linalg.svd(w, compute_uv=False)[:1], np.linalg.svd(result.output.tensors["w"], compute_uv=False)[:1], atol=1e-6)


def test_orthogonal_transform_preserves_singular_values():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((6, 4))
    pkg = WeightPackage.from_arrays({"w": w})
    out = invoke("w.decomp.orthogonal_transform", [pkg], {"method": "qr", "side": "right"}).output
    s_before = np.linalg.svd(w, compute_uv=False)
    s_after = np.linalg.svd(out.tensors["w"], compute_uv=False)
    np.testing.assert_allclose(s_before, s_after, atol=1e-5)


def test_sparsify_hits_requested_density_and_keeps_shape():
    rng = np.random.default_rng(3)
    pkg = WeightPackage.from_arrays({"w": rng.standard_normal((10, 10))})
    out = invoke("w.decomp.sparsify", [pkg], {"fraction": 0.3}).output
    zeros = np.mean(np.abs(out.tensors["w"]) < 1e-12)
    assert 0.6 <= zeros <= 0.75  # ~70% zeroed
    assert out.tensors["w"].shape == (10, 10)


# --------------------------------------------------------------------------- #
# merging
# --------------------------------------------------------------------------- #


def test_task_arithmetic_matches_definition():
    base = WeightPackage.from_arrays({"w": np.zeros((2, 2), dtype=np.float32)})
    m1 = WeightPackage.from_arrays({"w": np.full((2, 2), 1.0, dtype=np.float32)})
    m2 = WeightPackage.from_arrays({"w": np.full((2, 2), 3.0, dtype=np.float32)})
    result = invoke("w.merge.task_arithmetic", [base, m1, m2], {"weights": [1.0, 0.5]})
    np.testing.assert_allclose(result.output.tensors["w"], np.full((2, 2), 2.5, dtype=np.float32))


def test_soup_is_weighted_average():
    p1 = WeightPackage.from_arrays({"w": np.zeros(4, dtype=np.float32)})
    p2 = WeightPackage.from_arrays({"w": np.full(4, 2.0, dtype=np.float32)})
    result = invoke("w.merge.soup", [p1, p2], {"weights": [1.0, 1.0]})
    np.testing.assert_allclose(result.output.tensors["w"], np.full(4, 1.0, dtype=np.float32))


def test_ties_reduces_conflicts_and_is_deterministic():
    rng = np.random.default_rng(4)
    base = WeightPackage.from_arrays({"w": np.zeros((8, 8), dtype=np.float32)})
    m1 = WeightPackage.from_arrays({"w": rng.standard_normal((8, 8)).astype(np.float32)})
    m2 = WeightPackage.from_arrays({"w": rng.standard_normal((8, 8)).astype(np.float32)})
    a = invoke("w.merge.ties", [base, m1, m2], {"density": 0.3, "weights": [1.0, 1.0]}, seed=5)
    b = invoke("w.merge.ties", [base, m1, m2], {"density": 0.3, "weights": [1.0, 1.0]}, seed=5)
    np.testing.assert_array_equal(a.output.tensors["w"], b.output.tensors["w"])
    assert a.details["statistics"]["agree_fraction"][0] <= 1.0


def test_dare_is_reproducible_for_fixed_seed():
    rng = np.random.default_rng(5)
    base = WeightPackage.from_arrays({"w": np.zeros((6, 6), dtype=np.float32)})
    m1 = WeightPackage.from_arrays({"w": rng.standard_normal((6, 6)).astype(np.float32)})
    a = invoke("w.merge.dare", [base, m1], {"density": 0.5}, seed=42)
    b = invoke("w.merge.dare", [base, m1], {"density": 0.5}, seed=42)
    c = invoke("w.merge.dare", [base, m1], {"density": 0.5}, seed=43)
    np.testing.assert_array_equal(a.output.tensors["w"], b.output.tensors["w"])
    assert not np.array_equal(a.output.tensors["w"], c.output.tensors["w"])


# --------------------------------------------------------------------------- #
# structural / generation / reconstruction
# --------------------------------------------------------------------------- #


def test_parameter_tying_produces_identical_tensors():
    pkg = WeightPackage.from_arrays(
        {"a.weight": np.zeros((2, 2), dtype=np.float32), "b.weight": np.ones((2, 2), dtype=np.float32)}
    )
    out = invoke("w.structural.tie", [pkg], {"names": ["a.weight", "b.weight"], "strategy": "mean"}).output
    np.testing.assert_allclose(out.tensors["a.weight"], out.tensors["b.weight"])
    np.testing.assert_allclose(out.tensors["a.weight"], np.full((2, 2), 0.5, dtype=np.float32))


def test_structured_random_generation_is_labelled_and_seeded(mlp_package):
    result = invoke(
        "w.gen.structured",
        [mlp_package],
        {"distribution": "normal"},
        scope=TaskScope.WEIGHT_GENERATION,
        seed=123,
    )
    assert all(m.origin.value == "randomly_initialized" for m in result.output.metas.values())
    assert "STRUCTURED RANDOM" in result.details["note"]
    again = invoke(
        "w.gen.structured", [mlp_package], {"distribution": "normal"}, scope=TaskScope.WEIGHT_GENERATION, seed=123
    )
    np.testing.assert_array_equal(result.output.tensors["head.weight"], again.output.tensors["head.weight"])


def test_orthogonal_generation_has_orthonormal_columns(mlp_package):
    result = invoke(
        "w.gen.structured", [mlp_package], {"distribution": "orthogonal"}, scope=TaskScope.WEIGHT_GENERATION, seed=2
    )
    w = result.output.tensors["fc0.weight"].astype(np.float64)
    gram = w.T @ w
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off_diagonal)) < 1e-4


def test_reconstruction_fills_only_from_donors_or_fails(mlp_package):
    partial = mlp_package.select("head.weight", "head.bias")
    donor = mlp_package
    result = invoke(
        "w.recon.fill_missing",
        [partial, donor],
        {"allow_random_fill": False},
        scope=TaskScope.WEIGHT_RECONSTRUCTION,
    )
    # fill_missing completes the *template*'s parameter set from donors
    assert set(result.output.names()) == set(partial.names())
    np.testing.assert_array_equal(result.output.tensors["head.weight"], donor.tensors["head.weight"])
    assert result.details["sources"]["head.weight"] == "donor[0]"

    empty_template = WeightPackage.from_arrays({"missing.weight": np.zeros((3, 3), dtype=np.float32)})
    with pytest.raises(InvalidOperationParameters) as exc:
        invoke(
            "w.recon.fill_missing",
            [empty_template, partial],
            {"allow_random_fill": False},
            scope=TaskScope.WEIGHT_RECONSTRUCTION,
        )
    assert "missing.weight" in str(exc.value)


def test_rename_rebinds_names_without_changing_values(mlp_package):
    out = invoke(
        "w.recon.rename",
        [mlp_package],
        {"mapping": {"head.weight": "classifier.weight"}},
        scope=TaskScope.WEIGHT_RECONSTRUCTION,
    ).output
    assert "classifier.weight" in out
    assert "head.weight" not in out
    np.testing.assert_array_equal(out.tensors["classifier.weight"], mlp_package.tensors["head.weight"])


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #


def test_hungarian_assignment_finds_optimal_matching():
    cost = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    rows, cols = linear_sum_assignment(cost)
    assert sorted(rows.tolist()) == [0, 1, 2]
    assert len(set(cols.tolist())) == 3
    # brute force optimum
    import itertools

    best = min(sum(cost[i, p[i]] for i in range(3)) for p in itertools.permutations(range(3)))
    assert np.isclose(cost[rows, cols].sum(), best)


def test_procrustes_recovers_known_orthogonal_transform(linear_pair):
    a, b, q = linear_pair
    r, residual, diag = orthogonal_procrustes(a.tensors["layer.weight"], b.tensors["layer.weight"])
    np.testing.assert_allclose(r, q, atol=1e-6)
    assert residual < 1e-6


def test_alignment_auto_recovers_rotation_and_improves_similarity(linear_pair):
    a, b, q = linear_pair
    aligned, report = align_packages(a, b, method="auto", threshold=0.9)
    assert report.ok, report.reasons
    np.testing.assert_allclose(aligned.tensors["layer.weight"], b.tensors["layer.weight"], atol=1e-5)
    assert report.to_dict()["semantic_equivalence_claimed"] is False


def test_permutation_alignment_recovers_row_permutation():
    rng = np.random.default_rng(9)
    w = rng.standard_normal((5, 4))
    perm = np.array([3, 0, 4, 1, 2])
    a = WeightPackage.from_arrays({"w": w})
    b = WeightPackage.from_arrays({"w": w[perm]})
    aligner = ParameterAligner(method="permutation", threshold=0.5)
    report = aligner.align(a, b)
    assert report.ok, report.reasons
    aligned = aligner.apply(a, report)
    np.testing.assert_allclose(aligned.tensors["w"], b.tensors["w"], atol=1e-6)


def test_no_valid_alignment_for_unrelated_parameterizations():
    rng = np.random.default_rng(10)
    a = WeightPackage.from_arrays({"w": rng.standard_normal((8, 8))})
    b = WeightPackage.from_arrays({"w": rng.standard_normal((8, 8))})
    aligner = ParameterAligner(method="procrustes", threshold=0.9)
    report = aligner.align(a, b)
    assert report.ok is False
    with pytest.raises(NoValidAlignment) as exc:
        report.require()
    assert exc.value.code == "npa.no_valid_alignment"


def test_alignment_refuses_shape_incompatible_tensors():
    a = WeightPackage.from_arrays({"w": np.zeros((4, 4)), "x": np.zeros(4)})
    b = WeightPackage.from_arrays({"w": np.zeros((5, 4)), "x": np.zeros(4)})
    aligner = ParameterAligner(method="procrustes", threshold=0.1)
    report = aligner.align(a, b)
    assert "w" in report.skipped
    assert report.ok is False


def test_activation_alignment_reports_cka(mlp, mlp_package):
    from uzen.core.npa.aligner import activation_alignment

    inputs = mlp.sample_inputs(batch=3, seq=4, seed=0)
    acts = activation_alignment(mlp_package, mlp_package.copy(), mlp.graph, inputs)
    assert acts
    for name, entry in acts.items():
        assert 0.0 <= entry.cka <= 1.0001
        assert entry.cka > 0.99


def test_alignment_op_raises_when_no_correspondence_exists():
    rng = np.random.default_rng(11)
    a = WeightPackage.from_arrays({"w": rng.standard_normal((8, 8))})
    b = WeightPackage.from_arrays({"w": rng.standard_normal((8, 8))})
    with pytest.raises(NoValidAlignment):
        invoke("w.align.procrustes", [a, b], {"threshold": 0.99})
