"""Multi-objective search: NSGA-II must produce a genuine non-dominated front."""

from __future__ import annotations

import numpy as np
import pytest

from uzen.core.ane.search import MULTI_OBJECTIVE_ALGORITHMS
from uzen.core.nwpc.registry import SearchRequest
from uzen.errors import InvalidOperationParameters


def _bi_objective():
    def first(x):
        return -float((x[0] - 0.2) ** 2 + (x[1] - 0.2) ** 2)

    def second(x):
        return -float((x[0] - 0.8) ** 2 + (x[1] - 0.8) ** 2)

    return first, second


def test_nsga2_requires_a_second_objective():
    first, _ = _bi_objective()
    with pytest.raises(InvalidOperationParameters):
        MULTI_OBJECTIVE_ALGORITHMS["nsga2"](SearchRequest(first, [(0.0, 1.0)], 40, seed=0))


def test_nsga2_returns_a_nondominated_front_within_budget():
    first, second = _bi_objective()
    result = MULTI_OBJECTIVE_ALGORITHMS["nsga2"](
        SearchRequest(
            first, [(0.0, 1.0), (0.0, 1.0)], 200, seed=1, extra_objectives=(second,)
        )
    )
    assert result.algorithm == "nsga2"
    assert 0 < result.evaluations <= 200
    assert len(result.pareto_front) >= 2
    values = np.array([entry["objectives"] for entry in result.pareto_front])
    for i in range(len(values)):
        for j in range(len(values)):
            if i == j:
                continue
            dominated = np.all(values[i] >= values[j]) and np.any(values[i] > values[j])
            assert not dominated, (i, j, values[i], values[j])
    # Conflicting objectives must produce a trade-off, not a single collapsed point.
    assert len({round(float(v[0]), 6) for v in values}) > 1
    assert all(len(entry["params"]) == 2 for entry in result.pareto_front)


def test_nsga2_is_reproducible_under_a_seed():
    first, second = _bi_objective()
    request = SearchRequest(
        first, [(0.0, 1.0), (0.0, 1.0)], 120, seed=7, extra_objectives=(second,)
    )
    a = MULTI_OBJECTIVE_ALGORITHMS["nsga2"](request)
    b = MULTI_OBJECTIVE_ALGORITHMS["nsga2"](request)
    np.testing.assert_array_equal(a.best_params, b.best_params)
    assert a.evaluations == b.evaluations
    assert [entry["params"] for entry in a.pareto_front] == [
        entry["params"] for entry in b.pareto_front
    ]
    payload = a.to_dict()
    assert payload["pareto_front"] == a.pareto_front
