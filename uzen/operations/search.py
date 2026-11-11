"""Search operations registered in the operation catalogue.

Search algorithms operate on an objective function over low-dimensional
transformation parameters. They are registered as operations so that they are
subject to the same policy gate, audition and capability reporting as
parameter-transforming operations, but they are invoked through
``OperationRegistry.invoke_search``.
"""

from __future__ import annotations

from uzen.core.ane.search import MULTI_OBJECTIVE_ALGORITHMS, SEARCH_ALGORITHMS
from uzen.core.nwpc.registry import OperationSpec, ParamSpec, default_registry
from uzen.policies.strict_plus.effects import EngineEffect, Maturity, TaskScope

_SCOPES = frozenset(
    {
        TaskScope.WEIGHT_OPTIMIZATION,
        TaskScope.WEIGHT_GENERATION,
        TaskScope.WEIGHT_MODIFICATION,
        TaskScope.WEIGHT_RECONSTRUCTION,
    }
)

_DESCRIPTIONS = {
    "cma_es": "Covariance Matrix Adaptation Evolution Strategy with full covariance update.",
    "evolution_strategy": "(mu, lambda) evolution strategy with isotropic Gaussian mutation.",
    "coordinate_descent": "Cyclic coordinate search with adaptive step halving.",
    "simulated_annealing": "Metropolis acceptance with geometric cooling.",
    "bayesian_optimization": "Gaussian-process surrogate with Expected Improvement acquisition.",
    "beam_search": "Beam search over a discretized low-dimensional grid (<= 4 dimensions).",
    "nsga2": "NSGA-II multi-objective search; returns the non-dominated Pareto front.",
}


def _register_all() -> None:
    registry = default_registry()
    algorithms = {**SEARCH_ALGORITHMS, **MULTI_OBJECTIVE_ALGORITHMS}
    for name, fn in algorithms.items():
        registry.register(
            OperationSpec(
                code=f"w.search.{name}",
                title=f"Derivative-free search: {name}",
                family="search",
                effect=EngineEffect.DERIVATIVE_FREE,
                fn=None,  # search-only: invoked through registry.invoke_search
                description=_DESCRIPTIONS[name],
                maturity=Maturity.IMPLEMENTED,
                scopes=_SCOPES,
                params={
                    "iterations": ParamSpec("int", default=50, minimum=1, maximum=100000),
                    "budget": ParamSpec("int", default=100, minimum=1, maximum=100000),
                },
                assumptions=(
                    "the objective is a low-dimensional parameterization, not raw model weights",
                    "no gradient information is used or required",
                    "the search budget bounds total objective evaluations",
                ),
                search=fn,
                min_inputs=0,
                max_inputs=0,
            )
        )


_register_all()
