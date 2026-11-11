"""Derivative-free search algorithms.

ANE searches over *low-dimensional transformation parameters* (interpolation
weights, merge densities, ranks, scaling factors) — never over individual model
parameters. All algorithms are gradient-free by construction, deterministic under a
seed, budgeted by evaluation count, and return a full history for provenance.

Implemented:
  * CMA-ES with full covariance adaptation (Hansen's standard update)
  * (mu, lambda) evolution strategy with isotropic Gaussian mutation
  * coordinate descent with adaptive step sizes
  * simulated annealing with geometric cooling
  * Bayesian optimization with a GP (RBF kernel) and Expected Improvement
  * beam search over a discretized grid (useful for integer/categorical knobs)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from uzen.core.nwpc.registry import SearchRequest, SearchResult
from uzen.errors import BudgetExhausted, InvalidOperationParameters


def _clip(x: np.ndarray, bounds: Sequence[tuple[float, float]]) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    return np.clip(x, lo, hi)


def _scales(bounds: Sequence[tuple[float, float]]) -> np.ndarray:
    return np.array([max(1e-12, b[1] - b[0]) for b in bounds], dtype=np.float64)


def _random_point(rng: np.random.Generator, bounds: Sequence[tuple[float, float]]) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    return rng.uniform(lo, hi)


class _Budget:
    def __init__(self, objective: Callable[[np.ndarray], float], budget: int, maximize: bool) -> None:
        self.objective = objective
        self.budget = budget
        self.maximize = maximize
        self.used = 0
        self.history: list[dict[str, Any]] = []
        self.best_x: np.ndarray | None = None
        self.best_value = -np.inf if maximize else np.inf

    def evaluate(self, x: np.ndarray) -> float:
        if self.used >= self.budget:
            raise BudgetExhausted(f"search budget of {self.budget} evaluations exhausted")
        value = float(self.objective(np.asarray(x, dtype=np.float64)))
        self.used += 1
        improved = (value > self.best_value) if self.maximize else (value < self.best_value)
        if improved or self.best_x is None:
            self.best_x = np.asarray(x, dtype=np.float64).copy()
            self.best_value = value
        self.history.append({"x": [float(v) for v in np.asarray(x)], "value": value, "eval": self.used})
        if not np.isfinite(value) and self.maximize:
            return -1e9
        return value


def _result(algorithm: str, tracker: _Budget, *, converged: bool, details: dict[str, Any]) -> SearchResult:
    assert tracker.best_x is not None
    return SearchResult(
        best_params=tracker.best_x,
        best_value=float(tracker.best_value),
        history=tracker.history,
        evaluations=tracker.used,
        converged=converged,
        algorithm=algorithm,
        details=details,
    )


# --------------------------------------------------------------------------- #
# CMA-ES
# --------------------------------------------------------------------------- #


def cma_es(request: SearchRequest) -> SearchResult:
    d = len(request.bounds)
    if d == 0:
        raise InvalidOperationParameters("CMA-ES requires at least one search dimension")
    rng = np.random.default_rng(request.seed)
    sign = 1.0 if request.maximize else -1.0
    tracker = _Budget(lambda x: sign * request.objective(x), request.budget, maximize=True)

    lam = 4 + int(3 * math.log(d))
    mu = lam // 2
    weights = np.log((mu + 0.5) / np.arange(1, mu + 1))
    weights = weights / weights.sum()
    mueff = 1.0 / float(np.sum(weights**2))

    cc = 4.0 / (d + 4.0)
    cs = (mueff + 2.0) / (d + mueff + 3.0)
    c1 = 2.0 / ((d + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
    chi_n = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))

    scales = _scales(request.bounds)
    lo = np.array([b[0] for b in request.bounds], dtype=np.float64)
    mean = lo + scales * 0.5
    sigma = 0.25
    C = np.eye(d)
    pc = np.zeros(d)
    ps = np.zeros(d)
    eigen_updated = True
    generations = 0

    while tracker.used < request.budget:
        generations += 1
        try:
            vals, vecs = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            vals, vecs = np.ones(d), np.eye(d)
        vals = np.clip(vals, 1e-14, None)
        B = vecs
        D = np.sqrt(vals)
        arz = rng.standard_normal((lam, d))
        ary = (arz * D) @ B.T
        arx = mean + sigma * ary * scales
        arx = np.array([_clip(x, request.bounds) for x in arx])
        values = []
        for x in arx:
            values.append(tracker.evaluate(x))
            if tracker.used >= request.budget:
                break
        values = np.array(values)
        if values.size == 0:
            break
        order = np.argsort(-values)[: min(mu, values.size)]
        selected = ary[order]
        old_mean = mean.copy()
        mean = mean + sigma * (weights[: len(order)] @ selected) * scales
        y_mean = (mean - old_mean) / (sigma * scales + 1e-30)
        try:
            inv_sqrt = B @ np.diag(1.0 / D) @ B.T
        except np.linalg.LinAlgError:
            inv_sqrt = np.eye(d)
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt @ y_mean)
        hsig = float(np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * generations)) / chi_n) < (
            1.4 + 2.0 / (d + 1.0)
        )
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_mean
        artmp = selected
        rank_mu = sum(
            weights[i] * np.outer(artmp[i], artmp[i]) for i in range(min(len(order), len(weights)))
        )
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * rank_mu
        sigma = sigma * math.exp(min(1.0, (cs / damps) * (np.linalg.norm(ps) / chi_n - 1.0)))
        sigma = float(np.clip(sigma, 1e-8, 10.0))
        eigen_updated = True

    return _result(
        "cma_es",
        tracker,
        converged=sigma < 1e-3,
        details={
            "dimensions": d,
            "lambda": lam,
            "mu": mu,
            "generations": generations,
            "final_sigma": sigma,
            "eigen_update": eigen_updated,
        },
    )


# --------------------------------------------------------------------------- #
# evolution strategy
# --------------------------------------------------------------------------- #


def evolution_strategy(request: SearchRequest) -> SearchResult:
    d = len(request.bounds)
    rng = np.random.default_rng(request.seed)
    sign = 1.0 if request.maximize else -1.0
    tracker = _Budget(lambda x: sign * request.objective(x), request.budget, maximize=True)
    scales = _scales(request.bounds)
    lo = np.array([b[0] for b in request.bounds], dtype=np.float64)

    lam = max(4, min(32, 4 + d))
    mu = max(1, lam // 2)
    sigma0 = float(request.options.get("sigma", 0.25))
    sigma = sigma0
    center = _random_point(rng, request.bounds)
    generation = 0

    while tracker.used < request.budget:
        generation += 1
        candidates = []
        for _ in range(lam):
            if tracker.used >= request.budget:
                break
            x = _clip(center + rng.standard_normal(d) * sigma * scales, request.bounds)
            candidates.append((tracker.evaluate(x), x))
        if not candidates:
            break
        candidates.sort(key=lambda t: -t[0])
        elite = np.stack([c[1] for c in candidates[:mu]])
        center = elite.mean(axis=0)
        sigma = max(1e-6, sigma * (0.95 if generation > 3 else 1.0))

    return _result(
        "evolution_strategy",
        tracker,
        converged=sigma < 1e-4,
        details={"dimensions": d, "lambda": lam, "mu": mu, "generations": generation, "final_sigma": sigma},
    )


# --------------------------------------------------------------------------- #
# coordinate descent
# --------------------------------------------------------------------------- #


def coordinate_descent(request: SearchRequest) -> SearchResult:
    d = len(request.bounds)
    rng = np.random.default_rng(request.seed)
    sign = 1.0 if request.maximize else -1.0
    tracker = _Budget(lambda x: sign * request.objective(x), request.budget, maximize=True)
    scales = _scales(request.bounds)
    x = _random_point(rng, request.bounds)
    best = tracker.evaluate(x)
    step = np.full(d, 0.25)
    rounds = 0

    while tracker.used < request.budget:
        rounds += 1
        improved = False
        for i in range(d):
            for direction in (1.0, -1.0):
                if tracker.used >= request.budget:
                    break
                candidate = x.copy()
                candidate[i] = candidate[i] + direction * step[i] * scales[i]
                candidate = _clip(candidate, request.bounds)
                value = tracker.evaluate(candidate)
                if value > best:
                    best = value
                    x = candidate
                    improved = True
                    break
        if not improved:
            step *= 0.5
            if np.all(step < 1e-4):
                break

    return _result(
        "coordinate_descent",
        tracker,
        converged=bool(np.all(step < 1e-3)),
        details={"dimensions": d, "rounds": rounds, "final_step": [float(s) for s in step]},
    )


# --------------------------------------------------------------------------- #
# simulated annealing
# --------------------------------------------------------------------------- #


def simulated_annealing(request: SearchRequest) -> SearchResult:
    d = len(request.bounds)
    rng = np.random.default_rng(request.seed)
    sign = 1.0 if request.maximize else -1.0
    tracker = _Budget(lambda x: sign * request.objective(x), request.budget, maximize=True)
    scales = _scales(request.bounds)
    x = _random_point(rng, request.bounds)
    current = tracker.evaluate(x)
    best_x, best_value = x.copy(), current
    temp = float(request.options.get("temperature", 1.0))
    cooling = float(request.options.get("cooling", 0.97))
    steps = 0
    while tracker.used < request.budget:
        steps += 1
        candidate = _clip(x + rng.standard_normal(d) * 0.2 * scales * max(0.05, temp), request.bounds)
        value = tracker.evaluate(candidate)
        delta = value - current
        if delta >= 0 or rng.random() < math.exp(min(0.0, delta / max(temp, 1e-9))):
            x, current = candidate, value
            if current > best_value:
                best_x, best_value = x.copy(), current
        temp = max(1e-4, temp * cooling)

    tracker.best_x = best_x
    tracker.best_value = best_value
    return _result(
        "simulated_annealing",
        tracker,
        converged=temp <= 1e-3,
        details={"dimensions": d, "steps": steps, "final_temperature": temp},
    )


# --------------------------------------------------------------------------- #
# Bayesian optimization (GP + EI)
# --------------------------------------------------------------------------- #


def _rbf_kernel(a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
    sq = np.sum(a**2, axis=1)[:, None] + np.sum(b**2, axis=1)[None, :] - 2 * a @ b.T
    return np.exp(-0.5 * np.maximum(sq, 0.0) / (length**2))


def bayesian_optimization(request: SearchRequest) -> SearchResult:
    d = len(request.bounds)
    rng = np.random.default_rng(request.seed)
    sign = 1.0 if request.maximize else -1.0
    tracker = _Budget(lambda x: sign * request.objective(x), request.budget, maximize=True)
    scales = _scales(request.bounds)
    n_init = min(max(3, 2 * d + 1), max(1, request.budget // 3))
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for _ in range(n_init):
        if tracker.used >= request.budget:
            break
        x = _random_point(rng, request.bounds)
        xs.append(x)
        ys.append(tracker.evaluate(x))

    length = float(request.options.get("length_scale", 0.3))
    noise = float(request.options.get("noise", 1e-3))
    iterations = 0

    while tracker.used < request.budget:
        iterations += 1
        X = np.array(xs)
        Y = np.array(ys)
        # Normalize inputs to [0,1]^d for a stationary kernel.
        lo = np.array([b[0] for b in request.bounds], dtype=np.float64)
        Xn = (X - lo) / scales
        K = _rbf_kernel(Xn, Xn, length) + noise * np.eye(len(xs))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(len(xs)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))

        candidates = np.array([_random_point(rng, request.bounds) for _ in range(256)])
        Cn = (candidates - lo) / scales
        Ks = _rbf_kernel(Cn, Xn, length)
        mean = Ks @ alpha
        v = np.linalg.solve(L, Ks.T)
        var = np.clip(1.0 - np.sum(v**2, axis=0), 1e-12, None)
        std = np.sqrt(var)
        best_y = float(np.max(Y))
        improvement = mean - best_y - 0.01
        z = improvement / std
        ei = improvement * _norm_cdf(z) + std * _norm_pdf(z)
        next_x = candidates[int(np.argmax(ei))]
        xs.append(next_x)
        ys.append(tracker.evaluate(next_x))

    return _result(
        "bayesian_optimization",
        tracker,
        converged=False,
        details={
            "dimensions": d,
            "iterations": iterations,
            "init_points": n_init,
            "length_scale": length,
            "noise": noise,
            "model": "GP-RBF with Expected Improvement",
        },
    )


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# beam search over a discretized grid
# --------------------------------------------------------------------------- #


def beam_search(request: SearchRequest) -> SearchResult:
    d = len(request.bounds)
    if d > 4:
        raise InvalidOperationParameters(
            f"beam search over a grid is limited to 4 dimensions (requested {d}); "
            "use CMA-ES or Bayesian optimization for higher-dimensional search"
        )
    steps = int(request.options.get("steps", 5))
    beam = int(request.options.get("beam_width", 3))
    steps = max(2, steps)
    sign = 1.0 if request.maximize else -1.0
    tracker = _Budget(lambda x: sign * request.objective(x), request.budget, maximize=True)
    grids = [np.linspace(b[0], b[1], steps) for b in request.bounds]
    # Breadth-first with beam pruning: expand one dimension at a time.
    partials: list[np.ndarray] = [np.array([g[0] for g in grids])]
    for dim in range(d):
        expanded: list[np.ndarray] = []
        for base in partials:
            for value in grids[dim]:
                candidate = base.copy()
                candidate[dim] = value
                expanded.append(candidate)
        scored: list[tuple[float, np.ndarray]] = []
        for candidate in expanded:
            if tracker.used >= request.budget:
                break
            scored.append((tracker.evaluate(candidate), candidate))
        if not scored:
            break
        scored.sort(key=lambda t: -t[0])
        partials = [c for _, c in scored[:beam]]

    return _result(
        "beam_search",
        tracker,
        converged=True,
        details={"dimensions": d, "grid_steps": steps, "beam_width": beam},
    )


# --------------------------------------------------------------------------- #
# multi-objective NSGA-II
# --------------------------------------------------------------------------- #


def _pareto_dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def _non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Fast non-dominated sorting (Deb et al.); objectives are maximised."""
    n = len(objectives)
    dominated_by = [0] * n
    dominates: list[list[int]] = [[] for _ in range(n)]
    for p in range(n):
        for q in range(p + 1, n):
            if _pareto_dominates(objectives[p], objectives[q]):
                dominates[p].append(q)
                dominated_by[q] += 1
            elif _pareto_dominates(objectives[q], objectives[p]):
                dominates[q].append(p)
                dominated_by[p] += 1
    fronts: list[list[int]] = []
    current = [i for i in range(n) if dominated_by[i] == 0]
    while current:
        fronts.append(current)
        nxt: list[int] = []
        for p in current:
            for q in dominates[p]:
                dominated_by[q] -= 1
                if dominated_by[q] == 0:
                    nxt.append(q)
        current = nxt
    return fronts


def _crowding_distance(objectives: np.ndarray, front: Sequence[int]) -> dict[int, float]:
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    m = objectives.shape[1]
    for dim in range(m):
        order = sorted(front, key=lambda i: objectives[i, dim])
        lo = objectives[order[0], dim]
        hi = objectives[order[-1], dim]
        distance[order[0]] = distance[order[-1]] = float("inf")
        span = float(hi - lo)
        if span <= 0:
            continue
        for k in range(1, len(order) - 1):
            distance[order[k]] += float(
                objectives[order[k + 1], dim] - objectives[order[k - 1], dim]
            ) / span
    return distance


def nsga2(request: SearchRequest) -> SearchResult:
    """NSGA-II over low-dimensional transformation parameters.

    Objective 0 is ``request.objective``; further objectives come from
    ``request.extra_objectives``. All are maximised. Returns the non-dominated front,
    which is the honest answer for conflicting objectives: there is no single "best".
    """
    d = len(request.bounds)
    if d < 1:
        raise InvalidOperationParameters("NSGA-II requires at least one search dimension")
    objectives = (request.objective, *request.extra_objectives)
    if len(objectives) < 2:
        raise InvalidOperationParameters(
            "NSGA-II is multi-objective: supply at least one extra objective in "
            "SearchRequest.extra_objectives"
        )
    rng = np.random.default_rng(request.seed)
    population_size = int(request.options.get("population", max(8, 4 * d)))
    population_size = max(2, min(population_size, request.budget))
    eta_c = float(request.options.get("eta_crossover", 15.0))
    eta_m = float(request.options.get("eta_mutation", 20.0))
    mutation_rate = float(request.options.get("mutation_rate", 1.0 / d))
    scales = _scales(request.bounds)
    lo = np.array([b[0] for b in request.bounds], dtype=np.float64)
    hi = np.array([b[1] for b in request.bounds], dtype=np.float64)

    used = 0
    history: list[dict[str, Any]] = []

    def evaluate(x: np.ndarray) -> np.ndarray:
        nonlocal used
        values = np.array([float(fn(np.asarray(x, dtype=np.float64))) for fn in objectives])
        used += 1
        history.append(
            {"x": [float(v) for v in x], "values": [float(v) for v in values], "eval": used}
        )
        return values

    def rank_and_crowd(fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fronts = _non_dominated_sort(fit)
        rank = np.zeros(len(fit), dtype=int)
        crowd = np.zeros(len(fit), dtype=np.float64)
        for r, front in enumerate(fronts):
            for i in front:
                rank[i] = r
            for i, dist in _crowding_distance(fit, front).items():
                crowd[i] = dist
        return rank, crowd

    population = [rng.uniform(lo, hi) for _ in range(population_size)]
    fitness = np.array([evaluate(x) for x in population])

    while used < request.budget:
        rank, crowd = rank_and_crowd(fitness)

        def tournament() -> np.ndarray:
            a, b = rng.integers(0, len(population), size=2)
            if rank[a] != rank[b]:
                return population[a] if rank[a] < rank[b] else population[b]
            return population[a] if crowd[a] >= crowd[b] else population[b]

        offspring: list[np.ndarray] = []
        offspring_fitness: list[np.ndarray] = []
        while len(offspring) < population_size and used < request.budget:
            p1, p2 = tournament(), tournament()
            if rng.random() < 0.9:
                u = rng.random(d)
                beta = np.where(
                    u <= 0.5,
                    (2.0 * u) ** (1.0 / (eta_c + 1.0)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)),
                )
                c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
                c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for child in (c1, c2):
                if used >= request.budget or len(offspring) >= population_size:
                    break
                mutate = rng.random(d) < mutation_rate
                if mutate.any():
                    u = rng.random(d)
                    delta = np.where(
                        u < 0.5,
                        (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0,
                        1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0)),
                    )
                    child[mutate] = child[mutate] + delta[mutate] * scales[mutate]
                np.clip(child, lo, hi, out=child)
                offspring.append(child)
                offspring_fitness.append(evaluate(child))
        if not offspring:
            break
        combined = population + offspring
        combined_fitness = np.vstack([fitness, np.array(offspring_fitness)])
        rank, crowd = rank_and_crowd(combined_fitness)
        order = sorted(range(len(combined)), key=lambda i: (rank[i], -crowd[i]))
        selected = order[:population_size]
        population = [combined[i] for i in selected]
        fitness = combined_fitness[selected]

    rank, _crowd = rank_and_crowd(fitness)
    first_front = [i for i in range(len(fitness)) if rank[i] == 0]
    pareto_front = [
        {
            "params": [float(v) for v in population[i]],
            "objectives": [float(v) for v in fitness[i]],
        }
        for i in sorted(first_front, key=lambda i: -fitness[i, 0])
    ]
    representative = max(first_front, key=lambda i: fitness[i, 0])
    return SearchResult(
        best_params=population[representative],
        best_value=float(fitness[representative, 0]),
        history=history,
        evaluations=used,
        converged=False,
        algorithm="nsga2",
        details={
            "dimensions": d,
            "objectives": len(objectives),
            "population": population_size,
            "front_size": len(first_front),
            "eta_crossover": eta_c,
            "eta_mutation": eta_m,
            "selection": "fast non-dominated sort + crowding distance",
        },
        pareto_front=pareto_front,
    )


# Scalar algorithms optimise a single scalar objective; ANE selects from these.
SEARCH_ALGORITHMS: dict[str, Callable[[SearchRequest], SearchResult]] = {
    "cma_es": cma_es,
    "evolution_strategy": evolution_strategy,
    "coordinate_descent": coordinate_descent,
    "simulated_annealing": simulated_annealing,
    "bayesian_optimization": bayesian_optimization,
    "beam_search": beam_search,
}

# Multi-objective algorithms return a Pareto front and require extra objectives.
MULTI_OBJECTIVE_ALGORITHMS: dict[str, Callable[[SearchRequest], SearchResult]] = {
    "nsga2": nsga2,
}
