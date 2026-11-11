"""Neural Parameter Alignment (NPA).

Two checkpoints can have identical tensor shapes while living in different learned
parameter bases: neurons may be permuted, feature spaces rotated, normalization
statistics shifted. Direct arithmetic on such tensors is not meaningful, so NPA
attempts to *establish* a defensible correspondence first, and returns
``NO_VALID_ALIGNMENT`` when it cannot.

Implemented methods (all closed-form, no gradients):

``identity``      no transform; reports raw cosine similarity per tensor
``procrustes``    orthogonal R from the SVD of X^T Y (min ||X R - Y||_F)
``linear_map``    ridge-regularized least-squares map W via pseudo-inverse
``permutation``   Hungarian (Jonker-Volgenant) assignment on row cosine similarity
``basis``         symmetric whitening of the feature covariance, then orthogonal map
``activation``    forward-pass feature comparison (linear CKA + Hungarian on
                  activation correlations) for a supplied graph and inputs
``auto``          tries the candidate methods and selects the best above threshold

A numerical alignment score is *never* treated as proof of semantic equivalence.
Reports state this explicitly and carry ``semantic_equivalence_claimed: False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from uzen.core.nws.tensors import WeightPackage
from uzen.errors import NoValidAlignment, ShapeMismatch
from uzen.util.linalg import (
    cosine_similarity_matrix,
    linear_cka,
    linear_sum_assignment,
    orthogonal_procrustes,
    procrustes_projection,
    safe_norm,
)

METHODS = ("identity", "procrustes", "linear_map", "permutation", "basis", "activation", "auto")


@dataclass
class TensorAlignment:
    name: str
    method: str
    score: float
    applied: bool
    residual: float | None = None
    transform: np.ndarray | None = None
    permutation: np.ndarray | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "score": round(float(self.score), 6),
            "applied": self.applied,
            "residual": None if self.residual is None else round(float(self.residual), 6),
            "transform_shape": None if self.transform is None else list(self.transform.shape),
            "permutation": None if self.permutation is None else [int(v) for v in self.permutation[:64]],
            "details": {k: v for k, v in self.details.items() if k != "singular_values"},
        }


@dataclass
class AlignmentReport:
    method: str
    ok: bool
    score: float
    threshold: float
    tensors: dict[str, TensorAlignment] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    candidates: dict[str, float] = field(default_factory=dict)

    CODE_NO_ALIGNMENT = "NO_VALID_ALIGNMENT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "ok": self.ok,
            "score": round(float(self.score), 6),
            "threshold": self.threshold,
            "num_aligned": len(self.tensors),
            "num_skipped": len(self.skipped),
            "skipped": self.skipped,
            "reasons": self.reasons,
            "candidates": {k: round(float(v), 6) for k, v in self.candidates.items()},
            "tensors": {name: ta.summary() for name, ta in self.tensors.items()},
            "semantic_equivalence_claimed": False,
            "note": (
                "alignment scores measure numerical correspondence under a specific "
                "transformation; they do not prove that the two parameter sets compute "
                "the same function"
            ),
        }

    def require(self) -> None:
        if not self.ok:
            raise NoValidAlignment(
                "no defensible parameter correspondence could be established: "
                + "; ".join(self.reasons[:5]),
                method=self.method,
                score=float(self.score),
                threshold=self.threshold,
                code=self.CODE_NO_ALIGNMENT,
            )


# --------------------------------------------------------------------------- #
# per-tensor alignment strategies
# --------------------------------------------------------------------------- #


def _identity(x: np.ndarray, y: np.ndarray) -> TensorAlignment:
    xf = x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)
    yf = y.reshape(y.shape[0], -1) if y.ndim > 1 else y.reshape(1, -1)
    if xf.shape != yf.shape:
        return TensorAlignment("", "identity", 0.0, False, details={"reason": "shape mismatch"})
    sim = cosine_similarity_matrix(xf, yf)
    # identity correspondence means row i <-> row i
    diag = float(np.mean(np.diag(sim))) if xf.shape[0] else 0.0
    return TensorAlignment(
        "",
        "identity",
        (diag + 1.0) / 2.0,
        applied=diag > 0.0,
        residual=float(np.linalg.norm(xf - yf)),
        details={"mean_row_cosine": diag},
    )


def _procrustes(x: np.ndarray, y: np.ndarray) -> TensorAlignment:
    if x.shape != y.shape:
        return TensorAlignment("", "procrustes", 0.0, False, details={"reason": "shape mismatch"})
    xs = x.reshape(x.shape[0], -1)
    ys = y.reshape(y.shape[0], -1)
    r, residual, diag = orthogonal_procrustes(xs, ys)
    denom = safe_norm(ys) or 1.0
    score = max(0.0, 1.0 - residual / denom)
    # reshape R back into the tensor's trailing space so it can be applied
    transform = r.reshape(x.shape[1:] + y.shape[1:]) if x.ndim > 1 else r
    return TensorAlignment(
        "",
        "procrustes",
        score,
        applied=True,
        residual=residual,
        transform=transform,
        details=diag,
    )


def _linear_map(x: np.ndarray, y: np.ndarray) -> TensorAlignment:
    if x.shape != y.shape:
        return TensorAlignment("", "linear_map", 0.0, False, details={"reason": "shape mismatch"})
    xs = x.reshape(x.shape[0], -1)
    ys = y.reshape(y.shape[0], -1)
    w, residual, diag = procrustes_projection(xs, ys, ridge=1e-6)
    denom = safe_norm(ys) or 1.0
    score = max(0.0, 1.0 - residual / denom)
    return TensorAlignment(
        "",
        "linear_map",
        score,
        applied=True,
        residual=residual,
        transform=w.reshape(x.shape[1:] + y.shape[1:]),
        details=diag,
    )


def _permutation(x: np.ndarray, y: np.ndarray) -> TensorAlignment:
    """Row (output-neuron) permutation alignment via Hungarian assignment."""
    if x.shape != y.shape or x.ndim < 2:
        return TensorAlignment("", "permutation", 0.0, False, details={"reason": "requires equal rank>=2 shapes"})
    xs = x.reshape(x.shape[0], -1)
    ys = y.reshape(y.shape[0], -1)
    sim = cosine_similarity_matrix(xs, ys)
    cost = 1.0 - sim
    # Guard against NaN from zero-norm rows.
    cost = np.nan_to_num(cost, nan=2.0)
    rows, cols = linear_sum_assignment(cost)
    pairs = dict(zip(rows.tolist(), cols.tolist()))
    # ``pairs`` maps source row -> target row. Reordering needs the inverse
    # (target row -> source row) so that ``x[perm][c] == x[r] ~ y[c]``.
    perm = np.arange(xs.shape[0], dtype=np.int64)
    for r, c in pairs.items():
        perm[c] = r
    matched_sim = float(np.mean([sim[r, c] for r, c in pairs.items()])) if pairs else 0.0
    covered = len(pairs) / max(1, xs.shape[0])
    score = (matched_sim + 1.0) / 2.0 * covered
    residual = float(np.linalg.norm(xs[perm] - ys))
    return TensorAlignment(
        "",
        "permutation",
        score,
        applied=True,
        residual=residual,
        permutation=perm,
        details={
            "mean_matched_cosine": matched_sim,
            "assignment_coverage": covered,
            "assignment_cost": float(np.sum([cost[r, c] for r, c in pairs.items()])),
        },
    )


def _basis(x: np.ndarray, y: np.ndarray, *, eps: float = 1e-6) -> TensorAlignment:
    """Whitening + orthogonal map, then re-colouring to the target covariance."""
    if x.shape != y.shape or x.ndim < 2:
        return TensorAlignment("", "basis", 0.0, False, details={"reason": "requires equal rank>=2 shapes"})
    xs = x.reshape(x.shape[0], -1).astype(np.float64)
    ys = y.reshape(y.shape[0], -1).astype(np.float64)
    cx = xs.T @ xs / max(1, xs.shape[0])
    cy = ys.T @ ys / max(1, ys.shape[0])

    def _sym_inv_sqrt(m: np.ndarray, power: float) -> np.ndarray:
        vals, vecs = np.linalg.eigh(m + eps * np.eye(m.shape[0]))
        vals = np.clip(vals, eps, None)
        return (vecs * (vals**power)) @ vecs.T

    wx = _sym_inv_sqrt(cx, 0.5)  # whiten source
    wy = _sym_inv_sqrt(cy, -0.5)  # re-colour to target
    xw = xs @ wx
    yw = ys @ wy
    r, residual, diag = orthogonal_procrustes(xw, yw)
    total = wy @ r @ wx
    pred = xs @ total
    pred_residual = float(np.linalg.norm(pred - ys))
    denom = safe_norm(ys) or 1.0
    score = max(0.0, 1.0 - pred_residual / denom)
    return TensorAlignment(
        "",
        "basis",
        score,
        applied=True,
        residual=pred_residual,
        transform=total.reshape(x.shape[1:] + y.shape[1:]),
        details={
            "orthogonal_residual": residual,
            "whitening_epsilon": eps,
            "condition_source": float(np.linalg.cond(cx)),
            "condition_target": float(np.linalg.cond(cy)),
            **{k: v for k, v in diag.items() if k != "singular_values"},
        },
    )


_STRATEGIES: dict[str, Callable[[np.ndarray, np.ndarray], TensorAlignment]] = {
    "identity": _identity,
    "procrustes": _procrustes,
    "linear_map": _linear_map,
    "permutation": _permutation,
    "basis": _basis,
}


# --------------------------------------------------------------------------- #
# activation-space alignment
# --------------------------------------------------------------------------- #


@dataclass
class ActivationAlignment:
    node: str
    cka: float
    permutation: np.ndarray | None
    mean_matched_cosine: float
    shapes: tuple[int, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "cka": round(float(self.cka), 6),
            "mean_matched_cosine": round(float(self.mean_matched_cosine), 6),
            "shape": list(self.shapes),
            "permutation": None if self.permutation is None else [int(v) for v in self.permutation[:64]],
        }


def activation_alignment(
    source: WeightPackage,
    target: WeightPackage,
    graph,
    inputs: Mapping[str, np.ndarray],
    *,
    nodes: Sequence[str] | None = None,
    max_rows: int = 4096,
) -> dict[str, ActivationAlignment]:
    """Compare internal representations of two parameter sets on one graph.

    Runs a real forward pass with each parameter set, captures activations at the
    requested nodes, then reports linear CKA and a Hungarian neuron permutation per
    node. This is evidence about *feature space correspondence*, not proof of
    functional equivalence.
    """
    from uzen.core.nir.executor import execute

    node_ids = list(nodes) if nodes else _candidate_nodes(graph)
    result_src = execute(graph, inputs, source, capture=node_ids, check_finite=True)
    result_tgt = execute(graph, inputs, target, capture=node_ids, check_finite=True)
    out: dict[str, ActivationAlignment] = {}
    for nid in node_ids:
        a = result_src.intermediates.get(nid)
        b = result_tgt.intermediates.get(nid)
        if a is None or b is None:
            continue
        fa = _feature_matrix(a, max_rows)
        fb = _feature_matrix(b, max_rows)
        if fa.shape != fb.shape:
            continue
        cka = linear_cka(fa, fb)
        sim = cosine_similarity_matrix(fa, fb)
        cost = np.nan_to_num(1.0 - sim, nan=2.0)
        rows, cols = linear_sum_assignment(cost)
        perm = np.arange(fa.shape[0], dtype=np.int64)
        for r, c in zip(rows.tolist(), cols.tolist()):
            perm[r] = c
        matched = float(np.mean([sim[r, c] for r, c in zip(rows.tolist(), cols.tolist())])) if len(rows) else 0.0
        out[nid] = ActivationAlignment(
            node=nid,
            cka=cka,
            permutation=perm,
            mean_matched_cosine=matched,
            shapes=tuple(fa.shape),
        )
    return out


def _candidate_nodes(graph) -> list[str]:
    """Nodes whose outputs expose a (batch, ..., features) representation."""
    nodes: list[str] = []
    for nid, node in graph.nodes.items():
        if node.outputs and node.outputs[0].rank >= 2:
            nodes.append(nid)
    return nodes


def _feature_matrix(activation: np.ndarray, max_rows: int) -> np.ndarray:
    arr = np.asarray(activation, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    else:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[0] > max_rows:
        step = max(1, arr.shape[0] // max_rows)
        arr = arr[::step][:max_rows]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# aligner
# --------------------------------------------------------------------------- #


class ParameterAligner:
    """Establish (or refuse to establish) parameter correspondence."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        method: str = "auto",
        allow_partial: bool = False,
        graph=None,
        activation_inputs: Mapping[str, np.ndarray] | None = None,
        activation_nodes: Sequence[str] | None = None,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown alignment method '{method}'; choose from {METHODS}")
        self.method = method
        self.threshold = float(threshold)
        self.allow_partial = allow_partial
        self.graph = graph
        self.activation_inputs = dict(activation_inputs or {})
        self.activation_nodes = list(activation_nodes or [])

    # ------------------------------------------------------------------ #

    def align(self, source: WeightPackage, target: WeightPackage) -> AlignmentReport:
        common = sorted(set(source.names()) & set(target.names()))
        skipped: dict[str, str] = {}
        for name in sorted(set(source.names()) ^ set(target.names())):
            missing_in = "target" if name in source else "source"
            skipped[name] = f"tensor missing from {missing_in}"
        if not common:
            return AlignmentReport(
                method=self.method,
                ok=False,
                score=0.0,
                threshold=self.threshold,
                skipped=skipped,
                reasons=["the two parameter sets share no tensor names; names must be mapped first"],
            )

        if self.method == "activation":
            return self._align_activation(source, target, common, skipped)
        if self.method == "auto":
            return self._align_auto(source, target, common, skipped)
        return self._align_single(self.method, source, target, common, skipped)

    # ------------------------------------------------------------------ #

    def _align_single(
        self,
        method: str,
        source: WeightPackage,
        target: WeightPackage,
        common: Sequence[str],
        skipped: dict[str, str],
    ) -> AlignmentReport:
        strategy = _STRATEGIES[method]
        tensors: dict[str, TensorAlignment] = {}
        reasons: list[str] = []
        weights: list[float] = []
        scores: list[float] = []
        for name in common:
            x = source.tensors[name]
            y = target.tensors[name]
            if x.shape != y.shape:
                skipped[name] = f"shape mismatch {x.shape} vs {y.shape}"
                continue
            if x.dtype.kind not in "fc":
                skipped[name] = f"non-float tensor ({x.dtype})"
                continue
            if x.ndim < 2:
                # Rank-1 parameters (biases, norms, gains) have no basis ambiguity: a
                # rotation/permutation of a matrix does not apply to them. Align them by
                # identity and keep them out of the aggregate basis score.
                ta = _identity(x.astype(np.float64, copy=False), y.astype(np.float64, copy=False))
                ta.name = name
                ta.details["reason"] = "rank-1 tensor aligned by identity (no basis ambiguity)"
                tensors[name] = ta
                continue
            ta = strategy(x.astype(np.float64, copy=False), y.astype(np.float64, copy=False))
            ta.name = name
            tensors[name] = ta
            weight = float(max(1, x.size))
            weights.append(weight)
            scores.append(ta.score)
            if ta.details.get("reason"):
                reasons.append(f"{name}: {ta.details['reason']}")

        score = float(np.average(scores, weights=weights)) if scores else 1.0
        ok = bool(tensors) and score >= self.threshold
        if not ok:
            reasons.append(
                f"aggregate alignment score {score:.4f} is below threshold {self.threshold:.4f} "
                f"for method '{method}'; parameters do not appear to share a recoverable basis"
            )
        if not self.allow_partial and skipped and tensors:
            ok = False
            reasons.append(
                f"{len(skipped)} tensors could not be aligned ({list(skipped)[:5]}); "
                "partial alignment requires allow_partial"
            )
        return AlignmentReport(
            method=method,
            ok=ok,
            score=score,
            threshold=self.threshold,
            tensors=tensors,
            skipped=skipped,
            reasons=reasons,
        )

    def _align_auto(
        self,
        source: WeightPackage,
        target: WeightPackage,
        common: Sequence[str],
        skipped: dict[str, str],
    ) -> AlignmentReport:
        candidates: dict[str, float] = {}
        best: AlignmentReport | None = None
        # Order matters: cheapest/most conservative first.
        for method in ("identity", "procrustes", "permutation", "linear_map"):
            report = self._align_single(method, source, target, common, dict(skipped))
            candidates[method] = report.score
            if best is None or report.score > best.score:
                best = report
        assert best is not None
        best.candidates = candidates
        best.method = f"auto:{best.method}"
        best.ok = bool(best.tensors) and best.score >= self.threshold
        if not best.ok:
            best.reasons.append(
                "no candidate alignment method reached the threshold "
                f"({candidates}); returning NO_VALID_ALIGNMENT"
            )
        return best

    def _align_activation(
        self,
        source: WeightPackage,
        target: WeightPackage,
        common: Sequence[str],
        skipped: dict[str, str],
    ) -> AlignmentReport:
        if self.graph is None or not self.activation_inputs:
            raise NoValidAlignment(
                "activation alignment requires a graph and concrete inputs "
                "(pass graph=... and activation_inputs=...)",
                method="activation",
            )
        acts = activation_alignment(
            source,
            target,
            self.graph,
            self.activation_inputs,
            nodes=self.activation_nodes or None,
        )
        if not acts:
            return AlignmentReport(
                method="activation",
                ok=False,
                score=0.0,
                threshold=self.threshold,
                skipped=skipped,
                reasons=["no comparable activations were captured on the supplied graph"],
            )
        per_node = {nid: a.cka for nid, a in acts.items()}
        score = float(np.mean(list(per_node.values())))
        tensors = {
            nid: TensorAlignment(
                name=f"activation:{nid}",
                method="activation",
                score=float(a.cka),
                applied=False,
                permutation=a.permutation,
                details={
                    "mean_matched_cosine": a.mean_matched_cosine,
                    "shape": list(a.shapes),
                    "note": "activation-space correspondence; map node permutations to parameters explicitly",
                },
            )
            for nid, a in acts.items()
        }
        ok = score >= self.threshold
        reasons = []
        if not ok:
            reasons.append(
                f"mean activation CKA {score:.4f} is below threshold {self.threshold:.4f}"
            )
        return AlignmentReport(
            method="activation",
            ok=ok,
            score=score,
            threshold=self.threshold,
            tensors=tensors,
            skipped=skipped,
            reasons=reasons,
            candidates=per_node,
        )

    # ------------------------------------------------------------------ #

    def apply(
        self,
        source: WeightPackage,
        report: AlignmentReport,
        *,
        include_untransformed: bool = True,
    ) -> WeightPackage:
        """Apply an alignment report to produce aligned source weights."""
        tensors: dict[str, np.ndarray] = {}
        for name in source.names():
            arr = source.tensors[name]
            ta = report.tensors.get(name)
            if ta is None or not ta.applied:
                if include_untransformed:
                    tensors[name] = np.array(arr, copy=True)
                continue
            tensors[name] = apply_tensor_alignment(arr, ta)
        pkg = source.copy()
        pkg.tensors = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
        pkg.metadata = {
            **pkg.metadata,
            "alignment_method": report.method,
            "alignment_score": report.score,
            "semantic_equivalence_claimed": False,
        }
        return pkg


def apply_tensor_alignment(arr: np.ndarray, ta: TensorAlignment) -> np.ndarray:
    """Apply one tensor alignment (permutation or matrix transform)."""
    work = arr.astype(np.float64, copy=False)
    if ta.permutation is not None:
        if arr.ndim < 2:
            raise ShapeMismatch(
                f"tensor '{ta.name}' is rank-{arr.ndim}; a row permutation cannot be applied"
            )
        return work[ta.permutation].astype(arr.dtype, copy=False)
    if ta.transform is not None:
        if arr.ndim < 2:
            raise ShapeMismatch(
                f"tensor '{ta.name}' is rank-{arr.ndim}; a matrix transform cannot be applied"
            )
        trailing = arr.shape[1:]
        w = ta.transform.reshape(int(np.prod(trailing)), int(np.prod(trailing)))
        flat = work.reshape(arr.shape[0], -1)
        return (flat @ w).reshape(arr.shape).astype(arr.dtype, copy=False)
    return np.array(arr, copy=True)


def align_packages(
    source: WeightPackage,
    target: WeightPackage,
    *,
    method: str = "auto",
    threshold: float = 0.5,
    allow_partial: bool = False,
    graph=None,
    activation_inputs: Mapping[str, np.ndarray] | None = None,
) -> tuple[WeightPackage, AlignmentReport]:
    """Convenience wrapper: align then apply, raising on NO_VALID_ALIGNMENT."""
    aligner = ParameterAligner(
        method=method,
        threshold=threshold,
        allow_partial=allow_partial,
        graph=graph,
        activation_inputs=activation_inputs,
    )
    report = aligner.align(source, target)
    report.require()
    return aligner.apply(source, report), report
