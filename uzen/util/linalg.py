"""Numerical linear algebra primitives used by the weight-engineering engines.

All of these are pure-NumPy closed-form implementations (no gradient descent, no
autograd). They constitute the mathematical backbone of:

* linear / spherical interpolation and extrapolation,
* orthogonal Procrustes parameter alignment,
* assignment-based neuron/head permutation alignment,
* linear CKA representation comparison,
* orthogonal (semi-orthogonal) parameter construction,
* BF16/F16 conversion used by the artifact codec.

Nothing here is a placeholder: each function is exercised by the test-suite on
real arrays.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# numerical safety
# --------------------------------------------------------------------------- #

EPS = 1e-12


def as_float_array(x: np.ndarray, *, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind not in "fiu b".replace(" ", ""):
        raise TypeError(f"expected a real numeric array, got dtype {arr.dtype}")
    return arr.astype(dtype, copy=False)


def safe_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x).reshape(-1)))


def unit(x: np.ndarray) -> np.ndarray:
    n = safe_norm(x)
    if n < EPS:
        raise ValueError("cannot normalize a zero vector")
    return x / n


def is_finite(x: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(x)).all())


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    shifted = arr - np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    m = np.max(arr, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(arr - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


# --------------------------------------------------------------------------- #
# interpolation
# --------------------------------------------------------------------------- #


def linear_interpolate(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """(1 - t) * a + t * b, shape-preserving, no implicit broadcasting."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for interpolation: {a.shape} vs {b.shape}")
    return (1.0 - t) * a + t * b


def spherical_interpolate(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation of two tensors treated as flat vectors.

    Falls back to linear interpolation when the vectors are (near) collinear,
    which is exactly the limit where slerp is undefined. The result preserves
    the *interpolated magnitude* along the arc; it does not renormalize the
    output to one of the input norms (callers that want norm preservation use
    :func:`slerp_preserve_norm`).
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for slerp: {a.shape} vs {b.shape}")
    flat_a = np.asarray(a, dtype=np.float64).reshape(-1)
    flat_b = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(flat_a), np.linalg.norm(flat_b)
    if na < EPS or nb < EPS:
        return linear_interpolate(a, b, t)
    cos = float(np.dot(flat_a, flat_b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    if cos > 0.9995:
        out = linear_interpolate(flat_a, flat_b, t)
        return out.reshape(a.shape)
    if cos < -0.9995:
        # antipodal: the arc is not unique; choose the linear path (documented).
        return linear_interpolate(a, b, t)
    theta = math.acos(cos)
    sin_theta = math.sin(theta)
    wa = math.sin((1.0 - t) * theta) / sin_theta
    wb = math.sin(t * theta) / sin_theta
    out = wa * flat_a + wb * flat_b
    return out.reshape(a.shape)


def slerp_preserve_norm(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Slerp followed by rescaling to the lerp of the two input norms."""
    out = spherical_interpolate(a, b, t).astype(np.float64)
    target = (1.0 - t) * safe_norm(a) + t * safe_norm(b)
    current = safe_norm(out)
    if current < EPS:
        return linear_interpolate(a, b, t)
    return (out * (target / current)).astype(np.asarray(a).dtype, copy=False)


def extrapolate(a: np.ndarray, b: np.ndarray, factor: float) -> np.ndarray:
    """a + factor * (b - a). factor>1 extrapolates beyond b, factor<0 behind a."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for extrapolation: {a.shape} vs {b.shape}")
    return a + factor * (b - a)


# --------------------------------------------------------------------------- #
# orthogonal Procrustes
# --------------------------------------------------------------------------- #


def orthogonal_procrustes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    allow_reflection: bool = True,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float, dict]:
    """Solve min_R ||X R - Y||_F subject to R^T R = I.

    Closed form via SVD of ``X^T Y = U S V^T``, giving ``R = U V^T`` (optionally
    with the standard reflection correction that yields the closest *rotation*).

    Parameters
    ----------
    x, y : (n, d_x) and (n, d_y) arrays; rows are corresponding samples.
    allow_reflection : if False, enforce det(R) = +1.

    Returns
    -------
    R, residual_frobenius, diagnostics
    """
    x = as_float_array(x)
    y = as_float_array(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("orthogonal_procrustes expects 2-D matrices")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"sample count mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[1] < y.shape[1]:
        raise ValueError(
            "target dimensionality exceeds source dimensionality; "
            "use a rectangular projection (procrustes_projection) instead"
        )

    m = x.T @ y
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    # u: (dx, k), s: (k,), vt: (k, dy)
    r = u @ vt  # (dx, dy) semi-orthogonal: R^T R = I when dx >= dy
    reflected = False
    if not allow_reflection and x.shape[1] == y.shape[1]:
        if np.linalg.det(r) < 0:
            u_fixed = u.copy()
            u_fixed[:, -1] *= -1.0
            r = u_fixed @ vt
            reflected = True
    residual = float(np.linalg.norm(x @ r - y))
    denom = float(np.linalg.norm(y)) or 1.0
    diag = {
        "singular_values": [float(v) for v in s],
        "reflected_correction": reflected,
        "relative_residual": residual / denom,
        "condition": float(s[0] / s[-1]) if s[-1] > eps else float("inf"),
        "semi_orthogonal": bool(np.allclose(r.T @ r, np.eye(r.shape[1]), atol=1e-6)),
    }
    return r, residual, diag


def procrustes_projection(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, float, dict]:
    """Least-squares map W minimising ||X W - Y||_F (dimensionality-reducing OK).

    Uses the closed-form SVD/pseudo-inverse solution; a non-zero ``ridge`` adds
    Tikhonov regularization (closed form, not an iterative solver).
    """
    x = as_float_array(x)
    y = as_float_array(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("sample count mismatch")
    xtx = x.T @ x
    if ridge > 0:
        xtx = xtx + ridge * np.eye(xtx.shape[0])
    w = np.linalg.solve(xtx, x.T @ y) if np.linalg.cond(xtx) < 1e12 else np.linalg.pinv(xtx) @ (x.T @ y)
    residual = float(np.linalg.norm(x @ w - y))
    denom = float(np.linalg.norm(y)) or 1.0
    return w, residual, {
        "relative_residual": residual / denom,
        "rank_xtx": int(np.linalg.matrix_rank(xtx)),
        "ridge": ridge,
    }


# --------------------------------------------------------------------------- #
# Hungarian / Jonker-Volgenant assignment (O(n^3), pure NumPy)
# --------------------------------------------------------------------------- #


def linear_sum_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment for a rectangular cost matrix.

    Implements the shortest-augmenting-path (Jonker-Volgenant / e-maxx) method
    with potential updates. Pure Python loop over rows with vectorised inner
    scans; exact for finite costs and independent of SciPy.

    Returns (row_ind, col_ind) sorted by row.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost must be 2-D")
    if not np.isfinite(cost).all():
        raise ValueError("cost matrix contains non-finite entries")
    n, m = cost.shape
    transposed = False
    if n > m:  # algorithm requires rows <= cols
        cost = cost.T
        n, m = m, n
        transposed = True

    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)  # p[j] = row (1-based) matched to column j
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 < 0:  # pragma: no cover - only with degenerate inputs
                raise RuntimeError("assignment failed: no augmenting column")
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    rows: list[int] = []
    cols: list[int] = []
    for j in range(1, m + 1):
        if p[j] != 0:
            r = int(p[j] - 1)
            c = int(j - 1)
            if transposed:
                rows.append(c)
                cols.append(r)
            else:
                rows.append(r)
                cols.append(c)
    order = np.argsort(rows, kind="stable")
    return np.asarray(rows, dtype=np.int64)[order], np.asarray(cols, dtype=np.int64)[order]


def assignment_cost(cost: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    return float(np.sum(cost[rows, cols]))


# --------------------------------------------------------------------------- #
# representation comparison
# --------------------------------------------------------------------------- #


def linear_cka(x: np.ndarray, y: np.ndarray, *, unbiased: bool = False) -> float:
    """Linear Centered Kernel Alignment between two representation matrices.

    ``CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)`` after centering
    columns (features). Returns a value in [0, 1]; 1 means identical up to an
    orthogonal transformation of feature space.
    """
    x = as_float_array(x)
    y = as_float_array(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("linear_cka expects 2-D feature matrices (samples x features)")
    if x.shape[0] != y.shape[0]:
        raise ValueError("sample count mismatch")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xty = np.linalg.norm(y.T @ x, "fro") ** 2
    xtx = np.linalg.norm(x.T @ x, "fro")
    yty = np.linalg.norm(y.T @ y, "fro")
    if xtx < EPS or yty < EPS:
        return 0.0
    return float(xty / (xtx * yty))


def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (n, d) and (m, d) matrices."""
    x = as_float_array(x)
    y = as_float_array(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("shape mismatch for cosine similarity")
    xn = np.linalg.norm(x, axis=1, keepdims=True)
    yn = np.linalg.norm(y, axis=1, keepdims=True)
    xn = np.maximum(xn, EPS)
    yn = np.maximum(yn, EPS)
    return (x / xn) @ (y / yn).T


# --------------------------------------------------------------------------- #
# orthogonal / semi-orthogonal construction
# --------------------------------------------------------------------------- #


def orthogonal_matrix(n: int, m: int | None = None, *, seed: int | None = None, method: str = "qr") -> np.ndarray:
    """Construct an (n, m) matrix with orthonormal columns (semi-orthogonal).

    ``method``:
      * ``qr``     - QR of a seeded Gaussian matrix (Householder-based, real).
      * ``svd``    - SVD of a seeded Gaussian matrix.
      * ``hadamard`` - normalized Hadamard matrix (power-of-two sizes only).
    Deterministic for a fixed seed. This is *analytic construction*, not learned.
    """
    m = n if m is None else m
    if n <= 0 or m <= 0:
        raise ValueError("dimensions must be positive")
    if m > n:
        raise ValueError("semi-orthogonal construction requires m <= n")
    if method == "hadamard":
        q = hadamard_matrix(n)
        return q[:, :m]
    rng = np.random.default_rng(0 if seed is None else int(seed))
    g = rng.standard_normal((n, m))
    if method == "qr":
        q, r = np.linalg.qr(g)
        # Fix signs so the result is stable across platforms.
        d = np.sign(np.diag(r))
        d[d == 0] = 1.0
        q = q * d
        return q[:, :m]
    if method == "svd":
        u, _, _ = np.linalg.svd(g, full_matrices=False)
        return u[:, :m]
    raise ValueError(f"unknown orthogonal construction method: {method}")


def hadamard_matrix(n: int) -> np.ndarray:
    """Sylvester construction of an n x n normalized Hadamard matrix (n=2^k)."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("Hadamard matrix requires n to be a power of two")
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def topk_mask(values: np.ndarray, fraction: float, *, largest: bool = True) -> np.ndarray:
    """Boolean mask selecting the top (or bottom) ``fraction`` of |values|."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    flat = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    k = int(round(fraction * flat.size))
    mask = np.zeros(flat.size, dtype=bool)
    if k > 0:
        idx = np.argpartition(flat, -k)[-k:] if largest else np.argpartition(flat, k - 1)[:k]
        mask[idx] = True
    return mask.reshape(np.shape(values))


def low_rank_approximation(matrix: np.ndarray, rank: int) -> Tuple[np.ndarray, dict]:
    """Truncated SVD reconstruction (Eckart-Young optimal for Frobenius norm)."""
    m = as_float_array(matrix)
    if m.ndim != 2:
        raise ValueError("low_rank_approximation expects a 2-D matrix")
    rank = int(rank)
    if rank <= 0:
        raise ValueError("rank must be positive")
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    k = min(rank, s.size)
    approx = (u[:, :k] * s[:k]) @ vt[:k, :]
    energy = float(np.sum(s**2)) or 1.0
    captured = float(np.sum(s[:k] ** 2)) / energy
    return approx, {
        "rank_requested": rank,
        "rank_used": k,
        "spectral_energy_captured": captured,
        "residual_frobenius": float(np.linalg.norm(m - approx)),
        "singular_values": [float(v) for v in s[: k + 1]],
    }


# --------------------------------------------------------------------------- #
# bfloat16 / float16 conversion (artifact codec support)
# --------------------------------------------------------------------------- #


def float32_to_bfloat16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bfloat16, returned as uint16 bit patterns."""
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    bits = arr.view(np.uint32)
    # Rounding bias: 0x7FFF + LSB-of-kept-part for RNE.
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    out = (rounded >> 16).astype(np.uint16)
    # Preserve NaN payload / Inf exactly.
    nan_mask = np.isnan(arr)
    if np.any(nan_mask):
        out[nan_mask] = 0x7FC0
    return out


def bfloat16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    """bfloat16 bit patterns -> float32 (exact widening)."""
    b = np.ascontiguousarray(np.asarray(bits, dtype=np.uint16))
    return (b.astype(np.uint32) << 16).view(np.float32)


def to_float16(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float16)


# --------------------------------------------------------------------------- #
# statistics used by inspection / provenance
# --------------------------------------------------------------------------- #


def tensor_statistics(x: np.ndarray) -> dict:
    arr = np.asarray(x)
    flat = arr.reshape(-1)
    out: dict = {
        "shape": [int(d) for d in arr.shape],
        "dtype": str(arr.dtype),
        "numel": int(flat.size),
        "nbytes": int(arr.nbytes),
    }
    if flat.size == 0:
        out["finite"] = True
        return out
    if arr.dtype.kind in "fc":
        f = flat.astype(np.float64)
        finite = np.isfinite(f)
        out["finite"] = bool(finite.all())
        out["nan_count"] = int(np.isnan(f).sum())
        out["inf_count"] = int(np.isinf(f).sum())
        if finite.any():
            v = f[finite]
            out.update(
                {
                    "mean": float(v.mean()),
                    "std": float(v.std()),
                    "min": float(v.min()),
                    "max": float(v.max()),
                    "abs_mean": float(np.abs(v).mean()),
                    "l2_norm": float(np.linalg.norm(v)),
                    "sparsity": float(np.mean(np.abs(v) < 1e-12)),
                    "kurtosis": float(np.mean(((v - v.mean()) / (v.std() + EPS)) ** 4)),
                }
            )
    else:
        out["finite"] = True
        out["min"] = int(flat.min())
        out["max"] = int(flat.max())
    return out


def gram_spectrum(x: np.ndarray, top_k: int = 8) -> dict:
    """Top singular values of a matrix (weight-spectrum diagnostics)."""
    m = as_float_array(x)
    if m.ndim != 2:
        return {}
    s = np.linalg.svd(m, compute_uv=False)
    k = min(top_k, s.size)
    energy = float(np.sum(s**2)) or 1.0
    return {
        "top_singular_values": [float(v) for v in s[:k]],
        "stable_rank": float(np.sum(s**2) / (s[0] ** 2)) if s.size and s[0] > EPS else 0.0,
        "spectral_energy_top_k": float(np.sum(s[:k] ** 2) / energy),
        "condition": float(s[0] / s[-1]) if s.size and s[-1] > EPS else float("inf"),
    }


def derive_dtype_error(dtype: np.dtype) -> str | None:
    if dtype.kind not in "fiub":
        return f"unsupported dtype {dtype} for weight arithmetic"
    return None


def sum_tensors(values: Iterable[np.ndarray]) -> np.ndarray:
    """Sum with an explicit first element and strict shape agreement."""
    it = iter(values)
    try:
        total = np.array(next(it), copy=True)
    except StopIteration as exc:
        raise ValueError("cannot sum an empty sequence") from exc
    for v in it:
        if v.shape != total.shape:
            raise ValueError(f"shape mismatch while summing: {v.shape} vs {total.shape}")
        total = total + v
    return total
