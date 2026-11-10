"""Smith Normal Form and related linear algebra (Appendix A of the paper).

Two families of routines are provided:

* :func:`smith_normal_form` – an exact SNF *with transformation matrices*
  ``P @ A @ Q == D`` over Z or over a prime field Z/pZ.  It is written in pure
  Python and is meant for small/medium matrices (it is what we use to build the
  homology bases and the induced homology maps of an s²-model).

* :func:`elementary_divisors` – only the invariant factors of an integer
  matrix.  When `cypari2 <https://github.com/sagemath/cypari2>`_ is installed
  the computation is delegated to PARI/GP's ``matsnf`` (fast, exact), which is
  what makes the experiments on strongly regular graphs (315 x 315 matrices)
  practical.  Otherwise the pure Python algorithm is used.

* :func:`rank_mod_p` – vectorised Gaussian elimination over Z/pZ (used for
  Betti numbers over Z2).

Notation follows the paper: ``A = U_SNF @ D_SNF @ V_SNF`` with ``U_SNF`` and
``V_SNF`` invertible.  With the transforms returned here, ``U_SNF = P^{-1}``
and ``V_SNF = Q^{-1}``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .rings import Ring, Z, as_ring

__all__ = [
    "SNF",
    "smith_normal_form",
    "elementary_divisors",
    "snf_multiset",
    "rank_mod_p",
    "rank",
    "inverse_matrix",
    "set_snf_backend",
    "get_snf_backend",
]


# --------------------------------------------------------------------------
# Smith normal form with transforms (pure Python)
# --------------------------------------------------------------------------
@dataclass
class SNF:
    """Result of a Smith Normal Form factorisation ``P @ A @ Q == D``.

    Attributes
    ----------
    D : diagonal matrix (same shape as ``A``) with ``d_1 | d_2 | ... | d_r``
    P, P_inv : invertible ``n x n`` matrices (row operations)
    Q, Q_inv : invertible ``m x m`` matrices (column operations)
    divisors : the non-zero diagonal entries ``(d_1, ..., d_r)``
    ring : coefficient ring
    """

    D: np.ndarray
    P: np.ndarray
    P_inv: np.ndarray
    Q: np.ndarray
    Q_inv: np.ndarray
    divisors: tuple
    ring: Ring

    @property
    def rank(self) -> int:
        return len(self.divisors)

    @property
    def n_units(self) -> int:
        """Number of invariant factors equal to 1 (index ``t(G) - 1`` in the paper)."""
        return sum(1 for d in self.divisors if d == 1)

    # paper notation
    @property
    def U(self) -> np.ndarray:
        return self.P_inv

    @property
    def V(self) -> np.ndarray:
        return self.Q_inv


def _to_lists(A) -> list:
    A = np.asarray(A)
    return [[int(x) for x in row] for row in A]


def _from_lists(L, shape) -> np.ndarray:
    if not L or not L[0]:
        return np.zeros(shape, dtype=np.int64)
    flat = [x for row in L for x in row]
    big = max(abs(x) for x in flat) >= 2**62
    return np.array(L, dtype=object if big else np.int64).reshape(shape)


def smith_normal_form(A, ring=Z) -> SNF:
    """Smith Normal Form with transformation matrices.

    Returns an :class:`SNF` with ``P @ A @ Q == D`` (coefficients in ``ring``),
    ``D`` diagonal with non-negative entries such that each non-zero entry
    divides the next one.  Over a field all non-zero entries are 1.
    """
    ring = as_ring(ring)
    A = np.asarray(A)
    n, m = A.shape
    p = ring.characteristic
    D = _to_lists(ring.reduce(A))
    P = [[int(i == j) for j in range(n)] for i in range(n)]
    Pi = [[int(i == j) for j in range(n)] for i in range(n)]
    Q = [[int(i == j) for j in range(m)] for i in range(m)]
    Qi = [[int(i == j) for j in range(m)] for i in range(m)]

    def red(x):
        return x % p if p else x

    # elementary operations keeping P A Q = D, Pi = P^-1, Qi = Q^-1 -------
    def swap_rows(i, j):
        if i == j:
            return
        D[i], D[j] = D[j], D[i]
        P[i], P[j] = P[j], P[i]
        for row in Pi:
            row[i], row[j] = row[j], row[i]

    def swap_cols(i, j):
        if i == j:
            return
        for row in D:
            row[i], row[j] = row[j], row[i]
        for row in Q:
            row[i], row[j] = row[j], row[i]
        Qi[i], Qi[j] = Qi[j], Qi[i]

    def add_row(i, j, c):  # row_i += c * row_j
        if c == 0:
            return
        Di, Dj = D[i], D[j]
        for k in range(m):
            if Dj[k]:
                Di[k] = red(Di[k] + c * Dj[k])
        Pi_, Pj = P[i], P[j]
        for k in range(n):
            if Pj[k]:
                Pi_[k] = red(Pi_[k] + c * Pj[k])
        for row in Pi:  # column j of P^-1 -= c * column i
            if row[i]:
                row[j] = red(row[j] - c * row[i])

    def add_col(i, j, c):  # col_i += c * col_j
        if c == 0:
            return
        for row in D:
            if row[j]:
                row[i] = red(row[i] + c * row[j])
        for row in Q:
            if row[j]:
                row[i] = red(row[i] + c * row[j])
        Qi_, Qj = Qi[i], Qi[j]  # row j of Q^-1 -= c * row i
        for k in range(m):
            if Qi_[k]:
                Qj[k] = red(Qj[k] - c * Qi_[k])

    def scale_row(i, u):  # u is a unit
        D[i] = [red(u * x) for x in D[i]]
        P[i] = [red(u * x) for x in P[i]]
        ui = ring.inv(u)
        for row in Pi:
            row[i] = red(row[i] * ui)

    def size(x):
        return abs(x) if p == 0 else (0 if x == 0 else 1)

    t = 0
    while t < min(n, m):
        # choose the non-zero entry of smallest size in the remaining block
        best = None
        for i in range(t, n):
            Di = D[i]
            for j in range(t, m):
                if Di[j] and (best is None or size(Di[j]) < best[0]):
                    best = (size(Di[j]), i, j)
                    if best[0] == 1:
                        break
            if best is not None and best[0] == 1:
                break
        if best is None:
            break
        swap_rows(t, best[1])
        swap_cols(t, best[2])
        while True:
            piv = D[t][t]
            if p:
                for i in range(t + 1, n):
                    if D[i][t]:
                        add_row(i, t, red(-D[i][t] * ring.inv(piv)))
                for j in range(t + 1, m):
                    if D[t][j]:
                        add_col(j, t, red(-D[t][j] * ring.inv(piv)))
                break
            dirty = False
            for i in range(t + 1, n):
                if D[i][t]:
                    add_row(i, t, -(D[i][t] // piv))
                    dirty = dirty or D[i][t] != 0
            for j in range(t + 1, m):
                if D[t][j]:
                    add_col(j, t, -(D[t][j] // piv))
                    dirty = dirty or D[t][j] != 0
            if dirty:
                # bring the smallest remaining entry of row/column t to the pivot
                cand = [(abs(D[i][t]), i, t) for i in range(t + 1, n) if D[i][t]]
                cand += [(abs(D[t][j]), t, j) for j in range(t + 1, m) if D[t][j]]
                _, i, j = min(cand)
                swap_rows(t, i)
                swap_cols(t, j)
                continue
            # divisibility condition d_t | every remaining entry
            bad = None
            for i in range(t + 1, n):
                for j in range(t + 1, m):
                    if D[i][j] % piv:
                        bad = i
                        break
                if bad is not None:
                    break
            if bad is None:
                break
            add_row(t, bad, 1)
        if p:
            if D[t][t] != 1:
                scale_row(t, ring.inv(D[t][t]))
        elif D[t][t] < 0:
            scale_row(t, -1)
        t += 1

    divisors = tuple(D[i][i] for i in range(min(n, m)) if D[i][i] != 0)
    return SNF(
        D=_from_lists(D, (n, m)),
        P=_from_lists(P, (n, n)),
        P_inv=_from_lists(Pi, (n, n)),
        Q=_from_lists(Q, (m, m)),
        Q_inv=_from_lists(Qi, (m, m)),
        divisors=divisors,
        ring=ring,
    )


def inverse_matrix(A, ring=Z) -> np.ndarray:
    """Inverse of a square matrix that is invertible over ``ring``."""
    ring = as_ring(ring)
    A = np.asarray(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square")
    s = smith_normal_form(A, ring)
    if len(s.divisors) != A.shape[0] or any(d != 1 for d in s.divisors):
        raise ValueError(f"matrix is not invertible over {ring.name}")
    # P A Q = I  =>  A^-1 = Q P
    return ring.reduce(ring.matmul(s.Q, s.P))


# --------------------------------------------------------------------------
# Elementary divisors only (fast path through PARI when available)
# --------------------------------------------------------------------------
_BACKEND = None


@lru_cache(maxsize=1)
def _pari():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cypari2

            pari = cypari2.Pari()
            try:
                pari.default("debugmem", 0)  # silence "increasing stack size"
            except Exception:  # pragma: no cover
                pass
            try:
                pari.allocatemem(2**26, 2**32, silent=True)
            except TypeError:  # older cypari2
                pari.allocatemem(2**26, 2**32)
        return pari
    except Exception:  # pragma: no cover - depends on the environment
        return None


def set_snf_backend(name: str | None) -> None:
    """Force the backend for :func:`elementary_divisors` ("pari" or "python")."""
    global _BACKEND
    if name not in (None, "pari", "python"):
        raise ValueError("backend must be 'pari', 'python' or None (auto)")
    if name == "pari" and _pari() is None:
        raise RuntimeError("cypari2 is not installed")
    _BACKEND = name


def get_snf_backend() -> str:
    if _BACKEND is not None:
        return _BACKEND
    return "pari" if _pari() is not None else "python"


def elementary_divisors(A) -> tuple:
    """Non-zero invariant factors ``d_1 | d_2 | ... | d_r`` of an integer matrix.

    The number of returned values is the rank of ``A`` over Q; the multiset of
    elementary divisors in the sense of the paper is this tuple together with
    ``min(n, m) - r`` zeros.
    """
    A = np.asarray(A)
    n, m = A.shape
    if n == 0 or m == 0 or not np.any(A):
        return ()
    if get_snf_backend() == "pari":
        pari = _pari()
        k = max(n, m)
        # pad to a square matrix (matsnf's convention for rectangular input
        # differs between versions); padding only adds zero invariant factors
        X = pari.matrix(k, k)
        rows, cols = np.nonzero(A)
        for i, j in zip(rows.tolist(), cols.tolist()):
            X[i, j] = int(A[i, j])
        d = [abs(int(x)) for x in pari.matsnf(X)]
        return tuple(sorted(x for x in d if x != 0))
    return smith_normal_form(A, Z).divisors


def snf_multiset(A) -> tuple:
    """Hashable SNF summary: ``(n, m, (d_1, ..., d_r))`` (zeros implied)."""
    A = np.asarray(A)
    return (A.shape[0], A.shape[1], elementary_divisors(A))


# --------------------------------------------------------------------------
# Rank over a prime field (vectorised)
# --------------------------------------------------------------------------
def rank_mod_p(A, p: int) -> int:
    """Rank of an integer matrix over Z/pZ (Gaussian elimination)."""
    A = np.mod(np.asarray(A, dtype=object if p > 3_000_000_000 else np.int64), p)
    if A.dtype == object:  # pragma: no cover - huge primes
        return smith_normal_form(A, as_ring(p)).rank
    A = A.astype(np.int64)
    n, m = A.shape
    r = 0
    for c in range(m):
        if r == n:
            break
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        piv = r + nz[0]
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), -1, p)
        A[r] = (A[r] * inv) % p
        col = A[:, c].copy()
        col[r] = 0
        rows = np.nonzero(col)[0]
        if rows.size:
            A[rows] = (A[rows] - np.outer(col[rows], A[r])) % p
        r += 1
    return r


def rank(A, ring=Z) -> int:
    """Rank of ``A`` over ``ring`` (over Z this is the rank over Q)."""
    ring = as_ring(ring)
    if ring.is_field:
        return rank_mod_p(A, ring.characteristic)
    return len(elementary_divisors(A))
