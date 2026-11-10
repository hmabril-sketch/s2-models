"""Isomorphism-invariant certificates computed from s²-models (Section 6).

Intra-analysis metrics (evaluated on every component ``G(s)``):

* ``omega_d``   – increasing ordered sequence of vertex degrees of ``G(s)``;
* ``omega_beta`` – Z2 Betti numbers ``(beta_0, beta_1)`` of ``G(s)`` (free homology);
* ``omega_i``   – increasing ordered sequence of the elementary divisors != 1
  of the incidence matrix of ``G(s)`` taken with coefficients in Z
  (torsion of ``H_0``).  The diagonal of the SNF has ``min(|V|, |E|)``
  entries, so the zero divisors (which encode the rank) are included.

Inter-analysis metrics (transition maps):

* ``omega_rho+`` – SNF multisets (over Z) of the matrices of
  ``rho+(1)``, ``rho+(2)∘rho+(1)``, ``rho+(3)∘rho+(2)∘rho+(1)``, ...;
* ``omega_rho-`` – SNF multisets of ``rho-(2)``, ``rho-(2)∘rho-(3)``, ...

A truncated model ``S²_k`` contains ``G(1), ..., G(k)`` and the transition
maps between them, so it carries the composites of length ``1..k-1``.  We
attach to scale ``s >= 2`` the composites joining ``G(1)`` and ``G(s)``
(``rho+(s-1)∘...∘rho+(1)`` and ``rho-(2)∘...∘rho-(s)``); scale 1 carries none.
For the boundary-scale model these composites are ``((M Mᵀ)^{s-1}, I_E)``
and ``(I_V, (Mᵀ M)^{s-1})``.  This reading reproduces the paper's Tables 2–5
(see ``results/``).  All metrics are invariant under permutations of vertices
and edges, so different values certify non-isomorphism.

Hybrid and inter-analysis certificates inspired by colour refinement
(:func:`cr_inter_degree_certificate`, :func:`hybrid_certificate`) are also
provided.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .linalg import elementary_divisors, rank_mod_p, snf_multiset
from .models import BoundaryScaleModel, S2Model

__all__ = [
    "METRICS",
    "level_features",
    "model_features",
    "certificate",
    "omega_d",
    "omega_beta",
    "omega_i",
    "omega_rho_plus",
    "omega_rho_minus",
    "unique_fraction",
    "isomorphism_test",
    "IsoTestResult",
    "cr_inter_degree_certificate",
    "hybrid_certificate",
]

METRICS = ("beta", "d", "i", "rho-", "rho+")
METRIC_SYMBOLS = {"beta": "ω_β", "d": "ω_d", "i": "ω_i", "rho-": "ω_ρ-", "rho+": "ω_ρ+"}


def _is_identity(A: np.ndarray) -> bool:
    return A.shape[0] == A.shape[1] and np.array_equal(A, np.eye(A.shape[0], dtype=A.dtype))


def _nonunit_divisors(A: np.ndarray) -> tuple:
    """Diagonal entries != 1 of the SNF of ``A`` over Z, zeros included."""
    ed = elementary_divisors(A)
    zeros = min(A.shape) - len(ed)
    return tuple(d for d in ed if d != 1) + (0,) * zeros


def _snf_summary(A: np.ndarray) -> tuple:
    if _is_identity(A):
        return (A.shape[0], A.shape[1], (1,) * A.shape[0])
    return snf_multiset(A)


def _composite_plus(model: S2Model, j: int) -> tuple:
    if isinstance(model, BoundaryScaleModel):
        return model.vertex_laplacian_power(j), model.ring.eye(model.n_edges)
    f = model.rho_plus_composite(j)
    return f.f0, f.f1


def _composite_minus(model: S2Model, j: int) -> tuple:
    if isinstance(model, BoundaryScaleModel):
        return model.ring.eye(model.n_vertices), model.edge_laplacian_power(j)
    f = model.rho_minus_composite(j)
    return f.f0, f.f1


# ---------------------------------------------------------------------------
# per-level features
# ---------------------------------------------------------------------------
def level_features(model: S2Model, s: int, metrics: Iterable[str] = METRICS) -> dict:
    """Values of the requested metrics at scale ``s``.

    ``rho+`` at scale ``s`` is the SNF of ``rho+(s-1)∘...∘rho+(1)`` and
    ``rho-`` at scale ``s`` the SNF of ``rho-(2)∘...∘rho-(s)`` (both are
    ``None`` at ``s = 1``).
    """
    metrics = set(metrics)
    out = {}
    B = model.boundary(s)
    if "d" in metrics:
        out["d"] = tuple(sorted(int(x) for x in np.count_nonzero(B, axis=1)))
    if "beta" in metrics:
        r = rank_mod_p(B, 2)
        out["beta"] = (B.shape[0] - r, B.shape[1] - r)
    if "i" in metrics:
        out["i"] = _nonunit_divisors(B)
    if "rho+" in metrics:
        if s == 1:
            out["rho+"] = None
        else:
            f0, f1 = _composite_plus(model, s - 1)
            out["rho+"] = (_snf_summary(f0), _snf_summary(f1))
    if "rho-" in metrics:
        if s == 1:
            out["rho-"] = None
        else:
            f0, f1 = _composite_minus(model, s - 1)
            out["rho-"] = (_snf_summary(f0), _snf_summary(f1))
    return out


def model_features(model: S2Model, k: int = 3, metrics: Iterable[str] = METRICS) -> list:
    """List of :func:`level_features` for ``s = 1..k``."""
    return [level_features(model, s, metrics) for s in range(1, k + 1)]


def certificate(features: Sequence[dict], metrics: Iterable[str] = METRICS, k: int | None = None) -> tuple:
    """Certificate of the truncated model ``S²_k`` built from precomputed features."""
    k = len(features) if k is None else k
    metrics = tuple(metrics)
    return tuple(tuple(features[s][m] for m in metrics) for s in range(k))


def _single(model, k, metric):
    return tuple(level_features(model, s, (metric,))[metric] for s in range(1, k + 1))


def omega_d(model: S2Model, k: int = 3) -> tuple:
    return _single(model, k, "d")


def omega_beta(model: S2Model, k: int = 3) -> tuple:
    return _single(model, k, "beta")


def omega_i(model: S2Model, k: int = 3) -> tuple:
    return _single(model, k, "i")


def omega_rho_plus(model: S2Model, k: int = 3) -> tuple:
    return _single(model, k, "rho+")


def omega_rho_minus(model: S2Model, k: int = 3) -> tuple:
    return _single(model, k, "rho-")


# ---------------------------------------------------------------------------
# distinguishing power
# ---------------------------------------------------------------------------
def unique_fraction(certificates: Sequence) -> float:
    """Percentage of objects whose certificate is not shared with any other one.

    This is the quantity reported in Tables 2–5 of the paper ("percentage of
    graphs that can be uniquely distinguished").
    """
    if not certificates:
        return float("nan")
    c = Counter(certificates)
    return 100.0 * sum(1 for x in certificates if c[x] == 1) / len(certificates)


@dataclass
class IsoTestResult:
    distinguished: bool
    level: int | None = None
    metric: str | None = None

    def __str__(self) -> str:
        if self.distinguished:
            return f"non-isomorphic (certified by {self.metric} at level s={self.level})"
        return "not distinguished (possibly isomorphic)"


def isomorphism_test(G, H, k: int = 3, metrics: Iterable[str] = METRICS,
                     model_cls=BoundaryScaleModel, ring="Z2") -> IsoTestResult:
    """Intra/inter-analysis s²-HI test.

    Returns ``distinguished=True`` with the first level and metric whose values
    differ (which proves that ``G`` and ``H`` are not isomorphic), or
    ``distinguished=False`` if the truncated certificates coincide.
    """
    mG, mH = model_cls(G, ring=ring), model_cls(H, ring=ring)
    if mG.M.shape != mH.M.shape:
        return IsoTestResult(True, 1, "size")
    metrics = tuple(metrics)
    for s in range(1, k + 1):
        for m in metrics:
            if level_features(mG, s, (m,))[m] != level_features(mH, s, (m,))[m]:
                return IsoTestResult(True, s, m)
    return IsoTestResult(False)


# ---------------------------------------------------------------------------
# colour-refinement style certificates (Section 6, inter/hybrid analysis)
# ---------------------------------------------------------------------------
def cr_inter_degree_certificate(G, k: int = 3, ring="Z2") -> tuple:
    """Vertex inter-degree certificate of the CR s²-model.

    For each vertex ``v`` the formal sum ``sum_{1<=i<=k} |(dδ)^{i-1}(v)|`` is
    encoded as the tuple of degrees (number of non-null coefficients);
    the certificate is the sorted multiset of these tuples.
    """
    from .models import ColorRefinementModel

    m = ColorRefinementModel(G, ring=ring)
    L = m.vertex_laplacian
    cur = m.ring.eye(m.n_vertices)
    cols = []
    for _ in range(k):
        cols.append(np.count_nonzero(cur, axis=0))
        cur = m.ring.matmul(L, cur)
    per_vertex = sorted(zip(*[c.tolist() for c in cols]))
    return tuple(per_vertex)


def hybrid_certificate(G, k: int = 3, ring="Z2") -> tuple:
    """Hybrid scale-space certificate inspired by the WL strategy.

    For a vertex ``v``: ``sum_{0<=i<=k} (δ(dδ)^i + (dδ)^i)(v)`` in ``F[V ∪ E]``;
    for an edge ``e``: ``sum_{0<=i<=k} (d(δd)^i + (δd)^i)(e)``.  Each chain is
    summarised by the multiset of its non-zero coefficients per scale, and the
    certificate is the pair of sorted multisets over vertices and edges.
    """
    m = BoundaryScaleModel(G, ring=ring)
    r = m.ring
    M, Lv, Le = m.M, m.vertex_laplacian, m.edge_laplacian

    def summarise(X):
        out = []
        for col in X.T:
            nz = col[col != 0]
            out.append(tuple(sorted(int(x) for x in nz)))
        return out

    v_parts, e_parts = [], []
    Pv, Pe = r.eye(m.n_vertices), r.eye(m.n_edges)
    for _ in range(k + 1):
        v_parts.append(list(zip(summarise(r.matmul(M.T, Pv)), summarise(Pv))))
        e_parts.append(list(zip(summarise(r.matmul(M, Pe)), summarise(Pe))))
        Pv, Pe = r.matmul(Lv, Pv), r.matmul(Le, Pe)
    v_cert = tuple(sorted(zip(*v_parts)))
    e_cert = tuple(sorted(zip(*e_parts)))
    return (v_cert, e_cert)
