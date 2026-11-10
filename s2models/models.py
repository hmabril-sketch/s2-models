"""Scale-space topological models (s²-models) of a hypergraph (Section 4).

Definition 4.  An s²-model ``S²({V(s), E(s), d(s), rho±(s)}_{s>=1})`` of a
hypergraph ``G = (V, E, d)`` is a one-parameter family of chain complexes
``G(s) = (V(s), E(s), d(s))`` with transition functions
``rho+(s) : G(s) -> G(s+1)`` and ``rho-(s) : G(s) -> G(s-1)`` such that

* [S2.1] ``G(1) = C_*(G)``;
* [S2.2] ``|V(s)| - |E(s)| = chi(G)`` for all ``s``;
* [S2.3] ``d(s)`` is a differential of ``G(s)`` and ``d(1) = d``;
* [S2.4] the transition maps are chain maps;
* [S2.5] any two compositions of ``n`` maps ``rho+`` and ``m`` maps ``rho-``
  from ``G(s)`` to ``G(s+n-m)`` coincide.

All models implemented here keep ``V(s) = V`` and ``E(s) = E`` (the setting
of the paper's experiments), so they are described in matrix terms by the
sequence of incidence matrices ``M(s)`` and transition matrices.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
import numpy as np

from .chain import ChainComplex, ChainMap
from .homology_model import HomologyS2Model
from .hypergraph import Hypergraph
from .linalg import inverse_matrix
from .rings import Z2, Ring, as_ring

__all__ = [
    "S2Model",
    "BoundaryScaleModel",
    "ColorRefinementModel",
    "IteratedLaplacianModel",
    "TrivialAdjacencyModel",
    "BasisChangeModel",
]


def _as_hypergraph(G) -> Hypergraph:
    if isinstance(G, Hypergraph):
        return G
    try:  # networkx graph
        import networkx as nx

        if isinstance(G, nx.Graph):
            return Hypergraph.from_networkx(G)
    except ImportError:  # pragma: no cover
        pass
    return Hypergraph(np.asarray(G))


class S2Model(ABC):
    """Base class of s²-models whose components share the vertices and edges of ``G``.

    Subclasses define the boundary matrix ``M(s)`` (``_boundary``) and the
    matrices ``(f0, f1)`` of the transition maps (``_rho_plus``/``_rho_minus``).
    Results are cached, so building the levels of a truncated model is cheap.
    """

    kind = "generic"

    def __init__(self, G, ring=Z2):
        self.hypergraph = _as_hypergraph(G)
        self.ring: Ring = as_ring(ring)
        self.M = self.ring.reduce(self.hypergraph.M)
        self._boundaries: dict[int, np.ndarray] = {}
        self._levels: dict[int, ChainComplex] = {}

    # ---- to be provided by subclasses ------------------------------------
    @abstractmethod
    def _boundary(self, s: int) -> np.ndarray: ...

    @abstractmethod
    def _rho_plus(self, s: int) -> tuple: ...

    @abstractmethod
    def _rho_minus(self, s: int) -> tuple: ...

    # ---- public API -------------------------------------------------------
    @property
    def n_vertices(self) -> int:
        return self.M.shape[0]

    @property
    def n_edges(self) -> int:
        return self.M.shape[1]

    def boundary(self, s: int) -> np.ndarray:
        """Matrix of ``d(s)`` (incidence matrix of the algebraic hypergraph ``G(s)``)."""
        if s < 1:
            raise ValueError("scales start at s = 1")
        if s not in self._boundaries:
            self._boundaries[s] = self.ring.reduce(self._boundary(s))
        return self._boundaries[s]

    def level(self, s: int) -> ChainComplex:
        """The algebraic hypergraph (chain complex) ``G(s)``."""
        if s not in self._levels:
            self._levels[s] = ChainComplex(self.boundary(s), self.ring)
        return self._levels[s]

    def levels(self, k: int) -> list:
        """Components ``G(1), ..., G(k)`` of the truncated model ``S²_k``."""
        return [self.level(s) for s in range(1, k + 1)]

    def rho_plus(self, s: int) -> ChainMap:
        """Transition map ``rho+(s) : G(s) -> G(s+1)``."""
        f0, f1 = self._rho_plus(s)
        return ChainMap(f0, f1, self.level(s), self.level(s + 1), name=f"rho+({s})")

    def rho_minus(self, s: int) -> ChainMap:
        """Transition map ``rho-(s) : G(s) -> G(s-1)`` (defined for ``s >= 2``)."""
        if s < 2:
            raise ValueError("rho-(1) goes to the trivial complex; use s >= 2")
        f0, f1 = self._rho_minus(s)
        return ChainMap(f0, f1, self.level(s), self.level(s - 1), name=f"rho-({s})")

    def rho_plus_composite(self, j: int) -> ChainMap:
        """``rho+(j) ∘ ... ∘ rho+(1) : G(1) -> G(j+1)``."""
        f = self.rho_plus(1)
        for s in range(2, j + 1):
            f = self.rho_plus(s) @ f
        return f

    def rho_minus_composite(self, j: int) -> ChainMap:
        """``rho-(2) ∘ rho-(3) ∘ ... ∘ rho-(j+1) : G(j+1) -> G(1)``."""
        f = self.rho_minus(j + 1)
        for s in range(j, 1, -1):
            f = self.rho_minus(s) @ f
        return f

    def homology(self, ring=None) -> HomologyS2Model:
        """The homology s²-model ``H(S²(G))`` of Definition 8."""
        return HomologyS2Model(self, ring)

    # ---- axioms -----------------------------------------------------------
    def verify(self, k: int = 3, max_steps: int = 3) -> dict:
        """Check the conditions S2.1–S2.5 on the truncated model ``S²_k``.

        S2.5 is checked for every start level ``s <= k`` and every word of at
        most ``max_steps`` transition maps that stays inside ``1..k+1``.
        Returns a dict ``{condition: bool}``.
        """
        ring = self.ring
        out = {}
        out["S2.1 G(1) = C*(G)"] = bool(np.array_equal(self.boundary(1), self.M))
        chi = self.hypergraph.euler_characteristic()
        out["S2.2 Euler-Poincaré preserved"] = all(
            self.level(s).euler_characteristic() == chi for s in range(1, k + 2))
        out["S2.3 d(1) = d"] = out["S2.1 G(1) = C*(G)"]
        ok = True
        for s in range(1, k + 1):
            ok &= self.rho_plus(s).is_chain_map()
            if s >= 2:
                ok &= self.rho_minus(s).is_chain_map()
        ok &= self.rho_minus(k + 1).is_chain_map()
        out["S2.4 transition maps are chain maps"] = bool(ok)

        ok = True
        for s in range(1, k + 1):
            for length in range(1, max_steps + 1):
                by_target: dict = {}
                for word in itertools.product((+1, -1), repeat=length):
                    cur, f = s, None
                    valid = True
                    for step in word:
                        if step == +1:
                            if cur + 1 > k + 1:
                                valid = False
                                break
                            g = self.rho_plus(cur)
                        else:
                            if cur - 1 < 1:
                                valid = False
                                break
                            g = self.rho_minus(cur)
                        f = g if f is None else g @ f
                        cur += step
                    if not valid:
                        continue
                    key = (cur, word.count(+1), word.count(-1))
                    mats = (ring.reduce(f.f0), ring.reduce(f.f1))
                    if key in by_target:
                        ref = by_target[key]
                        ok &= np.array_equal(ref[0], mats[0]) and np.array_equal(ref[1], mats[1])
                    else:
                        by_target[key] = mats
        out["S2.5 composition invariance"] = bool(ok)
        return out

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(|V|={self.n_vertices}, |E|={self.n_edges}, "
                f"ring={self.ring.name})")


# ---------------------------------------------------------------------------
# Incidence s²-models
# ---------------------------------------------------------------------------
class BoundaryScaleModel(S2Model):
    """Boundary-scale model (Definition 7).

    ``S²(V, E, {d(δd)^{s-1}, (dδ, 1_E), (1_V, δd)})``, i.e. in matrix terms

        M(s)   = M (Mᵀ M)^{s-1}
        rho+(s) = (M Mᵀ, I_E) : G(s) -> G(s+1)
        rho-(s) = (I_V, Mᵀ M) : G(s) -> G(s-1)

    (The text of Definition 7 writes the exponent ``s``; the matrix form and
    Fig. 5 use ``s-1`` so that ``G(1) = C_*(G)``, which is what we implement.)
    ``M Mᵀ`` is the signless vertex Laplacian ``dδ`` and ``Mᵀ M`` the signless
    edge Laplacian ``δd``.  The experiments of the paper use ``ring = Z2``.
    """

    kind = "boundary-scale"

    def __init__(self, G, ring=Z2):
        super().__init__(G, ring)
        r = self.ring
        self.vertex_laplacian = r.reduce(r.matmul(self.M, self.M.T))  # dδ
        self.edge_laplacian = r.reduce(r.matmul(self.M.T, self.M))  # δd

    def _boundary(self, s):
        if s == 1:
            return self.M
        return self.ring.matmul(self.boundary(s - 1), self.edge_laplacian)

    def _rho_plus(self, s):
        return self.vertex_laplacian, self.ring.eye(self.n_edges)

    def _rho_minus(self, s):
        return self.ring.eye(self.n_vertices), self.edge_laplacian

    def vertex_laplacian_power(self, j: int) -> np.ndarray:
        """``(M Mᵀ)^j`` – the ``f0`` part of ``rho+(j) ∘ ... ∘ rho+(1)``."""
        if j == 0:
            return self.ring.eye(self.n_vertices)
        cache = self.__dict__.setdefault("_vpow", {1: self.vertex_laplacian})
        if j not in cache:
            cache[j] = self.ring.matmul(self.vertex_laplacian_power(j - 1), self.vertex_laplacian)
        return cache[j]

    def edge_laplacian_power(self, j: int) -> np.ndarray:
        """``(Mᵀ M)^j`` – the ``f1`` part of ``rho-(2) ∘ ... ∘ rho-(j+1)``."""
        if j == 0:
            return self.ring.eye(self.n_edges)
        cache = self.__dict__.setdefault("_epow", {1: self.edge_laplacian})
        if j not in cache:
            cache[j] = self.ring.matmul(self.edge_laplacian_power(j - 1), self.edge_laplacian)
        return cache[j]


class BasisChangeModel(S2Model):
    """Incidence model given by changes of basis ``P`` (vertices) and ``Q`` (edges).

    ``S²(V, E, {P^{s-1} M Q^{s-1}, (P, Q^{-1}), (P^{-1}, Q)})`` with ``P`` and
    ``Q`` square matrices invertible over the coefficient ring.
    """

    kind = "basis-change"

    def __init__(self, G, P, Q, ring=Z2):
        super().__init__(G, ring)
        r = self.ring
        self.P = r.reduce(np.asarray(P))
        self.Q = r.reduce(np.asarray(Q))
        self.P_inv = inverse_matrix(self.P, r)
        self.Q_inv = inverse_matrix(self.Q, r)

    def _boundary(self, s):
        if s == 1:
            return self.M
        r = self.ring
        return r.matmul(r.matmul(self.P, self.boundary(s - 1)), self.Q)

    def _rho_plus(self, s):
        return self.P, self.Q_inv

    def _rho_minus(self, s):
        return self.P_inv, self.Q


# ---------------------------------------------------------------------------
# Adjacency s²-models: d(s) = d for every s
# ---------------------------------------------------------------------------
class _ConstantModel(S2Model):
    """Adjacency model generated by a single chain map ``f : C_*(G) -> C_*(G)``
    used for every transition, ``rho+(s) = rho-(s+1) = f``."""

    kind = "adjacency"

    def _boundary(self, s):
        return self.M

    @abstractmethod
    def _map(self) -> tuple: ...

    def _rho_plus(self, s):
        return self._map()

    def _rho_minus(self, s):
        return self._map()


class TrivialAdjacencyModel(_ConstantModel):
    """``S²(V, E, d, (1, 1), (1, 1))``."""

    def _map(self):
        return self.ring.eye(self.n_vertices), self.ring.eye(self.n_edges)


class ColorRefinementModel(_ConstantModel):
    """Color-refinement (CR) s²-model ``S²(V, E, d, (dδ, δd), (dδ, δd))``.

    Inspired by Weisfeiler–Leman colour refinement: the transition maps are the
    signless vertex and edge Laplacians.
    """

    def __init__(self, G, ring=Z2):
        super().__init__(G, ring)
        r = self.ring
        self.vertex_laplacian = r.reduce(r.matmul(self.M, self.M.T))
        self.edge_laplacian = r.reduce(r.matmul(self.M.T, self.M))

    def _map(self):
        return self.vertex_laplacian, self.edge_laplacian


class IteratedLaplacianModel(ColorRefinementModel):
    """Generalised CR model with ``f = (Σ_{i<=k} (-1)^i (dδ)^i, Σ_{i<=k} (-1)^i (δd)^i)``."""

    def __init__(self, G, k: int = 2, ring=Z2):
        super().__init__(G, ring)
        r = self.ring
        f0 = r.zeros((self.n_vertices, self.n_vertices))
        f1 = r.zeros((self.n_edges, self.n_edges))
        for i in range(k + 1):
            sign = -1 if i % 2 else 1
            f0 = r.reduce(f0 + sign * r.matpow(self.vertex_laplacian, i))
            f1 = r.reduce(f1 + sign * r.matpow(self.edge_laplacian, i))
        self._f = (f0, f1)

    def _map(self):
        return self._f
