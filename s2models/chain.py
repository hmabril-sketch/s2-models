"""Algebraic hypergraphs: 1-dimensional chain complexes, chain maps and homology.

A chain complex associated to a hypergraph ``G = (V, E, d)`` is

    0 --> C_1 = F[E] --d--> C_0 = F[V] --> 0,

so it is fully described by the matrix of ``d`` (a ``|V| x |E|`` matrix whose
column ``j`` is the boundary of the edge ``e_j``).  The paper calls such an
object an *algebraic hypergraph* ``(V~, E~, F[d])``.

Homology is computed through the Smith Normal Form (Appendix A):
``P @ M @ Q = D``.  The columns ``w_i`` of ``P^{-1}`` and ``c_j`` of ``Q`` are
the new bases of ``F[V]`` and ``F[E]`` in which ``d(c_j) = d_jj * w_j``.
Then (``r`` = rank, ``t`` = index of the first invariant factor != 1):

* ``{w_t, ..., w_n}`` are representative 0-cycles of ``H_0`` (torsion classes
  for ``t <= i <= r`` with order ``d_ii``, free classes for ``i > r``);
* ``{c_{r+1}, ..., c_m}`` are representative 1-cycles of ``H_1`` (always free);
* ``beta_0 = |V| - r`` and ``beta_1 = |E| - r``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .linalg import SNF, elementary_divisors, rank, smith_normal_form
from .rings import Ring, Z, as_ring

__all__ = ["ChainComplex", "ChainMap", "Homology", "identity_map"]


class ChainComplex:
    """A 1-dimensional chain complex ``C_1 --d--> C_0`` over ``ring``.

    Parameters
    ----------
    boundary : array-like, shape (n_vertices, n_edges)
        Matrix of the differential ``d : F[E] -> F[V]`` in the combinatorial
        bases (a vertex-edge incidence matrix with coefficients in ``ring``).
    ring : Ring or str
        Coefficient ring (``"Z"``, ``"Z2"``, ...).
    """

    def __init__(self, boundary, ring=Z):
        self.ring = as_ring(ring)
        self.boundary = self.ring.reduce(np.asarray(boundary))
        if self.boundary.ndim != 2:
            raise ValueError("boundary must be a 2-D matrix")

    # ----------------------------------------------------------- dimensions
    @property
    def n_vertices(self) -> int:
        return self.boundary.shape[0]

    @property
    def n_edges(self) -> int:
        return self.boundary.shape[1]

    @property
    def dims(self) -> tuple:
        return (self.n_vertices, self.n_edges)

    def euler_characteristic(self) -> int:
        """``chi = dim C_0 - dim C_1`` (= ``beta_0 - beta_1``)."""
        return self.n_vertices - self.n_edges

    @property
    def coboundary(self) -> np.ndarray:
        """Matrix of the codifferential ``delta : F[V] -> F[E]`` (transpose)."""
        return self.boundary.T.copy()

    # ------------------------------------------------------------- homology
    @cached_property
    def snf(self) -> SNF:
        return smith_normal_form(self.boundary, self.ring)

    @cached_property
    def rank(self) -> int:
        return rank(self.boundary, self.ring)

    def betti_numbers(self) -> tuple:
        """``(beta_0, beta_1)`` over the coefficient ring (rank of the free part)."""
        r = self.rank
        return (self.n_vertices - r, self.n_edges - r)

    def elementary_divisors(self) -> tuple:
        """Non-zero invariant factors of the boundary matrix (over Z)."""
        if self.ring.is_field:
            return (1,) * self.rank
        return elementary_divisors(self.boundary)

    def torsion(self) -> tuple:
        """Torsion coefficients of ``H_0`` (invariant factors different from 1)."""
        return tuple(d for d in self.elementary_divisors() if d != 1)

    @cached_property
    def homology(self) -> "Homology":
        s = self.snf
        r = s.rank
        t = s.n_units  # number of unit invariant factors (t(G) - 1 in the paper)
        W = s.P_inv  # columns = new basis of C_0
        C = s.Q  # columns = new basis of C_1
        return Homology(
            complex=self,
            snf=s,
            rank=r,
            n_units=t,
            torsion=tuple(s.divisors[t:]),
            H0_generators=W[:, t:],
            H1_generators=C[:, r:],
        )

    def __repr__(self) -> str:
        return f"ChainComplex(|V|={self.n_vertices}, |E|={self.n_edges}, ring={self.ring.name})"


@dataclass
class Homology:
    """Homology of a 1-D chain complex, seen as an algebraic hypergraph.

    ``H0_generators`` (columns) are representative 0-cycles: the first
    ``len(torsion)`` ones are torsion classes (orders given by ``torsion``),
    the remaining ``beta_0`` ones are free.  ``H1_generators`` are
    representative 1-cycles (a basis of ``ker d``).
    The boundary of the homology algebraic hypergraph,
    ``D^H_SNF`` in the paper, is ``boundary_matrix``.
    """

    complex: ChainComplex
    snf: SNF
    rank: int
    n_units: int
    torsion: tuple
    H0_generators: np.ndarray
    H1_generators: np.ndarray

    @property
    def betti(self) -> tuple:
        return (self.complex.n_vertices - self.rank, self.complex.n_edges - self.rank)

    @property
    def boundary_matrix(self) -> np.ndarray:
        """``D^H_SNF``: the SNF diagonal block after removing the unit part."""
        t = self.n_units
        return self.snf.D[t:, t:]

    # coordinates of chains in the SNF bases
    def vertex_coordinates(self, x) -> np.ndarray:
        """Coordinates of a 0-chain (or matrix of column chains) in the basis ``w``."""
        return self.complex.ring.reduce(self.complex.ring.matmul(self.snf.P, np.asarray(x)))

    def edge_coordinates(self, y) -> np.ndarray:
        """Coordinates of a 1-chain (or matrix of column chains) in the basis ``c``."""
        return self.complex.ring.reduce(self.complex.ring.matmul(self.snf.Q_inv, np.asarray(y)))

    def class_of_0chain(self, x) -> np.ndarray:
        """Homology class of 0-chains (columns of ``x``) as coordinates on the generators.

        Torsion coordinates are reduced modulo the corresponding invariant factor.
        """
        x = np.asarray(x)
        squeeze = x.ndim == 1
        if squeeze:
            x = x.reshape(-1, 1)
        coords = self.vertex_coordinates(x)[self.n_units :, :].astype(object)
        for k, d in enumerate(self.torsion):
            coords[k, :] = [int(v) % d for v in coords[k, :]]
        coords = coords.astype(np.int64)
        return coords[:, 0] if squeeze else coords

    def class_of_1cycle(self, y) -> np.ndarray:
        """Homology class of 1-cycles (columns of ``y``) in the basis of ``H_1``."""
        y = np.asarray(y)
        squeeze = y.ndim == 1
        if squeeze:
            y = y.reshape(-1, 1)
        coords = self.edge_coordinates(y)
        if np.any(coords[: self.rank, :] != 0):
            raise ValueError("the given 1-chain is not a cycle")
        out = coords[self.rank :, :]
        return out[:, 0] if squeeze else out


@dataclass
class ChainMap:
    """A pair ``f = (f_0, f_1)`` of linear maps between 1-D chain complexes.

    ``f0`` has shape ``(|V'|, |V|)`` and ``f1`` shape ``(|E'|, |E|)``; the chain
    condition is ``f0 @ d == d' @ f1``.
    """

    f0: np.ndarray
    f1: np.ndarray
    source: ChainComplex
    target: ChainComplex
    name: str = field(default="")

    def __post_init__(self):
        ring = self.source.ring
        self.f0 = ring.reduce(np.asarray(self.f0))
        self.f1 = ring.reduce(np.asarray(self.f1))
        if self.f0.shape != (self.target.n_vertices, self.source.n_vertices):
            raise ValueError(f"f0 has shape {self.f0.shape}, expected "
                             f"{(self.target.n_vertices, self.source.n_vertices)}")
        if self.f1.shape != (self.target.n_edges, self.source.n_edges):
            raise ValueError(f"f1 has shape {self.f1.shape}, expected "
                             f"{(self.target.n_edges, self.source.n_edges)}")

    @property
    def ring(self) -> Ring:
        return self.source.ring

    def is_chain_map(self) -> bool:
        ring = self.ring
        lhs = ring.reduce(ring.matmul(self.f0, self.source.boundary))
        rhs = ring.reduce(ring.matmul(self.target.boundary, self.f1))
        return bool(np.array_equal(lhs, rhs))

    def __matmul__(self, other: "ChainMap") -> "ChainMap":
        """Composition ``self ∘ other``."""
        if other.target.dims != self.source.dims:
            raise ValueError("maps are not composable")
        ring = self.ring
        return ChainMap(
            ring.reduce(ring.matmul(self.f0, other.f0)),
            ring.reduce(ring.matmul(self.f1, other.f1)),
            other.source,
            self.target,
            name=f"{self.name}∘{other.name}" if self.name and other.name else "",
        )

    def homology_map(self) -> tuple:
        """Matrices of the induced map ``[f]_H = ([f]_0, [f]_1)`` on homology.

        ``[f]_0`` maps the generators of ``H_0(source)`` (columns of
        ``source.homology.H0_generators``) to coordinates in ``H_0(target)``;
        likewise for ``[f]_1``.  This is the construction of Appendix A:
        restrict ``snf(f_i)`` to the bases of representative cycles.
        """
        if not self.is_chain_map():
            raise ValueError("homology is only functorial for chain maps")
        Hs, Ht = self.source.homology, self.target.homology
        ring = self.ring
        img0 = ring.reduce(ring.matmul(self.f0, Hs.H0_generators))
        img1 = ring.reduce(ring.matmul(self.f1, Hs.H1_generators))
        h0 = Ht.class_of_0chain(img0) if img0.size else np.zeros((Ht.H0_generators.shape[1], 0), dtype=np.int64)
        h1 = Ht.class_of_1cycle(img1) if img1.size else np.zeros((Ht.H1_generators.shape[1], 0), dtype=np.int64)
        return h0, h1

    def __repr__(self) -> str:
        return (f"ChainMap({self.name or 'f'}: {self.source.dims} -> {self.target.dims}, "
                f"ring={self.ring.name})")


def identity_map(C: ChainComplex) -> ChainMap:
    return ChainMap(C.ring.eye(C.n_vertices), C.ring.eye(C.n_edges), C, C, name="1")
