"""Homology of an s²-model (Section 5, Definition 8).

    H(S²(G)) = S²({B_0^H(M(s)), B_1^H(M(s)), D^H_SNF(M(s)), [rho±(s)]_H}_{s>=1})

Each component ``G(s)`` is replaced by its homology, seen again as an
algebraic hypergraph whose "vertices" are representative 0-cycles, whose
"edges" are representative 1-cycles and whose boundary is the (torsion) block
of the Smith Normal Form.  Transition maps are replaced by the maps induced
in homology, which are well defined because the transition maps are chain
maps (Theorem 3 shows that the result is an isomorphism invariant up to
s²-isomorphism).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chain import ChainComplex, Homology
from .linalg import rank
from .rings import as_ring

__all__ = ["HomologyS2Model", "HomologyLevel"]


@dataclass
class HomologyLevel:
    """Homology of one component ``G(s)``."""

    s: int
    homology: Homology

    @property
    def betti(self) -> tuple:
        return self.homology.betti

    @property
    def torsion(self) -> tuple:
        return self.homology.torsion

    @property
    def vertices(self) -> np.ndarray:
        """``B_0^H``: representative 0-cycles (columns, in the basis V)."""
        return self.homology.H0_generators

    @property
    def edges(self) -> np.ndarray:
        """``B_1^H``: representative 1-cycles (columns, in the basis E)."""
        return self.homology.H1_generators

    @property
    def boundary(self) -> np.ndarray:
        """``D^H_SNF``."""
        return self.homology.boundary_matrix


class HomologyS2Model:
    """``H(S²(G))`` for an :class:`~s2models.models.S2Model`.

    Parameters
    ----------
    model : S2Model
    ring : optional coefficient ring for homology; defaults to the ring of
        the model (the paper takes both rings equal "for simplicity").
    """

    def __init__(self, model, ring=None):
        self.model = model
        self.ring = as_ring(ring) if ring is not None else model.ring
        self._levels: dict[int, HomologyLevel] = {}
        self._complexes: dict[int, ChainComplex] = {}

    def _complex(self, s: int) -> ChainComplex:
        if s not in self._complexes:
            if self.ring == self.model.ring:
                self._complexes[s] = self.model.level(s)
            else:
                self._complexes[s] = ChainComplex(self.model.boundary(s), self.ring)
        return self._complexes[s]

    def level(self, s: int) -> HomologyLevel:
        if s not in self._levels:
            self._levels[s] = HomologyLevel(s, self._complex(s).homology)
        return self._levels[s]

    def levels(self, k: int) -> list:
        return [self.level(s) for s in range(1, k + 1)]

    def _induced(self, f, s_from: int, s_to: int) -> tuple:
        from .chain import ChainMap

        g = ChainMap(f.f0, f.f1, self._complex(s_from), self._complex(s_to), name=f.name)
        return g.homology_map()

    def rho_plus(self, s: int) -> tuple:
        """Matrices of ``[rho+(s)]_H : H(G(s)) -> H(G(s+1))``."""
        return self._induced(self.model.rho_plus(s), s, s + 1)

    def rho_minus(self, s: int) -> tuple:
        """Matrices of ``[rho-(s)]_H : H(G(s)) -> H(G(s-1))``."""
        return self._induced(self.model.rho_minus(s), s, s - 1)

    def betti_sequence(self, k: int) -> list:
        return [self.level(s).betti for s in range(1, k + 1)]

    def summary(self, k: int = 3) -> str:
        """Human readable description of the truncated homology model."""
        lines = [f"H(S²(G)) over {self.ring.name}, levels 1..{k}"]
        for s in range(1, k + 1):
            L = self.level(s)
            lines.append(f"  s={s}: beta0={L.betti[0]}, beta1={L.betti[1]}, "
                         f"torsion={list(L.torsion)}")
            if s < k:
                h0, h1 = self.rho_plus(s)
                r0 = rank(h0, self.ring) if h0.size else 0
                r1 = rank(h1, self.ring) if h1.size else 0
                lines.append(f"     [rho+({s})]_H : H0 ({h0.shape[1]} -> {h0.shape[0]} generators, "
                             f"rank {r0}), H1 ({h1.shape[1]} -> {h1.shape[0]}, rank {r1})")
        return "\n".join(lines)
