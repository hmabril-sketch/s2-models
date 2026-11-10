"""Cellular hypergraphs (Section 3.1 of the paper).

A (cellular) hypergraph is ``G = ((V, l_V), (E, l_E), I_G)`` where vertices and
edges are enumerated and ``I_G : V x E -> N`` is the vertex-edge incidence map
(multiple degrees of contact are allowed).  Edges are primary entities, so two
different edges may have the same set of vertices.  Everything is stored in
the vertex-edge incidence matrix ``M_ve(G)`` of shape ``|V| x |E|``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from .chain import ChainComplex
from .rings import Z, as_ring

__all__ = ["Hypergraph"]


class Hypergraph:
    """A cellular hypergraph given by its vertex-edge incidence matrix.

    Parameters
    ----------
    incidence : array-like of non-negative integers, shape (|V|, |E|)
        ``incidence[i, j] = I_G(v_i, e_j)``, the degree of contact between
        vertex ``v_i`` and edge ``e_j``.
    vertex_labels, edge_labels : optional sequences
        Names for the vertices and edges (the enumeration maps ``l_V``, ``l_E``
        are the positions in these sequences).

    Examples
    --------
    The hypergraph of Fig. 1 of the paper:

    >>> G = Hypergraph.from_edges(6, [{1: 1, 2: 3}, {}, {1: 3, 3: 1, 4: 1},
    ...                               {3: 1}, {3: 2, 4: 1, 5: 1}], one_based=True)
    >>> G.euler_characteristic(), G.connectivity_euler_characteristic()
    (1, 2)
    """

    def __init__(self, incidence, vertex_labels: Sequence | None = None,
                 edge_labels: Sequence | None = None):
        M = np.asarray(incidence)
        if M.ndim != 2:
            raise ValueError("incidence must be a 2-D matrix")
        if M.size and (M.min() < 0):
            raise ValueError("incidence degrees must be non-negative integers")
        self.M = M.astype(np.int64)
        n, m = self.M.shape
        self.vertex_labels = list(vertex_labels) if vertex_labels is not None else [f"v{i+1}" for i in range(n)]
        self.edge_labels = list(edge_labels) if edge_labels is not None else [f"e{j+1}" for j in range(m)]
        if len(self.vertex_labels) != n or len(self.edge_labels) != m:
            raise ValueError("label lengths do not match the incidence matrix")

    # ------------------------------------------------------------ builders
    @classmethod
    def from_edges(cls, n_vertices: int, edges: Iterable, one_based: bool = False,
                   **kwargs) -> "Hypergraph":
        """Build a hypergraph from boundaries of edges.

        Each edge is either a mapping ``{vertex: degree_of_contact}`` or an
        iterable of vertices (a vertex repeated ``k`` times has contact ``k``).
        This is the boundary combinatorial map ``d_G(e) = sum n_v v``.
        """
        edges = list(edges)
        M = np.zeros((n_vertices, len(edges)), dtype=np.int64)
        off = 1 if one_based else 0
        for j, e in enumerate(edges):
            items = e.items() if isinstance(e, Mapping) else ((v, 1) for v in e)
            for v, k in items:
                M[v - off, j] += int(k)
        return cls(M, **kwargs)

    @classmethod
    def from_networkx(cls, graph) -> "Hypergraph":
        """Classical (multi)graph as a hypergraph: every edge has degree and
        degree of contact 2 (a self-loop has a single vertex with contact 2)."""
        nodes = list(graph.nodes())
        index = {v: i for i, v in enumerate(nodes)}
        if graph.is_multigraph():
            edge_list = list(graph.edges(keys=True))
            pairs = [(u, v) for u, v, _ in edge_list]
        else:
            edge_list = list(graph.edges())
            pairs = edge_list
        M = np.zeros((len(nodes), len(pairs)), dtype=np.int64)
        for j, (u, v) in enumerate(pairs):
            M[index[u], j] += 1
            M[index[v], j] += 1
        return cls(M, vertex_labels=[str(v) for v in nodes],
                   edge_labels=[f"{u}-{v}" for u, v in pairs])

    @classmethod
    def from_graph6(cls, line: str) -> "Hypergraph":
        import networkx as nx

        return cls.from_networkx(nx.from_graph6_bytes(line.strip().encode()))

    # ----------------------------------------------------------- basic data
    @property
    def n_vertices(self) -> int:
        return self.M.shape[0]

    @property
    def n_edges(self) -> int:
        return self.M.shape[1]

    @property
    def incidence_matrix(self) -> np.ndarray:
        """Vertex-edge incidence matrix ``M_ve``."""
        return self.M.copy()

    @property
    def edge_vertex_matrix(self) -> np.ndarray:
        """Dual (edge-vertex) incidence matrix ``M_ev = M_ve^T``."""
        return self.M.T.copy()

    def is_empty(self) -> bool:
        return not np.any(self.M)

    def boundary(self, e: int) -> dict:
        """Boundary multiset ``d(e) = sum_v I(v, e) v`` of edge ``e`` (index)."""
        col = self.M[:, e]
        return {int(i): int(col[i]) for i in np.nonzero(col)[0]}

    def coboundary(self, v: int) -> dict:
        """Coboundary multiset ``delta(v) = sum_e I(v, e) e`` of vertex ``v``."""
        row = self.M[v, :]
        return {int(j): int(row[j]) for j in np.nonzero(row)[0]}

    def vertex_degrees(self) -> np.ndarray:
        """``|v| = #{e : I(v, e) != 0}``."""
        return (self.M != 0).sum(axis=1)

    def vertex_contact_degrees(self) -> np.ndarray:
        """``|v|_c = sum_e I(v, e)``."""
        return self.M.sum(axis=1)

    def edge_degrees(self) -> np.ndarray:
        return (self.M != 0).sum(axis=0)

    def edge_contact_degrees(self) -> np.ndarray:
        return self.M.sum(axis=0)

    def is_graph(self) -> bool:
        """True if every edge has degree and degree of contact 2."""
        return bool(np.all(self.edge_degrees() == 2) and np.all(self.edge_contact_degrees() == 2))

    # ------------------------------------------------- derived structures
    def minimum_connectivity(self) -> "Hypergraph":
        """``m(G)``: same support, every non-null degree of contact set to 1."""
        return Hypergraph((self.M != 0).astype(np.int64), self.vertex_labels, self.edge_labels)

    def connectivity_graph(self):
        """``g(G)``: bipartite graph on ``V ∪ E`` with an edge ``(v, e)`` of
        weight ``I(v, e)`` whenever ``I(v, e) != 0`` (returns a networkx Graph)."""
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from((("v", i) for i in range(self.n_vertices)), bipartite=0)
        g.add_nodes_from((("e", j) for j in range(self.n_edges)), bipartite=1)
        for i, j in zip(*np.nonzero(self.M)):
            g.add_edge(("v", int(i)), ("e", int(j)), weight=int(self.M[i, j]))
        return g

    def euler_characteristic(self) -> int:
        """``chi(G) = |V| - |E|``."""
        return self.n_vertices - self.n_edges

    def connectivity_euler_characteristic(self) -> int:
        """``chi(g(G)) = |V| + |E| - #{(v, e) : I(v, e) != 0}``."""
        return self.n_vertices + self.n_edges - int(np.count_nonzero(self.M))

    def chain_complex(self, ring=Z) -> ChainComplex:
        """``C_*(G, F)``: the algebraic version of the 1D-cellulation of ``G``."""
        return ChainComplex(as_ring(ring).reduce(self.M), ring)

    # ----------------------------------------------------- maps & symmetry
    def permuted(self, vertex_perm: Sequence[int], edge_perm: Sequence[int]) -> "Hypergraph":
        """Isomorphic copy obtained by permuting rows and columns of ``M_ve``.

        ``vertex_perm[i]`` is the new position of vertex ``i`` (same for edges).
        """
        vp = np.asarray(vertex_perm)
        ep = np.asarray(edge_perm)
        M = np.zeros_like(self.M)
        M[np.ix_(vp, ep)] = self.M
        vl = [None] * self.n_vertices
        el = [None] * self.n_edges
        for i, k in enumerate(vp):
            vl[k] = self.vertex_labels[i]
        for j, k in enumerate(ep):
            el[k] = self.edge_labels[j]
        return Hypergraph(M, vl, el)

    def is_hypergraph_map(self, other: "Hypergraph", f0: Sequence[int], f1: Sequence[int]) -> bool:
        """Definition 2: ``0 <= I(v, e) <= I'(f0(v), f1(e))`` for all ``(v, e)``."""
        f0 = np.asarray(f0)
        f1 = np.asarray(f1)
        return bool(np.all(self.M <= other.M[np.ix_(f0, f1)]))

    def is_isomorphism(self, other: "Hypergraph", f0: Sequence[int], f1: Sequence[int]) -> bool:
        """Definition 3: bijective hypergraph map preserving all degrees of contact."""
        f0 = np.asarray(f0)
        f1 = np.asarray(f1)
        if self.M.shape != other.M.shape:
            return False
        if len(set(f0.tolist())) != self.n_vertices or len(set(f1.tolist())) != self.n_edges:
            return False
        return bool(np.array_equal(self.M, other.M[np.ix_(f0, f1)]))

    def __repr__(self) -> str:
        return f"Hypergraph(|V|={self.n_vertices}, |E|={self.n_edges})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Hypergraph) and np.array_equal(self.M, other.M)

    def __hash__(self) -> int:
        return hash((self.M.shape, self.M.tobytes()))
