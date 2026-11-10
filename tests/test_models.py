import networkx as nx
import numpy as np
import pytest

from s2models import (
    BasisChangeModel,
    BoundaryScaleModel,
    ColorRefinementModel,
    Hypergraph,
    IteratedLaplacianModel,
    TrivialAdjacencyModel,
)

rng = np.random.default_rng(1)


def random_hypergraph(n=7, m=6, max_contact=3):
    M = rng.integers(0, max_contact + 1, size=(n, m))
    M[rng.random((n, m)) > 0.45] = 0
    return Hypergraph(M)


@pytest.mark.parametrize("ring", ["Z2", "Z3", "Z"])
@pytest.mark.parametrize("cls", [BoundaryScaleModel, ColorRefinementModel,
                                 TrivialAdjacencyModel, IteratedLaplacianModel])
def test_axioms_hold(cls, ring):
    for _ in range(3):
        G = random_hypergraph()
        model = cls(G, ring=ring)
        checks = model.verify(k=3, max_steps=3)
        assert all(checks.values()), checks


def test_basis_change_model():
    G = random_hypergraph(5, 4)
    # unimodular (over Z) upper triangular changes of basis
    P = np.triu(rng.integers(-2, 3, size=(5, 5)), 1) + np.eye(5, dtype=int)
    Q = np.triu(rng.integers(-2, 3, size=(4, 4)), 1) + np.eye(4, dtype=int)
    for ring in ("Z", "Z2"):
        model = BasisChangeModel(G, P, Q, ring=ring)
        assert all(model.verify(3).values())


def test_boundary_scale_matrix_form():
    G = random_hypergraph()
    M = G.M
    m = BoundaryScaleModel(G, ring="Z")
    for s in (1, 2, 3, 4):
        assert np.array_equal(m.boundary(s), M @ np.linalg.matrix_power(M.T @ M, s - 1))
    m2 = BoundaryScaleModel(G, ring="Z2")
    for s in (1, 2, 3):
        assert np.array_equal(m2.boundary(s), m.boundary(s) % 2)


def test_composites():
    G = Hypergraph.from_networkx(nx.petersen_graph())
    m = BoundaryScaleModel(G, ring="Z2")
    f = m.rho_plus_composite(3)
    assert np.array_equal(f.f0, m.vertex_laplacian_power(3))
    g = m.rho_minus_composite(3)
    assert np.array_equal(g.f1, m.edge_laplacian_power(3))
    assert f.is_chain_map() and g.is_chain_map()


def test_homology_functoriality():
    """[g∘f]_H = [g]_H [f]_H and [1]_H = 1 for the boundary-scale model."""
    for ring in ("Z2", "Z"):
        G = random_hypergraph(6, 5, 2)
        m = BoundaryScaleModel(G, ring=ring)
        H = m.homology()
        a0, a1 = H.rho_plus(1)
        b0, b1 = H.rho_plus(2)
        c0, c1 = (m.rho_plus(2) @ m.rho_plus(1)).homology_map()
        if ring == "Z2":
            assert np.array_equal((b0 @ a0) % 2, c0)
            assert np.array_equal((b1 @ a1) % 2, c1)
        else:
            assert np.array_equal(b1 @ a1, c1)  # H1 is free
        from s2models import identity_map

        i0, i1 = identity_map(m.level(2)).homology_map()
        assert np.array_equal(i0 % 2 if ring == "Z2" else i0, np.eye(i0.shape[0], dtype=int))
        assert np.array_equal(i1, np.eye(i1.shape[0], dtype=int))


def test_torsion_homology_over_z():
    # unsigned incidence of an odd cycle has H0 = Z/2 (torsion) and H1 = Z
    C5 = Hypergraph.from_networkx(nx.cycle_graph(5)).chain_complex("Z")
    assert C5.betti_numbers() == (0, 0)  # rank 5 over Q
    assert C5.torsion() == (2,)
    C4 = Hypergraph.from_networkx(nx.cycle_graph(4)).chain_complex("Z")
    assert C4.betti_numbers() == (1, 1)
    assert C4.torsion() == ()
