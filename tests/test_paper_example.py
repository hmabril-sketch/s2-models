"""The running example of the paper (Figs. 1, 2, 3 and 5)."""

import numpy as np
import pytest

from s2models import BoundaryScaleModel, Hypergraph


@pytest.fixture
def G():
    # Fig. 1: |V| = 6, |E| = 5 (e2 is an empty edge, v6 an isolated vertex)
    return Hypergraph.from_edges(
        6, [{1: 1, 2: 3}, {}, {1: 3, 3: 1, 4: 1}, {3: 1}, {3: 2, 4: 1, 5: 1}], one_based=True)


def test_incidence_matrix(G):
    expected = np.array([[1, 0, 3, 0, 0],
                         [3, 0, 0, 0, 0],
                         [0, 0, 1, 1, 2],
                         [0, 0, 1, 0, 1],
                         [0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0]])
    assert np.array_equal(G.M, expected)


def test_euler_characteristics(G):
    # Section 3.2: chi(G) = chi(m(G)) = 1 and chi(g(G)) = 11 - 9 = 2
    assert G.euler_characteristic() == 1
    assert G.minimum_connectivity().euler_characteristic() == 1
    g = G.connectivity_graph()
    assert g.number_of_nodes() == 11 and g.number_of_edges() == 9
    assert G.connectivity_euler_characteristic() == 2


def test_degrees(G):
    # identities (a) and (b) of Section 3.1
    assert G.vertex_degrees().sum() == G.edge_degrees().sum()
    assert G.vertex_contact_degrees().sum() == G.edge_contact_degrees().sum()
    assert G.boundary(4) == {2: 2, 3: 1, 4: 1}  # d(e5) = 2 v3 + v4 + v5 (0-based keys)


def test_algebraic_setting_z2(G):
    # Fig. 3 (right) and Fig. 5 (top): d(1) = Z2[d]
    d1 = np.array([[1, 0, 1, 0, 0],
                   [1, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0],
                   [0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0]])
    assert np.array_equal(G.chain_complex("Z2").boundary, d1)


def test_boundary_scale_levels_fig5(G):
    m = BoundaryScaleModel(G, ring="Z2")
    d2 = np.array([[1, 0, 0, 1, 1],
                   [0, 0, 1, 0, 0],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 1, 1],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0]])
    d3 = np.array([[0, 0, 1, 1, 0],
                   [1, 0, 1, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0],
                   [1, 0, 1, 1, 1],
                   [0, 0, 0, 0, 0]])
    rho_plus_2_f0 = np.array([[0, 1, 1, 1, 0, 0],
                              [1, 1, 0, 0, 0, 0],
                              [1, 0, 0, 1, 0, 0],
                              [1, 0, 1, 0, 1, 0],
                              [0, 0, 0, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0]])
    rho_minus_3_f1 = np.array([[0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [1, 0, 1, 1, 1],
                               [0, 0, 1, 1, 0],
                               [0, 0, 1, 0, 0]])
    assert np.array_equal(m.boundary(2), d2)
    assert np.array_equal(m.boundary(3), d3)
    assert np.array_equal(m.rho_plus(2).f0, rho_plus_2_f0)
    assert np.array_equal(m.rho_plus(2).f1, np.eye(5, dtype=int))
    assert np.array_equal(m.rho_minus(3).f0, np.eye(6, dtype=int))
    assert np.array_equal(m.rho_minus(3).f1, rho_minus_3_f1)


def test_betti_numbers_section_4_2(G):
    # "beta0(G(1)) = 2, beta1(G(1)) = 1 ... beta0(G(2)) = 3, beta1(G(2)) = 2,
    #  beta0(G(3)) = 4 and beta1(G(3)) = 3"
    m = BoundaryScaleModel(G, ring="Z2")
    assert [m.level(s).betti_numbers() for s in (1, 2, 3)] == [(2, 1), (3, 2), (4, 3)]
    H = m.homology()
    assert H.betti_sequence(3) == [(2, 1), (3, 2), (4, 3)]


def test_axioms(G):
    for ring in ("Z2", "Z"):
        assert all(BoundaryScaleModel(G, ring=ring).verify(3).values())
