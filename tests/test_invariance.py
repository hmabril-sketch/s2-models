"""Isomorphism invariance (Theorem 3) and discrimination power."""

import itertools

import networkx as nx
import numpy as np
import pytest

from s2models import (
    BoundaryScaleModel,
    Hypergraph,
    cr_inter_degree_certificate,
    hybrid_certificate,
    isomorphism_test,
    model_features,
)

rng = np.random.default_rng(2)


def rook_4x4():
    return nx.convert_node_labels_to_integers(
        nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4)))


def shrikhande():
    g = nx.Graph()
    nodes = list(itertools.product(range(4), range(4)))
    gens = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]
    for a, b in nodes:
        for x, y in gens:
            g.add_edge((a, b), ((a + x) % 4, (b + y) % 4))
    return nx.convert_node_labels_to_integers(g)


def random_relabel(H: Hypergraph) -> Hypergraph:
    return H.permuted(rng.permutation(H.n_vertices), rng.permutation(H.n_edges))


def test_isomorphism_definition():
    H = Hypergraph(rng.integers(0, 3, size=(5, 4)))
    vp, ep = rng.permutation(5), rng.permutation(4)
    K = H.permuted(vp, ep)
    assert H.is_isomorphism(K, vp, ep)
    assert H.is_hypergraph_map(K, vp, ep)


@pytest.mark.parametrize("trial", range(5))
def test_certificates_are_invariant(trial):
    M = rng.integers(0, 3, size=(8, 9))
    M[rng.random(M.shape) > 0.4] = 0
    H = Hypergraph(M)
    K = random_relabel(H)
    for ring in ("Z2", "Z"):
        assert model_features(BoundaryScaleModel(H, ring), 3) == \
            model_features(BoundaryScaleModel(K, ring), 3)
    assert cr_inter_degree_certificate(H) == cr_inter_degree_certificate(K)
    assert hybrid_certificate(H) == hybrid_certificate(K)
    hH = BoundaryScaleModel(H).homology()
    hK = BoundaryScaleModel(K).homology()
    assert hH.betti_sequence(3) == hK.betti_sequence(3)


def test_srg_relabel_not_distinguished():
    g = shrikhande()
    H = Hypergraph.from_networkx(g)
    assert not isomorphism_test(H, random_relabel(H)).distinguished


def test_cubic06_distinguished_at_level_1():
    # K_{3,3} vs the prism: the elementary divisors of the incidence matrices differ
    res = isomorphism_test(nx.complete_bipartite_graph(3, 3), nx.circular_ladder_graph(3))
    assert res.distinguished and res.level == 1 and res.metric == "i"


def test_shrikhande_vs_rook():
    """The two srg(16,6,2,2) are 3-WL equivalent but the s²-model separates them."""
    a, b = shrikhande(), rook_4x4()
    assert nx.is_strongly_regular(a) and nx.is_strongly_regular(b)
    assert not nx.is_isomorphic(a, b)
    res = isomorphism_test(a, b, k=3)
    assert res.distinguished
    # Table 4: omega_beta, omega_d and omega_rho+ do not separate them
    for metric in ("beta", "d", "rho+"):
        assert not isomorphism_test(a, b, k=3, metrics=(metric,)).distinguished
    for metric in ("i", "rho-"):
        assert isomorphism_test(a, b, k=3, metrics=(metric,)).distinguished
