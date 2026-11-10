#!/usr/bin/env python
"""s²-model isomorphism tests on a few classical hard pairs.

    python examples/isomorphism_test.py
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from s2models import Hypergraph, isomorphism_test  # noqa: E402


def shrikhande():
    g = nx.Graph()
    for a, b in itertools.product(range(4), range(4)):
        for x, y in [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]:
            g.add_edge((a, b), ((a + x) % 4, (b + y) % 4))
    return nx.convert_node_labels_to_integers(g)


def rook_4x4():
    return nx.convert_node_labels_to_integers(
        nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4)))


def main():
    pairs = {
        "K_{3,3} vs prism (cubic06)": (nx.complete_bipartite_graph(3, 3), nx.circular_ladder_graph(3)),
        "Shrikhande vs 4x4 rook (sr16622, 3-WL equivalent)": (shrikhande(), rook_4x4()),
        "Dodecahedron vs Desargues (cubic, 20 vertices)": (nx.dodecahedral_graph(), nx.desargues_graph()),
    }
    for name, (a, b) in pairs.items():
        print(f"{name}:\n   {isomorphism_test(a, b, k=3)}")

    # a relabelled copy is never distinguished (Theorem 3)
    rng = np.random.default_rng(0)
    H = Hypergraph.from_networkx(shrikhande())
    K = H.permuted(rng.permutation(H.n_vertices), rng.permutation(H.n_edges))
    print(f"Shrikhande vs a random relabelling:\n   {isomorphism_test(H, K, k=3)}")

    # hypergraphs with multiple degrees of contact work the same way
    A = Hypergraph.from_edges(5, [{0: 2, 1: 1}, {1: 1, 2: 1, 3: 1}, {3: 3, 4: 1}, {0: 1, 4: 1}])
    B = Hypergraph.from_edges(5, [{0: 2, 1: 1}, {1: 1, 2: 1, 3: 1}, {3: 1, 4: 3}, {0: 1, 4: 1}])
    print(f"Two hypergraphs with different contact degrees:\n   {isomorphism_test(A, B, k=3, ring='Z')}")


if __name__ == "__main__":
    main()
