#!/usr/bin/env python
"""Walk through the running example of the paper (Figs. 1, 2, 3 and 5).

    python examples/paper_example.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from s2models import BoundaryScaleModel, Hypergraph  # noqa: E402


def show(title, A):
    print(f"\n{title}")
    print(np.array2string(np.asarray(A), separator=" "))


def main():
    # Fig. 1: relational description given by the boundary of each edge
    G = Hypergraph.from_edges(
        6, [{1: 1, 2: 3}, {}, {1: 3, 3: 1, 4: 1}, {3: 1}, {3: 2, 4: 1, 5: 1}], one_based=True)
    show("Vertex-edge incidence matrix M_ve(G) (Fig. 1)", G.M)
    print(f"\nchi(G) = {G.euler_characteristic()},  chi(m(G)) = "
          f"{G.minimum_connectivity().euler_characteristic()},  chi(g(G)) = "
          f"{G.connectivity_euler_characteristic()}   (Fig. 2)")

    show("Algebraic setting over Z2, d(1) = Z2[d] (Fig. 3, right)", G.chain_complex("Z2").boundary)

    model = BoundaryScaleModel(G, ring="Z2")
    show("rho+(2) = (Z2[dδ], 1)  –  f0 (Fig. 5)", model.rho_plus(2).f0)
    show("d(2) = Z2[d(δd)] (Fig. 5)", model.boundary(2))
    show("d(3) = Z2[d(δd)^2] (Fig. 5)", model.boundary(3))
    show("rho-(3) = (1, Z2[δd])  –  f1 (Fig. 5)", model.rho_minus(3).f1)

    print("\nAxioms of Definition 4 on S²_3:")
    for k, v in model.verify(3).items():
        print(f"  {k:40s} {v}")

    print("\nBetti numbers along the boundary-scale model (Section 4.2):")
    for s in (1, 2, 3):
        print(f"  beta(G({s})) = {model.level(s).betti_numbers()}")

    print()
    print(model.homology().summary(3))

    print("\nSame model over Z (torsion appears at higher scales):")
    print(BoundaryScaleModel(G, ring="Z").homology().summary(3))


if __name__ == "__main__":
    main()
