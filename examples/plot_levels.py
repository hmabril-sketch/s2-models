#!/usr/bin/env python
"""Connectivity representation of the boundary-scale levels (Table 1 of the paper).

Draws, for the two connected cubic graphs on 6 vertices (K_{3,3} and the
prism), the hypergraph components G(1), G(2), G(3) of their boundary-scale
s²-model over Z2: vertices are the nodes on the left, edges the nodes on the
right, and a segment joins v and e whenever the incidence is non-zero.

    python examples/plot_levels.py [--out results/figures/cubic06_levels.png]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from s2models import BoundaryScaleModel  # noqa: E402


def draw_level(ax, B, title=None):
    n, m = B.shape
    vy = np.linspace(1, 0, n)
    ey = np.linspace(1, 0, m)
    cmap = plt.get_cmap("tab20")
    for j in range(m):
        for i in np.nonzero(B[:, j])[0]:
            ax.plot([0, 1], [vy[i], ey[j]], color=cmap(j % 20), lw=1.2)
    ax.scatter(np.zeros(n), vy, s=22, color="#222", zorder=3)
    ax.scatter(np.ones(m), ey, s=22, color="#222", marker="s", zorder=3)
    for i in range(n):
        ax.text(-0.06, vy[i], f"v{i + 1}", ha="right", va="center", fontsize=7)
    for j in range(m):
        ax.text(1.06, ey[j], f"e{j + 1}", ha="left", va="center", fontsize=7)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.08, 1.08)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "cubic06_levels.png"))
    args = ap.parse_args()
    graphs = {"cubic06-0 (K3,3)": nx.complete_bipartite_graph(3, 3),
              "cubic06-1 (prism)": nx.circular_ladder_graph(3)}
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.5))
    for r, (name, g) in enumerate(graphs.items()):
        model = BoundaryScaleModel(g, ring="Z2")
        for s in (1, 2, 3):
            draw_level(axes[r, s - 1], model.boundary(s), f"s = {s}" if r == 0 else None)
        axes[r, 0].text(-0.45, 0.5, name, rotation=90, va="center", ha="center",
                        transform=axes[r, 0].transData, fontsize=10)
    fig.suptitle("Boundary-scale s²-model over Z2: hypergraph components G(1), G(2), G(3)", fontsize=11)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
