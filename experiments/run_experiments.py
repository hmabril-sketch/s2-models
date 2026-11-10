#!/usr/bin/env python
"""Graph isomorphism experiments of Section 6.1 (Tables 2–5).

For every graph of every dataset we build the truncated boundary-scale
s²-model S²_k (k = 3 by default, coefficients in Z2) and compute, at each
scale s = 1..k, the metrics

    ω_β  Z2 Betti numbers of G(s)
    ω_d  sorted vertex degrees of G(s)
    ω_i  elementary divisors != 1 (over Z) of the incidence matrix of G(s)
    ω_ρ- SNF multiset (over Z) of rho-(2)∘...∘rho-(s+1) = (I, (MᵀM)^s)
    ω_ρ+ SNF multiset (over Z) of rho+(s)∘...∘rho+(1) = ((MMᵀ)^s, I)

and report the percentage of graphs of each dataset whose certificate is not
shared by any other graph of the same dataset.

Examples
--------
    python experiments/run_experiments.py                     # everything
    python experiments/run_experiments.py --family cubic      # Table 3 only
    python experiments/run_experiments.py --family sr --jobs 4

Features are cached in ``results/cache`` so tables can be regenerated
instantly (``--tables-only``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import networkx as nx  # noqa: E402

from s2models import BoundaryScaleModel, Hypergraph, unique_fraction  # noqa: E402
from s2models.datasets import load_family  # noqa: E402
from s2models.invariants import METRIC_SYMBOLS, METRICS, level_features  # noqa: E402

import paper_results  # noqa: E402

FAMILIES = ("cubic", "cayley", "sr")
FAMILY_TITLES = {"cubic": "Connected cubic graphs (Table 3)",
                 "sr": "Strongly regular graphs (Table 4)",
                 "cayley": "Minimal Cayley graphs (Table 5)"}


# ---------------------------------------------------------------------------
# feature computation
# ---------------------------------------------------------------------------
def _hashable(x):
    if isinstance(x, (list, tuple)):
        return tuple(_hashable(y) for y in x)
    if isinstance(x, dict):
        return {k: _hashable(v) for k, v in x.items()}
    return x


def graph_features(args) -> list:
    g6, levels, ring = args
    G = nx.from_graph6_bytes(g6.encode())
    model = BoundaryScaleModel(Hypergraph.from_networkx(G), ring=ring)
    return [level_features(model, s, METRICS) for s in range(1, levels + 1)]


def compute_dataset(name, graphs, levels, ring, cache_dir: Path, jobs: int, verbose=True):
    cache = cache_dir / ring / f"{name}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        if data["levels"] >= levels and data["n"] == len(graphs):
            return [[_hashable(f) for f in feats[:levels]] for feats in data["features"]]
    g6s = [nx.to_graph6_bytes(g, header=False).decode().strip() for g in graphs]
    t0 = time.time()
    work = [(g, levels, ring) for g in g6s]
    if jobs > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            feats = list(ex.map(graph_features, work, chunksize=1))
    else:
        feats = [graph_features(w) for w in work]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"dataset": name, "ring": ring, "levels": levels,
                                 "n": len(graphs), "graph6": g6s, "features": feats}))
    if verbose:
        print(f"  {name:13s} {len(graphs):4d} graphs  {time.time() - t0:7.1f}s", flush=True)
    return [[_hashable(f) for f in fs] for fs in feats]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def certificates(feats, metrics, k):
    return [tuple(tuple(fs[s][m] for m in metrics) for s in range(k)) for fs in feats]


def per_metric_row(feats, k):
    row = {"n": len(feats)}
    for m in METRICS:
        row[m] = unique_fraction(certificates(feats, (m,), k))
    row["all"] = unique_fraction(certificates(feats, METRICS, k))
    return row


def per_level_row(feats, k):
    return [unique_fraction(certificates(feats, METRICS, s)) for s in range(1, k + 1)]


def fmt(x):
    if x is None:
        return "–"
    return f"{x:.1f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def cell(ours, paper):
    if paper is None:
        return fmt(ours)
    same = abs(float(ours) - float(paper)) < 0.06
    return f"{fmt(ours)}" if same else f"**{fmt(ours)}** ({fmt(float(paper))})"


def write_tables(all_rows, all_levels, levels, ring, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [f"# Reproduced results (boundary-scale s²-model, ring {ring}, k = {levels})",
          "",
          "Percentages of graphs of each dataset that are uniquely distinguished.",
          "A value in **bold** differs from the paper; the paper's value is given in",
          "parentheses. Values equal to the paper (±0.05) are printed plainly.",
          ""]
    csv_metric = ["family,dataset,n," + ",".join(METRICS) + ",all"]
    csv_level = ["family,dataset,n," + ",".join(f"s{s}" for s in range(1, levels + 1))]
    cols = ("n",) + METRICS + ("all",)
    header = "| dataset | # | " + " | ".join(METRIC_SYMBOLS[m] for m in METRICS) + " | Disting. |"
    for fam, rows in all_rows.items():
        md += [f"## {FAMILY_TITLES[fam]}", "", header, "|" + "---|" * (len(cols) + 1)]
        for name, row in rows.items():
            paper = paper_results.PER_METRIC.get(name)
            cells = []
            for i, c in enumerate(cols):
                p = None if paper is None or levels != 3 else paper[i]
                cells.append(cell(row[c], p) if c != "n" else str(row[c]))
            md.append(f"| {name} | " + " | ".join(cells) + " |")
            csv_metric.append(f"{fam},{name}," + ",".join(f"{row[c]:.4g}" if c != "n" else str(row[c]) for c in cols))
        md.append("")
    md += ["## Distinguished graphs per truncation level (Table 2)", "",
           "| dataset | # | " + " | ".join(f"s = {s}" for s in range(1, levels + 1)) + " |",
           "|" + "---|" * (levels + 2)]
    for fam, rows in all_levels.items():
        for name, vals in rows.items():
            paper = paper_results.TABLE2.get(name)
            cells = [cell(v, paper[i] if paper and i < 3 else None) for i, v in enumerate(vals)]
            n = all_rows[fam][name]["n"]
            md.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
            csv_level.append(f"{fam},{name},{n}," + ",".join(f"{v:.4g}" for v in vals))
    md.append("")
    (out_dir / f"results_{ring}_k{levels}.md").write_text("\n".join(md))
    (out_dir / f"per_metric_{ring}_k{levels}.csv").write_text("\n".join(csv_metric) + "\n")
    (out_dir / f"per_level_{ring}_k{levels}.csv").write_text("\n".join(csv_level) + "\n")
    return out_dir / f"results_{ring}_k{levels}.md"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    ap.add_argument("--datasets", nargs="+", default=None, help="restrict to these dataset names")
    ap.add_argument("--levels", type=int, default=3, help="truncation level k (default 3)")
    ap.add_argument("--ring", default="Z2", help="coefficient ring of the s²-model (default Z2)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--cache-dir", default=str(ROOT / "results" / "cache"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--extra-cayley", action="store_true",
                    help="also run the 36-vertex Cayley groups (not reported in the paper)")
    ap.add_argument("--tables-only", action="store_true", help="only use cached features")
    args = ap.parse_args(argv)

    all_rows, all_levels = {}, {}
    for fam in args.family:
        print(f"[{fam}]", flush=True)
        sets = load_family(fam, args.data_dir, include_extra=args.extra_cayley) if fam == "cayley" \
            else load_family(fam, args.data_dir)
        all_rows[fam], all_levels[fam] = {}, {}
        for name, graphs in sets.items():
            if args.datasets and name not in args.datasets:
                continue
            if args.tables_only and not (Path(args.cache_dir) / args.ring / f"{name}.json").exists():
                continue
            feats = compute_dataset(name, graphs, args.levels, args.ring, Path(args.cache_dir), args.jobs)
            all_rows[fam][name] = per_metric_row(feats, args.levels)
            all_levels[fam][name] = per_level_row(feats, args.levels)
    path = write_tables(all_rows, all_levels, args.levels, args.ring, Path(args.out_dir))
    print(f"tables written to {path}")
    print(path.read_text())


if __name__ == "__main__":
    main()
