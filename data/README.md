# Benchmark graphs

The graph6 files are not stored in this repository. Download them with

```bash
python scripts/download_data.py
```

They are fetched from a pinned commit of
[aidos-lab/PH_expressivity](https://github.com/aidos-lab/PH_expressivity)
(Ballester & Rieck, *On the expressivity of persistent homology in graph
learning*, LoG 2024), which ships exactly the datasets used in the paper, and
are checked against the SHA-256 hashes listed in `s2models/datasets.py`.

| family | files | original source |
|---|---|---|
| connected cubic graphs | `cubic/cub06.g6` … `cub14.g6` | generated with nauty's `geng -c -d3 -D3 n` |
| minimal Cayley graphs | `cay/minCayleyGraphs{12,…,63}Vertices.g6` | [House of Graphs](https://houseofgraphs.org) |
| strongly regular graphs | `SR/sr*.g6` | [B. McKay's collection](https://users.cecs.anu.edu.au/~bdm/data/graphs.html) |

Cayley graphs are grouped by `(|V|, |E|)` into datasets `cayleyN-M`, as in
Table 5 of the paper.
