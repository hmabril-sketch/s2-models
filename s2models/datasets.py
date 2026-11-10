"""Benchmark graph families used in Section 6.1 of the paper.

* connected cubic graphs on 6–14 vertices (``cubic06`` ... ``cubic14``);
* minimal Cayley graphs (House of Graphs), grouped by ``(|V|, |E|)`` as
  ``cayleyN-M``; groups with a single graph are dropped (nothing to
  distinguish) and, following the paper, the 36-vertex groups are reported
  separately;
* strongly regular graphs from B. McKay's collection (``srNKLM``:
  ``srg(N, K, L, M)``).

The graph6 files are the ones distributed with Ballester & Rieck,
"On the expressivity of persistent homology in graph learning" (LoG 2024),
https://github.com/aidos-lab/PH_expressivity, which contains exactly the same
datasets (and the same number of graphs per dataset) as Tables 3–5.
:func:`download_datasets` fetches them (pinned to a fixed commit and checked
against SHA-256 hashes) so that this repository does not redistribute them.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

import networkx as nx

__all__ = [
    "DATA_FILES",
    "PAPER_CAYLEY_GROUPS",
    "download_datasets",
    "default_data_dir",
    "load_family",
    "load_all",
    "read_graph6",
]

_COMMIT = "6a3058f652a84365e6469f86c8a7ce5373cac837"
_BASE = ("https://raw.githubusercontent.com/aidos-lab/PH_expressivity/"
         f"{_COMMIT}/data/simple_isomorphism_datasets")

# relative path -> sha256 of the file at the pinned commit
DATA_FILES = {
    "cubic/cub06.g6": "49a7391d96ad84fd13c8b5267ef23c215085ed317e6146e479df2bcdfd672949",
    "cubic/cub08.g6": "6946d13a0aec85386d47d087ad2a9f2561b05d8580f0ec937414f8d31359d34f",
    "cubic/cub10.g6": "0c3182bcbbfdc38fc7b84f3a1ac510834c3064524db00afeac212a3292f17c7b",
    "cubic/cub12.g6": "21bab16fbf7db826928315b9ca1d64684c0b1f7014b02fab0c6f2fb2f247104b",
    "cubic/cub14.g6": "6657f77868567a7e604a2325b5d748e5a04848b4ab75310ffa3ba695d26c66fe",
    "cay/minCayleyGraphs12Vertices.g6": "b7f4f3a1e1e77b15c417e406e0fe652c371320a92fa05bed43cee323d5fc7265",
    "cay/minCayleyGraphs16Vertices.g6": "70f47292e73e3a9f1d17f4554139ff763741116cd7c41f2e4e1d7971d59d9097",
    "cay/minCayleyGraphs20Vertices.g6": "16378bf41612ee7543bc60afeea3f3d76248d00de3fcfe2553c9d0245f954f22",
    "cay/minCayleyGraphs24Vertices.g6": "895ae84c58566985452baa2093cc32a7e0933c217aaa314d2086d4ba572a7d41",
    "cay/minCayleyGraphs32Vertices.g6": "8099dd6482491cf850a071574c5f528eb371006ca9b1aeab7c6452571aa86366",
    "cay/minCayleyGraphs36Vertices.g6": "a2518c779a44ec8314fe75a1a102cff574a14fcb64164396b31d7eb7f2136d67",
    "cay/minCayleyGraphs60Vertices.g6": "8d94b94a8832363462e4d7466a0ac16f941631604387ec0f2858a6d8fec5d903",
    "cay/minCayleyGraphs63Vertices.g6": "04fbd4e64c2cc6f5b063cddf6818ae89b986e1e9ae6c245bbe59d2cb19c8eca5",
    "SR/sr16622.g6": "2e258d3499c8e0f4bf1a9b98edb8bb5956bca794a0c07787d5227121044c50c1",
    "SR/sr251256.g6": "f55f86eb48cd61c3315c62341cc3a8c59289c58504edb3ab859d7ef2068b9aa4",
    "SR/sr261034.g6": "f0b4b4ad4507835703444c0f0990733e2f7cc31524cf91d04ba4a109f1fd52ac",
    "SR/sr281264.g6": "3859424c3e8facd4d3be701ff91fe38d5c43c7b7c1c0562a5d912b096a7ed2aa",
    "SR/sr291467.g6": "ea303aa60f98f01e6cc619884cd7d5716bfc00c38b7feaee080c7ff312781625",
    "SR/sr351899.g6": "a63871afc7e8a3a0d390d705c38a40ecc58e22563dcec8fafc9ea2879a361eb7",
    "SR/sr361446.g6": "f47de3ca58e807e9edbd13002ce3e71ed3ab813aa1517f8babde7dd0ab075854",
    "SR/sr401224.g6": "b899979459e6304bfff848d112d0ac8d738c837e5e21dc6e9d310a386aee8be6",
}

# Cayley groups that appear in Table 5 of the paper (in that order)
PAPER_CAYLEY_GROUPS = [
    "cayley12-18", "cayley12-24", "cayley16-24", "cayley16-32", "cayley20-30",
    "cayley20-40", "cayley24-36", "cayley24-48", "cayley24-60", "cayley24-72",
    "cayley32-48", "cayley32-64", "cayley32-80", "cayley32-96", "cayley60-90",
    "cayley60-120", "cayley60-150", "cayley60-180", "cayley63-126", "cayley63-189",
]

CUBIC = ["cubic06", "cubic08", "cubic10", "cubic12", "cubic14"]
STRONGLY_REGULAR = ["sr16622", "sr251256", "sr261034", "sr281264", "sr291467",
                    "sr351899", "sr361446", "sr401224"]


def default_data_dir() -> Path:
    env = os.environ.get("S2MODELS_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_datasets(data_dir: str | os.PathLike | None = None, force: bool = False,
                      verbose: bool = True) -> Path:
    """Download the graph6 files into ``data_dir`` (default: ``<repo>/data``)."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    for rel, digest in DATA_FILES.items():
        dest = data_dir / rel
        if dest.exists() and not force:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{_BASE}/{rel}"
        if verbose:
            print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as r:
            content = r.read()
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(content)
        if digest is not None and _sha256(tmp) != digest:
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch for {rel}")
        tmp.replace(dest)
    return data_dir


def read_graph6(path) -> list:
    """Read every graph of a graph6 file (one graph per line)."""
    graphs = []
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(b">>graph6<<"):
                line = line[len(b">>graph6<<"):]
            if not line:
                continue
            graphs.append(nx.from_graph6_bytes(line))
    return graphs


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` "
            "(or s2models.datasets.download_datasets()) first.")
    return path


def load_family(family: str, data_dir=None, include_extra: bool = False) -> dict:
    """Load a benchmark family as ``{dataset_name: [networkx graphs]}``.

    ``family`` is one of ``"cubic"``, ``"sr"`` (strongly regular) or
    ``"cayley"``.  For Cayley graphs, ``include_extra=True`` also returns the
    groups not reported in the paper (36 vertices).
    """
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    family = family.lower()
    if family == "cubic":
        return {name: read_graph6(_require(data_dir / "cubic" / f"cub{name[5:]}.g6")) for name in CUBIC}
    if family in ("sr", "strongly_regular"):
        return {name: read_graph6(_require(data_dir / "SR" / f"{name}.g6")) for name in STRONGLY_REGULAR}
    if family == "cayley":
        groups: dict = defaultdict(list)
        for n in (12, 16, 20, 24, 32, 36, 60, 63):
            for g in read_graph6(_require(data_dir / "cay" / f"minCayleyGraphs{n}Vertices.g6")):
                groups[f"cayley{g.number_of_nodes()}-{g.number_of_edges()}"].append(g)
        groups = {k: v for k, v in groups.items() if len(v) > 1}
        ordered = {k: groups[k] for k in PAPER_CAYLEY_GROUPS}
        if include_extra:
            extra = sorted((k for k in groups if k not in ordered),
                           key=lambda s: tuple(int(x) for x in s[6:].split("-")))
            ordered.update({k: groups[k] for k in extra})
        return ordered
    raise ValueError(f"unknown family {family!r}")


def load_all(data_dir=None) -> dict:
    return {f: load_family(f, data_dir) for f in ("cubic", "cayley", "sr")}
