#!/usr/bin/env python
"""Download the benchmark graphs (graph6) used in the experiments.

Usage:  python scripts/download_data.py [--data-dir data] [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s2models.datasets import download_datasets, load_family  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None, help="destination (default: <repo>/data)")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()
    d = download_datasets(args.data_dir, force=args.force)
    print(f"data stored in {d}")
    for fam in ("cubic", "cayley", "sr"):
        sets = load_family(fam, d)
        print(f"  {fam:7s}: " + ", ".join(f"{k} ({len(v)})" for k, v in sets.items()))


if __name__ == "__main__":
    main()
