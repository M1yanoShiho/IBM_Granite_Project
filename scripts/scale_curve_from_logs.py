#!/usr/bin/env python3
"""Extract the scale-degradation curve from run_niah_scale logs into one tidy CSV.

The scale jobs (one retriever each) can share an output tag and clobber the per-size
result CSVs, but each job's LOG keeps its own per-size numbers. This parses those logs
back into a single ``retriever,max_docs,n_docs,needle_found@10,mrr`` table — the
two-line figure data — sorted by retriever then corpus size.

    python scripts/scale_curve_from_logs.py logs/niah-scale-18019720.out \
        logs/niah-scale-18019721.out > results/niah_scale_curve.csv
"""
from __future__ import annotations

import re
import sys
from typing import Iterator, List, Tuple

# "=== max_docs=10000 (q2d_granite, hnsw) ===" -> the swept corpus size + retriever.
_HEADER = re.compile(r"max_docs=(\d+)\s*\((\w+),")
# "q2d_granite: needle_found@10=0.590 MRR=0.334 (n_docs=54935)"
_RESULT = re.compile(r"(\w+):\s*needle_found@\d+=([\d.]+)\s+MRR=([\d.]+)\s*\(n_docs=(\d+)\)")

Row = Tuple[str, str, str, str, str]  # retriever, max_docs, n_docs, needle_found, mrr


def parse_logs(paths: List[str]) -> Iterator[Row]:
    """Yield one row per (retriever, corpus size) block across the given log files."""
    for path in paths:
        max_docs = ""
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                header = _HEADER.search(line)
                if header:
                    max_docs = header.group(1)
                    continue
                result = _RESULT.search(line)
                if result:
                    retriever, needle_found, mrr, n_docs = result.groups()
                    yield retriever, max_docs, n_docs, needle_found, mrr


def main(argv: List[str]) -> int:
    if not argv:
        print("usage: scale_curve_from_logs.py <log> [<log> ...]", file=sys.stderr)
        return 1
    rows = sorted(parse_logs(argv), key=lambda r: (r[0], int(r[1]) if r[1] else 0))
    print("retriever,max_docs,n_docs,needle_found@10,mrr")
    for row in rows:
        print(",".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
