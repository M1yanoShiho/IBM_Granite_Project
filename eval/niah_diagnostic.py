"""Ranking-vs-recall decomposition for the NIAH diagnostic (finding 12 / Table 4a).

Pure arithmetic on two `run_niah` aggregate CSVs — the SAME task/retriever at k=10 and
k=100. The needle is retrieved into the top-100 pool `found@100` (= R@100) of the time,
but only `found@10` reach the top-10; the gap is retrieved-but-**buried** (a RANKING
failure), and `1 - found@100` is **never-retrieved** (a RECALL failure). "The bottleneck
is ranking, not recall" = buried > unreachable, quantified here. No GPU, no model — just
combines two numbers `run_niah` already wrote.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_needle_found(csv_path: Path, retriever: str) -> Tuple[int, float]:
    """Return ``(k, needle_found@k)`` for ``retriever`` from a run_niah aggregate CSV.

    ``k`` is parsed from the ``needle_found@<k>`` column name (it differs between the k=10
    and k=100 runs), so one reader handles both. Raises ``ValueError`` if the column or the
    retriever row is absent.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = next(
            (c for c in reader.fieldnames or [] if c.startswith("needle_found@")), None
        )
        if col is None:
            raise ValueError(f"no needle_found@k column in {csv_path}")
        k = int(col.split("@", 1)[1])
        for row in reader:
            if row["retriever"] == retriever:
                return k, float(row[col])
    raise ValueError(f"retriever {retriever!r} not found in {csv_path}")


def decompose(found_at_10: float, found_at_100: float) -> Dict[str, float]:
    """Split the miss into ranking vs recall failure.

    ``buried`` (ranking failure) = in the top-100 pool but ranked below 10
    (``found@100 - found@10``); ``unreachable`` (recall failure) = never in the pool
    (``1 - found@100``); ``ratio`` = buried / unreachable (``inf`` if no recall failure).
    """
    buried = found_at_100 - found_at_10
    unreachable = 1.0 - found_at_100
    ratio = buried / unreachable if unreachable else float("inf")
    return {
        "found@10": found_at_10,
        "found@100": found_at_100,
        "buried": buried,
        "unreachable": unreachable,
        "ratio": ratio,
    }


def _write_csv(row: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(
            {k: ("inf" if v == float("inf") else round(v, 4)) for k, v in row.items()}
        )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.niah_diagnostic",
        description="Ranking-vs-recall decomposition (Table 4a) from two run_niah aggregate "
        "CSVs (k=10 and k=100). Pure arithmetic, no GPU.",
    )
    p.add_argument("--k10-csv", type=Path, required=True, dest="k10_csv")
    p.add_argument("--k100-csv", type=Path, required=True, dest="k100_csv")
    p.add_argument("--retriever", default="granite_dense")
    p.add_argument("--out", type=Path, default=Path("results/niah_diag_decomposition.csv"))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    k10, f10 = read_needle_found(args.k10_csv, args.retriever)
    k100, f100 = read_needle_found(args.k100_csv, args.retriever)
    if (k10, k100) != (10, 100):
        raise SystemExit(f"expected a k=10 and a k=100 CSV; got k={k10} and k={k100}.")
    d = decompose(f10, f100)
    _write_csv(d, args.out)

    ratio = d["ratio"]
    ratio_str = "inf (no recall failure)" if ratio == float("inf") else f"{ratio:.2f}"
    print(f"retriever={args.retriever}")
    print(f"  needle-found@10  = {d['found@10']:.4f}")
    print(f"  needle-found@100 = {d['found@100']:.4f}   (R@100)")
    print(f"  buried 11-100  (RANKING failure) = {d['buried']:.4f}")
    print(f"  unreachable    (RECALL  failure) = {d['unreachable']:.4f}")
    print(f"  ranking / recall failure ratio   = {ratio_str}")
    print(f"-> {'ranking' if d['buried'] > d['unreachable'] else 'recall'} is the bottleneck")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
