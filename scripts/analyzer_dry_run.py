"""Dry run of the production query analyzer over the calibration questions.

Required before the completeness harness is rebuilt. `RuleBasedQueryAnalyzer` is
purely rule-based -- regex tokenisation, a stopword list, a year pattern, and a
`METRICS` vocabulary that is enterprise-oriented (revenue, margin, clause,
termination, ...). ASQA is open-domain Wikipedia, and nobody has looked at what
the analyzer actually produces there.

Reports the checklist distribution, whether each checklist came from the METRICS
vocabulary or from the focus fallback, and verbatim samples -- enough to judge
whether the items are meaningful before any GPU time is spent on them.

No models, no gold fields, CPU only.

Usage:
  PYTHONPATH=src python scripts/analyzer_dry_run.py --limit 400 --seed 13
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import g3_baseline_comparison as g3  # noqa: E402, I001  identical case construction
from evidence_rag.contracts.models import Query  # noqa: E402
from evidence_rag.query_analysis import METRICS, RuleBasedQueryAnalyzer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--dump", type=Path, default=None)
    args = parser.parse_args()

    cases = g3.build_cases(g3.vt.ensure_asqa(), args.limit, random.Random(args.seed), args.top_k)
    print(f"[data] {len(cases)} cases (same construction/seed as the G3 run)\n", flush=True)

    analyzer = RuleBasedQueryAnalyzer()
    counts: Counter[int] = Counter()
    source: Counter[str] = Counter()
    constraint_counts: Counter[int] = Counter()
    rows = []
    for case in cases:
        checklist = analyzer.analyze(Query(query_id=case.query_id, text=case.question))
        n = len(checklist.required_facts)
        counts[n] += 1
        constraint_counts[len(checklist.constraints)] += 1
        # did the METRICS vocabulary fire, or did it fall back to the focus string?
        from_metrics = any(f.lower() in METRICS for f in checklist.required_facts)
        source["METRICS vocabulary" if from_metrics else "focus fallback"] += 1
        rows.append(
            {
                "query_id": case.query_id,
                "question": case.question,
                "focus": checklist.focus,
                "required_facts": list(checklist.required_facts),
                "constraints": list(checklist.constraints),
                "from_metrics": from_metrics,
            }
        )

    total = len(rows)
    mean = sum(k * v for k, v in counts.items()) / total
    print("== required_facts per query ==")
    print(f"mean: {mean:.2f}")
    for n in sorted(counts):
        print(f"  {n} fact(s): {counts[n]:4d} ({100 * counts[n] / total:5.1f}%)")
    print(f"empty checklists : {counts.get(0, 0)} ({100 * counts.get(0, 0) / total:.1f}%)")
    print(f"single-item      : {counts.get(1, 0)} ({100 * counts.get(1, 0) / total:.1f}%)")
    print()
    print("== where required_facts came from ==")
    for key, value in source.most_common():
        print(f"  {key}: {value} ({100 * value / total:.1f}%)")
    print()
    print("== constraints per query ==")
    print(f"mean: {sum(k * v for k, v in constraint_counts.items()) / total:.2f}")
    print()
    print(f"== verbatim samples (first {args.samples}) ==")
    for row in rows[: args.samples]:
        print(f"Q: {row['question']}")
        print(f"   required_facts: {row['required_facts']}")
        print(f"   constraints   : {row['constraints'][:6]}")
    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n[dump] {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
