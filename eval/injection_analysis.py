# eval/injection_analysis.py
"""V2 injection-removal + V3 natural-conflict-rate (validity experiments, zero GPU).

Both answer "is the counterfactual-needle pathology our own construction?", from
dumped artifacts only:

- **V2 (removal):** delete the INJECTED distractors (Source A counterfactual +
  Source B generative; mined Source-C docs are natural corpus passages and stay)
  from each dumped ranking and re-place the needle. Removing docs cannot reorder
  the rest (per-doc scores are independent), so within the dumped depth this is the
  EXACT no-injection ranking. Each miss then splits into ``injection_buried``
  (needle enters top-k once injections are gone) vs ``natural_buried`` (natural
  docs alone keep it out) vs ``unreachable`` (outside the dumped depth; unknown).
- **V3 (conflict):** among the NATURAL docs of each query's extraction pool (the
  corroboration dump's answers, injected ids filtered out), how often do valid
  extracted answers disagree? Per-doc extraction is independent, so these answers
  are unaffected by the injections' presence. Caveat (report it): the natural pool
  here is the injected-corpus top-20 minus injections — a prefix of the true
  natural top-20, so the conflict rate is slightly conservative.

    python -m eval.injection_analysis --task results/niah_nq300_frozen.json \
        --runs results/niah_nq300v_runs_granite_dense.json \
               results/niah_nq300v_runs_q2d_granite.json \
        --tune-dump results/corroboration_runs_nq300v.json \
        --k 10 --out-prefix results/injection_nq300v
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

from eval.ir_metrics import Run
from eval.run_niah import load_niah_run
from eval.tune_corroboration import load_runs_with_answers, per_query_hits
from src.niah.types import SOURCE_COUNTERFACTUAL, SOURCE_GENERATIVE
from src.retrieval.corroboration import is_valid_answer, normalize_answer

_INJECTED_SOURCES = {SOURCE_COUNTERFACTUAL, SOURCE_GENERATIVE}


def injected_ids(payload: dict) -> Set[str]:
    """All doc_ids injected into the corpus (cf + generative), across ALL queries.

    Mined distractors are real corpus docs — recorded, never injected — so they are
    part of the natural haystack and are NOT removed.
    """
    out: Set[str] = set()
    for e in payload["examples"]:
        for d in e["distractors"]:
            if d["source"] in _INJECTED_SOURCES:
                out.add(d["doc_id"])
    return out


def filter_run(run: Run, injected: Set[str]) -> Run:
    """The run with every injected doc removed (scores untouched)."""
    return {qid: {d: s for d, s in scores.items() if d not in injected}
            for qid, scores in run.items()}


def _topk(scores: Dict[str, float], k: int) -> List[str]:
    # Same deterministic tie-break as per_query_hits: score desc, doc_id asc.
    return sorted(sorted(scores), key=lambda d: scores[d], reverse=True)[:k]


def classify_query(scores: Dict[str, float], needle: str, injected: Set[str], k: int) -> str:
    """found | injection_buried | natural_buried | unreachable for one query."""
    if needle not in scores:
        return "unreachable"
    if needle in _topk(scores, k):
        return "found"
    natural = {d: s for d, s in scores.items() if d not in injected}
    return "injection_buried" if needle in _topk(natural, k) else "natural_buried"


def natural_conflict(answers: Dict[str, str], injected: Set[str]) -> dict:
    """Do the NATURAL docs of one query's pool give conflicting valid answers?"""
    norms = [
        normalize_answer(a)
        for doc_id, a in answers.items()
        if doc_id not in injected and is_valid_answer(a)
    ]
    counts = Counter(norms)
    n = len(norms)
    return {
        "n_natural_valid": n,
        "n_distinct": len(counts),
        "conflict": len(counts) >= 2,
        "majority_share": (max(counts.values()) / n) if n else 0.0,
    }


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.injection_analysis",
        description="Injection-removal burial split (V2) + natural conflict rate (V3).",
    )
    p.add_argument("--task", type=Path, required=True, help="Task recipe JSON (distractor ids).")
    p.add_argument("--runs", type=Path, nargs="*", default=[], help="run_niah --dump-runs files.")
    p.add_argument("--tune-dump", type=Path, default=None, dest="tune_dump",
                   help="tune_corroboration dump WITH answers (V3 needs the answer strings).")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out-prefix", type=Path, default=Path("results/injection"), dest="out_prefix")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = json.loads(args.task.read_text(encoding="utf-8"))
    injected = injected_ids(payload)
    print(f"injected distractor docs (cf+gen, all queries): {len(injected)}")

    from eval.run_benchmark import write_per_query_csv

    # ---- V2: removal split per arm --------------------------------------- #
    removal_cols: Dict[str, Dict[str, float]] = {}
    for p in args.runs:
        dump = load_niah_run(p)
        name, run, needles = dump["retriever"], dump["run"], dump["needles"]
        counts = Counter(
            classify_query(run[qid], needle, injected, args.k)
            for qid, needle in needles.items() if qid in run
        )
        n = sum(counts.values())
        with_inj = per_query_hits(run, needles, args.k)
        without = per_query_hits(filter_run(run, injected), needles, args.k)
        removal_cols[f"{name}_with_injection"] = with_inj
        removal_cols[f"{name}_no_injection"] = without
        print(f"\n[V2] {name} (n={n}): found@{args.k} {sum(with_inj.values())/n:.4f} "
              f"-> no-injection {sum(without.values())/n:.4f}")
        for label in ("found", "injection_buried", "natural_buried", "unreachable"):
            c = counts.get(label, 0)
            print(f"    {label:<17} {c:>4}  ({c/n:.1%})")
    if removal_cols:
        path = Path(f"{args.out_prefix}_removal_per_query.csv")
        write_per_query_csv(removal_cols, path)
        print(f"wrote {path}")

    # ---- V3: natural conflict rate over the answers dump ------------------ #
    if args.tune_dump is not None:
        _, _, needles, answers_run, _ = load_runs_with_answers(args.tune_dump)
        if answers_run is None:
            raise SystemExit(
                f"{args.tune_dump} carries no answer strings (pre-WS-0 dump) — "
                "re-dump with the current eval.tune_corroboration --dump-runs first."
            )
        rows = []
        for qid, answers in answers_run.items():
            rec = natural_conflict(answers, injected)
            rec["query_id"] = qid
            rows.append(rec)
        n = len(rows)
        conflict_rate = sum(r["conflict"] for r in rows) / n if n else 0.0
        mean_distinct = sum(r["n_distinct"] for r in rows) / n if n else 0.0
        mean_valid = sum(r["n_natural_valid"] for r in rows) / n if n else 0.0
        print(f"\n[V3] natural pools (n={n}, injected filtered out): "
              f"conflict_rate={conflict_rate:.4f} mean_distinct_answers={mean_distinct:.2f} "
              f"mean_valid_natural_answers={mean_valid:.2f}")
        path = Path(f"{args.out_prefix}_conflict_per_query.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "query_id", "n_natural_valid", "n_distinct", "conflict", "majority_share"
            ])
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r[k] for k in writer.fieldnames})
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
