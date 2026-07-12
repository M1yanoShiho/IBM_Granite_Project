# eval/anygold_rescore.py
"""V1 any-gold sensitivity re-score (validity experiment, zero GPU).

The certified NIAH metric tracks ONE designated needle per query (needle-found@k).
Critique: a query with several golds should count as a hit when ANY gold reaches
top-k. This tool re-scores dumped runs under that any-gold rule, next to the
designated rule, from the SAME rankings — pure arithmetic over
``run_niah --dump-runs`` files and/or a ``tune_corroboration --dump-runs`` file,
plus the task recipe's ``needle_ids`` (the full gold set per query).

Pre-registered readout (docs/hpc-run-log.md): absolute levels rise under any-gold;
the story survives if the q2d and corroboration deltas keep their sign/significance
(paired CSVs are written for ``eval.significance``). Either outcome is reported.

    python -m eval.anygold_rescore --task results/niah_nq300_frozen.json \
        --runs results/niah_nq300v_runs_granite_dense.json \
               results/niah_nq300v_runs_q2d_granite.json \
        --tune-dump results/corroboration_runs_nq300v.json --alpha 0.6 \
        --k 10 --out-prefix results/anygold_nq300v
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from eval.ir_metrics import Run
from eval.run_niah import load_niah_run
from eval.tune_corroboration import load_runs, per_query_hits
from src.retrieval.fusion import convex_fuse


def golds_from_recipe(payload: dict) -> Dict[str, Set[str]]:
    """``{query_id: set(needle_ids)}`` from a task recipe payload (full gold set)."""
    return {e["query_id"]: set(e["needle_ids"]) for e in payload["examples"]}


def any_gold_hits(run: Run, golds: Dict[str, Set[str]], k: int) -> Dict[str, float]:
    """Per-query 1.0/0.0 — is ANY of the query's golds in the top-``k``.

    Same deterministic tie-break as the designated metric (``per_query_hits``):
    score descending, then doc_id ascending — the two metrics differ ONLY in the
    target set, never in the ranking rule.
    """
    hits: Dict[str, float] = {}
    for qid, scores in run.items():
        ranked = sorted(sorted(scores), key=lambda d: scores[d], reverse=True)[:k]
        gold = golds.get(qid, set())
        hits[qid] = 1.0 if any(d in gold for d in ranked) else 0.0
    return hits


def _mean(hits: Dict[str, float]) -> float:
    return sum(hits.values()) / len(hits) if hits else 0.0


def collect_arms(
    run_paths: List[Path], tune_dump: Path | None, alpha: float
) -> List[Tuple[str, Run, Dict[str, str]]]:
    """``(name, run, designated_needles)`` per arm from the two dump kinds."""
    arms: List[Tuple[str, Run, Dict[str, str]]] = []
    for p in run_paths:
        payload = load_niah_run(p)
        arms.append((payload["retriever"], payload["run"], payload["needles"]))
    if tune_dump is not None:
        rel, corr, needles = load_runs(tune_dump)
        arms.append(("q2d_pool", convex_fuse(rel, corr, 1.0), needles))
        arms.append((f"q2d_corroborate_a{alpha}", convex_fuse(rel, corr, alpha), needles))
    return arms


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.anygold_rescore",
        description="Re-score dumped NIAH runs under the any-gold rule (V1).",
    )
    p.add_argument("--task", type=Path, required=True, help="Task recipe JSON (needle_ids source).")
    p.add_argument("--runs", type=Path, nargs="*", default=[], help="run_niah --dump-runs files.")
    p.add_argument("--tune-dump", type=Path, default=None, dest="tune_dump",
                   help="tune_corroboration --dump-runs file (adds q2d + blend arms).")
    p.add_argument("--alpha", type=float, default=0.6, help="Blend weight for the corroborate arm.")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out-prefix", type=Path, default=Path("results/anygold"), dest="out_prefix")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    if not args.runs and args.tune_dump is None:
        raise SystemExit("give --runs and/or --tune-dump (nothing to re-score).")
    golds = golds_from_recipe(json.loads(args.task.read_text(encoding="utf-8")))
    arms = collect_arms(args.runs, args.tune_dump, args.alpha)

    from eval.run_benchmark import write_per_query_csv

    designated_cols: Dict[str, Dict[str, float]] = {}
    anygold_cols: Dict[str, Dict[str, float]] = {}
    print(f"{'arm':<28} {'n':>4} {'designated@'+str(args.k):>14} {'any-gold@'+str(args.k):>12} {'gap':>7}")
    for name, run, needles in arms:
        designated = per_query_hits(run, needles, args.k)
        anygold = any_gold_hits(run, golds, args.k)
        designated_cols[name] = designated
        anygold_cols[name] = anygold
        d, a = _mean(designated), _mean(anygold)
        print(f"{name:<28} {len(anygold):>4} {d:>14.4f} {a:>12.4f} {a - d:>+7.4f}")

    des_path = Path(f"{args.out_prefix}_designated_per_query.csv")
    any_path = Path(f"{args.out_prefix}_anygold_per_query.csv")
    write_per_query_csv(designated_cols, des_path)
    write_per_query_csv(anygold_cols, any_path)
    print(f"wrote {des_path} and {any_path}")
    print("significance (any-gold deltas): "
          f"python -m eval.significance --per-query-csv {any_path} --reference <arm>")


if __name__ == "__main__":
    main()
