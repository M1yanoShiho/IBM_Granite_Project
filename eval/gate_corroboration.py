"""Offline gated dynamic-alpha analysis for the corroboration reranker.

Everything here is pure arithmetic on a runs dump produced by
``eval.tune_corroboration --dump-runs`` (no LLM, no GPU, no index): dev/test split,
per-query gate signals, gated convex blend, dev-only selection, test-only
certification, and the descriptive flip table. Two deliverables: re-certify
finding 15 under an honest dev/test protocol, and test whether a per-query gate
(blend only where consensus exists / where the first stage is unsure) beats the
global blend. Design spec: docs/superpowers/specs/2026-07-06-gated-corroboration-design.md.

A structural fact this leans on: ``minmax_normalize`` maps an all-equal score set to
all-1.0, and a convex blend with a constant is order-preserving -- so zero-consensus
queries are untouched by the global blend already; the gate's room to act is the
weak-consensus region (votes gate tau=1 must therefore reproduce the global blend).
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from eval.ir_metrics import Run
from eval.tune_alpha import _grid
from eval.tune_corroboration import load_runs, per_query_hits
from src.retrieval.fusion import convex_fuse, fuse_one, minmax_normalize


def max_votes_signal(corroboration_run: Run) -> Dict[str, float]:
    """Per-query strongest consensus: max raw vote count (0.0 for an empty pool)."""
    return {
        qid: (max(scores.values()) if scores else 0.0)
        for qid, scores in corroboration_run.items()
    }


def margin_signal(relevance_run: Run) -> Dict[str, float]:
    """Per-query first-stage confidence: top1 - top2 of the min-max-normalised
    relevance scores (0.0 when fewer than 2 docs; all-equal scores normalise to
    all-1.0 so the margin is 0.0 and a margin gate fires -- harmless, the blend
    then decides)."""
    out: Dict[str, float] = {}
    for qid, scores in relevance_run.items():
        norm = sorted(minmax_normalize(scores).values(), reverse=True)
        out[qid] = norm[0] - norm[1] if len(norm) >= 2 else 0.0
    return out


def gate_mask(
    family: str,
    param: Optional[float],
    votes: Dict[str, float],
    margins: Dict[str, float],
) -> Dict[str, bool]:
    """Per-query blend/keep decision. ``global`` always blends; ``votes`` blends iff
    ``max_votes >= param``; ``margin`` blends iff ``margin < param`` (blend only where
    the first stage is unsure)."""
    if family == "global":
        return {qid: True for qid in votes}
    if family == "votes":
        return {qid: v >= param for qid, v in votes.items()}
    if family == "margin":
        return {qid: m < param for qid, m in margins.items()}
    raise ValueError(f"unknown gate family: {family!r}")


def gated_fuse(
    relevance_run: Run, corroboration_run: Run, alpha: float, gate: Dict[str, bool]
) -> Run:
    """Blend gated-on queries with ``fuse_one``; gated-off queries keep their raw
    first-stage scores (identical ranking to alpha=1 for that query)."""
    fused: Run = {}
    for qid, rel in relevance_run.items():
        if gate.get(qid, False):
            fused[qid] = fuse_one(rel, corroboration_run.get(qid, {}), alpha)
        else:
            fused[qid] = dict(rel)
    return fused


def per_query_reciprocal_rank(
    fused_run: Run, needles: Dict[str, str]
) -> Dict[str, float]:
    """Per-query ``1/rank`` of the designated needle in ``fused_run`` (0.0 if the
    needle is absent from the pool). Uses the SAME deterministic tie-break as
    ``per_query_hits`` (score descending, then doc_id ascending) so needle-found@k
    and MRR agree on the ranking of tied documents."""
    out: Dict[str, float] = {}
    for qid, needle in needles.items():
        scores = fused_run.get(qid, {})
        ranked = sorted(sorted(scores), key=lambda d: scores[d], reverse=True)
        out[qid] = 1.0 / (ranked.index(needle) + 1) if needle in ranked else 0.0
    return out


def split_queries(
    qids: List[str], seed: int = 0, dev_fraction: float = 0.5
) -> Tuple[List[str], List[str]]:
    """Deterministic dev/test split: sort, seeded shuffle, cut at ``dev_fraction``.

    Sorting first makes the split a function of (qid set, seed) alone -- input
    order (dict iteration, file order) cannot change who lands in test.
    """
    ordered = sorted(qids)
    random.Random(seed).shuffle(ordered)
    n_dev = round(len(ordered) * dev_fraction)
    return ordered[:n_dev], ordered[n_dev:]


def kfold_folds(qids: List[str], k: int, seed: int = 0) -> List[List[str]]:
    """Partition ``qids`` into ``k`` near-equal disjoint folds, deterministically.

    Sort first (so the partition depends only on the qid set + seed, not input
    order), seeded-shuffle, then cut into ``k`` contiguous chunks with round-based
    boundaries (sizes differ by at most 1). Union of the folds = all qids.
    """
    ordered = sorted(qids)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    return [ordered[round(f * n / k):round((f + 1) * n / k)] for f in range(k)]


def fuse_for_config(
    relevance_run: Run,
    corroboration_run: Run,
    qids: List[str],
    family: str,
    param: Optional[float],
    alpha: float,
) -> Run:
    """The gated fused run for ONE (family, param, alpha) config on a qid subset.

    Signals over the full runs -> gate mask -> restrict to ``qids`` -> ``gated_fuse``.
    Extracted so needle-found@k and MRR score the exact same fused ranking, and so
    the sweep/selection/nested-CV paths share one definition of "fuse a config".
    """
    votes = max_votes_signal(corroboration_run)
    margins = margin_signal(relevance_run)
    mask = gate_mask(family, param, votes, margins)
    rel = {q: relevance_run[q] for q in qids}
    cor = {q: corroboration_run.get(q, {}) for q in qids}
    return gated_fuse(rel, cor, alpha, mask)


def evaluate_config(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    qids: List[str],
    family: str,
    param: Optional[float],
    alpha: float,
    k: int,
) -> Dict[str, float]:
    """Per-query needle-found hits for ONE (family, param, alpha) config, restricted
    to ``qids`` (the dev or test half). The single definition of "score a config" --
    the sweep, the selection, and the test arms all go through here."""
    fused = fuse_for_config(relevance_run, corroboration_run, qids, family, param, alpha)
    nee = {q: needles[q] for q in qids}
    return per_query_hits(fused, nee, k)


def _mean(hits: Dict[str, float]) -> float:
    return sum(hits.values()) / len(hits) if hits else 0.0


# Gate grids (spec section 4). tau=1 reproduces the global blend (structural fact,
# asserted in tests); the real hypothesis space is tau >= 2.
VOTE_TAUS = [1.0, 2.0, 3.0, 4.0]
MARGIN_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


class CurveRow(NamedTuple):
    """One point of the dev sensitivity surface."""

    family: str
    param: Optional[float]
    alpha: float
    score: float
    n_gated: int


def sweep_gated(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    qids: List[str],
    alpha_grid: List[float],
    k: int,
) -> List[CurveRow]:
    """Score every (family, param, alpha) config on ``qids`` (the dev half).

    Pure arithmetic; publish the WHOLE surface (sensitivity artifact, no
    cherry-picking), selection happens separately in :func:`select_on_dev`.
    """
    votes = max_votes_signal(corroboration_run)
    margins = margin_signal(relevance_run)
    rows: List[CurveRow] = []
    for family, params in (
        ("global", [None]),
        ("votes", VOTE_TAUS),
        ("margin", MARGIN_THRESHOLDS),
    ):
        for param in params:
            mask = gate_mask(family, param, votes, margins)
            n_gated = sum(1 for q in qids if mask.get(q, False))
            for alpha in alpha_grid:
                hits = evaluate_config(
                    relevance_run, corroboration_run, needles, qids, family, param, alpha, k
                )
                rows.append(CurveRow(family, param, alpha, _mean(hits), n_gated))
    return rows


_FAMILY_ORDER = {"global": 0, "votes": 1, "margin": 2}  # ties -> simpler wins


def _strictness(row: CurveRow) -> float:
    """Larger = gate fires less often (spec tie-break: prefer the stricter gate)."""
    if row.family == "votes":
        return row.param
    if row.family == "margin":
        return -row.param
    return 0.0


def select_on_dev(
    rows: List[CurveRow],
) -> Tuple[Dict[str, CurveRow], CurveRow, CurveRow]:
    """``(best_per_family, overall_winner, best_gated)`` under the spec tie-breaks.

    Within a family: max score, ties to larger alpha (closer to pure relevance,
    matching ``tune_corroboration.best_alpha``), then to the stricter gate. Across
    families: max score, ties to the simpler family (global > votes > margin).
    ``best_gated`` is the winner among the votes/margin families only -- always
    certified on test so "gated vs global" is measured even when global wins dev.
    """
    best: Dict[str, CurveRow] = {}
    for family in _FAMILY_ORDER:
        candidates = [r for r in rows if r.family == family]
        best[family] = max(candidates, key=lambda r: (r.score, r.alpha, _strictness(r)))
    winner = max(
        best.values(), key=lambda r: (r.score, -_FAMILY_ORDER[r.family], r.alpha)
    )
    best_gated = max(
        (best["votes"], best["margin"]),
        key=lambda r: (r.score, -_FAMILY_ORDER[r.family], r.alpha),
    )
    return best, winner, best_gated


def nested_cv(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    k: int,
    seed: int,
    folds: int,
    alpha_grid: List[float],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[Dict[str, tuple]]]:
    """Outer K-fold, per-fold selection -> honest out-of-fold scores for the 3 arms.

    For each outer fold: select the arms' configs on the OTHER folds (train) via
    ``sweep_gated`` + ``select_on_dev`` (selection metric = needle-found@k), then apply
    each selected config to the held-out fold, recording needle-found@k AND MRR. Every
    query is thus scored by a config that never saw it. Returns
    ``(hits_by_arm, rr_by_arm, per_fold_configs)`` -- the two score maps span all qids;
    ``per_fold_configs[i]`` is fold i's ``{arm: (family, param, alpha)}`` (transparency:
    stable selection vs fold-to-fold drift). No inner CV: the config is fitting-free, so
    inner selection reduces to selecting on the training partition (see the design spec).
    """
    arms = ("q2d", "global_corroborate", "gated_corroborate")
    hits_by_arm: Dict[str, Dict[str, float]] = {a: {} for a in arms}
    rr_by_arm: Dict[str, Dict[str, float]] = {a: {} for a in arms}
    per_fold_configs: List[Dict[str, tuple]] = []
    fold_lists = kfold_folds(list(needles), folds, seed)
    for i, test_qids in enumerate(fold_lists):
        train_qids = [q for j, fold in enumerate(fold_lists) if j != i for q in fold]
        rows = sweep_gated(relevance_run, corroboration_run, needles, train_qids, alpha_grid, k)
        best, _winner, best_gated = select_on_dev(rows)
        configs = {
            "q2d": ("global", None, 1.0),
            "global_corroborate": ("global", None, best["global"].alpha),
            "gated_corroborate": (best_gated.family, best_gated.param, best_gated.alpha),
        }
        per_fold_configs.append(configs)
        nee_test = {q: needles[q] for q in test_qids}
        for name, (fam, par, al) in configs.items():
            fused = fuse_for_config(relevance_run, corroboration_run, test_qids, fam, par, al)
            hits_by_arm[name].update(per_query_hits(fused, nee_test, k))
            rr_by_arm[name].update(per_query_reciprocal_rank(fused, nee_test))
    return hits_by_arm, rr_by_arm, per_fold_configs


_VOTE_BUCKETS = ["0", "1", "2", "3", "4+"]
_MARGIN_BUCKETS = ["<0.05", "0.05-0.1", "0.1-0.2", ">=0.2"]
_STATUSES = ["fixed", "broken", "unchanged_hit", "unchanged_miss"]


def vote_bucket(votes: float) -> str:
    return "4+" if votes >= 4 else str(int(votes))


def margin_bucket(margin: float) -> str:
    if margin < 0.05:
        return "<0.05"
    if margin < 0.1:
        return "0.05-0.1"
    if margin < 0.2:
        return "0.1-0.2"
    return ">=0.2"


def flip_status(base_hit: float, blended_hit: float) -> str:
    """fixed (miss->hit) / broken (hit->miss) / unchanged_hit / unchanged_miss."""
    if blended_hit and not base_hit:
        return "fixed"
    if base_hit and not blended_hit:
        return "broken"
    return "unchanged_hit" if base_hit else "unchanged_miss"


def flip_table(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    alpha: float,
    k: int,
) -> List[Dict[str, object]]:
    """Descriptive flip analysis on the FULL query set at one alpha (no tuning):
    who does the global blend fix/break, bucketed by each gate signal. The
    evidence for/against gating's premise; goes in the report either way."""
    votes = max_votes_signal(corroboration_run)
    margins = margin_signal(relevance_run)
    base = per_query_hits(relevance_run, needles, k)
    blended = per_query_hits(convex_fuse(relevance_run, corroboration_run, alpha), needles, k)
    statuses = {qid: flip_status(base[qid], blended[qid]) for qid in needles}
    rows: List[Dict[str, object]] = []
    for signal, sig_map, buckets, bucket_fn in (
        ("max_votes", votes, _VOTE_BUCKETS, vote_bucket),
        ("margin", margins, _MARGIN_BUCKETS, margin_bucket),
    ):
        counts = {b: {s: 0 for s in _STATUSES} for b in buckets}
        for qid in needles:
            counts[bucket_fn(sig_map[qid])][statuses[qid]] += 1
        for b in buckets:
            rows.append({"signal": signal, "bucket": b, **counts[b]})
    return rows


def write_dev_curves(rows: List[CurveRow], path: Path, k: int) -> None:
    """The dev sensitivity surface CSV (spec 6): whole curves, no cherry-picking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "param", "alpha", f"needle_found@{k}", "n_gated"])
        for r in rows:
            param = "" if r.param is None else r.param
            writer.writerow([r.family, param, r.alpha, round(r.score, 4), r.n_gated])


def write_flip_table(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["signal", "bucket", *_STATUSES])
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.gate_corroboration",
        description="Offline gated dynamic-alpha analysis of the corroboration "
        "reranker from a runs dump: dev/test re-certification of the global blend "
        "+ gated-vs-global comparison + flip table. Pure arithmetic, no GPU.",
    )
    p.add_argument("--from-runs", type=Path, required=True, dest="from_runs",
                   help="Runs dump JSON from eval.tune_corroboration --dump-runs.")
    p.add_argument("--k", type=int, default=10, help="needle-found cut-off (default: %(default)s).")
    p.add_argument("--seed", type=int, default=0, help="dev/test split seed (default: %(default)s).")
    p.add_argument("--alpha-step", type=float, default=0.1, dest="alpha_step")
    p.add_argument("--flip-alpha", type=float, default=0.6, dest="flip_alpha",
                   help="Alpha for the descriptive flip table (default: the certified "
                        "alpha*=%(default)s).")
    p.add_argument("--out-dir", type=Path, default=Path("results"), dest="out_dir")
    p.add_argument("--nested-cv", action="store_true", dest="nested_cv",
                   help="Also run outer K-fold nested-CV (honest n=300 out-of-sample) "
                        "+ MRR, from the same dump.")
    p.add_argument("--folds", type=int, default=5,
                   help="Nested-CV outer folds (default: %(default)s).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    from eval.run_benchmark import write_per_query_csv  # lazy: run_benchmark is heavy

    args = _parse_args(argv)
    relevance_run, corroboration_run, needles = load_runs(args.from_runs)

    # 1. Descriptive flip table on the FULL set (no tuning -- spec section 4).
    flips = flip_table(relevance_run, corroboration_run, needles, args.flip_alpha, args.k)
    write_flip_table(flips, args.out_dir / "corroboration_flip_table.csv")

    # 2. Split, sweep on dev, select on dev.
    dev_qids, test_qids = split_queries(list(needles), args.seed)
    rows = sweep_gated(
        relevance_run, corroboration_run, needles, dev_qids, _grid(args.alpha_step), args.k
    )
    write_dev_curves(rows, args.out_dir / "corroboration_gate_dev_curves.csv", args.k)
    best, winner, best_gated = select_on_dev(rows)

    # 3. Certify exactly three arms on the held-out test half (params frozen).
    arms = {
        "q2d": ("global", None, 1.0),
        "global_corroborate": ("global", None, best["global"].alpha),
        "gated_corroborate": (best_gated.family, best_gated.param, best_gated.alpha),
    }
    per_query = {
        name: evaluate_config(
            relevance_run, corroboration_run, needles, test_qids, fam, par, al, args.k
        )
        for name, (fam, par, al) in arms.items()
    }
    test_csv = args.out_dir / "corroboration_gate_test_per_query.csv"
    write_per_query_csv(per_query, test_csv)

    print(f"seed={args.seed}  dev={len(dev_qids)}  test={len(test_qids)}  k={args.k}")
    for family, row in best.items():
        print(f"  dev best [{family}]: param={row.param} alpha={row.alpha} "
              f"needle_found@{args.k}={row.score:.4f} n_gated={row.n_gated}")
    print(f"dev winner: {winner.family} (param={winner.param}, alpha={winner.alpha})")
    for name, (fam, par, al) in arms.items():
        print(f"  test {name}: needle_found@{args.k}={_mean(per_query[name]):.4f} "
              f"(family={fam}, param={par}, alpha={al})")
    print(f"wrote {test_csv} -> significance:")
    print(f"  python -m eval.significance --per-query-csv {test_csv} --reference q2d")
    print(f"  python -m eval.significance --per-query-csv {test_csv} --reference global_corroborate")

    if args.nested_cv:
        hits_cv, rr_cv, fold_cfgs = nested_cv(
            relevance_run, corroboration_run, needles,
            args.k, args.seed, args.folds, _grid(args.alpha_step),
        )
        hits_csv = args.out_dir / "corroboration_nested_cv_per_query.csv"
        mrr_csv = args.out_dir / "corroboration_nested_cv_mrr.csv"
        write_per_query_csv(hits_cv, hits_csv)
        write_per_query_csv(rr_cv, mrr_csv)
        print(f"nested-CV: folds={args.folds} seed={args.seed} "
              f"(fitting-free selection -> outer CV only, no inner loop)")
        for i, cfg in enumerate(fold_cfgs):
            print(f"  fold {i}: gated={cfg['gated_corroborate']} "
                  f"global_alpha={cfg['global_corroborate'][2]}")
        for arm in ("q2d", "global_corroborate", "gated_corroborate"):
            print(f"  cv {arm}: needle_found@{args.k}={_mean(hits_cv[arm]):.4f} "
                  f"MRR={_mean(rr_cv[arm]):.4f}")
        print(f"wrote {hits_csv} and {mrr_csv} -> significance:")
        print(f"  python -m eval.significance --per-query-csv {hits_csv} --reference q2d")
        print(f"  python -m eval.significance --per-query-csv {mrr_csv} --reference q2d")


if __name__ == "__main__":
    main()
