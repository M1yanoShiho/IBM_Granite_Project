"""Apply frozen NIAH selector models to FinanceBench and RAMDocs without retraining."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from eval.niah_selector_pilot import (
    CORE_FEATURES,
    FULL_FEATURES,
    _attach_blend_scores,
    _paired_bootstrap,
    _query_metric_vector,
    attach_diagnostic_scores,
    holm_adjust,
    minmax,
    rank_candidates,
    selector_metrics,
)


def selection_diagnostics(
    groups: Mapping[str, Sequence[Mapping[str, object]]], *, score_key: str, k: int
) -> dict[str, float]:
    """Report query-level useful coverage, harmful exposure, and their conjunction."""

    covered = harmful = strict = 0
    for rows in groups.values():
        selected = rank_candidates(rows, score_key)[:k]
        has_required = any(int(row["utility_grade"]) >= 3 for row in selected)
        has_harmful = any(int(row["utility_grade"]) == 0 for row in selected)
        covered += has_required
        harmful += has_harmful
        strict += has_required and not has_harmful
    count = len(groups)
    return {
        f"required_query_coverage@{k}": covered / count if count else 0.0,
        f"harmful_query_exposure@{k}": harmful / count if count else 0.0,
        f"strict_selection_success@{k}": strict / count if count else 0.0,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _groups(rows: Sequence[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        if query_id in output:
            raise ValueError(f"duplicate query_id {query_id!r}")
        output[query_id] = [dict(candidate) for candidate in row["candidates"]]
    return output


def _predict_booster_ensemble(
    groups: Mapping[str, list[dict[str, object]]],
    *,
    model_paths: Sequence[Path],
    features: Sequence[str],
    score_key: str,
) -> None:
    import lightgbm as lgb
    import numpy as np

    ordered = [(query_id, groups[query_id]) for query_id in sorted(groups)]
    matrix = np.asarray(
        [[float(row[feature]) for feature in features] for _, rows in ordered for row in rows],
        dtype=np.float32,
    )
    predictions = []
    for path in model_paths:
        booster = lgb.Booster(model_file=str(path))
        predictions.append(booster.predict(matrix, num_iteration=booster.best_iteration or -1))
    mean_predictions = np.mean(predictions, axis=0)
    offset = 0
    for _, rows in ordered:
        for row, score in zip(rows, mean_predictions[offset : offset + len(rows)]):
            row[score_key] = float(score)
        offset += len(rows)


def evaluate_external(
    *,
    feature_cache: Path,
    model_dir: Path,
    training_result: Path,
    dataset: str,
    out_dir: Path,
) -> None:
    """Evaluate the NIAH-trained ensemble as a leave-one-domain-out selector."""

    rows = _read_jsonl(feature_cache)
    groups = _groups(rows)
    training = json.loads(training_result.read_text(encoding="utf-8"))
    alpha_star = float(training["protocol"]["alpha_star"])
    _attach_blend_scores(groups, alpha=0.6)
    attach_diagnostic_scores(groups)
    for candidates in groups.values():
        votes = minmax([float(row["exact_vote_count"]) for row in candidates])
        for index, row in enumerate(candidates):
            row["alpha_star_score"] = (
                alpha_star * float(row["relevance_normalized"])
                + (1.0 - alpha_star) * votes[index]
            )
    seeds = (13, 42, 73)
    _predict_booster_ensemble(
        groups,
        model_paths=[model_dir / f"core_seed{seed}.txt" for seed in seeds],
        features=CORE_FEATURES,
        score_key="ml_core_score",
    )
    _predict_booster_ensemble(
        groups,
        model_paths=[model_dir / f"full_seed{seed}.txt" for seed in seeds],
        features=FULL_FEATURES,
        score_key="ml_full_score",
    )
    methods = {
        "q2d": "q2d_score",
        "fixed_0.6": "fixed_score",
        "alpha_star": "alpha_star_score",
        "source_dedup_fixed": "source_dedup_fixed_score",
        "support_only": "support_only_score",
        "ml_core": "ml_core_score",
        "ml_full": "ml_full_score",
        "oracle_at_20": "oracle_score",
    }
    subsets: dict[str, tuple[dict[str, list[dict[str, object]]], int]] = {
        "adapted_top20_to_10": (groups, 10)
    }
    if dataset == "ramdocs":
        subsets["official_pool_to_3"] = (
            {
                query_id: [
                    row for row in candidates if row["source"] == "ramdocs_official"
                ]
                for query_id, candidates in groups.items()
            },
            3,
        )
    metrics: dict[str, object] = {}
    for subset_name, (subset_groups, k) in subsets.items():
        metrics[subset_name] = {
            method: {
                **selector_metrics(subset_groups, score_key=score_key, k=k),
                **selection_diagnostics(subset_groups, score_key=score_key, k=k),
            }
            for method, score_key in methods.items()
        }
    fixed_vector = _query_metric_vector(groups, score_key="fixed_score", k=10)
    full_vector = _query_metric_vector(groups, score_key="ml_full_score", k=10)
    statistics = {
        "ndcg_improvement": _paired_bootstrap(
            fixed_vector, full_vector, metric="ndcg", improvement_sign=1.0
        ),
        "harmful_rate_reduction": _paired_bootstrap(
            fixed_vector, full_vector, metric="harmful_rate", improvement_sign=-1.0
        ),
        "required_recall_change": _paired_bootstrap(
            fixed_vector, full_vector, metric="required_recall", improvement_sign=1.0
        ),
    }
    adjusted = holm_adjust(
        {name: values["p_two_sided"] for name, values in statistics.items()}
    )
    for name, value in adjusted.items():
        statistics[name]["holm_adjusted_p"] = value
    result = {
        "dataset": dataset,
        "training_domain": "controlled_niah_only",
        "protocol": {"candidate_pool": 20, "context_k": 10, "alpha_star": alpha_star},
        "metrics": metrics,
        "statistics_full_vs_fixed": statistics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["query_id", "candidate_id", "utility_grade", "source", *methods]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for query_id in sorted(groups):
            for row in groups[query_id]:
                writer.writerow(
                    {
                        "query_id": query_id,
                        "candidate_id": row["candidate_id"],
                        "utility_grade": row["utility_grade"],
                        "source": row["source"],
                        **{method: row[score_key] for method, score_key in methods.items()},
                    }
                )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=("ramdocs", "financebench", "contractnli"), required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    evaluate_external(
        feature_cache=args.feature_cache,
        model_dir=args.model_dir,
        training_result=args.training_result,
        dataset=args.dataset,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
