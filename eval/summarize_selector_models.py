"""Summarize normalized LightGBM selector feature gain across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence


def normalize_importance(values: Sequence[float]) -> list[float]:
    total = float(sum(values))
    if total <= 0:
        return [0.0 for _ in values]
    return [float(value) / total for value in values]


def summarize_model_importance(
    *, model_dir: Path, training_result: Path, out: Path
) -> None:
    import lightgbm as lgb

    training = json.loads(training_result.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in training["protocol"]["seeds"]]
    summaries: dict[str, object] = {}
    for model_name in ("core", "full"):
        features = [str(value) for value in training["protocol"][f"{model_name}_features"]]
        per_seed: list[list[float]] = []
        for seed in seeds:
            booster = lgb.Booster(model_file=str(model_dir / f"{model_name}_seed{seed}.txt"))
            per_seed.append(
                normalize_importance(booster.feature_importance(importance_type="gain"))
            )
        summaries[model_name] = {
            feature: {
                "mean_normalized_gain": mean(values),
                "seed_sd": pstdev(values),
            }
            for feature, values in zip(features, zip(*per_seed))
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_model_importance(
        model_dir=args.model_dir, training_result=args.training_result, out=args.out
    )


if __name__ == "__main__":
    main()
