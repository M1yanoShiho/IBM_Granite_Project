"""Generate meeting-ready figures from frozen ML-selector result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


COLORS = {
    "fixed": "#0072B2",
    "ml": "#D55E00",
    "oracle": "#009E73",
    "neutral": "#6B7280",
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 220,
        }
    )


def _external_metrics(result: Mapping[str, object]) -> Mapping[str, object]:
    return result["metrics"]["adapted_top20_to_10"]


def plot_main_comparison(
    datasets: Mapping[str, Mapping[str, object]], *, out: Path
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metric_specs = (
        ("ndcg@10", "Ranking quality (NDCG@10)", True),
        ("required_evidence_recall@10", "Required evidence recall@10", True),
        ("harmful_rate@10", "Harmful evidence rate@10", False),
    )
    names = list(datasets)
    positions = np.arange(len(names))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.55))
    for axis, (metric, title, higher_is_better) in zip(axes, metric_specs):
        fixed = [float(datasets[name]["fixed_0.6"][metric]) for name in names]
        ml = [float(datasets[name]["ml_full"][metric]) for name in names]
        axis.bar(
            positions - width / 2,
            fixed,
            width,
            color=COLORS["fixed"],
            label="Fixed 0.6/0.4",
        )
        axis.bar(
            positions + width / 2,
            ml,
            width,
            color=COLORS["ml"],
            label="ML Full",
        )
        axis.set_title(title)
        axis.set_xticks(positions, names)
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", alpha=0.2)
        direction = "higher is better" if higher_is_better else "lower is better"
        axis.set_xlabel(direction)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_niah_tradeoff(metrics: Mapping[str, Mapping[str, float]], *, out: Path) -> None:
    import matplotlib.pyplot as plt

    display = {
        "q2d": "Q2D",
        "fixed_0.6": "Fixed",
        "ml_relevance_rank": "ML relevance",
        "ml_core": "ML Core",
        "ml_full": "ML Full",
        "ml_full_no_relevance": "ML no relevance",
        "oracle_at_20": "Oracle@20",
    }
    offsets = {
        "q2d": (-46, -18),
        "fixed_0.6": (6, 8),
        "ml_relevance_rank": (8, -18),
        "ml_core": (8, 2),
        "ml_full": (8, 14),
        "ml_full_no_relevance": (8, 6),
        "oracle_at_20": (6, 7),
    }
    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for method, label in display.items():
        row = metrics[method]
        color = (
            COLORS["ml"]
            if method.startswith("ml_")
            else COLORS["oracle"]
            if method == "oracle_at_20"
            else COLORS["fixed"]
        )
        axis.scatter(
            float(row["harmful_rate@10"]),
            float(row["ndcg@10"]),
            s=45 + 90 * float(row["required_evidence_recall@10"]),
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            label,
            (float(row["harmful_rate@10"]), float(row["ndcg@10"])),
            xytext=offsets[method],
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlabel("Harmful evidence rate@10 (lower is better)")
    axis.set_ylabel("NDCG@10 (higher is better)")
    axis.set_title("Controlled NIAH: quality-safety trade-off")
    axis.grid(alpha=0.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance: Mapping[str, object], *, out: Path) -> None:
    import matplotlib.pyplot as plt

    values = {
        feature: float(payload["mean_normalized_gain"])
        for feature, payload in importance["full"].items()
    }
    ordered = sorted(values, key=values.get)
    colors = [
        COLORS["fixed"]
        if feature in {"relevance_score", "relevance_normalized", "original_rank", "reciprocal_rank"}
        else COLORS["ml"]
        for feature in ordered
    ]
    fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    axis.barh(ordered, [values[name] for name in ordered], color=colors)
    axis.set_xlabel("Mean normalized LightGBM gain across 3 seeds")
    axis.set_title("What the Full selector learned")
    axis.grid(axis="x", alpha=0.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    pilot = _load(args.snapshot / "pilot/model_results/results.json")
    ramdocs = _load(args.snapshot / "ramdocs/niah_lodo/results.json")
    finance_path = args.snapshot / "financebench/full_niah_lodo/results.json"
    datasets: dict[str, Mapping[str, object]] = {
        "NIAH": pilot["metrics"]["test"],
        "RAMDocs": _external_metrics(ramdocs),
    }
    if finance_path.is_file():
        datasets["Finance"] = _external_metrics(_load(finance_path))
    plot_main_comparison(datasets, out=args.out_dir / "main_comparison.png")
    plot_niah_tradeoff(
        pilot["metrics"]["test"], out=args.out_dir / "niah_tradeoff.png"
    )
    plot_feature_importance(
        _load(args.snapshot / "pilot/model_results/feature_importance.json"),
        out=args.out_dir / "feature_importance.png",
    )


if __name__ == "__main__":
    main()
