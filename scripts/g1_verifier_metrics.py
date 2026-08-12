"""G1 verifier selection — recomputed from the raw scores, with explicit formulas.

The published triage tables report derived figures whose slice definitions were
not recorded in the artefact. This recomputes everything from
``results/verifier-triage/scores.jsonl`` with every definition stated, so each
number in the write-up carries its slice, its formula and its n.

**Whatever comes out is what is reported**, including where it does not match a
previously published figure.

    PYTHONPATH=src python scripts/g1_verifier_metrics.py \
        --scores local/results/g1/scores.jsonl \
        --out local/report-writing/g1-verifier-metrics.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

BACKENDS = ("true", "granite3b", "granite8b")
THRESHOLD = 0.50

# Slice composition, established by counting the raw file rather than assumed.
# Only `asqa` carries both polarities; every 2wiki-* cell is single-polarity and
# `counterfactual` is neutral-only.
NAMED_SLICES: dict[str, tuple[str, Callable[[str], bool]]] = {
    "asqa": (
        "ASQA claim/evidence pairs, both polarities",
        lambda c: c == "asqa",
    ),
    "2wiki-entailment": (
        "all 2WikiMultihop supported pairs (atomic + composed, plain + union)",
        lambda c: c.startswith("2wiki-") and c != "2wiki-neutral",
    ),
    "2wiki-neutral": (
        "2WikiMultihop hard negatives",
        lambda c: c == "2wiki-neutral",
    ),
    "counterfactual": (
        "entity-substituted pairs: evidence reads as supporting, entity swapped",
        lambda c: c == "counterfactual",
    ),
    "asqa+hard-negatives": (
        "ASQA both polarities, plus the 2wiki hard negatives",
        lambda c: c in ("asqa", "2wiki-neutral"),
    ),
    "all-cells": ("every row in the file", lambda c: True),
}


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fires(row: dict[str, Any], backend: str, threshold: float) -> bool:
    """The verifier's decision: p(entail) >= threshold."""
    return bool(row["scores"][backend]["p_entail"] >= threshold)


def metrics(rows: list[dict[str, Any]], backend: str, threshold: float) -> dict[str, Any]:
    pos = [r for r in rows if r["gold"] == "entailment"]
    neg = [r for r in rows if r["gold"] == "neutral"]
    tp = sum(1 for r in pos if fires(r, backend, threshold))
    fp = sum(1 for r in neg if fires(r, backend, threshold))
    return {
        "n_pos": len(pos),
        "n_neg": len(neg),
        "tp": tp,
        "fp": fp,
        # recall = TP / (gold-entailment rows)
        "recall": tp / len(pos) if pos else None,
        # false-positive rate = FP / (gold-neutral rows)
        "fp_rate": fp / len(neg) if neg else None,
        # rejection rate on a neutral-only slice = 1 - fp_rate
        "rejection": 1 - fp / len(neg) if neg else None,
        # precision = TP / (all rows the verifier fired on)
        "precision": tp / (tp + fp) if (tp + fp) else None,
    }


def fmt(value: Any) -> str:
    return "—" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    rows = load(args.scores)
    composition = Counter((r["cell"], r["gold"]) for r in rows)
    cells = sorted({c for c, _ in composition})

    doc: list[str] = [
        "# G1 verifier selection — recomputed with explicit definitions",
        "",
        f"Source: `{args.scores.as_posix()}`, **{len(rows)} rows**. "
        "Recomputed by `scripts/g1_verifier_metrics.py`; nothing is copied from a "
        "previously published table.",
        "",
        "## Definitions",
        "",
        f"A verifier **fires** on a pair when `p_entail >= {args.threshold:.2f}`. "
        "`p_entail` is the value stored per backend in the raw file.",
        "",
        "| quantity | formula | denominator |",
        "|---|---|---|",
        "| recall | TP / (gold-entailment rows) | rows whose gold label is `entailment` |",
        "| false-positive rate | FP / (gold-neutral rows) | rows whose gold label is `neutral` |",
        "| rejection rate | 1 − false-positive rate | neutral-only slices |",
        "| precision | TP / (TP + FP) | **rows the verifier fired on**, both polarities |",
        "",
        "Precision is only defined where a slice contains **both** polarities; on a "
        "single-polarity slice it is either 1.0 or 0.0 by construction and is shown "
        "as `—`.",
        "",
        "## Slice composition, counted from the file",
        "",
        "| cell | gold entailment | gold neutral | total |",
        "|---|---|---|---|",
    ]
    for cell in cells:
        pos, neg = composition[(cell, "entailment")], composition[(cell, "neutral")]
        doc.append(f"| `{cell}` | {pos} | {neg} | {pos + neg} |")
    total_pos = sum(v for (_, g), v in composition.items() if g == "entailment")
    total_neg = sum(v for (_, g), v in composition.items() if g == "neutral")
    doc += [
        f"| **total** | **{total_pos}** | **{total_neg}** | **{len(rows)}** |",
        "",
        "**Only `asqa` carries both polarities.** Every `2wiki-*` cell is "
        "entailment-only except `2wiki-neutral`, and `counterfactual` is "
        "neutral-only. That is why precision is reported on unions rather than "
        "per cell.",
        "",
    ]

    for slice_name, (description, predicate) in NAMED_SLICES.items():
        subset = [r for r in rows if predicate(r["cell"])]
        if not subset:
            continue
        cells_in = sorted({r["cell"] for r in subset})
        pos = sum(1 for r in subset if r["gold"] == "entailment")
        neg = len(subset) - pos
        doc += [
            f"## Slice `{slice_name}`",
            "",
            f"{description}.",
            "",
            f"- cells: {', '.join(f'`{c}`' for c in cells_in)}",
            f"- **n = {len(subset)}** ({pos} gold-entailment, {neg} gold-neutral)",
            f"- threshold: p_entail >= {args.threshold:.2f}",
            "",
            "| backend | recall | FP rate | rejection | precision | TP | FP |",
            "|---|---|---|---|---|---|---|",
        ]
        for backend in BACKENDS:
            m = metrics(subset, backend, args.threshold)
            precision = fmt(m["precision"]) if pos and neg else "—"
            recall = fmt(m["recall"]) if pos else "—"
            fp_rate = fmt(m["fp_rate"]) if neg else "—"
            rejection = fmt(m["rejection"]) if neg and not pos else "—"
            doc.append(
                f"| `{backend}` | {recall} | {fp_rate} | {rejection} | {precision} | "
                f"{m['tp']} | {m['fp']} |"
            )
        doc.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(doc) + "\n", encoding="utf-8")
    print(f"written: {args.out}")

    # console summary of the two headline figures
    asqa = [r for r in rows if r["cell"] == "asqa"]
    cf = [r for r in rows if r["cell"] == "counterfactual"]
    for backend in BACKENDS:
        a, c = metrics(asqa, backend, args.threshold), metrics(cf, backend, args.threshold)
        print(
            f"  {backend:10s} ASQA recall={fmt(a['recall'])} FP={fmt(a['fp_rate'])} "
            f"precision={fmt(a['precision'])} | counterfactual rejection={fmt(c['rejection'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
