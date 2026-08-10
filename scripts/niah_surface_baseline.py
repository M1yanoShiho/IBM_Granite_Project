"""Is the NIAH half's .9695 a semantic judgement, or string matching wearing one?

    export PYTHONPATH=src
    python scripts/niah_surface_baseline.py \
      --niah-manifest runs/niah-train-injected/manifest.json \
      --niah-provenance runs/niah-train-injected/provenance.jsonl \
      --niah-parents runs/niah-train-injected/source_parent.jsonl \
      --niah-dev-manifest runs/niah-injected/manifest.json \
      --sealed-dir runs/niah-sealed600 --niah-twin-label REFUTES

R013 seed 13 reaches macro-F1 .9695 on the 5284 NIAH rows. Those rows come in pairs: a needle
labelled SUPPORTS and its counterfactual twin labelled REFUTES, and the twin was made by
REPLACING ONE ENTITY in the needle. So the two premises are near-identical and the hypothesis is
the same for both, which means a model can separate them by asking whether the answer string in
the claim appears in the passage -- surface overlap, no judgement of contradiction at all.

Both stories predict .97 on this data. They predict very different things on a real retrieval
pool, where misleading evidence is not generally an entity-swapped copy of something else, and
where S6 measured an isolated-probe result failing to cascade by 38.3 points.

So this scores the surface story directly, with no model. If a trivial overlap rule reaches
about the same number, .9695 is not evidence of the discrimination the selector needs and
should not be reported as if it were. If it falls well short, the trained model is doing
something the surface cannot.

The baseline is deliberately STEELMANNED: its threshold is chosen after seeing the labels, by
sweeping for the best macro-F1. A baseline tuned with hindsight that still loses is a real
result; one crippled by a guessed threshold would prove nothing.

Rebuilt through `train_relations._load_niah_examples`, the same private path the training run
used, so the rows scored here are the rows that were trained on. Re-deriving them would let the
two drift apart silently, which is the class of defect this repo keeps finding.
"""

import argparse
import collections
import json
import re
import sys
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidence_rag.cli.train_relations import _load_niah_examples  # noqa: E402
from evidence_rag.relations.training import TrainingExample  # noqa: E402

# Only the words that carry no evidence either way. Kept short on purpose: a long list is a
# tuning surface, and the point is to give the baseline every reasonable advantage without
# quietly hand-building a feature.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were with"
    " what which who when where does did do".split()
)


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"\w+", text.casefold()) if word not in STOPWORDS}


def _overlap(example: TrainingExample) -> float:
    """Fraction of the claim's content words that appear in the passage."""

    claim = _tokens(example.hypothesis)
    if not claim:
        return 0.0
    return len(claim & _tokens(example.premise)) / len(claim)


def _macro_f1(pairs: Sequence[tuple[str, str]]) -> tuple[float, dict[str, float]]:
    labels = sorted({gold for gold, _ in pairs} | {pred for _, pred in pairs})
    per_label: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(per_label.values()) / len(per_label), per_label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in (
        "--niah-manifest",
        "--niah-provenance",
        "--niah-parents",
        "--niah-dev-manifest",
        "--sealed-dir",
    ):
        parser.add_argument(flag, required=True, type=Path)
    parser.add_argument("--niah-twin-label", required=True)
    parser.add_argument("--model-macro-f1", type=float, default=0.9695)
    parser.add_argument(
        "--oof",
        type=Path,
        help="oof_predictions.jsonl. Adds the ablation: how the model scores on the pairs where"
        " the surface cue is absent or inverted, which is the only reading that separates"
        " 'learned contradiction' from 'learned a finer surface'",
    )
    arguments = parser.parse_args(argv)

    niah, _sealed, _dev, _excluded = _load_niah_examples(
        Namespace(
            niah_manifest=arguments.niah_manifest,
            niah_provenance=arguments.niah_provenance,
            niah_parents=arguments.niah_parents,
            niah_dev_manifest=arguments.niah_dev_manifest,
            sealed_dir=arguments.sealed_dir,
            niah_twin_label=arguments.niah_twin_label,
        )
    )
    if not niah:
        raise SystemExit("no NIAH rows were built; check the six flags")

    scored = [(row.label.value, _overlap(row)) for row in niah]
    supports = sorted({gold for gold, _ in scored})
    if len(supports) != 2:
        raise SystemExit(f"expected two labels on the NIAH half, saw {supports}")
    high, low = "SUPPORTS", arguments.niah_twin_label

    best = (0.0, 0.0, {})
    # Sweep every distinct score as a cut point: the baseline gets the threshold it would have
    # picked knowing the answers.
    for threshold in sorted({score for _, score in scored}):
        pairs = [(gold, high if score >= threshold else low) for gold, score in scored]
        macro, per_label = _macro_f1(pairs)
        if macro > best[0]:
            best = (macro, threshold, per_label)

    macro, threshold, per_label = best
    majority, _ = _macro_f1([(gold, high) for gold, _ in scored])
    print(f"rows                 {len(niah)}")
    print(f"labels               {dict(_counts(scored))}")
    print(f"always-{high:<14} macro-F1 {majority:.4f}   (degenerate floor)")
    print(f"surface overlap      macro-F1 {macro:.4f}   at threshold {threshold:.4f}")
    for label in sorted(per_label):
        print(f"    {label:<10} F1 {per_label[label]:.4f}")
    print(f"trained model        macro-F1 {arguments.model_macro_f1:.4f}")
    gap = arguments.model_macro_f1 - macro
    print(f"\ngap (model - surface) {gap:+.4f}")
    print(
        "READ: a small gap means .9695 is reachable without judging contradiction at all, so it"
        " is not evidence of the discrimination the selector needs.\n"
        "      a large gap means the trained model is doing something the surface cannot."
    )
    if arguments.oof is not None:
        _ablate(niah, scored, arguments.oof, twin_label=arguments.niah_twin_label)
    return 0


def _ablate(
    niah: Sequence[TrainingExample],
    scored: Sequence[tuple[str, float]],
    oof_path: Path,
    *,
    twin_label: str,
) -> None:
    """Does the model still separate the pairs where the surface cue is absent or inverted?

    A gap over the surface baseline shows the model does MORE than overlap. It does not show the
    model is not ALSO leaning on overlap -- and the surface cue is strong in this data, so a
    model trained here will use it. On a real pool that cue mostly disappears, which is the shape
    S6 measured when an isolated-probe result failed to cascade.

    Pairs make the split threshold-free. Each synthetic family contributes exactly one needle and
    one twin, so the cue is INFORMATIVE for a pair when the needle's overlap beats the twin's, and
    ABSENT OR INVERTED when it does not. No band, no tuned cut-off, nothing to choose.
    """

    rows = [
        json.loads(line)
        for line in oof_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    niah_rows = [row for row in rows if row.get("source") == "niah"]

    # The chain is VitaminC then NIAH and the OOF is written in chain order, so the i-th NIAH row
    # in the file is niah[i]. That is an assumption about two files agreeing, so it is CHECKED:
    # a silent misalignment here would produce a perfectly plausible accuracy for the wrong rows.
    if len(niah_rows) != len(niah):
        raise SystemExit(f"OOF has {len(niah_rows)} NIAH rows, the rebuild has {len(niah)}")
    mismatched = sum(
        1 for row, example in zip(niah_rows, niah, strict=True) if row["gold"] != example.label.value
    )
    if mismatched:
        raise SystemExit(
            f"{mismatched} of {len(niah)} rows disagree on the gold label; the OOF and the"
            " rebuilt chain are not aligned and no number from them would mean anything"
        )

    # The family key is the group key shared by exactly two rows: one needle, one twin. Derived
    # rather than named, then verified -- if the derivation is wrong the pair check fails loudly.
    counts: collections.Counter[str] = collections.Counter(
        key for example in niah for key in example.group_keys
    )
    families: dict[str, list[int]] = collections.defaultdict(list)
    for index, example in enumerate(niah):
        pair_keys = [key for key in example.group_keys if counts[key] == 2]
        if len(pair_keys) != 1:
            raise SystemExit(
                f"row {index} has {len(pair_keys)} group keys occurring exactly twice;"
                " the needle/twin pairing cannot be derived"
            )
        families[pair_keys[0]].append(index)
    for key, members in families.items():
        labels = {niah[i].label.value for i in members}
        if len(members) != 2 or labels != {"SUPPORTS", twin_label}:
            raise SystemExit(f"family {key!r} is not one needle and one twin: {sorted(labels)}")

    informative: list[list[int]] = []
    uninformative: list[list[int]] = []
    for members in families.values():
        needle = next(i for i in members if niah[i].label.value == "SUPPORTS")
        twin = next(i for i in members if i != needle)
        (informative if scored[needle][1] > scored[twin][1] else uninformative).append(
            [needle, twin]
        )

    print("\n--- ablation: does the model survive without the surface cue? ---")
    print(f"pairs total            {len(families)}")
    for title, group in (("cue informative", informative), ("cue absent/inverted", uninformative)):
        if not group:
            print(f"{title:<22} 0 pairs")
            continue
        indices = [i for pair in group for i in pair]
        per_row = sum(
            1 for i in indices if niah_rows[i]["predicted"] == niah[i].label.value
        ) / len(indices)
        per_pair = sum(
            1
            for pair in group
            if all(niah_rows[i]["predicted"] == niah[i].label.value for i in pair)
        ) / len(group)
        share = len(group) / len(families)
        print(
            f"{title:<22} {len(group):>5} pairs ({share:6.1%})   "
            f"row acc {per_row:.4f}   both-correct {per_pair:.4f}"
        )
    print(
        "READ: if row accuracy holds up where the cue is absent or inverted, the model learned"
        " something the surface cannot give it.\n"
        "      if it collapses there, the gap above is a finer surface rule and will not survive"
        " a real pool.\n"
        "      check the pair count too: a verdict drawn from a handful of pairs is not one."
    )


def _counts(scored: Sequence[tuple[str, float]]) -> list[tuple[str, int]]:
    labels = sorted({gold for gold, _ in scored})
    return [(label, sum(1 for gold, _ in scored if gold == label)) for label in labels]


if __name__ == "__main__":
    sys.exit(main())
