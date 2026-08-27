"""Cross-backbone agreement over Gate 0B dumps (R012f).

WHY THIS EXISTS. R012/R012b/R012c report each arm's aggregate rates, and RADAR
(arXiv 2605.22041) reports that swapping the NLI backbone changes its end-to-end numbers only
slightly — from which one might conclude the relation model is not load-bearing. Aggregate rates
cannot settle that: two classifiers can post identical rates while agreeing on no pair at all.
The question needs a per-pair join, and the dumps already carry the key for it
(`premise_hash`, `hypothesis_hash`).

WHAT THE ANSWER IS USED FOR. If the arms NEST — one arm's SUPPORTS set contains the other's —
the backbone is a conservatism dial and RADAR's reading survives. If they CROSS, they are
genuinely different classifiers, and a union-of-SUPPORTS over models already run is the cheapest
conceivable route past Gate 0B: no training, no new checkpoint. The union's twin cost is
computed in the same pass, because a rule that fires SUPPORTS whenever EITHER arm does must fail
a twin whenever either arm fails it. That trade is the recall/twin exchange rate which
`related-work.md` §8 records as unmeasured in the published literature.

WHAT IT IS NOT. This is a diagnostic over dumps that already exist. It is not a new Gate 0B
reading, and a union arm is not a pre-registered arm — proposing one would require an amendment.
"""

from collections.abc import Mapping, Sequence
from typing import Any

SUPPORTS = "SUPPORTS"

# The probe's four kinds. `gold_supports_recall` is measured on the needle document paired with
# the gold answer; the twin metric is measured on the two crossed pairs, which are the ones a
# model must NOT call SUPPORTS. `cf_replacement` is the clean control (a counterfactual document
# does support its own replaced answer) and belongs to neither metric.
GOLD_KIND = "needle_gold"
TWIN_KINDS = ("cf_gold", "needle_replacement")

PairKey = tuple[str, str]


def _index(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[PairKey, Mapping[str, Any]]:
    indexed: dict[PairKey, Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["premise_hash"]), str(row["hypothesis_hash"]))
        if key in indexed:
            raise ValueError(
                f"arm {name!r} has a duplicate pair {key}: the join key is ambiguous, so an"
                " agreement rate computed over it would silently drop or double-count pairs."
                " Both produce a believable number."
            )
        indexed[key] = row
    return indexed


def _rate(rows: Sequence[Mapping[str, Any]], *, kind: str | tuple[str, ...], supports: bool) -> tuple[int, float]:
    kinds = (kind,) if isinstance(kind, str) else kind
    selected = [row for row in rows if row.get("kind") in kinds]
    if not selected:
        return 0, float("nan")
    hits = sum(1 for row in selected if (row["predicted"] == SUPPORTS) is supports)
    return len(selected), hits / len(selected)


def compare_backbones(
    dumps: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Per-arm rates, per-pair agreement, and the union rule, over dumps of the same probe.

    `dumps` maps an arm name to that arm's dump rows (task tier only — external rows carry a
    null `kind` and are excluded by the kind filters).
    """
    indexed = {name: _index(name, rows) for name, rows in dumps.items()}

    arms: dict[str, Any] = {}
    for name, rows in dumps.items():
        n_gold, gold_recall = _rate(rows, kind=GOLD_KIND, supports=True)
        n_twin, twin_accuracy = _rate(rows, kind=TWIN_KINDS, supports=False)
        arms[name] = {
            "n_rows": len(rows),
            "n_gold": n_gold,
            "gold_supports_recall": gold_recall,
            "n_twin": n_twin,
            "twin_not_supported_accuracy": twin_accuracy,
        }

    pairwise: dict[tuple[str, str], Any] = {}
    union: dict[tuple[str, str], Any] = {}
    names = list(dumps)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a, b = indexed[left], indexed[right]
            common = sorted(set(a) & set(b))
            if not common:
                raise ValueError(
                    f"arms {left!r} and {right!r} have no pairs in common. They were not scored"
                    " on the same probe, so no agreement statistic over them means what it"
                    " appears to mean — most likely the probe was rebuilt between the two sweeps."
                )
            agree = sum(1 for key in common if a[key]["predicted"] == b[key]["predicted"])
            gold = [key for key in common if a[key].get("kind") == GOLD_KIND]
            only_a = sum(
                1
                for key in gold
                if a[key]["predicted"] == SUPPORTS and b[key]["predicted"] != SUPPORTS
            )
            only_b = sum(
                1
                for key in gold
                if b[key]["predicted"] == SUPPORTS and a[key]["predicted"] != SUPPORTS
            )
            pairwise[(left, right)] = {
                "n_common": len(common),
                "n_only_in_a": len(set(a) - set(b)),
                "n_only_in_b": len(set(b) - set(a)),
                "raw_agreement": agree / len(common),
                "cohen_kappa": _kappa([a[key]["predicted"] for key in common],
                                      [b[key]["predicted"] for key in common]),
                "gold_only_a": only_a,
                "gold_only_b": only_b,
                # Crossing means each arm recovers gold pairs the other misses, so the arms are
                # not one dial at two settings. This is the finding that makes a union worth
                # measuring, and the one that contradicts a "backbone choice barely matters"
                # reading.
                "crossing": only_a > 0 and only_b > 0,
            }
            twin = [key for key in common if a[key].get("kind") in TWIN_KINDS]
            union[(left, right)] = {
                "gold_supports_recall": (
                    sum(
                        1
                        for key in gold
                        if SUPPORTS in (a[key]["predicted"], b[key]["predicted"])
                    )
                    / len(gold)
                    if gold
                    else float("nan")
                ),
                "twin_not_supported_accuracy": (
                    sum(
                        1
                        for key in twin
                        if SUPPORTS not in (a[key]["predicted"], b[key]["predicted"])
                    )
                    / len(twin)
                    if twin
                    else float("nan")
                ),
                "rule": "SUPPORTS if either arm says SUPPORTS",
            }
    return {"arms": arms, "pairwise": pairwise, "union": union}


def _kappa(left: Sequence[str], right: Sequence[str]) -> float:
    """Cohen's kappa. Reported alongside raw agreement because raw agreement is inflated when
    one label dominates — and albert's UNKNOWN rate is .48, exactly that situation."""
    labels = sorted(set(left) | set(right))
    n = len(left)
    observed = sum(1 for a, b in zip(left, right, strict=True) if a == b) / n
    expected = sum(
        (left.count(label) / n) * (right.count(label) / n) for label in labels
    )
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)
