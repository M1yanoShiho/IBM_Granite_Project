"""Tests for the cross-backbone agreement analysis (R012f).

R012/R012b/R012c reported each arm's AGGREGATE rates. Two classifiers can post the same rate
while agreeing on nothing, so the aggregate numbers cannot answer the question RADAR
(arXiv 2605.22041) raises: is the relation model a load-bearing component, or is backbone choice
a dial that the aggregation absorbs? That needs per-pair joins, which is what this module does.

The union rule is included because it is the cheapest possible escape from Gate 0B: if the arms
CROSS (each recovers gold pairs the other misses) rather than NEST, a union-of-SUPPORTS ensemble
of models we have already run could clear .85 with no training at all. It also measures the
recall/twin exchange rate, which §8 of related-work.md records as unmeasured in the literature.
"""

import pytest

from evidence_rag.relations.backbone_agreement import compare_backbones

_GOLD = "needle_gold"
_TWIN = "cf_gold"


def _row(pair: str, predicted: str, *, kind: str = _GOLD, gold: str = "SUPPORTS") -> dict:
    return {
        "premise_hash": f"p{pair}",
        "hypothesis_hash": f"h{pair}",
        "kind": kind,
        "gold": gold,
        "predicted": predicted,
    }


def test_identical_predictions_agree_perfectly() -> None:
    rows = [_row("1", "SUPPORTS"), _row("2", "UNKNOWN")]
    result = compare_backbones({"a": rows, "b": list(rows)})
    pair = result["pairwise"][("a", "b")]
    assert pair["n_common"] == 2
    assert pair["raw_agreement"] == 1.0


def test_pairs_are_joined_by_hash_not_by_position() -> None:
    """The two dumps need not list pairs in the same order, and a positional zip would silently
    compare unrelated pairs — producing a plausible agreement number from a meaningless join."""
    a = [_row("1", "SUPPORTS"), _row("2", "UNKNOWN")]
    b = [_row("2", "UNKNOWN"), _row("1", "SUPPORTS")]
    assert compare_backbones({"a": a, "b": b})["pairwise"][("a", "b")]["raw_agreement"] == 1.0


def test_disagreement_is_counted() -> None:
    a = [_row("1", "SUPPORTS"), _row("2", "SUPPORTS")]
    b = [_row("1", "SUPPORTS"), _row("2", "UNKNOWN")]
    assert compare_backbones({"a": a, "b": b})["pairwise"][("a", "b")]["raw_agreement"] == 0.5


def test_nested_arms_are_reported_as_nested() -> None:
    """b's SUPPORTS set contains a's. That is one conservatism dial, not two classifiers —
    and it is the RADAR-consistent outcome."""
    a = [_row("1", "SUPPORTS"), _row("2", "UNKNOWN"), _row("3", "UNKNOWN")]
    b = [_row("1", "SUPPORTS"), _row("2", "SUPPORTS"), _row("3", "UNKNOWN")]
    pair = compare_backbones({"a": a, "b": b})["pairwise"][("a", "b")]
    assert pair["gold_only_a"] == 0
    assert pair["gold_only_b"] == 1
    assert pair["crossing"] is False


def test_crossing_arms_are_reported_as_crossing() -> None:
    """Each arm recovers a gold pair the other misses. Then the arms are genuinely different
    classifiers, backbone choice is load-bearing, and a union is worth measuring."""
    a = [_row("1", "SUPPORTS"), _row("2", "UNKNOWN")]
    b = [_row("1", "UNKNOWN"), _row("2", "SUPPORTS")]
    pair = compare_backbones({"a": a, "b": b})["pairwise"][("a", "b")]
    assert pair["gold_only_a"] == 1
    assert pair["gold_only_b"] == 1
    assert pair["crossing"] is True


def test_union_recall_beats_both_arms_when_they_cross() -> None:
    a = [_row("1", "SUPPORTS"), _row("2", "UNKNOWN")]
    b = [_row("1", "UNKNOWN"), _row("2", "SUPPORTS")]
    union = compare_backbones({"a": a, "b": b})["union"][("a", "b")]
    assert union["gold_supports_recall"] == 1.0


def test_union_pays_for_recall_on_the_twin_metric() -> None:
    """The exchange rate, which is the whole point of computing the union: a rule that says
    SUPPORTS whenever EITHER arm does necessarily fails a twin whenever either arm fails it."""
    a = [_row("t1", "UNKNOWN", kind=_TWIN, gold="REFUTES"),
         _row("t2", "SUPPORTS", kind=_TWIN, gold="REFUTES")]
    b = [_row("t1", "SUPPORTS", kind=_TWIN, gold="REFUTES"),
         _row("t2", "UNKNOWN", kind=_TWIN, gold="REFUTES")]
    union = compare_backbones({"a": a, "b": b})["union"][("a", "b")]
    assert union["twin_not_supported_accuracy"] == 0.0


def test_per_arm_rates_reproduce_the_published_metric_definitions() -> None:
    """Sanity anchor: the per-arm numbers this module computes must be the same quantity the
    sweep reported, or every comparison built on them is against a different metric."""
    rows = [_row("1", "SUPPORTS"), _row("2", "UNKNOWN"), _row("3", "SUPPORTS"), _row("4", "REFUTES")]
    arm = compare_backbones({"a": rows})["arms"]["a"]
    assert arm["gold_supports_recall"] == 0.5
    assert arm["n_gold"] == 4


def test_a_duplicate_pair_within_one_dump_is_refused() -> None:
    """A duplicated join key would make the join ambiguous and quietly drop or double-count
    pairs. Both outcomes produce a believable agreement number."""
    rows = [_row("1", "SUPPORTS"), _row("1", "UNKNOWN")]
    with pytest.raises(ValueError, match="duplicate"):
        compare_backbones({"a": rows})


def test_arms_scored_on_different_pair_sets_are_refused() -> None:
    """If two dumps do not cover the same pairs, an agreement rate computed on the intersection
    is not the quantity anyone thinks it is. This is the guard that would have caught a probe
    rebuild between two sweeps."""
    a = [_row("1", "SUPPORTS")]
    b = [_row("2", "SUPPORTS")]
    with pytest.raises(ValueError, match="no pairs in common"):
        compare_backbones({"a": a, "b": b})


def test_partial_overlap_is_reported_rather_than_silently_intersected() -> None:
    a = [_row("1", "SUPPORTS"), _row("2", "SUPPORTS")]
    b = [_row("1", "SUPPORTS")]
    pair = compare_backbones({"a": a, "b": b})["pairwise"][("a", "b")]
    assert pair["n_common"] == 1
    assert pair["n_only_in_a"] == 1
