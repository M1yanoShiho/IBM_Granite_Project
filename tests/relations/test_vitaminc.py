import pytest

from evidence_rag.relations.models import RelationLabel
from evidence_rag.relations.vitaminc import decontaminate, to_pairs


def _row(claim: str, evidence: str, label: str, page: str) -> dict[str, str]:
    return {"claim": claim, "evidence": evidence, "label": label, "page": page}


def test_to_pairs_maps_the_three_official_labels() -> None:
    pairs = to_pairs(
        [
            _row("c1", "e1", "SUPPORTS", "p1"),
            _row("c2", "e2", "REFUTES", "p1"),
            _row("c3", "e3", "NOT ENOUGH INFO", "p2"),
        ]
    )
    assert [pair.label for pair in pairs] == [
        RelationLabel.SUPPORTS,
        RelationLabel.REFUTES,
        RelationLabel.UNKNOWN,
    ]


def test_evidence_is_the_premise_and_claim_is_the_hypothesis() -> None:
    """Getting this backwards would invert every REFUTES edge while still looking plausible."""
    pair = to_pairs([_row("c1", "e1", "SUPPORTS", "p1")])[0]
    assert pair.premise == "e1"
    assert pair.hypothesis == "c1"
    assert pair.group == "p1"


def test_to_pairs_rejects_an_unknown_label_string() -> None:
    with pytest.raises(ValueError, match="unknown VitaminC label"):
        to_pairs([_row("c", "e", "MAYBE", "p")])


def test_to_pairs_rejects_a_row_missing_a_required_column() -> None:
    with pytest.raises(KeyError):
        to_pairs([{"claim": "c", "evidence": "e", "label": "SUPPORTS"}])


def test_decontaminate_removes_train_pages_that_appear_in_dev_or_test() -> None:
    result = decontaminate(
        train=to_pairs(
            [_row("c1", "e1", "SUPPORTS", "shared"), _row("c2", "e2", "SUPPORTS", "safe")]
        ),
        dev=to_pairs([_row("c3", "e3", "REFUTES", "shared")]),
        test=to_pairs([_row("c4", "e4", "REFUTES", "test-only")]),
    )
    assert [pair.group for pair in result.train] == ["safe"]
    assert result.removed_train_groups == ("shared",)


def test_decontaminate_removes_dev_pages_that_appear_in_test() -> None:
    result = decontaminate(
        train=to_pairs([_row("c1", "e1", "SUPPORTS", "safe")]),
        dev=to_pairs([_row("c2", "e2", "SUPPORTS", "leaky"), _row("c3", "e3", "SUPPORTS", "ok")]),
        test=to_pairs([_row("c4", "e4", "REFUTES", "leaky")]),
    )
    assert [pair.group for pair in result.dev] == ["ok"]
    assert result.removed_dev_groups == ("leaky",)


def test_decontaminate_never_moves_official_test() -> None:
    test = to_pairs([_row("c3", "e3", "REFUTES", "x"), _row("c4", "e4", "SUPPORTS", "y")])
    result = decontaminate(
        train=to_pairs([_row("c1", "e1", "SUPPORTS", "x")]),
        dev=to_pairs([_row("c2", "e2", "SUPPORTS", "x")]),
        test=test,
    )
    assert result.test == test


def test_a_page_in_all_three_splits_is_purged_from_train_and_dev() -> None:
    result = decontaminate(
        train=to_pairs([_row("c1", "e1", "SUPPORTS", "leaky")]),
        dev=to_pairs([_row("c2", "e2", "SUPPORTS", "leaky")]),
        test=to_pairs([_row("c3", "e3", "REFUTES", "leaky")]),
    )
    assert result.train == ()
    assert result.dev == ()
    assert result.removed_train_groups == ("leaky",)


def test_train_loses_a_page_shared_only_with_dev() -> None:
    """Distinguishes the dev axis from the test axis: this page never touches test, so only the
    train-vs-dev rule can remove it."""
    result = decontaminate(
        train=to_pairs([_row("c1", "e1", "SUPPORTS", "dev-only"),
                        _row("c2", "e2", "SUPPORTS", "safe")]),
        dev=to_pairs([_row("c3", "e3", "SUPPORTS", "dev-only")]),
        test=to_pairs([_row("c4", "e4", "REFUTES", "elsewhere")]),
    )
    assert [pair.group for pair in result.train] == ["safe"]
    assert [pair.group for pair in result.dev] == ["dev-only"]
    assert result.removed_train_groups == ("dev-only",)


def test_two_spellings_of_one_page_are_one_page() -> None:
    """The collision measured on bp1 2026-08-08, verbatim.

    VitaminC ships this article as `XXx-COLON- ...` in train and `XXX-COLON- ...` in dev. Raw
    string comparison sees two pages and removes neither; `page_key` — which every downstream
    leakage check uses — casefolds, so `assert_decontaminated` sees one and refuses the run.
    The exporter must partition the way the audit reads, or it cannot remove what the audit
    will find.
    """
    result = decontaminate(
        train=to_pairs([_row("c1", "e1", "SUPPORTS", "XXx-COLON- Return of Xander Cage")]),
        dev=to_pairs([_row("c2", "e2", "SUPPORTS", "XXX-COLON- Return of Xander Cage")]),
        test=(),
    )
    assert result.train == ()
    assert result.removed_train_groups == ("XXx-COLON- Return of Xander Cage",)


def test_spelling_variants_are_matched_against_test_too() -> None:
    """The axis with no downstream guard.

    `assert_decontaminated` only compares train against dev, so a spelling collision with
    official test would train on the evaluation surface and nothing would report it. Measured
    zero on the real export, which is a fact about this snapshot of VitaminC and not a property
    of the code — so it is asserted here instead.
    """
    result = decontaminate(
        train=to_pairs([_row("c1", "e1", "SUPPORTS", "Anna  Karenina")]),
        dev=to_pairs([_row("c2", "e2", "SUPPORTS", "anna karenina")]),
        test=to_pairs([_row("c3", "e3", "REFUTES", "Anna Karenina")]),
    )
    assert result.train == ()
    assert result.dev == ()
    assert result.removed_dev_groups == ("anna karenina",)
