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


def test_train_is_purged_against_the_kept_dev_not_the_raw_dev() -> None:
    """A page dropped from dev for leaking into test must not be re-admitted into train."""
    result = decontaminate(
        train=to_pairs([_row("c1", "e1", "SUPPORTS", "leaky")]),
        dev=to_pairs([_row("c2", "e2", "SUPPORTS", "leaky")]),
        test=to_pairs([_row("c3", "e3", "REFUTES", "leaky")]),
    )
    assert result.train == ()
    assert result.dev == ()
    assert result.removed_train_groups == ("leaky",)
