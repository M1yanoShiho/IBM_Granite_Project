from evidence_rag.relations.models import (
    BINARY_PREDICTED_LABELS,
    PREDICTED_LABELS,
    RelationLabel,
)


def test_a_three_class_head_emits_supports_refutes_or_unknown() -> None:
    """A2 (g2-proto-3, M0 §10.2) narrowed A1 back to the reading, so the OUTPUT space is three
    classes again and the collapse happens at each consumer.

    Pinned as a tuple rather than as set membership because the ORDER is the frozen §2.4
    tie-break: a tie resolves UNKNOWN -> REFUTES -> SUPPORTS and must never land on SUPPORTS,
    since a SUPPORTS edge is what makes a candidate droppable.
    """
    assert PREDICTED_LABELS == (
        RelationLabel.UNKNOWN,
        RelationLabel.REFUTES,
        RelationLabel.SUPPORTS,
    )


def test_a_natively_binary_head_emits_the_two_class_space_in_the_same_tie_break_order() -> None:
    """MiniCheck-FT5 has no third class to emit, so it speaks a second legitimate shape. The
    tie-break rule is identical and for the identical reason: never toward SUPPORTS.

    The two spaces overlap only on SUPPORTS, which is what lets `_output_space` tell them apart
    by exact key match instead of by guessing (M0 §10.6 cost 3).
    """
    assert BINARY_PREDICTED_LABELS == (RelationLabel.NOT_SUPPORTED, RelationLabel.SUPPORTS)
    assert PREDICTED_LABELS[-1] is BINARY_PREDICTED_LABELS[-1] is RelationLabel.SUPPORTS
    assert set(PREDICTED_LABELS) & set(BINARY_PREDICTED_LABELS) == {RelationLabel.SUPPORTS}


def test_not_supported_survives_as_a_derived_label_a_three_class_head_never_emits() -> None:
    """A2 demoted NOT_SUPPORTED from a predicted class to a derived one, but did not delete it.

    It is still what the 0B-2 metric and the graph read a non-SUPPORTS prediction AS; A1-era
    dumps, warm cache entries and probe files carry it and must keep parsing; and a
    natively-binary checkpoint emits it directly. Removing it would make all three unreadable.
    """
    assert RelationLabel.NOT_SUPPORTED not in PREDICTED_LABELS
    assert set(RelationLabel) == {
        RelationLabel.SUPPORTS,
        RelationLabel.NOT_SUPPORTED,
        RelationLabel.REFUTES,
        RelationLabel.UNKNOWN,
    }
