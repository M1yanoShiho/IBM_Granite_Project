from evidence_rag.relations.models import PREDICTED_LABELS, RelationLabel


def test_the_relation_model_may_only_emit_supports_or_not_supported() -> None:
    """A1 (g2-proto-2) §9.1: the relation model's output space is binary.

    Pinned as a tuple rather than a set membership check because the ORDER is the frozen §2.4
    tie-break — a tie must land on NOT_SUPPORTED, never on SUPPORTS.
    """
    assert PREDICTED_LABELS == (RelationLabel.NOT_SUPPORTED, RelationLabel.SUPPORTS)


def test_refutes_and_unknown_survive_as_schema_members_the_model_no_longer_emits() -> None:
    """A1 §9.1 retains CLAIM_REFUTES as an edge type; 0B-1's VitaminC gold still carries
    REFUTES and NEI (-> UNKNOWN), and historical dumps and cache entries are read back through
    this enum. Deleting either member would make those unreadable."""
    assert RelationLabel.REFUTES not in PREDICTED_LABELS
    assert RelationLabel.UNKNOWN not in PREDICTED_LABELS
    assert set(RelationLabel) == {
        RelationLabel.SUPPORTS,
        RelationLabel.NOT_SUPPORTED,
        RelationLabel.REFUTES,
        RelationLabel.UNKNOWN,
    }
