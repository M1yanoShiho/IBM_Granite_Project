from evidence_rag.materializer.provenance import MutationRecord
from evidence_rag.relations.claims import HYPOTHESIS_TEMPLATE, build_hypothesis
from evidence_rag.relations.models import RelationLabel
from evidence_rag.relations.task_probe import (
    CF_GOLD,
    GOLD_SUPPORTS,
    NEEDLE_REPLACEMENT,
    TWIN_REFUTES,
    build_probe_pairs,
)


def _record(query_id: str = "q1") -> MutationRecord:
    return MutationRecord(
        query_id=query_id,
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_alias_used="Kennedy",
        replacement_value="Nixon",
        string_class="proper_name_1",
        seed=42,
        char_span=(0, 7),
        text_hash_before="a" * 8,
        text_hash_after="b" * 8,
        answer_bank_hash="c" * 8,
    )


_TEXTS = {"needle": "Kennedy won", "cf::needle": "Nixon won"}
_QUESTIONS = {"q1": "who won?"}


def test_hypothesis_uses_the_frozen_template() -> None:
    assert build_hypothesis("who won?", "Kennedy") == (
        'The answer to the question "who won?" is Kennedy.'
    )


def test_hypothesis_is_deterministic() -> None:
    assert build_hypothesis("q", "a") == build_hypothesis("q", "a")


def test_hypothesis_strips_surrounding_whitespace() -> None:
    assert build_hypothesis("  q  ", "  a  ") == 'The answer to the question "q" is a.'


def test_template_is_exported_for_the_protocol_record() -> None:
    assert "{question}" in HYPOTHESIS_TEMPLATE
    assert "{answer}" in HYPOTHESIS_TEMPLATE


def test_builds_the_four_deterministic_pair_types() -> None:
    pairs = build_probe_pairs(
        records=(_record(),), question_by_query=_QUESTIONS, text_by_document=_TEXTS
    )
    assert len(pairs) == 4
    by_key = {(pair.premise, pair.label) for pair in pairs}
    assert ("Kennedy won", RelationLabel.SUPPORTS) in by_key
    assert ("Kennedy won", RelationLabel.REFUTES) in by_key
    assert ("Nixon won", RelationLabel.SUPPORTS) in by_key
    assert ("Nixon won", RelationLabel.REFUTES) in by_key


def test_the_twin_row_pairs_the_counterfactual_against_the_gold_claim() -> None:
    """The cf_gold row is the whole point of 0B-2: the twin discrimination the extraction
    layer could not do."""
    pairs = build_probe_pairs(
        records=(_record(),), question_by_query=_QUESTIONS, text_by_document=_TEXTS
    )
    cf_gold = next(pair for pair in pairs if pair.kind == CF_GOLD)
    assert cf_gold.premise == "Nixon won"
    assert "Kennedy" in cf_gold.hypothesis
    assert cf_gold.label is RelationLabel.REFUTES


def test_pairs_carry_the_synthetic_family_as_the_leakage_group() -> None:
    pairs = build_probe_pairs(
        records=(_record(),), question_by_query=_QUESTIONS, text_by_document=_TEXTS
    )
    assert {pair.group for pair in pairs} == {"Kennedy|Nixon|proper_name_1"}


def test_kind_partitions_are_the_two_gate_metrics() -> None:
    pairs = build_probe_pairs(
        records=(_record(),), question_by_query=_QUESTIONS, text_by_document=_TEXTS
    )
    assert len([pair for pair in pairs if pair.kind in TWIN_REFUTES]) == 2
    assert len([pair for pair in pairs if pair.kind in GOLD_SUPPORTS]) == 1
    assert NEEDLE_REPLACEMENT in TWIN_REFUTES


def test_skips_a_record_whose_documents_are_missing() -> None:
    """A half-built probe would silently change the denominators Gate 0B is judged on."""
    pairs = build_probe_pairs(
        records=(_record(),),
        question_by_query=_QUESTIONS,
        text_by_document={"needle": "Kennedy won"},
    )
    assert pairs == ()


def test_skips_a_record_whose_query_is_missing() -> None:
    pairs = build_probe_pairs(
        records=(_record(),), question_by_query={}, text_by_document=_TEXTS
    )
    assert pairs == ()


def test_multiple_records_are_emitted_independently() -> None:
    pairs = build_probe_pairs(
        records=(_record("q1"), _record("q2")),
        question_by_query={"q1": "who won?", "q2": "who won?"},
        text_by_document=_TEXTS,
    )
    assert len(pairs) == 8
    assert {pair.query_id for pair in pairs} == {"q1", "q2"}
