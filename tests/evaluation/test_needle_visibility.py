from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.evaluation.needle_visibility import (
    ABSENT,
    TRUNCATED,
    VISIBLE,
    classify_needle,
    summarize_visibility,
)


def _cand(evidence_id: str, document_id: str, text: str, rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id,
        chunk_id=f"{document_id}::c0",
        text=text,
        source_uri=f"synthetic://{document_id}",
        retrieval_score=1.0 / rank,
        retrieval_rank=rank,
    )


def test_visible_when_alias_in_truncated_text():
    window = [_cand("e", "needle", "Kennedy is the answer " + "x " * 800, 1)]
    assert classify_needle(window, "needle", ["Kennedy"], passage_chars=600) == VISIBLE


def test_truncated_when_alias_only_after_cut():
    text = "x" * 700 + " Kennedy"
    window = [_cand("e", "needle", text, 1)]
    assert classify_needle(window, "needle", ["Kennedy"], passage_chars=600) == TRUNCATED


def test_absent_when_alias_not_in_chunk():
    window = [_cand("e", "needle", "no answer in this passage", 1)]
    assert classify_needle(window, "needle", ["Kennedy"], passage_chars=600) == ABSENT


def test_none_when_needle_not_in_window():
    window = [_cand("e", "other", "Kennedy", 1)]
    assert classify_needle(window, "needle", ["Kennedy"], passage_chars=600) is None


def test_visible_wins_across_multiple_needle_chunks():
    window = [
        _cand("e1", "needle", "x" * 700 + " Kennedy", 1),  # truncated on its own
        _cand("e2", "needle", "Kennedy up front", 2),  # visible
    ]
    assert classify_needle(window, "needle", ["Kennedy"], passage_chars=600) == VISIBLE


def test_summarize_rates():
    summary = summarize_visibility([VISIBLE, VISIBLE, TRUNCATED, ABSENT, None])
    assert summary.n_needle_in_window == 4
    assert summary.visible == 2
    assert summary.truncated == 1
    assert summary.absent == 1
    assert summary.visible_rate == 0.5
    assert summary.truncated_rate == 0.25
    assert summary.absent_rate == 0.25


def test_summarize_empty_is_none():
    summary = summarize_visibility([None, None])
    assert summary.n_needle_in_window == 0
    assert summary.visible_rate is None
