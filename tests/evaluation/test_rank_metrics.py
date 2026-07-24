from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate
from evidence_rag.evaluation.models import GoldCase
from evidence_rag.evaluation.scoring import recall_at_k, reciprocal_rank
from evidence_rag.evaluation.stage_evaluators import (
    RETRIEVER_METRICS,
    evaluate_retriever_stage,
)


def candidate(evidence_id: str, doc: str, rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=doc,
        chunk_id=f"chunk-{evidence_id}",
        text=f"text {evidence_id}",
        source_uri=f"fixture://{doc}",
        retrieval_score=1.0 / rank,
        retrieval_rank=rank,
    )


def ranked(*docs: str) -> tuple[EvidenceCandidate, ...]:
    return tuple(candidate(f"ev-{i}", doc, i) for i, doc in enumerate(docs, start=1))


def test_reciprocal_rank_uses_first_relevant_rank() -> None:
    candidates = ranked("d1", "d2", "d3")
    assert reciprocal_rank(candidates, {"d1"}).value == 1.0
    assert reciprocal_rank(candidates, {"d3"}).value == 1.0 / 3.0
    assert reciprocal_rank(candidates, {"d2", "d3"}).value == 1.0 / 2.0


def test_reciprocal_rank_zero_when_no_relevant_retrieved() -> None:
    assert reciprocal_rank(ranked("d1", "d2"), {"other"}).value == 0.0


def test_reciprocal_rank_is_order_independent_of_input() -> None:
    shuffled = tuple(reversed(ranked("d1", "d2", "d3")))
    assert reciprocal_rank(shuffled, {"d3"}).value == 1.0 / 3.0


def test_reciprocal_rank_unscored_without_labels() -> None:
    assert reciprocal_rank(ranked("d1"), None).value is None
    assert reciprocal_rank(ranked("d1"), set()).value is None


def test_recall_at_k_restricts_to_top_k() -> None:
    candidates = ranked("d1", "d2", "d3", "d4")
    relevant = {"d1", "d4"}
    assert recall_at_k(candidates, relevant, 1).value == 0.5
    assert recall_at_k(candidates, relevant, 4).value == 1.0


def test_recall_at_k_is_monotonic_non_decreasing() -> None:
    candidates = ranked("d1", "d2", "d3")
    relevant = {"d1", "d2", "d3"}
    values = [recall_at_k(candidates, relevant, k).value for k in (1, 2, 3)]
    assert values == sorted(values)


def test_recall_at_k_unscored_without_labels() -> None:
    assert recall_at_k(ranked("d1"), None, 5).value is None


def test_retriever_stage_report_exposes_rank_aware_metrics() -> None:
    candidates = CandidateSet(query_id="q", candidates=ranked("d1", "d2", "d3"))
    gold = GoldCase(query_id="q", relevant_document_ids=("d2",))

    report = evaluate_retriever_stage(
        (candidates,), (gold,), dataset_signature="sig"
    )

    metrics = report.per_case[0].metrics
    assert set(metrics) == set(RETRIEVER_METRICS)
    assert metrics["retriever.core.document_mrr"].value == 1.0 / 2.0
    assert metrics["retriever.core.document_recall_at_5"].value == 1.0
