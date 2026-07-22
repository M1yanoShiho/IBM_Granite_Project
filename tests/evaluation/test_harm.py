from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, SelectedEvidenceSet
from evidence_rag.evaluation.harm import (
    HARM_METRIC,
    counterfactual_pool_hit_rate,
    evaluate_selector_harm,
    harmful_in_context,
    provenance_harm_map,
)
from evidence_rag.materializer.provenance import MutationRecord


def ev(evidence_id: str, document_id: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id,
        chunk_id=f"c-{evidence_id}",
        text=f"text {evidence_id}",
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=int(evidence_id[-1]) if evidence_id[-1].isdigit() else 1,
    )


def selected(query_id: str, *document_ids: str) -> SelectedEvidenceSet:
    return SelectedEvidenceSet(
        query_id=query_id,
        evidence=tuple(ev(f"e{i}", d) for i, d in enumerate(document_ids, start=1)),
    )


def record(query_id: str, cf_doc: str) -> MutationRecord:
    return MutationRecord(
        query_id=query_id,
        needle_document_id="needle",
        counterfactual_document_id=cf_doc,
        gold_value="18",
        gold_alias_used="18%",
        replacement_value="23",
        string_class="integer",
        seed=42,
        char_span=(0, 3),
        text_hash_before="a" * 64,
        text_hash_after="b" * 64,
        answer_bank_hash="c" * 64,
    )


def test_harmful_in_context_flags_poison_in_selection() -> None:
    assert harmful_in_context({"d1", "cf::q1::n"}, "cf::q1::n").value == 1.0
    assert harmful_in_context({"d1", "d2"}, "cf::q1::n").value == 0.0


def test_harmful_in_context_is_none_for_non_injected() -> None:
    assert harmful_in_context({"d1"}, None).value is None


def test_provenance_harm_map_builds_query_to_counterfactual() -> None:
    mapping = provenance_harm_map((record("q1", "cf::q1::n"), record("q2", "cf::q2::n")))
    assert mapping == {"q1": "cf::q1::n", "q2": "cf::q2::n"}


def test_evaluate_selector_harm_aggregates_over_injected_only() -> None:
    selected_sets = (
        selected("q1", "cf::q1::n", "d2"),
        selected("q2", "d3", "d4"),
        selected("q3", "d5"),
    )
    harm_map = {"q1": "cf::q1::n", "q2": "cf::q2::n"}
    report = evaluate_selector_harm(selected_sets, harm_map, dataset_signature="sig")
    aggregate = report.aggregate[HARM_METRIC]
    assert aggregate.mean == 0.5
    assert aggregate.n_scored == 2
    assert aggregate.n_total == 3


def test_pool_hit_rate_counts_injected_queries_only() -> None:
    candidate_sets = (
        CandidateSet(query_id="q1", candidates=(ev("e1", "cf::q1::n"), ev("e2", "d2"))),
        CandidateSet(query_id="q2", candidates=(ev("e3", "d3"),)),
    )
    harm_map = {"q1": "cf::q1::n", "q2": "cf::q2::n"}
    assert counterfactual_pool_hit_rate(candidate_sets, harm_map) == 0.5


def test_pool_hit_rate_is_none_without_injected_queries() -> None:
    candidate_sets = (CandidateSet(query_id="q1", candidates=(ev("e1", "d1"),)),)
    assert counterfactual_pool_hit_rate(candidate_sets, {}) is None
