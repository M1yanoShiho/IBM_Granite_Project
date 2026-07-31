from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.evaluation.cluster_eval import (
    ClusterEvalCase,
    aggregate,
    contains_alias,
    evaluate_case,
    selection_bias,
    summarize,
    wilson_interval,
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


def test_contains_alias_word_boundary():
    assert contains_alias("born in Paris today", ["Paris"]) is True
    assert contains_alias("Parisian cafe", ["Paris"]) is False
    assert contains_alias("no match here", ["Paris", "Lyon"]) is False


def test_missed_conflict_true_when_needle_and_cf_share_cluster():
    window = [
        _cand("e_needle", "needle", "the needle passage", 1),
        _cand("e_cf", "cf::needle", "the counterfactual passage", 2),
    ]
    answers = ["Kennedy", "Kennedy"]
    case = evaluate_case(
        window,
        answers,
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.missed_conflict is True
    assert case.needle_gold_recovery is True


def test_missed_conflict_false_when_distinct_clusters():
    window = [
        _cand("e_needle", "needle", "the needle passage", 1),
        _cand("e_cf", "cf::needle", "the counterfactual passage", 2),
    ]
    answers = ["Kennedy", "Nixon"]
    case = evaluate_case(
        window,
        answers,
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.missed_conflict is False


def test_missed_conflict_unscored_when_needle_has_no_valid_answer():
    window = [
        _cand("e_needle", "needle", "the needle passage", 1),
        _cand("e_cf", "cf::needle", "the counterfactual passage", 2),
    ]
    answers = ["NONE", "Nixon"]  # needle not clustered
    case = evaluate_case(
        window,
        answers,
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.missed_conflict is None
    assert case.needle_gold_recovery is False  # in window, but did not recover gold


def test_false_conflict_true_when_second_gold_doc_splits():
    window = [
        _cand("e_needle", "needle", "answer is Kennedy", 1),
        _cand("e_other", "hay1", "also mentions JFK here", 2),
        _cand("e_cf", "cf::needle", "counterfactual", 3),
    ]
    answers = ["Kennedy", "JFK", "Nixon"]
    case = evaluate_case(
        window,
        answers,
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy", "JFK"),
    )
    assert case.false_conflict is True


def test_false_conflict_unscored_when_no_second_gold_doc():
    window = [
        _cand("e_needle", "needle", "answer is Kennedy", 1),
        _cand("e_cf", "cf::needle", "counterfactual", 2),
    ]
    answers = ["Kennedy", "Nixon"]
    case = evaluate_case(
        window,
        answers,
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy", "JFK"),
    )
    assert case.false_conflict is None


def test_needle_not_in_window_leaves_metrics_unscored():
    window = [_cand("e_cf", "cf::needle", "counterfactual", 1)]
    answers = ["Nixon"]
    case = evaluate_case(
        window,
        answers,
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.needle_in_window is False
    assert case.cf_in_window is True
    assert case.missed_conflict is None
    assert case.false_conflict is None
    assert case.needle_gold_recovery is None


def test_wilson_interval_bounds():
    low, high = wilson_interval(5, 10)
    assert 0.0 <= low < 0.5 < high <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    zlow, zhigh = wilson_interval(0, 20)
    assert zlow == 0.0
    assert 0.0 < zhigh < 0.3


def test_summarize_skips_none():
    summary = summarize([True, False, None, True])
    assert summary.n_scored == 3
    assert summary.successes == 2
    assert summary.rate == 2 / 3
    assert summary.ci_low <= summary.rate <= summary.ci_high


def test_summarize_all_none_is_unscored():
    summary = summarize([None, None])
    assert summary.n_scored == 0
    assert summary.rate is None


def test_aggregate_counts_window_presence():
    cases = [
        ClusterEvalCase("q1", True, True, True, None, True, True, False),
        ClusterEvalCase("q2", True, False, None, False, False, False, None),
    ]
    report = aggregate(cases)
    assert report.n_cases == 2
    assert report.needle_in_window == 2
    assert report.cf_in_window == 1
    assert report.missed_conflict.n_scored == 1
    assert report.needle_gold_recovery.n_scored == 2
    assert report.n_fixed_eligible == 1
    assert report.fixed_false_conflict.n_scored == 1


def test_selection_bias_counts_multi_canonical_key():
    reference_answers = [
        ("JFK", "John F. Kennedy"),
        ("1,200", "1200"),
        ("Paris",),
        None,
    ]
    bias = selection_bias(reference_answers, n_injected=2)
    assert bias.n_gold_cases == 4
    assert bias.n_answerable == 3
    assert bias.n_multi_key == 1
    assert bias.multi_key_rate == 1 / 3
    assert bias.skip_rate == (4 - 2) / 4


def test_fixed_eligible_needs_needle_and_another_gold_alias_passage() -> None:
    window = (
        _cand("e_needle", "needle", "Kennedy won the race", 1),
        _cand("e_other", "other", "Kennedy also appears here", 2),
    )
    case = evaluate_case(
        window,
        ("Kennedy", "Kennedy"),
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.fixed_eligible is True
    assert case.fixed_false_conflict is False


def test_fixed_eligible_is_false_without_a_second_gold_alias_passage() -> None:
    window = (
        _cand("e_needle", "needle", "Kennedy won the race", 1),
        _cand("e_noise", "noise", "unrelated text", 2),
    )
    case = evaluate_case(
        window,
        ("Kennedy", "NONE"),
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.fixed_eligible is False
    assert case.fixed_false_conflict is None


def test_fixed_false_conflict_true_when_gold_alias_passage_lands_elsewhere() -> None:
    window = (
        _cand("e_needle", "needle", "Kennedy won the race", 1),
        _cand("e_other", "other", "Kennedy also appears here", 2),
    )
    case = evaluate_case(
        window,
        ("Kennedy", "Nixon"),
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.fixed_false_conflict is True


def test_abstention_never_creates_a_fixed_false_conflict() -> None:
    """UNKNOWN/NONE counts as 'did not create a conflict' — frozen in the M0 G-FC guardrail."""
    window = (
        _cand("e_needle", "needle", "Kennedy won the race", 1),
        _cand("e_other", "other", "Kennedy also appears here", 2),
    )
    case = evaluate_case(
        window,
        ("Kennedy", "NONE"),
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.fixed_eligible is True
    assert case.fixed_false_conflict is False


def test_fixed_false_conflict_false_when_needle_itself_abstains() -> None:
    """Eligibility is system-independent, so an abstaining needle scores False, not None."""
    window = (
        _cand("e_needle", "needle", "Kennedy won the race", 1),
        _cand("e_other", "other", "Kennedy also appears here", 2),
    )
    case = evaluate_case(
        window,
        ("NONE", "Kennedy"),
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.fixed_eligible is True
    assert case.fixed_false_conflict is False
