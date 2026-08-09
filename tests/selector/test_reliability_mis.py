from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.selector.reliability_mis import (
    LazyAnswerGenerator,
    MISSelectionEvent,
    ReliabilityMISSelector,
    SelectorBackendError,
    answer_statement,
    build_conflict_graph,
    build_pair_batch,
    clean_isolated_answer,
    is_unanswerable,
    select_capacity_limited_independent_set,
)
from evidence_rag.selector.top_k import TopKSelector


def candidate(
    evidence_id: str,
    score: float,
    rank: int,
    *,
    document_id: str | None = None,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id or f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=f"passage-{evidence_id}",
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=score,
        retrieval_rank=rank,
    )


def candidate_set(*items: EvidenceCandidate) -> CandidateSet:
    return CandidateSet(query_id="q-1", candidates=items)


class PassageAnswerGenerator:
    def __init__(self, answers: Mapping[str, str], *, failure: Exception | None = None) -> None:
        self.answers = answers
        self.failure = failure
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.failure is not None:
            raise self.failure
        passage = prompt.split("Passage: ", maxsplit=1)[1].rsplit("\nAnswer:", maxsplit=1)[0]
        return self.answers[passage]


def _statement_answer(statement: str) -> str:
    return statement.rsplit("\nis ", maxsplit=1)[1].removesuffix(".")


class PairScorer:
    model_id = "fake-nli"
    model_revision = "test-revision"

    def __init__(
        self,
        scores: Mapping[tuple[str, str], float] | None = None,
        *,
        failure: SelectorBackendError | None = None,
        wrong_length: bool = False,
    ) -> None:
        self.scores = scores or {}
        self.failure = failure
        self.wrong_length = wrong_length
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        batch = tuple(pairs)
        self.calls.append(batch)
        if self.failure is not None:
            raise self.failure
        result = tuple(
            self.scores.get((_statement_answer(left), _statement_answer(right)), 0.0)
            for left, right in batch
        )
        return result[:-1] if self.wrong_length and result else result


def selector(
    answers: Mapping[str, str],
    scores: Mapping[tuple[str, str], float] | None = None,
    *,
    parents: Mapping[str, str] | None = None,
    top_n: int = 20,
    threshold: float = 0.5,
    events: list[MISSelectionEvent] | None = None,
    scorer: PairScorer | None = None,
) -> tuple[ReliabilityMISSelector, PassageAnswerGenerator, PairScorer]:
    generator = PassageAnswerGenerator(answers)
    actual_scorer = scorer or PairScorer(scores)
    implementation = ReliabilityMISSelector(
        generator,
        actual_scorer,
        parents or {},
        top_n=top_n,
        contradiction_threshold=threshold,
        on_event=events.append if events is not None else None,
    )
    return implementation, generator, actual_scorer


def selected_ids(result: object) -> tuple[str, ...]:
    return tuple(item.evidence_id for item in result.items)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "NONE",
        "Answer: none.",
        "unknown",
        "I don't know",
        "I do not know.",
        "not in the passage",
        "The answer is not in the evidence provided.",
    ],
)
def test_unanswerable_recognizer_is_deliberately_small(raw: str) -> None:
    assert is_unanswerable(raw)


def test_answer_cleanup_and_statement_are_fixed() -> None:
    assert clean_isolated_answer(" Answer:  Bristol ") == "Bristol"
    assert answer_statement("Where?", "Answer: Bristol") == (
        "The answer to the question: Where?\nis Bristol."
    )
    assert not is_unanswerable("There is no evidence of rain")


def test_pair_batch_uses_only_valid_answers_in_i_less_than_j_order() -> None:
    pairs, indices = build_pair_batch("Where?", ("Bristol", "NONE", "London"))
    assert indices == ((0, 2),)
    assert tuple(map(_statement_answer, pairs[0])) == ("Bristol", "London")


def test_conflict_threshold_is_inclusive_and_graph_is_undirected() -> None:
    adjacency, edge_count = build_conflict_graph(
        3,
        ((0, 1), (0, 2), (1, 2)),
        (0.5, 0.4999, 0.8),
        threshold=0.5,
    )
    assert edge_count == 2
    assert adjacency == (1 << 1, (1 << 0) | (1 << 2), 1 << 1)


@pytest.mark.parametrize("probability", [float("nan"), -0.1, 1.1, "invalid"])
def test_invalid_nli_probability_is_a_backend_error(probability: object) -> None:
    with pytest.raises(SelectorBackendError):
        build_conflict_graph(2, ((0, 1),), (probability,), threshold=0.5)


def test_capacity_limited_independent_set_uses_parent_count_then_size_then_rank() -> None:
    # Vertices 0/1/2 all conflict with 3/4; the left side has more passages but
    # only one parent, so the independently sourced right side must win.
    adjacency = (
        (1 << 3) | (1 << 4),
        (1 << 3) | (1 << 4),
        (1 << 3) | (1 << 4),
        (1 << 0) | (1 << 1) | (1 << 2),
        (1 << 0) | (1 << 1) | (1 << 2),
    )
    assert select_capacity_limited_independent_set(
        adjacency,
        ("same", "same", "same", "independent-a", "independent-b"),
        capacity=4,
    ) == (3, 4)

    # Equal parent count and size falls to canonical retrieval order.
    assert select_capacity_limited_independent_set((2, 1), ("a", "b"), capacity=1) == (0,)


def test_path_conflict_retains_compatible_endpoints() -> None:
    items = candidate_set(
        candidate("a", 0.9, 1),
        candidate("b", 0.8, 2),
        candidate("c", 0.7, 3),
    )
    implementation, _generator, _scorer = selector(
        {"passage-a": "A", "passage-b": "B", "passage-c": "C"},
        {("A", "B"): 0.9, ("B", "C"): 0.9},
    )
    result = implementation.select(Query(query_id="q-1", text="Which?"), items, 3)
    assert selected_ids(result) == ("a", "c")


def test_single_candidate_skips_nli_and_returns_top_k() -> None:
    items = candidate_set(candidate("a", 0.9, 1))
    query = Query(query_id="q-1", text="Which?")
    implementation, _generator, scorer = selector({"passage-a": "A"})
    assert implementation.select(query, items, 1) == TopKSelector().select(query, items, 1)
    assert scorer.calls == []


def test_three_consistent_passages_beat_one_conflicting_passage() -> None:
    items = candidate_set(
        candidate("a1", 0.9, 1),
        candidate("a2", 0.8, 2),
        candidate("a3", 0.7, 3),
        candidate("b", 0.6, 4),
    )
    implementation, _generator, _scorer = selector(
        {
            "passage-a1": "A",
            "passage-a2": "A",
            "passage-a3": "A",
            "passage-b": "B",
        },
        {("A", "B"): 0.9},
    )
    result = implementation.select(Query(query_id="q-1", text="Which?"), items, 4)
    assert selected_ids(result) == ("a1", "a2", "a3")


def test_repeated_same_parent_does_not_outvote_two_independent_sources() -> None:
    items = candidate_set(
        candidate("a1", 0.95, 1, document_id="doc-a1"),
        candidate("a2", 0.94, 2, document_id="doc-a2"),
        candidate("a3", 0.93, 3, document_id="doc-a3"),
        candidate("b1", 0.8, 4, document_id="doc-b1"),
        candidate("b2", 0.7, 5, document_id="doc-b2"),
    )
    answers = {
        "passage-a1": "A",
        "passage-a2": "A",
        "passage-a3": "A",
        "passage-b1": "B",
        "passage-b2": "B",
    }
    cross_edges = {("A", "B"): 0.99}
    implementation, _generator, _scorer = selector(
        answers,
        cross_edges,
        parents={
            "doc-a1": "parent-a",
            "doc-a2": "parent-a",
            "doc-a3": "parent-a",
            "doc-b1": "parent-b1",
            "doc-b2": "parent-b2",
        },
    )
    assert selected_ids(implementation.select(Query(query_id="q-1", text="Which?"), items, 4)) == (
        "b1",
        "b2",
    )


def test_conflicting_twins_from_one_parent_remain_separate_passage_nodes() -> None:
    items = candidate_set(
        candidate("original", 0.9, 1, document_id="doc-original"),
        candidate("twin", 0.8, 2, document_id="doc-twin"),
    )
    implementation, _generator, _scorer = selector(
        {"passage-original": "True", "passage-twin": "False"},
        {("True", "False"): 0.9},
        parents={"doc-original": "same-parent", "doc-twin": "same-parent"},
    )
    assert selected_ids(implementation.select(Query(query_id="q-1", text="Which?"), items, 2)) == (
        "original",
    )


def test_no_conflict_is_exact_existing_top_k_even_beyond_analysis_window() -> None:
    items = candidate_set(
        candidate("c", 0.7, 3),
        candidate("a", 0.9, 1),
        candidate("b", 0.8, 2),
    )
    query = Query(query_id="q-1", text="Which?")
    implementation, _generator, _scorer = selector(
        {"passage-a": "A", "passage-b": "A"},
        top_n=2,
    )
    expected = TopKSelector().select(query, items, 3)
    assert implementation.select(query, items, 3) == expected


def test_all_none_skips_nli_and_returns_exact_top_k() -> None:
    items = candidate_set(candidate("a", 0.9, 1), candidate("b", 0.8, 2))
    query = Query(query_id="q-1", text="Which?")
    implementation, _generator, scorer = selector(
        {"passage-a": "NONE", "passage-b": "Answer: unknown"}
    )
    assert implementation.select(query, items, 1) == TopKSelector().select(query, items, 1)
    assert scorer.calls == []


def test_none_node_is_retained_but_does_not_enter_conflict_pairs() -> None:
    items = candidate_set(
        candidate("none", 0.95, 1),
        candidate("a", 0.9, 2),
        candidate("b", 0.8, 3),
    )
    implementation, _generator, scorer = selector(
        {"passage-none": "NONE", "passage-a": "A", "passage-b": "B"},
        {("A", "B"): 0.9},
    )
    result = implementation.select(Query(query_id="q-1", text="Which?"), items, 3)
    assert selected_ids(result) == ("none", "a")
    assert len(scorer.calls[0]) == 1


def test_conflict_path_never_backfills_unchecked_tail() -> None:
    items = candidate_set(
        candidate("a", 0.9, 1),
        candidate("b", 0.8, 2),
        candidate("tail", 0.7, 3),
    )
    implementation, _generator, _scorer = selector(
        {"passage-a": "A", "passage-b": "B"},
        {("A", "B"): 0.9},
        top_n=2,
    )
    assert selected_ids(implementation.select(Query(query_id="q-1", text="Which?"), items, 2)) == (
        "a",
    )


def test_input_permutation_does_not_change_answer_pair_direction_or_output() -> None:
    high = candidate("high", 0.9, 1)
    low = candidate("low", 0.8, 2)
    query = Query(query_id="q-1", text="Which?")
    outputs = []
    pair_answers = []
    for items in (candidate_set(high, low), candidate_set(low, high)):
        implementation, _generator, scorer = selector(
            {"passage-high": "A", "passage-low": "B"},
            {("A", "B"): 0.9},
        )
        outputs.append(selected_ids(implementation.select(query, items, 2)))
        pair_answers.append(tuple(map(_statement_answer, scorer.calls[0][0])))
    assert outputs == [("high",), ("high",)]
    assert pair_answers == [("A", "B"), ("A", "B")]


def test_same_input_repeatedly_produces_the_same_contract_valid_result() -> None:
    items = candidate_set(
        candidate("a", 0.9, 1),
        candidate("b", 0.8, 2),
        candidate("c", 0.7, 3),
    )
    implementation, _generator, _scorer = selector(
        {"passage-a": "A", "passage-b": "B", "passage-c": "C"},
        {("A", "B"): 0.9, ("B", "C"): 0.9},
    )
    query = Query(query_id="q-1", text="Which?")
    first = implementation.select(query, items, 2)
    second = implementation.select(query, items, 2)
    assert first == second
    assert len(first.items) <= 2
    assert tuple(item.selection_rank for item in first.items) == (1, 2)
    by_id = {candidate.evidence_id: candidate for candidate in items.candidates}
    assert all(
        item.selection_score == by_id[item.evidence_id].retrieval_score for item in first.items
    )


def test_answer_backend_failure_atomically_falls_back_to_top_k() -> None:
    items = candidate_set(candidate("a", 0.9, 1), candidate("b", 0.8, 2))
    query = Query(query_id="q-1", text="Which?")
    events: list[MISSelectionEvent] = []
    failing = PassageAnswerGenerator({}, failure=RuntimeError("offline"))
    implementation = ReliabilityMISSelector(
        LazyAnswerGenerator(lambda: failing),
        PairScorer(),
        {},
        on_event=events.append,
    )
    assert implementation.select(query, items, 1) == TopKSelector().select(query, items, 1)
    assert events[-1].mode == "topk_backend_fallback"
    assert events[-1].fallback_stage == "answer"


def test_nli_backend_failure_atomically_falls_back_to_top_k() -> None:
    items = candidate_set(candidate("a", 0.9, 1), candidate("b", 0.8, 2))
    query = Query(query_id="q-1", text="Which?")
    events: list[MISSelectionEvent] = []
    scorer = PairScorer(failure=SelectorBackendError("offline"))
    implementation, _generator, _scorer = selector(
        {"passage-a": "A", "passage-b": "B"},
        events=events,
        scorer=scorer,
    )
    assert implementation.select(query, items, 1) == TopKSelector().select(query, items, 1)
    assert events[-1].fallback_stage == "nli"


def test_wrong_nli_length_is_treated_as_backend_failure() -> None:
    items = candidate_set(candidate("a", 0.9, 1), candidate("b", 0.8, 2))
    events: list[MISSelectionEvent] = []
    implementation, _generator, _scorer = selector(
        {"passage-a": "A", "passage-b": "B"},
        events=events,
        scorer=PairScorer(wrong_length=True),
    )
    result = implementation.select(Query(query_id="q-1", text="Which?"), items, 2)
    assert selected_ids(result) == ("a", "b")
    assert events[-1].mode == "topk_backend_fallback"


def test_event_reports_unresolved_parent_without_leaking_passage_or_answer() -> None:
    items = candidate_set(
        candidate("a", 0.9, 1, document_id="known"),
        candidate("b", 0.8, 2, document_id="missing"),
    )
    events: list[MISSelectionEvent] = []
    implementation, _generator, _scorer = selector(
        {"passage-a": "A", "passage-b": "B"},
        {("A", "B"): 0.9},
        parents={"known": "parent-a"},
        events=events,
    )
    implementation.select(Query(query_id="q-1", text="Which?"), items, 2)
    event = events[-1]
    assert event.unresolved_parent_count == 1
    assert event.model_id == "fake-nli"
    assert event.contradiction_edge_count == 1
    assert event.threshold == 0.5
    assert "passage" not in event.__dict__
    assert "answers" not in event.__dict__


def test_event_sink_failure_never_changes_selection() -> None:
    items = candidate_set(candidate("a", 0.9, 1))

    def broken_sink(_event: MISSelectionEvent) -> None:
        raise RuntimeError("logging unavailable")

    implementation = ReliabilityMISSelector(
        PassageAnswerGenerator({"passage-a": "A"}),
        PairScorer(),
        {},
        on_event=broken_sink,
    )
    assert selected_ids(implementation.select(Query(query_id="q-1", text="Which?"), items, 1)) == (
        "a",
    )


def test_empty_candidates_do_not_construct_lazy_answer_backend() -> None:
    constructed = 0

    def factory() -> PassageAnswerGenerator:
        nonlocal constructed
        constructed += 1
        return PassageAnswerGenerator({})

    implementation = ReliabilityMISSelector(LazyAnswerGenerator(factory), PairScorer(), {})
    result = implementation.select(
        Query(query_id="q-1", text="Which?"),
        CandidateSet(query_id="q-1", candidates=()),
        2,
    )
    assert result.items == ()
    assert constructed == 0


def test_lazy_answer_backend_is_constructed_once_and_reused() -> None:
    constructed = 0
    backend = PassageAnswerGenerator({"passage-a": "A"})

    def factory() -> PassageAnswerGenerator:
        nonlocal constructed
        constructed += 1
        return backend

    lazy = LazyAnswerGenerator(factory)
    assert lazy.generate("Passage: passage-a\nAnswer:") == "A"
    assert lazy.generate("Passage: passage-a\nAnswer:") == "A"
    assert constructed == 1


def test_configuration_and_programming_errors_fail_loudly() -> None:
    with pytest.raises(ValueError, match="top_n"):
        ReliabilityMISSelector(PassageAnswerGenerator({}), PairScorer(), {}, top_n=21)
    with pytest.raises(ValueError, match="query and candidates"):
        ReliabilityMISSelector(PassageAnswerGenerator({}), PairScorer(), {}).select(
            Query(query_id="q-other", text="Which?"),
            CandidateSet(query_id="q-1", candidates=()),
            1,
        )
