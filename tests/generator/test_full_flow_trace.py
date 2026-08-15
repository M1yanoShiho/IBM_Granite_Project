from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.draft import DraftAnswerGenerator, KeyFactDraftAnswerGenerator
from evidence_rag.generator.verify_annotate import (
    CitationRoutedVerifier,
    VerifyAnnotateGenerator,
)


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


class EntailingNLI:
    def classify(self, *, premise: str, hypothesis: str) -> str:
        return "entailment" if "Revenue rose" in premise and "Revenue rose" in hypothesis else "neutral"


@dataclass(frozen=True)
class _Consistency:
    consistent: bool = True
    mismatches: tuple[object, ...] = ()


class ConsistentEntityChecker:
    def check(self, claim: str, evidence: str) -> _Consistency:
        return _Consistency()


def _selected() -> SelectedEvidenceSet:
    return SelectedEvidenceSet(
        query_id="q-1",
        evidence=(
            EvidenceCandidate(
                evidence_id="ev-1",
                document_id="doc-1",
                chunk_id="chunk-1",
                text="Revenue rose by 8%.",
                source_uri="fixture://ev-1",
                retrieval_score=1.0,
                retrieval_rank=1,
            ),
        ),
    )


def _run(responses: list[str], *, trace_enabled: bool):
    llm = ScriptedLLM(responses)
    draft = DraftAnswerGenerator(llm=llm, trace_enabled=trace_enabled)
    generator = VerifyAnnotateGenerator(
        draft_generator=draft,
        verifier=CitationRoutedVerifier(
            EntailingNLI(), ConsistentEntityChecker(), entity_gate="observe"
        ),
        trace_enabled=trace_enabled,
    )
    result = generator.generate(
        Query(query_id="q-1", text="What changed?"),
        QueryChecklist(query_id="q-1", focus="change", required_facts=()),
        _selected(),
    )
    return result, generator, draft, llm


STRUCTURED_RESPONSES = [
    "Revenue rose 8% [1].",
    '{"claims":[{"source_text":"Revenue rose 8% [1].",'
    '"text":"Revenue rose by 8%."}]}',
    '{"results":[{"claim_id":"claim-1","faithful":true}]}',
]


def test_trace_on_and_off_preserve_calls_and_final_output() -> None:
    off_result, off_generator, off_draft, off_llm = _run(
        STRUCTURED_RESPONSES.copy(), trace_enabled=False
    )
    on_result, on_generator, on_draft, on_llm = _run(
        STRUCTURED_RESPONSES.copy(), trace_enabled=True
    )

    assert on_result == off_result
    assert on_llm.prompts == off_llm.prompts
    assert off_generator.last_trace is None
    assert off_draft.last_trace is None

    trace = on_generator.last_trace
    assert trace is not None
    assert trace.draft.raw_draft_text == STRUCTURED_RESPONSES[0]
    assert trace.draft.splitter.status == "structured"
    assert trace.draft.splitter.split_raw_output == STRUCTURED_RESPONSES[1]
    assert trace.draft.splitter.faithfulness_raw_output == STRUCTURED_RESPONSES[2]
    assert trace.claims[0].final_disposition == "verified"
    assert trace.claims[0].citation == "ev-1"
    assert trace.final_answer == on_result.answer
    assert trace.final_empty_reason == ""


def test_parser_failure_is_distinct_from_an_empty_model_draft() -> None:
    _, degraded_generator, _, _ = _run(
        ["Revenue rose 8% [1].", "not json"], trace_enabled=True
    )
    degraded = degraded_generator.last_trace
    assert degraded is not None
    assert degraded.draft.draft_empty_reason == ""
    assert degraded.draft.splitter.status == "degraded_fallback"
    assert degraded.draft.splitter.failure_type == "ValueError"
    assert degraded.draft.splitter.output_claim_count == 1
    assert degraded.claims[0].degraded_splitter_fallback is True

    result, empty_generator, _, empty_llm = _run(["I don't know."], trace_enabled=True)
    empty = empty_generator.last_trace
    assert result.answer == ""
    assert empty is not None
    assert empty.draft.draft_empty_reason == "model_decline_or_empty"
    assert empty.draft.splitter.status == "not_run_empty_draft"
    assert empty.final_empty_reason == "empty_draft"
    assert len(empty_llm.prompts) == 1


def test_nonempty_draft_with_zero_split_claims_has_an_explicit_empty_reason() -> None:
    result, generator, _, _ = _run(
        ["Revenue rose 8% [1].", '{"claims":[]}'], trace_enabled=True
    )

    assert result.answer == ""
    assert generator.last_trace is not None
    assert generator.last_trace.draft.splitter.status == "structured"
    assert generator.last_trace.final_empty_reason == "splitter_no_claims"


def test_runtime_contract_rejects_gold_fields_and_trace_contains_no_gold_keys() -> None:
    with pytest.raises(ValidationError, match="gold_answer"):
        Query(query_id="q-1", text="What changed?", gold_answer="8%")  # type: ignore[call-arg]

    _, generator, _, _ = _run(STRUCTURED_RESPONSES.copy(), trace_enabled=True)
    assert generator.last_trace is not None
    serialized = generator.last_trace.model_dump_json()
    assert "gold" not in serialized.casefold()
    assert "reference_answer" not in serialized.casefold()


def test_key_fact_path_exposes_the_same_draft_and_splitter_trace() -> None:
    llm = ScriptedLLM(
        [
            '{"facts":[{"slot":"DATE_OR_YEAR","value":"2024","evidence":[1]}]}',
            "Revenue rose 8% [1].",
            STRUCTURED_RESPONSES[1],
            STRUCTURED_RESPONSES[2],
        ]
    )
    draft = KeyFactDraftAnswerGenerator(llm, trace_enabled=True)
    generator = VerifyAnnotateGenerator(
        draft_generator=draft,
        verifier=CitationRoutedVerifier(
            EntailingNLI(), ConsistentEntityChecker(), entity_gate="observe"
        ),
        trace_enabled=True,
    )

    generator.generate(
        Query(query_id="q-1", text="When did revenue change?"),
        QueryChecklist(query_id="q-1", focus="date", required_facts=()),
        _selected(),
    )

    assert generator.last_trace is not None
    assert generator.last_trace.draft.mode == "key_fact_notes"
    assert generator.last_trace.draft.key_fact_note_count == 1
    assert generator.last_trace.draft.splitter.status == "structured"
