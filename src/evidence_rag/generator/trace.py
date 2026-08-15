"""Runtime-only observability models for the Generator evidence flow.

These records expose existing decisions without changing them. They intentionally
contain no reference answer, gold provenance, utility label, or evaluation score.
"""

from typing import Literal

from evidence_rag.contracts.models import FrozenModel, NonEmpty

SplitterStatus = Literal[
    "not_run_empty_draft",
    "structured",
    "degraded_fallback",
    "failed",
    "unavailable",
]
DraftEmptyReason = Literal["", "no_selected_evidence", "model_decline_or_empty", "unknown"]
DraftMode = Literal["ordinary", "key_fact_notes", "key_fact_fallback", "external"]
FinalEmptyReason = Literal[
    "",
    "empty_draft",
    "splitter_no_claims",
    "all_claims_unfaithful",
    "all_claims_removed_or_empty",
    "capped_no_verified",
]
ClaimDisposition = Literal[
    "verified",
    "unverified",
    "dropped_entity_conflict",
    "skipped_unfaithful",
    "skipped_empty_sentence",
]


class SplitterTrace(FrozenModel):
    schema_version: Literal["full-flow-splitter-trace-v1"] = "full-flow-splitter-trace-v1"
    status: SplitterStatus
    split_raw_output: str | None = None
    faithfulness_raw_output: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    raw_claim_count: int = 0
    located_claim_count: int = 0
    retained_claim_count: int = 0
    output_claim_count: int = 0


class DraftTextTrace(FrozenModel):
    schema_version: Literal["full-flow-draft-text-trace-v1"] = (
        "full-flow-draft-text-trace-v1"
    )
    raw_output: str
    raw_output_available: bool
    empty_reason: DraftEmptyReason = ""


class DraftStageTrace(FrozenModel):
    schema_version: Literal["full-flow-draft-stage-trace-v1"] = (
        "full-flow-draft-stage-trace-v1"
    )
    mode: DraftMode
    raw_draft_text: str
    normalized_draft_text: str
    raw_output_available: bool
    draft_empty_reason: DraftEmptyReason = ""
    key_fact_note_count: int = 0
    splitter: SplitterTrace


class ClaimTrace(FrozenModel):
    schema_version: Literal["full-flow-claim-trace-v1"] = "full-flow-claim-trace-v1"
    claim_id: NonEmpty
    claim_text: NonEmpty
    span_start: int
    span_end: int
    faithful_to_answer: bool
    degraded_splitter_fallback: bool
    final_disposition: ClaimDisposition
    routing_outcome: str = ""
    citation: str | None = None
    declared_indices: tuple[int, ...] = ()
    gated_outcome: str = ""
    gated_citation: str | None = None
    conflict_evidence_id: str | None = None
    final_sentence: str = ""


class GeneratorTrace(FrozenModel):
    """One query's complete runtime Generator trace.

    The schema is deliberately limited to fields available after Selector output.
    Evaluation code may join this trace to gold only after generation completes.
    """

    schema_version: Literal["full-flow-generator-trace-v1"] = (
        "full-flow-generator-trace-v1"
    )
    query_id: NonEmpty
    selected_evidence_ids: tuple[NonEmpty, ...]
    draft: DraftStageTrace
    claims: tuple[ClaimTrace, ...]
    final_answer: str
    final_cited_evidence_ids: tuple[NonEmpty, ...]
    final_empty_reason: FinalEmptyReason = ""


def unavailable_draft_trace(answer_text: str) -> DraftStageTrace:
    """Trace an injected legacy draft producer that has no internal instrumentation."""

    return DraftStageTrace(
        mode="external",
        raw_draft_text=answer_text,
        normalized_draft_text=answer_text,
        raw_output_available=False,
        draft_empty_reason="unknown" if not answer_text.strip() else "",
        splitter=SplitterTrace(status="unavailable", output_claim_count=0),
    )
