from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

NonEmpty = Annotated[str, Field(min_length=1)]
PositiveRank = Annotated[int, Field(ge=1)]
PositiveLimit = Annotated[int, Field(ge=1)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Query(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    text: NonEmpty


class QueryChecklist(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    focus: NonEmpty
    required_facts: tuple[NonEmpty, ...]
    constraints: tuple[NonEmpty, ...] = ()


class Document(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    document_id: NonEmpty
    text: NonEmpty
    source_uri: NonEmpty


class EvidenceCandidate(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_id: NonEmpty
    document_id: NonEmpty
    chunk_id: NonEmpty
    text: NonEmpty
    source_uri: NonEmpty
    retrieval_score: FiniteFloat
    retrieval_rank: PositiveRank


class CandidateSet(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    candidates: tuple[EvidenceCandidate, ...]

    @model_validator(mode="after")
    def unique_ids_and_ranks(self) -> "CandidateSet":
        ids = tuple(item.evidence_id for item in self.candidates)
        ranks = tuple(item.retrieval_rank for item in self.candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("candidate evidence IDs must be unique")
        if len(ranks) != len(set(ranks)):
            raise ValueError("candidate retrieval ranks must be unique")
        return self


class SelectionItem(FrozenModel):
    evidence_id: NonEmpty
    selection_score: FiniteFloat
    selection_rank: PositiveRank


class SelectionResult(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    items: tuple[SelectionItem, ...]

    @model_validator(mode="after")
    def unique_ids_and_ranks(self) -> "SelectionResult":
        ids = tuple(item.evidence_id for item in self.items)
        ranks = tuple(item.selection_rank for item in self.items)
        if len(ids) != len(set(ids)):
            raise ValueError("selected evidence IDs must be unique")
        if len(ranks) != len(set(ranks)):
            raise ValueError("selection ranks must be unique")
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("selection items must be ordered by consecutive ranks")
        return self


class SelectedEvidenceSet(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    evidence: tuple[EvidenceCandidate, ...]

    @model_validator(mode="after")
    def unique_ids(self) -> "SelectedEvidenceSet":
        ids = tuple(item.evidence_id for item in self.evidence)
        if len(ids) != len(set(ids)):
            raise ValueError("selected evidence IDs must be unique")
        return self


class GenerationResult(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    answer: str
    cited_evidence_ids: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def answer_and_citations_match(self) -> "GenerationResult":
        if self.answer.strip() and not self.cited_evidence_ids:
            raise ValueError("nonempty answer requires at least one citation")
        if not self.answer.strip() and self.cited_evidence_ids:
            raise ValueError("empty answer cannot contain citations")
        if len(self.cited_evidence_ids) != len(set(self.cited_evidence_ids)):
            raise ValueError("citations must be unique")
        return self


class PipelineRun(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query: Query
    checklist: QueryChecklist
    top_k: PositiveLimit
    max_selected: PositiveLimit
    candidates: CandidateSet
    selection: SelectionResult
    selected: SelectedEvidenceSet
    generation: GenerationResult

    @model_validator(mode="after")
    def stages_are_consistent(self) -> "PipelineRun":
        expected = self.query.query_id
        if self.checklist.query_id != expected:
            raise ValueError("checklist query ID differs")
        stage_query_ids = (
            self.candidates.query_id,
            self.selection.query_id,
            self.selected.query_id,
            self.generation.query_id,
        )
        if any(query_id != expected for query_id in stage_query_ids):
            raise ValueError("pipeline run query IDs differ")
        if len(self.candidates.candidates) > self.top_k:
            raise ValueError("candidate count exceeds top_k")
        if len(self.selection.items) > self.max_selected:
            raise ValueError("selection count exceeds max_selected")

        candidates_by_id = {
            item.evidence_id: item for item in self.candidates.candidates
        }
        selected_ids = tuple(item.evidence_id for item in self.selection.items)
        unknown = tuple(
            evidence_id
            for evidence_id in selected_ids
            if evidence_id not in candidates_by_id
        )
        if unknown:
            raise ValueError(f"selection contains unknown evidence: {unknown}")
        expected_selected = tuple(
            candidates_by_id[evidence_id] for evidence_id in selected_ids
        )
        if self.selected.evidence != expected_selected:
            raise ValueError("selected evidence does not match selection")

        cited = set(self.generation.cited_evidence_ids)
        unknown_citations = cited - set(selected_ids)
        if unknown_citations:
            raise ValueError(
                f"generation cites unselected evidence: {sorted(unknown_citations)}"
            )
        return self
