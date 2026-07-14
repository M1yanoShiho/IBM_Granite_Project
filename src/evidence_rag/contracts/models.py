from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

NonEmpty = Annotated[str, Field(min_length=1)]
PositiveRank = Annotated[int, Field(ge=1)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Query(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    text: NonEmpty


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
