"""Versioned request and response models exposed to frontend clients."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

NonEmpty = Annotated[str, Field(min_length=1)]
PositiveRank = Annotated[int, Field(ge=1)]


class ApiModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class QueryRequest(ApiModel):
    schema_version: Literal["1.0"] = "1.0"
    query: NonEmpty
    query_id: NonEmpty | None = None
    session_id: NonEmpty | None = None
    top_k: Annotated[int, Field(ge=1, le=100)] = 10
    max_selected: Annotated[int, Field(ge=1, le=10)] = 10

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class EvidenceItem(ApiModel):
    evidence_id: NonEmpty
    document_id: NonEmpty
    text: NonEmpty
    source_uri: NonEmpty
    retrieval_score: FiniteFloat
    retrieval_rank: PositiveRank


class RuntimeDiagnostics(ApiModel):
    retriever: Literal["hybrid"] = "hybrid"
    selector: Literal["nli-risk-controlled"] = "nli-risk-controlled"
    generator: Literal["grounded-grc"] = "grounded-grc"
    candidate_count: Annotated[int, Field(ge=0)]
    selected_count: Annotated[int, Field(ge=0)]
    dropped_count: Annotated[int, Field(ge=0)]
    citation_count: Annotated[int, Field(ge=0)]


class QueryResponse(ApiModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    session_id: NonEmpty | None = None
    answer: str
    candidates: tuple[EvidenceItem, ...]
    selected_evidence: tuple[EvidenceItem, ...]
    citations: tuple[EvidenceItem, ...]
    diagnostics: RuntimeDiagnostics


class HealthResponse(ApiModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["ok"] = "ok"
    model_loaded: bool
    runtime: Literal["hybrid-nli-grc-seed13"] = "hybrid-nli-grc-seed13"
