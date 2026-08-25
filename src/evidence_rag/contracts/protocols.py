from typing import Protocol

from evidence_rag.contracts.models import (
    CandidateSet,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    SelectionResult,
)


class Retriever(Protocol):
    def retrieve(self, query: Query, top_k: int) -> CandidateSet: ...


class Selector(Protocol):
    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult: ...


class Generator(Protocol):
    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult: ...
