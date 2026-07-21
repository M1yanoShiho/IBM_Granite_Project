from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.pipeline.service import EvidenceRAGPipeline


def evidence(evidence_id: str, text: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


class RetrieverA:
    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        return CandidateSet(query_id=query.query_id, candidates=(evidence("ev-a", "A"),))


class RetrieverB:
    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        return CandidateSet(query_id=query.query_id, candidates=(evidence("ev-b", "B"),))


class SelectorA:
    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        item = candidates.candidates[0]
        return SelectionResult(
            query_id=query.query_id,
            items=(
                SelectionItem(
                    evidence_id=item.evidence_id,
                    selection_score=1.0,
                    selection_rank=1,
                ),
            ),
        )


class SelectorB:
    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        item = candidates.candidates[0]
        return SelectionResult(
            query_id=query.query_id,
            items=(
                SelectionItem(
                    evidence_id=item.evidence_id,
                    selection_score=0.5,
                    selection_rank=1,
                ),
            ),
        )


class GeneratorA:
    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        item = selected.evidence[0]
        return GenerationResult(
            query_id=query.query_id,
            answer=f"A:{item.text}",
            cited_evidence_ids=(item.evidence_id,),
        )


class GeneratorB:
    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        item = selected.evidence[0]
        return GenerationResult(
            query_id=query.query_id,
            answer=f"B:{item.text}",
            cited_evidence_ids=(item.evidence_id,),
        )


def run(
    retriever: Retriever,
    selector: Selector,
    generator: Generator,
) -> GenerationResult:
    return EvidenceRAGPipeline(retriever, selector, generator).run(
        Query(query_id="q", text="question"),
        top_k=1,
        max_selected=1,
    )


def test_retriever_can_be_replaced_alone() -> None:
    assert run(RetrieverA(), SelectorA(), GeneratorA()).answer == "A:A"
    assert run(RetrieverB(), SelectorA(), GeneratorA()).answer == "A:B"


def test_selector_can_be_replaced_alone() -> None:
    assert run(RetrieverA(), SelectorA(), GeneratorA()).answer == "A:A"
    assert run(RetrieverA(), SelectorB(), GeneratorA()).answer == "A:A"


def test_generator_can_be_replaced_alone() -> None:
    assert run(RetrieverA(), SelectorA(), GeneratorA()).answer == "A:A"
    assert run(RetrieverA(), SelectorA(), GeneratorB()).answer == "B:A"


def test_every_module_combination_runs() -> None:
    retrievers = (RetrieverA, RetrieverB)
    selectors = (SelectorA, SelectorB)
    generators = (GeneratorA, GeneratorB)

    answers = {
        run(retriever(), selector(), generator()).answer
        for retriever in retrievers
        for selector in selectors
        for generator in generators
    }

    assert answers == {"A:A", "A:B", "B:A", "B:B"}
