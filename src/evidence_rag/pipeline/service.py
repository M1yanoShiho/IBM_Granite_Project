from evidence_rag.contracts.models import GenerationResult, Query
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.contracts.validation import resolve_selection, validate_generation


class EvidenceRAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        selector: Selector,
        generator: Generator,
    ) -> None:
        self.retriever = retriever
        self.selector = selector
        self.generator = generator

    def run(
        self,
        query: Query,
        top_k: int = 20,
        max_selected: int = 10,
    ) -> GenerationResult:
        candidates = self.retriever.retrieve(query, top_k)
        if candidates.query_id != query.query_id:
            raise ValueError("retriever returned the wrong query ID")
        selection = self.selector.select(query, candidates, max_selected)
        if selection.query_id != query.query_id:
            raise ValueError("selector returned the wrong query ID")
        selected = resolve_selection(candidates, selection)
        result = self.generator.generate(query, selected)
        validate_generation(selected, result)
        return result
