from evidence_rag.contracts.models import GenerationResult, PipelineRun, Query
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.contracts.validation import resolve_selection, validate_generation
from evidence_rag.query_analysis import QueryAnalyzer, RuleBasedQueryAnalyzer


class EvidenceRAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        selector: Selector,
        generator: Generator,
        query_analyzer: QueryAnalyzer | None = None,
    ) -> None:
        self.retriever = retriever
        self.selector = selector
        self.generator = generator
        self.query_analyzer = query_analyzer or RuleBasedQueryAnalyzer()

    def run(
        self,
        query: Query,
        top_k: int = 20,
        max_selected: int = 10,
    ) -> GenerationResult:
        return self.run_with_trace(
            query,
            top_k=top_k,
            max_selected=max_selected,
        ).generation

    def run_with_trace(
        self,
        query: Query,
        top_k: int = 20,
        max_selected: int = 10,
    ) -> PipelineRun:
        checklist = self.query_analyzer.analyze(query)
        if checklist.query_id != query.query_id:
            raise ValueError("query analyzer returned the wrong query ID")
        candidates = self.retriever.retrieve(query, top_k)
        if candidates.query_id != query.query_id:
            raise ValueError("retriever returned the wrong query ID")
        selection = self.selector.select(query, candidates, max_selected)
        if selection.query_id != query.query_id:
            raise ValueError("selector returned the wrong query ID")
        selected = resolve_selection(candidates, selection)
        result = self.generator.generate(query, checklist, selected)
        validate_generation(selected, result)
        return PipelineRun(
            query=query,
            checklist=checklist,
            top_k=top_k,
            max_selected=max_selected,
            candidates=candidates,
            selection=selection,
            selected=selected,
            generation=result,
        )
