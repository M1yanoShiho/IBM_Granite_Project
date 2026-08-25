from evidence_rag.contracts.models import (
    CandidateSet,
    Query,
    SelectionItem,
    SelectionResult,
)


class TopKSelector:
    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        if query.query_id != candidates.query_id:
            raise ValueError("query and candidates query IDs differ")
        if max_selected <= 0:
            raise ValueError("max_selected must be positive")
        ranked = sorted(
            candidates.candidates,
            key=lambda item: (item.retrieval_rank, item.evidence_id),
        )
        return SelectionResult(
            query_id=query.query_id,
            items=tuple(
                SelectionItem(
                    evidence_id=item.evidence_id,
                    selection_score=item.retrieval_score,
                    selection_rank=rank,
                )
                for rank, item in enumerate(ranked[:max_selected], start=1)
            ),
        )
