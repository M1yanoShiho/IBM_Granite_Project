from evidence_rag.contracts.models import (
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)


class ExtractiveGenerator:
    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        if query.query_id != selected.query_id:
            raise ValueError("query and selected evidence query IDs differ")
        if not selected.evidence:
            return GenerationResult(
                query_id=query.query_id,
                answer="",
                cited_evidence_ids=(),
            )
        answer = "\n".join(
            f"- {item.text} [{item.evidence_id}]" for item in selected.evidence
        )
        return GenerationResult(
            query_id=query.query_id,
            answer=answer,
            cited_evidence_ids=tuple(item.evidence_id for item in selected.evidence),
        )
