from evidence_rag.contracts.models import (
    CandidateSet,
    GenerationResult,
    SelectedEvidenceSet,
    SelectionResult,
)


def resolve_selection(
    candidates: CandidateSet,
    selection: SelectionResult,
) -> SelectedEvidenceSet:
    if candidates.query_id != selection.query_id:
        raise ValueError("candidate and selection query IDs differ")
    by_id = {item.evidence_id: item for item in candidates.candidates}
    selected_ids = tuple(item.evidence_id for item in selection.items)
    unknown = tuple(evidence_id for evidence_id in selected_ids if evidence_id not in by_id)
    if unknown:
        raise ValueError(f"selection contains unknown evidence: {unknown}")
    return SelectedEvidenceSet(
        query_id=selection.query_id,
        evidence=tuple(by_id[evidence_id] for evidence_id in selected_ids),
    )


def validate_generation(
    selected: SelectedEvidenceSet,
    result: GenerationResult,
) -> None:
    if selected.query_id != result.query_id:
        raise ValueError("selected evidence and generation query IDs differ")
    allowed = {item.evidence_id for item in selected.evidence}
    unknown = set(result.cited_evidence_ids) - allowed
    if unknown:
        raise ValueError(f"generation cites unselected evidence: {sorted(unknown)}")
