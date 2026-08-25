"""Build the runtime-safe Selector→Generator guidance sidecar.

The builder accepts a Selector trace only after selection has finished and copies
an explicit allowlist for evidence that remains in ``SelectedEvidenceSet``.  It
does not carry deleted text/IDs or any evaluation-only benchmark fields.
"""

from evidence_rag.contracts.models import (
    RetainedEvidenceGuidance,
    SelectedEvidenceSet,
    SelectionGuidance,
)
from evidence_rag.selector.models import SelectorAction, SelectorDecisionTrace


def build_selection_guidance(
    trace: SelectorDecisionTrace,
    selected: SelectedEvidenceSet,
) -> SelectionGuidance:
    if trace.query_id != selected.query_id:
        raise ValueError("Selector trace and selected evidence query IDs differ")
    selected_ids = tuple(item.evidence_id for item in selected.evidence)
    if trace.selected_evidence_ids != selected_ids:
        raise ValueError("Selector trace and selected evidence IDs differ")

    decision_by_id = {item.evidence_id: item for item in trace.decisions}
    retained: list[RetainedEvidenceGuidance] = []
    for evidence in selected.evidence:
        decision = decision_by_id.get(evidence.evidence_id)
        if decision is None:
            raise ValueError(f"Selector trace omits retained evidence {evidence.evidence_id!r}")
        if decision.action is SelectorAction.DROP_HARM:
            raise ValueError("dropped evidence cannot enter SelectionGuidance")
        retained.append(
            RetainedEvidenceGuidance(
                evidence_id=decision.evidence_id,
                retrieval_rank=decision.retrieval_rank,
                protect_signal=decision.protect_score,
                harm_signal=decision.harm_score,
                action=(
                    "KEEP"
                    if decision.action is SelectorAction.KEEP
                    else "ABSTAIN_KEEP"
                ),
                reason=decision.reason.value,
            )
        )
    return SelectionGuidance(
        query_id=trace.query_id,
        selector_changed=bool(trace.dropped_evidence_ids),
        dropped_count=len(trace.dropped_evidence_ids),
        retained=tuple(retained),
    )
