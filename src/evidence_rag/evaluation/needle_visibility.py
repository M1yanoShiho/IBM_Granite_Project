"""Needle-visibility triage for the low needle-gold-recovery (systematic-debugging Phase 1).

Model-free: partitions each injected query's needle by WHERE the injected gold alias sits
relative to what the extractor actually reads (chunk text truncated to passage_chars).
Separates "extractor never saw the answer" (truncation / chunk-misalignment — a bigger model
cannot help) from "answer was visible but extraction missed it" (the only regime where a
larger extractor could matter). Reuses the same word-boundary alias match as the eval.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.evaluation.cluster_eval import contains_alias

VISIBLE = "visible"
TRUNCATED = "truncated"
ABSENT = "absent"


def classify_needle(
    window: Sequence[EvidenceCandidate],
    needle_document_id: str,
    gold_aliases: Sequence[str],
    *,
    passage_chars: int,
) -> str | None:
    needle_chunks = [c for c in window if c.document_id == needle_document_id]
    if not needle_chunks:
        return None
    if any(contains_alias(c.text[:passage_chars], gold_aliases) for c in needle_chunks):
        return VISIBLE
    if any(contains_alias(c.text, gold_aliases) for c in needle_chunks):
        return TRUNCATED
    return ABSENT


@dataclass(frozen=True)
class VisibilitySummary:
    n_needle_in_window: int
    visible: int
    truncated: int
    absent: int
    visible_rate: float | None
    truncated_rate: float | None
    absent_rate: float | None


def summarize_visibility(classes: Iterable[str | None]) -> VisibilitySummary:
    scored = [value for value in classes if value is not None]
    n = len(scored)
    visible = sum(1 for value in scored if value == VISIBLE)
    truncated = sum(1 for value in scored if value == TRUNCATED)
    absent = sum(1 for value in scored if value == ABSENT)
    return VisibilitySummary(
        n_needle_in_window=n,
        visible=visible,
        truncated=truncated,
        absent=absent,
        visible_rate=(visible / n) if n else None,
        truncated_rate=(truncated / n) if n else None,
        absent_rate=(absent / n) if n else None,
    )
