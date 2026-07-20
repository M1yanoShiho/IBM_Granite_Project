"""Answer clusters over the rerank window (spec §6.3).

Independent support counts distinct document_id values — chunks of one document
are one vote. Parametric answers never enter clusters (spec §8: signal graduation).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.selector.answer_norm import canonicalize_answer, is_valid_answer


@dataclass(frozen=True)
class AnswerCluster:
    answer: str
    member_ids: tuple[str, ...]
    independent_support: int


def build_clusters(
    window: Sequence[EvidenceCandidate],
    answers: Sequence[str],
) -> tuple[AnswerCluster, ...]:
    if len(window) != len(answers):
        raise ValueError("window and answers lengths differ")
    groups: dict[str, list[EvidenceCandidate]] = {}
    for candidate, answer in zip(window, answers, strict=True):
        if not is_valid_answer(answer):
            continue
        groups.setdefault(canonicalize_answer(answer), []).append(candidate)
    return tuple(
        AnswerCluster(
            answer=answer,
            member_ids=tuple(candidate.evidence_id for candidate in members),
            independent_support=len({candidate.document_id for candidate in members}),
        )
        for answer, members in groups.items()
    )
