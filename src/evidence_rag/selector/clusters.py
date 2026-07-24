"""Answer clusters over the rerank window (spec §6.3).

Independent support counts distinct document_id values — chunks of one document
are one vote. Parametric answers never enter clusters (spec §8: signal graduation).
"""

from collections.abc import Callable, Sequence
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


def build_clusters_lenient(
    window: Sequence[EvidenceCandidate],
    answers: Sequence[str],
    equivalence: Callable[[str, str], bool],
) -> tuple[AnswerCluster, ...]:
    """Representative-anchored greedy clustering under a lenient equivalence (spec 2026-07-23).

    Processes candidates in window order (the caller sorts by retrieval_rank). Each valid answer
    joins the first existing cluster whose representative — its founding member's raw answer — it is
    `equivalence`-equal to, else opens a new cluster. Comparing only against representatives avoids
    transitive chaining, so a bridging answer cannot merge two otherwise-distinct clusters. Independent
    support counts distinct document_id values, matching build_clusters. cluster.answer is the
    representative's canonical form; since a lenient equivalence that is a superset of canonical
    equality yields pairwise non-equivalent representatives, distinct clusters keep distinct answers.
    """
    if len(window) != len(answers):
        raise ValueError("window and answers lengths differ")
    representatives: list[str] = []
    members: list[list[EvidenceCandidate]] = []
    for candidate, answer in zip(window, answers, strict=True):
        if not is_valid_answer(answer):
            continue
        for index, representative in enumerate(representatives):
            if equivalence(answer, representative):
                members[index].append(candidate)
                break
        else:
            representatives.append(answer)
            members.append([candidate])
    return tuple(
        AnswerCluster(
            answer=canonicalize_answer(representatives[index]),
            member_ids=tuple(candidate.evidence_id for candidate in group),
            independent_support=len({candidate.document_id for candidate in group}),
        )
        for index, group in enumerate(members)
    )
