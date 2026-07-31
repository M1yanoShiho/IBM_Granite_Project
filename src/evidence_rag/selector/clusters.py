"""Answer clusters over the rerank window (spec §6.3).

Independent support counts distinct sources. The unit is selectable: `document_id` (chunks of one
document are one vote) or, when a `parent_by_document` sidecar is supplied, `source_parent_id`
(passages of one Wikipedia article are one vote). The parent unit is the correct reading of
"independent" on dpr-w100, where one article spans many document_ids — see
`materializer/source_parent.py`. Parametric answers never enter clusters (spec §8: signal
graduation).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.selector.answer_norm import canonicalize_answer, is_valid_answer


def _independent_support(
    members: Sequence[EvidenceCandidate],
    parent_by_document: Mapping[str, str] | None,
) -> int:
    """Distinct sources backing a cluster.

    Under the parent unit an unmapped document is its own parent, so a missing sidecar entry can
    never merge two genuinely distinct sources — it can only fail to merge two related ones.
    """
    if parent_by_document is None:
        return len({candidate.document_id for candidate in members})
    return len(
        {
            parent_by_document.get(candidate.document_id, candidate.document_id)
            for candidate in members
        }
    )


@dataclass(frozen=True)
class AnswerCluster:
    answer: str
    member_ids: tuple[str, ...]
    independent_support: int


def build_clusters(
    window: Sequence[EvidenceCandidate],
    answers: Sequence[str],
    *,
    parent_by_document: Mapping[str, str] | None = None,
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
            independent_support=_independent_support(members, parent_by_document),
        )
        for answer, members in groups.items()
    )


def build_clusters_lenient(
    window: Sequence[EvidenceCandidate],
    answers: Sequence[str],
    equivalence: Callable[[str, str], bool],
    *,
    parent_by_document: Mapping[str, str] | None = None,
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
            independent_support=_independent_support(group, parent_by_document),
        )
        for index, group in enumerate(members)
    )
