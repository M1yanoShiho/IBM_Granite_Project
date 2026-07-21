"""Set-level coverage selection (A2 spec §5-§7).

Complementarity is a property of the selected set, so selection moves from
pointwise-score-and-truncate to greedy set cover over deterministic query+answer
features. No NLI, no absolute threshold; the drop gate still governs removal — this
module only packs survivors. Feature extraction is a deterministic heuristic whose
error the E1-support component eval measures (spec §9).
"""

import re
from collections.abc import Mapping, Sequence

from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.selector.answer_norm import STOPWORDS, canonicalize_answer

_NUMBER_TOKEN = re.compile(
    r"(?:us\$|\$|£|€)?\d[\d,]*(?:\.\d+)?\s?"
    r"(?:%|percent|k|m|b|thousand|million|billion|trillion)?",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9&.\-]{2,}")
_UNIT_WORDS = frozenset({"percent", "thousand", "million", "billion", "trillion"})


def salient_features(text: str) -> frozenset[str]:
    features: set[str] = set()
    for match in _NUMBER_TOKEN.finditer(text):
        token = match.group(0).strip()
        if not any(char.isdigit() for char in token):
            continue
        canonical = canonicalize_answer(token)
        if canonical:
            features.add(f"num:{canonical}")
    for match in _WORD.finditer(text):
        token = match.group(0).strip(".").lower()
        if token and token not in STOPWORDS and token not in _UNIT_WORDS and not token.isdigit():
            features.add(f"tok:{token}")
    return frozenset(features)


def coverage_select(
    survivors: Sequence[EvidenceCandidate],
    *,
    blended_by_id: Mapping[str, float],
    answer_by_id: Mapping[str, str | None],
    query_text: str,
    max_selected: int,
) -> tuple[EvidenceCandidate, ...]:
    if max_selected <= 0 or not survivors:
        return ()

    clusters: dict[str, list[EvidenceCandidate]] = {}
    for candidate in survivors:
        answer = answer_by_id.get(candidate.evidence_id)
        if answer is not None:
            clusters.setdefault(answer, []).append(candidate)

    def support(answer: str) -> int:
        return len({member.document_id for member in clusters[answer]})

    ordered_clusters = sorted(clusters, key=lambda answer: (-support(answer), answer))

    feature_cache: dict[str, frozenset[str]] = {
        candidate.evidence_id: salient_features(candidate.text) for candidate in survivors
    }
    subject: set[str] = set(salient_features(query_text))
    if ordered_clusters:
        for member in clusters[ordered_clusters[0]]:
            subject |= feature_cache[member.evidence_id]

    universe: set[str] = set()
    for candidate in survivors:
        if feature_cache[candidate.evidence_id] & subject:
            universe |= feature_cache[candidate.evidence_id]

    def features(candidate: EvidenceCandidate) -> frozenset[str]:
        return feature_cache[candidate.evidence_id] & universe

    def blended(candidate: EvidenceCandidate) -> float:
        return blended_by_id.get(candidate.evidence_id, candidate.retrieval_score)

    output: list[EvidenceCandidate] = []
    pinned: set[str] = set()
    covered: set[str] = set()
    for answer in ordered_clusters:
        if len(output) >= max_selected:
            break
        representative = min(
            clusters[answer],
            key=lambda c: (-blended(c), c.retrieval_rank, c.evidence_id),
        )
        output.append(representative)
        pinned.add(representative.evidence_id)
        covered |= features(representative)

    remaining = [c for c in survivors if c.evidence_id not in pinned]
    while len(output) < max_selected and remaining:
        remaining.sort(
            key=lambda c: (
                -len(features(c) - covered),
                -blended(c),
                c.retrieval_rank,
                c.evidence_id,
            )
        )
        chosen = remaining.pop(0)
        output.append(chosen)
        covered |= features(chosen)

    return tuple(output)
