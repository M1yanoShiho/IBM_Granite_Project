from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    Query,
)
from evidence_rag.selector.corroboration import (
    CorroborationSelector,
    corroboration_scores,
    normalize_answer,
)


class FakeExtractor:
    def generate(self, prompt: str) -> str:
        if "own knowledge" in prompt:
            return "Acme"
        if "Passage: Acme" in prompt:
            return "Acme"
        if "Passage: Globex" in prompt:
            return "Globex"
        return "NONE"


def evidence(
    evidence_id: str,
    text: str,
    score: float,
    rank: int,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=score,
        retrieval_rank=rank,
    )


def test_corroboration_scores_count_matching_answers() -> None:
    assert normalize_answer("The Acme.") == "acme"
    assert corroboration_scores(("Acme", "Globex", "Acme"), "Acme") == (2.0, 0.0, 2.0)


def test_corroboration_selector_can_promote_cross_supported_evidence() -> None:
    selector = CorroborationSelector(FakeExtractor(), alpha=0.2, use_parametric=True)
    candidates = CandidateSet(
        query_id="q",
        candidates=(
            evidence("wrong", "Globex is named in a distractor.", 1.0, 1),
            evidence("right-a", "Acme is named in one source.", 0.7, 2),
            evidence("right-b", "Acme is named in another source.", 0.6, 3),
        ),
    )

    result = selector.select(Query(query_id="q", text="Which company?"), candidates, 2)

    assert tuple(item.evidence_id for item in result.items) == ("right-a", "right-b")
