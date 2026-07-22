from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.selector.gated import GatedCoverageSelector


class MappedExtractor:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def generate(self, prompt: str) -> str:
        if "own knowledge" in prompt:
            return "NONE"
        for text, answer in self.mapping.items():
            if f"Passage: {text}" in prompt:
                return answer
        return "NONE"


def cand(evidence_id: str, text: str, score: float, rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        chunk_id=f"c-{evidence_id}",
        text=text,
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=score,
        retrieval_rank=rank,
    )


QUERY = Query(query_id="q", text="IBM 2025 operating margin")


def test_coverage_keeps_answer_and_complementary_drops_noise() -> None:
    pool = CandidateSet(
        query_id="q",
        candidates=(
            cand("ans", "IBM 2025 operating margin was 18 percent", 0.9, 1),
            cand("comp", "IBM 2025 revenue 1.2B and operating income detail", 0.5, 2),
            cand("noise", "A recipe for chocolate cake with sugar", 0.6, 3),
        ),
    )
    extractor = MappedExtractor(
        {"IBM 2025 operating margin was 18 percent": "18 percent"}
    )
    selector = GatedCoverageSelector(extractor, use_parametric=False)
    ids = tuple(item.evidence_id for item in selector.select(QUERY, pool, 2).items)
    assert ids[0] == "ans"
    assert "comp" in ids and "noise" not in ids


def test_coverage_selector_respects_shortfall() -> None:
    pool = CandidateSet(
        query_id="q",
        candidates=(cand("ans", "IBM 2025 operating margin was 18 percent", 0.9, 1),),
    )
    extractor = MappedExtractor(
        {"IBM 2025 operating margin was 18 percent": "18 percent"}
    )
    selector = GatedCoverageSelector(extractor, use_parametric=False)
    assert len(selector.select(QUERY, pool, 5).items) == 1
