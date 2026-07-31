from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.pipeline.service import EvidenceRAGPipeline
from evidence_rag.selector.corroboration import CorroborationSelector
from evidence_rag.selector.gated import GatedCorroborationSelector, GateDecision


class MappedExtractor:
    """Maps passage text -> extracted answer; parametric answer configurable."""

    def __init__(self, mapping: dict[str, str], parametric: str = "NONE") -> None:
        self.mapping = mapping
        self.parametric = parametric

    def generate(self, prompt: str) -> str:
        if "own knowledge" in prompt:
            return self.parametric
        for text, answer in self.mapping.items():
            if f"Passage: {text}" in prompt:
                return answer
        return "NONE"


def evidence(
    evidence_id: str,
    text: str,
    score: float,
    rank: int,
    document_id: str | None = None,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id or f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=score,
        retrieval_rank=rank,
    )


QUERY = Query(query_id="q", text="Which value?")


def candidate_set(*items: EvidenceCandidate) -> CandidateSet:
    return CandidateSet(query_id="q", candidates=items)


def selected_ids(
    selector: GatedCorroborationSelector,
    candidates: CandidateSet,
    k: int,
) -> tuple[str, ...]:
    return tuple(item.evidence_id for item in selector.select(QUERY, candidates, k).items)


def three_vs_one_pool() -> tuple[CandidateSet, MappedExtractor]:
    pool = candidate_set(
        evidence("lone", "text lone", 1.0, 1),
        evidence("w1", "text w1", 0.9, 2),
        evidence("w2", "text w2", 0.8, 3),
        evidence("w3", "text w3", 0.7, 4),
    )
    extractor = MappedExtractor(
        {"text lone": "Globex", "text w1": "Acme", "text w2": "Acme", "text w3": "Acme"}
    )
    return pool, extractor


def test_lone_source_with_no_competitor_is_never_dropped() -> None:
    pool = candidate_set(
        evidence("only", "text only", 1.0, 1),
        evidence("bg1", "text bg1", 0.9, 2),
        evidence("bg2", "text bg2", 0.8, 3),
    )
    extractor = MappedExtractor({"text only": "Acme"})
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    assert set(selected_ids(selector, pool, 3)) == {"only", "bg1", "bg2"}


def test_one_vs_one_conflict_keeps_both() -> None:
    pool = candidate_set(
        evidence("a", "text a", 1.0, 1),
        evidence("b", "text b", 0.9, 2),
    )
    extractor = MappedExtractor({"text a": "Acme", "text b": "Globex"})
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    assert set(selected_ids(selector, pool, 2)) == {"a", "b"}


def test_three_vs_one_drops_isolated_claim_without_padding() -> None:
    pool, extractor = three_vs_one_pool()
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    ids = selected_ids(selector, pool, 4)
    assert "lone" not in ids
    assert len(ids) == 3


def test_two_vs_one_within_margin_keeps_both() -> None:
    pool = candidate_set(
        evidence("y", "text y", 1.0, 1),
        evidence("x1", "text x1", 0.9, 2),
        evidence("x2", "text x2", 0.8, 3),
    )
    extractor = MappedExtractor({"text y": "Globex", "text x1": "Acme", "text x2": "Acme"})
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    assert "y" in selected_ids(selector, pool, 3)


def test_support_cap_blocks_drop_of_supported_cluster() -> None:
    pool = candidate_set(
        evidence("y1", "text y1", 1.0, 1),
        evidence("y2", "text y2", 0.9, 2),
        evidence("x1", "text x1", 0.8, 3),
        evidence("x2", "text x2", 0.7, 4),
        evidence("x3", "text x3", 0.6, 5),
        evidence("x4", "text x4", 0.5, 6),
    )
    extractor = MappedExtractor(
        {
            "text y1": "Globex",
            "text y2": "Globex",
            "text x1": "Acme",
            "text x2": "Acme",
            "text x3": "Acme",
            "text x4": "Acme",
        }
    )
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    ids = selected_ids(selector, pool, 6)
    assert "y1" in ids and "y2" in ids


def test_alias_answers_cluster_together_and_gate_uses_merged_votes() -> None:
    pool = candidate_set(
        evidence("b", "text b", 1.0, 1),
        evidence("a1", "text a1", 0.9, 2),
        evidence("a2", "text a2", 0.8, 3),
    )
    extractor = MappedExtractor(
        {"text b": "3 billion", "text a1": "$1.2B", "text a2": "1,200 million"}
    )
    selector = GatedCorroborationSelector(extractor, margin=1, use_parametric=False)
    assert "b" not in selected_ids(selector, pool, 3)


def test_lenient_clustering_prevents_gold_fragment_false_drop() -> None:
    # Gold appears as two aliases across two documents; a distractor has 3 votes.
    pool = candidate_set(
        evidence("g1", "text g1", 1.0, 1, document_id="gold-1"),
        evidence("g2", "text g2", 0.95, 2, document_id="gold-2"),
        evidence("d1", "text d1", 0.9, 3, document_id="dist-1"),
        evidence("d2", "text d2", 0.8, 4, document_id="dist-2"),
        evidence("d3", "text d3", 0.7, 5, document_id="dist-3"),
    )
    extractor = MappedExtractor(
        {
            "text g1": "Apostle Paul",
            "text g2": "Paul",
            "text d1": "Acme",
            "text d2": "Acme",
            "text d3": "Acme",
        }
    )
    # exact: gold fragments to one vote each, both dropped by the 3-vote distractor.
    exact = GatedCorroborationSelector(extractor, use_parametric=False)
    exact_ids = selected_ids(exact, pool, 5)
    assert "g1" not in exact_ids and "g2" not in exact_ids
    # lenient: "Apostle Paul" + "Paul" merge to a 2-vote cluster, so gold survives.
    lenient = GatedCorroborationSelector(extractor, equivalence="lenient", use_parametric=False)
    lenient_ids = selected_ids(lenient, pool, 5)
    assert "g1" in lenient_ids and "g2" in lenient_ids


def test_same_document_chunks_collapse_to_one_vote() -> None:
    pool = candidate_set(
        evidence("x1", "text x1", 1.0, 1, document_id="doc-X"),
        evidence("x2", "text x2", 0.9, 2, document_id="doc-X"),
        evidence("x3", "text x3", 0.8, 3, document_id="doc-X"),
        evidence("y", "text y", 0.7, 4),
    )
    extractor = MappedExtractor(
        {"text x1": "Acme", "text x2": "Acme", "text x3": "Acme", "text y": "Globex"}
    )
    selector = GatedCorroborationSelector(extractor, margin=1, use_parametric=False)
    assert "y" in selected_ids(selector, pool, 4)


def test_gate_silent_when_nothing_extracts() -> None:
    pool = candidate_set(
        evidence("r1", "text r1", 1.0, 1),
        evidence("r2", "text r2", 0.9, 2),
        evidence("r3", "text r3", 0.8, 3),
    )
    selector = GatedCorroborationSelector(MappedExtractor({}), use_parametric=False)
    assert selected_ids(selector, pool, 3) == ("r1", "r2", "r3")


def test_drop_frees_budget_slot_for_replacement() -> None:
    pool, extractor = three_vs_one_pool()
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    ids = selected_ids(selector, pool, 3)
    assert set(ids) == {"w1", "w2", "w3"}


def test_matches_plain_corroboration_when_gate_never_fires() -> None:
    pool = candidate_set(
        evidence("wrong", "Globex is named in a distractor.", 1.0, 1),
        evidence("right-a", "Acme is named in one source.", 0.7, 2),
        evidence("right-b", "Acme is named in another source.", 0.6, 3),
    )
    extractor = MappedExtractor(
        {
            "Globex is named in a distractor.": "Globex",
            "Acme is named in one source.": "Acme",
            "Acme is named in another source.": "Acme",
        },
        parametric="Acme",
    )
    gated = GatedCorroborationSelector(extractor, alpha=0.2)
    plain = CorroborationSelector(extractor, alpha=0.2)
    assert gated.select(QUERY, pool, 2) == plain.select(QUERY, pool, 2)


def test_sink_receives_structured_decisions() -> None:
    decisions: list[GateDecision] = []
    pool, extractor = three_vs_one_pool()
    selector = GatedCorroborationSelector(
        extractor, use_parametric=False, on_gate_decision=decisions.append
    )
    selector.select(QUERY, pool, 4)
    by_id = {decision.evidence_id: decision for decision in decisions}
    assert len(decisions) == 4
    lone = by_id["lone"]
    assert lone.action == "drop"
    assert lone.own_support == 1
    assert lone.winner_support == 3
    assert lone.margin == 2
    assert by_id["w1"].action == "keep"


def test_deterministic_across_repeats() -> None:
    pool, extractor = three_vs_one_pool()
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    assert selector.select(QUERY, pool, 4) == selector.select(QUERY, pool, 4)


def test_invalid_arguments_raise() -> None:
    extractor = MappedExtractor({})
    for kwargs in (
        {"alpha": 1.5},
        {"margin": 0},
        {"support_cap": -1},
        {"top_n": 0},
        {"equivalence": "fuzzy"},
    ):
        try:
            GatedCorroborationSelector(extractor, **kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    selector = GatedCorroborationSelector(extractor)
    try:
        selector.select(Query(query_id="other", text="t"), candidate_set(), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected query ID mismatch error")


class PoolRetriever:
    def __init__(self, pool: CandidateSet) -> None:
        self.pool = pool

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        return self.pool


class CitingGenerator:
    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        first = selected.evidence[0]
        return GenerationResult(
            query_id=query.query_id,
            answer=first.text,
            cited_evidence_ids=(first.evidence_id,),
        )


def test_pipeline_accepts_gated_shortfall() -> None:
    pool, extractor = three_vs_one_pool()
    pipeline = EvidenceRAGPipeline(
        PoolRetriever(pool),
        GatedCorroborationSelector(extractor, use_parametric=False),
        CitingGenerator(),
    )
    run = pipeline.run_with_trace(QUERY, top_k=4, max_selected=4)
    assert len(run.selection.items) == 3
    assert "lone" not in {item.evidence_id for item in run.selection.items}


class _TwoChunkExtractor:
    """d1 and d2 are two passages of ONE article; d3 is a genuinely different source."""

    def generate(self, prompt: str) -> str:
        if "Passage: P_A1" in prompt or "Passage: P_A2" in prompt:
            return "Nixon"
        if "Passage: P_B" in prompt:
            return "Kennedy"
        return "NONE"


def _two_chunk_pool() -> tuple[Query, CandidateSet]:
    def candidate(evidence_id: str, document_id: str, text: str, rank: int) -> EvidenceCandidate:
        return EvidenceCandidate(
            evidence_id=evidence_id,
            document_id=document_id,
            chunk_id=f"{document_id}::c0",
            text=text,
            source_uri=f"s://{document_id}",
            retrieval_score=1.0 / rank,
            retrieval_rank=rank,
        )

    query = Query(query_id="q1", text="who won?")
    candidates = CandidateSet(
        query_id="q1",
        candidates=(
            candidate("a1", "d1", "P_A1", 1),
            candidate("a2", "d2", "P_A2", 2),
            candidate("b", "d3", "P_B", 3),
        ),
    )
    return query, candidates


def test_document_unit_lets_two_chunks_of_one_article_out_vote_a_real_source() -> None:
    """Pins the defect: under the document unit d1+d2 are one article but cast two votes,
    out-voting the genuinely independent d3 by the margin and dropping it."""
    query, candidates = _two_chunk_pool()
    selector = GatedCorroborationSelector(
        _TwoChunkExtractor(), margin=1, support_cap=1, use_parametric=False
    )
    result = selector.select(query, candidates, max_selected=3)
    assert "b" not in {item.evidence_id for item in result.items}


def test_parent_unit_stops_the_same_article_from_out_voting_it() -> None:
    query, candidates = _two_chunk_pool()
    selector = GatedCorroborationSelector(
        _TwoChunkExtractor(),
        margin=1,
        support_cap=1,
        use_parametric=False,
        parent_by_document={"d1": "page a", "d2": "page a", "d3": "page b"},
    )
    result = selector.select(query, candidates, max_selected=3)
    assert "b" in {item.evidence_id for item in result.items}
