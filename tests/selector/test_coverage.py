from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.selector.coverage import coverage_select, salient_features


def test_numbers_are_canonicalized_features() -> None:
    feats = salient_features("Revenue was $1.2B in 2025.")
    assert "num:1200000000" in feats
    assert "num:2025" in feats


def test_number_aliases_collapse_to_same_feature() -> None:
    assert salient_features("1,200 million") == salient_features("$1.2B")


def test_content_words_become_features_stopwords_excluded() -> None:
    feats = salient_features("The operating margin of IBM")
    assert "tok:operating" in feats
    assert "tok:margin" in feats
    assert "tok:ibm" in feats
    assert "tok:the" not in feats


def test_pure_digits_do_not_duplicate_as_tokens() -> None:
    feats = salient_features("18%")
    assert feats == frozenset({"num:18"})


def test_features_are_deterministic() -> None:
    text = "IBM 2025 operating margin was 18 percent"
    assert salient_features(text) == salient_features(text)


def cand(
    evidence_id: str,
    text: str,
    rank: int,
    document_id: str | None = None,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id or f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=rank,
    )


def test_answer_cluster_representative_is_pinned() -> None:
    survivors = (
        cand("ans", "IBM 2025 operating margin was 18 percent", 1),
        cand("noise", "Unrelated weather report about rain", 2),
    )
    out = coverage_select(
        survivors,
        blended_by_id={"ans": 0.4, "noise": 0.9},
        answer_by_id={"ans": "18", "noise": None},
        query_text="IBM 2025 operating margin",
        max_selected=1,
    )
    assert tuple(c.evidence_id for c in out) == ("ans",)


def test_complementary_support_beats_irrelevant_after_pin() -> None:
    survivors = (
        cand("ans", "IBM 2025 operating margin was 18 percent", 1),
        cand("comp", "IBM 2025 revenue 1.2B and operating income", 2),
        cand("noise", "A recipe for chocolate cake", 3),
    )
    out = coverage_select(
        survivors,
        blended_by_id={"ans": 0.9, "comp": 0.5, "noise": 0.6},
        answer_by_id={"ans": "18", "comp": None, "noise": None},
        query_text="IBM 2025 operating margin",
        max_selected=2,
    )
    ids = tuple(c.evidence_id for c in out)
    assert ids[0] == "ans"
    assert "comp" in ids and "noise" not in ids


def test_redundant_duplicate_is_demoted() -> None:
    survivors = (
        cand("ans", "IBM 2025 operating margin was 18 percent", 1),
        cand("dup", "IBM 2025 operating margin was 18 percent", 2, document_id="doc-ans"),
        cand("comp", "IBM 2025 revenue 1.2B operating income detail", 3),
    )
    out = coverage_select(
        survivors,
        blended_by_id={"ans": 0.9, "dup": 0.8, "comp": 0.1},
        answer_by_id={"ans": "18", "dup": "18", "comp": None},
        query_text="IBM 2025 operating margin",
        max_selected=2,
    )
    ids = tuple(c.evidence_id for c in out)
    assert "ans" in ids and "comp" in ids and "dup" not in ids


def test_two_answer_clusters_each_get_a_pinned_rep() -> None:
    survivors = (
        cand("a", "IBM margin was 18 percent", 1),
        cand("b", "IBM margin was 16 percent", 2),
        cand("c", "IBM margin context filler", 3),
    )
    out = coverage_select(
        survivors,
        blended_by_id={"a": 0.9, "b": 0.8, "c": 0.7},
        answer_by_id={"a": "18", "b": "16", "c": None},
        query_text="IBM margin",
        max_selected=2,
    )
    ids = {c.evidence_id for c in out}
    assert ids == {"a", "b"}


def test_shortfall_not_padded_and_deterministic() -> None:
    survivors = (cand("a", "IBM margin 18 percent", 1),)
    kwargs = dict(
        blended_by_id={"a": 0.5},
        answer_by_id={"a": "18"},
        query_text="IBM margin",
        max_selected=5,
    )
    out1 = coverage_select(survivors, **kwargs)  # type: ignore[arg-type]
    out2 = coverage_select(survivors, **kwargs)  # type: ignore[arg-type]
    assert tuple(c.evidence_id for c in out1) == ("a",)
    assert out1 == out2


def test_empty_universe_falls_back_to_blended_after_pin() -> None:
    survivors = (
        cand("x", "zzz", 1),
        cand("y", "qqq", 2),
    )
    out = coverage_select(
        survivors,
        blended_by_id={"x": 0.2, "y": 0.9},
        answer_by_id={"x": None, "y": None},
        query_text="nothing matches",
        max_selected=2,
    )
    assert tuple(c.evidence_id for c in out) == ("y", "x")
