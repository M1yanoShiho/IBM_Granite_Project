"""Pin the exact ALCE citation-metric semantics.

These tests exist because the precision definition involves a redundancy
ablation that is easy to implement subtly wrong, and everything downstream in the
G3 remediation depends on it being right. Each test names the reference behaviour
it pins.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from alce_metrics import (  # noqa: E402
    CachedEntailment,
    ScoredExample,
    compute_citation_metrics,
)


def entails(premise: str, hypothesis: str) -> bool:
    """Toy judge: the premise entails the hypothesis iff it contains all its tokens."""
    return set(hypothesis.replace(".", "").split()) <= set(premise.replace(".", "").split())


def example(sentences, citations, docs, example_id="ex") -> ScoredExample:
    return ScoredExample(
        example_id=example_id,
        sentences=tuple(sentences),
        citations=tuple(tuple(c) for c in citations),
        docs=docs,
    )


def test_single_citation_is_precise_iff_it_entails() -> None:
    docs = {"d1": "x"}
    supported = example(["x"], [["d1"]], docs)
    unsupported = example(["y"], [["d1"]], docs)

    assert compute_citation_metrics([supported], entails).citation_prec == 100.0
    assert compute_citation_metrics([supported], entails).citation_rec == 100.0
    assert compute_citation_metrics([unsupported], entails).citation_prec == 0.0
    assert compute_citation_metrics([unsupported], entails).citation_rec == 0.0


def test_uncited_sentence_scores_zero_recall_and_adds_no_citations() -> None:
    # ALCE: len(ref) == 0 -> joint_entail = 0, and total_citations is NOT incremented
    docs = {"d1": "x"}
    ex = example(["x", "y"], [["d1"], []], docs)

    report = compute_citation_metrics([ex], entails)

    assert report.citation_rec == 50.0  # 1 of 2 sentences entailed
    assert report.citation_prec == 100.0  # 1 precise citation of 1 total
    assert report.n_citations == 1


def test_example_with_no_citations_at_all_scores_zero_precision_not_skipped() -> None:
    # ALCE: entail_prec / total_citations if total_citations > 0 else 0
    ex = example(["x"], [[]], {"d1": "x"})

    report = compute_citation_metrics([ex], entails)

    assert report.n_examples == 1  # counted, not skipped
    assert report.citation_prec == 0.0
    assert report.citation_rec == 0.0


def test_unknown_citation_id_is_treated_as_out_of_range() -> None:
    # ALCE: any(ref_id >= len(docs)) -> joint_entail = 0 and no citations counted
    ex = example(["x"], [["missing"]], {"d1": "x"})

    report = compute_citation_metrics([ex], entails)

    assert report.citation_rec == 0.0
    assert report.n_citations == 0
    assert report.citation_prec == 0.0


def test_multi_citation_both_necessary_are_both_precise() -> None:
    # joint entails; neither alone entails; removing either breaks entailment
    docs = {"d1": "x", "d2": "y"}
    ex = example(["x y"], [["d1", "d2"]], docs)

    report = compute_citation_metrics([ex], entails)

    assert report.citation_rec == 100.0
    assert report.citation_prec == 100.0  # 2 precise of 2
    assert report.sent_mcite == 1
    assert report.sent_mcite_support == 1
    assert report.sent_mcite_overcite == 0


def test_multi_citation_redundant_one_is_counted_as_overcite() -> None:
    # d1 alone entails; d2 does not, and dropping d2 still entails -> d2 is redundant
    docs = {"d1": "x", "d2": "z"}
    ex = example(["x"], [["d1", "d2"]], docs)

    report = compute_citation_metrics([ex], entails)

    assert report.citation_rec == 100.0
    assert report.citation_prec == 50.0  # 1 precise of 2 citations
    assert report.sent_mcite_overcite == 1


def test_redundancy_ablation_does_not_run_for_a_lone_citation() -> None:
    # a single citation that does not entail is simply imprecise; no ablation
    docs = {"d1": "z"}
    ex = example(["x"], [["d1"]], docs)

    report = compute_citation_metrics([ex], entails)

    assert report.citation_prec == 0.0
    assert report.sent_mcite == 0
    assert report.sent_mcite_overcite == 0


def test_metrics_are_macro_averaged_over_examples_not_pooled() -> None:
    docs = {"d1": "x", "d2": "z"}
    good = example(["x"], [["d1"]], docs, example_id="good")  # prec 1.0 over 1 citation
    bad = example(["x x x"], [["d2", "d2", "d2"]], {"d2": "z"}, example_id="bad")

    report = compute_citation_metrics([good, bad], entails)

    # macro: (1.0 + 0.0) / 2 = 50.0 ; pooled over citations would be 1/4 = 25.0
    assert report.citation_prec == 50.0


def test_longer_unsupported_answers_lose_recall() -> None:
    """The property that makes sentence-level scoring length-fair: padding an
    answer with uncited prose costs recall, instead of silently diluting every
    citation the way answer-level scoring does."""
    docs = {"d1": "x"}
    short = example(["x"], [["d1"]], docs, example_id="short")
    padded = example(["x", "unrelated", "filler"], [["d1"], [], []], docs, example_id="padded")

    assert compute_citation_metrics([short], entails).citation_rec == 100.0
    assert compute_citation_metrics([padded], entails).citation_rec == pytest.approx(33.333, abs=0.01)
    # but the citation itself stays fully precise -- dilution no longer punishes it
    assert compute_citation_metrics([padded], entails).citation_prec == 100.0


def test_cache_collapses_repeated_judgements() -> None:
    calls = {"n": 0}

    def counting(premise: str, hypothesis: str) -> bool:
        calls["n"] += 1
        return entails(premise, hypothesis)

    cached = CachedEntailment(counting)
    cached("x", "x")
    cached("x", "x")

    assert calls["n"] == 1
    assert cached.calls == 1


def test_sentence_and_citation_lists_must_line_up() -> None:
    with pytest.raises(ValueError, match="one citation list"):
        ScoredExample(example_id="e", sentences=("a", "b"), citations=(("d1",),), docs={})
