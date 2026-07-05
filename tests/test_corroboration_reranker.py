"""Tests for CorroborationReranker (LLM answer-extraction + corroboration blend)."""
from src.retrieval.base import RetrievedChunk
from src.retrieval.reranker import CorroborationReranker


class FakeAnswerLLM:
    """Extracts an answer per passage via a text->answer map; parametric via a fixed
    reply. The extraction prompt contains 'Passage:'; the parametric prompt does not."""

    def __init__(self, answer_by_text, parametric="NONE"):
        self.answer_by_text = answer_by_text
        self.parametric = parametric

    def generate(self, prompt: str) -> str:
        if "Passage:" not in prompt:            # parametric elicitation
            return self.parametric
        for text, ans in self.answer_by_text.items():
            if text in prompt:
                return ans
        return "NONE"


def test_corroborated_needle_outranks_lone_counterfactual():
    # The counterfactual has the HIGHEST relevance but a lone (wrong) answer; the needle
    # and another gold share the correct answer, so corroboration lifts them above it.
    cand = [
        RetrievedChunk("cf", "counterfactual passage", 0.99),   # answer Berlin (lone)
        RetrievedChunk("needle", "needle passage", 0.90),       # answer Paris
        RetrievedChunk("gold2", "second gold passage", 0.80),   # answer Paris (corroborates)
    ]
    llm = FakeAnswerLLM({
        "counterfactual passage": "Berlin",
        "needle passage": "Paris",
        "second gold passage": "Paris",
    })
    rr = CorroborationReranker(llm, top_n=3, alpha=0.5, use_parametric=False)

    out = [c.doc_id for c in rr.rerank("capital of France?", cand, top_k=3)]

    assert out[0] in {"needle", "gold2"}         # a corroborated (correct) passage leads
    assert out.index("needle") < out.index("cf")  # needle beat its counterfactual twin


def test_alpha_one_is_pure_first_stage_order():
    cand = [RetrievedChunk("a", "alpha passage", 0.9), RetrievedChunk("b", "bravo passage", 0.5)]
    llm = FakeAnswerLLM({"alpha passage": "Paris", "bravo passage": "Paris"})
    rr = CorroborationReranker(llm, top_n=2, alpha=1.0, use_parametric=False)

    out = [c.doc_id for c in rr.rerank("q", cand, top_k=2)]

    assert out == ["a", "b"]                     # alpha=1 ignores corroboration entirely


def test_top_n_caps_extraction_and_keeps_tail_below():
    cand = [
        RetrievedChunk("a", "alpha passage", 0.9),
        RetrievedChunk("b", "bravo passage", 0.8),
        RetrievedChunk("c", "charlie passage", 0.7),   # beyond top_n -> tail
    ]
    llm = FakeAnswerLLM({"alpha passage": "Paris", "bravo passage": "NONE"})
    rr = CorroborationReranker(llm, top_n=2, alpha=0.0, use_parametric=False)

    out = [c.doc_id for c in rr.rerank("q", cand, top_k=3)]

    assert out[-1] == "c"                        # the un-scored tail stays below the window


def test_empty_pool_returns_empty():
    rr = CorroborationReranker(FakeAnswerLLM({}), top_n=5)
    assert rr.rerank("q", [], top_k=3) == []


def test_score_docs_returns_relevance_and_corroboration_aligned():
    # The raw signals the reranker blends (and the lambda-sweep dumps once): relevance =
    # first-stage scores; corroboration = cross-source answer votes over the docs.
    docs = [
        RetrievedChunk("a", "alpha passage", 0.9),
        RetrievedChunk("b", "bravo passage", 0.5),
    ]
    llm = FakeAnswerLLM({"alpha passage": "Paris", "bravo passage": "Paris"})
    rr = CorroborationReranker(llm, top_n=2, use_parametric=False)

    relevance, corroboration = rr.score_docs("q", docs)

    assert relevance == [0.9, 0.5]        # first-stage relevance, aligned to docs
    assert corroboration == [1.0, 1.0]    # both answer Paris -> corroborate each other
