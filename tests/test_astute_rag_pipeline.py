"""Tests for the Astute RAG pipeline (source-aware internal/external consolidation).

``AstuteRAGPipeline`` turns the static retrieve-then-generate flow into the three
steps of Astute RAG (Wang et al., 2024): elicit the model's own internal-knowledge
passage, consolidate it with the retrieved passages *with source awareness* (so a
lone counterfactual document can be cross-checked and down-weighted), then answer
from the consolidated notes. It is named for the family, staying a lightweight,
prompt-only variant that reuses the ONE injected LLM — no fine-tuning.
"""

from __future__ import annotations

from src.rag_pipeline import AstuteRAGPipeline, RAGResult
from src.retrieval.base import RetrievedChunk


class ScriptedRetriever:
    """Returns canned results per query text and records the queries it saw."""

    def __init__(self, responses) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def retrieve(self, query: str):
        self.calls.append(query)
        return self.responses.get(query, [])


class EchoLLM:
    def generate(self, prompt: str) -> str:
        return "ANSWER"


class StepLLM:
    """Routes on the prompt's trailing cue (each Astute step ends differently) and
    records every prompt, so tests can assert the three-step flow behaviourally."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        tail = prompt.rstrip()
        if tail.endswith("Passage:"):
            return "INTERNAL_KNOWLEDGE"
        if tail.endswith("Consolidated notes:"):
            return "CONSOLIDATED_NOTES"
        return "FINAL_ANSWER"


def test_query_returns_ragresult_with_answer_and_retrieved_chunks() -> None:
    retriever = ScriptedRetriever({"q": [RetrievedChunk("d1", "ctx", 0.9)]})
    pipeline = AstuteRAGPipeline(retriever, EchoLLM(), top_k=1)

    result = pipeline.query("q")

    assert isinstance(result, RAGResult)
    assert result.answer == "ANSWER"
    assert [c.doc_id for c in result.retrieved_chunks] == ["d1"]


def test_elicits_internal_knowledge_from_question_without_retrieved_text() -> None:
    # Step 1 of Astute: the model writes its own passage from parametric knowledge,
    # grounded on the question and NOT on the retrieved documents (so it is an
    # independent source to cross-check them against).
    retriever = ScriptedRetriever({"q": [RetrievedChunk("d1", "RETRIEVED_TEXT", 0.9)]})
    llm = StepLLM()
    pipeline = AstuteRAGPipeline(retriever, llm, top_k=1)

    pipeline.query("q")

    elicit = [p for p in llm.prompts if p.rstrip().endswith("Passage:")]
    assert len(elicit) == 1                    # exactly one elicitation
    assert "q" in elicit[0]                     # grounded on the question
    assert "RETRIEVED_TEXT" not in elicit[0]    # from its own knowledge, not the docs


def test_consolidation_sees_both_retrieved_documents_and_internal_knowledge() -> None:
    # Step 2: retrieved passages AND the model's own passage go into one
    # consolidation prompt, so a lone counterfactual document can be checked against
    # the others and the model's parametric knowledge.
    retriever = ScriptedRetriever(
        {"q": [RetrievedChunk("d1", "DOC_ONE_TEXT", 0.9),
               RetrievedChunk("d2", "DOC_TWO_TEXT", 0.5)]}
    )
    llm = StepLLM()
    pipeline = AstuteRAGPipeline(retriever, llm, top_k=2)

    pipeline.query("q")

    consolidate = [p for p in llm.prompts if p.rstrip().endswith("Consolidated notes:")]
    assert len(consolidate) == 1
    prompt = consolidate[0]
    assert "DOC_ONE_TEXT" in prompt and "DOC_TWO_TEXT" in prompt  # retrieved passages
    assert "INTERNAL_KNOWLEDGE" in prompt                         # the model's own passage


def test_consolidation_labels_documents_and_model_as_distinct_sources() -> None:
    # Source-awareness is the crux of Astute: retrieved docs and the model's own
    # knowledge must be tagged differently so the LLM can weigh their reliability.
    retriever = ScriptedRetriever({"q": [RetrievedChunk("d1", "DOC_TEXT", 0.9)]})
    llm = StepLLM()
    pipeline = AstuteRAGPipeline(retriever, llm, top_k=1)

    pipeline.query("q")

    prompt = next(p for p in llm.prompts if p.rstrip().endswith("Consolidated notes:"))
    assert "Document" in prompt   # retrieved-source tag
    assert "Model" in prompt      # internal-knowledge-source tag


def test_final_answer_is_generated_from_the_consolidated_notes() -> None:
    # Step 3: answer from the resolved consolidation, not the raw (possibly
    # conflicting) passages.
    retriever = ScriptedRetriever({"q": [RetrievedChunk("d1", "DOC_TEXT", 0.9)]})
    llm = StepLLM()
    pipeline = AstuteRAGPipeline(retriever, llm, top_k=1)

    result = pipeline.query("q")

    finalize = [p for p in llm.prompts if p.rstrip().endswith("Answer:")]
    assert len(finalize) == 1
    assert "CONSOLIDATED_NOTES" in finalize[0]   # answers from the consolidation...
    assert "DOC_TEXT" not in finalize[0]         # ...not straight from the raw passage
    assert result.answer == "FINAL_ANSWER"


def test_consolidation_can_override_a_lone_counterfactual_document() -> None:
    # The whole point of Astute: a single wrong retrieved document should not decide
    # the answer when the model's own knowledge (and the consolidation step) can
    # cross-check it. Scored end-to-end with an LLM stand-in that trusts the resolved
    # notes over the raw passage.
    class ConflictResolvingLLM:
        def generate(self, prompt: str) -> str:
            tail = prompt.rstrip()
            if tail.endswith("Passage:"):
                return "The capital of France is Paris."          # correct internal knowledge
            if tail.endswith("Consolidated notes:"):
                assert "Berlin" in prompt and "Paris" in prompt   # sees both sources
                return "The capital of France is Paris."          # resolves to the truth
            return "Paris" if "Paris" in prompt else "Berlin"     # finalize follows the notes

    retriever = ScriptedRetriever(
        {"capital of France?": [RetrievedChunk("d1", "The capital of France is Berlin.", 0.9)]}
    )
    pipeline = AstuteRAGPipeline(retriever, ConflictResolvingLLM(), top_k=1)

    result = pipeline.query("capital of France?")

    assert result.answer == "Paris"   # the counterfactual "Berlin" did not win


def test_consolidation_context_is_capped_at_top_k() -> None:
    # Only top_k retrieved passages reach consolidation (and the result), even if the
    # retriever returns more — the same context-depth contract as RAGPipeline.
    retriever = ScriptedRetriever(
        {"q": [RetrievedChunk("d1", "KEEP1", 0.9),
               RetrievedChunk("d2", "KEEP2", 0.8),
               RetrievedChunk("d3", "DROP", 0.1)]}
    )
    llm = StepLLM()
    pipeline = AstuteRAGPipeline(retriever, llm, top_k=2)

    result = pipeline.query("q")

    prompt = next(p for p in llm.prompts if p.rstrip().endswith("Consolidated notes:"))
    assert "KEEP1" in prompt and "KEEP2" in prompt and "DROP" not in prompt
    assert [c.doc_id for c in result.retrieved_chunks] == ["d1", "d2"]
