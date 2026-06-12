"""Unit tests for the project's data structures and configuration defaults.

These cover the parts of the codebase that are already implemented (the
dataclasses and their defaults), giving a green baseline that the CI / Bristol
marker can run today. As the skeleton functions in ``src`` and ``eval`` are
filled in, the ``xfail`` tests at the bottom should be turned into real
assertions.
"""

from __future__ import annotations

from src.data_processing import HaystackSample, Needle
from src.llm_client import GenerationConfig
from src.rag_pipeline import RAGResult
from src.retrieval.base import RetrievedChunk
from eval.niah_runner import NIAHConfig


# --------------------------------------------------------------------------- #
# Needle
# --------------------------------------------------------------------------- #
def test_needle_stores_fields() -> None:
    needle = Needle(
        text="The secret code is 42.",
        question="What is the secret code?",
        answer="42",
    )
    assert needle.text == "The secret code is 42."
    assert needle.question == "What is the secret code?"
    assert needle.answer == "42"


# --------------------------------------------------------------------------- #
# HaystackSample
# --------------------------------------------------------------------------- #
def test_haystack_sample_defaults_metadata_to_empty_dict() -> None:
    needle = Needle(text="x", question="q", answer="a")
    sample = HaystackSample(
        context="...needle...",
        needle=needle,
        context_length=4000,
        depth_percent=50.0,
    )
    assert sample.context_length == 4000
    assert sample.depth_percent == 50.0
    assert sample.metadata == {}


def test_haystack_sample_metadata_is_independent_per_instance() -> None:
    """Guards against a mutable-default-argument bug."""
    needle = Needle(text="x", question="q", answer="a")
    a = HaystackSample(context="a", needle=needle, context_length=1000, depth_percent=0)
    b = HaystackSample(context="b", needle=needle, context_length=1000, depth_percent=0)
    a.metadata["k"] = "v"
    assert b.metadata == {}


# --------------------------------------------------------------------------- #
# GenerationConfig
# --------------------------------------------------------------------------- #
def test_generation_config_defaults_are_deterministic() -> None:
    cfg = GenerationConfig()
    # Temperature 0.0 matters: retrieval evaluation must be deterministic.
    assert cfg.temperature == 0.0
    assert cfg.max_new_tokens == 256
    assert cfg.top_p == 1.0


# --------------------------------------------------------------------------- #
# RAGResult
# --------------------------------------------------------------------------- #
def test_rag_result_stores_answer_and_chunks() -> None:
    chunks = [
        RetrievedChunk(doc_id="c1", text="Bristol is a city.", score=0.9),
        RetrievedChunk(doc_id="c2", text="It is in the UK.", score=0.7),
    ]
    result = RAGResult(answer="Bristol", retrieved_chunks=chunks)
    assert result.answer == "Bristol"
    assert [c.doc_id for c in result.retrieved_chunks] == ["c1", "c2"]


# --------------------------------------------------------------------------- #
# NIAHConfig
# --------------------------------------------------------------------------- #
def test_niah_config_grid_matches_proposal() -> None:
    cfg = NIAHConfig()
    assert cfg.context_lengths == [1000, 2000, 4000, 8000, 16000]
    assert cfg.depth_percents == [0, 10, 25, 50, 75, 90, 100]
    # The proposal claims a 5 x 7 = 35-cell evaluation grid.
    assert len(cfg.context_lengths) * len(cfg.depth_percents) == 35
