"""Placeholder tests for skeleton functions that are not implemented yet.

Each test documents the *intended* behaviour and is marked ``xfail`` because
the implementation currently raises ``NotImplementedError``. As each function
is built out, remove the ``xfail`` marker and replace the body with a real
assertion. ``--strict-markers`` + ``xfail_strict`` would flip these to failures
the moment they unexpectedly pass, signalling "this is done now".
"""

from __future__ import annotations

import pytest

from src.data_processing import Needle, inject_needle
from src.ingestion.chunker import chunk_document


@pytest.mark.xfail(reason="inject_needle not implemented yet", strict=True)
def test_inject_needle_places_text_in_haystack() -> None:
    needle = Needle(text="NEEDLE", question="q", answer="a")
    result = inject_needle("haystack " * 100, needle, depth_percent=50)
    assert "NEEDLE" in result


def test_chunk_document_returns_non_empty_list() -> None:
    chunks = chunk_document("doc1", "word " * 1000, chunk_size=512, chunk_overlap=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 1
