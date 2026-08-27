"""What the structure-aware chunker must do that ``WordChunker`` demonstrably does not.

The chunk is the unit of evidence for all three modules, and a fixed word window cuts
through whatever happens to be at word 120. These tests pin the three cases where that is
not a cosmetic difference: a table cut in half, a heading severed from its section, and a
paragraph cut mid-sentence when a boundary was available a few words away.
"""

import pytest
from pydantic import ValidationError

from evidence_rag.contracts.models import Document
from evidence_rag.infrastructure.config import ChunkerConfig
from evidence_rag.infrastructure.corpus import (
    CorpusBuilder,
    SectionChunker,
    WordChunker,
    build_chunker,
)

TABLE = "\n".join(
    ["| quarter | revenue |", "| --- | --- |"] + [f"| Q{index} | {index}00 |" for index in range(1, 9)]
)

REPORT = Document(
    document_id="report",
    text=(
        "# Annual report\n\n"
        + " ".join(f"intro{index}" for index in range(40))
        + "\n\n## Revenue by quarter\n\n"
        + TABLE
        + "\n\n## Outlook\n\n"
        + " ".join(f"outlook{index}" for index in range(30))
    ),
    source_uri="fixture://report",
)


def chunks_of(document: Document, **kwargs: int) -> tuple[str, ...]:
    return tuple(chunk.text for chunk in SectionChunker(**kwargs).chunk(document))


def test_a_table_is_never_split_across_chunks() -> None:
    # Half a table is not evidence: the rows that keep the header answer the question and
    # the rows that lose it cannot be read at all.
    texts = chunks_of(REPORT, chunk_size=30, overlap=5)
    holding_table = [text for text in texts if "| Q1 |" in text]
    assert len(holding_table) == 1
    for row in range(1, 9):
        assert f"| Q{row} |" in holding_table[0]
    assert "| quarter | revenue |" in holding_table[0]


def test_an_oversized_table_stays_whole_rather_than_being_cut() -> None:
    # Deliberate trade, documented on the class: a table longer than chunk_size becomes one
    # oversized chunk, because rows without their header are worse than a long chunk.
    texts = chunks_of(REPORT, chunk_size=12, overlap=2)
    holding_table = [text for text in texts if "| Q1 |" in text]
    assert len(holding_table) == 1
    assert len(holding_table[0].split()) > 12


def test_a_heading_stays_with_the_section_it_titles() -> None:
    texts = chunks_of(REPORT, chunk_size=30, overlap=5)
    for heading, body in (("## Revenue by quarter", "| Q1 |"), ("## Outlook", "outlook0")):
        owning = [text for text in texts if body in text]
        assert owning, f"no chunk contains {body}"
        assert heading in owning[0], f"{heading} was severed from its section"


def test_no_chunk_is_a_bare_heading() -> None:
    # The first version of this chunker emitted one. A heading alone is worthless as
    # evidence and leaves the section it titles anonymous, so a pending heading must never
    # trigger a split — it rides along even when that overflows chunk_size.
    for size in (10, 12, 20, 30, 60):
        for text in chunks_of(REPORT, chunk_size=size, overlap=2):
            lines = [line for line in text.strip().splitlines() if line.strip()]
            assert not (len(lines) == 1 and lines[0].lstrip().startswith("#")), (
                f"chunk_size={size} produced a chunk that is only a heading: {text!r}"
            )


def test_a_heading_is_carried_into_an_oversized_section_it_titles() -> None:
    # Falling back to the window must not orphan the title either.
    document = Document(
        document_id="long-section",
        text="## Methodology\n\n" + " ".join(f"step{index}" for index in range(100)),
        source_uri="fixture://long-section",
    )
    texts = chunks_of(document, chunk_size=30, overlap=5)
    assert len(texts) > 1
    assert "## Methodology" in texts[0]
    # ...and only into the first piece: repeating it would inflate every chunk's term
    # counts with the same words.
    assert all("## Methodology" not in text for text in texts[1:])


def test_word_chunker_really_does_split_the_table_so_this_is_not_a_strawman() -> None:
    # The comparison this whole chunker is justified by. If WordChunker ever stops cutting
    # the table, the justification weakens and this test says so.
    word_texts = [chunk.text for chunk in WordChunker(chunk_size=30, overlap=5).chunk(REPORT)]
    holding_rows = [text for text in word_texts if "| Q1 |" in text or "| Q8 |" in text]
    assert len(holding_rows) > 1, "expected WordChunker to spread the table over chunks"


def test_prose_without_any_markup_is_cut_on_paragraphs_not_mid_paragraph() -> None:
    # SciFact/2Wiki look like this: no headings, no tables. The chunker must still improve
    # on a blind window by preferring the paragraph boundary that is already there.
    document = Document(
        document_id="prose",
        text=(
            " ".join(f"first{index}" for index in range(20))
            + "\n\n"
            + " ".join(f"second{index}" for index in range(20))
        ),
        source_uri="fixture://prose",
    )
    texts = chunks_of(document, chunk_size=25, overlap=5)
    assert len(texts) == 2
    assert "second0" not in texts[0]
    assert "first19" not in texts[1]


def test_a_single_paragraph_longer_than_chunk_size_falls_back_to_the_window() -> None:
    # Nothing structural is left to cut on, so refusing to split would produce one chunk
    # the size of the document — the failure mode this chunker must not have.
    document = Document(
        document_id="long",
        text=" ".join(f"word{index}" for index in range(100)),
        source_uri="fixture://long",
    )
    texts = chunks_of(document, chunk_size=30, overlap=5)
    assert len(texts) > 1
    assert all(len(text.split()) <= 30 for text in texts)
    # Overlap applies here, and only here.
    assert texts[0].split()[-5:] == texts[1].split()[:5]


def test_chunking_is_deterministic_and_ids_are_stable() -> None:
    first = SectionChunker(chunk_size=30, overlap=5).chunk(REPORT)
    second = SectionChunker(chunk_size=30, overlap=5).chunk(REPORT)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert len({chunk.evidence_id for chunk in first}) == len(first)


def test_section_chunking_gives_a_different_corpus_signature_from_word_chunking() -> None:
    # Otherwise a section-chunked run could reuse a word-chunked index and be scored
    # against chunks it never produced, with nothing to reveal it.
    documents = (REPORT,)
    word = CorpusBuilder(WordChunker(chunk_size=30, overlap=5)).build(documents, "sig")
    section = CorpusBuilder(SectionChunker(chunk_size=30, overlap=5)).build(documents, "sig")
    assert word.manifest.corpus_signature != section.manifest.corpus_signature
    assert section.manifest.chunker_version == "section-v1"
    assert (section.manifest.chunk_size, section.manifest.overlap) == (30, 5)


def test_section_is_selectable_from_a_config() -> None:
    config = ChunkerConfig(name="section", chunk_size=200, overlap=40)
    chunker = build_chunker(config.name, chunk_size=config.chunk_size, overlap=config.overlap)
    assert isinstance(chunker, SectionChunker)
    assert (chunker.chunk_size, chunker.overlap) == (200, 40)


def test_window_settings_are_still_required_to_be_coherent() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        ChunkerConfig(name="section", chunk_size=50, overlap=50)
    with pytest.raises(ValueError, match="overlap"):
        SectionChunker(chunk_size=50, overlap=50)
