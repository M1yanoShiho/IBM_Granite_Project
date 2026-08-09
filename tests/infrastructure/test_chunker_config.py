"""Pins the two properties that make chunking safe to vary from a config file.

Chunk size and overlap were fixed at ``WordChunker``'s defaults and unreachable from an
experiment config, so they had never been swept even though the chunk is the unit of
evidence for all three modules. Making them configurable is only safe if two things
hold, and both are easy to break silently:

1. A config with no ``[chunker]`` table must build the *identical* corpus, byte for
   byte, as before. Every recorded SciFact/NQ/2Wiki number depends on it.
2. Two different settings must produce different ``corpus_signature``s, so a sweep
   cannot reuse another point's index. The signature already folds the chunker in;
   this test is here so a later refactor cannot quietly drop it.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_rag.contracts.models import Document
from evidence_rag.infrastructure.config import ChunkerConfig, load_experiment_config
from evidence_rag.infrastructure.corpus import CorpusBuilder, WordChunker

DOCUMENTS = (
    Document(
        document_id="doc-1",
        text=" ".join(f"word{index}" for index in range(500)),
        source_uri="fixture://doc-1",
    ),
    Document(
        document_id="doc-2",
        text=" ".join(f"term{index}" for index in range(300)),
        source_uri="fixture://doc-2",
    ),
)


def build(chunk_size: int, overlap: int):
    return CorpusBuilder(WordChunker(chunk_size=chunk_size, overlap=overlap)).build(
        DOCUMENTS, "fixture-signature"
    )


def test_defaults_match_wordchunker_so_existing_configs_are_unaffected() -> None:
    config = ChunkerConfig()
    assert (config.chunk_size, config.overlap) == (120, 20)
    reference = WordChunker()
    assert (config.chunk_size, config.overlap) == (reference.chunk_size, reference.overlap)


def test_a_config_without_a_chunker_table_builds_the_identical_corpus(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "no-chunker.toml"
    config_path.write_text(
        '[dataset]\nmanifest = "m.json"\n\n'
        '[output]\ndirectory = "./out"\n\n'
        '[retriever]\nname = "bm25"\n\n'
        '[selector]\nname = "top-k"\n\n'
        '[generator]\nname = "extractive"\n\n'
        "[run]\ntop_k = 5\nmax_selected = 2\nseed = 7\n",
        encoding="utf-8",
    )
    (tmp_path / "m.json").write_text("{}", encoding="utf-8")

    config = load_experiment_config(config_path)
    assert (config.chunker.chunk_size, config.chunker.overlap) == (120, 20)

    from_config = build(config.chunker.chunk_size, config.chunker.overlap)
    before = CorpusBuilder().build(DOCUMENTS, "fixture-signature")
    assert from_config.manifest.corpus_signature == before.manifest.corpus_signature
    assert from_config.chunks == before.chunks


def test_a_chunker_table_is_read_and_reaches_the_corpus(tmp_path: Path) -> None:
    config_path = tmp_path / "with-chunker.toml"
    config_path.write_text(
        '[dataset]\nmanifest = "m.json"\n\n'
        '[output]\ndirectory = "./out"\n\n'
        '[retriever]\nname = "bm25"\n\n'
        '[selector]\nname = "top-k"\n\n'
        '[generator]\nname = "extractive"\n\n'
        "[chunker]\nchunk_size = 180\noverlap = 30\n\n"
        "[run]\ntop_k = 5\nmax_selected = 2\nseed = 7\n",
        encoding="utf-8",
    )
    (tmp_path / "m.json").write_text("{}", encoding="utf-8")

    config = load_experiment_config(config_path)
    assert (config.chunker.chunk_size, config.chunker.overlap) == (180, 30)
    corpus = build(config.chunker.chunk_size, config.chunker.overlap)
    assert corpus.manifest.chunk_size == 180
    assert corpus.manifest.overlap == 30


def test_different_settings_give_different_corpus_signatures() -> None:
    # Without this, two sweep points could share a persisted index and the second would
    # be scored against the first's chunks with nothing to reveal it.
    signatures = {
        (size, overlap): build(size, overlap).manifest.corpus_signature
        for size, overlap in ((120, 20), (180, 30), (180, 0), (60, 20))
    }
    assert len(set(signatures.values())) == len(signatures)


def test_overlap_at_or_above_chunk_size_is_rejected_at_config_load() -> None:
    # WordChunker rejects this too, but only once a job is already running; catching it
    # in the config names the offending file before anything reaches the cluster.
    with pytest.raises(ValidationError, match="overlap"):
        ChunkerConfig(chunk_size=100, overlap=100)
    with pytest.raises(ValidationError, match="overlap"):
        ChunkerConfig(chunk_size=100, overlap=140)


def test_unknown_chunker_name_is_rejected() -> None:
    # Only WordChunker exists today; a typo like "words" must fail loudly rather than
    # silently falling back to the default and producing a corpus nobody asked for.
    with pytest.raises(ValidationError):
        ChunkerConfig(name="words")
