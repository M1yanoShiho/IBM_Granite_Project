"""Measure what a built BM25 index *retains*, using the instrument that can see it.

This gap has been open twice. R5 estimated ~2.9 KB/chunk for the cached forward index
from `sys.getsizeof`, then withdrew the estimate because peak RSS showed no rise at all
on the real corpus — twice. The diagnosis recorded at the time was that peak RSS is the
wrong instrument, being dominated by build-phase transients, and that settling it needs
steady-state sampling. R9 (the inverted index) could then only repeat "memory is
unmeasured", and its claim that postings are a *transpose* of the per-chunk counters
rather than an addition to them is so far an argument, not a number.

So this measures retained bytes, not peak: `tracemalloc` snapshots either side of the
build, with `gc.collect()` before the second one, so transients that have been freed do
not count and only what the retriever still holds does.

Three quantities are separated because they answer different questions:

- **corpus** — the `CorpusSnapshot` itself (chunk objects and their text). Paid by every
  retriever, so it is the baseline the index cost should be read against rather than
  folded into.
- **index** — everything `BM25Retriever._set_chunks` retains on top of that corpus.
  This is the number R5 wanted and the one that extrapolates.
- **per chunk** — index bytes / chunk count, reported at several corpus sizes so the
  linearity is checked rather than assumed. R5's extrapolation to a million chunks is
  only meaningful if the per-chunk cost is flat.

**Use real text.** The synthetic generator is a fallback, and a misleading one: postings cost
scales with *distinct terms per chunk*, and a hand-tuned Pareto draw gets that badly wrong. The
default generator yields ~13.6 distinct terms per chunk against ~95 for real prose, which
understated the inverted index's memory by about 7× and overstated its saving over the forward
index by about 4× the first time this was run. `--documents DIR` measures a directory of
.md/.txt; `--manifest` measures a benchmark.

Usage (CPU only):

    PYTHONPATH=src python scripts/retriever_memory.py --sizes 2000 4000 8000
    PYTHONPATH=src python scripts/retriever_memory.py \
        --manifest data/benchmarks/scifact/test/manifest.json --sizes 5183
"""

import argparse
import gc
import json
import platform
import random
import sys
import tracemalloc
from pathlib import Path
from typing import Any

from evidence_rag.contracts.models import Document
from evidence_rag.infrastructure.corpus import CorpusBuilder, CorpusSnapshot, WordChunker
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.retriever.bm25 import BM25Retriever

VOCABULARY_SIZE = 20000


def _synthetic_documents(count: int, *, seed: int) -> tuple[Document, ...]:
    """Zipf-like word draw, so document frequencies span the range postings care about."""
    vocabulary = [f"term{index}" for index in range(VOCABULARY_SIZE)]
    rng = random.Random(seed)
    documents = []
    for index in range(count):
        words = [
            vocabulary[min(int(rng.paretovariate(1.2)) - 1, VOCABULARY_SIZE - 1)]
            for _ in range(200)
        ]
        documents.append(
            Document(
                document_id=f"doc-{index}",
                text=" ".join(words),
                source_uri=f"synthetic://doc-{index}",
            )
        )
    return tuple(documents)


def _retained(build: Any) -> tuple[Any, int]:
    """Run ``build`` and return its result plus the bytes it still holds afterwards.

    ``gc.collect()`` before the second snapshot is the whole point: without it the
    measurement includes garbage that merely has not been reclaimed yet, which is how a
    transient gets mistaken for a retained cost.
    """
    gc.collect()
    before = tracemalloc.take_snapshot()
    result = build()
    gc.collect()
    after = tracemalloc.take_snapshot()
    retained = sum(entry.size_diff for entry in after.compare_to(before, "filename"))
    return result, retained


def _measure(documents: tuple[Document, ...], *, chunk_size: int, overlap: int) -> dict[str, Any]:
    corpus: CorpusSnapshot
    corpus, corpus_bytes = _retained(
        lambda: CorpusBuilder(WordChunker(chunk_size=chunk_size, overlap=overlap)).build(
            documents, "memory-probe"
        )
    )
    retriever, index_bytes = _retained(lambda: BM25Retriever.from_corpus(corpus))
    chunk_count = len(corpus.chunks)
    # Referenced so the retriever cannot be collected before the snapshot is taken.
    assert len(retriever.chunks) == chunk_count
    return {
        "documents": len(documents),
        "chunks": chunk_count,
        "corpus_bytes": corpus_bytes,
        "index_bytes": index_bytes,
        "index_bytes_per_chunk": index_bytes / chunk_count if chunk_count else 0.0,
        "index_over_corpus": index_bytes / corpus_bytes if corpus_bytes else 0.0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="measure a real corpus instead of synthetic")
    parser.add_argument(
        "--documents",
        type=Path,
        help="measure a directory of .md/.txt files -- the most realistic option available "
        "without a benchmark, and the one that matters: postings cost scales with DISTINCT "
        "terms per chunk, which a synthetic vocabulary gets wrong by several fold",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[2000, 4000, 8000])
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.manifest:
        source = JsonlDatasetAdapter.load(arguments.manifest).documents
    elif arguments.documents:
        source = tuple(
            Document(
                document_id=str(path.relative_to(arguments.documents)).replace("\\", "/"),
                text=text,
                source_uri=str(path),
            )
            for path in sorted(arguments.documents.rglob("*"))
            if path.suffix.lower() in {".md", ".txt"}
            and path.is_file()
            and (text := path.read_text(encoding="utf-8", errors="replace")).strip()
        )
    else:
        source = None

    tracemalloc.start()
    rows = []
    for size in arguments.sizes:
        documents = (
            tuple(source[:size])
            if source is not None
            else _synthetic_documents(size, seed=arguments.seed)
        )
        rows.append(
            _measure(documents, chunk_size=arguments.chunk_size, overlap=arguments.overlap)
        )
    tracemalloc.stop()

    report = {
        "corpus": str(arguments.manifest or arguments.documents or "synthetic"),
        "chunk_size": arguments.chunk_size,
        "overlap": arguments.overlap,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "measurements": rows,
    }
    print(f"{'chunks':>8} {'index MB':>9} {'B/chunk':>9} {'index/corpus':>13}")
    for row in rows:
        print(
            f"{row['chunks']:8d} {row['index_bytes'] / 1e6:9.2f} "
            f"{row['index_bytes_per_chunk']:9.0f} {row['index_over_corpus']:12.2f}x"
        )
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
