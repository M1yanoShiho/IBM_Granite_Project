"""Measure how retrieval cost grows with corpus size.

Times two things separately at each corpus size, because they scale for different
reasons and a single number hides which one hurts:

- **index build** — chunking, tokenising and the document-frequency pass
  (``BM25Retriever._set_chunks``), paid once per corpus.
- **per-query retrieval** — reported as mean/p50/p95 so a long tail is visible
  rather than averaged away.

Deliberately standalone. The evaluation layer's metric registry is bound into every
report's ``metric_registry_signature``, so adding a latency metric there would change
that signature for the Selector and Generator stages too; a performance study has no
business doing that. This writes its own JSON instead.

What this does and does not measure: it times *this* Python BM25 implementation, not
BM25 as an algorithm. Conclusions belong to the implementation.

Usage (CPU only, no GPU or LLM needed):

    PYTHONPATH=src python scripts/retriever_scaling.py \
        --manifest data/benchmarks/scifact/test/manifest.json \
        --sizes 500 1000 2500 5183 --queries 50 \
        --output results/retriever-scaling-scifact.json
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from evidence_rag.infrastructure.corpus import CorpusBuilder, WordChunker
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.retriever.strong_bm25 import StrongBM25Retriever


def _peak_rss_mb() -> float | None:
    """Peak resident set size, or None where the platform cannot report it."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows has no resource module
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return peak / 1024 if platform.system() == "Linux" else peak / (1024 * 1024)


def _measure(
    documents: list[Any], queries: list[Any], *, top_k: int, chunk_size: int, overlap: int
) -> dict[str, Any]:
    builder = CorpusBuilder(WordChunker(chunk_size=chunk_size, overlap=overlap))

    build_start = time.perf_counter()
    snapshot = builder.build(documents, dataset_signature="scaling-probe")
    retriever = StrongBM25Retriever.from_corpus(snapshot)
    build_seconds = time.perf_counter() - build_start

    latencies: list[float] = []
    for query in queries:
        query_start = time.perf_counter()
        retriever.retrieve(query, top_k)
        latencies.append((time.perf_counter() - query_start) * 1000.0)

    ordered = sorted(latencies)
    return {
        "documents": len(documents),
        "chunks": len(snapshot.chunks),
        "queries": len(queries),
        "build_seconds": round(build_seconds, 4),
        "ms_per_query_mean": round(statistics.fmean(latencies), 3),
        "ms_per_query_p50": round(ordered[len(ordered) // 2], 3),
        "ms_per_query_p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
        "peak_rss_mb": None if (rss := _peak_rss_mb()) is None else round(rss, 1),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--sizes",
        required=True,
        type=int,
        nargs="+",
        help="document counts to measure; each is a prefix of the corpus",
    )
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    bundle = JsonlDatasetAdapter.load(arguments.manifest)
    documents = list(bundle.documents)
    queries = list(bundle.queries)[: arguments.queries]
    if not queries:
        raise SystemExit(f"no queries in {arguments.manifest}")

    oversized = [size for size in arguments.sizes if size > len(documents)]
    if oversized:
        raise SystemExit(
            f"requested sizes {oversized} exceed the corpus ({len(documents)} documents)"
        )

    rows = []
    for size in sorted(arguments.sizes):
        row = _measure(
            documents[:size],
            queries,
            top_k=arguments.top_k,
            chunk_size=arguments.chunk_size,
            overlap=arguments.overlap,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {
        "schema_version": "1.0",
        "retriever": "strong-bm25",
        "manifest": str(arguments.manifest),
        "top_k": arguments.top_k,
        "chunker": {"chunk_size": arguments.chunk_size, "overlap": arguments.overlap},
        "python": sys.version.split()[0],
        "measurements": rows,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("=== scaling: strong-bm25 ===")
    print(f"{'docs':>8} {'chunks':>8} {'build_s':>9} {'mean_ms':>9} {'p95_ms':>9} {'ms/1k_chunks':>13}")
    for row in rows:
        per_thousand = row["ms_per_query_mean"] / (row["chunks"] / 1000) if row["chunks"] else 0.0
        print(
            f"{row['documents']:>8} {row['chunks']:>8} {row['build_seconds']:>9.3f} "
            f"{row['ms_per_query_mean']:>9.2f} {row['ms_per_query_p95']:>9.2f} {per_thousand:>13.3f}"
        )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
