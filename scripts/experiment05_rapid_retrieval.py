#!/usr/bin/env python3
"""Run Experiment 05 full-corpus BM25 candidate retrieval and neural reranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.evaluation.experiment05_data import read_runtime_bundle  # noqa: E402
from evidence_rag.evaluation.experiment05_retrieval import (  # noqa: E402
    RapidRetrievalConfig,
    build_rapid_retrieval_trace,
    canonical_sha256,
    validate_rapid_retrieval_trace,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                validate_rapid_retrieval_trace(row)
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--bm25-index", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidate-pool-size", type=int, default=1_000)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument("--reranker-pool-size", type=int, default=40)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    # Pyserini imports its optional OpenAI encoder eagerly in some releases.
    # This experiment never instantiates that encoder or calls the API, but the
    # dependency still requires a non-empty value merely to complete the import.
    os.environ.setdefault("OPENAI_API_KEY", "unused-local-pyserini-import")
    from pyserini.search.lucene import LuceneSearcher
    from sentence_transformers import SentenceTransformer

    config = RapidRetrievalConfig(
        candidate_pool_size=args.candidate_pool_size,
        final_top_k=args.final_top_k,
        reranker_pool_size=args.reranker_pool_size,
        rrf_k=args.rrf_k,
        dense_batch_size=args.dense_batch_size,
    )
    config.validate()
    runtime = list(read_runtime_bundle(args.runtime, dataset=args.dataset))
    if args.limit is not None:
        runtime = runtime[: args.limit]
    existing = _read_existing(args.output)
    expected_ids = [str(row["query_id"]) for row in runtime]
    observed_ids = [str(row["query_id"]) for row in existing]
    if observed_ids != expected_ids[: len(observed_ids)]:
        raise ValueError("retrieval resume file is not an ordered runtime prefix")
    for row in existing:
        observed_config = dict(row.get("config", {}))
        observed_config.pop("dense_batch_size", None)
        expected_config = dict(config.__dict__)
        expected_config.pop("dense_batch_size", None)
        if observed_config != expected_config:
            raise ValueError("retrieval resume file changes a semantic retrieval parameter")

    searcher = LuceneSearcher(str(args.bm25_index))
    searcher.set_bm25(0.9, 0.4)
    encoder = SentenceTransformer(str(args.model_snapshot), device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for runtime_row in runtime[len(existing) :]:
            hits = searcher.search(str(runtime_row["question"]), config.candidate_pool_size)
            candidates: list[dict[str, Any]] = []
            for rank, hit in enumerate(hits, start=1):
                document = searcher.doc(hit.docid)
                if document is None:
                    raise ValueError("BM25 hit cannot be resolved to a stored document")
                raw = json.loads(document.raw())
                if str(raw.get("id")) != str(hit.docid):
                    raise ValueError("BM25 hit ID differs from stored raw document ID")
                raw["bm25_rank"] = rank
                raw["bm25_score"] = float(hit.score)
                candidates.append(raw)
            trace = build_rapid_retrieval_trace(
                dataset=args.dataset,
                query_id=str(runtime_row["query_id"]),
                question=str(runtime_row["question"]),
                bm25_candidates=candidates,
                encoder=encoder,
                config=config,
            )
            handle.write(json.dumps(trace, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()

    completed = _read_existing(args.output)
    if [str(row["query_id"]) for row in completed] != expected_ids:
        raise ValueError("rapid retrieval output does not cover the frozen runtime IDs")
    manifest = {
        "schema_version": "experiment05.rapid_retrieval_manifest.v1",
        "status": "PASS",
        "dataset": args.dataset,
        "count": len(completed),
        "ordered_query_ids_sha256": canonical_sha256(expected_ids),
        "runtime_sha256": _file_sha256(args.runtime),
        "bm25_index": str(args.bm25_index),
        "model_snapshot": str(args.model_snapshot),
        "config": config.__dict__,
        "resume_existing_count": len(existing),
        "observed_dense_batch_sizes": sorted(
            {int(row["config"]["dense_batch_size"]) for row in completed}
        ),
        "output": str(args.output),
        "output_sha256": _file_sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dataset": args.dataset, "count": len(completed), "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
