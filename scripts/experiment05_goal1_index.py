#!/usr/bin/env python3
"""Build and audit Experiment 05 full-corpus retrieval indices."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.evaluation.experiment05_index import (  # noqa: E402
    DenseIndexParameters,
    build_dense_index,
    freeze_bm25_manifest,
    normalize_corpus,
    pyserini_index_command,
    read_corpus_manifest,
)


def normalize_stage(args: argparse.Namespace) -> dict[str, object]:
    manifest = normalize_corpus(
        dataset=args.dataset,
        source_kind=args.source_kind,
        source_path=args.source,
        output_dir=args.output_dir,
        max_words=args.max_words,
        shard_size=args.shard_size,
    )
    return {
        "dataset": manifest["dataset"],
        "passage_count": manifest["passage_count"],
        "ordered_passage_ids_sha256": manifest["ordered_passage_ids_sha256"],
        "shard_count": manifest["shard_count"],
        "source_sha256": manifest["source_sha256"],
    }


def dense_stage(args: argparse.Namespace) -> dict[str, object]:
    manifest = build_dense_index(
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir,
        model_snapshot=args.model_snapshot,
        device=args.device,
        parameters=DenseIndexParameters(
            dimension=args.dimension,
            nlist=args.nlist,
            m=args.m,
            nbits=args.nbits,
            nprobe=args.nprobe,
            train_size=args.train_size,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            seed=args.seed,
        ),
    )
    return {
        "dataset": manifest["dataset"],
        "passage_count": manifest["passage_count"],
        "faiss_ntotal": manifest["faiss_ntotal"],
        "build_signature": manifest["build_signature"],
        "index_sha256": manifest["index_sha256"],
    }


def bm25_stage(args: argparse.Namespace) -> dict[str, object]:
    corpus = read_corpus_manifest(args.corpus_dir)
    args.index_dir.parent.mkdir(parents=True, exist_ok=True)
    command = pyserini_index_command(
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir,
        threads=args.threads,
    )
    subprocess.run(command, check=True)
    from pyserini.index.lucene import LuceneIndexReader

    reader = LuceneIndexReader(str(args.index_dir))
    stats = reader.stats()
    document_count = int(stats["documents"])
    manifest = freeze_bm25_manifest(
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir,
        document_count=document_count,
        pyserini_version=importlib.metadata.version("pyserini"),
        k1=args.k1,
        b=args.b,
    )
    if document_count != int(corpus["passage_count"]):
        raise ValueError("BM25 index document count differs after manifest freeze")
    return {
        "dataset": manifest["dataset"],
        "passage_count": manifest["passage_count"],
        "document_count": manifest["document_count"],
        "index_tree_sha256": manifest["index_tree_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--dataset", required=True)
    normalize.add_argument("--source-kind", required=True, choices=("kilt", "dpr"))
    normalize.add_argument("--source", required=True, type=Path)
    normalize.add_argument("--output-dir", required=True, type=Path)
    normalize.add_argument("--max-words", type=int, default=100)
    normalize.add_argument("--shard-size", type=int, default=250_000)

    dense = subparsers.add_parser("dense")
    dense.add_argument("--corpus-dir", required=True, type=Path)
    dense.add_argument("--index-dir", required=True, type=Path)
    dense.add_argument("--model-snapshot", required=True, type=Path)
    dense.add_argument("--device", default="cuda")
    dense.add_argument("--dimension", type=int, default=768)
    dense.add_argument("--nlist", type=int, default=8192)
    dense.add_argument("--m", type=int, default=96)
    dense.add_argument("--nbits", type=int, default=8)
    dense.add_argument("--nprobe", type=int, default=64)
    dense.add_argument("--train-size", type=int, default=327_680)
    dense.add_argument("--batch-size", type=int, default=512)
    dense.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    dense.add_argument("--seed", type=int, default=13)

    bm25 = subparsers.add_parser("bm25")
    bm25.add_argument("--corpus-dir", required=True, type=Path)
    bm25.add_argument("--index-dir", required=True, type=Path)
    bm25.add_argument("--threads", type=int, default=16)
    bm25.add_argument("--k1", type=float, default=0.9)
    bm25.add_argument("--b", type=float, default=0.4)
    args = parser.parse_args()

    if args.stage == "normalize":
        summary = normalize_stage(args)
    elif args.stage == "dense":
        summary = dense_stage(args)
    elif args.stage == "bm25":
        summary = bm25_stage(args)
    else:  # pragma: no cover - argparse prevents this
        raise AssertionError(args.stage)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
