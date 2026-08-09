"""Audit and freeze one Hybrid-RRF Top-N candidate pool for Selector experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.evaluation.selector_experiment import (
    audit_candidate_pool,
    read_jsonl,
)
from evidence_rag.infrastructure.config import load_experiment_config
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.materializer.source_parent import read_parent_index
from evidence_rag.retriever.granite import DEFAULT_GRANITE_EMBEDDING_MODEL_ID


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze and audit a Selector candidate pool")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--source-parent", required=True, type=Path)
    parser.add_argument("--retriever-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--embedding-model-id", default=DEFAULT_GRANITE_EMBEDDING_MODEL_ID
    )
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--git-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_experiment_config(arguments.retriever_config)
    if config.top_k != arguments.top_n:
        raise ValueError("Retriever config top_k differs from --top-n")
    if config.retriever.name != "hybrid" or config.retriever.parameters.get("fusion") != "rrf":
        raise ValueError("Selector experiment requires the frozen Hybrid RRF Retriever")

    bundle = JsonlDatasetAdapter.load(arguments.dataset_manifest)
    candidates = read_jsonl(arguments.candidates, CandidateSet)
    parents = read_parent_index(arguments.source_parent)
    harm_by_query = (
        None
        if arguments.provenance is None
        else {
            record.query_id: record.counterfactual_document_id
            for record in read_provenance(arguments.provenance)
        }
    )
    manifest = audit_candidate_pool(
        bundle,
        candidates,
        parents,
        candidate_pool_path=arguments.candidates,
        dataset_manifest_path=arguments.dataset_manifest,
        retriever_config_path=arguments.retriever_config,
        top_n=arguments.top_n,
        seed=arguments.seed,
        harm_by_query=harm_by_query,
        embedding_model_id=arguments.embedding_model_id,
        embedding_revision=arguments.embedding_revision,
        git_root=arguments.git_root,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pool = manifest["candidate_pool"]
    audit = manifest["audit"]
    if not isinstance(pool, dict) or not isinstance(audit, dict):
        raise AssertionError("internal pool audit shape error")
    if pool["exact_top_n_rate"] != 1.0:
        raise ValueError("not every query has exactly Top-N candidates")
    if audit["unresolved_parent_count"] != 0:
        raise ValueError("candidate pool has unresolved source parents")
    print(
        json.dumps(
            {
                "candidate_pool_sha256": pool["sha256"],
                "exact_top_n_rate": pool["exact_top_n_rate"],
                "harmful_pool_hit_rate": audit["harmful_pool_hit_rate"],
                "query_count": pool["query_count"],
                "required_recall_at_top_n": audit["required_recall_at_top_n"],
                "unresolved_parent_count": audit["unresolved_parent_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
