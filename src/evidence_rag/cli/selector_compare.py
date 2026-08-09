"""Run TopK and Reliability-MIS over the same frozen candidate-pool file."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.evaluation.selector_experiment import (
    align_inputs,
    compare_arm_results,
    read_jsonl,
    run_selector_arm,
    sha256_file,
    validate_pool_manifest,
    write_comparison_outputs,
)
from evidence_rag.generator.granite import (
    DEFAULT_GRANITE_MODEL_ID,
    GraniteGenerationConfig,
    GraniteLLMClient,
)
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.materializer.source_parent import read_parent_index
from evidence_rag.selector.deberta_contradiction import (
    DEFAULT_CONTRADICTION_MODEL_ID,
    DEFAULT_CONTRADICTION_MODEL_REVISION,
    DebertaContradictionScorer,
)
from evidence_rag.selector.reliability_mis import (
    DEFAULT_CONTRADICTION_THRESHOLD,
    LazyAnswerGenerator,
    MISSelectionEvent,
    ReliabilityMISSelector,
)
from evidence_rag.selector.top_k import TopKSelector


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Selectors on one frozen pool")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--source-parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--query-ids", type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-selected", type=int, default=10)
    parser.add_argument("--run-seed", type=int, default=13)
    parser.add_argument("--stats-seed", type=int, default=20260809)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--nli-batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--git-root", type=Path, default=Path.cwd())
    return parser


def _query_id_filter(path: Path | None) -> tuple[str, ...] | None:
    if path is None:
        return None
    ids = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(ids) != len(set(ids)):
        raise ValueError("--query-ids contains duplicates")
    if not ids:
        raise ValueError("--query-ids is empty")
    return ids


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    pool_manifest = json.loads(arguments.pool_manifest.read_text(encoding="utf-8"))
    pool_sha = validate_pool_manifest(
        pool_manifest,
        candidate_pool_path=arguments.candidates,
        expected_top_n=arguments.top_n,
    )
    bundle = JsonlDatasetAdapter.load(arguments.dataset_manifest)
    candidates = read_jsonl(arguments.candidates, CandidateSet)
    aligned = list(align_inputs(bundle, candidates))
    requested_ids = _query_id_filter(arguments.query_ids)
    if requested_ids is not None:
        requested = set(requested_ids)
        unknown = requested - {query.query_id for query, _candidates, _gold in aligned}
        if unknown:
            raise ValueError(f"--query-ids contains unknown ID: {sorted(unknown)[0]}")
        by_query = {row[0].query_id: row for row in aligned}
        aligned = [by_query[query_id] for query_id in requested_ids]

    parents = read_parent_index(arguments.source_parent)
    source_parent_sha = sha256_file(arguments.source_parent)
    harm_by_query = (
        None
        if arguments.provenance is None
        else {
            record.query_id: record.counterfactual_document_id
            for record in read_provenance(arguments.provenance)
        }
    )

    top_rows, _top_selections, _top_selected = run_selector_arm(
        "top-k",
        TopKSelector(),
        aligned,
        output_directory=arguments.output / "top-k",
        candidate_pool_sha256=pool_sha,
        source_parent_sha256=source_parent_sha,
        max_selected=arguments.max_selected,
        run_seed=arguments.run_seed,
        dataset_signature=bundle.dataset_signature,
        harm_by_query=harm_by_query,
        resume=arguments.resume,
        git_root=arguments.git_root,
    )

    events: dict[str, MISSelectionEvent] = {}

    def record_event(event: MISSelectionEvent) -> None:
        events[event.query_id] = event

    device = os.getenv("LLM_DEVICE", "auto")

    def granite_factory() -> GraniteLLMClient:
        return GraniteLLMClient(
            model_id=DEFAULT_GRANITE_MODEL_ID,
            config=GraniteGenerationConfig(max_new_tokens=32, temperature=0.0),
            device=device,
        )

    scorer = DebertaContradictionScorer(
        model_id=DEFAULT_CONTRADICTION_MODEL_ID,
        model_revision=DEFAULT_CONTRADICTION_MODEL_REVISION,
        device=device,
        batch_size=arguments.nli_batch_size,
    )
    reliability_mis = ReliabilityMISSelector(
        LazyAnswerGenerator(granite_factory),
        scorer,
        parents.parent_by_document,
        top_n=arguments.top_n,
        contradiction_threshold=DEFAULT_CONTRADICTION_THRESHOLD,
        on_event=record_event,
    )

    def event_lookup(query_id: str) -> object | None:
        event = events.pop(query_id, None)
        return None if event is None else asdict(event)

    mis_rows, _mis_selections, _mis_selected = run_selector_arm(
        "reliability-mis",
        reliability_mis,
        aligned,
        output_directory=arguments.output / "reliability-mis",
        candidate_pool_sha256=pool_sha,
        source_parent_sha256=source_parent_sha,
        max_selected=arguments.max_selected,
        run_seed=arguments.run_seed,
        dataset_signature=bundle.dataset_signature,
        harm_by_query=harm_by_query,
        event_lookup=event_lookup,
        resume=arguments.resume,
        git_root=arguments.git_root,
    )
    summary = compare_arm_results(
        top_rows,
        mis_rows,
        stats_seed=arguments.stats_seed,
        iterations=arguments.iterations,
    )
    summary = {
        **summary,
        "candidate_pool_sha256": pool_sha,
        "source_parent_sha256": source_parent_sha,
        "query_count": len(aligned),
        "top_n": arguments.top_n,
        "max_selected": arguments.max_selected,
        "models": {
            "answer_extraction": DEFAULT_GRANITE_MODEL_ID,
            "nli": DEFAULT_CONTRADICTION_MODEL_ID,
            "nli_revision": DEFAULT_CONTRADICTION_MODEL_REVISION,
            "nli_batch_size": arguments.nli_batch_size,
            "contradiction_threshold": DEFAULT_CONTRADICTION_THRESHOLD,
        },
    }
    write_comparison_outputs(arguments.output, top_rows, mis_rows, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
