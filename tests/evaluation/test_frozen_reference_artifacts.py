from pathlib import Path

from evidence_rag.contracts.models import CandidateSet, SelectedEvidenceSet
from evidence_rag.evaluation.runners import run_generator_stage, run_selector_stage
from evidence_rag.generator.extractive import ExtractiveGenerator
from evidence_rag.infrastructure.artifacts import ArtifactStore
from evidence_rag.infrastructure.config import ChunkerConfig
from evidence_rag.infrastructure.corpus import CorpusBuilder, build_chunker
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.selector.top_k import TopKSelector

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_the_committed_frozen_chain_still_reproduces_from_current_code() -> None:
    """The frozen chain must be rebuildable, not merely self-consistent.

    Everything else in this file checks that the committed artifacts agree with *each
    other* — every sidecar carries the same corpus signature, so the chain stays green
    even when no code in the tree can produce that signature any more. It drifted exactly
    that way once: `1a38f52` added chunk metadata to the hashed corpus manifest, which
    moved `corpus_signature` from `7de07bd…` to `f359854…` while the committed chain kept
    the old value for three weeks with nothing failing.

    That matters beyond tidiness. `docs/SHARED_RAG_INFRASTRUCTURE_PLAN.md` §4.5 makes this
    chain the frozen baseline the Selector and Generator groups read, so a chain claiming a
    corpus the current code cannot rebuild is a chain nobody can verify.
    """

    dataset = JsonlDatasetAdapter.load(FIXTURES / "reference_dataset/manifest.json")
    store = ArtifactStore(FIXTURES / "reference_baseline_artifacts")
    manifest = store.read_manifest(expected_dataset_signature=dataset.dataset_signature)

    # configs/experiments/reference_baseline.toml has no [chunker] table, so the frozen
    # chain is the default chunker's output.
    defaults = ChunkerConfig()
    rebuilt = CorpusBuilder(
        build_chunker(defaults.name, chunk_size=defaults.chunk_size, overlap=defaults.overlap)
    ).build(dataset.documents, dataset.dataset_signature)

    assert rebuilt.manifest.corpus_signature == manifest.corpus_signature, (
        "the committed frozen chain no longer reproduces: current code builds "
        f"{rebuilt.manifest.corpus_signature} where the chain records "
        f"{manifest.corpus_signature}. Either the change that moved it was unintended, "
        "or the chain needs re-freezing alongside it."
    )


def test_committed_frozen_artifacts_support_selector_and_generator_experiments() -> None:
    dataset = JsonlDatasetAdapter.load(FIXTURES / "reference_dataset/manifest.json")
    store = ArtifactStore(FIXTURES / "reference_baseline_artifacts")
    manifest = store.read_manifest(
        expected_dataset_signature=dataset.dataset_signature,
    )
    candidate_upstream = {
        filename: store.artifact_hash(filename)
        for filename in ("corpus_snapshot.json", "queries.jsonl")
    }
    candidates = store.read_jsonl(
        "candidate_sets.jsonl",
        CandidateSet,
        expected_dataset_signature=dataset.dataset_signature,
        expected_corpus_signature=manifest.corpus_signature,
        expected_producer="ExperimentWorkflow",
        expected_stage="retriever",
        expected_upstream_artifact_hashes=candidate_upstream,
    )
    selected = store.read_jsonl(
        "selected_evidence_sets.jsonl",
        SelectedEvidenceSet,
        expected_dataset_signature=dataset.dataset_signature,
        expected_corpus_signature=manifest.corpus_signature,
        expected_producer="ExperimentWorkflow",
        expected_stage="selector",
        expected_upstream_artifact_hashes={
            "candidate_sets.jsonl": store.artifact_hash("candidate_sets.jsonl")
        },
    )

    selector_run = run_selector_stage(
        TopKSelector(),
        dataset.queries,
        candidates,
        dataset.gold_cases,
        dataset_signature=dataset.dataset_signature,
        max_selected=manifest.max_selected,
    )
    generator_run = run_generator_stage(
        ExtractiveGenerator(),
        dataset.queries,
        selected,
        dataset.gold_cases,
        dataset_signature=dataset.dataset_signature,
    )

    assert selector_run.selected_sets == selected
    assert generator_run.report.case_ids == tuple(query.query_id for query in dataset.queries)
    assert generator_run.report.dataset_signature == dataset.dataset_signature
