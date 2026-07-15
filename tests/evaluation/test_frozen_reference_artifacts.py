from pathlib import Path

from evidence_rag.contracts.models import CandidateSet, SelectedEvidenceSet
from evidence_rag.evaluation.runners import run_generator_stage, run_selector_stage
from evidence_rag.generator.extractive import ExtractiveGenerator
from evidence_rag.infrastructure.artifacts import ArtifactStore
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.selector.top_k import TopKSelector

FIXTURES = Path(__file__).parents[1] / "fixtures"


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
