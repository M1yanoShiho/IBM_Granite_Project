import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

import evidence_rag.composition as composition_module
import evidence_rag.evaluation.experiment as experiment_module
from evidence_rag.composition import (
    build_generator,
    build_pipeline_from_config,
    build_retriever,
    build_selector,
)
from evidence_rag.contracts.models import (
    CandidateSet,
    GenerationResult,
    PipelineRun,
    Query,
    SelectedEvidenceSet,
    SelectionResult,
)
from evidence_rag.evaluation.experiment import ExperimentWorkflow
from evidence_rag.evaluation.models import EvaluationReport, GoldCase, StageEvaluationReport
from evidence_rag.infrastructure.config import ExperimentConfig, ModuleConfig
from evidence_rag.infrastructure.corpus import CorpusBuilder, CorpusSnapshot
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.retriever.indexing import IndexManifest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MANIFEST = ROOT / "tests/fixtures/reference_dataset/manifest.json"
DOCS_README = ROOT / "docs/README.md"


def write_config(root: Path, *, output_directory: str = "run") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "experiment.toml"
    config_path.write_text(
        f"""
[dataset]
manifest = "{REFERENCE_MANIFEST.as_posix()}"

[output]
directory = "{output_directory}"

[retriever]
name = "bm25"
parameters = {{ k1 = 1.5, b = 0.75 }}

[selector]
name = "top-k"

[generator]
name = "extractive"

[run]
top_k = 3
max_selected = 2
seed = 7
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def workflow(tmp_path: Path) -> ExperimentWorkflow:
    return ExperimentWorkflow.from_toml(write_config(tmp_path))


def assert_artifact_readable(
    workflow: ExperimentWorkflow,
    filename: str,
    model_type: type[object],
) -> tuple[object, ...]:
    return workflow.store.read_jsonl(
        filename,
        model_type,
        expected_dataset_signature=workflow.dataset.dataset_signature,
        expected_corpus_signature=workflow.corpus.manifest.corpus_signature,
    )


def test_reference_dataset_and_corpus_signatures_repeat() -> None:
    first_dataset = JsonlDatasetAdapter.load(REFERENCE_MANIFEST)
    second_dataset = JsonlDatasetAdapter.load(REFERENCE_MANIFEST)
    first_corpus = CorpusBuilder().build(first_dataset.documents, first_dataset.dataset_signature)
    second_corpus = CorpusBuilder().build(
        second_dataset.documents, second_dataset.dataset_signature
    )

    assert first_dataset.dataset_signature == second_dataset.dataset_signature
    assert first_corpus.manifest.corpus_signature == second_corpus.manifest.corpus_signature
    assert len(first_dataset.documents) == 3
    assert len(first_dataset.queries) >= 2


@pytest.mark.parametrize(
    ("factory", "config", "message"),
    (
        (build_retriever, ModuleConfig(name="dense"), "unknown retriever"),
        (
            build_retriever,
            ModuleConfig(name="bm25", parameters={"alpha": 0.5}),
            "unknown retriever parameter",
        ),
        (
            build_retriever,
            ModuleConfig(name="bm25", parameters={"k1": True}),
            "must be numeric",
        ),
        (build_selector, ModuleConfig(name="reranker"), "unknown selector"),
        (
            build_selector,
            ModuleConfig(name="top-k", parameters={"limit": 2}),
            "does not accept parameters",
        ),
        (build_generator, ModuleConfig(name="llm"), "unknown generator"),
        (
            build_generator,
            ModuleConfig(name="extractive", parameters={"style": "short"}),
            "does not accept parameters",
        ),
    ),
)
def test_module_factories_reject_unknown_names_parameters_and_bool_numbers(
    factory: Callable[..., object],
    config: ModuleConfig,
    message: str,
) -> None:
    dataset = JsonlDatasetAdapter.load(REFERENCE_MANIFEST)
    corpus = CorpusBuilder().build(dataset.documents, dataset.dataset_signature)

    with pytest.raises(ValueError, match=message):
        if factory is build_retriever:
            factory(config, corpus)
        else:
            factory(config)


def test_live_pipeline_factory_builds_supported_modules() -> None:
    dataset = JsonlDatasetAdapter.load(REFERENCE_MANIFEST)
    corpus = CorpusBuilder().build(dataset.documents, dataset.dataset_signature)
    config = ExperimentConfig(
        dataset_manifest_path=REFERENCE_MANIFEST,
        output_directory=ROOT / "runs/test-unused",
        retriever=ModuleConfig(name="bm25", parameters={"k1": 1, "b": 0.5}),
        selector=ModuleConfig(name="top-k"),
        generator=ModuleConfig(name="extractive"),
        top_k=3,
        max_selected=2,
        seed=1,
    )

    run = build_pipeline_from_config(config, corpus).run_with_trace(
        dataset.queries[0], top_k=config.top_k, max_selected=config.max_selected
    )

    assert run.candidates.candidates
    assert run.generation.answer


def test_prepare_writes_compatible_base_artifacts(tmp_path: Path) -> None:
    current = workflow(tmp_path)

    assert current.manifest is None

    summary = current.prepare()

    expected = {
        "run_manifest.json",
        "queries.jsonl",
        "gold_cases.jsonl",
        "corpus_snapshot.json",
        "index/index_manifest.json",
        "index/corpus_snapshot.json",
    }
    expected_with_metadata = expected | {
        f"{filename}.metadata.json" for filename in expected if not filename.startswith("index/")
    }
    assert set(summary.artifacts) == expected_with_metadata
    assert (current.store.root / "index/index_manifest.json").is_file()
    assert (current.store.root / "index/corpus_snapshot.json").is_file()
    assert (
        current.store.read_manifest(
            expected_dataset_signature=current.dataset.dataset_signature,
            expected_corpus_signature=current.corpus.manifest.corpus_signature,
        )
        == current.manifest
    )
    assert assert_artifact_readable(current, "queries.jsonl", Query) == current.dataset.queries
    assert (
        assert_artifact_readable(current, "gold_cases.jsonl", GoldCase)
        == current.dataset.gold_cases
    )
    assert (
        current.store.read_json(
            "corpus_snapshot.json",
            CorpusSnapshot,
            expected_dataset_signature=current.dataset.dataset_signature,
            expected_corpus_signature=current.corpus.manifest.corpus_signature,
        )
        == current.corpus
    )


def test_manifest_records_current_repository_commit(tmp_path: Path) -> None:
    expected = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    current = workflow(tmp_path)
    current.prepare()

    assert current.manifest is not None
    assert current.manifest.git_commit == expected
    assert isinstance(current.manifest.git_dirty, bool)
    assert len(current.manifest.source_tree_signature) == 64
    int(current.manifest.source_tree_signature, 16)


def test_manifest_records_chunker_and_validated_index_identity(tmp_path: Path) -> None:
    current = workflow(tmp_path)

    current.prepare()

    assert current.manifest is not None
    index_manifest = IndexManifest.model_validate_json(
        (current.store.root / "index/index_manifest.json").read_text(encoding="utf-8")
    )
    assert current.manifest.chunker_name == current.corpus.manifest.chunker_name
    assert current.manifest.chunker_version == current.corpus.manifest.chunker_version
    assert current.manifest.chunk_size == current.corpus.manifest.chunk_size
    assert current.manifest.overlap == current.corpus.manifest.overlap
    assert current.manifest.index_implementation == index_manifest.implementation
    assert current.manifest.index_implementation_version == index_manifest.implementation_version
    assert current.manifest.index_signature == index_manifest.index_signature


def test_prepare_rejects_existing_output_from_different_config(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    ExperimentWorkflow.from_toml(config_path).prepare()
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("top_k = 3", "top_k = 2"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run manifest does not match"):
        ExperimentWorkflow.from_toml(config_path).prepare()


def test_independent_stages_write_gated_typed_artifacts(tmp_path: Path) -> None:
    current = workflow(tmp_path)

    current.prepare()
    current.run_retriever()
    current.run_selector()
    current.run_generator()

    assert len(assert_artifact_readable(current, "candidate_sets.jsonl", CandidateSet)) == 2
    assert len(assert_artifact_readable(current, "selection_results.jsonl", SelectionResult)) == 2
    assert (
        len(assert_artifact_readable(current, "selected_evidence_sets.jsonl", SelectedEvidenceSet))
        == 2
    )
    assert len(assert_artifact_readable(current, "generation_results.jsonl", GenerationResult)) == 2
    for filename, stage in (
        ("retriever_report.json", "retriever"),
        ("selector_report.json", "selector"),
        ("generator_report.json", "generator"),
    ):
        report = current.store.read_json(
            filename,
            StageEvaluationReport,
            expected_dataset_signature=current.dataset.dataset_signature,
            expected_corpus_signature=current.corpus.manifest.corpus_signature,
        )
        assert report.stage == stage
        assert all(metric.n_scored for metric in report.aggregate.values())


def test_retriever_and_pipeline_load_the_prepared_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = workflow(tmp_path)
    current.prepare()
    index_directory = current.store.root / "index"
    load_calls: list[Path] = []
    original_load = composition_module.load_index

    def tracked_load(directory: Path, **kwargs: object) -> object:
        load_calls.append(directory)
        return original_load(directory, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(composition_module, "load_index", tracked_load)

    current.run_retriever()
    current.run_pipeline()

    assert load_calls == [index_directory, index_directory]


@pytest.mark.parametrize("factory_name", ("retriever", "pipeline"))
@pytest.mark.parametrize("missing_filename", ("index_manifest.json", "corpus_snapshot.json"))
def test_runtime_index_paths_fail_without_rebuilding_missing_files(
    tmp_path: Path,
    factory_name: str,
    missing_filename: str,
) -> None:
    current = workflow(tmp_path)
    current.prepare()
    missing_path = current.store.root / "index" / missing_filename
    missing_path.unlink()

    with pytest.raises(ValueError, match=missing_filename):
        if factory_name == "retriever":
            build_retriever(
                current.config.retriever,
                current.corpus,
                index_directory=current.store.root / "index",
            )
        else:
            build_pipeline_from_config(
                current.config,
                current.corpus,
                index_directory=current.store.root / "index",
            )

    assert not missing_path.exists()


def test_runtime_index_paths_do_not_build_an_absent_index(tmp_path: Path) -> None:
    current = workflow(tmp_path)
    index_directory = current.store.root / "never-prepared"

    with pytest.raises(ValueError, match="index_manifest.json"):
        build_retriever(
            current.config.retriever,
            current.corpus,
            index_directory=index_directory,
        )

    assert not index_directory.exists()


@pytest.mark.parametrize("command", ("retriever", "pipeline"))
def test_runtime_workflow_requires_explicit_prepare(
    tmp_path: Path,
    command: str,
) -> None:
    current = workflow(tmp_path)

    with pytest.raises(ValueError, match="run prepare first"):
        getattr(current, f"run_{command}")()

    assert not (current.store.root / "index").exists()


def fail_factory(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("an upstream factory was called")


def test_selector_reads_frozen_candidates_without_retriever_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = workflow(tmp_path)
    current.prepare()
    current.run_retriever()
    shutil.rmtree(current.store.root / "index")
    monkeypatch.setattr(experiment_module, "build_retriever", fail_factory)
    monkeypatch.setattr(experiment_module, "build_pipeline_from_config", fail_factory)

    current.run_selector()

    assert len(assert_artifact_readable(current, "selection_results.jsonl", SelectionResult)) == 2


def test_generator_reads_frozen_evidence_without_upstream_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = workflow(tmp_path)
    current.prepare()
    current.run_retriever()
    current.run_selector()
    shutil.rmtree(current.store.root / "index")
    monkeypatch.setattr(experiment_module, "build_retriever", fail_factory)
    monkeypatch.setattr(experiment_module, "build_selector", fail_factory)
    monkeypatch.setattr(experiment_module, "build_pipeline_from_config", fail_factory)

    current.run_generator()

    assert len(assert_artifact_readable(current, "generation_results.jsonl", GenerationResult)) == 2


@pytest.mark.parametrize(
    ("stage", "artifact"),
    (
        ("selector", "selected_evidence_sets.jsonl"),
        ("generator", "generation_results.jsonl"),
    ),
)
def test_independent_stage_accepts_new_source_identity_and_records_it(
    tmp_path: Path,
    stage: str,
    artifact: str,
) -> None:
    current = workflow(tmp_path)
    current.prepare()
    current.run_retriever()
    if stage == "generator":
        current.run_selector()
    shutil.rmtree(current.store.root / "index")
    current.git_commit = "new-stage-commit"
    current.git_dirty = True
    current.source_tree_signature = "f" * 64

    getattr(current, f"run_{stage}")()

    metadata = json.loads(
        (current.store.root / f"{artifact}.metadata.json").read_text(encoding="utf-8")
    )
    provenance = metadata["producer_provenance"]
    assert provenance["git_commit"] == "new-stage-commit"
    assert provenance["git_dirty"] is True
    assert provenance["source_tree_signature"] == "f" * 64
    assert provenance["module"] == current.config.model_dump(mode="json")[stage]


@pytest.mark.parametrize(
    ("artifact", "sidecar", "runner", "message"),
    (
        (
            "candidate_sets.jsonl",
            "candidate_sets.jsonl.metadata.json",
            "selector",
            "run manifest signature mismatch",
        ),
        (
            "selected_evidence_sets.jsonl",
            "selected_evidence_sets.jsonl.metadata.json",
            "generator",
            "run manifest signature mismatch",
        ),
    ),
)
def test_workflow_rejects_cross_run_frozen_artifact_substitution(
    tmp_path: Path,
    artifact: str,
    sidecar: str,
    runner: str,
    message: str,
) -> None:
    first = workflow(tmp_path / "first")
    second_config = write_config(tmp_path / "second")
    second_config.write_text(
        second_config.read_text(encoding="utf-8").replace("seed = 7", "seed = 99"),
        encoding="utf-8",
    )
    second = ExperimentWorkflow.from_toml(second_config)
    for current in (first, second):
        current.prepare()
        current.run_retriever()
        if runner == "generator":
            current.run_selector()

    shutil.copyfile(second.store.root / artifact, first.store.root / artifact)
    shutil.copyfile(second.store.root / sidecar, first.store.root / sidecar)

    with pytest.raises(ValueError, match=message):
        if runner == "selector":
            first.run_selector()
        else:
            first.run_generator()


def test_workflow_sidecars_record_upstream_artifact_hashes(tmp_path: Path) -> None:
    current = workflow(tmp_path)
    current.prepare()
    current.run_retriever()
    current.run_selector()
    current.run_generator()

    candidate_metadata = json.loads(
        (current.store.root / "candidate_sets.jsonl.metadata.json").read_text()
    )
    selected_metadata = json.loads(
        (current.store.root / "selected_evidence_sets.jsonl.metadata.json").read_text()
    )
    generation_metadata = json.loads(
        (current.store.root / "generation_results.jsonl.metadata.json").read_text()
    )
    assert set(candidate_metadata["upstream_artifact_hashes"]) == {
        "corpus_snapshot.json",
        "queries.jsonl",
    }
    assert set(selected_metadata["upstream_artifact_hashes"]) == {"candidate_sets.jsonl"}
    assert set(generation_metadata["upstream_artifact_hashes"]) == {"selected_evidence_sets.jsonl"}


def test_workflow_rejects_tampered_upstream_hash_provenance(tmp_path: Path) -> None:
    current = workflow(tmp_path)
    current.prepare()
    current.run_retriever()
    current.run_selector()
    sidecar_path = current.store.root / "selected_evidence_sets.jsonl.metadata.json"
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    metadata["upstream_artifact_hashes"]["candidate_sets.jsonl"] = "0" * 64
    sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="upstream artifact hash mismatch"):
        current.run_generator()


def test_retriever_rejects_tampered_index_provenance_in_run_manifest(
    tmp_path: Path,
) -> None:
    current = workflow(tmp_path)
    current.prepare()
    sidecar_path = current.store.root / "run_manifest.json.metadata.json"
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    metadata["upstream_artifact_hashes"]["index/index_manifest.json"] = "0" * 64
    sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="upstream artifact hashes mismatch"):
        current.run_retriever()


def test_independent_stages_fail_when_upstream_artifact_is_missing(tmp_path: Path) -> None:
    selector_workflow = workflow(tmp_path / "selector")
    selector_workflow.prepare()
    with pytest.raises(ValueError, match=r"candidate_sets\.jsonl.*retriever"):
        selector_workflow.run_selector()

    generator_workflow = workflow(tmp_path / "generator")
    generator_workflow.prepare()
    with pytest.raises(ValueError, match=r"selected_evidence_sets\.jsonl.*selector"):
        generator_workflow.run_generator()


def test_live_pipeline_writes_one_trace_per_query_and_full_report(tmp_path: Path) -> None:
    current = workflow(tmp_path)

    current.prepare()
    current.run_pipeline()

    runs = assert_artifact_readable(current, "pipeline_runs.jsonl", PipelineRun)
    report = current.store.read_json(
        "evaluation_report.json",
        EvaluationReport,
        expected_dataset_signature=current.dataset.dataset_signature,
        expected_corpus_signature=current.corpus.manifest.corpus_signature,
    )
    assert tuple(run.query.query_id for run in runs) == tuple(
        query.query_id for query in current.dataset.queries
    )
    assert report.case_ids == tuple(query.query_id for query in current.dataset.queries)
    assert {key.partition(".")[0] for key in report.aggregate} == {
        "retriever",
        "selector",
        "generator",
        "system",
    }
    assert all(metric.n_scored for metric in report.aggregate.values())


def test_run_all_completes_reference_workflow(tmp_path: Path) -> None:
    current = workflow(tmp_path)

    summary = current.run_all()

    assert summary.command == "all"
    assert summary.query_count == 2
    assert "index/index_manifest.json" in summary.artifacts
    assert "index/corpus_snapshot.json" in summary.artifacts
    assert (current.config.output_directory / "evaluation_report.json").is_file()
    assert (current.config.output_directory / "generator_report.json").is_file()


def test_docs_describe_actual_experiment_commands_and_index_files() -> None:
    documentation = DOCS_README.read_text(encoding="utf-8")
    command_prefix = (
        "python -m evidence_rag.cli.experiment --config configs/experiments/reference_baseline.toml"
    )
    for command in ("prepare", "retriever", "selector", "generator", "pipeline", "all"):
        assert f"{command_prefix} {command}" in documentation
    assert "runs/reference-baseline/index/index_manifest.json" in documentation
    assert "runs/reference-baseline/index/corpus_snapshot.json" in documentation
    assert "冻结 artifact" in documentation
    assert "实时 Pipeline" in documentation
