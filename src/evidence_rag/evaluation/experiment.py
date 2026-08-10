import subprocess
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from evidence_rag.composition import (
    build_generator,
    build_pipeline_from_config,
    build_retriever,
    build_selector,
    prepare_retriever_index,
    retriever_provenance,
    source_parent_provenance,
)
from evidence_rag.contracts.models import (
    CandidateSet,
    SelectedEvidenceSet,
)
from evidence_rag.evaluation.evaluator import evaluate_dataset
from evidence_rag.evaluation.models import GoldCase
from evidence_rag.evaluation.runners import (
    run_generator_stage,
    run_retriever_stage,
    run_selector_stage,
)
from evidence_rag.infrastructure.artifacts import (
    ArtifactStore,
    ProducerProvenance,
    RunManifest,
    metadata_filename,
)
from evidence_rag.infrastructure.config import (
    ExperimentConfig,
    ModuleConfig,
    load_experiment_config,
)
from evidence_rag.infrastructure.corpus import CorpusBuilder, CorpusSnapshot, build_chunker
from evidence_rag.infrastructure.datasets import DatasetBundle, JsonlDatasetAdapter
from evidence_rag.retriever.indexing import IndexManifest, read_index_manifest

NonEmpty = Annotated[str, Field(min_length=1)]
ArtifactModelT = TypeVar("ArtifactModelT", bound=BaseModel)
Command = Literal["prepare", "retriever", "selector", "generator", "pipeline", "all"]
PRODUCER = "ExperimentWorkflow"
INDEX_ARTIFACTS = frozenset({"index/index_manifest.json", "index/corpus_snapshot.json"})


class WorkflowSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    command: Command
    output_directory: Path
    query_count: int
    artifacts: tuple[NonEmpty, ...]


def _git_state(config_path: Path) -> tuple[str, bool]:
    search_roots = (config_path.parent, Path(__file__).resolve().parents[3])
    for root in search_roots:
        try:
            commit = subprocess.run(
                ("git", "-C", str(root), "rev-parse", "HEAD"),
                check=True,
                capture_output=True,
                text=True,
            )
            status = subprocess.run(
                ("git", "-C", str(root), "status", "--porcelain"),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        if commit_hash := commit.stdout.strip():
            return commit_hash, bool(status.stdout.strip())
    return "git-unavailable", False


def _source_tree_signature() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class ExperimentWorkflow:
    def __init__(self, config: ExperimentConfig, *, config_path: Path) -> None:
        self.config = config
        self.dataset: DatasetBundle = JsonlDatasetAdapter.load(config.dataset_manifest_path)
        self.corpus: CorpusSnapshot = CorpusBuilder(
            build_chunker(
                config.chunker.name,
                chunk_size=config.chunker.chunk_size,
                overlap=config.chunker.overlap,
            )
        ).build(
            self.dataset.documents,
            self.dataset.dataset_signature,
        )
        self.git_commit, self.git_dirty = _git_state(config_path)
        self.source_tree_signature = _source_tree_signature()
        self.manifest: RunManifest | None = None
        self.store = ArtifactStore(config.output_directory)

    def _build_manifest(self, index_manifest: IndexManifest) -> RunManifest:
        return RunManifest(
            dataset_id=self.dataset.manifest.dataset_id,
            dataset_version=self.dataset.manifest.dataset_version,
            split=self.dataset.manifest.split,
            dataset_signature=self.dataset.dataset_signature,
            corpus_signature=self.corpus.manifest.corpus_signature,
            chunker_name=self.corpus.manifest.chunker_name,
            chunker_version=self.corpus.manifest.chunker_version,
            chunk_size=self.corpus.manifest.chunk_size,
            overlap=self.corpus.manifest.overlap,
            index_implementation=index_manifest.implementation,
            index_implementation_version=index_manifest.implementation_version,
            index_signature=index_manifest.index_signature,
            retriever=self.config.retriever,
            selector=self.config.selector,
            generator=self.config.generator,
            top_k=self.config.top_k,
            max_selected=self.config.max_selected,
            seed=self.config.seed,
            git_commit=self.git_commit,
            git_dirty=self.git_dirty,
            source_tree_signature=self.source_tree_signature,
        )

    @classmethod
    def from_toml(cls, path: Path) -> "ExperimentWorkflow":
        config_path = Path(path).resolve()
        return cls(load_experiment_config(config_path), config_path=config_path)

    def prepare(self) -> WorkflowSummary:
        base_artifacts = (
            "run_manifest.json",
            "queries.jsonl",
            "gold_cases.jsonl",
            "corpus_snapshot.json",
            "index/index_manifest.json",
            "index/corpus_snapshot.json",
        )
        index_manifest = prepare_retriever_index(
            self.config.retriever,
            self.corpus,
            self._index_directory,
        )
        expected_manifest = self._build_manifest(index_manifest)
        if (self.store.root / "run_manifest.json").exists():
            self._validate_stored_manifest(index_manifest=index_manifest, strict=True)
        self.manifest = expected_manifest
        self.store.write_manifest(
            expected_manifest,
            producer=PRODUCER,
            stage="prepare",
            producer_provenance=self._stage_provenance("prepare"),
            upstream_artifact_hashes={
                filename: self._raw_artifact_hash(filename) for filename in sorted(INDEX_ARTIFACTS)
            },
        )
        self.store.write_jsonl(
            "queries.jsonl",
            self.dataset.queries,
            producer=PRODUCER,
            stage="prepare",
            producer_provenance=self._stage_provenance("prepare"),
        )
        self.store.write_jsonl(
            "gold_cases.jsonl",
            self.dataset.gold_cases,
            producer=PRODUCER,
            stage="prepare",
            producer_provenance=self._stage_provenance("prepare"),
        )
        self.store.write_json(
            "corpus_snapshot.json",
            self.corpus,
            producer=PRODUCER,
            stage="prepare",
            producer_provenance=self._stage_provenance("prepare"),
        )
        return self._summary("prepare", self._with_metadata(base_artifacts))

    def run_retriever(self) -> WorkflowSummary:
        self._ensure_prepared()
        retriever = build_retriever(
            self.config.retriever,
            self.corpus,
            index_directory=self._index_directory,
        )
        index_manifest = read_index_manifest(self._index_directory)
        self._validate_stored_manifest(index_manifest=index_manifest)
        # The pool records its own producer. `.metadata.json` already carried the retriever
        # config, but a sidecar is a separate file: Gate 0A is handed a path to
        # candidate_sets.jsonl, and a copied or concatenated pool arrives without it. M0 §4's
        # freeze has to be checkable from the artefact that is actually read.
        run = run_retriever_stage(
            retriever,
            self.dataset.queries,
            self.dataset.gold_cases,
            dataset_signature=self.dataset.dataset_signature,
            top_k=self.config.top_k,
            retriever_provenance=retriever_provenance(index_manifest),
        )
        self._write_jsonl(
            "candidate_sets.jsonl",
            run.candidate_sets,
            stage="retriever",
            upstream=("corpus_snapshot.json", "queries.jsonl"),
        )
        self._write_json(
            "retriever_report.json",
            run.report,
            stage="retriever",
            upstream=("candidate_sets.jsonl",),
        )
        return self._summary(
            "retriever",
            self._with_metadata(("candidate_sets.jsonl", "retriever_report.json")),
        )

    def run_selector(self) -> WorkflowSummary:
        self._ensure_prepared()
        self._require_artifact("candidate_sets.jsonl", "retriever")
        candidate_sets = self._read_jsonl(
            "candidate_sets.jsonl",
            CandidateSet,
            expected_stage="retriever",
            expected_upstream=("corpus_snapshot.json", "queries.jsonl"),
        )
        run = run_selector_stage(
            build_selector(self.config.selector),
            self.dataset.queries,
            candidate_sets,
            self.dataset.gold_cases,
            dataset_signature=self.dataset.dataset_signature,
            max_selected=self.config.max_selected,
        )
        self._write_jsonl(
            "selection_results.jsonl",
            run.selection_results,
            stage="selector",
            upstream=("candidate_sets.jsonl",),
        )
        self._write_jsonl(
            "selected_evidence_sets.jsonl",
            run.selected_sets,
            stage="selector",
            upstream=("candidate_sets.jsonl",),
        )
        self._write_json(
            "selector_report.json",
            run.report,
            stage="selector",
            upstream=("selection_results.jsonl", "selected_evidence_sets.jsonl"),
        )
        return self._summary(
            "selector",
            self._with_metadata(
                (
                    "selection_results.jsonl",
                    "selected_evidence_sets.jsonl",
                    "selector_report.json",
                )
            ),
        )

    def run_generator(self) -> WorkflowSummary:
        self._ensure_prepared()
        self._require_artifact("selected_evidence_sets.jsonl", "selector")
        selected_sets = self._read_jsonl(
            "selected_evidence_sets.jsonl",
            SelectedEvidenceSet,
            expected_stage="selector",
            expected_upstream=("candidate_sets.jsonl",),
        )
        run = run_generator_stage(
            build_generator(self.config.generator),
            self.dataset.queries,
            selected_sets,
            self.dataset.gold_cases,
            dataset_signature=self.dataset.dataset_signature,
        )
        self._write_jsonl(
            "generation_results.jsonl",
            run.generation_results,
            stage="generator",
            upstream=("selected_evidence_sets.jsonl",),
        )
        self._write_json(
            "generator_report.json",
            run.report,
            stage="generator",
            upstream=("generation_results.jsonl",),
        )
        return self._summary(
            "generator",
            self._with_metadata(("generation_results.jsonl", "generator_report.json")),
        )

    def run_pipeline(self) -> WorkflowSummary:
        self._ensure_prepared()
        pipeline = build_pipeline_from_config(
            self.config,
            self.corpus,
            index_directory=self._index_directory,
        )
        self._validate_stored_manifest(index_manifest=read_index_manifest(self._index_directory))
        runs = tuple(
            pipeline.run_with_trace(
                query,
                top_k=self.config.top_k,
                max_selected=self.config.max_selected,
            )
            for query in self.dataset.queries
        )
        gold_cases = self._ordered_gold_cases()
        report = evaluate_dataset(
            zip(runs, gold_cases, strict=True),
            dataset_signature=self.dataset.dataset_signature,
        )
        self._write_jsonl(
            "pipeline_runs.jsonl",
            runs,
            stage="pipeline",
            upstream=("corpus_snapshot.json", "gold_cases.jsonl", "queries.jsonl"),
        )
        self._write_json(
            "evaluation_report.json",
            report,
            stage="pipeline",
            upstream=("pipeline_runs.jsonl",),
        )
        return self._summary(
            "pipeline",
            self._with_metadata(("pipeline_runs.jsonl", "evaluation_report.json")),
        )

    def run_all(self) -> WorkflowSummary:
        self.prepare()
        self.run_retriever()
        self.run_selector()
        self.run_generator()
        self.run_pipeline()
        return self._summary(
            "all",
            self._with_metadata(
                (
                    "run_manifest.json",
                    "queries.jsonl",
                    "gold_cases.jsonl",
                    "corpus_snapshot.json",
                    "index/index_manifest.json",
                    "index/corpus_snapshot.json",
                    "candidate_sets.jsonl",
                    "retriever_report.json",
                    "selection_results.jsonl",
                    "selected_evidence_sets.jsonl",
                    "selector_report.json",
                    "generation_results.jsonl",
                    "generator_report.json",
                    "pipeline_runs.jsonl",
                    "evaluation_report.json",
                )
            ),
        )

    @property
    def _index_directory(self) -> Path:
        return self.store.root / "index"

    def _ensure_prepared(self) -> None:
        if not (self.store.root / "run_manifest.json").is_file():
            raise ValueError("experiment is not prepared; run prepare first")
        self._validate_stored_manifest()

    def _validate_stored_manifest(
        self,
        *,
        index_manifest: IndexManifest | None = None,
        strict: bool = False,
    ) -> None:
        index_hashes = (
            {
                filename: self._raw_artifact_hash(filename)
                for filename in sorted(INDEX_ARTIFACTS)
            }
            if index_manifest is not None
            else None
        )
        stored = self.store.read_manifest(
            expected_dataset_signature=self.dataset.dataset_signature,
            expected_corpus_signature=self.corpus.manifest.corpus_signature,
            expected_upstream_artifact_hashes=index_hashes,
        )
        base_fields = (
            "schema_version",
            "contract_version",
            "dataset_id",
            "dataset_version",
            "split",
            "dataset_signature",
            "corpus_signature",
            "chunker_name",
            "chunker_version",
            "chunk_size",
            "overlap",
        )
        if any(
            getattr(stored, field) != getattr(self._build_manifest_from_stored(stored), field)
            for field in base_fields
        ):
            raise ValueError("stored run manifest does not match experiment config")
        if index_manifest is not None:
            expected = self._build_manifest(index_manifest)
            index_fields = (
                "index_implementation",
                "index_implementation_version",
                "index_signature",
                "retriever",
            )
            if any(getattr(stored, field) != getattr(expected, field) for field in index_fields):
                raise ValueError("stored run manifest does not match experiment config")
            if strict and stored != expected:
                raise ValueError("stored run manifest does not match experiment config")
        self.manifest = stored

    def _build_manifest_from_stored(self, stored: RunManifest) -> RunManifest:
        return stored.model_copy(
            update={
                "dataset_id": self.dataset.manifest.dataset_id,
                "dataset_version": self.dataset.manifest.dataset_version,
                "split": self.dataset.manifest.split,
                "dataset_signature": self.dataset.dataset_signature,
                "corpus_signature": self.corpus.manifest.corpus_signature,
                "chunker_name": self.corpus.manifest.chunker_name,
                "chunker_version": self.corpus.manifest.chunker_version,
                "chunk_size": self.corpus.manifest.chunk_size,
                "overlap": self.corpus.manifest.overlap,
            }
        )

    def _stage_provenance(self, stage: str) -> ProducerProvenance:
        if stage == "retriever":
            module = self.config.retriever
            parameters: dict[str, object] = {"top_k": self.config.top_k}
        elif stage == "selector":
            module = self.config.selector
            parameters = {"max_selected": self.config.max_selected}
            # The SAME_SOURCE sidecar is resolved from the environment, not the config, so
            # its identity has to be recorded here or two runs against different sidecars
            # would be indistinguishable in the archived provenance.
            parameters.update(source_parent_provenance(self.config.selector))
        elif stage == "generator":
            module = self.config.generator
            parameters = {}
        elif stage == "pipeline":
            module = ModuleConfig(
                name="pipeline",
                parameters={
                    "retriever": self.config.retriever.model_dump(mode="json"),
                    "selector": self.config.selector.model_dump(mode="json"),
                    "generator": self.config.generator.model_dump(mode="json"),
                },
            )
            parameters = {
                "top_k": self.config.top_k,
                "max_selected": self.config.max_selected,
            }
        else:
            module = ModuleConfig(
                name="infrastructure",
                parameters={
                    "chunker_name": self.corpus.manifest.chunker_name,
                    "chunker_version": self.corpus.manifest.chunker_version,
                    "index_implementation": (
                        self.manifest.index_implementation if self.manifest else "unknown"
                    ),
                    "index_implementation_version": (
                        self.manifest.index_implementation_version
                        if self.manifest
                        else "unknown"
                    ),
                },
            )
            parameters = {}
        return ProducerProvenance(
            git_commit=self.git_commit,
            git_dirty=self.git_dirty,
            source_tree_signature=self.source_tree_signature,
            module=module,
            execution=ModuleConfig(name=stage, parameters=parameters),
        )

    def _read_jsonl(
        self,
        filename: str,
        model_type: type[ArtifactModelT],
        *,
        expected_stage: str,
        expected_upstream: tuple[str, ...],
    ) -> tuple[ArtifactModelT, ...]:
        return self.store.read_jsonl(
            filename,
            model_type,
            expected_dataset_signature=self.dataset.dataset_signature,
            expected_corpus_signature=self.corpus.manifest.corpus_signature,
            expected_producer=PRODUCER,
            expected_stage=expected_stage,
            expected_upstream_artifact_hashes=self._upstream_hashes(expected_upstream),
        )

    def _write_jsonl(
        self,
        filename: str,
        records: Iterable[BaseModel],
        *,
        stage: str,
        upstream: tuple[str, ...],
    ) -> None:
        self.store.write_jsonl(
            filename,
            records,
            producer=PRODUCER,
            stage=stage,
            producer_provenance=self._stage_provenance(stage),
            upstream_artifact_hashes=self._upstream_hashes(upstream),
        )

    def _write_json(
        self,
        filename: str,
        model: BaseModel,
        *,
        stage: str,
        upstream: tuple[str, ...],
    ) -> None:
        self.store.write_json(
            filename,
            model,
            producer=PRODUCER,
            stage=stage,
            producer_provenance=self._stage_provenance(stage),
            upstream_artifact_hashes=self._upstream_hashes(upstream),
        )

    def _upstream_hashes(self, filenames: tuple[str, ...]) -> dict[str, str]:
        return {filename: self.store.artifact_hash(filename) for filename in filenames}

    def _raw_artifact_hash(self, filename: str) -> str:
        try:
            content = (self.store.root / filename).read_bytes()
        except OSError as error:
            raise ValueError(f"unable to read upstream artifact {filename}: {error}") from error
        return sha256(content).hexdigest()

    @staticmethod
    def _with_metadata(artifacts: tuple[str, ...]) -> tuple[str, ...]:
        expanded: list[str] = []
        for filename in artifacts:
            expanded.append(filename)
            if filename not in INDEX_ARTIFACTS:
                expanded.append(metadata_filename(filename))
        return tuple(expanded)

    def _require_artifact(self, filename: str, upstream_command: str) -> None:
        if not (self.store.root / filename).is_file():
            raise ValueError(
                f"required artifact {filename} is missing; run {upstream_command} first"
            )

    def _ordered_gold_cases(self) -> tuple[GoldCase, ...]:
        by_query_id = {gold.query_id: gold for gold in self.dataset.gold_cases}
        missing = tuple(
            query.query_id for query in self.dataset.queries if query.query_id not in by_query_id
        )
        if missing:
            raise ValueError(f"missing gold case for query ID: {missing[0]}")
        return tuple(by_query_id[query.query_id] for query in self.dataset.queries)

    def _summary(
        self,
        command: Command,
        artifacts: tuple[str, ...],
    ) -> WorkflowSummary:
        return WorkflowSummary(
            command=command,
            output_directory=self.config.output_directory,
            query_count=len(self.dataset.queries),
            artifacts=artifacts,
        )
