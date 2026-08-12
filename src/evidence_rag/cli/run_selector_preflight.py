"""Run or independently verify the formal R004 dual-head resource preflight.

This command never saves a checkpoint and never constructs a Selector policy.  It verifies the
two write-once label bundles, freezes an ID-only 100+100 query sample, audits token truncation on
all train-modelval pairs, times an untrained two-head forward pass, then discards a short
forward/backward resource probe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import socket
import subprocess
import sys
import time
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

import evidence_rag.evaluation.selector_preflight as selector_preflight_module
import evidence_rag.materializer.selector_labels as selector_labels_module
import evidence_rag.selector.dual_head as dual_head_module
from evidence_rag.contracts.models import CandidateSet, Query
from evidence_rag.evaluation.selector_components import OUTPUT_FILES as COMPONENT_OUTPUT_FILES
from evidence_rag.evaluation.selector_preflight import (
    FORWARD_TRACE_FILE,
    PREFLIGHT_CANDIDATES_PER_QUERY,
    PREFLIGHT_FORWARD_PAIRS,
    PREFLIGHT_PROTOCOL_VERSION,
    PREFLIGHT_QUESTIONS_PER_DATASET,
    PREFLIGHT_SAMPLE_SEED,
    PREFLIGHT_TRAINING_MICROBATCHES,
    PREFLIGHT_TRAINING_WARMUP_MICROBATCHES,
    TOKEN_AUDIT_FILE,
    DatasetKind,
    PreflightSampleQuery,
    R004ResourcePreflightReport,
    TokenLengthAuditRow,
    TrainingGpuEstimate,
    UntrainedForwardTraceRow,
    build_resource_preflight_artifacts,
    estimate_seed_training_gpu_hours,
    freeze_resource_preflight_artifacts,
    nearest_rank_percentile,
    select_preflight_queries,
    summarize_token_lengths,
    verify_resource_preflight_artifacts,
)
from evidence_rag.infrastructure.datasets import DatasetManifest
from evidence_rag.materializer.selector_labels import (
    LABELS_FILE,
    SelectorLabelArtifacts,
    SelectorLabelRow,
    project_text_pair,
    text_pair_sha256,
    verify_selector_label_artifacts,
)
from evidence_rag.materializer.selector_labels import (
    OUTPUT_FILES as LABEL_OUTPUT_FILES,
)
from evidence_rag.materializer.selector_labels import (
    PROTOCOL_VERSION as LABEL_PROTOCOL_VERSION,
)
from evidence_rag.materializer.selector_pool import (
    CANDIDATE_FILE,
    SELECTOR_POOL_MANIFEST_FILE,
)
from evidence_rag.selector.dual_head import (
    MAX_LENGTH,
    audit_token_lengths,
    fingerprint_dual_head_model,
    load_dual_head_model,
    masked_dual_head_bce,
)


@dataclass(frozen=True)
class _FrozenConfig:
    raw: Mapping[str, object]
    model_id: str
    revision: str
    model_sha256: str
    snapshot_files: Mapping[str, str]
    max_length: int
    seed: int
    learning_rate: float
    epochs: int
    batch_size: int
    accumulation: int
    sample_seed: int
    questions_per_dataset: int
    candidates_per_query: int
    forward_pairs: int
    forward_warmup_batches: int
    training_probe_microbatches: int
    training_probe_warmup: int
    expected_label_counts: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class _Pair:
    dataset_kind: DatasetKind
    query_id: str
    evidence_id: str
    document_id: str
    retrieval_rank: int
    question: str
    candidate_text: str
    role: str
    component_id: str
    protect_label: int | None
    protect_mask: bool
    harm_label: int | None
    harm_mask: bool


@dataclass(frozen=True)
class _PreparedDataset:
    dataset_kind: DatasetKind
    pairs: tuple[_Pair, ...]
    query_role: Mapping[str, str]
    report: Mapping[str, object]

    @property
    def modelval_query_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                query_id for query_id, role in self.query_role.items() if role == "train-modelval"
            )
        )

    @property
    def modelval_pairs(self) -> tuple[_Pair, ...]:
        return tuple(pair for pair in self.pairs if pair.role == "train-modelval")

    @property
    def supervised_train_fit_pairs(self) -> tuple[_Pair, ...]:
        return tuple(
            pair
            for pair in self.pairs
            if pair.role == "train-fit" and (pair.protect_mask or pair.harm_mask)
        )


@dataclass(frozen=True)
class _DatasetArguments:
    kind: DatasetKind
    dataset_manifest: Path
    source_parent: Path
    candidate_pool: Path
    components_dir: Path
    labels_dir: Path
    assignment: Path | None = None
    provenance: Path | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run/verify formal R004 Selector preflight")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--niah-dataset-manifest", required=True, type=Path)
    parser.add_argument("--niah-source-parent", required=True, type=Path)
    parser.add_argument("--niah-candidate-pool", required=True, type=Path)
    parser.add_argument("--niah-components-dir", required=True, type=Path)
    parser.add_argument("--niah-assignment", required=True, type=Path)
    parser.add_argument("--niah-provenance", required=True, type=Path)
    parser.add_argument("--niah-labels-dir", required=True, type=Path)
    parser.add_argument("--twowiki-dataset-manifest", required=True, type=Path)
    parser.add_argument("--twowiki-source-parent", required=True, type=Path)
    parser.add_argument("--twowiki-candidate-pool", required=True, type=Path)
    parser.add_argument("--twowiki-components-dir", required=True, type=Path)
    parser.add_argument("--twowiki-labels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"config {label} must be a string-keyed table")
    return cast(Mapping[str, object], value)


def _text(section: Mapping[str, object], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config value {name} must be non-empty text")
    return value


def _integer(section: Mapping[str, object], name: str) -> int:
    value = section.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"config value {name} must be an integer")
    return value


def _number(section: Mapping[str, object], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise ValueError(f"config value {name} must be numeric")
    return float(value)


def _load_config(path: Path) -> _FrozenConfig:
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"unable to load R004 config {path}: {error}") from error
    model = _mapping(raw.get("model"), label="[model]")
    labels = _mapping(raw.get("labels"), label="[labels]")
    training = _mapping(raw.get("training"), label="[training]")
    preflight = _mapping(raw.get("preflight"), label="[preflight]")
    snapshot_files_raw = _mapping(model.get("snapshot_files"), label="[model.snapshot_files]")
    snapshot_files: dict[str, str] = {}
    for filename, digest in snapshot_files_raw.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"model snapshot digest for {filename!r} is invalid")
        snapshot_files[filename] = digest
    expected_raw = _mapping(
        preflight.get("expected_label_counts"), label="[preflight.expected_label_counts]"
    )
    expected: dict[str, dict[str, int]] = {}
    expected_count_fields = {
        "niah": {
            "queries",
            "candidate_rows",
            "train_fit_rows",
            "train_modelval_rows",
            "train_modelval_queries",
            "protect_0_harm_1",
            "protect_1_harm_0",
            "protect_1_harm_mask",
            "protect_mask_harm_mask",
        },
        "2wiki": {
            "queries",
            "candidate_rows",
            "train_fit_rows",
            "train_modelval_rows",
            "train_modelval_queries",
            "protect_1_harm_mask",
            "protect_mask_harm_mask",
        },
    }
    for kind in ("niah", "2wiki"):
        values = _mapping(expected_raw.get(kind), label=f"expected label counts {kind}")
        if set(values) != expected_count_fields[kind]:
            raise ValueError(f"expected label count fields differ for {kind}")
        expected[kind] = {}
        for key, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"expected label count {kind}.{key} must be non-negative int")
            expected[kind][key] = value

    seeds = training.get("seeds")
    if seeds != [13, 42, 73]:
        raise ValueError("training seeds must remain [13,42,73]")
    frozen = _FrozenConfig(
        raw=raw,
        model_id=_text(model, "model_id"),
        revision=_text(model, "revision"),
        model_sha256=_text(model, "model_sha256"),
        snapshot_files=snapshot_files,
        max_length=_integer(model, "max_length"),
        seed=13,
        learning_rate=_number(training, "learning_rate"),
        epochs=_integer(training, "epochs"),
        batch_size=_integer(training, "batch_size"),
        accumulation=_integer(training, "gradient_accumulation_steps"),
        sample_seed=_integer(preflight, "sample_seed"),
        questions_per_dataset=_integer(preflight, "questions_per_dataset"),
        candidates_per_query=_integer(preflight, "candidates_per_query"),
        forward_pairs=_integer(preflight, "forward_pairs"),
        forward_warmup_batches=_integer(preflight, "forward_warmup_batches"),
        training_probe_microbatches=_integer(preflight, "training_probe_microbatches"),
        training_probe_warmup=_integer(preflight, "training_probe_warmup_microbatches"),
        expected_label_counts=expected,
    )
    frozen_requirements = {
        "model.model_id": _text(model, "model_id") == "cross-encoder/nli-deberta-v3-base",
        "model.revision": _text(model, "revision") == "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
        "labels.protocol_version": _text(labels, "protocol_version") == LABEL_PROTOCOL_VERSION,
        "labels.unjudged_policy": _text(labels, "unjudged_policy") == "mask",
        "labels.provenance_is_model_input": labels.get("provenance_is_model_input") is False,
        "training.unjudged_hard_negatives_per_query": training.get(
            "unjudged_hard_negatives_per_query"
        )
        == 0,
        "training.negative_sampling": _text(training, "negative_sampling")
        == "active-mask-supervision-only",
        "training.niah_twowiki_ratio": _text(training, "niah_twowiki_ratio") == "1:1",
        "training.class_weighting": _text(training, "class_weighting")
        == "inverse-sqrt-frequency-per-source-head-active-classes-only",
        "training.zero_frequency_policy": _text(training, "zero_frequency_policy")
        == "masked-head-or-absent-class-contributes-no-loss-and-no-weight",
        "preflight.protocol": _text(preflight, "sample_protocol_version")
        == PREFLIGHT_PROTOCOL_VERSION,
        "preflight.role": _text(preflight, "sample_role") == "train-modelval",
        "preflight.training_role": _text(preflight, "training_probe_source_role") == "train-fit",
        "model.activation": _text(model, "output_activation") == "independent-sigmoid",
        "model.truncation": _text(model, "truncation") == "only_second",
        "model.dtype": _text(model, "torch_dtype") == "float32",
    }
    failed = sorted(name for name, passed in frozen_requirements.items() if not passed)
    if failed:
        raise ValueError(f"R004 config violates frozen requirements: {failed}")
    if model.get("input_fields") != ["question", "candidate_text"]:
        raise ValueError("model input_fields must be exactly question,candidate_text")
    expected_snapshot_files = {
        "added_tokens.json",
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "spm.model",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if set(frozen.snapshot_files) != expected_snapshot_files:
        raise ValueError("model snapshot must contain exactly the seven pre-registered files")
    if (
        frozen.max_length != MAX_LENGTH
        or not math.isclose(frozen.learning_rate, 0.00002, rel_tol=0.0, abs_tol=0.0)
        or frozen.epochs != 3
        or frozen.batch_size != 4
        or frozen.accumulation != 4
        or frozen.sample_seed != PREFLIGHT_SAMPLE_SEED
        or frozen.questions_per_dataset != PREFLIGHT_QUESTIONS_PER_DATASET
        or frozen.candidates_per_query != PREFLIGHT_CANDIDATES_PER_QUERY
        or frozen.forward_pairs != PREFLIGHT_FORWARD_PAIRS
        or frozen.forward_warmup_batches != 4
        or frozen.training_probe_microbatches != PREFLIGHT_TRAINING_MICROBATCHES
        or frozen.training_probe_warmup != PREFLIGHT_TRAINING_WARMUP_MICROBATCHES
    ):
        raise ValueError("R004 config differs from the pre-registered protocol constants")
    if frozen.model_sha256 != snapshot_files.get("model.safetensors"):
        raise ValueError("model_sha256 and snapshot model.safetensors pin disagree")
    return frozen


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_pin(path: Path, *, recorded_path: str | None = None) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"missing input file: {resolved}")
    return {
        "path": recorded_path if recorded_path is not None else str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _module_path(module: Any) -> Path:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str) or not value:
        raise ValueError(f"cannot resolve source file for module {module!r}")
    return Path(value).resolve()


def _audit_model_snapshot(path: Path, config: _FrozenConfig) -> dict[str, dict[str, object]]:
    root = Path(path).resolve()
    if root.name != config.revision:
        raise ValueError(f"model snapshot directory must be the frozen revision {config.revision}")
    actual_files = {item.name for item in root.iterdir() if item.is_file()}
    expected_files = set(config.snapshot_files)
    if actual_files != expected_files:
        raise ValueError(
            "model snapshot file set differs from config: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    pins: dict[str, dict[str, object]] = {}
    for filename in sorted(expected_files):
        file_path = root / filename
        pin = _file_pin(file_path)
        if pin["sha256"] != config.snapshot_files[filename]:
            raise ValueError(f"model snapshot SHA-256 mismatch for {filename}")
        pins[filename] = pin
    return pins


def _expected_model_identity(
    config: _FrozenConfig, snapshot_pins: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    snapshot_identity = hashlib.sha256(
        json.dumps(
            {name: pin["sha256"] for name, pin in sorted(snapshot_pins.items())},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "model_id": config.model_id,
        "revision": config.revision,
        "snapshot_identity_sha256": snapshot_identity,
        "snapshot_files": {name: dict(pin) for name, pin in sorted(snapshot_pins.items())},
    }


def _git_snapshot(repo_root: Path, *, required_tracked_paths: Sequence[Path]) -> dict[str, object]:
    root = Path(repo_root).resolve()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    actual_root = Path(git("rev-parse", "--show-toplevel")).resolve()
    if actual_root != root:
        raise ValueError(f"--repo-root is not the Git top-level directory: {root}")
    for raw_path in required_tracked_paths:
        resolved = Path(raw_path).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"formal R004 code/config is outside --repo-root: {resolved}"
            ) from error
        git("ls-files", "--error-unmatch", relative.as_posix())
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    dirty_lines = git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    if dirty_lines:
        raise ValueError(f"formal R004 requires a clean git worktree; found {dirty_lines[:5]}")
    return {"branch": branch, "commit": commit, "dirty": False}


def _read_queries(manifest_path: Path) -> dict[str, Query]:
    manifest = DatasetManifest.model_validate_json(Path(manifest_path).read_text(encoding="utf-8"))
    query_path = Path(manifest_path).parent / manifest.queries_file
    queries: dict[str, Query] = {}
    for number, line in enumerate(query_path.read_bytes().splitlines(), start=1):
        try:
            query = Query.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid query at {query_path}:{number}: {error}") from error
        if query.query_id in queries:
            raise ValueError(f"duplicate query ID in {query_path}: {query.query_id}")
        queries[query.query_id] = query
    return queries


def _read_candidate_sets(pool: Path, wanted: set[str]) -> dict[str, CandidateSet]:
    path = Path(pool) / CANDIDATE_FILE
    values: dict[str, CandidateSet] = {}
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        try:
            candidate_set = CandidateSet.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid candidate set at {path}:{number}: {error}") from error
        if candidate_set.query_id not in wanted:
            continue
        if candidate_set.query_id in values:
            raise ValueError(f"duplicate candidate query ID: {candidate_set.query_id}")
        values[candidate_set.query_id] = candidate_set
    if set(values) != wanted:
        raise ValueError(f"candidate/query keys differ; missing={sorted(wanted - set(values))[:5]}")
    return values


def _check_expected_label_counts(
    *, dataset_kind: DatasetKind, report: Mapping[str, object], expected: Mapping[str, int]
) -> None:
    counts = report.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError(f"{dataset_kind} label report has no counts")
    roles = counts.get("roles")
    pairs = counts.get("label_pairs")
    if not isinstance(roles, Mapping) or not isinstance(pairs, Mapping):
        raise ValueError(f"{dataset_kind} label report has invalid role/label counts")
    actual: dict[str, object] = {
        "queries": counts.get("queries"),
        "candidate_rows": counts.get("candidate_rows"),
        "train_fit_rows": roles.get("train-fit"),
        "train_modelval_rows": roles.get("train-modelval"),
        "protect_0_harm_1": pairs.get("protect=0,harm=1", 0),
        "protect_1_harm_0": pairs.get("protect=1,harm=0", 0),
        "protect_1_harm_mask": pairs.get("protect=1,harm=mask", 0),
        "protect_mask_harm_mask": pairs.get("protect=mask,harm=mask", 0),
    }
    for name, wanted in expected.items():
        if name == "train_modelval_queries":
            continue
        if actual.get(name) != wanted:
            raise ValueError(
                f"{dataset_kind} formal label count {name}={actual.get(name)}, expected {wanted}"
            )


def _prepare_dataset(
    arguments: _DatasetArguments,
    artifacts: SelectorLabelArtifacts,
    expected: Mapping[str, int],
) -> _PreparedDataset:
    _check_expected_label_counts(
        dataset_kind=arguments.kind, report=artifacts.report, expected=expected
    )
    label_rows: list[SelectorLabelRow] = []
    for number, line in enumerate(artifacts.files[LABELS_FILE].splitlines(), start=1):
        try:
            row = SelectorLabelRow.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid {arguments.kind} label row {number}: {error}") from error
        if row.dataset_kind != arguments.kind:
            raise ValueError(f"label dataset kind mismatch at row {number}")
        label_rows.append(row)
    label_queries = {row.query_id for row in label_rows}
    queries = _read_queries(arguments.dataset_manifest)
    missing_queries = label_queries - set(queries)
    if missing_queries:
        raise ValueError(f"labels reference unknown queries: {sorted(missing_queries)[:5]}")
    candidate_sets = _read_candidate_sets(arguments.candidate_pool, label_queries)
    query_role: dict[str, str] = {}
    rows_by_query: dict[str, int] = defaultdict(int)
    pairs: list[_Pair] = []
    seen: set[tuple[str, str]] = set()
    for row in label_rows:
        previous_role = query_role.setdefault(row.query_id, row.role)
        if previous_role != row.role:
            raise ValueError(f"query {row.query_id} crosses label roles")
        candidate_by_id = {
            item.evidence_id: item for item in candidate_sets[row.query_id].candidates
        }
        candidate = candidate_by_id.get(row.evidence_id)
        if candidate is None or candidate.document_id != row.document_id:
            raise ValueError(
                f"label/candidate identity mismatch for {row.query_id}/{row.evidence_id}"
            )
        projected = project_text_pair(
            question=queries[row.query_id].text, candidate_text=candidate.text
        )
        if text_pair_sha256(projected) != row.text_pair_sha256:
            raise ValueError(f"label/model text hash mismatch for {row.query_id}/{row.evidence_id}")
        key = (row.query_id, row.evidence_id)
        if key in seen:
            raise ValueError(f"duplicate label pair: {key}")
        seen.add(key)
        rows_by_query[row.query_id] += 1
        pairs.append(
            _Pair(
                dataset_kind=arguments.kind,
                query_id=row.query_id,
                evidence_id=row.evidence_id,
                document_id=row.document_id,
                retrieval_rank=candidate.retrieval_rank,
                question=projected.question,
                candidate_text=projected.candidate_text,
                role=row.role,
                component_id=row.component_id,
                protect_label=row.protect_label,
                protect_mask=row.protect_mask,
                harm_label=row.harm_label,
                harm_mask=row.harm_mask,
            )
        )
    bad_windows = sorted(
        query_id
        for query_id, count in rows_by_query.items()
        if count != PREFLIGHT_CANDIDATES_PER_QUERY
    )
    if bad_windows:
        raise ValueError(f"label windows are not exact Top20: {bad_windows[:5]}")
    ranks_by_query: dict[str, set[int]] = defaultdict(set)
    for pair in pairs:
        ranks_by_query[pair.query_id].add(pair.retrieval_rank)
    expected_ranks = set(range(1, PREFLIGHT_CANDIDATES_PER_QUERY + 1))
    bad_ranks = sorted(
        query_id for query_id, ranks in ranks_by_query.items() if ranks != expected_ranks
    )
    if bad_ranks:
        raise ValueError(f"label/candidate ranks are not exact 1..20: {bad_ranks[:5]}")
    modelval_queries = sum(role == "train-modelval" for role in query_role.values())
    if modelval_queries != expected["train_modelval_queries"]:
        raise ValueError(
            f"{arguments.kind} train-modelval queries={modelval_queries}, "
            f"expected {expected['train_modelval_queries']}"
        )
    return _PreparedDataset(
        dataset_kind=arguments.kind,
        pairs=tuple(
            sorted(
                pairs,
                key=lambda item: (item.query_id, item.retrieval_rank, item.evidence_id),
            )
        ),
        query_role=query_role,
        report=artifacts.report,
    )


def _sample_pairs(
    datasets: Mapping[DatasetKind, _PreparedDataset], sample: Sequence[PreflightSampleQuery]
) -> tuple[_Pair, ...]:
    by_key: dict[tuple[DatasetKind, str], list[_Pair]] = {
        (pair.dataset_kind, pair.query_id): []
        for dataset in datasets.values()
        for pair in dataset.modelval_pairs
    }
    for dataset in datasets.values():
        for pair in dataset.modelval_pairs:
            by_key[(pair.dataset_kind, pair.query_id)].append(pair)
    output: list[_Pair] = []
    for row in sample:
        pairs = sorted(
            by_key.get((row.dataset_kind, row.query_id), []),
            key=lambda item: (item.retrieval_rank, item.evidence_id),
        )
        if len(pairs) != PREFLIGHT_CANDIDATES_PER_QUERY:
            raise ValueError(f"sampled query {row.dataset_kind}/{row.query_id} is not exact Top20")
        output.extend(pairs)
    if len(output) != PREFLIGHT_FORWARD_PAIRS:
        raise ValueError(f"sample forward pairs={len(output)}, expected {PREFLIGHT_FORWARD_PAIRS}")
    return tuple(output)


def _token_audit_rows(tokenizer: Any, pairs: Sequence[_Pair]) -> tuple[TokenLengthAuditRow, ...]:
    rows: list[TokenLengthAuditRow] = []
    batch_size = 256
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        audits = audit_token_lengths(
            tokenizer,
            question=[pair.question for pair in batch],
            candidate_text=[pair.candidate_text for pair in batch],
        )
        for pair, audit in zip(batch, audits, strict=True):
            special = audit.raw_pair_tokens - audit.question_tokens - audit.candidate_tokens
            truncated_count = audit.raw_pair_tokens - audit.encoded_pair_tokens
            row = TokenLengthAuditRow(
                dataset_kind=pair.dataset_kind,
                query_id=pair.query_id,
                evidence_id=pair.evidence_id,
                retrieval_rank=pair.retrieval_rank,
                question_tokens=audit.question_tokens,
                candidate_tokens=audit.candidate_tokens,
                special_tokens=special,
                raw_pair_tokens=audit.raw_pair_tokens,
                encoded_tokens=audit.encoded_pair_tokens,
                truncated_candidate_tokens=truncated_count,
                truncated=audit.candidate_truncated,
            )
            if row.encoded_tokens > MAX_LENGTH:
                raise ValueError("tokenizer emitted a pair longer than max_length")
            rows.append(row)
    return tuple(rows)


def _sync_cuda(torch: Any) -> None:
    torch.cuda.synchronize()


def _forward_probe(
    *, model: Any, torch: Any, pairs: Sequence[_Pair], config: _FrozenConfig
) -> tuple[tuple[UntrainedForwardTraceRow, ...], dict[str, object]]:
    model.eval()
    warmup = pairs[: config.batch_size]
    with torch.inference_mode():
        for _ in range(config.forward_warmup_batches):
            model(
                question=[pair.question for pair in warmup],
                candidate_text=[pair.candidate_text for pair in warmup],
            )
        _sync_cuda(torch)
        torch.cuda.reset_peak_memory_stats()
        rows: list[UntrainedForwardTraceRow] = []
        batch_seconds: list[float] = []
        for start in range(0, len(pairs), config.batch_size):
            batch = pairs[start : start + config.batch_size]
            _sync_cuda(torch)
            started = time.perf_counter()
            output = model(
                question=[pair.question for pair in batch],
                candidate_text=[pair.candidate_text for pair in batch],
            )
            _sync_cuda(torch)
            batch_seconds.append(time.perf_counter() - started)
            protect = [float(value) for value in output.protect_scores.detach().cpu().tolist()]
            harm = [float(value) for value in output.harm_scores.detach().cpu().tolist()]
            for pair, protect_score, harm_score in zip(batch, protect, harm, strict=True):
                rows.append(
                    UntrainedForwardTraceRow(
                        dataset_kind=pair.dataset_kind,
                        query_id=pair.query_id,
                        evidence_id=pair.evidence_id,
                        retrieval_rank=pair.retrieval_rank,
                        protect_score=protect_score,
                        harm_score=harm_score,
                    )
                )
    wall = sum(batch_seconds)
    differences = [abs(row.protect_score - row.harm_score) for row in rows]
    if not rows or not all(math.isfinite(value) for value in differences):
        raise RuntimeError("untrained forward emitted non-finite scores")
    if max(differences) == 0.0:
        raise RuntimeError("protect and harm heads emitted identical scores for every pair")
    return tuple(rows), {
        "interpretation": "UNTRAINED_RESOURCE_PROBE_DO_NOT_INTERPRET",
        "queries": 2 * config.questions_per_dataset,
        "pairs": len(rows),
        "batch_size": config.batch_size,
        "warmup_batches": config.forward_warmup_batches,
        "timed_batches": len(batch_seconds),
        "batch_seconds": batch_seconds,
        "wall_time_seconds": wall,
        "pairs_per_second": len(rows) / wall,
        "queries_per_second": (2 * config.questions_per_dataset) / wall,
        "batch_seconds_mean": sum(batch_seconds) / len(batch_seconds),
        "batch_seconds_p50": nearest_rank_percentile(batch_seconds, 50),
        "batch_seconds_p95": nearest_rank_percentile(batch_seconds, 95),
        "max_head_score_difference": max(differences),
        "mean_head_score_difference": sum(differences) / len(differences),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "oom_encountered": False,
        "nonfinite_encountered": False,
    }


def _probe_order(pair: _Pair, *, seed: int) -> str:
    return hashlib.sha256(
        f"{pair.dataset_kind}\n{pair.query_id}\n{pair.evidence_id}\n{seed}".encode()
    ).hexdigest()


def _training_probe(
    *,
    model: Any,
    torch: Any,
    datasets: Mapping[DatasetKind, _PreparedDataset],
    config: _FrozenConfig,
) -> tuple[dict[str, object], TrainingGpuEstimate]:
    source_pairs = {
        kind: tuple(
            sorted(
                dataset.supervised_train_fit_pairs,
                key=lambda pair: (
                    _probe_order(pair, seed=config.seed),
                    pair.query_id,
                    pair.evidence_id,
                ),
            )
        )
        for kind, dataset in datasets.items()
    }
    if any(len(values) < config.batch_size for values in source_pairs.values()):
        raise ValueError("not enough active-mask train-fit pairs for the resource probe")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    offsets: dict[str, int] = {"niah": 0, "2wiki": 0}
    seconds: list[float] = []
    losses: list[float] = []
    protect_counts: list[int] = []
    harm_counts: list[int] = []
    optimizer_steps = 0
    for index in range(config.training_probe_microbatches):
        kind: DatasetKind = "niah" if index % 2 == 0 else "2wiki"
        values = source_pairs[kind]
        offset = offsets[kind]
        batch = tuple(values[(offset + inner) % len(values)] for inner in range(config.batch_size))
        offsets[kind] += config.batch_size
        if index == config.training_probe_warmup:
            _sync_cuda(torch)
            torch.cuda.reset_peak_memory_stats()
        _sync_cuda(torch)
        started = time.perf_counter()
        output = model(
            question=[pair.question for pair in batch],
            candidate_text=[pair.candidate_text for pair in batch],
        )
        nan = float("nan")
        protect_labels = torch.tensor(
            [nan if pair.protect_label is None else float(pair.protect_label) for pair in batch],
            device=output.protect_logits.device,
            dtype=output.protect_logits.dtype,
        )
        harm_labels = torch.tensor(
            [nan if pair.harm_label is None else float(pair.harm_label) for pair in batch],
            device=output.harm_logits.device,
            dtype=output.harm_logits.dtype,
        )
        protect_mask = torch.tensor(
            [pair.protect_mask for pair in batch], device=output.protect_logits.device
        )
        harm_mask = torch.tensor(
            [pair.harm_mask for pair in batch], device=output.harm_logits.device
        )
        loss = masked_dual_head_bce(
            protect_logits=output.protect_logits,
            harm_logits=output.harm_logits,
            protect_labels=protect_labels,
            harm_labels=harm_labels,
            protect_mask=protect_mask,
            harm_mask=harm_mask,
        )
        loss_value = float(loss.total.detach().cpu().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite ephemeral training loss at microbatch {index + 1}")
        (loss.total / config.accumulation).backward()
        if (index + 1) % config.accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        _sync_cuda(torch)
        seconds.append(time.perf_counter() - started)
        losses.append(loss_value)
        protect_counts.append(loss.protect_count)
        harm_counts.append(loss.harm_count)
    if optimizer_steps != config.training_probe_microbatches // config.accumulation:
        raise AssertionError("ephemeral optimizer-step accounting failed")
    estimate = estimate_seed_training_gpu_hours(
        niah_supervised_pairs=len(source_pairs["niah"]),
        twowiki_supervised_pairs=len(source_pairs["2wiki"]),
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.accumulation,
        epochs=config.epochs,
        microbatch_seconds=seconds,
        warmup_microbatches=config.training_probe_warmup,
    )
    report = {
        "interpretation": "EPHEMERAL_RESOURCE_PROBE_NO_CHECKPOINT",
        "source_role": "train-fit",
        "pair_order": "ascending-sha256(dataset-kind,newline,query-id,newline,evidence-id,newline,seed)",
        "loss": "unweighted independent masked BCE for resource measurement only",
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.accumulation,
        "niah_supervised_pairs": len(source_pairs["niah"]),
        "twowiki_supervised_pairs": len(source_pairs["2wiki"]),
        "microbatches": len(seconds),
        "warmup_microbatches": config.training_probe_warmup,
        "timed_microbatches": len(seconds) - config.training_probe_warmup,
        "microbatch_seconds": seconds,
        "losses": losses,
        "protect_effective_labels": protect_counts,
        "harm_effective_labels": harm_counts,
        "optimizer_steps": optimizer_steps,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "checkpoint": "NOT_APPLICABLE_EPHEMERAL_MODEL_DISCARDED",
        "oom_encountered": False,
        "nonfinite_encountered": False,
    }
    return report, estimate


def _label_artifacts(arguments: _DatasetArguments) -> SelectorLabelArtifacts:
    return verify_selector_label_artifacts(
        output_directory=arguments.labels_dir,
        dataset_kind=arguments.kind,
        dataset_manifest_path=arguments.dataset_manifest,
        source_parent_path=arguments.source_parent,
        candidate_pool_path=arguments.candidate_pool,
        component_directory=arguments.components_dir,
        assignment_path=arguments.assignment,
        provenance_path=arguments.provenance,
    )


def _input_pins(
    *,
    config_path: Path,
    snapshot_pins: Mapping[str, Mapping[str, object]],
    dataset_arguments: Sequence[_DatasetArguments],
    local_artifact_root: Path,
) -> dict[str, Mapping[str, object]]:
    artifact_root = Path(local_artifact_root).resolve()
    pins: dict[str, Mapping[str, object]] = {"config": _file_pin(config_path)}
    for filename, pin in snapshot_pins.items():
        pins[f"model_snapshot/{filename}"] = pin
    for arguments in dataset_arguments:
        prefix = arguments.kind
        paths = {
            "dataset_manifest": arguments.dataset_manifest,
            "source_parent": arguments.source_parent,
            "candidate_pool": arguments.candidate_pool / CANDIDATE_FILE,
            "pool_manifest": arguments.candidate_pool / SELECTOR_POOL_MANIFEST_FILE,
            "assignment": arguments.assignment,
            "provenance": arguments.provenance,
        }
        for name, path in paths.items():
            if path is not None:
                pins[f"{prefix}/{name}"] = _file_pin(path)
        for filename in LABEL_OUTPUT_FILES:
            label_path = (arguments.labels_dir / filename).resolve()
            try:
                recorded = label_path.relative_to(artifact_root).as_posix()
            except ValueError as error:
                raise ValueError(
                    f"R004 label artifact is outside the atomic run root: {label_path}"
                ) from error
            pins[f"{prefix}/labels/{filename}"] = _file_pin(label_path, recorded_path=recorded)
        for filename in COMPONENT_OUTPUT_FILES:
            pins[f"{prefix}/components/{filename}"] = _file_pin(arguments.components_dir / filename)
    return pins


def _dataset_arguments(arguments: argparse.Namespace) -> tuple[_DatasetArguments, ...]:
    datasets = (
        _DatasetArguments(
            kind="niah",
            dataset_manifest=arguments.niah_dataset_manifest,
            source_parent=arguments.niah_source_parent,
            candidate_pool=arguments.niah_candidate_pool,
            components_dir=arguments.niah_components_dir,
            labels_dir=arguments.niah_labels_dir,
            assignment=arguments.niah_assignment,
            provenance=arguments.niah_provenance,
        ),
        _DatasetArguments(
            kind="2wiki",
            dataset_manifest=arguments.twowiki_dataset_manifest,
            source_parent=arguments.twowiki_source_parent,
            candidate_pool=arguments.twowiki_candidate_pool,
            components_dir=arguments.twowiki_components_dir,
            labels_dir=arguments.twowiki_labels_dir,
        ),
    )
    forbidden_words = ("sealed", "heldout")
    for dataset in datasets:
        paths = (
            dataset.dataset_manifest,
            dataset.source_parent,
            dataset.candidate_pool,
            dataset.components_dir,
            dataset.labels_dir,
            dataset.assignment,
            dataset.provenance,
        )
        for path in paths:
            if path is None:
                continue
            raw = str(path).lower()
            resolved = str(Path(path).resolve(strict=False)).lower()
            if any(word in raw or word in resolved for word in forbidden_words):
                raise ValueError("R004 refuses sealed/heldout data paths")
    return datasets


def _load_tokenizer(snapshot: Path) -> Any:
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    tokenizer.truncation_side = "right"
    return tokenizer


def _verify_only(
    *,
    arguments: argparse.Namespace,
    config: _FrozenConfig,
    datasets: Mapping[DatasetKind, _PreparedDataset],
    sample: Sequence[PreflightSampleQuery],
    sample_pairs: Sequence[_Pair],
    input_pins: Mapping[str, Mapping[str, object]],
    git: Mapping[str, object],
    model_identity: Mapping[str, object],
) -> None:
    verified = verify_resource_preflight_artifacts(
        arguments.output_dir,
        expected_sample=sample,
        expected_input_pins=input_pins,
        expected_git=git,
        expected_model=model_identity,
    )
    typed_report = R004ResourcePreflightReport.model_validate_json(
        json.dumps(verified.report, allow_nan=False, sort_keys=True)
    )
    expected_label_audit = {kind: dataset.report for kind, dataset in datasets.items()}
    if json.dumps(typed_report.label_audit, sort_keys=True) != json.dumps(
        expected_label_audit, sort_keys=True
    ):
        raise ValueError("label audit report differs from independently verified label bundles")
    actual_supervised = {
        kind: len(dataset.supervised_train_fit_pairs) for kind, dataset in datasets.items()
    }
    if (
        typed_report.training_probe.niah_supervised_pairs != actual_supervised["niah"]
        or typed_report.training_probe.twowiki_supervised_pairs != actual_supervised["2wiki"]
    ):
        raise ValueError("training probe pair counts differ from active-mask train-fit labels")
    estimate = typed_report.seed13_training_gpu_estimate
    if (
        estimate.epochs != config.epochs
        or estimate.batch_size != config.batch_size
        or estimate.gradient_accumulation_steps != config.accumulation
    ):
        raise ValueError("GPU estimate differs from the frozen training config")
    dataset_order: tuple[DatasetKind, ...] = ("niah", "2wiki")
    all_modelval_pairs = tuple(
        pair for kind in dataset_order for pair in datasets[kind].modelval_pairs
    )
    tokenizer = _load_tokenizer(arguments.model_snapshot)
    expected_tokens = _token_audit_rows(tokenizer, all_modelval_pairs)
    actual_tokens = tuple(
        TokenLengthAuditRow.model_validate_json(line)
        for line in verified.files[TOKEN_AUDIT_FILE].splitlines()
    )
    if actual_tokens != expected_tokens:
        raise ValueError("token audit differs from an independent tokenizer-only recomputation")
    expected_forward_keys = {
        (pair.dataset_kind, pair.query_id, pair.evidence_id, pair.retrieval_rank)
        for pair in sample_pairs
    }
    actual_forward = tuple(
        UntrainedForwardTraceRow.model_validate_json(line)
        for line in verified.files[FORWARD_TRACE_FILE].splitlines()
    )
    actual_forward_keys = {
        (row.dataset_kind, row.query_id, row.evidence_id, row.retrieval_rank)
        for row in actual_forward
    }
    if actual_forward_keys != expected_forward_keys or len(actual_forward) != len(
        expected_forward_keys
    ):
        raise ValueError("forward trace keys differ from the frozen 200-query Top20 sample")
    if config.forward_pairs != len(actual_forward):
        raise ValueError("forward trace count differs from the frozen config")


def _run(arguments: argparse.Namespace) -> dict[str, object]:
    started = datetime.now(UTC)
    started_clock = time.perf_counter()
    runtime_log: list[dict[str, object]] = [
        {"event": "start", "time_utc": started.isoformat(), "run_id": "R004"}
    ]
    config = _load_config(arguments.config)
    git = _git_snapshot(
        arguments.repo_root,
        required_tracked_paths=(
            Path(__file__),
            _module_path(selector_preflight_module),
            _module_path(selector_labels_module),
            _module_path(dual_head_module),
            arguments.config,
        ),
    )
    snapshot_pins = _audit_model_snapshot(arguments.model_snapshot, config)
    model_identity = _expected_model_identity(config, snapshot_pins)
    dataset_arguments = _dataset_arguments(arguments)
    label_artifacts = {item.kind: _label_artifacts(item) for item in dataset_arguments}
    runtime_log.append({"event": "labels-verified", "time_utc": datetime.now(UTC).isoformat()})
    datasets = {
        item.kind: _prepare_dataset(
            item,
            label_artifacts[item.kind],
            config.expected_label_counts[item.kind],
        )
        for item in dataset_arguments
    }
    sample = select_preflight_queries(
        {kind: dataset.modelval_query_ids for kind, dataset in datasets.items()},
        count_per_dataset=config.questions_per_dataset,
        seed=config.sample_seed,
    )
    sample_pairs = _sample_pairs(datasets, sample)
    dataset_order: tuple[DatasetKind, ...] = ("niah", "2wiki")
    all_modelval_pairs = tuple(
        pair for kind in dataset_order for pair in datasets[kind].modelval_pairs
    )
    if len(all_modelval_pairs) != 8060:
        raise ValueError(f"token audit pairs={len(all_modelval_pairs)}, expected 8060")
    input_pins = _input_pins(
        config_path=arguments.config,
        snapshot_pins=snapshot_pins,
        dataset_arguments=dataset_arguments,
        local_artifact_root=arguments.output_dir.parent,
    )
    if arguments.verify_only:
        _verify_only(
            arguments=arguments,
            config=config,
            datasets=datasets,
            sample=sample,
            sample_pairs=sample_pairs,
            input_pins=input_pins,
            git=git,
            model_identity=model_identity,
        )
        return {
            "action": "verified",
            "status": "PASS",
            "output_dir": str(arguments.output_dir),
            "sample_queries": len(sample),
            "token_audit_pairs": len(all_modelval_pairs),
            "forward_pairs": len(sample_pairs),
        }

    if not arguments.device.startswith("cuda"):
        raise ValueError("formal R004 resource measurements require one CUDA device")
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the formal R004 resource preflight")
    torch.cuda.set_device(torch.device(arguments.device))
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = load_dual_head_model(
        str(Path(arguments.model_snapshot).resolve()),
        revision=config.revision,
        local_files_only=True,
        device=arguments.device,
    )
    fingerprint = fingerprint_dual_head_model(model)
    if model.protect_head is model.harm_head:
        raise RuntimeError("protect and harm heads unexpectedly share one module")
    if any(
        left.data_ptr() == right.data_ptr()
        for left, right in zip(
            model.protect_head.parameters(), model.harm_head.parameters(), strict=True
        )
    ):
        raise RuntimeError("protect and harm heads unexpectedly share parameter storage")
    token_rows = _token_audit_rows(model.tokenizer, all_modelval_pairs)
    if len(token_rows) != 8060:
        raise AssertionError("token audit cardinality changed")
    runtime_log.append({"event": "token-audit-complete", "time_utc": datetime.now(UTC).isoformat()})
    forward_rows, forward_report = _forward_probe(
        model=model, torch=torch, pairs=sample_pairs, config=config
    )
    runtime_log.append({"event": "forward-complete", "time_utc": datetime.now(UTC).isoformat()})
    training_report, estimate = _training_probe(
        model=model, torch=torch, datasets=datasets, config=config
    )
    runtime_log.append(
        {"event": "ephemeral-training-probe-complete", "time_utc": datetime.now(UTC).isoformat()}
    )
    properties = torch.cuda.get_device_properties(torch.device(arguments.device))
    finished = datetime.now(UTC)
    peak_memory = max(
        cast(int, forward_report["peak_memory_allocated_bytes"]),
        cast(int, training_report["peak_memory_allocated_bytes"]),
    )
    token_summary = summarize_token_lengths(token_rows)
    per_dataset_tokens = {
        kind: asdict(
            summarize_token_lengths([row for row in token_rows if row.dataset_kind == kind])
        )
        for kind in dataset_order
    }
    report: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": "R004",
        "stage": "label-audit-and-200q-preflight",
        "status": "PASS",
        "interpretation": "RESOURCE_AND_LABEL_FEASIBILITY_ONLY_NOT_SELECTOR_EFFECT",
        "git": git,
        "config_sha256": _sha256(arguments.config),
        "model": {
            **model_identity,
            "random_head_seed": config.seed,
            "initialized_full_state_fingerprint": fingerprint.to_dict(),
            "independent_linear_heads": True,
            "activation": "independent-sigmoid",
            "checkpoint": "NOT_APPLICABLE_EPHEMERAL_MODEL_DISCARDED",
        },
        "label_audit": {kind: dataset.report for kind, dataset in datasets.items()},
        "sampling": {
            "protocol_version": PREFLIGHT_PROTOCOL_VERSION,
            "seed": config.sample_seed,
            "role": "train-modelval",
            "queries_per_dataset": config.questions_per_dataset,
            "sample_queries": len(sample),
            "candidates_per_query": config.candidates_per_query,
            "forward_pairs": len(sample_pairs),
            "selection_uses_labels_or_lengths": False,
        },
        "token_audit": {
            "scope": "all-train-modelval-top20",
            "queries": sum(len(dataset.modelval_query_ids) for dataset in datasets.values()),
            "pairs": len(token_rows),
            "max_length": config.max_length,
            "truncation": "only_second",
            "overall": asdict(token_summary),
            "by_dataset": per_dataset_tokens,
        },
        "forward_probe": forward_report,
        "training_probe": training_report,
        "seed13_training_gpu_estimate": estimate.model_dump(mode="json"),
        "device": {
            "kind": "cuda",
            "requested": arguments.device,
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": str(torch.version.cuda),
            "precision": "float32",
            "peak_memory_allocated_bytes": peak_memory,
        },
        "runtime": {
            "host": socket.gethostname(),
            "python_version": platform.python_version(),
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "wall_time_seconds": time.perf_counter() - started_clock,
            "command": sys.argv,
        },
        "boundaries": {
            "model_input_fields": ["question", "candidate_text"],
            "provenance_is_model_input": False,
            "unjudged_as_negative": False,
            "selector_policy_constructed": False,
            "checkpoint_written": False,
            "sealed_or_heldout_accessed": False,
            "production_default_changed": False,
        },
    }
    runtime_log.append({"event": "pass", "time_utc": finished.isoformat()})
    artifacts = build_resource_preflight_artifacts(
        sample=sample,
        token_audit=token_rows,
        forward_trace=forward_rows,
        report=report,
        runtime_log=runtime_log,
        input_pins=input_pins,
        expected_git=git,
        expected_model=model_identity,
    )
    freeze_resource_preflight_artifacts(arguments.output_dir, artifacts)
    return {
        "action": "frozen",
        "status": "PASS",
        "output_dir": str(arguments.output_dir),
        "sample_queries": len(sample),
        "token_audit_pairs": len(token_rows),
        "forward_pairs": len(forward_rows),
        "forward_pairs_per_second": forward_report["pairs_per_second"],
        "peak_memory_allocated_bytes": peak_memory,
        "seed13_training_gpu_hours": estimate.point_gpu_hours,
        "seed13_training_gpu_hours_conservative": estimate.conservative_gpu_hours,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = _run(arguments)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
