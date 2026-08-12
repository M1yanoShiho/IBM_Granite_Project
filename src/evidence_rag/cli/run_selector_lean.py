"""Lean v3 Selector entry points.

The command intentionally has only three ordinary stages: fit the two fixed checkpoints,
calibrate one conservative policy on development data, and evaluate that already-frozen policy.
It does not implement a registry, lock protocol, or reveal state machine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from evidence_rag.cli.finalize_selector_r004 import finalize_selector_r004
from evidence_rag.contracts.models import CandidateSet, Query, QueryChecklist, SelectedEvidenceSet
from evidence_rag.evaluation.scoring import answer_match
from evidence_rag.evaluation.selector_artifacts import (
    SelectorRunStatus,
    verify_file_pin,
)
from evidence_rag.evaluation.selector_lean import (
    DevelopmentCandidateResult,
    EvaluationProjection,
    FrozenPolicy,
    PolicyQueryResult,
    apply_seed_policy,
    build_evaluation_projection,
    build_policy_candidates,
    combine_development_metrics,
    combine_projection_sha256,
    evaluate_seed_policy,
    evidence_inference,
    freeze_policy,
    macro_answer_inference,
    require_frozen_final_policy,
    select_development_policy,
    thresholds_from_train_scores,
)
from evidence_rag.evaluation.selector_sanity import compute_class_weights
from evidence_rag.generator.granite import CITATION_RAG_PROMPT, GraniteGenerator
from evidence_rag.infrastructure.datasets import DatasetManifest
from evidence_rag.materializer.selector_labels import (
    SelectorLabelRow,
    project_text_pair,
    text_pair_sha256,
)
from evidence_rag.selector.dual_head import (
    load_dual_head_checkpoint,
    masked_dual_head_bce,
    save_dual_head_checkpoint,
)
from evidence_rag.selector.models import CandidateRiskScore
from evidence_rag.selector.nli_dual_head import (
    NLI_ARCHITECTURE_VERSION,
    NLI_LABEL_MAP,
    load_nli_dual_head_model,
    pairwise_logistic_loss,
)

DatasetKind = Literal["niah", "2wiki"]
Variant = Literal["NLI-base", "NLI-pair"]
HeadName = Literal["protect", "harm"]
BinaryLabel = Literal[0, 1]
ClassWeightKey = tuple[DatasetKind, HeadName, BinaryLabel]

_SEEDS = (13, 42)
_EPOCHS = (1, 2, 3)
_DATASET_KINDS: tuple[DatasetKind, DatasetKind] = ("niah", "2wiki")
_MODEL_ID = "cross-encoder/nli-deberta-v3-base"
_MODEL_REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
_MODEL_WEIGHTS_SHA256 = "d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa"
_GENERATOR_ID = "ibm-granite/granite-4.1-3b"
_GENERATOR_REVISION = "c0650403e44e78ec0262dab1c90914c65b196c4e"
_GENERATOR_WEIGHT_SHA256 = {
    "895bf5f2d7c8b06ca902499567d3c3d9ed30061e4c5ad94bf8216286ca67e2fd",
    "de8c9efdaa6f669d595bda8b949213cba92ea69689fd2a27fbceed3d1ebeb2f7",
}
_GENERATOR_PROMPT_SHA256 = "691fb659d6f81a5df84c89e205de56858de4926ffc6aed9af3f24d7c5670b45f"
_TRAIN_SCORE_FILE = "train_scores.jsonl"
_CHECKPOINT_FILE = "model.safetensors"
_SIDECAR_FILE = "checkpoint_sidecar.json"
_BASE_STOP_SCHEMA = "selector-lean-base-stop-v1"


@dataclass(frozen=True, slots=True)
class LeanTrainingRow:
    """One train-fit candidate projected to the model's two permitted text fields."""

    dataset_kind: DatasetKind
    query_id: str
    evidence_id: str
    retrieval_rank: int
    question: str
    candidate_text: str
    protect_label: int | None
    protect_mask: bool
    harm_label: int | None
    harm_mask: bool
    role: Literal["train-fit"] = "train-fit"

    @property
    def identity(self) -> tuple[DatasetKind, str, str]:
        return self.dataset_kind, self.query_id, self.evidence_id

    @property
    def active(self) -> bool:
        return self.protect_mask or self.harm_mask


@dataclass(frozen=True, slots=True)
class LeanMicrobatch:
    """One source-homogeneous microbatch, optionally carrying one strict NIAH pair."""

    dataset_kind: DatasetKind
    rows: tuple[LeanTrainingRow, ...]
    strict_pair_positions: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class _LeanConfig:
    raw: Mapping[str, object]
    snapshot_files: Mapping[str, str]
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    batch_size: int
    accumulation: int
    expected_active_rows: Mapping[DatasetKind, int]
    expected_strict_pairs: int
    expected_microbatches_per_source: int
    expected_optimizer_steps_per_epoch: int
    pair_weights: Mapping[Variant, float]


@dataclass(frozen=True, slots=True)
class _FitData:
    active_by_dataset: Mapping[DatasetKind, tuple[LeanTrainingRow, ...]]
    threshold_rows: tuple[LeanTrainingRow, ...]


def _table(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"config {label} must be a string-keyed table")
    return cast(Mapping[str, object], value)


def _integer(section: Mapping[str, object], name: str) -> int:
    value = section.get(name)
    if type(value) is not int:
        raise ValueError(f"config {name} must be an integer")
    return value


def _number(section: Mapping[str, object], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"config {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"config {name} must be finite")
    return result


def _load_lean_config(path: Path) -> _LeanConfig:
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"unable to load Lean v3 config {path}: {error}") from error
    model = _table(raw.get("model"), label="[model]")
    label_map = _table(model.get("label_map"), label="[model.label_map]")
    snapshot_raw = _table(model.get("snapshot_files"), label="[model.snapshot_files]")
    labels = _table(raw.get("labels"), label="[labels]")
    data = _table(raw.get("data"), label="[data]")
    training = _table(raw.get("training"), label="[training]")
    expected = _table(training.get("expected"), label="[training.expected]")
    pair = _table(raw.get("pair_objective"), label="[pair_objective]")
    selection = _table(raw.get("selection"), label="[selection]")
    generator = _table(raw.get("generator"), label="[generator]")
    generator_weights = generator.get("weight_shard_sha256")

    exact = {
        "schema_version": raw.get("schema_version") == "1.0",
        "experiment_id": raw.get("experiment_id") == "selector-r005ab-v3-lean",
        "architecture": model.get("architecture_version") == NLI_ARCHITECTURE_VERSION,
        "model_id": model.get("model_id") == _MODEL_ID,
        "revision": model.get("revision") == _MODEL_REVISION,
        "weights": model.get("weights_sha256") == _MODEL_WEIGHTS_SHA256,
        "max_length": model.get("max_length") == 512,
        "truncation": model.get("truncation") == "only_second",
        "input_fields": model.get("input_fields") == ["question", "candidate_text"],
        "label_map": dict(label_map) == dict(NLI_LABEL_MAP),
        "training_role": labels.get("training_role") == "train-fit",
        "development_role": data.get("development_role") == "crc-calibration",
        "final_role": data.get("final_role") == "decision-dev",
        "seeds": training.get("seeds") == [13, 42],
        "optimizer": training.get("optimizer") == "AdamW",
        "learning_rate": training.get("learning_rate") == 0.00002,
        "weight_decay": training.get("weight_decay") == 0.01,
        "adam_beta1": training.get("adam_beta1") == 0.9,
        "adam_beta2": training.get("adam_beta2") == 0.999,
        "adam_epsilon": training.get("adam_epsilon") == 0.00000001,
        "epochs": training.get("epochs") == 3,
        "batch_size": training.get("batch_size") == 4,
        "accumulation": training.get("gradient_accumulation_steps") == 4,
        "scheduler": training.get("scheduler") == "none",
        "gradient_clipping": training.get("gradient_clipping") == "none",
        "niah_active_rows": expected.get("niah_active_rows") == 5312,
        "twowiki_active_rows": expected.get("twowiki_active_rows") == 5311,
        "niah_strict_pairs": expected.get("niah_strict_pairs") == 870,
        "microbatches_per_source": expected.get("microbatches_per_source_per_epoch") == 1328,
        "total_microbatches": expected.get("total_microbatches_per_epoch") == 2656,
        "optimizer_steps": expected.get("optimizer_steps_per_epoch") == 664,
        "nli_base_weight": pair.get("nli_base_weight") == 0.0,
        "nli_pair_weight": pair.get("nli_pair_weight") == 0.5,
        "quantiles": selection.get("quantiles") == [0.995, 0.99, 0.975, 0.95],
        "caps": selection.get("candidate_caps") == [1, 2],
        "safe_score": selection.get("safe_score") == "min(harm_score,1-protect_score)",
        "generator_id": generator.get("model_id") == _GENERATOR_ID,
        "generator_revision": generator.get("revision") == _GENERATOR_REVISION,
        "generator_weights": isinstance(generator_weights, list)
        and set(cast(list[str], generator_weights)) == _GENERATOR_WEIGHT_SHA256,
        "generator_prompt": generator.get("prompt_utf8_sha256") == _GENERATOR_PROMPT_SHA256,
        "generator_tokens": generator.get("max_new_tokens") == 32,
        "generator_temperature": generator.get("temperature") == 0.0,
        "generator_top_p": generator.get("top_p") == 1.0,
        "generator_sample": generator.get("do_sample") is False,
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"Lean v3 config differs from the frozen method: {failed}")

    snapshot_files: dict[str, str] = {}
    for filename, digest in snapshot_raw.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid model snapshot digest for {filename!r}")
        snapshot_files[filename] = digest
    if snapshot_files.get("model.safetensors") != _MODEL_WEIGHTS_SHA256:
        raise ValueError("model weight and snapshot-file digests disagree")
    return _LeanConfig(
        raw=raw,
        snapshot_files=snapshot_files,
        learning_rate=_number(training, "learning_rate"),
        weight_decay=_number(training, "weight_decay"),
        beta1=_number(training, "adam_beta1"),
        beta2=_number(training, "adam_beta2"),
        epsilon=_number(training, "adam_epsilon"),
        batch_size=_integer(training, "batch_size"),
        accumulation=_integer(training, "gradient_accumulation_steps"),
        expected_active_rows={
            "niah": _integer(expected, "niah_active_rows"),
            "2wiki": _integer(expected, "twowiki_active_rows"),
        },
        expected_strict_pairs=_integer(expected, "niah_strict_pairs"),
        expected_microbatches_per_source=_integer(expected, "microbatches_per_source_per_epoch"),
        expected_optimizer_steps_per_epoch=_integer(expected, "optimizer_steps_per_epoch"),
        pair_weights={
            "NLI-base": _number(pair, "nli_base_weight"),
            "NLI-pair": _number(pair, "nli_pair_weight"),
        },
    )


def _digest(*parts: object) -> str:
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def pair_digest(
    *, seed: int, epoch: int, query_id: str, clean_evidence_id: str, cf_evidence_id: str
) -> str:
    """Return the exact frozen digest for one strict clean/counterfactual pair."""

    return _digest(
        "selector-lean-v3",
        seed,
        epoch,
        "PAIR",
        query_id,
        clean_evidence_id,
        cf_evidence_id,
    )


def row_digest(*, seed: int, epoch: int, row: LeanTrainingRow) -> str:
    """Return the exact frozen digest for one active singleton."""

    return _digest(
        "selector-lean-v3",
        seed,
        epoch,
        row.dataset_kind,
        row.query_id,
        row.evidence_id,
    )


def _strict_niah_pairs(
    rows: Sequence[LeanTrainingRow],
) -> tuple[tuple[LeanTrainingRow, LeanTrainingRow], ...]:
    by_query: dict[str, list[LeanTrainingRow]] = defaultdict(list)
    for row in rows:
        by_query[row.query_id].append(row)
    output: list[tuple[LeanTrainingRow, LeanTrainingRow]] = []
    for query_id in sorted(by_query):
        values = by_query[query_id]
        clean = [
            row
            for row in values
            if row.protect_mask and row.harm_mask and row.protect_label == 1 and row.harm_label == 0
        ]
        counterfactual = [
            row
            for row in values
            if row.protect_mask and row.harm_mask and row.protect_label == 0 and row.harm_label == 1
        ]
        if len(clean) == 1 and len(counterfactual) == 1:
            output.append((clean[0], counterfactual[0]))
    return tuple(output)


def _validate_training_rows(rows: Sequence[LeanTrainingRow], *, dataset_kind: DatasetKind) -> None:
    if not rows:
        raise ValueError(f"{dataset_kind} has no active train-fit rows")
    identities: set[tuple[DatasetKind, str, str]] = set()
    for row in rows:
        if row.dataset_kind != dataset_kind:
            raise ValueError(f"{dataset_kind} batch input contains {row.dataset_kind} row")
        if row.role != "train-fit" or not row.active:
            raise ValueError("batch input may contain active train-fit rows only")
        for head in ("protect", "harm"):
            label = getattr(row, f"{head}_label")
            mask = getattr(row, f"{head}_mask")
            if mask != (label is not None) or (label is not None and label not in (0, 1)):
                raise ValueError(f"{head} label and mask disagree for {row.identity}")
        if row.identity in identities:
            raise ValueError(f"duplicate active training row: {row.identity}")
        identities.add(row.identity)
        if dataset_kind == "2wiki" and row.harm_mask:
            raise ValueError("2Wiki must not carry an active harm label")


def build_epoch_microbatches(
    *,
    niah_rows: Sequence[LeanTrainingRow],
    twowiki_rows: Sequence[LeanTrainingRow],
    seed: int,
    epoch: int,
    expected_active_rows: Mapping[DatasetKind, int] | None = None,
    expected_strict_pairs: int | None = None,
    expected_microbatches_per_source: int | None = None,
) -> tuple[LeanMicrobatch, ...]:
    """Pack the frozen pair-preserving schedule without repetition or source cycling."""

    if seed not in _SEEDS:
        raise ValueError(f"seed must be one of {_SEEDS}")
    if epoch not in _EPOCHS:
        raise ValueError(f"epoch must be one of {_EPOCHS}")
    _validate_training_rows(niah_rows, dataset_kind="niah")
    _validate_training_rows(twowiki_rows, dataset_kind="2wiki")
    if expected_active_rows is not None:
        actual = {"niah": len(niah_rows), "2wiki": len(twowiki_rows)}
        if actual != dict(expected_active_rows):
            raise ValueError(f"active-row counts differ from frozen contract: {actual}")

    strict_pairs = _strict_niah_pairs(niah_rows)
    if expected_strict_pairs is not None and len(strict_pairs) != expected_strict_pairs:
        raise ValueError(f"strict NIAH pairs={len(strict_pairs)}, expected {expected_strict_pairs}")
    ordered_pairs = sorted(
        strict_pairs,
        key=lambda value: (
            pair_digest(
                seed=seed,
                epoch=epoch,
                query_id=value[0].query_id,
                clean_evidence_id=value[0].evidence_id,
                cf_evidence_id=value[1].evidence_id,
            ),
            value[0].query_id,
            value[0].evidence_id,
            value[1].evidence_id,
        ),
    )
    paired_ids = {row.identity for pair in strict_pairs for row in pair}
    niah_singletons = sorted(
        (row for row in niah_rows if row.identity not in paired_ids),
        key=lambda row: (
            row_digest(seed=seed, epoch=epoch, row=row),
            row.query_id,
            row.evidence_id,
        ),
    )
    niah_batches: list[LeanMicrobatch] = []
    singleton_index = 0
    for clean, counterfactual in ordered_pairs:
        extras = tuple(niah_singletons[singleton_index : singleton_index + 2])
        singleton_index += len(extras)
        niah_batches.append(
            LeanMicrobatch(
                dataset_kind="niah",
                rows=(clean, counterfactual, *extras),
                strict_pair_positions=(0, 1),
            )
        )
    for start in range(singleton_index, len(niah_singletons), 4):
        niah_batches.append(
            LeanMicrobatch(dataset_kind="niah", rows=tuple(niah_singletons[start : start + 4]))
        )

    ordered_twowiki = sorted(
        twowiki_rows,
        key=lambda row: (
            row_digest(seed=seed, epoch=epoch, row=row),
            row.query_id,
            row.evidence_id,
        ),
    )
    twowiki_batches = [
        LeanMicrobatch(dataset_kind="2wiki", rows=tuple(ordered_twowiki[start : start + 4]))
        for start in range(0, len(ordered_twowiki), 4)
    ]
    if len(niah_batches) != len(twowiki_batches):
        raise ValueError(
            "frozen 1:1 schedule requires equal source microbatch counts: "
            f"NIAH={len(niah_batches)}, 2Wiki={len(twowiki_batches)}"
        )
    if (
        expected_microbatches_per_source is not None
        and len(niah_batches) != expected_microbatches_per_source
    ):
        raise ValueError(
            f"microbatches/source={len(niah_batches)}, expected {expected_microbatches_per_source}"
        )

    schedule = tuple(
        batch
        for index in range(len(niah_batches))
        for batch in (niah_batches[index], twowiki_batches[index])
    )
    scheduled = [row.identity for batch in schedule for row in batch.rows]
    expected_identities = {row.identity for row in (*niah_rows, *twowiki_rows)}
    if len(scheduled) != len(set(scheduled)) or set(scheduled) != expected_identities:
        raise AssertionError("epoch schedule repeated or omitted an active row")
    return schedule


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_identity(snapshot_files: Mapping[str, str]) -> str:
    payload = json.dumps(
        dict(sorted(snapshot_files.items())), separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_pin(path: Path) -> dict[str, object]:
    source = Path(path).resolve()
    if source.is_file():
        return {
            "kind": "file",
            "path": source.as_posix(),
            "sha256": _sha256_file(source),
        }
    if source.is_dir():
        files = tuple(sorted(item for item in source.rglob("*") if item.is_file()))
        if not files:
            raise ValueError(f"final input directory contains no files: {source}")
        return {
            "kind": "directory",
            "path": source.as_posix(),
            "files": {item.relative_to(source).as_posix(): _sha256_file(item) for item in files},
        }
    raise ValueError(f"final input path does not exist: {source}")


def final_input_sha256(
    *,
    niah_dataset_manifest: Path,
    niah_source_parent: Path,
    niah_candidate_pool: Path,
    niah_components_directory: Path,
    niah_assignment: Path,
    niah_provenance: Path,
    twowiki_dataset_manifest: Path,
    twowiki_source_parent: Path,
    twowiki_candidate_pool: Path,
    twowiki_components_directory: Path,
) -> str:
    """Hash the actual raw inputs that L003 will use, without reading any effect output."""

    paths = {
        "niah/dataset": Path(niah_dataset_manifest).resolve().parent,
        "niah/source_parent": niah_source_parent,
        "niah/candidate_pool": niah_candidate_pool,
        "niah/components": niah_components_directory,
        "niah/assignment": niah_assignment,
        "niah/provenance": niah_provenance,
        "2wiki/dataset": Path(twowiki_dataset_manifest).resolve().parent,
        "2wiki/source_parent": twowiki_source_parent,
        "2wiki/candidate_pool": twowiki_candidate_pool,
        "2wiki/components": twowiki_components_directory,
    }
    return _snapshot_identity(
        {
            label: hashlib.sha256(
                json.dumps(
                    _path_pin(path),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            for label, path in sorted(paths.items())
        }
    )


def _audit_generator_snapshot(path: Path, config: _LeanConfig) -> str:
    root = Path(path).resolve()
    generator = _table(config.raw.get("generator"), label="[generator]")
    frozen_files = _table(
        generator.get("snapshot_files"), label="[generator.snapshot_files]"
    )
    observed: dict[str, str] = {}
    for filename, expected in frozen_files.items():
        source = root / filename
        if not source.is_file() or not isinstance(expected, str):
            raise ValueError(f"pinned Generator snapshot file is missing: {filename}")
        digest = _sha256_file(source)
        if digest != expected:
            raise ValueError(f"Generator snapshot digest mismatch: {filename}")
        observed[filename] = digest
    weights = {
        _sha256_file(source): source.name for source in root.glob("*.safetensors") if source.is_file()
    }
    if set(weights) != _GENERATOR_WEIGHT_SHA256:
        raise ValueError("Generator weight shards differ from the frozen pair")
    observed.update({filename: digest for digest, filename in weights.items()})
    identity = {
        "model_id": _GENERATOR_ID,
        "revision": _GENERATOR_REVISION,
        "prompt_sha256": _GENERATOR_PROMPT_SHA256,
        "max_new_tokens": 32,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "files": dict(sorted(observed.items())),
    }
    return hashlib.sha256(
        json.dumps(identity, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _audit_model_snapshot(path: Path, config: _LeanConfig) -> str:
    root = Path(path).resolve()
    if not root.is_dir() or root.name != _MODEL_REVISION:
        raise ValueError(f"model snapshot directory must be the pinned revision {_MODEL_REVISION}")
    actual = {item.name for item in root.iterdir() if item.is_file()}
    expected = set(config.snapshot_files)
    if actual != expected:
        raise ValueError(
            f"model snapshot file set differs: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    for filename, wanted in config.snapshot_files.items():
        if _sha256_file(root / filename) != wanted:
            raise ValueError(f"model snapshot SHA-256 mismatch for {filename}")
    return _snapshot_identity(config.snapshot_files)


def _r004_fit_paths(r004_root: Path) -> Mapping[DatasetKind, Mapping[str, Path]]:
    manifest = finalize_selector_r004(r004_root, verify_only=True)
    if manifest.status is not SelectorRunStatus.PASS:
        raise ValueError("Lean fit requires a verified R004 PASS")
    index = {pin.label: pin for pin in manifest.inputs.all_pins()}
    if len(index) != len(manifest.inputs.all_pins()):
        raise ValueError("R004 input pin labels are not unique")
    root = Path(r004_root).resolve()

    def required(label: str) -> Path:
        try:
            pin = index[label]
        except KeyError as error:
            raise ValueError(f"R004 manifest omits {label!r}") from error
        path = verify_file_pin(pin, root=root).resolve()
        lowered = path.as_posix().lower()
        if "sealed" in lowered or "heldout" in lowered:
            raise ValueError(f"fit refuses sealed/heldout input: {path}")
        return path

    return {
        kind: {
            "dataset_manifest": required(f"{kind}/dataset_manifest"),
            "candidate_pool": required(f"{kind}/candidate_pool"),
            "labels": required(f"{kind}/labels/selector_labels.jsonl"),
        }
        for kind in _DATASET_KINDS
    }


def _read_queries(dataset_manifest_path: Path) -> Mapping[str, Query]:
    try:
        manifest = DatasetManifest.model_validate_json(
            Path(dataset_manifest_path).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        raise ValueError(f"invalid dataset manifest {dataset_manifest_path}: {error}") from error
    path = Path(dataset_manifest_path).parent / manifest.queries_file
    values: dict[str, Query] = {}
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        try:
            query = Query.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid query {path}:{number}: {error}") from error
        if query.query_id in values:
            raise ValueError(f"duplicate query ID: {query.query_id}")
        values[query.query_id] = query
    return values


def _read_candidate_sets(path: Path) -> Mapping[str, CandidateSet]:
    values: dict[str, CandidateSet] = {}
    for number, line in enumerate(Path(path).read_bytes().splitlines(), start=1):
        try:
            candidates = CandidateSet.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid candidate set {path}:{number}: {error}") from error
        if candidates.query_id in values:
            raise ValueError(f"duplicate candidate-set query ID: {candidates.query_id}")
        values[candidates.query_id] = candidates
    return values


def _load_dataset_fit_rows(
    *, dataset_kind: DatasetKind, paths: Mapping[str, Path]
) -> tuple[tuple[LeanTrainingRow, ...], tuple[LeanTrainingRow, ...]]:
    queries = _read_queries(paths["dataset_manifest"])
    candidates = _read_candidate_sets(paths["candidate_pool"])
    label_rows: list[SelectorLabelRow] = []
    for number, line in enumerate(paths["labels"].read_bytes().splitlines(), start=1):
        try:
            row = SelectorLabelRow.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(
                f"invalid Selector label {paths['labels']}:{number}: {error}"
            ) from error
        if row.dataset_kind == dataset_kind and row.role == "train-fit":
            label_rows.append(row)

    active: list[LeanTrainingRow] = []
    threshold: list[LeanTrainingRow] = []
    seen: set[tuple[str, str]] = set()
    for label in label_rows:
        key = (label.query_id, label.evidence_id)
        if key in seen:
            raise ValueError(f"duplicate train-fit label identity: {dataset_kind}/{key}")
        seen.add(key)
        try:
            query = queries[label.query_id]
            candidate_set = candidates[label.query_id]
        except KeyError as error:
            raise ValueError(
                f"label references missing query/candidate set: {dataset_kind}/{key}"
            ) from error
        by_id = {item.evidence_id: item for item in candidate_set.candidates}
        candidate = by_id.get(label.evidence_id)
        if candidate is None or candidate.document_id != label.document_id:
            raise ValueError(f"label/candidate identity mismatch: {dataset_kind}/{key}")
        projected = project_text_pair(question=query.text, candidate_text=candidate.text)
        if text_pair_sha256(projected) != label.text_pair_sha256:
            raise ValueError(f"label/model text hash mismatch: {dataset_kind}/{key}")
        projected_row = LeanTrainingRow(
            dataset_kind=dataset_kind,
            query_id=label.query_id,
            evidence_id=label.evidence_id,
            retrieval_rank=candidate.retrieval_rank,
            question=projected.question,
            candidate_text=projected.candidate_text,
            protect_label=label.protect_label,
            protect_mask=label.protect_mask,
            harm_label=label.harm_label,
            harm_mask=label.harm_mask,
        )
        if projected_row.active:
            active.append(projected_row)
        if candidate.retrieval_rank <= 10:
            threshold.append(projected_row)

    query_ranks: dict[str, set[int]] = defaultdict(set)
    for training_row in threshold:
        query_ranks[training_row.query_id].add(training_row.retrieval_rank)
    malformed = sorted(
        query_id for query_id, ranks in query_ranks.items() if ranks != set(range(1, 11))
    )
    if malformed:
        raise ValueError(f"train-fit threshold windows are not exact TopK10: {malformed[:5]}")
    return (
        tuple(sorted(active, key=lambda row: (row.query_id, row.retrieval_rank, row.evidence_id))),
        tuple(
            sorted(threshold, key=lambda row: (row.query_id, row.retrieval_rank, row.evidence_id))
        ),
    )


def _load_fit_data(r004_root: Path, config: _LeanConfig) -> _FitData:
    paths = _r004_fit_paths(r004_root)
    active: dict[DatasetKind, tuple[LeanTrainingRow, ...]] = {}
    threshold_rows: list[LeanTrainingRow] = []
    expected_queries: Mapping[DatasetKind, int] = {"niah": 920, "2wiki": 2700}
    for kind in _DATASET_KINDS:
        source_active, source_threshold = _load_dataset_fit_rows(
            dataset_kind=kind, paths=paths[kind]
        )
        active[kind] = source_active
        threshold_rows.extend(source_threshold)
        actual_queries = len({row.query_id for row in source_threshold})
        if actual_queries != expected_queries[kind] or len(source_threshold) != 10 * actual_queries:
            raise ValueError(
                f"{kind} train-fit TopK10 scope differs from frozen contract: "
                f"queries={actual_queries}, rows={len(source_threshold)}"
            )
    actual_active = {kind: len(rows) for kind, rows in active.items()}
    if actual_active != dict(config.expected_active_rows):
        raise ValueError(f"train-fit active counts differ from frozen contract: {actual_active}")
    return _FitData(active_by_dataset=active, threshold_rows=tuple(threshold_rows))


def _class_weight_table(
    active_by_dataset: Mapping[DatasetKind, Sequence[LeanTrainingRow]],
) -> tuple[Mapping[ClassWeightKey, float], tuple[Mapping[str, object], ...]]:
    records = [
        {
            "dataset_kind": row.dataset_kind,
            "query_id": row.query_id,
            "evidence_id": row.evidence_id,
            "role": row.role,
            "protect_label": row.protect_label,
            "protect_mask": row.protect_mask,
            "harm_label": row.harm_label,
            "harm_mask": row.harm_mask,
        }
        for kind in _DATASET_KINDS
        for row in active_by_dataset[kind]
    ]
    typed = compute_class_weights(records)
    values: dict[ClassWeightKey, float] = {
        (row.dataset_kind, row.head, row.class_label): float(row.normalized_weight) for row in typed
    }
    return values, tuple(row.model_dump(mode="json") for row in typed)


def _loss_for_batch(
    *,
    torch: Any,
    output: Any,
    batch: LeanMicrobatch,
    class_weights: Mapping[ClassWeightKey, float],
    pair_weight: float,
) -> tuple[Any, Any]:
    nan = float("nan")
    rows = batch.rows
    protect_labels = torch.tensor(
        [nan if row.protect_label is None else float(row.protect_label) for row in rows],
        device=output.protect_logits.device,
        dtype=output.protect_logits.dtype,
    )
    harm_labels = torch.tensor(
        [nan if row.harm_label is None else float(row.harm_label) for row in rows],
        device=output.harm_logits.device,
        dtype=output.harm_logits.dtype,
    )
    protect_mask = torch.tensor(
        [row.protect_mask for row in rows], device=output.protect_logits.device
    )
    harm_mask = torch.tensor([row.harm_mask for row in rows], device=output.harm_logits.device)
    protect_weights = torch.tensor(
        [
            class_weights[(row.dataset_kind, "protect", cast(BinaryLabel, row.protect_label))]
            if row.protect_mask
            else nan
            for row in rows
        ],
        device=output.protect_logits.device,
        dtype=output.protect_logits.dtype,
    )
    harm_weights = torch.tensor(
        [
            class_weights[(row.dataset_kind, "harm", cast(BinaryLabel, row.harm_label))]
            if row.harm_mask
            else nan
            for row in rows
        ],
        device=output.harm_logits.device,
        dtype=output.harm_logits.dtype,
    )
    bce = masked_dual_head_bce(
        protect_logits=output.protect_logits,
        harm_logits=output.harm_logits,
        protect_labels=protect_labels,
        harm_labels=harm_labels,
        protect_mask=protect_mask,
        harm_mask=harm_mask,
        protect_weights=protect_weights,
        harm_weights=harm_weights,
        weight_normalization="active_count",
    )
    pair_loss = (output.protect_logits.sum() + output.harm_logits.sum()) * 0.0
    if batch.strict_pair_positions is not None:
        clean_index, counterfactual_index = batch.strict_pair_positions
        pair_loss = pairwise_logistic_loss(
            protect_clean_logits=output.protect_logits[clean_index : clean_index + 1],
            protect_counterfactual_logits=output.protect_logits[
                counterfactual_index : counterfactual_index + 1
            ],
            harm_clean_logits=output.harm_logits[clean_index : clean_index + 1],
            harm_counterfactual_logits=output.harm_logits[
                counterfactual_index : counterfactual_index + 1
            ],
        )
    return bce, bce.total + pair_weight * pair_loss


def _module_state(module: Any) -> Mapping[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _module_changed(module: Any, initial: Mapping[str, Any], torch: Any) -> bool:
    current = module.state_dict()
    return set(current) == set(initial) and any(
        not bool(torch.equal(current[name].detach().cpu(), initial[name])) for name in current
    )


def _score_train_fit_topk10(
    *, model: Any, torch: Any, rows: Sequence[LeanTrainingRow], batch_size: int, seed: int
) -> tuple[Mapping[str, object], ...]:
    model.eval()
    output_rows: list[Mapping[str, object]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            output = model(
                question=[row.question for row in batch],
                candidate_text=[row.candidate_text for row in batch],
            )
            protects = output.protect_scores.detach().float().cpu().tolist()
            harms = output.harm_scores.detach().float().cpu().tolist()
            for row, protect_raw, harm_raw in zip(batch, protects, harms, strict=True):
                protect = float(protect_raw)
                harm = float(harm_raw)
                safe = min(harm, 1.0 - protect)
                if not all(
                    math.isfinite(value) and 0.0 <= value <= 1.0 for value in (protect, harm, safe)
                ):
                    raise RuntimeError("model emitted a non-finite or out-of-range train-fit score")
                output_rows.append(
                    {
                        "schema_version": "1.0",
                        "protocol_version": "selector-lean-v3",
                        "seed": seed,
                        "dataset_kind": row.dataset_kind,
                        "query_id": row.query_id,
                        "evidence_id": row.evidence_id,
                        "retrieval_rank": row.retrieval_rank,
                        "role": "train-fit",
                        "protect_score": protect,
                        "harm_score": harm,
                        "safe_score": safe,
                    }
                )
    return tuple(output_rows)


def _write_new(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite Lean Selector output: {destination}"
        ) from error


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Sequence[Mapping[str, object]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
            )
            for value in values
        )
        + ("\n" if values else "")
    ).encode("utf-8")


def fit_selector_lean(
    *,
    config_path: Path,
    r004_root: Path,
    model_snapshot: Path,
    variant: Variant,
    seed: int,
    output_directory: Path,
    device: str,
) -> Mapping[str, object]:
    """Run the real minimal Lean fit and emit one final-epoch checkpoint plus train scores."""

    config = _load_lean_config(config_path)
    if variant not in config.pair_weights:
        raise ValueError("variant must be NLI-base or NLI-pair")
    if seed not in _SEEDS:
        raise ValueError(f"seed must be one of {_SEEDS}")
    base_snapshot_sha256 = _audit_model_snapshot(model_snapshot, config)
    fit_data = _load_fit_data(r004_root, config)
    class_weights, class_weight_rows = _class_weight_table(fit_data.active_by_dataset)

    root = Path(output_directory).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"output path is not a directory: {root}")
    if root.is_dir() and any(root.iterdir()):
        raise ValueError(f"output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    torch = importlib.import_module("torch")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = load_nli_dual_head_model(
        str(Path(model_snapshot).resolve()),
        revision=_MODEL_REVISION,
        identity_model_id=_MODEL_ID,
        local_files_only=True,
        device=device,
    )
    protect_classifier = model.protect_classifier
    harm_classifier = model.harm_classifier
    if protect_classifier is harm_classifier:
        raise RuntimeError("protect and harm classifiers must be independent modules")
    initial_protect = _module_state(protect_classifier)
    initial_harm = _module_state(harm_classifier)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )

    epoch_rows: list[Mapping[str, object]] = []
    total_optimizer_steps = 0
    pair_weight = config.pair_weights[variant]
    for epoch in _EPOCHS:
        schedule = build_epoch_microbatches(
            niah_rows=fit_data.active_by_dataset["niah"],
            twowiki_rows=fit_data.active_by_dataset["2wiki"],
            seed=seed,
            epoch=epoch,
            expected_active_rows=config.expected_active_rows,
            expected_strict_pairs=config.expected_strict_pairs,
            expected_microbatches_per_source=config.expected_microbatches_per_source,
        )
        if len(schedule) % config.accumulation:
            raise ValueError("microbatch schedule does not fill complete accumulation windows")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_losses: list[float] = []
        protect_losses: list[float] = []
        harm_losses: list[float] = []
        epoch_steps = 0
        for microbatch_index, batch in enumerate(schedule, start=1):
            output = model(
                question=[row.question for row in batch.rows],
                candidate_text=[row.candidate_text for row in batch.rows],
            )
            bce, total = _loss_for_batch(
                torch=torch,
                output=output,
                batch=batch,
                class_weights=class_weights,
                pair_weight=pair_weight,
            )
            values = {
                "total": float(total.detach().cpu().item()),
                "protect": float(bce.protect.detach().cpu().item()),
                "harm": float(bce.harm.detach().cpu().item()),
            }
            active_values = [values["total"]]
            if bce.protect_count:
                active_values.append(values["protect"])
            if bce.harm_count:
                active_values.append(values["harm"])
            if not all(math.isfinite(value) for value in active_values):
                raise RuntimeError(f"non-finite active loss at epoch {epoch}")
            (total / config.accumulation).backward()
            total_losses.append(values["total"])
            protect_losses.append(values["protect"])
            harm_losses.append(values["harm"])
            if microbatch_index % config.accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_steps += 1
        if epoch_steps != config.expected_optimizer_steps_per_epoch:
            raise ValueError(
                f"optimizer steps in epoch {epoch}={epoch_steps}, expected "
                f"{config.expected_optimizer_steps_per_epoch}"
            )
        total_optimizer_steps += epoch_steps
        epoch_rows.append(
            {
                "epoch": epoch,
                "microbatches": len(schedule),
                "optimizer_steps": epoch_steps,
                "mean_total_loss": sum(total_losses) / len(total_losses),
                "mean_protect_loss": sum(protect_losses) / len(protect_losses),
                "mean_harm_loss": sum(harm_losses) / len(harm_losses),
            }
        )

    if not _module_changed(protect_classifier, initial_protect, torch):
        raise RuntimeError("protect classifier parameters did not change during fit")
    if not _module_changed(harm_classifier, initial_harm, torch):
        raise RuntimeError("harm classifier parameters did not change during fit")

    checkpoint_path = root / _CHECKPOINT_FILE
    fingerprint = save_dual_head_checkpoint(model, checkpoint_path)
    train_scores = _score_train_fit_topk10(
        model=model,
        torch=torch,
        rows=fit_data.threshold_rows,
        batch_size=config.batch_size,
        seed=seed,
    )
    score_path = root / _TRAIN_SCORE_FILE
    _write_new(score_path, _jsonl_bytes(train_scores))
    sidecar = {
        "schema_version": "1.0",
        "protocol_version": "selector-lean-v3",
        "variant": variant,
        "seed": seed,
        "architecture_version": NLI_ARCHITECTURE_VERSION,
        "label_map": dict(NLI_LABEL_MAP),
        "model_id": _MODEL_ID,
        "revision": _MODEL_REVISION,
        "base_model_weights_sha256": _MODEL_WEIGHTS_SHA256,
        "base_snapshot_sha256": base_snapshot_sha256,
        "config_sha256": _sha256_file(config_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_state_sha256": fingerprint.weights_sha256,
        "checkpoint_epoch": 3,
        "optimizer_steps": total_optimizer_steps,
        "active_rows": {kind: len(fit_data.active_by_dataset[kind]) for kind in _DATASET_KINDS},
        "train_score_rows": len(train_scores),
        "train_scores_sha256": _sha256_file(score_path),
        "class_weights": list(class_weight_rows),
        "epochs": epoch_rows,
        "health_gate": {
            "status": "PASS",
            "nonfinite_or_oom": False,
            "protect_classifier_changed": True,
            "harm_classifier_changed": True,
        },
    }
    sidecar_path = root / _SIDECAR_FILE
    _write_new(sidecar_path, _json_bytes(sidecar))
    return {
        "action": "fit-complete",
        "variant": variant,
        "seed": seed,
        "output_dir": root.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "train_scores": score_path.as_posix(),
        "train_scores_sha256": _sha256_file(score_path),
        "sidecar": sidecar_path.as_posix(),
    }


@dataclass(frozen=True, slots=True)
class _FitBundle:
    seed: int
    variant: Variant
    checkpoint_sha256: str
    safe_scores: tuple[float, ...]


def _read_json_object(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_fit_bundle(
    *,
    fit_directory: Path,
    expected_seed: int,
    expected_variant: Variant,
    config_path: Path,
) -> _FitBundle:
    root = Path(fit_directory).resolve()
    checkpoint = root / _CHECKPOINT_FILE
    scores_path = root / _TRAIN_SCORE_FILE
    sidecar_path = root / _SIDECAR_FILE
    if not root.is_dir():
        raise ValueError(f"fit directory does not exist: {root}")
    sidecar = _read_json_object(sidecar_path, label="fit sidecar")
    health = _table(sidecar.get("health_gate"), label="fit sidecar health_gate")
    exact = {
        "protocol_version": sidecar.get("protocol_version") == "selector-lean-v3",
        "seed": sidecar.get("seed") == expected_seed,
        "variant": sidecar.get("variant") == expected_variant,
        "architecture_version": sidecar.get("architecture_version") == NLI_ARCHITECTURE_VERSION,
        "label_map": sidecar.get("label_map") == dict(NLI_LABEL_MAP),
        "model_id": sidecar.get("model_id") == _MODEL_ID,
        "revision": sidecar.get("revision") == _MODEL_REVISION,
        "base_model_weights_sha256": sidecar.get("base_model_weights_sha256")
        == _MODEL_WEIGHTS_SHA256,
        "config_sha256": sidecar.get("config_sha256") == _sha256_file(config_path),
        "checkpoint_epoch": sidecar.get("checkpoint_epoch") == 3,
        "optimizer_steps": sidecar.get("optimizer_steps") == 3 * 664,
        "health_gate": health.get("status") == "PASS"
        and health.get("nonfinite_or_oom") is False
        and health.get("protect_classifier_changed") is True
        and health.get("harm_classifier_changed") is True,
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"fit sidecar differs from Lean v3 contract: {failed}")

    checkpoint_sha256 = _sha256_file(checkpoint)
    if checkpoint_sha256 != _require_digest(
        sidecar.get("checkpoint_sha256"), label="fit checkpoint_sha256"
    ):
        raise ValueError(f"fit checkpoint hash differs from sidecar: {checkpoint}")
    if _sha256_file(scores_path) != _require_digest(
        sidecar.get("train_scores_sha256"), label="fit train_scores_sha256"
    ):
        raise ValueError(f"fit train-score hash differs from sidecar: {scores_path}")

    safe_scores: list[float] = []
    identities: set[tuple[str, str, str]] = set()
    query_ranks: dict[tuple[str, str], set[int]] = defaultdict(set)
    dataset_counts = {"niah": 0, "2wiki": 0}
    for line_number, line in enumerate(scores_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid train score at {scores_path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"train score at {scores_path}:{line_number} must be an object")
        dataset_kind = row.get("dataset_kind")
        query_id = row.get("query_id")
        evidence_id = row.get("evidence_id")
        rank = row.get("retrieval_rank")
        if (
            row.get("role") != "train-fit"
            or row.get("seed") != expected_seed
            or dataset_kind not in _DATASET_KINDS
            or not isinstance(query_id, str)
            or not query_id
            or not isinstance(evidence_id, str)
            or not evidence_id
            or type(rank) is not int
            or not 1 <= rank <= 10
        ):
            raise ValueError(f"invalid train-score identity at {scores_path}:{line_number}")
        protect = _finite_number(row.get("protect_score"), label="train protect_score")
        harm = _finite_number(row.get("harm_score"), label="train harm_score")
        safe = _finite_number(row.get("safe_score"), label="train safe_score")
        if any(not 0.0 <= value <= 1.0 for value in (protect, harm, safe)):
            raise ValueError(f"invalid train score at {scores_path}:{line_number}")
        expected_safe = min(harm, 1.0 - protect)
        if not math.isclose(safe, expected_safe, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError(f"safe score formula differs at {scores_path}:{line_number}")
        identity = (dataset_kind, query_id, evidence_id)
        if identity in identities:
            raise ValueError(f"duplicate train-score identity: {identity}")
        identities.add(identity)
        query_ranks[(dataset_kind, query_id)].add(rank)
        dataset_counts[dataset_kind] += 1
        safe_scores.append(safe)
    malformed = [identity for identity, ranks in query_ranks.items() if ranks != set(range(1, 11))]
    if malformed:
        raise ValueError(f"train scores do not form exact TopK10 windows: {malformed[:5]}")
    expected_counts = {"niah": 9200, "2wiki": 27000}
    if dataset_counts != expected_counts or len(safe_scores) != 36200:
        raise ValueError(
            f"train-score scope differs from Lean v3: counts={dataset_counts}, "
            f"total={len(safe_scores)}"
        )
    if sidecar.get("train_score_rows") != len(safe_scores):
        raise ValueError("fit sidecar train_score_rows differs from train score file")
    return _FitBundle(
        seed=expected_seed,
        variant=expected_variant,
        checkpoint_sha256=checkpoint_sha256,
        safe_scores=tuple(safe_scores),
    )


def _evaluation_projections(
    *,
    role: Literal["crc-calibration", "decision-dev"],
    niah_dataset_manifest: Path,
    niah_source_parent: Path,
    niah_candidate_pool: Path,
    niah_components_directory: Path,
    niah_assignment: Path,
    niah_provenance: Path,
    twowiki_dataset_manifest: Path,
    twowiki_source_parent: Path,
    twowiki_candidate_pool: Path,
    twowiki_components_directory: Path,
) -> Mapping[DatasetKind, EvaluationProjection]:
    return {
        "niah": build_evaluation_projection(
            dataset_kind="niah",
            role=role,
            dataset_manifest_path=niah_dataset_manifest,
            source_parent_path=niah_source_parent,
            candidate_pool_path=niah_candidate_pool,
            component_directory=niah_components_directory,
            assignment_path=niah_assignment,
            provenance_path=niah_provenance,
        ),
        "2wiki": build_evaluation_projection(
            dataset_kind="2wiki",
            role=role,
            dataset_manifest_path=twowiki_dataset_manifest,
            source_parent_path=twowiki_source_parent,
            candidate_pool_path=twowiki_candidate_pool,
            component_directory=twowiki_components_directory,
        ),
    }


def _score_evaluation_projections(
    *,
    config: _LeanConfig,
    model_snapshot: Path,
    checkpoint_path: Path,
    projections: Mapping[DatasetKind, EvaluationProjection],
    device: str,
    batch_size: int,
) -> Mapping[DatasetKind, Mapping[str, Mapping[str, CandidateRiskScore]]]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("evaluation batch_size must be a positive integer")
    _audit_model_snapshot(model_snapshot, config)
    torch = importlib.import_module("torch")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = load_nli_dual_head_model(
        str(Path(model_snapshot).resolve()),
        revision=_MODEL_REVISION,
        identity_model_id=_MODEL_ID,
        local_files_only=True,
        device=device,
    )
    load_dual_head_checkpoint(model, checkpoint_path)
    model.eval()
    output: dict[DatasetKind, dict[str, dict[str, CandidateRiskScore]]] = {
        "niah": {},
        "2wiki": {},
    }
    with torch.inference_mode():
        for dataset_kind in _DATASET_KINDS:
            flat = [
                (row, candidate)
                for row in projections[dataset_kind].rows
                for candidate in row.topk10
            ]
            for start in range(0, len(flat), batch_size):
                batch = flat[start : start + batch_size]
                scores = model(
                    question=[row.question for row, _ in batch],
                    candidate_text=[candidate.text for _, candidate in batch],
                )
                protects = scores.protect_scores.detach().float().cpu().tolist()
                harms = scores.harm_scores.detach().float().cpu().tolist()
                for (row, candidate), protect_raw, harm_raw in zip(
                    batch, protects, harms, strict=True
                ):
                    protect = float(protect_raw)
                    harm = float(harm_raw)
                    if not all(
                        math.isfinite(value) and 0.0 <= value <= 1.0
                        for value in (protect, harm)
                    ):
                        raise RuntimeError("model emitted an invalid development score")
                    per_query = output[dataset_kind].setdefault(row.query_id, {})
                    if candidate.evidence_id in per_query:
                        raise ValueError("duplicate development evidence score")
                    per_query[candidate.evidence_id] = CandidateRiskScore(
                        protect_score=protect,
                        harm_score=harm,
                    )
    return output


def evaluate_development_selector_lean(
    *,
    config_path: Path,
    variant: Variant,
    model_snapshot: Path,
    seed13_fit_directory: Path,
    seed42_fit_directory: Path,
    niah_dataset_manifest: Path,
    niah_source_parent: Path,
    niah_candidate_pool: Path,
    niah_components_directory: Path,
    niah_assignment: Path,
    niah_provenance: Path,
    twowiki_dataset_manifest: Path,
    twowiki_source_parent: Path,
    twowiki_candidate_pool: Path,
    twowiki_components_directory: Path,
    output_results_path: Path,
    device: str,
    batch_size: int = 32,
) -> Mapping[str, object]:
    """Score ordinary development data and fill the complete pre-registered 4x2 grid."""

    config = _load_lean_config(config_path)
    bundles = {
        seed: _read_fit_bundle(
            fit_directory=seed13_fit_directory if seed == 13 else seed42_fit_directory,
            expected_seed=seed,
            expected_variant=variant,
            config_path=config_path,
        )
        for seed in _SEEDS
    }
    projections = _evaluation_projections(
        role="crc-calibration",
        niah_dataset_manifest=niah_dataset_manifest,
        niah_source_parent=niah_source_parent,
        niah_candidate_pool=niah_candidate_pool,
        niah_components_directory=niah_components_directory,
        niah_assignment=niah_assignment,
        niah_provenance=niah_provenance,
        twowiki_dataset_manifest=twowiki_dataset_manifest,
        twowiki_source_parent=twowiki_source_parent,
        twowiki_candidate_pool=twowiki_candidate_pool,
        twowiki_components_directory=twowiki_components_directory,
    )
    projection_sha256 = combine_projection_sha256(projections)
    thresholds = thresholds_from_train_scores(
        {seed: bundle.safe_scores for seed, bundle in bundles.items()}
    )
    candidates = build_policy_candidates(thresholds)
    scores_by_seed = {
        seed: _score_evaluation_projections(
            config=config,
            model_snapshot=model_snapshot,
            checkpoint_path=(
                seed13_fit_directory if seed == 13 else seed42_fit_directory
            )
            / _CHECKPOINT_FILE,
            projections=projections,
            device=device,
            batch_size=batch_size,
        )
        for seed in _SEEDS
    }
    results: list[DevelopmentCandidateResult] = []
    for candidate in candidates:
        if not candidate.policy_enabled:
            continue
        if candidate.quantile is None or candidate.cap is None:
            raise AssertionError("active Lean policy is missing quantile/cap")
        thresholds_by_seed = dict(candidate.thresholds_by_seed)
        seed13 = evaluate_seed_policy(
            seed=13,
            projections=projections,
            scores_by_dataset=scores_by_seed[13],
            threshold=thresholds_by_seed[13],
            cap=candidate.cap,
            include_controls=True,
        )
        seed42 = evaluate_seed_policy(
            seed=42,
            projections=projections,
            scores_by_dataset=scores_by_seed[42],
            threshold=thresholds_by_seed[42],
            cap=candidate.cap,
            include_controls=False,
        )
        results.append(
            combine_development_metrics(
                quantile=candidate.quantile,
                cap=candidate.cap,
                thresholds_by_seed=candidate.thresholds_by_seed,
                seed13=seed13,
                seed42=seed42,
            )
        )
    payload: dict[str, object] = {
        "schema_version": "selector-lean-development-results-v1",
        "role": "crc-calibration",
        "variant": variant,
        "development_projection_sha256": projection_sha256,
        "checkpoint_sha256_by_seed": {
            seed: bundle.checkpoint_sha256 for seed, bundle in bundles.items()
        },
        "datasets": {
            kind: {
                "queries": len(projections[kind].rows),
                "projection_sha256": projections[kind].sha256,
            }
            for kind in _DATASET_KINDS
        },
        "results": [asdict(result) for result in results],
    }
    _write_new(output_results_path, _json_bytes(payload))
    return payload


def _seed_thresholds(value: object) -> tuple[tuple[int, float], ...]:
    if not isinstance(value, Mapping):
        raise ValueError("development thresholds_by_seed must be a seed-keyed object")
    try:
        rows = tuple(sorted((int(seed), float(threshold)) for seed, threshold in value.items()))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid development thresholds_by_seed") from error
    return rows


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _strict_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _development_result(value: object) -> DevelopmentCandidateResult:
    if not isinstance(value, Mapping):
        raise ValueError("every development result must be an object")

    def number(name: str) -> float:
        return _finite_number(value.get(name), label=f"development {name}")

    cap = value.get("cap")
    if type(cap) is not int:
        raise ValueError("development cap must be an integer")
    return DevelopmentCandidateResult(
        quantile=number("quantile"),
        cap=cap,
        thresholds_by_seed=_seed_thresholds(value.get("thresholds_by_seed")),
        mean_deletions_per_query=number("mean_deletions_per_query"),
        harmful_reduction_seed13=number("harmful_reduction_seed13"),
        harmful_reduction_seed42=number("harmful_reduction_seed42"),
        niah_recall_loss_seed13_pp=number("niah_recall_loss_seed13_pp"),
        niah_recall_loss_seed42_pp=number("niah_recall_loss_seed42_pp"),
        twowiki_recall_loss_seed13_pp=number("twowiki_recall_loss_seed13_pp"),
        twowiki_recall_loss_seed42_pp=number("twowiki_recall_loss_seed42_pp"),
        niah_chain_loss_seed13_pp=number("niah_chain_loss_seed13_pp"),
        niah_chain_loss_seed42_pp=number("niah_chain_loss_seed42_pp"),
        twowiki_chain_loss_seed13_pp=number("twowiki_chain_loss_seed13_pp"),
        twowiki_chain_loss_seed42_pp=number("twowiki_chain_loss_seed42_pp"),
        seed13_harm_beats_random=_strict_bool(
            value.get("seed13_harm_beats_random"), label="seed13_harm_beats_random"
        ),
        seed13_harm_beats_bottom=_strict_bool(
            value.get("seed13_harm_beats_bottom"), label="seed13_harm_beats_bottom"
        ),
        seed13_precision_beats_random=_strict_bool(
            value.get("seed13_precision_beats_random"),
            label="seed13_precision_beats_random",
        ),
        seed13_precision_beats_bottom=_strict_bool(
            value.get("seed13_precision_beats_bottom"),
            label="seed13_precision_beats_bottom",
        ),
    )


def _read_development_results(
    path: Path,
    *,
    expected_variant: Variant,
    development_projection_sha256: str,
    checkpoint_sha256_by_seed: Mapping[int, str],
) -> tuple[DevelopmentCandidateResult, ...]:
    bundle = _read_json_object(path, label="development results")
    if (
        bundle.get("schema_version") != "selector-lean-development-results-v1"
        or bundle.get("role") != "crc-calibration"
        or bundle.get("variant") != expected_variant
        or bundle.get("development_projection_sha256") != development_projection_sha256
    ):
        raise ValueError("development result metadata differs from the requested Lean run")
    checkpoint_map = bundle.get("checkpoint_sha256_by_seed")
    if not isinstance(checkpoint_map, Mapping):
        raise ValueError("development results omit checkpoint_sha256_by_seed")
    try:
        observed_checkpoints = tuple(
            sorted((int(seed), str(digest)) for seed, digest in checkpoint_map.items())
        )
    except (TypeError, ValueError) as error:
        raise ValueError("development checkpoint map is invalid") from error
    if observed_checkpoints != tuple(sorted(checkpoint_sha256_by_seed.items())):
        raise ValueError("development results were produced by different checkpoints")
    values = bundle.get("results")
    if not isinstance(values, list):
        raise ValueError("development results must contain a results list")
    return tuple(_development_result(value) for value in values)


def calibrate_selector_lean(
    *,
    config_path: Path,
    variant: Variant,
    seed13_fit_directory: Path,
    seed42_fit_directory: Path,
    development_results_path: Path,
    code_commit: str,
    final_input_sha256: str,
    development_projection_sha256: str,
    generator_sha256: str,
    output_policy_path: Path,
    base_stop_path: Path | None = None,
) -> Mapping[str, object]:
    """Choose once on ordinary development data and freeze the resulting policy."""

    _load_lean_config(config_path)
    if variant == "NLI-pair":
        if base_stop_path is None:
            raise ValueError("NLI-pair requires the prior NLI-base STOP record")
        base_stop = _read_json_object(base_stop_path, label="NLI-base STOP record")
        expected_stop = {
            "schema_version": _BASE_STOP_SCHEMA,
            "action": "stop-keep-topk10",
            "variant": "NLI-base",
            "code_commit": code_commit,
            "final_input_sha256": final_input_sha256,
            "development_projection_sha256": development_projection_sha256,
            "generator_sha256": generator_sha256,
        }
        if any(base_stop.get(name) != value for name, value in expected_stop.items()):
            raise ValueError("NLI-base STOP record differs from the requested NLI-pair run")
    elif base_stop_path is not None:
        raise ValueError("base_stop_path is valid only for NLI-pair")
    bundles = {
        seed: _read_fit_bundle(
            fit_directory=seed13_fit_directory if seed == 13 else seed42_fit_directory,
            expected_seed=seed,
            expected_variant=variant,
            config_path=config_path,
        )
        for seed in _SEEDS
    }
    thresholds = thresholds_from_train_scores(
        {seed: bundle.safe_scores for seed, bundle in bundles.items()}
    )
    candidates = build_policy_candidates(thresholds)
    results = _read_development_results(
        development_results_path,
        expected_variant=variant,
        development_projection_sha256=_require_digest(
            development_projection_sha256,
            label="development_projection_sha256",
        ),
        checkpoint_sha256_by_seed={
            seed: bundle.checkpoint_sha256 for seed, bundle in bundles.items()
        },
    )
    selection = select_development_policy(results, policy_candidates=candidates)
    if selection is None:
        stop: dict[str, object] = {
            "schema_version": _BASE_STOP_SCHEMA,
            "action": "stop-keep-topk10",
            "variant": variant,
            "reason": "no development policy passed the frozen protection gate",
            "code_commit": code_commit,
            "final_input_sha256": final_input_sha256,
            "development_projection_sha256": development_projection_sha256,
            "generator_sha256": generator_sha256,
            "checkpoint_sha256_by_seed": {
                seed: bundle.checkpoint_sha256 for seed, bundle in bundles.items()
            },
        }
        _write_new(output_policy_path, _json_bytes(stop))
        return stop
    policy = freeze_policy(
        selection,
        variant=variant,
        checkpoint_sha256_by_seed={
            seed: bundle.checkpoint_sha256 for seed, bundle in bundles.items()
        },
        code_commit=code_commit,
        final_input_sha256=_require_digest(final_input_sha256, label="final_input_sha256"),
        development_projection_sha256=development_projection_sha256,
        generator_sha256=_require_digest(generator_sha256, label="generator_sha256"),
    )
    _write_new(output_policy_path, _json_bytes(policy.to_dict()))
    return {
        "action": "policy-frozen",
        "variant": variant,
        "quantile": policy.quantile,
        "cap": policy.cap,
        "output_policy": Path(output_policy_path).resolve().as_posix(),
        "output_policy_sha256": _sha256_file(output_policy_path),
    }


def _read_frozen_policy(path: Path) -> FrozenPolicy:
    return FrozenPolicy.from_dict(_read_json_object(path, label="frozen policy"))


class _PinnedGraniteTextGenerator:
    def __init__(self, *, snapshot: Path, device: str) -> None:
        transformers = importlib.import_module("transformers")
        self._torch = importlib.import_module("torch")
        root = str(Path(snapshot).resolve())
        load = {"local_files_only": True, "trust_remote_code": False}
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(root, **load)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            root,
            use_safetensors=True,
            dtype="auto",
            **load,
        ).to(device)
        self._model.eval()

    def generate(self, prompt: str) -> str:
        tokenizer = self._tokenizer
        if getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            encoded = tokenizer(prompt, return_tensors="pt")
        model_inputs = {
            name: value.to(self._model.device) if hasattr(value, "to") else value
            for name, value in encoded.items()
        }
        with self._torch.inference_mode():
            output = self._model.generate(
                **model_inputs,
                max_new_tokens=32,
                do_sample=False,
            )
        input_length = int(model_inputs["input_ids"].shape[-1])
        return str(
            tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        ).strip()


def _decision_rows(
    *,
    seed: int,
    decisions: Sequence[PolicyQueryResult],
    projections: Mapping[DatasetKind, EvaluationProjection],
    scores: Mapping[DatasetKind, Mapping[str, Mapping[str, CandidateRiskScore]]],
) -> tuple[Mapping[str, object], ...]:
    query_by_key = {
        (kind, row.query_id): row for kind, projection in projections.items() for row in projection.rows
    }
    output: list[Mapping[str, object]] = []
    for decision in decisions:
        row = query_by_key[(decision.dataset_kind, decision.query_id)]
        dropped = set(decision.dropped_evidence_ids)
        output.append(
            {
                "schema_version": "selector-lean-final-decision-v1",
                "seed": seed,
                "dataset_kind": decision.dataset_kind,
                "query_id": decision.query_id,
                "component_id": decision.component_id,
                "selected_evidence_ids": list(decision.selected_evidence_ids),
                "dropped_evidence_ids": list(decision.dropped_evidence_ids),
                "recall_loss": decision.recall_loss,
                "chain_loss": decision.chain_loss,
                "harmful_reduction": decision.harmful_reduction,
                "candidates": [
                    {
                        "evidence_id": candidate.evidence_id,
                        "retrieval_rank": candidate.retrieval_rank,
                        "protect_score": scores[decision.dataset_kind][row.query_id][
                            candidate.evidence_id
                        ].protect_score,
                        "harm_score": scores[decision.dataset_kind][row.query_id][
                            candidate.evidence_id
                        ].harm_score,
                        "safe_score": scores[decision.dataset_kind][row.query_id][
                            candidate.evidence_id
                        ].safe_score(),
                        "action": "DROP_HARM"
                        if candidate.evidence_id in dropped
                        else "KEEP",
                    }
                    for candidate in row.topk10
                ],
            }
        )
    return tuple(output)


def _evidence_gate(
    *,
    seed13_metrics: Any,
    seed42_metrics: Any,
    seed13_inference: Mapping[str, object],
) -> tuple[bool, Mapping[str, bool]]:
    niah_inference = cast(Mapping[str, object], seed13_inference["niah"])
    twowiki_inference = cast(Mapping[str, object], seed13_inference["2wiki"])
    harm_inference = cast(
        Mapping[str, object], seed13_inference["niah_harmful_reduction"]
    )
    harm_vs_topk = cast(Mapping[str, object], harm_inference["selector_vs_topk10"])

    def ci_high(dataset: Mapping[str, object], metric: str) -> float:
        return _finite_number(
            cast(Mapping[str, object], dataset[metric])["ci_high"],
            label=f"{metric} ci_high",
        )

    conditions = {
        "seed13_harm_ci_lower_gt_zero": _finite_number(
            harm_vs_topk["ci_low"], label="harm ci_low"
        )
        > 0.0,
        "seed42_harm_point_gt_zero": seed42_metrics.niah_harmful_reduction > 0.0,
        "seed13_niah_recall_point_le_1pp": seed13_metrics.niah_recall_loss_pp <= 1.0,
        "seed13_2wiki_recall_point_le_1pp": seed13_metrics.twowiki_recall_loss_pp <= 1.0,
        "seed13_niah_chain_point_le_1pp": seed13_metrics.niah_chain_loss_pp <= 1.0,
        "seed13_2wiki_chain_point_le_1pp": seed13_metrics.twowiki_chain_loss_pp <= 1.0,
        "seed13_niah_recall_ci_upper_le_3pp": ci_high(niah_inference, "recall_loss")
        <= 0.03,
        "seed13_2wiki_recall_ci_upper_le_3pp": ci_high(twowiki_inference, "recall_loss")
        <= 0.03,
        "seed13_niah_chain_ci_upper_le_3pp": ci_high(niah_inference, "chain_loss")
        <= 0.03,
        "seed13_2wiki_chain_ci_upper_le_3pp": ci_high(twowiki_inference, "chain_loss")
        <= 0.03,
        "seed42_niah_recall_point_le_3pp": seed42_metrics.niah_recall_loss_pp <= 3.0,
        "seed42_2wiki_recall_point_le_3pp": seed42_metrics.twowiki_recall_loss_pp <= 3.0,
        "seed42_niah_chain_point_le_3pp": seed42_metrics.niah_chain_loss_pp <= 3.0,
        "seed42_2wiki_chain_point_le_3pp": seed42_metrics.twowiki_chain_loss_pp <= 3.0,
        "seed13_harm_beats_random": seed13_metrics.niah_harmful_reduction
        > cast(float, seed13_metrics.random_harmful_reduction),
        "seed13_harm_beats_bottom": seed13_metrics.niah_harmful_reduction
        > cast(float, seed13_metrics.bottom_harmful_reduction),
        "seed13_precision_beats_random": seed13_metrics.niah_deletion_precision
        > cast(float, seed13_metrics.random_deletion_precision),
        "seed13_precision_beats_bottom": seed13_metrics.niah_deletion_precision
        > cast(float, seed13_metrics.bottom_deletion_precision),
    }
    return all(conditions.values()), conditions


def final_evaluate_selector_lean(
    *,
    config_path: Path,
    frozen_policy_path: Path,
    model_snapshot: Path,
    generator_snapshot: Path,
    seed13_fit_directory: Path,
    seed42_fit_directory: Path,
    code_commit: str,
    development_projection_sha256: str,
    niah_dataset_manifest: Path,
    niah_source_parent: Path,
    niah_candidate_pool: Path,
    niah_components_directory: Path,
    niah_assignment: Path,
    niah_provenance: Path,
    twowiki_dataset_manifest: Path,
    twowiki_source_parent: Path,
    twowiki_candidate_pool: Path,
    twowiki_components_directory: Path,
    output_directory: Path,
    device: str = "cuda:0",
    evaluation_batch_size: int = 32,
    generator_device: str = "cuda:0",
) -> Mapping[str, object]:
    """Run the once-only fixed-policy final evidence gate, then paired Granite answers."""

    config = _load_lean_config(config_path)
    root = Path(output_directory).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"final output directory must be absent or empty: {root}")
    generator_sha256 = _audit_generator_snapshot(generator_snapshot, config)
    policy = _read_frozen_policy(frozen_policy_path)
    bundles = {
        seed: _read_fit_bundle(
            fit_directory=seed13_fit_directory if seed == 13 else seed42_fit_directory,
            expected_seed=seed,
            expected_variant=policy.variant,
            config_path=config_path,
        )
        for seed in _SEEDS
    }
    observed_final_input_sha256 = final_input_sha256(
        niah_dataset_manifest=niah_dataset_manifest,
        niah_source_parent=niah_source_parent,
        niah_candidate_pool=niah_candidate_pool,
        niah_components_directory=niah_components_directory,
        niah_assignment=niah_assignment,
        niah_provenance=niah_provenance,
        twowiki_dataset_manifest=twowiki_dataset_manifest,
        twowiki_source_parent=twowiki_source_parent,
        twowiki_candidate_pool=twowiki_candidate_pool,
        twowiki_components_directory=twowiki_components_directory,
    )
    require_frozen_final_policy(
        policy,
        checkpoint_sha256_by_seed={
            seed: bundle.checkpoint_sha256 for seed, bundle in bundles.items()
        },
        code_commit=code_commit,
        final_input_sha256=observed_final_input_sha256,
        development_projection_sha256=_require_digest(
            development_projection_sha256,
            label="development_projection_sha256",
        ),
        generator_sha256=_require_digest(generator_sha256, label="generator_sha256"),
    )
    projections = _evaluation_projections(
        role="decision-dev",
        niah_dataset_manifest=niah_dataset_manifest,
        niah_source_parent=niah_source_parent,
        niah_candidate_pool=niah_candidate_pool,
        niah_components_directory=niah_components_directory,
        niah_assignment=niah_assignment,
        niah_provenance=niah_provenance,
        twowiki_dataset_manifest=twowiki_dataset_manifest,
        twowiki_source_parent=twowiki_source_parent,
        twowiki_candidate_pool=twowiki_candidate_pool,
        twowiki_components_directory=twowiki_components_directory,
    )
    decision_projection_sha256 = combine_projection_sha256(projections)
    thresholds = dict(policy.thresholds_by_seed)
    scores_by_seed = {
        seed: _score_evaluation_projections(
            config=config,
            model_snapshot=model_snapshot,
            checkpoint_path=(seed13_fit_directory if seed == 13 else seed42_fit_directory)
            / _CHECKPOINT_FILE,
            projections=projections,
            device=device,
            batch_size=evaluation_batch_size,
        )
        for seed in _SEEDS
    }
    decisions = {
        seed: apply_seed_policy(
            projections=projections,
            scores_by_dataset=scores_by_seed[seed],
            threshold=thresholds[seed],
            cap=policy.cap,
        )
        for seed in _SEEDS
    }
    metrics = {
        seed: evaluate_seed_policy(
            seed=seed,
            projections=projections,
            scores_by_dataset=scores_by_seed[seed],
            threshold=thresholds[seed],
            cap=policy.cap,
            include_controls=seed == 13,
        )
        for seed in _SEEDS
    }
    inference = {
        seed: evidence_inference(
            decisions=decisions[seed],
            projections=projections,
            include_controls=seed == 13,
        )
        for seed in _SEEDS
    }
    evidence_pass, evidence_conditions = _evidence_gate(
        seed13_metrics=metrics[13],
        seed42_metrics=metrics[42],
        seed13_inference=inference[13],
    )
    answer_rows: list[Mapping[str, object]] = []
    answer_inference: Mapping[str, object] | None = None
    answer_pass = False
    if evidence_pass:
        torch = importlib.import_module("torch")
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        generator = GraniteGenerator(
            llm=_PinnedGraniteTextGenerator(
                snapshot=generator_snapshot,
                device=generator_device,
            ),
            prompt_template=CITATION_RAG_PROMPT,
        )
        decision13 = {(row.dataset_kind, row.query_id): row for row in decisions[13]}
        selector_answers: dict[DatasetKind, dict[str, float]] = {"niah": {}, "2wiki": {}}
        topk_answers: dict[DatasetKind, dict[str, float]] = {"niah": {}, "2wiki": {}}
        components: dict[DatasetKind, dict[str, str]] = {"niah": {}, "2wiki": {}}
        for kind in _DATASET_KINDS:
            for row in projections[kind].rows:
                query = Query(query_id=row.query_id, text=row.question)
                checklist = QueryChecklist(
                    query_id=row.query_id,
                    focus=row.question,
                    required_facts=(),
                )
                baseline = SelectedEvidenceSet(query_id=row.query_id, evidence=row.topk10)
                baseline_generation = generator.generate(query, checklist, baseline)
                selected_ids = decision13[(kind, row.query_id)].selected_evidence_ids
                if selected_ids == tuple(candidate.evidence_id for candidate in row.topk10):
                    selector_generation = baseline_generation
                    reused = True
                else:
                    selected_set = set(selected_ids)
                    selected = SelectedEvidenceSet(
                        query_id=row.query_id,
                        evidence=tuple(
                            candidate for candidate in row.topk10
                            if candidate.evidence_id in selected_set
                        ),
                    )
                    selector_generation = generator.generate(query, checklist, selected)
                    reused = False
                baseline_match = answer_match(
                    baseline_generation.answer, row.reference_answers
                ).value
                selector_match = answer_match(
                    selector_generation.answer, row.reference_answers
                ).value
                if baseline_match is None or selector_match is None:
                    raise ValueError("final answer_match unexpectedly has no score")
                topk_answers[kind][row.query_id] = baseline_match
                selector_answers[kind][row.query_id] = selector_match
                components[kind][row.query_id] = row.component_id
                answer_rows.append(
                    {
                        "schema_version": "selector-lean-final-answer-v1",
                        "dataset_kind": kind,
                        "query_id": row.query_id,
                        "component_id": row.component_id,
                        "topk10": baseline_generation.model_dump(mode="json"),
                        "selector": selector_generation.model_dump(mode="json"),
                        "topk10_answer_match": baseline_match,
                        "selector_answer_match": selector_match,
                        "identical_context_reused": reused,
                    }
                )
        answer_inference = macro_answer_inference(
            selector_by_dataset=selector_answers,
            topk10_by_dataset=topk_answers,
            components_by_dataset=components,
        )
        dataset_delta = cast(Mapping[str, float], answer_inference["dataset_delta"])
        answer_pass = (
            _finite_number(answer_inference["macro_delta"], label="macro answer delta") > 0.0
            and dataset_delta["niah"] >= -0.01
            and dataset_delta["2wiki"] >= -0.01
        )

    report: dict[str, object] = {
        "schema_version": "selector-lean-final-report-v1",
        "status": "PASS" if evidence_pass and answer_pass else "FAIL",
        "role": "decision-dev",
        "frozen_policy_sha256": _sha256_file(frozen_policy_path),
        "decision_projection_sha256": decision_projection_sha256,
        "datasets": {
            kind: {
                "queries": len(projections[kind].rows),
                "projection_sha256": projections[kind].sha256,
            }
            for kind in _DATASET_KINDS
        },
        "policy": policy.to_dict(),
        "evidence_gate": {
            "status": "PASS" if evidence_pass else "FAIL",
            "conditions": evidence_conditions,
            "metrics_by_seed": {seed: asdict(metrics[seed]) for seed in _SEEDS},
            "inference_by_seed": inference,
        },
        "answer_gate": {
            "status": "PASS" if answer_pass else "FAIL",
            "skipped": not evidence_pass,
            "inference": answer_inference,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    _write_new(root / "final_report.json", _json_bytes(report))
    for seed in _SEEDS:
        _write_new(
            root / f"decision_trace_seed{seed}.jsonl",
            _jsonl_bytes(
                _decision_rows(
                    seed=seed,
                    decisions=decisions[seed],
                    projections=projections,
                    scores=scores_by_seed[seed],
                )
            ),
        )
    if answer_rows:
        _write_new(root / "answer_rows_seed13.jsonl", _jsonl_bytes(answer_rows))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal Selector Lean v3 experiment")
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="fit one frozen seed/variant checkpoint")
    fit.add_argument("--config", required=True, type=Path)
    fit.add_argument("--r004-root", required=True, type=Path)
    fit.add_argument("--model-snapshot", required=True, type=Path)
    fit.add_argument("--variant", required=True, choices=("NLI-base", "NLI-pair"))
    fit.add_argument("--seed", required=True, type=int, choices=_SEEDS)
    fit.add_argument("--output-dir", required=True, type=Path)
    fit.add_argument("--device", default="cuda:0")

    calibrate = commands.add_parser(
        "calibrate", help="choose one policy on crc-calibration and freeze it"
    )
    calibrate.add_argument("--config", required=True, type=Path)
    calibrate.add_argument("--variant", required=True, choices=("NLI-base", "NLI-pair"))
    calibrate.add_argument("--model-snapshot", required=True, type=Path)
    calibrate.add_argument("--generator-snapshot", required=True, type=Path)
    calibrate.add_argument("--seed13-fit-dir", required=True, type=Path)
    calibrate.add_argument("--seed42-fit-dir", required=True, type=Path)
    calibrate.add_argument(
        "--development-results",
        required=True,
        type=Path,
        help="new JSON path where the measured 4x2 development grid will be written",
    )
    calibrate.add_argument("--niah-dataset-manifest", required=True, type=Path)
    calibrate.add_argument("--niah-source-parent", required=True, type=Path)
    calibrate.add_argument("--niah-candidate-pool", required=True, type=Path)
    calibrate.add_argument("--niah-components-dir", required=True, type=Path)
    calibrate.add_argument("--niah-assignment", required=True, type=Path)
    calibrate.add_argument("--niah-provenance", required=True, type=Path)
    calibrate.add_argument("--twowiki-dataset-manifest", required=True, type=Path)
    calibrate.add_argument("--twowiki-source-parent", required=True, type=Path)
    calibrate.add_argument("--twowiki-candidate-pool", required=True, type=Path)
    calibrate.add_argument("--twowiki-components-dir", required=True, type=Path)
    calibrate.add_argument("--device", default="cuda:0")
    calibrate.add_argument("--evaluation-batch-size", type=int, default=32)
    calibrate.add_argument("--code-commit", required=True)
    calibrate.add_argument("--output-policy", required=True, type=Path)
    calibrate.add_argument(
        "--base-stop",
        type=Path,
        help="required only for NLI-pair; the matching NLI-base STOP JSON",
    )

    final = commands.add_parser(
        "final-evaluate", help="verify the frozen policy and open decision-dev once"
    )
    final.add_argument("--config", required=True, type=Path)
    final.add_argument("--frozen-policy", required=True, type=Path)
    final.add_argument("--model-snapshot", required=True, type=Path)
    final.add_argument("--generator-snapshot", required=True, type=Path)
    final.add_argument("--seed13-fit-dir", required=True, type=Path)
    final.add_argument("--seed42-fit-dir", required=True, type=Path)
    final.add_argument("--code-commit", required=True)
    final.add_argument("--development-projection-sha256", required=True)
    final.add_argument("--niah-dataset-manifest", required=True, type=Path)
    final.add_argument("--niah-source-parent", required=True, type=Path)
    final.add_argument("--niah-candidate-pool", required=True, type=Path)
    final.add_argument("--niah-components-dir", required=True, type=Path)
    final.add_argument("--niah-assignment", required=True, type=Path)
    final.add_argument("--niah-provenance", required=True, type=Path)
    final.add_argument("--twowiki-dataset-manifest", required=True, type=Path)
    final.add_argument("--twowiki-source-parent", required=True, type=Path)
    final.add_argument("--twowiki-candidate-pool", required=True, type=Path)
    final.add_argument("--twowiki-components-dir", required=True, type=Path)
    final.add_argument("--output-dir", required=True, type=Path)
    final.add_argument("--device", default="cuda:0")
    final.add_argument("--evaluation-batch-size", type=int, default=32)
    final.add_argument("--generator-device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "fit":
        result = fit_selector_lean(
            config_path=arguments.config,
            r004_root=arguments.r004_root,
            model_snapshot=arguments.model_snapshot,
            variant=cast(Variant, arguments.variant),
            seed=arguments.seed,
            output_directory=arguments.output_dir,
            device=arguments.device,
        )
    elif arguments.command == "calibrate":
        development = evaluate_development_selector_lean(
            config_path=arguments.config,
            variant=cast(Variant, arguments.variant),
            model_snapshot=arguments.model_snapshot,
            seed13_fit_directory=arguments.seed13_fit_dir,
            seed42_fit_directory=arguments.seed42_fit_dir,
            niah_dataset_manifest=arguments.niah_dataset_manifest,
            niah_source_parent=arguments.niah_source_parent,
            niah_candidate_pool=arguments.niah_candidate_pool,
            niah_components_directory=arguments.niah_components_dir,
            niah_assignment=arguments.niah_assignment,
            niah_provenance=arguments.niah_provenance,
            twowiki_dataset_manifest=arguments.twowiki_dataset_manifest,
            twowiki_source_parent=arguments.twowiki_source_parent,
            twowiki_candidate_pool=arguments.twowiki_candidate_pool,
            twowiki_components_directory=arguments.twowiki_components_dir,
            output_results_path=arguments.development_results,
            device=arguments.device,
            batch_size=arguments.evaluation_batch_size,
        )
        development_projection_sha256 = _require_digest(
            development.get("development_projection_sha256"),
            label="development projection hash",
        )
        config = _load_lean_config(arguments.config)
        pinned_final_input = final_input_sha256(
            niah_dataset_manifest=arguments.niah_dataset_manifest,
            niah_source_parent=arguments.niah_source_parent,
            niah_candidate_pool=arguments.niah_candidate_pool,
            niah_components_directory=arguments.niah_components_dir,
            niah_assignment=arguments.niah_assignment,
            niah_provenance=arguments.niah_provenance,
            twowiki_dataset_manifest=arguments.twowiki_dataset_manifest,
            twowiki_source_parent=arguments.twowiki_source_parent,
            twowiki_candidate_pool=arguments.twowiki_candidate_pool,
            twowiki_components_directory=arguments.twowiki_components_dir,
        )
        pinned_generator = _audit_generator_snapshot(arguments.generator_snapshot, config)
        result = calibrate_selector_lean(
            config_path=arguments.config,
            variant=cast(Variant, arguments.variant),
            seed13_fit_directory=arguments.seed13_fit_dir,
            seed42_fit_directory=arguments.seed42_fit_dir,
            development_results_path=arguments.development_results,
            code_commit=arguments.code_commit,
            final_input_sha256=pinned_final_input,
            development_projection_sha256=development_projection_sha256,
            generator_sha256=pinned_generator,
            output_policy_path=arguments.output_policy,
            base_stop_path=arguments.base_stop,
        )
    elif arguments.command == "final-evaluate":
        result = final_evaluate_selector_lean(
            config_path=arguments.config,
            frozen_policy_path=arguments.frozen_policy,
            model_snapshot=arguments.model_snapshot,
            generator_snapshot=arguments.generator_snapshot,
            seed13_fit_directory=arguments.seed13_fit_dir,
            seed42_fit_directory=arguments.seed42_fit_dir,
            code_commit=arguments.code_commit,
            development_projection_sha256=arguments.development_projection_sha256,
            niah_dataset_manifest=arguments.niah_dataset_manifest,
            niah_source_parent=arguments.niah_source_parent,
            niah_candidate_pool=arguments.niah_candidate_pool,
            niah_components_directory=arguments.niah_components_dir,
            niah_assignment=arguments.niah_assignment,
            niah_provenance=arguments.niah_provenance,
            twowiki_dataset_manifest=arguments.twowiki_dataset_manifest,
            twowiki_source_parent=arguments.twowiki_source_parent,
            twowiki_candidate_pool=arguments.twowiki_candidate_pool,
            twowiki_components_directory=arguments.twowiki_components_dir,
            output_directory=arguments.output_dir,
            device=arguments.device,
            evaluation_batch_size=arguments.evaluation_batch_size,
            generator_device=arguments.generator_device,
        )
    else:  # pragma: no cover - argparse enforces the finite command set
        raise AssertionError(f"unhandled command: {arguments.command}")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
