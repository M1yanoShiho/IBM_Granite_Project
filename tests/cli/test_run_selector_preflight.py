import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

import evidence_rag.cli.run_selector_preflight as preflight_cli
from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate
from evidence_rag.evaluation.selector_components import OUTPUT_FILES as COMPONENT_OUTPUT_FILES
from evidence_rag.evaluation.selector_preflight import PreflightSampleQuery
from evidence_rag.materializer.selector_labels import (
    LABELS_FILE,
    SelectorLabelArtifacts,
    SelectorLabelRow,
    project_text_pair,
    text_pair_sha256,
)
from evidence_rag.materializer.selector_labels import (
    OUTPUT_FILES as LABEL_OUTPUT_FILES,
)
from evidence_rag.materializer.selector_pool import CANDIDATE_FILE, SELECTOR_POOL_MANIFEST_FILE

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = REPO_ROOT / "configs/selector/adaptive_risk_v1.toml"
DATASET_PATH_FIELDS = (
    "niah_dataset_manifest",
    "niah_source_parent",
    "niah_candidate_pool",
    "niah_components_dir",
    "niah_assignment",
    "niah_provenance",
    "niah_labels_dir",
    "twowiki_dataset_manifest",
    "twowiki_source_parent",
    "twowiki_candidate_pool",
    "twowiki_components_dir",
    "twowiki_labels_dir",
)


def _write(path: Path, payload: bytes = b"fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _real_config() -> preflight_cli._FrozenConfig:
    return preflight_cli._load_config(REAL_CONFIG)


def _label_report(expected: Mapping[str, int]) -> dict[str, object]:
    return {
        "counts": {
            "queries": expected["queries"],
            "candidate_rows": expected["candidate_rows"],
            "roles": {
                "train-fit": expected["train_fit_rows"],
                "train-modelval": expected["train_modelval_rows"],
            },
            "label_pairs": {
                "protect=0,harm=1": expected.get("protect_0_harm_1", 0),
                "protect=1,harm=0": expected.get("protect_1_harm_0", 0),
                "protect=1,harm=mask": expected.get("protect_1_harm_mask", 0),
                "protect=mask,harm=mask": expected.get("protect_mask_harm_mask", 0),
            },
        }
    }


def _dataset_namespace(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        niah_dataset_manifest=tmp_path / "niah/train/manifest.json",
        niah_source_parent=tmp_path / "niah/train/source_parent.jsonl",
        niah_candidate_pool=tmp_path / "niah/train/pool",
        niah_components_dir=tmp_path / "niah/train/components",
        niah_assignment=tmp_path / "niah/train/assignment.jsonl",
        niah_provenance=tmp_path / "niah/train/provenance.jsonl",
        niah_labels_dir=tmp_path / "niah/train/labels",
        twowiki_dataset_manifest=tmp_path / "2wiki/train/manifest.json",
        twowiki_source_parent=tmp_path / "2wiki/train/source_parent.jsonl",
        twowiki_candidate_pool=tmp_path / "2wiki/train/pool",
        twowiki_components_dir=tmp_path / "2wiki/train/components",
        twowiki_labels_dir=tmp_path / "2wiki/train/labels",
    )


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _clean_git_repository(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "R004 test")
    _git(repo, "config", "user.email", "r004@example.invalid")
    required = _write(repo / "src/evidence_rag/cli/run_selector_preflight.py", b"tracked code\n")
    _git(repo, "add", required.relative_to(repo).as_posix())
    _git(repo, "commit", "-m", "initial")
    return repo, required


def _mini_snapshot(
    tmp_path: Path,
) -> tuple[preflight_cli._FrozenConfig, Path, dict[str, dict[str, object]]]:
    base = _real_config()
    root = tmp_path / "mini-revision"
    payloads = {filename: f"frozen:{filename}\n".encode() for filename in base.snapshot_files}
    for filename, payload in payloads.items():
        _write(root / filename, payload)
    digests = {filename: _sha256(payload) for filename, payload in payloads.items()}
    config = replace(
        base,
        revision=root.name,
        model_sha256=digests["model.safetensors"],
        snapshot_files=digests,
    )
    return config, root, preflight_cli._audit_model_snapshot(root, config)


def _pin_inputs(tmp_path: Path, kind: str) -> preflight_cli._DatasetArguments:
    root = tmp_path / kind
    manifest = _write(root / "manifest.json")
    source_parent = _write(root / "source_parent.jsonl")
    pool = root / "pool"
    _write(pool / CANDIDATE_FILE)
    _write(pool / SELECTOR_POOL_MANIFEST_FILE)
    components = root / "components"
    for filename in COMPONENT_OUTPUT_FILES:
        _write(components / filename)
    labels = root / "labels"
    for filename in LABEL_OUTPUT_FILES:
        _write(labels / filename)
    assignment = _write(root / "assignment.jsonl") if kind == "niah" else None
    provenance = _write(root / "provenance.jsonl") if kind == "niah" else None
    return preflight_cli._DatasetArguments(
        kind=kind,
        dataset_manifest=manifest,
        source_parent=source_parent,
        candidate_pool=pool,
        components_dir=components,
        labels_dir=labels,
        assignment=assignment,
        provenance=provenance,
    )


def _prepared_fixture(
    tmp_path: Path, *, kind: str, query_id: str, bad_text_hash: bool = False
) -> tuple[preflight_cli._DatasetArguments, SelectorLabelArtifacts, dict[str, int]]:
    root = tmp_path / kind
    question = f"Question for {kind}?"
    manifest = {
        "schema_version": "1.0",
        "dataset_id": f"toy-{kind}",
        "dataset_version": "v1",
        "split": "train",
        "documents_file": "documents.jsonl",
        "queries_file": "queries.jsonl",
        "gold_cases_file": "gold_cases.jsonl",
    }
    _write(root / "manifest.json", (json.dumps(manifest) + "\n").encode())
    _write(
        root / "queries.jsonl",
        (
            json.dumps({"schema_version": "1.0", "query_id": query_id, "text": question}) + "\n"
        ).encode(),
    )
    candidates = tuple(
        EvidenceCandidate(
            evidence_id=f"{query_id}-e{rank}",
            document_id=f"{query_id}-d{rank}",
            chunk_id=f"{query_id}-c{rank}",
            text=f"candidate text {kind} {rank}",
            source_uri=f"toy://{kind}/{rank}",
            retrieval_score=1.0 / rank,
            retrieval_rank=rank,
        )
        for rank in (2, 1)
    )
    pool = root / "pool"
    candidate_set = CandidateSet(query_id=query_id, candidates=candidates)
    _write(pool / CANDIDATE_FILE, (candidate_set.model_dump_json() + "\n").encode())

    rows: list[SelectorLabelRow] = []
    for candidate in candidates:
        digest = text_pair_sha256(
            project_text_pair(question=question, candidate_text=candidate.text)
        )
        if bad_text_hash and candidate.retrieval_rank == 1:
            digest = "f" * 64
        supervised = candidate.retrieval_rank == 1
        rows.append(
            SelectorLabelRow(
                dataset_id=f"toy-{kind}",
                dataset_kind=kind,
                query_id=query_id,
                component_id=f"component-{query_id}",
                role="train-modelval",
                evidence_id=candidate.evidence_id,
                document_id=candidate.document_id,
                text_pair_sha256=digest,
                protect_label=1 if supervised else None,
                protect_mask=supervised,
                protect_source="official-support" if supervised else "unjudged",
                harm_label=0 if supervised else None,
                harm_mask=supervised,
                harm_source="verified-clean" if supervised else "unjudged",
            )
        )
    expected = {
        "queries": 1,
        "candidate_rows": 2,
        "train_fit_rows": 0,
        "train_modelval_rows": 2,
        "train_modelval_queries": 1,
        "protect_1_harm_0": 1,
        "protect_mask_harm_mask": 1,
    }
    artifacts = SelectorLabelArtifacts(
        files={LABELS_FILE: "".join(row.model_dump_json() + "\n" for row in rows).encode()},
        report=_label_report(expected),
        manifest={},
    )
    arguments = preflight_cli._DatasetArguments(
        kind=kind,
        dataset_manifest=root / "manifest.json",
        source_parent=root / "source_parent.jsonl",
        candidate_pool=pool,
        components_dir=root / "components",
        labels_dir=root / "labels",
    )
    return arguments, artifacts, expected


def test_real_adaptive_risk_config_loads_the_frozen_contract() -> None:
    config = _real_config()

    assert config.model_id == "cross-encoder/nli-deberta-v3-base"
    assert config.max_length == 512
    assert config.sample_seed == 20260811
    assert config.questions_per_dataset == 100
    assert config.candidates_per_query == 20
    assert config.forward_pairs == 4000
    assert len(config.snapshot_files) == 7
    assert set(config.snapshot_files) == {
        "added_tokens.json",
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "spm.model",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert config.expected_label_counts["niah"]["candidate_rows"] == 20_460
    assert config.expected_label_counts["2wiki"]["candidate_rows"] == 60_000


@pytest.mark.parametrize(
    "field",
    DATASET_PATH_FIELDS,
)
@pytest.mark.parametrize("forbidden", ("heldout", "SEALED"))
def test_dataset_arguments_reject_heldout_and_sealed_paths(
    tmp_path: Path, field: str, forbidden: str
) -> None:
    arguments = _dataset_namespace(tmp_path)
    setattr(arguments, field, tmp_path / forbidden / field)

    with pytest.raises(ValueError, match="refuses sealed/heldout"):
        preflight_cli._dataset_arguments(arguments)


@pytest.mark.parametrize("field", DATASET_PATH_FIELDS)
@pytest.mark.parametrize("forbidden", ("heldout", "SEALED"))
def test_dataset_arguments_reject_neutral_symlink_to_forbidden_path(
    tmp_path: Path, field: str, forbidden: str
) -> None:
    arguments = _dataset_namespace(tmp_path)
    target = _write(tmp_path / "storage" / forbidden / field)
    neutral = tmp_path / "neutral" / field
    neutral.parent.mkdir(parents=True, exist_ok=True)
    neutral.symlink_to(target)
    assert forbidden.lower() not in str(neutral).lower()
    setattr(arguments, field, neutral)

    with pytest.raises(ValueError, match="refuses sealed/heldout"):
        preflight_cli._dataset_arguments(arguments)


def test_git_snapshot_accepts_clean_repo_with_required_tracked_code(tmp_path: Path) -> None:
    repo, required = _clean_git_repository(tmp_path)

    snapshot = preflight_cli._git_snapshot(repo, required_tracked_paths=(required,))

    assert snapshot == {
        "branch": _git(repo, "branch", "--show-current"),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": False,
    }


def test_git_snapshot_rejects_git_subdirectory_as_repo_root(tmp_path: Path) -> None:
    repo, required = _clean_git_repository(tmp_path)

    with pytest.raises(ValueError, match="not the Git top-level"):
        preflight_cli._git_snapshot(repo / "src", required_tracked_paths=(required,))


def test_git_snapshot_rejects_untracked_required_code_path(tmp_path: Path) -> None:
    repo, required = _clean_git_repository(tmp_path)
    untracked_required = _write(repo / "src/evidence_rag/selector/dual_head.py")

    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        preflight_cli._git_snapshot(repo, required_tracked_paths=(required, untracked_required))


def test_git_snapshot_rejects_any_other_untracked_file(tmp_path: Path) -> None:
    repo, required = _clean_git_repository(tmp_path)
    _write(repo / "untracked.txt")

    with pytest.raises(ValueError, match="clean git worktree"):
        preflight_cli._git_snapshot(repo, required_tracked_paths=(required,))


@pytest.mark.parametrize("kind", ("niah", "2wiki"))
def test_expected_label_counts_accept_real_counts_and_reject_drift(kind: str) -> None:
    expected = _real_config().expected_label_counts[kind]
    report = _label_report(expected)

    preflight_cli._check_expected_label_counts(dataset_kind=kind, report=report, expected=expected)
    counts = report["counts"]
    assert isinstance(counts, dict)
    counts["candidate_rows"] = int(expected["candidate_rows"]) + 1
    with pytest.raises(ValueError, match=f"{kind} formal label count candidate_rows"):
        preflight_cli._check_expected_label_counts(
            dataset_kind=kind, report=report, expected=expected
        )


def test_snapshot_gate_accepts_exactly_the_seven_pinned_files(tmp_path: Path) -> None:
    config, root, pins = _mini_snapshot(tmp_path)

    assert len(pins) == 7
    assert set(pins) == set(config.snapshot_files)
    for filename, pin in pins.items():
        assert pin == {
            "path": str((root / filename).resolve()),
            "bytes": (root / filename).stat().st_size,
            "sha256": config.snapshot_files[filename],
        }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "file set differs"),
        ("unexpected", "file set differs"),
        ("tampered", "SHA-256 mismatch"),
    ),
)
def test_snapshot_gate_rejects_file_set_or_hash_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    config, root, _ = _mini_snapshot(tmp_path)
    target = root / "model.safetensors"
    if mutation == "missing":
        target.unlink()
    elif mutation == "unexpected":
        _write(root / "unexpected.bin")
    else:
        target.write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match=message):
        preflight_cli._audit_model_snapshot(root, config)


def test_input_pins_bind_snapshot_and_every_dataset_sidecar(tmp_path: Path) -> None:
    _, _, snapshot_pins = _mini_snapshot(tmp_path / "snapshot")
    config_path = _write(tmp_path / "config.toml", b"frozen-config\n")
    datasets = (_pin_inputs(tmp_path, "niah"), _pin_inputs(tmp_path, "2wiki"))

    pins = preflight_cli._input_pins(
        config_path=config_path,
        snapshot_pins=snapshot_pins,
        dataset_arguments=datasets,
        local_artifact_root=tmp_path,
    )

    expected_keys = {"config"}
    expected_keys.update(f"model_snapshot/{name}" for name in snapshot_pins)
    for dataset in datasets:
        prefix = dataset.kind
        expected_keys.update(
            {
                f"{prefix}/dataset_manifest",
                f"{prefix}/source_parent",
                f"{prefix}/candidate_pool",
                f"{prefix}/pool_manifest",
            }
        )
        expected_keys.update(f"{prefix}/labels/{name}" for name in LABEL_OUTPUT_FILES)
        expected_keys.update(f"{prefix}/components/{name}" for name in COMPONENT_OUTPUT_FILES)
    expected_keys.update({"niah/assignment", "niah/provenance"})
    assert set(pins) == expected_keys
    assert "2wiki/assignment" not in pins
    assert "2wiki/provenance" not in pins
    for name, pin in pins.items():
        path = Path(str(pin["path"]))
        if not path.is_absolute():
            path = tmp_path / path
        assert pin["bytes"] == path.stat().st_size
        assert pin["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        if "/labels/" in name:
            assert not Path(str(pin["path"])).is_absolute()


def test_prepare_and_sample_join_exact_query_candidate_and_text_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight_cli, "PREFLIGHT_CANDIDATES_PER_QUERY", 2)
    monkeypatch.setattr(preflight_cli, "PREFLIGHT_FORWARD_PAIRS", 4)
    niah_args, niah_artifacts, niah_expected = _prepared_fixture(
        tmp_path, kind="niah", query_id="n-q"
    )
    wiki_args, wiki_artifacts, wiki_expected = _prepared_fixture(
        tmp_path, kind="2wiki", query_id="w-q"
    )
    datasets = {
        "niah": preflight_cli._prepare_dataset(niah_args, niah_artifacts, niah_expected),
        "2wiki": preflight_cli._prepare_dataset(wiki_args, wiki_artifacts, wiki_expected),
    }
    sample = (
        PreflightSampleQuery(dataset_kind="2wiki", query_id="w-q", sample_digest="a" * 64),
        PreflightSampleQuery(dataset_kind="niah", query_id="n-q", sample_digest="b" * 64),
    )

    pairs = preflight_cli._sample_pairs(datasets, sample)

    assert [(pair.dataset_kind, pair.query_id, pair.retrieval_rank) for pair in pairs] == [
        ("2wiki", "w-q", 1),
        ("2wiki", "w-q", 2),
        ("niah", "n-q", 1),
        ("niah", "n-q", 2),
    ]
    assert [pair.candidate_text for pair in pairs[:2]] == [
        "candidate text 2wiki 1",
        "candidate text 2wiki 2",
    ]


def test_prepare_dataset_rejects_label_to_model_text_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight_cli, "PREFLIGHT_CANDIDATES_PER_QUERY", 2)
    arguments, artifacts, expected = _prepared_fixture(
        tmp_path, kind="niah", query_id="n-q", bad_text_hash=True
    )

    with pytest.raises(ValueError, match="label/model text hash mismatch"):
        preflight_cli._prepare_dataset(arguments, artifacts, expected)
