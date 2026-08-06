"""Tests for the R013-R015 training CLI (M0 §3.8).

The fits themselves are replaced by a fake fitter: no GPU here, and the point of these tests is
the data path, not the optimiser. Everything asserted below is a way to finish a training run
and get a plausible model out of the wrong data.
"""

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from evidence_rag.cli import train_relations
from evidence_rag.materializer.provenance import MutationRecord, write_provenance
from evidence_rag.relations.models import RelationLabel
from evidence_rag.relations.training import (
    BASE_LABEL_ORDER,
    Fold,
    FoldFit,
    FoldFitter,
)


def _pairs_file(path: Path, pages: list[str]) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "premise": f"evidence {index}",
                    "hypothesis": f"claim {index}",
                    "label": ("SUPPORTS", "REFUTES", "UNKNOWN")[index % 3],
                    "group": page,
                }
            )
            + "\n"
            for index, page in enumerate(pages)
        ),
        encoding="utf-8",
    )
    return path


def _fake_fitter(**_: object) -> FoldFitter:
    def fit(*, fold: Fold, seed: int, base_model: str) -> FoldFit:
        return FoldFit(
            model_version=f"fake/seed-{seed}-fold-{fold.index}@{fold.index:016x}",
            label_order=BASE_LABEL_ORDER,
            predictions=tuple(RelationLabel.SUPPORTS for _ in fold.held_out),
        )

    return fit


def _corpus(tmp_path: Path, n_train: int = 12, n_dev: int = 2) -> dict[str, Path]:
    train = _pairs_file(tmp_path / "train.jsonl", [f"train-page-{i}" for i in range(n_train)])
    dev = _pairs_file(tmp_path / "dev.jsonl", [f"dev-page-{i}" for i in range(n_dev)])
    log = tmp_path / "decontamination.json"
    log.write_text(
        json.dumps(
            {
                "n_train": n_train,
                "n_dev": n_dev,
                "n_test": 99,
                "removed_train_groups": [],
                "removed_dev_groups": [],
            }
        ),
        encoding="utf-8",
    )
    return {"train": train, "dev": dev, "log": log}


def _argv(files: dict[str, Path], output: Path, *extra: str) -> list[str]:
    return [
        "--vitaminc-train",
        str(files["train"]),
        "--vitaminc-dev",
        str(files["dev"]),
        "--decontamination-log",
        str(files["log"]),
        "--seed",
        "13",
        "--output-dir",
        str(output),
        *extra,
    ]


@pytest.fixture(autouse=True)
def _no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing in this file may reach sentence-transformers. If the seam ever stops being the
    only door to it, these tests would start needing a GPU to say anything."""
    monkeypatch.setattr(train_relations, "load_fold_fitter", _fake_fitter)


def test_a_vitaminc_only_run_writes_a_manifest_and_out_of_fold_predictions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    assert train_relations.main(_argv(_corpus(tmp_path), output)) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 13
    assert manifest["base_model"] == "cross-encoder/nli-deberta-v3-base"
    assert manifest["chain"]["n_vitaminc"] == 12
    assert manifest["chain"]["n_niah"] == 0
    assert len(manifest["folds"]) == 5
    rows = (output / "oof_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 12


def test_the_manifest_says_in_words_that_the_niah_half_did_not_run(tmp_path: Path) -> None:
    """A VitaminC-only run is a legitimate partial execution of §3.8, and the difference has to
    be legible in the artefact — otherwise a later reader cannot tell this run from one that did
    the domain adaptation."""
    output = tmp_path / "run"
    train_relations.main(_argv(_corpus(tmp_path), output))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["niah_domain_adaptation"]["status"] == "not run"
    assert "sealed-600" in manifest["niah_domain_adaptation"]["reason"]


def test_dev_rows_never_enter_the_training_chain(tmp_path: Path) -> None:
    """§2.4 and §11.2a reserve VitaminC official dev as the one legal calibration surface. Dev is
    loaded here only to check decontamination ran, and the chain size proves it stopped there."""
    output = tmp_path / "run"
    train_relations.main(_argv(_corpus(tmp_path, n_train=12, n_dev=2), output))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chain"]["n_examples"] == 12


def test_a_train_file_that_does_not_match_the_decontamination_log_is_refused(
    tmp_path: Path,
) -> None:
    """The likeliest slip on this path is pointing --vitaminc-train at vitaminc_test.jsonl. Same
    schema, trains fine, and it is the split §3.8 permits exactly one run on."""
    files = _corpus(tmp_path)
    _pairs_file(files["train"], [f"other-page-{i}" for i in range(30)])
    with pytest.raises(ValueError, match="decontamination"):
        train_relations.main(_argv(files, tmp_path / "run"))


def test_undecontaminated_splits_are_refused(tmp_path: Path) -> None:
    """A page in both train and dev means `decontaminate` never ran."""
    files = _corpus(tmp_path, n_train=12, n_dev=2)
    _pairs_file(files["dev"], ["train-page-0", "dev-page-1"])
    with pytest.raises(ValueError, match="decontamination"):
        train_relations.main(_argv(files, tmp_path / "run"))


def test_a_seed_outside_the_pre_registered_three_is_refused(tmp_path: Path) -> None:
    argv = _argv(_corpus(tmp_path), tmp_path / "run")
    argv[argv.index("13")] = "99"
    with pytest.raises(ValueError, match="pre-registered seeds"):
        train_relations.main(argv)


def test_asking_for_the_niah_half_without_sealed_600_is_refused(tmp_path: Path) -> None:
    """The blocker, at the CLI boundary. There is no flag that turns the check off, and giving
    the NIAH inputs without the sealed corpus stops the run instead of quietly training on
    VitaminC alone — which would leave a manifest claiming domain adaptation happened."""
    files = _corpus(tmp_path)
    with pytest.raises(ValueError, match="sealed-600"):
        train_relations.main(
            _argv(
                files,
                tmp_path / "run",
                "--niah-manifest",
                str(tmp_path / "manifest.json"),
                "--niah-provenance",
                str(tmp_path / "provenance.jsonl"),
                "--niah-parents",
                str(tmp_path / "parents.jsonl"),
            )
        )


def _niah_fixture(tmp_path: Path) -> list[str]:
    """A one-record NIAH train split: manifest + mutation log + parent sidecar."""
    niah = tmp_path / "niah"
    niah.mkdir()

    def write(name: str, rows: list[dict[str, object]]) -> None:
        (niah / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    write(
        "documents.jsonl",
        [
            {
                "schema_version": "1.0",
                "document_id": "needle",
                "text": "JFK\n\nKennedy won",
                "source_uri": "s://n",
            },
            {
                "schema_version": "1.0",
                "document_id": "cf::needle",
                "text": "JFK\n\nNixon won",
                "source_uri": "s://c",
            },
        ],
    )
    write("queries.jsonl", [{"schema_version": "1.0", "query_id": "q1", "text": "who won?"}])
    write(
        "gold_cases.jsonl",
        [{"query_id": "q1", "relevant_document_ids": ["needle"], "reference_answers": ["JFK"]}],
    )
    (niah / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "niah",
                "dataset_version": "train+cf42",
                "split": "train",
                "documents_file": "documents.jsonl",
                "queries_file": "queries.jsonl",
                "gold_cases_file": "gold_cases.jsonl",
            }
        ),
        encoding="utf-8",
    )
    write_provenance(
        niah / "provenance.jsonl",
        [
            MutationRecord(
                query_id="q1",
                needle_document_id="needle",
                counterfactual_document_id="cf::needle",
                gold_value="kennedy",
                gold_alias_used="Kennedy",
                replacement_value="Nixon",
                string_class="proper_name_1",
                seed=42,
                char_span=(0, 7),
                text_hash_before="a" * 8,
                text_hash_after="b" * 8,
                answer_bank_hash="c" * 8,
            )
        ],
    )
    write(
        "source_parent.jsonl",
        [
            {"document_id": "needle", "source_parent_id": "JFK"},
            {"document_id": "cf::needle", "source_parent_id": "JFK"},
        ],
    )
    return [
        "--niah-manifest",
        str(niah / "manifest.json"),
        "--niah-provenance",
        str(niah / "provenance.jsonl"),
        "--niah-parents",
        str(niah / "source_parent.jsonl"),
        "--niah-twin-label",
        "REFUTES",
    ]


@dataclass(frozen=True)
class _Fingerprint:
    parent_pages: tuple[str, ...]


def _sealed_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *titles: str
) -> list[str]:
    """Fake the corpus line's `materializer.sealed600`, which has not landed here yet.

    Only the published contract is faked -- `read_split_fingerprint(directory).parent_pages`.
    """
    module = types.ModuleType("evidence_rag.materializer.sealed600")
    module.read_split_fingerprint = lambda directory: _Fingerprint(titles)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "evidence_rag.materializer.sealed600", module)
    return ["--sealed-dir", str(tmp_path / "sealed600")]


def test_the_niah_half_runs_once_a_disjoint_sealed_600_is_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The half is implemented, not merely stubbed: given a sealed-600 the pairs clear, it adds
    its four rows per record and the manifest records what it was checked against."""
    output = tmp_path / "run"
    argv = _argv(
        _corpus(tmp_path),
        output,
        *_niah_fixture(tmp_path),
        *_sealed_dir(tmp_path, monkeypatch, "Some other article"),
    )
    assert train_relations.main(argv) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chain"]["n_niah"] == 4
    adaptation = manifest["niah_domain_adaptation"]
    assert adaptation["status"] == "enforced"
    assert adaptation["twin_label"] == "REFUTES"
    assert adaptation["twin_label_ruling"] == "M0 §3.8(a)"
    assert adaptation["sealed_n_parent_pages"] == 1
    assert len(adaptation["sealed_parent_pages_sha256"]) == 64


def test_a_sealed_600_that_shares_an_article_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The constraint §3.8 states, at the CLI boundary: the needle's article is in the sealed
    evaluation corpus, so adapting on it would put the eval set into the training data."""
    argv = _argv(
        _corpus(tmp_path),
        tmp_path / "run",
        *_niah_fixture(tmp_path),
        *_sealed_dir(tmp_path, monkeypatch, "jfk"),
    )
    with pytest.raises(ValueError, match="overlap sealed-600"):
        train_relations.main(argv)


def test_plan_only_builds_the_folds_without_fitting_anything(tmp_path: Path) -> None:
    """Runnable on a login node: the whole data path can fail before a GPU hour is spent."""
    output = tmp_path / "run"
    assert train_relations.main(_argv(_corpus(tmp_path), output, "--plan-only")) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["is_plan_only"] is True
    assert manifest["folds"] == []
    assert manifest["chain"]["fold_sizes"] == [3, 3, 2, 2, 2]
    assert not (output / "oof_predictions.jsonl").exists()


def test_a_smoke_run_is_marked_as_one(tmp_path: Path) -> None:
    """A truncated chain must not be readable later as the real run."""
    output = tmp_path / "run"
    train_relations.main(_argv(_corpus(tmp_path), output, "--max-examples", "10"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["is_smoke_run"] is True
    assert manifest["chain"]["n_examples"] == 10


def test_the_manifest_carries_the_line_needed_to_score_the_checkpoint(tmp_path: Path) -> None:
    """§11.10 item 7: a base id and its id2label reach the code through the registry, never
    hardcoded at a call site. A trained checkpoint's id is run-specific, so the run emits the
    exact LABEL_ORDER line and a human adds it — visible and reviewable."""
    output = tmp_path / "run"
    train_relations.main(_argv(_corpus(tmp_path), output))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert "'REFUTES', 'SUPPORTS', 'UNKNOWN'" in manifest["gate0b_registry_line"]
    assert str(output) in manifest["gate0b_registry_line"]


def test_the_manifest_records_the_frozen_3_8_b_hyperparameters(tmp_path: Path) -> None:
    """§3.8(b) freezes every one of these. They are pinned here because the protocol says no
    legal tuning surface exists for them -- dev is reserved by §2.4 and §11.2a and this path
    trains on train only -- so a diff to any of them is a protocol change, not a code change."""
    output = tmp_path / "run"
    train_relations.main(_argv(_corpus(tmp_path), output))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hyperparameters"] == {
        "optimizer": "adamw",
        "schedule": "linear_decay",
        "epochs": 2,
        "learning_rate": 2e-5,
        "batch_size": 32,
        "warmup_ratio": 0.06,
        "max_length": 256,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "bf16": True,
    }
    assert "§3.8(b)" in manifest["hyperparameters_note"]


def test_no_flag_can_move_a_frozen_hyperparameter(tmp_path: Path) -> None:
    """§3.8(b) leaves nothing to tune, so the CLI must offer no door. argparse exits 2 on an
    unrecognised flag; a run that silently ignored --epochs would be worse than one that stops."""
    with pytest.raises(SystemExit):
        train_relations.main(_argv(_corpus(tmp_path), tmp_path / "run", "--epochs", "5"))
