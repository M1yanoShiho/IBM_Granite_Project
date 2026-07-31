import json
from pathlib import Path

import pytest

from evidence_rag.cli.export_vitaminc import main


def _fake_splits() -> dict[str, list[dict[str, str]]]:
    return {
        "train": [
            {"claim": "c1", "evidence": "e1", "label": "SUPPORTS", "page": "shared"},
            {"claim": "c2", "evidence": "e2", "label": "SUPPORTS", "page": "safe"},
        ],
        "validation": [
            {"claim": "c3", "evidence": "e3", "label": "REFUTES", "page": "shared"},
        ],
        "test": [
            {"claim": "c4", "evidence": "e4", "label": "REFUTES", "page": "t1"},
            {"claim": "c5", "evidence": "e5", "label": "NOT ENOUGH INFO", "page": "t2"},
        ],
    }


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evidence_rag.cli.export_vitaminc.load_splits", _fake_splits)


def test_exports_official_test_pairs_and_a_decontamination_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch(monkeypatch)
    out_test = tmp_path / "vitaminc_test.jsonl"
    removed = tmp_path / "removed.json"
    assert main(["--out-test", str(out_test), "--removed-log", str(removed)]) == 0

    rows = [json.loads(line) for line in out_test.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 2
    assert rows[0] == {"premise": "e4", "hypothesis": "c4", "label": "REFUTES", "group": "t1"}
    assert rows[1]["label"] == "UNKNOWN"

    report = json.loads(capsys.readouterr().out)
    assert report["n_test"] == 2
    assert report["removed_train_groups"] == ["shared"]
    assert json.loads(removed.read_text(encoding="utf-8")) == report


def test_train_and_dev_exports_are_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch)
    unexpected = tmp_path / "train.jsonl"
    main([
        "--out-test", str(tmp_path / "test.jsonl"),
        "--removed-log", str(tmp_path / "removed.json"),
    ])
    assert not unexpected.exists()


def test_decontaminated_train_is_written_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch)
    out_train = tmp_path / "train.jsonl"
    main([
        "--out-test", str(tmp_path / "test.jsonl"),
        "--out-train", str(out_train),
        "--removed-log", str(tmp_path / "removed.json"),
    ])
    rows = [json.loads(line) for line in out_train.read_text(encoding="utf-8").splitlines() if line]
    assert [row["group"] for row in rows] == ["safe"]
