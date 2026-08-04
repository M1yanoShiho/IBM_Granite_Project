import json
from pathlib import Path

import pytest

from evidence_rag.cli.recompute_binary import main, recompute_from_dump


def _dump(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _three_class_rows() -> list[dict[str, object]]:
    """A pre-A1 dump: predictions still carry REFUTES and UNKNOWN."""
    return [
        {"model_id": "m1", "tier": "task", "kind": "needle_gold",
         "gold": "SUPPORTS", "predicted": "SUPPORTS"},
        {"model_id": "m1", "tier": "task", "kind": "needle_gold",
         "gold": "SUPPORTS", "predicted": "UNKNOWN"},
        {"model_id": "m1", "tier": "task", "kind": "cf_gold",
         "gold": "REFUTES", "predicted": "REFUTES"},
        {"model_id": "m1", "tier": "task", "kind": "needle_replacement",
         "gold": "REFUTES", "predicted": "UNKNOWN"},
        {"model_id": "m1", "tier": "task", "kind": "cf_replacement",
         "gold": "SUPPORTS", "predicted": "SUPPORTS"},
        {"model_id": "m1", "tier": "external", "kind": None,
         "gold": "SUPPORTS", "predicted": "SUPPORTS"},
    ]


def test_recomputes_the_binary_task_report_from_a_pre_A1_dump(tmp_path: Path) -> None:
    """M0 §9.10a rules that R012 and R012b are recomputed from their dumps rather than re-run.
    Nothing implemented that ruling, so it was an obligation with no mechanism — the same gap
    the casing-rate requirement had. A three-class prediction column is the input here, not an
    error: both A1 metrics are defined against SUPPORTS, never against gold."""
    reports = recompute_from_dump(_dump(tmp_path / "dump.jsonl", _three_class_rows()))
    report = reports["m1"]
    # twin = cf_gold + needle_replacement; neither predicted SUPPORTS, so both count as
    # not-supported even though one of them said UNKNOWN.
    assert report.n_twin == 2
    assert report.twin_not_supported_accuracy == pytest.approx(1.0)
    # gold = needle_gold only; one of the two said SUPPORTS.
    assert report.n_gold == 2
    assert report.gold_supports_recall == pytest.approx(0.5)


def test_the_external_tier_is_ignored(tmp_path: Path) -> None:
    """0B-1 is suspended (M0 §9.11) and its rows carry no `kind`, so they are not scorable by
    the task report at all."""
    reports = recompute_from_dump(_dump(tmp_path / "dump.jsonl", _three_class_rows()))
    assert reports["m1"].n_twin + reports["m1"].n_gold == 4


def test_gold_supports_recall_must_match_the_original_sweep(tmp_path: Path) -> None:
    """The coherence check that makes §9.10a's equivalence claim verifiable rather than asserted.
    A1 §9.1 pins `gold_supports_recall` as UNCHANGED, so a recomputation that moves it has a bug
    — most likely a collapse by sum, which is a threshold at .5 and silently reclassifies every
    pair whose winning probability was below it."""
    dump = _dump(tmp_path / "dump.jsonl", _three_class_rows())
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps({"models": {"m1": {"task": {"gold_supports_recall": 0.5}}}}), encoding="utf-8"
    )
    assert main(["--dump", str(dump), "--against", str(sweep)]) == 0


def test_a_disagreeing_sweep_is_a_loud_failure_not_a_warning(tmp_path: Path) -> None:
    """If the historical number cannot be reproduced, every binary reading published from these
    dumps is suspect. That is not something to print and carry on from."""
    dump = _dump(tmp_path / "dump.jsonl", _three_class_rows())
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps({"models": {"m1": {"task": {"gold_supports_recall": 0.99}}}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="gold_supports_recall"):
        main(["--dump", str(dump), "--against", str(sweep)])


def test_report_is_written_to_stdout_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dump = _dump(tmp_path / "dump.jsonl", _three_class_rows())
    assert main(["--dump", str(dump)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_dump"].endswith("dump.jsonl")
    assert payload["models"]["m1"]["gold_supports_recall"] == pytest.approx(0.5)
    assert payload["models"]["m1"]["twin_not_supported_accuracy"] == pytest.approx(1.0)
