import json
from pathlib import Path

from evidence_rag.evaluation.paired_metric_cli import main

METRIC = "selector.core.conditional_document_recall"


def _report(path: Path, values: dict[str, float | None]) -> Path:
    per_case = [
        {"query_id": qid, "schema_version": "1.0",
         "metrics": {METRIC: {"reason": None, "value": value}}}
        for qid, value in values.items()
    ]
    path.write_text(json.dumps({"per_case": per_case, "stage": "selector"}), encoding="utf-8")
    return path


def test_cli_pairs_metric_and_prints_stats(tmp_path, capsys) -> None:
    on = _report(tmp_path / "on.json", {"q1": 1.0, "q2": 1.0, "q3": 0.0})
    off = _report(tmp_path / "off.json", {"q1": 0.0, "q2": 1.0, "q3": 0.0})

    exit_code = main(
        ["--on-report", str(on), "--off-report", str(off), "--metric", METRIC, "--iterations", "2000"]
    )

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metric"] == METRIC
    assert out["n_paired"] == 3
    assert abs(out["delta"] - 1 / 3) < 1e-9
    assert 0.0 <= out["p_value"] <= 1.0
