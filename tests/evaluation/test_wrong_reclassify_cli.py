import json
from pathlib import Path

from evidence_rag.evaluation.wrong_reclassify_cli import main


def _dump(tmp_path: Path) -> Path:
    rows = [
        {"query_id": "a", "question": "q", "gold_value": "paul", "extracted": "Apostle Paul",
         "visible": True, "outcome": "wrong"},
        {"query_id": "b", "question": "q", "gold_value": "linda davis",
         "extracted": "Reba McEntire", "visible": True, "outcome": "wrong"},
        {"query_id": "c", "question": "q", "gold_value": "kennedy", "extracted": "kennedy",
         "visible": True, "outcome": "recovered"},  # not wrong -> excluded
        {"query_id": "d", "question": "q", "gold_value": "leu", "extracted": "Moldovan leu",
         "visible": False, "outcome": "wrong"},  # not visible -> excluded by default
    ]
    path = tmp_path / "probe_dump.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_cli_reports_recoverable_over_visible_wrong(tmp_path, capsys):
    output = tmp_path / "reclass.json"
    exit_code = main(["--dump", str(_dump(tmp_path)), "--output", str(output)])

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    # only the two visible wrong rows count; paul is recoverable, reba is different
    assert report["n"] == 2
    assert report["gold_in_extracted"] == 1
    assert report["different"] == 1
    assert report["recoverable"] == 1
    assert report["recoverable_rate"] == 0.5
    printed = capsys.readouterr().out
    assert "DIFFERENT residual" in printed


def test_cli_all_includes_non_visible(tmp_path):
    output = tmp_path / "reclass_all.json"
    main(["--dump", str(_dump(tmp_path)), "--output", str(output), "--all"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["n"] == 3  # paul, reba, moldovan leu
    assert report["gold_in_extracted"] == 2  # paul + moldovan leu
