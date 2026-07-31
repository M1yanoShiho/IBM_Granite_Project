import json
from pathlib import Path

from evidence_rag.evaluation.cluster_rescore_cli import main


def _dump(tmp_path: Path) -> Path:
    dump = tmp_path / "dump.jsonl"
    dump.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "needle_document_id": "needle",
                "counterfactual_document_id": "cf::needle",
                "gold_value": "Kennedy",
                "gold_aliases": ["Kennedy"],
                "window": [
                    {"evidence_id": "e0", "document_id": "needle", "retrieval_rank": 1,
                     "answer": "Kennedy", "contains_gold_alias": True},
                    {"evidence_id": "e1", "document_id": "cf::needle", "retrieval_rank": 2,
                     "answer": "Nixon", "contains_gold_alias": False},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return dump


def test_cli_emits_both_scorings(tmp_path: Path) -> None:
    output = tmp_path / "rescore.json"
    assert main(["--dump", str(_dump(tmp_path)), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["n_rows"] == 1
    # no second gold-alias passage in this window, so the fixed-denominator metric is not scored
    assert payload["exact"]["fixed_false_conflict"]["n_scored"] == 0
    assert payload["exact"]["n_fixed_eligible"] == 0
    assert payload["lenient"]["needle_gold_recovery"]["rate"] == 1.0
    assert payload["exact"]["missed_conflict"]["rate"] == 0.0


def test_cli_writes_per_query_maps_when_asked(tmp_path: Path) -> None:
    output = tmp_path / "rescore.json"
    per_query = tmp_path / "per_query.json"
    assert (
        main(
            [
                "--dump", str(_dump(tmp_path)),
                "--output", str(output),
                "--per-query", str(per_query),
            ]
        )
        == 0
    )
    mapping = json.loads(per_query.read_text(encoding="utf-8"))
    assert mapping["lenient"]["needle_gold_recovery"]["q1"] == 1.0
    # unscored metrics must survive as null so paired testing can drop them
    assert mapping["exact"]["fixed_false_conflict"]["q1"] is None


def test_per_query_is_optional(tmp_path: Path) -> None:
    output = tmp_path / "rescore.json"
    unexpected = tmp_path / "per_query.json"
    assert main(["--dump", str(_dump(tmp_path)), "--output", str(output)]) == 0
    assert not unexpected.exists()
