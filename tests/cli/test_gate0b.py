import json
from pathlib import Path

import pytest

from evidence_rag.cli.gate0b import LABEL_ORDER, load_score_fn, main


def _write(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _task_pairs(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "task.jsonl",
        [
            {"premise": "Kennedy won", "hypothesis": "the answer is Kennedy",
             "label": "SUPPORTS", "group": "g1", "kind": "needle_gold", "query_id": "q1"},
            {"premise": "Nixon won", "hypothesis": "the answer is Kennedy",
             "label": "REFUTES", "group": "g1", "kind": "cf_gold", "query_id": "q1"},
            {"premise": "Kennedy won", "hypothesis": "the answer is Nixon",
             "label": "REFUTES", "group": "g1", "kind": "needle_replacement", "query_id": "q1"},
        ],
    )


def _perfect_scorer(model_id: str):  # type: ignore[no-untyped-def]
    """Predicts SUPPORTS when the premise's subject appears in the hypothesis, else REFUTES."""

    def score(pairs):  # type: ignore[no-untyped-def]
        return [
            {"SUPPORTS": 1.0, "REFUTES": 0.0, "UNKNOWN": 0.0}
            if premise.split()[0] in hypothesis
            else {"SUPPORTS": 0.0, "REFUTES": 1.0, "UNKNOWN": 0.0}
            for premise, hypothesis in pairs
        ]

    return score


def test_runner_scores_every_model_in_one_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _perfect_scorer)
    output = tmp_path / "gate0b.json"
    assert (
        main([
            "--task-pairs", str(_task_pairs(tmp_path)),
            "--output", str(output),
            "--models", "m1", "m2", "m3",
        ])
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload["models"]) == {"m1", "m2", "m3"}
    assert payload["n_task_pairs"] == 3
    assert payload["models"]["m1"]["task"]["twin_refutes_accuracy"] == 1.0
    assert payload["models"]["m1"]["task"]["gold_supports_recall"] == 1.0
    assert payload["models"]["m1"]["task"]["failures"] == []


def test_external_tier_is_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _perfect_scorer)
    output = tmp_path / "gate0b.json"
    main([
        "--task-pairs", str(_task_pairs(tmp_path)),
        "--output", str(output),
        "--models", "m1",
    ])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "external" not in payload["models"]["m1"]
    assert payload["n_external_pairs"] == 0


def test_external_tier_is_scored_when_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _perfect_scorer)
    external = _write(
        tmp_path / "external.jsonl",
        [
            {"premise": "Kennedy won", "hypothesis": "the answer is Kennedy",
             "label": "SUPPORTS", "group": "p1"},
            {"premise": "Nixon won", "hypothesis": "the answer is Kennedy",
             "label": "REFUTES", "group": "p2"},
        ],
    )
    output = tmp_path / "gate0b.json"
    main([
        "--task-pairs", str(_task_pairs(tmp_path)),
        "--external-pairs", str(external),
        "--output", str(output),
        "--models", "m1",
    ])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["models"]["m1"]["external"]["n"] == 2
    assert payload["models"]["m1"]["external"]["failures"] == []


def test_load_score_fn_rejects_an_unverified_checkpoint() -> None:
    """A wrong label order silently swaps REFUTES and SUPPORTS while the numbers stay
    plausible, so guessing is not an option."""
    with pytest.raises(ValueError, match="no verified label order"):
        load_score_fn("some/unknown-model")


def test_label_order_covers_the_planned_checkpoints() -> None:
    assert "tals/albert-xlarge-vitaminc-mnli" in LABEL_ORDER
    for order in LABEL_ORDER.values():
        assert sorted(order) == ["REFUTES", "SUPPORTS", "UNKNOWN"]
