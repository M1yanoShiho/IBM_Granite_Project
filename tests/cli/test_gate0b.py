import json
from pathlib import Path

import pytest

from evidence_rag.cli.gate0b import LABEL_ORDER, _load_tokenizer, load_score_fn, main


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

    return score, "fake"


def _external_pairs(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "external.jsonl",
        [
            {"premise": "Kennedy won", "hypothesis": "the answer is Kennedy",
             "label": "SUPPORTS", "group": "p1"},
            {"premise": "Nixon won", "hypothesis": "the answer is Kennedy",
             "label": "REFUTES", "group": "p2"},
        ],
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


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


def test_label_order_matches_the_verified_id2label_of_each_checkpoint() -> None:
    """Pins the ORDER, not just completeness.

    Verified against the real checkpoints on 2026-07-30:
      tals/albert-xlarge-vitaminc-mnli -> {0: SUPPORTS, 1: REFUTES, 2: NOT ENOUGH INFO}
      MoritzLaurer/DeBERTa-...-wanli   -> {0: entailment, 1: neutral, 2: contradiction}

    A test that only checked the three labels are present would let a future wrong entry swap
    REFUTES and SUPPORTS while every downstream number stayed plausible.
    """
    assert LABEL_ORDER["tals/albert-xlarge-vitaminc-mnli"] == (
        "SUPPORTS", "REFUTES", "UNKNOWN"
    )
    assert LABEL_ORDER["MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"] == (
        "SUPPORTS", "UNKNOWN", "REFUTES"
    )
    for order in LABEL_ORDER.values():
        assert sorted(order) == ["REFUTES", "SUPPORTS", "UNKNOWN"]


class _FakeTransformers:
    """Reproduces the observed failure: the fast tokenizer raises, the slow one loads."""

    class AutoTokenizer:
        fast_error: Exception | None = AttributeError(
            "'NoneType' object has no attribute 'endswith'"
        )
        slow_error: Exception | None = None

        @classmethod
        def from_pretrained(cls, model_id: str, use_fast: bool = True):  # type: ignore[no-untyped-def]
            if use_fast:
                if cls.fast_error is not None:
                    raise cls.fast_error
                return "fast-tokenizer"
            if cls.slow_error is not None:
                raise cls.slow_error
            return "slow-tokenizer"


def test_tokenizer_falls_back_to_slow_and_reports_it() -> None:
    """ALBERT ships no tokenizer.json, so the on-the-fly fast conversion fails. The fallback is
    safe, but which variant ran must be reported — it feeds a go/no-go decision."""
    tokenizer, variant = _load_tokenizer(_FakeTransformers, "tals/albert-xlarge-vitaminc-mnli")
    assert tokenizer == "slow-tokenizer"
    assert variant == "slow"


def test_tokenizer_prefers_fast_when_it_works() -> None:
    class Works(_FakeTransformers):
        class AutoTokenizer(_FakeTransformers.AutoTokenizer):
            fast_error = None

    tokenizer, variant = _load_tokenizer(Works, "any/model")
    assert tokenizer == "fast-tokenizer"
    assert variant == "fast"


def test_both_tokenizer_paths_failing_raises_with_both_causes() -> None:
    """A checkpoint that loads under neither path must stop the sweep, not run untokenized."""

    class Broken(_FakeTransformers):
        class AutoTokenizer(_FakeTransformers.AutoTokenizer):
            slow_error = OSError("no sentencepiece")

    with pytest.raises(RuntimeError, match="could not load a tokenizer"):
        _load_tokenizer(Broken, "any/model")


def test_identical_failures_are_diagnosed_as_a_missing_backend() -> None:
    """AutoTokenizer silently reuses the FAST class when the slow one cannot be imported, so
    use_fast=False becomes a no-op and both attempts fail the same way. That signature means a
    missing backend, not a broken checkpoint, and the error must say so."""

    error = AttributeError("'NoneType' object has no attribute 'endswith'")

    class NoSlowClass(_FakeTransformers):
        class AutoTokenizer(_FakeTransformers.AutoTokenizer):
            fast_error = error
            slow_error = error

    with pytest.raises(RuntimeError, match="sentencepiece"):
        _load_tokenizer(NoSlowClass, "tals/albert-xlarge-vitaminc-mnli")


def test_tokenizer_variant_is_recorded_in_the_sweep_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _perfect_scorer)
    output = tmp_path / "gate0b.json"
    main([
        "--task-pairs", str(_task_pairs(tmp_path)),
        "--output", str(output),
        "--models", "m1",
    ])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["models"]["m1"]["tokenizer_variant"] == "fake"


def _always_unknown_scorer(model_id: str):  # type: ignore[no-untyped-def]
    """Disagrees with every gold label, so gold and predicted cannot be confused for each other."""

    def score(pairs):  # type: ignore[no-untyped-def]
        return [{"SUPPORTS": 0.1, "REFUTES": 0.2, "UNKNOWN": 0.7} for _ in pairs]

    return score, "fake"


def test_dump_writes_one_row_per_model_and_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The aggregate reports how many pairs failed but not which ones, and the anomaly the sweep
    surfaced can only be read per pair."""
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _perfect_scorer)
    dump = tmp_path / "dump.jsonl"
    assert (
        main([
            "--task-pairs", str(_task_pairs(tmp_path)),
            "--external-pairs", str(_external_pairs(tmp_path)),
            "--output", str(tmp_path / "gate0b.json"),
            "--dump", str(dump),
            "--models", "m1", "m2",
        ])
        == 0
    )
    rows = _read_jsonl(dump)
    assert len(rows) == 2 * (3 + 2)
    assert [row["model_id"] for row in rows].count("m1") == 5
    assert [row["model_id"] for row in rows].count("m2") == 5
    assert {row["tier"] for row in rows} == {"task", "external"}


def test_dump_rows_carry_tier_kind_query_id_and_class_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`kind` is the breakdown the dump exists for; external rows have none and say so with null
    rather than omitting the key, so every row shares one schema."""
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _perfect_scorer)
    dump = tmp_path / "dump.jsonl"
    main([
        "--task-pairs", str(_task_pairs(tmp_path)),
        "--external-pairs", str(_external_pairs(tmp_path)),
        "--output", str(tmp_path / "gate0b.json"),
        "--dump", str(dump),
        "--models", "m1",
    ])
    rows = _read_jsonl(dump)
    task = [row for row in rows if row["tier"] == "task"]
    external = [row for row in rows if row["tier"] == "external"]
    assert [row["kind"] for row in task] == ["needle_gold", "cf_gold", "needle_replacement"]
    assert [row["query_id"] for row in task] == ["q1", "q1", "q1"]
    assert [row["kind"] for row in external] == [None, None]
    assert all("query_id" not in row for row in external)
    assert [row["gold"] for row in task] == ["SUPPORTS", "REFUTES", "REFUTES"]
    assert [row["predicted"] for row in task] == ["SUPPORTS", "REFUTES", "REFUTES"]
    assert task[0]["probabilities"] == {"SUPPORTS": 1.0, "REFUTES": 0.0, "UNKNOWN": 0.0}
    assert task[1]["probabilities"] == {"SUPPORTS": 0.0, "REFUTES": 1.0, "UNKNOWN": 0.0}


def test_dump_records_the_prediction_and_not_a_second_copy_of_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A perfect scorer makes gold and predicted identical on every row, so a dump that echoed
    the gold label back would still look correct. This pins the two to different sources."""
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _always_unknown_scorer)
    dump = tmp_path / "dump.jsonl"
    main([
        "--task-pairs", str(_task_pairs(tmp_path)),
        "--external-pairs", str(_external_pairs(tmp_path)),
        "--output", str(tmp_path / "gate0b.json"),
        "--dump", str(dump),
        "--models", "m1",
    ])
    rows = _read_jsonl(dump)
    assert {row["gold"] for row in rows} == {"SUPPORTS", "REFUTES"}
    assert {row["predicted"] for row in rows} == {"UNKNOWN"}
    assert all(
        row["probabilities"] == {"SUPPORTS": 0.1, "REFUTES": 0.2, "UNKNOWN": 0.7} for row in rows
    )


def test_aggregate_output_is_byte_identical_with_and_without_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dump is a diagnostic read-out. If turning it on could move the reported metrics, the
    gate's go/no-go numbers would depend on how the job was launched."""
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _perfect_scorer)
    base = [
        "--task-pairs", str(_task_pairs(tmp_path)),
        "--external-pairs", str(_external_pairs(tmp_path)),
        "--models", "m1", "m2",
    ]
    plain = tmp_path / "plain.json"
    dumped = tmp_path / "dumped.json"
    dump = tmp_path / "dump.jsonl"
    main([*base, "--output", str(plain)])
    assert not dump.exists()
    main([*base, "--output", str(dumped), "--dump", str(dump)])
    assert dump.exists()
    assert plain.read_bytes() == dumped.read_bytes()


def test_dump_creates_its_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _perfect_scorer)
    dump = tmp_path / "nested" / "deeper" / "dump.jsonl"
    main([
        "--task-pairs", str(_task_pairs(tmp_path)),
        "--output", str(tmp_path / "gate0b.json"),
        "--dump", str(dump),
        "--models", "m1",
    ])
    assert len(_read_jsonl(dump)) == 3
