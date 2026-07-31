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
