import json
from pathlib import Path

import pytest

from evidence_rag.cli.gate0b import (
    LABEL_ORDER,
    _collapse_to_binary,
    _load_tokenizer,
    _weight_buffers,
    load_score_fn,
    main,
)
from evidence_rag.relations.minicheck import (
    MINICHECK_FLAN_T5_LARGE,
    VERIFIED_BINARY_PROTOCOLS,
)


def _write(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _task_pairs(tmp_path: Path) -> Path:
    """Twin rows carry NOT_SUPPORTED, the A1 gold label emitted by `task_probe.build_probe_pairs`.

    The external fixture below still carries REFUTES: 0B-1 reads VitaminC official gold and A1
    §9.1 changed only the 0B-2 metric, so the two tiers deliberately speak different label sets.
    """
    return _write(
        tmp_path / "task.jsonl",
        [
            {"premise": "Kennedy won", "hypothesis": "the answer is Kennedy",
             "label": "SUPPORTS", "group": "g1", "kind": "needle_gold", "query_id": "q1"},
            {"premise": "Nixon won", "hypothesis": "the answer is Kennedy",
             "label": "NOT_SUPPORTED", "group": "g1", "kind": "cf_gold", "query_id": "q1"},
            {"premise": "Kennedy won", "hypothesis": "the answer is Nixon",
             "label": "NOT_SUPPORTED", "group": "g1", "kind": "needle_replacement",
             "query_id": "q1"},
        ],
    )


def _perfect_scorer(model_id: str):  # type: ignore[no-untyped-def]
    """SUPPORTS when the premise's subject appears in the hypothesis, else NOT_SUPPORTED.

    Speaks A1's binary contract, which is what `load_score_fn` returns after collapsing a
    checkpoint's three classes."""

    def score(pairs):  # type: ignore[no-untyped-def]
        return [
            {"SUPPORTS": 1.0, "NOT_SUPPORTED": 0.0}
            if premise.split()[0] in hypothesis
            else {"SUPPORTS": 0.0, "NOT_SUPPORTED": 1.0}
            for premise, hypothesis in pairs
        ]

    return score, "fake", "fake@0123456789abcdef"


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
    assert payload["models"]["m1"]["task"]["twin_not_supported_accuracy"] == 1.0
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


def test_0b1_is_UNRUNNABLE_after_A1_and_this_test_records_it_rather_than_fixing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OPEN PROTOCOL GAP — needs a human ruling, do not "fix" this test.

    A1 §9.1 changes the 0B-2 metric and says NOTHING about 0B-1, but three of 0B-1's five frozen
    thresholds are defined on the three-class space. Feeding VitaminC official gold (SUPPORTS /
    REFUTES / NEI) to a model that can only emit SUPPORTS / NOT_SUPPORTED gives, on a scorer
    that is perfectly correct about the underlying relation:

        refutes_precision    0.0  -> automatic FAIL (the model can never predict REFUTES)
        refutes_coverage     0.0  -> automatic FAIL (same)
        macro_f1             0.5  -> automatic FAIL (REFUTES f1 is structurally 0, capping it)
        non_unknown_coverage 1.0  -> VACUOUS PASS   (the model can never predict UNKNOWN)
        support_coverage     1.0  -> the only threshold still measuring anything

    The vacuous pass is the dangerous one: it reports .80+ forever while measuring nothing.

    `external_report`, its thresholds and `vitaminc.py` are deliberately left exactly as they
    were — inventing a gold mapping here would be deciding an unapproved protocol question, and
    §9.5a's pre-registered threshold remedy is triggered by "REFUTES precision < .85", which
    cannot even be evaluated once REFUTES is unreachable.
    """
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _perfect_scorer)
    output = tmp_path / "gate0b.json"
    main([
        "--task-pairs", str(_task_pairs(tmp_path)),
        "--external-pairs", str(_external_pairs(tmp_path)),
        "--output", str(output),
        "--models", "m1",
    ])
    external = json.loads(output.read_text(encoding="utf-8"))["models"]["m1"]["external"]
    assert external["refutes_precision"] == 0.0
    assert external["refutes_coverage"] == 0.0
    assert external["macro_f1"] == 0.5
    assert external["non_unknown_coverage"] == 1.0
    assert external["support_coverage"] == 1.0
    assert sorted(external["failures"]) == ["macro_f1", "refutes_coverage", "refutes_precision"]


def test_three_class_logits_collapse_by_max_so_binary_argmax_equals_relabelling() -> None:
    """THE load-bearing case. A1 §9.10a says recomputing the binary reading from a dump is
    equivalent to a natively-binary model, and the dump only carries the three-class ARGMAX. So
    the collapse must be the one that makes `argmax(SUPPORTS, NOT_SUPPORTED)` identical to
    `argmax(SUPPORTS, REFUTES, UNKNOWN)` followed by relabelling — that is max, not sum.

    Here they disagree: max keeps SUPPORTS (.40 > .35), sum flips to NOT_SUPPORTED (.60 > .40).
    Summing would silently impose a threshold at P(SUPPORTS) > .5, which is the very thing
    §9.5a forbids, and it would move `gold_supports_recall`, which §9.1's change table pins as
    UNCHANGED. The published binary twin readings (.9980 / .9871 / .8689 / .9008) reproduce
    under max and not under sum.
    """
    assert _collapse_to_binary((0.40, 0.35, 0.25), ("SUPPORTS", "REFUTES", "UNKNOWN")) == {
        "SUPPORTS": 0.40,
        "NOT_SUPPORTED": 0.35,
    }


def test_collapse_reads_the_per_checkpoint_label_order() -> None:
    """DeBERTa's head is (entailment, neutral, contradiction), so position 1 is UNKNOWN and
    position 2 is REFUTES. Collapsing positionally instead of by verified name would swap
    REFUTES and UNKNOWN — invisible after the collapse, but it corrupts the recorded
    confidence."""
    order = LABEL_ORDER["MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"]
    assert _collapse_to_binary((0.1, 0.2, 0.7), order) == {
        "SUPPORTS": 0.1,
        "NOT_SUPPORTED": 0.7,
    }


def test_load_score_fn_rejects_an_unverified_checkpoint() -> None:
    """A wrong label order silently swaps REFUTES and SUPPORTS while the numbers stay
    plausible, so guessing is not an option."""
    with pytest.raises(ValueError, match="no verified label order"):
        load_score_fn("some/unknown-model")


def test_load_score_fn_names_both_registries_when_an_id_is_in_neither() -> None:
    """There are now two ways to be verified and an id must fail against both.

    A message naming only LABEL_ORDER would send whoever hits it to add a three-name order for a
    checkpoint that has no classes to order — which is exactly the fabrication the binary path
    exists to prevent.
    """
    with pytest.raises(ValueError, match="no verified binary protocol"):
        load_score_fn("some/unknown-model")


def test_a_natively_binary_checkpoint_never_gets_a_three_class_label_order() -> None:
    """The registries are disjoint, and this is the guard that keeps them so.

    MiniCheck is a `T5ForConditionalGeneration` with no `id2label`. Adding it to LABEL_ORDER
    would require inventing a REFUTES/UNKNOWN split it cannot express, and `_collapse_to_binary`
    would then read those invented columns and return a number for them.
    """
    assert MINICHECK_FLAN_T5_LARGE not in LABEL_ORDER
    assert not set(LABEL_ORDER) & set(VERIFIED_BINARY_PROTOCOLS)


def test_load_score_fn_dispatches_each_checkpoint_to_its_own_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch is by registry membership, not by a name heuristic."""
    monkeypatch.setattr(
        "evidence_rag.cli.gate0b._load_three_class_score_fn",
        lambda model_id: ("three-class", model_id, "v"),
    )
    monkeypatch.setattr(
        "evidence_rag.cli.gate0b._load_binary_score_fn",
        lambda model_id: ("binary", model_id, "v"),
    )
    assert load_score_fn("tals/albert-xlarge-vitaminc-mnli")[0] == "three-class"
    assert load_score_fn(MINICHECK_FLAN_T5_LARGE)[0] == "binary"


def test_the_binary_arm_scores_through_the_unchanged_metrics_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A natively-binary arm and a three-class arm in ONE sweep, reported identically.

    This is the seam the arm was designed around (M0 §9.10a's exception): the binary checkpoint
    is reported as an independent arm, but `task_report` and its thresholds do not learn that it
    exists. Both entries must therefore carry the same report fields — if the binary arm needed
    its own metric keys, the two dicts would differ here.
    """
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _perfect_scorer)
    output = tmp_path / "sweep.json"
    assert (
        main(
            [
                "--task-pairs", str(_task_pairs(tmp_path)),
                "--output", str(output),
                "--models", "tals/albert-xlarge-vitaminc-mnli", MINICHECK_FLAN_T5_LARGE,
            ]
        )
        == 0
    )
    models = json.loads(output.read_text(encoding="utf-8"))["models"]
    assert set(models) == {"tals/albert-xlarge-vitaminc-mnli", MINICHECK_FLAN_T5_LARGE}
    three_class, binary = (models[name]["task"] for name in models)
    assert three_class.keys() == binary.keys()
    assert "twin_not_supported_accuracy" in binary
    assert "gold_supports_recall" in binary


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


def _always_not_supported_scorer(model_id: str):  # type: ignore[no-untyped-def]
    """Abstains on every pair. Under A1 abstention and contradiction are the same output, so
    this also stands in for the old always-UNKNOWN degenerate model."""

    def score(pairs):  # type: ignore[no-untyped-def]
        return [{"SUPPORTS": 0.3, "NOT_SUPPORTED": 0.7} for _ in pairs]

    return score, "fake", "fake@0123456789abcdef"


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
    assert [row["gold"] for row in task] == ["SUPPORTS", "NOT_SUPPORTED", "NOT_SUPPORTED"]
    assert [row["predicted"] for row in task] == ["SUPPORTS", "NOT_SUPPORTED", "NOT_SUPPORTED"]
    assert task[0]["probabilities"] == {"SUPPORTS": 1.0, "NOT_SUPPORTED": 0.0}
    assert task[1]["probabilities"] == {"SUPPORTS": 0.0, "NOT_SUPPORTED": 1.0}


def test_dump_records_the_prediction_and_not_a_second_copy_of_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A perfect scorer makes gold and predicted identical on every row, so a dump that echoed
    the gold label back would still look correct. This pins the two to different sources."""
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _always_not_supported_scorer)
    dump = tmp_path / "dump.jsonl"
    main([
        "--task-pairs", str(_task_pairs(tmp_path)),
        "--external-pairs", str(_external_pairs(tmp_path)),
        "--output", str(tmp_path / "gate0b.json"),
        "--dump", str(dump),
        "--models", "m1",
    ])
    rows = _read_jsonl(dump)
    assert {row["gold"] for row in rows} == {"SUPPORTS", "NOT_SUPPORTED", "REFUTES"}
    assert {row["predicted"] for row in rows} == {"NOT_SUPPORTED"}
    assert all(
        row["probabilities"] == {"SUPPORTS": 0.3, "NOT_SUPPORTED": 0.7} for row in rows
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


def test_weight_buffers_fold_name_shape_and_dtype_into_the_hashed_spec() -> None:
    """Two fine-tuning seeds share every name, shape and dtype, so the VALUES have to reach the
    hash. Shape and dtype ride along in the spec so a reshape or a dtype change cannot produce
    the same fingerprint from the same bytes."""

    class _Tensor:
        shape = (2, 2)
        dtype = "float32"

        def detach(self) -> "_Tensor":
            return self

        def cpu(self) -> "_Tensor":
            return self

        def contiguous(self) -> "_Tensor":
            return self

        def numpy(self) -> "_Tensor":
            return self

        def tobytes(self) -> bytes:
            return b"\x01\x02\x03\x04"

    class _Model:
        def state_dict(self) -> dict[str, "_Tensor"]:
            return {"encoder.weight": _Tensor()}

    assert list(_weight_buffers(_Model())) == [
        ("encoder.weight|(2, 2)|float32", b"\x01\x02\x03\x04")
    ]


def test_a_0b1_report_carries_its_own_health_warning_in_the_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 0B-1 numbers are meaningless under g2-proto-2 but they are still emitted, because the
    pending ruling needs to see them. Meaningless numbers that travel without a marker get read
    as results — this project has already lost a day to output that looked like other output. So
    the warning rides inside the artefact rather than in a log line someone may not scroll to.

    This does NOT pick between the ruling's options; it only refuses to let the numbers travel
    silently. Remove it when 0B-1 is amended, not before.
    """
    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _perfect_scorer)
    output = tmp_path / "sweep.json"
    main([
        "--task-pairs", str(_task_pairs(tmp_path)),
        "--external-pairs", str(_external_pairs(tmp_path)),
        "--output", str(output),
        "--models", "m1",
    ])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "external_tier_status" in payload
    assert "g2-proto-2" in payload["external_tier_status"]
    assert "not runnable" in payload["external_tier_status"].lower()

    monkeypatch.setattr("evidence_rag.cli.gate0b.load_score_fn", _perfect_scorer)
    task_only = tmp_path / "task_only.json"
    main([
        "--task-pairs", str(_task_pairs(tmp_path)),
        "--output", str(task_only),
        "--models", "m1",
    ])
    assert "external_tier_status" not in json.loads(task_only.read_text(encoding="utf-8"))
