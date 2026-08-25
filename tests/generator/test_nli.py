import pytest

from evidence_rag.generator.nli import (
    DEFAULT_BINARY_ENTAIL_THRESHOLD,
    DEFAULT_NLI_MODEL_ID,
    DebertaNLIModel,
    MiniCheckNLIModel,
    TrueNLIModel,
    _binary_entailment_label,
    build_nli_model,
    normalize_nli_label,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("entailment", "entailment"),
        ("ENTAILMENT", "entailment"),
        ("entail", "entailment"),
        ("contradiction", "contradiction"),
        ("CONTRADICT", "contradiction"),
        ("neutral", "neutral"),
        ("LABEL_0", "neutral"),
    ),
)
def test_model_label_vocabularies_map_onto_our_three_labels(raw: str, expected: str) -> None:
    assert normalize_nli_label(raw) == expected


def test_constructing_the_model_loads_no_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NLI_MODEL_ID", raising=False)

    model = DebertaNLIModel()

    assert model.model_id == DEFAULT_NLI_MODEL_ID
    assert model._model is None
    assert model._tokenizer is None


def test_model_id_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NLI_MODEL_ID", "cross-encoder/nli-deberta-v3-large")

    assert DebertaNLIModel().model_id == "cross-encoder/nli-deberta-v3-large"


def test_binary_backends_construct_without_loading_weights() -> None:
    true_model = TrueNLIModel()
    mini_model = MiniCheckNLIModel()

    assert true_model.model_id == "google/t5_xxl_true_nli_mixture"
    assert mini_model.model_id == "lytang/MiniCheck-Flan-T5-Large"
    assert true_model.threshold == DEFAULT_BINARY_ENTAIL_THRESHOLD
    assert mini_model.threshold == DEFAULT_BINARY_ENTAIL_THRESHOLD
    assert true_model._model is None and true_model._tokenizer is None
    assert mini_model._model is None and mini_model._tokenizer is None


def test_binary_backends_do_not_produce_contradiction() -> None:
    # the annotation the team agreed to: binary backends never emit a
    # contradiction, so ClaimVerification.contradicted is not computed under them
    assert TrueNLIModel().produces_contradiction is False
    assert MiniCheckNLIModel().produces_contradiction is False
    assert DebertaNLIModel().produces_contradiction is True


@pytest.mark.parametrize(
    ("p_entail", "threshold", "expected"),
    (
        (0.9, 0.5, "entailment"),
        (0.5, 0.5, "entailment"),  # threshold is inclusive
        (0.49, 0.5, "neutral"),
        (0.0, 0.5, "neutral"),
    ),
)
def test_binary_label_thresholding(p_entail: float, threshold: float, expected: str) -> None:
    assert _binary_entailment_label(p_entail, threshold) == expected
    # a binary backend can never route to contradiction
    assert _binary_entailment_label(p_entail, threshold) != "contradiction"


def test_build_nli_model_defaults_to_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NLI_BACKEND", raising=False)
    assert isinstance(build_nli_model(), TrueNLIModel)


@pytest.mark.parametrize(
    ("name", "cls"),
    (("true", TrueNLIModel), ("minicheck", MiniCheckNLIModel), ("deberta", DebertaNLIModel)),
)
def test_build_nli_model_selects_by_name(name: str, cls: type) -> None:
    assert isinstance(build_nli_model(name), cls)


def test_build_nli_model_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NLI_BACKEND", "minicheck")
    assert isinstance(build_nli_model(), MiniCheckNLIModel)
    # explicit argument overrides the environment
    assert isinstance(build_nli_model("true"), TrueNLIModel)


def test_build_nli_model_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown NLI backend"):
        build_nli_model("granite")
