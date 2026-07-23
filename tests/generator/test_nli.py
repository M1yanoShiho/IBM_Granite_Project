import pytest

from evidence_rag.generator.nli import DEFAULT_NLI_MODEL_ID, DebertaNLIModel, normalize_nli_label


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
