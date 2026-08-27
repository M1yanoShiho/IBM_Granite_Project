from __future__ import annotations

from types import SimpleNamespace

import pytest

from evidence_rag.generator.granite import NamedAdapterTextGenerator, PeftGraniteLLMClient


class _FakeModel:
    def __init__(self) -> None:
        self.events: list[str] = []

    def disable_adapter(self) -> _FakeModel:
        self.events.append("disable_enter")
        return self

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        self.events.append("disable_exit")

    def set_adapter(self, name: str) -> None:
        self.events.append(f"set:{name}")


def _client() -> PeftGraniteLLMClient:
    client = object.__new__(PeftGraniteLLMClient)
    client.adapter_paths = {"clean": "/clean", "mixed": "/mixed"}
    client._model = _FakeModel()
    client._tokenizer = SimpleNamespace()
    return client


def test_base_generate_explicitly_disables_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    monkeypatch.setattr(client, "_generate_current_model", lambda prompt: f"base:{prompt}")

    assert client.generate("hello") == "base:hello"
    assert client._model.events == ["disable_enter", "disable_exit"]


def test_named_adapter_view_selects_only_requested_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    monkeypatch.setattr(client, "_generate_current_model", lambda prompt: f"adapted:{prompt}")
    view = NamedAdapterTextGenerator(client, "mixed")

    assert view.generate("hello") == "adapted:hello"
    assert client._model.events == ["set:mixed"]


def test_named_adapter_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown LoRA adapter"):
        NamedAdapterTextGenerator(_client(), "missing")
