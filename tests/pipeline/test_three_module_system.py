from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

import evidence_rag.composition as composition_module
from evidence_rag.composition import build_generator, build_pipeline_from_config
from evidence_rag.contracts.models import (
    EvidenceCandidate,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.infrastructure.config import ModuleConfig, load_experiment_config
from evidence_rag.infrastructure.corpus import CorpusBuilder
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_CONFIG = ROOT / "configs/experiments/systemf_three_module_smoke_seed13.toml"


class _Embedder:
    def embed_documents(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vector(text) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0)

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        if "Red Hat" in text:
            return (1.0, 0.0)
        if "poison" in text:
            return (0.8, 0.2)
        return (0.0, 1.0)


class _SelectorOutput:
    def __init__(self, protect: list[float], harm: list[float]) -> None:
        self.protect_scores = protect
        self.harm_scores = harm


class _SelectorModel:
    def __call__(
        self,
        *,
        question: list[str],
        candidate_text: list[str],
    ) -> _SelectorOutput:
        return _SelectorOutput(
            [0.01 if "poison" in text else 0.99 for text in candidate_text],
            [0.99 if "poison" in text else 0.01 for text in candidate_text],
        )


class _GrcClient:
    adapter_paths = {"grc": "fixture://grc"}

    def __init__(self) -> None:
        self.adapter_prompts: list[str] = []

    def generate_with_adapter(self, prompt: str, adapter_name: str) -> str:
        assert adapter_name == "grc"
        self.adapter_prompts.append(prompt)
        return "IBM acquired Red Hat in 2019 [1]."

    def generate(self, prompt: str) -> str:
        if prompt.startswith("Split the answer"):
            return (
                '{"claims":[{"source_text":"IBM acquired Red Hat in 2019 [1].",'
                '"text":"IBM acquired Red Hat in 2019."}]}'
            )
        if prompt.startswith("Check whether each rewritten claim"):
            return '{"results":[{"claim_id":"claim-1","faithful":true}]}'
        raise AssertionError(f"unexpected base-model prompt: {prompt[:60]}")


class _LoadedGrcClient(_GrcClient):
    def __init__(
        self,
        *,
        model_id: str,
        adapters: dict[str, str],
        config: object,
        device: str,
    ) -> None:
        super().__init__()
        self.adapter_paths = adapters
        self.model_id = model_id
        self.config = config
        self.device = device


class _EntailingNLI:
    def classify(self, *, premise: str, hypothesis: str) -> str:
        return "entailment"


@dataclass(frozen=True)
class _Consistency:
    consistent: bool = True
    mismatches: tuple[object, ...] = ()


class _EntityChecker:
    def check(self, claim: str, evidence: str) -> _Consistency:
        return _Consistency()


def _corpus(config):
    dataset = JsonlDatasetAdapter.load(config.dataset_manifest_path)
    return CorpusBuilder().build(dataset.documents, dataset.dataset_signature)


def _config():
    return load_experiment_config(SYSTEM_CONFIG)


def test_three_module_pipeline_removes_harm_before_grounded_generation() -> None:
    client = _GrcClient()
    config = _config()
    pipeline = build_pipeline_from_config(
        config,
        _corpus(config),
        embedder=_Embedder(),
        selector_model=_SelectorModel(),
        grc_client=client,
        nli=_EntailingNLI(),
        entity_checker=_EntityChecker(),
    )

    run = pipeline.run_with_trace(
        Query(query_id="q-system", text="What company did IBM acquire in 2019?"),
        top_k=10,
        max_selected=10,
    )

    assert len(run.candidates.candidates) == 10
    assert any("poison" in item.text for item in run.candidates.candidates)
    assert all("poison" not in item.text for item in run.selected.evidence)
    assert client.adapter_prompts and "poison" not in client.adapter_prompts[0]
    assert run.generation.answer == "IBM acquired Red Hat in 2019."
    assert run.generation.cited_evidence_ids


def test_grounded_generator_loads_frozen_adapter_and_true_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_snapshot = tmp_path / "granite"
    adapter_path = tmp_path / "grc-adapter"
    true_snapshot = tmp_path / "true"
    model_snapshot.mkdir()
    adapter_path.mkdir()
    true_snapshot.mkdir()
    model_config = model_snapshot / "config.json"
    adapter_weights = adapter_path / "adapter_model.safetensors"
    adapter_config = adapter_path / "adapter_config.json"
    true_config = true_snapshot / "config.json"
    model_config.write_bytes(b"granite-config")
    adapter_weights.write_bytes(b"grc-weights")
    adapter_config.write_bytes(b"grc-config")
    true_config.write_bytes(b"true-config")
    monkeypatch.setenv("TEST_GRANITE_MODEL", str(model_snapshot))
    monkeypatch.setenv("TEST_GRC_ADAPTER", str(adapter_path))
    monkeypatch.setenv("TEST_TRUE_MODEL", str(true_snapshot))

    monkeypatch.setattr(composition_module, "PeftGraniteLLMClient", _LoadedGrcClient)
    monkeypatch.setattr(
        composition_module,
        "TrueNLIModel",
        lambda *, model_id: _EntailingNLI(),
    )

    generator = build_generator(
        ModuleConfig(
            name="grounded-grc",
            parameters={
                "model_snapshot": "${TEST_GRANITE_MODEL}",
                "model_config_sha256": hashlib.sha256(model_config.read_bytes()).hexdigest(),
                "adapter_path": "${TEST_GRC_ADAPTER}",
                "adapter_name": "grc",
                "adapter_weights_sha256": hashlib.sha256(
                    adapter_weights.read_bytes()
                ).hexdigest(),
                "adapter_config_sha256": hashlib.sha256(
                    adapter_config.read_bytes()
                ).hexdigest(),
                "true_snapshot": "${TEST_TRUE_MODEL}",
                "true_config_sha256": hashlib.sha256(true_config.read_bytes()).hexdigest(),
                "device": "cpu",
                "max_new_tokens": 256,
                "temperature": 0.0,
                "top_p": 1.0,
                "entity_gate": "observe",
            },
        ),
        entity_checker=_EntityChecker(),
    )
    selected = SelectedEvidenceSet(
        query_id="q-generator",
        evidence=(
            EvidenceCandidate(
                evidence_id="ev-clean",
                document_id="doc-clean",
                chunk_id="chunk-clean",
                text="IBM acquired Red Hat in 2019.",
                source_uri="fixture://clean",
                retrieval_score=1.0,
                retrieval_rank=1,
            ),
        ),
    )

    result = generator.generate(
        Query(query_id="q-generator", text="What company did IBM acquire in 2019?"),
        QueryChecklist(query_id="q-generator", focus="company", required_facts=()),
        selected,
    )

    assert result.answer == "IBM acquired Red Hat in 2019."
    assert result.cited_evidence_ids == ("ev-clean",)
