"""Offline smoke for the final Hybrid → trained-NLI-policy → grounded-GR-C wiring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from evidence_rag.composition import build_pipeline_from_config
from evidence_rag.contracts.models import PipelineRun, Query
from evidence_rag.infrastructure.config import ExperimentConfig, load_experiment_config
from evidence_rag.infrastructure.corpus import CorpusBuilder
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.pipeline.service import EvidenceRAGPipeline

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CPU_SMOKE_CONFIG = _ROOT / "configs/runtime/cpu_smoke.toml"


class _OfflineEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
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


class _OfflineSelectorModel:
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


class _OfflineGrcClient:
    adapter_paths = {"grc": "fixture://offline-grc"}

    def generate_with_adapter(self, prompt: str, adapter_name: str) -> str:
        if adapter_name != "grc":
            raise ValueError(f"unexpected smoke adapter: {adapter_name}")
        if "poison" in prompt:
            raise ValueError("Selector boundary failed: dropped evidence reached Generator")
        return "IBM acquired Red Hat in 2019 [1]."

    def generate(self, prompt: str) -> str:
        if prompt.startswith("Split the answer"):
            return (
                '{"claims":[{"source_text":"IBM acquired Red Hat in 2019 [1].",'
                '"text":"IBM acquired Red Hat in 2019."}]}'
            )
        if prompt.startswith("Check whether each rewritten claim"):
            return '{"results":[{"claim_id":"claim-1","faithful":true}]}'
        raise ValueError(f"unexpected offline smoke prompt: {prompt[:60]}")


class _EntailingNli:
    def classify(
        self,
        premise: str,
        hypothesis: str,
    ) -> Literal["entailment", "neutral", "contradiction"]:
        return "entailment"


@dataclass(frozen=True)
class _Consistency:
    consistent: bool = True
    mismatches: tuple[object, ...] = ()


class _EntityChecker:
    def check(self, claim: str, evidence: str) -> _Consistency:
        return _Consistency()


def build_cpu_smoke_pipeline(
    config_path: Path = DEFAULT_CPU_SMOKE_CONFIG,
) -> tuple[EvidenceRAGPipeline, ExperimentConfig]:
    """Build final module classes with deterministic doubles and no external I/O."""

    config = load_experiment_config(config_path)
    dataset = JsonlDatasetAdapter.load(config.dataset_manifest_path)
    corpus = CorpusBuilder().build(dataset.documents, dataset.dataset_signature)
    pipeline = build_pipeline_from_config(
        config,
        corpus,
        embedder=_OfflineEmbedder(),
        selector_model=_OfflineSelectorModel(),
        grc_client=_OfflineGrcClient(),
        nli=_EntailingNli(),
        entity_checker=_EntityChecker(),
    )
    return pipeline, config


def run_cpu_smoke(
    *,
    pipeline: EvidenceRAGPipeline | None = None,
    config: ExperimentConfig | None = None,
) -> PipelineRun:
    if pipeline is None or config is None:
        pipeline, config = build_cpu_smoke_pipeline()
    return pipeline.run_with_trace(
        Query(
            query_id="cpu-smoke-query",
            text="What company did IBM acquire in 2019?",
        ),
        top_k=config.top_k,
        max_selected=config.max_selected,
    )


def main() -> None:
    print(run_cpu_smoke().model_dump_json(indent=2))


if __name__ == "__main__":
    main()
