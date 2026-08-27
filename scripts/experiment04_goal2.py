#!/usr/bin/env python3
"""Run the Experiment 04 Goal 2 ten-arm development-only wiring smoke.

The full per-arm traces are written under ``runs/``.  The committed manifest is
content-free: it exposes only system identity, structural booleans, counts, and
canonical hashes.  This script accepts only the revealed synthetic fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from evidence_rag.composition import build_generator
from evidence_rag.contracts.models import GenerationResult, Query
from evidence_rag.contracts.protocols import Retriever, Selector
from evidence_rag.evaluation.experiment04_runner import (
    ARM_MATRIX,
    ArmComponents,
    Experiment04ArmRunner,
    canonical_output_bytes,
    load_arm_matrix,
)
from evidence_rag.evaluation.system_scorer import SCORER_SCHEMA_VERSION
from evidence_rag.generator.draft import DRAFT_PROMPT
from evidence_rag.generator.granite import InlineCitationGraniteGenerator
from evidence_rag.infrastructure.config import ModuleConfig
from evidence_rag.infrastructure.corpus import CorpusBuilder
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.retriever.granite import GraniteDenseRetriever
from evidence_rag.retriever.hybrid import HybridRetriever
from evidence_rag.retriever.rerank import RerankingRetriever
from evidence_rag.retriever.strong_bm25 import StrongBM25Retriever
from evidence_rag.selector.nli_runtime import NliRiskControlledSelector
from evidence_rag.selector.provence import ProvenceSelector
from evidence_rag.selector.top_k import TopKSelector

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiments/experiment04"
FIXTURE = ROOT / "tests/fixtures/three_module_smoke_dataset/manifest.json"
DEFAULT_RUN_DIR = ROOT / "runs/experiment04/goal2-development-smoke"
DEFAULT_MANIFEST = (
    ROOT
    / "docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21"
    / "artifacts/goal2_smoke_manifest.json"
)
FORMAL_DIR = ROOT / "runs/experiment04/formal"


class FixtureEmbedder:
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


class FixtureReranker:
    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        del query
        return tuple(
            100.0 if "Red Hat" in text else 50.0 if "poison" in text else -index
            for index, text in enumerate(passages)
        )


class FixturePruner:
    def prune(self, *, question: str, title: str, text: str) -> str:
        del question, title
        return "" if "poison" in text else text


class SelectorOutput:
    def __init__(self, protect: list[float], harm: list[float]) -> None:
        self.protect_scores = protect
        self.harm_scores = harm


class FixtureSelectorModel:
    def __call__(self, *, question: list[str], candidate_text: list[str]) -> SelectorOutput:
        del question
        return SelectorOutput(
            [0.01 if "poison" in text else 0.99 for text in candidate_text],
            [0.99 if "poison" in text else 0.01 for text in candidate_text],
        )


class FixtureDirectLLM:
    def generate(self, prompt: str) -> str:
        if "Evidence:" not in prompt or "Question:" not in prompt:
            raise ValueError("shared direct prompt contract is missing")
        return "IBM acquired Red Hat in 2019 [1]."


class FixtureGrcClient:
    adapter_paths = {"grc": "fixture://grc"}

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def generate_with_adapter(self, prompt: str, adapter_name: str) -> str:
        if adapter_name != "grc" or "Evidence:" not in prompt:
            raise ValueError("grounded draft did not use the frozen adapter route")
        return "IBM acquired Red Hat in 2019 [1]."

    def generate(self, prompt: str) -> str:
        if prompt.startswith("Split the answer"):
            return (
                '{"claims":[{"source_text":"IBM acquired Red Hat in 2019 [1].",'
                '"text":"IBM acquired Red Hat in 2019."}]}'
            )
        if prompt.startswith("Check whether each rewritten claim"):
            return '{"results":[{"claim_id":"claim-1","faithful":true}]}'
        raise ValueError("unexpected base-model prompt in fixture GRC client")


class EntailingNLI:
    def classify(
        self,
        premise: str,
        hypothesis: str,
    ) -> Literal["entailment", "neutral", "contradiction"]:
        del premise, hypothesis
        return "entailment"


@dataclass(frozen=True)
class Consistency:
    consistent: bool = True
    mismatches: tuple[object, ...] = ()


class EntityChecker:
    def check(self, claim: str, evidence: str) -> Consistency:
        del claim, evidence
        return Consistency()


def _recursive_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).casefold())
            keys.update(_recursive_keys(child))
    elif isinstance(value, list | tuple):
        for child in value:
            keys.update(_recursive_keys(child))
    return keys


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sidecar(bundle: Any) -> dict[str, Any]:
    if len(bundle.queries) != 1 or len(bundle.gold_cases) != 1:
        raise ValueError("Goal 2 fixture must contain exactly one revealed development case")
    gold = bundle.gold_cases[0]
    documents = {document.document_id: document for document in bundle.documents}
    relevant = gold.relevant_document_ids or ()
    aliases = gold.reference_answers or ()
    if not relevant or not aliases:
        raise ValueError("revealed fixture lacks expected development labels")
    units = []
    for document_id in relevant:
        document = documents[document_id]
        units.append(
            {
                "unit_id": f"{document_id}:u000",
                "source_id": document_id,
                "text": document.text,
                "text_sha256": hashlib.sha256(document.text.encode()).hexdigest(),
            }
        )
    return {
        "schema_version": SCORER_SCHEMA_VERSION,
        "dataset": "hotpotqa",
        "query_id": gold.query_id,
        "gold_answer_aliases": list(aliases),
        "support_units": units,
        "component_id": "revealed-development-fixture",
    }


def _citation_judge(
    query: Query,
    selected: Any,
    result: GenerationResult,
) -> tuple[float, float]:
    del query
    cited = {
        item.evidence_id: item.text
        for item in selected.evidence
        if item.evidence_id in result.cited_evidence_ids
    }
    fully_supported = bool(cited) and "Red Hat" in " ".join(cited.values())
    return (1.0, 1.0) if fully_supported else (0.0, 0.0)


def _grounded_generator(seed: int) -> Any:
    return build_generator(
        ModuleConfig(
            name="grounded-grc",
            parameters={
                "adapter_name": "grc",
                "max_input_tokens": 2304,
                "max_new_tokens": 256,
                "temperature": 0.0,
                "top_p": 1.0,
                "entity_gate": "observe",
            },
        ),
        grc_client=FixtureGrcClient(seed),
        nli=EntailingNLI(),
        entity_checker=EntityChecker(),
    )


def _components(config: Any, corpus: Any) -> ArmComponents:
    dense = GraniteDenseRetriever.from_corpus(corpus, embedder=FixtureEmbedder())
    hybrid = HybridRetriever(
        (
            StrongBM25Retriever.from_corpus(corpus, k1=0.9, b=0.4),
            dense,
        ),
        k=60,
        pool_size=config.budget.hybrid_pool_size,
    )
    retrievers: dict[str, Retriever] = {
        "dense": dense,
        "hybrid": hybrid,
        "granite-rerank": RerankingRetriever(
            hybrid,
            FixtureReranker(),
            pool_size=config.budget.reranker_pool_size,
        ),
    }
    selectors: dict[str, Selector] = {
        "keep-all": TopKSelector(),
        "nli-risk-controlled": NliRiskControlledSelector(
            model=FixtureSelectorModel(),
            safe_threshold=0.9212157130241394,
            max_delete=2,
        ),
        "provence": ProvenceSelector(FixturePruner()),
    }
    if config.modules.generator == "direct":
        generator = InlineCitationGraniteGenerator(
            llm=FixtureDirectLLM(),
            prompt_template=DRAFT_PROMPT,
        )
    else:
        seed = config.modules.generator_seed
        if seed is None:
            raise ValueError("grounded arm has no frozen generator seed")
        generator = _grounded_generator(seed)
    return ArmComponents(
        retriever=retrievers[config.modules.retriever],
        selector=selectors[config.modules.selector],
        generator=generator,
    )


def run_smoke(run_dir: Path, manifest_path: Path) -> dict[str, Any]:
    configs = load_arm_matrix(CONFIG_DIR)
    bundle = JsonlDatasetAdapter.load(FIXTURE)
    corpus = CorpusBuilder().build(bundle.documents, bundle.dataset_signature)
    query = bundle.queries[0]
    sidecar = _sidecar(bundle)
    records: list[dict[str, Any]] = []
    for arm_id in ARM_MATRIX:
        config = configs[arm_id]
        runner = Experiment04ArmRunner(
            config,
            _components(config, corpus),
            citation_judge=_citation_judge,
        )
        output = runner.run(query, scorer_sidecar=sidecar)
        output_path = run_dir / f"{arm_id}.json"
        _write_json(output_path, output)
        metrics = output["score"]["metrics"]
        forbidden = _recursive_keys(output) & {
            "gold",
            "gold_answer",
            "gold_answers",
            "gold_answer_aliases",
            "support_units",
            "component_id",
        }
        records.append(
            {
                "arm_id": arm_id,
                "modules": config.modules.model_dump(mode="json"),
                "completed_queries": 1,
                "retrieval_trace_present": bool(output["retrieval"]["candidate_count"]),
                "selected_context_present": bool(output["selection"]["selected_count"]),
                "answer_present": bool(output["generation"]["answer"]),
                "score_present": set(metrics) == {"ret", "sel", "ans", "cit", "rar"},
                "scorer_readable": output["score"]["n_queries"] == 1,
                "gold_leakage_keys": sorted(forbidden),
                "canonical_output_sha256": hashlib.sha256(
                    canonical_output_bytes(output)
                ).hexdigest(),
            }
        )
    all_pass = all(
        row["completed_queries"] == 1
        and row["retrieval_trace_present"]
        and row["selected_context_present"]
        and row["answer_present"]
        and row["score_present"]
        and row["scorer_readable"]
        and not row["gold_leakage_keys"]
        for row in records
    )
    formal_empty = not FORMAL_DIR.exists() or not any(FORMAL_DIR.rglob("*"))
    manifest = {
        "schema_version": "experiment04.goal2_smoke.v1",
        "fixture_kind": "revealed_synthetic_development",
        "fixture_query_count": 1,
        "arm_count": len(records),
        "expected_arm_count": 10,
        "all_arms_completed": all_pass and len(records) == 10,
        "formal_output_directory_empty": formal_empty,
        "runtime_output_directory": str(run_dir.relative_to(ROOT)),
        "arms": records,
        "status": "PASS" if all_pass and len(records) == 10 and formal_empty else "FAIL",
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = run_smoke(args.run_dir.resolve(), args.manifest.resolve())
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "arm_count": manifest["arm_count"],
                "formal_output_directory_empty": manifest[
                    "formal_output_directory_empty"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
