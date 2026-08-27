#!/usr/bin/env python3
"""Real-model Goal 2 smoke for the four Experiment 04 baseline systems.

This entry accepts only the repository's revealed synthetic fixture.  It loads
every baseline model at the revision frozen in the Goal 2 manifest, verifies the
config/weight hashes, and executes one query through the unified arm runner.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from evidence_rag.contracts.models import GenerationResult, Query, SelectedEvidenceSet
from evidence_rag.contracts.protocols import Retriever, Selector
from evidence_rag.evaluation.experiment04_runner import (
    ArmComponents,
    Experiment04ArmRunner,
    canonical_output_bytes,
    load_arm_matrix,
)
from evidence_rag.evaluation.system_scorer import SCORER_SCHEMA_VERSION
from evidence_rag.generator.draft import DRAFT_PROMPT
from evidence_rag.generator.granite import (
    GraniteGenerationConfig,
    GraniteLLMClient,
    InlineCitationGraniteGenerator,
)
from evidence_rag.generator.nli import MiniCheckNLIModel
from evidence_rag.infrastructure.corpus import CorpusBuilder
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.retriever.granite import GraniteDenseRetriever, GraniteEmbedder
from evidence_rag.retriever.hybrid import HybridRetriever
from evidence_rag.retriever.rerank import GraniteCrossEncoderReranker, RerankingRetriever
from evidence_rag.retriever.strong_bm25 import StrongBM25Retriever
from evidence_rag.selector.provence import ProvencePassagePruner, ProvenceSelector
from evidence_rag.selector.top_k import TopKSelector

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = (
    ROOT
    / "docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21"
)
MANIFEST_PATH = EXPERIMENT / "artifacts/goal2_model_config_manifest.json"
CONFIG_DIR = ROOT / "configs/experiments/experiment04"
FIXTURE = ROOT / "tests/fixtures/three_module_smoke_dataset/manifest.json"
BASELINE_ARMS = ("dense_rag", "hybrid_rag", "granite_rerank_rag", "provence_rag")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _download_and_verify(cache_dir: Path, identity: dict[str, Any]) -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id=str(identity["model_id"]),
            revision=str(identity["revision"]),
            cache_dir=cache_dir,
        )
    )
    if _sha256(snapshot / "config.json") != identity["config_sha256"]:
        raise ValueError(f"config hash mismatch for {identity['model_id']}")
    for weight in identity.get("weights", []):
        if _sha256(snapshot / weight["file"]) != weight["sha256"]:
            raise ValueError(f"weight hash mismatch for {identity['model_id']}")
    return snapshot


def _sidecar(bundle: Any) -> dict[str, Any]:
    gold = bundle.gold_cases[0]
    documents = {document.document_id: document for document in bundle.documents}
    relevant = gold.relevant_document_ids or ()
    aliases = gold.reference_answers or ()
    units = [
        {
            "unit_id": f"{document_id}:u000",
            "source_id": document_id,
            "text": documents[document_id].text,
            "text_sha256": hashlib.sha256(documents[document_id].text.encode()).hexdigest(),
        }
        for document_id in relevant
    ]
    return {
        "schema_version": SCORER_SCHEMA_VERSION,
        "dataset": "hotpotqa",
        "query_id": gold.query_id,
        "gold_answer_aliases": list(aliases),
        "support_units": units,
        "component_id": "revealed-development-fixture",
    }


def _judge(
    model: MiniCheckNLIModel,
) -> Callable[[Query, SelectedEvidenceSet, GenerationResult], tuple[float, float]]:
    def score(
        query: Query,
        selected: SelectedEvidenceSet,
        result: GenerationResult,
    ) -> tuple[float, float]:
        del query
        cited = [
            item.text
            for item in selected.evidence
            if item.evidence_id in result.cited_evidence_ids
        ]
        if not cited or not result.answer.strip():
            return 0.0, 0.0
        supported = sum(
            model.classify(premise=text, hypothesis=result.answer) == "entailment"
            for text in cited
        )
        precision = supported / len(cited)
        recall = float(supported > 0)
        return precision, recall

    return score


def _cuda_cleanup() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - server dependency
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    models = manifest["models"]
    snapshots = {
        key: _download_and_verify(cache_dir, models[key])
        for key in (
            "dense_embedder",
            "granite_reranker",
            "provence",
            "direct_and_grounded_base",
            "minicheck_scorer_only",
        )
    }
    bundle = JsonlDatasetAdapter.load(FIXTURE)
    if len(bundle.queries) != 1:
        raise ValueError("server smoke accepts exactly one revealed development query")
    corpus = CorpusBuilder().build(bundle.documents, bundle.dataset_signature)
    configs = load_arm_matrix(CONFIG_DIR)
    query = bundle.queries[0]
    sidecar = _sidecar(bundle)

    embedder = GraniteEmbedder(
        model_id=str(snapshots["dense_embedder"]),
        device="cpu",
    )
    dense = GraniteDenseRetriever.from_corpus(corpus, embedder=embedder)
    hybrid = HybridRetriever(
        (
            StrongBM25Retriever.from_corpus(corpus, k1=0.9, b=0.4),
            dense,
        ),
        k=60,
        pool_size=10,
    )
    direct_llm = GraniteLLMClient(
        model_id=str(snapshots["direct_and_grounded_base"]),
        config=GraniteGenerationConfig(
            max_input_tokens=2304,
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
        ),
        device="cuda",
        dtype="float16",
    )
    generator = InlineCitationGraniteGenerator(
        llm=direct_llm,
        prompt_template=DRAFT_PROMPT,
    )
    minicheck = MiniCheckNLIModel(model_id=str(snapshots["minicheck_scorer_only"]))
    minicheck._ensure_loaded()
    citation_judge = _judge(minicheck)

    reranker = GraniteCrossEncoderReranker(
        model_id=str(snapshots["granite_reranker"]),
        revision=str(models["granite_reranker"]["revision"]),
        device="cpu",
        local_files_only=True,
    )
    granite_rerank = RerankingRetriever(hybrid, reranker, pool_size=40)
    provence = ProvenceSelector(
        ProvencePassagePruner(
            model_id=str(snapshots["provence"]),
            revision=str(models["provence"]["revision"]),
            threshold=0.1,
            always_select_title=True,
            reorder=False,
            local_files_only=True,
        )
    )
    retrievers: dict[str, Retriever] = {
        "dense_rag": dense,
        "hybrid_rag": hybrid,
        "granite_rerank_rag": granite_rerank,
        "provence_rag": hybrid,
    }
    selectors: dict[str, Selector] = {
        "dense_rag": TopKSelector(),
        "hybrid_rag": TopKSelector(),
        "granite_rerank_rag": TopKSelector(),
        "provence_rag": provence,
    }
    records = []
    for arm_id in BASELINE_ARMS:
        runner = Experiment04ArmRunner(
            configs[arm_id],
            ArmComponents(
                retriever=retrievers[arm_id],
                selector=selectors[arm_id],
                generator=generator,
            ),
            citation_judge=citation_judge,
        )
        output = runner.run(query, scorer_sidecar=sidecar)
        _write_json(output_dir / "full" / f"{arm_id}.json", output)
        records.append(
            {
                "arm_id": arm_id,
                "completed_queries": 1,
                "retrieval_trace_present": bool(output["retrieval"]["candidate_count"]),
                "selected_context_present": bool(output["selection"]["selected_count"]),
                "answer_present": bool(output["generation"]["answer"]),
                "score_present": set(output["score"]["metrics"])
                == {"ret", "sel", "ans", "cit", "rar"},
                "canonical_output_sha256": hashlib.sha256(
                    canonical_output_bytes(output)
                ).hexdigest(),
            }
        )
    del reranker, granite_rerank, provence, direct_llm, generator, minicheck
    _cuda_cleanup()
    try:
        import torch
        import transformers

        runtime = {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:  # pragma: no cover - server dependency
        runtime = {"host": platform.node(), "python": platform.python_version()}
    passed = all(
        row["completed_queries"] == 1
        and row["retrieval_trace_present"]
        and row["selected_context_present"]
        and row["answer_present"]
        and row["score_present"]
        for row in records
    )
    summary = {
        "schema_version": "experiment04.goal2_real_baseline_smoke.v1",
        "fixture_kind": "revealed_synthetic_development",
        "model_hashes_verified": True,
        "minicheck_real_model": True,
        "arm_count": len(records),
        "arms": records,
        "runtime": runtime,
        "status": "PASS" if passed and len(records) == 4 else "FAIL",
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.cache_dir.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "status": summary["status"],
                "arm_count": summary["arm_count"],
                "model_hashes_verified": summary["model_hashes_verified"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
