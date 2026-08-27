from __future__ import annotations

import hashlib
from pathlib import Path

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    Query,
    SelectedEvidenceSet,
)
from evidence_rag.evaluation.experiment04_runner import (
    ARM_MATRIX,
    ArmComponents,
    Experiment04ArmRunner,
    load_arm_matrix,
)
from evidence_rag.evaluation.system_scorer import SCORER_SCHEMA_VERSION
from evidence_rag.selector.top_k import TopKSelector

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/experiments/experiment04"


class _Retriever:
    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        assert top_k == 10
        return CandidateSet(
            query_id=query.query_id,
            candidates=(
                EvidenceCandidate(
                    evidence_id="clean::chunk-00000",
                    document_id="clean",
                    chunk_id="chunk-00000",
                    text="IBM acquired Red Hat in 2019.",
                    source_uri="fixture://clean",
                    retrieval_score=1.0,
                    retrieval_rank=1,
                ),
            ),
        )


class _Generator:
    def generate(self, query: Query, checklist: object, selected: SelectedEvidenceSet) -> GenerationResult:
        del checklist
        assert selected.evidence
        return GenerationResult(
            query_id=query.query_id,
            answer="IBM acquired Red Hat in 2019.",
            cited_evidence_ids=(selected.evidence[0].evidence_id,),
        )


def _sidecar() -> dict[str, object]:
    text = "IBM acquired Red Hat in 2019."
    return {
        "schema_version": SCORER_SCHEMA_VERSION,
        "dataset": "hotpotqa",
        "query_id": "q",
        "gold_answer_aliases": [text],
        "support_units": [
            {
                "unit_id": "clean:u000",
                "source_id": "clean",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        ],
        "component_id": "fixture-component",
    }


def test_frozen_matrix_contains_exactly_ten_registered_identities() -> None:
    configs = load_arm_matrix(CONFIG_DIR)

    assert set(configs) == set(ARM_MATRIX)
    assert len(configs) == 10
    assert configs["granite_rerank_rag"].modules.retriever == "granite-rerank"
    assert configs["provence_rag"].modules.selector == "provence"
    assert [configs[f"ours_seed{seed}"].modules.generator_seed for seed in (13, 42, 73)] == [
        13,
        42,
        73,
    ]


def test_unified_runner_outputs_all_five_metrics_without_gold_fields() -> None:
    config = load_arm_matrix(CONFIG_DIR)["dense_rag"]
    runner = Experiment04ArmRunner(
        config,
        ArmComponents(
            retriever=_Retriever(),
            selector=TopKSelector(),
            generator=_Generator(),
        ),
        citation_judge=lambda query, selected, result: (1.0, 1.0),
    )

    output = runner.run(Query(query_id="q", text="What did IBM acquire?"), scorer_sidecar=_sidecar())

    assert output["score"]["metrics"] == {
        "ret": 1.0,
        "sel": 1.0,
        "ans": 1.0,
        "cit": 1.0,
        "rar": 1.0,
    }
    serialized = str(output).casefold()
    assert "gold_answer" not in serialized
    assert "support_units" not in serialized
    assert "component_id" not in serialized
