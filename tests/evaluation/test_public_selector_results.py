from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/selector"
SOURCE_SHA256 = "b031b6f29051fed94a76220ac829ebe2acffe086e253dec9f3b24c090252495c"


def _load(name: str) -> dict[str, object]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_public_selector_results_preserve_positive_and_negative_gates() -> None:
    evidence = _load("misleading_evidence_summary.json")
    answer = _load("blind_answer_gate.json")

    assert evidence["source"]["sha256"] == SOURCE_SHA256
    assert answer["source"]["sha256"] == SOURCE_SHA256
    assert evidence["status"] == "PASS"
    assert answer["status"] == "FAIL"
    assert answer["overall_decision"] == "KEEP_TOPK10"


def test_public_selector_metrics_match_the_frozen_lean_v3_report() -> None:
    evidence = _load("misleading_evidence_summary.json")
    answer = _load("blind_answer_gate.json")

    assert evidence["metrics_by_seed"]["13"]["harmful_reduction"] == (
        0.12951807228915663
    )
    assert evidence["metrics_by_seed"]["42"]["harmful_reduction"] == (
        0.13102409638554216
    )
    assert evidence["metrics_by_seed"]["13"]["required_recall_loss_pp"] == 0.0
    assert evidence["metrics_by_seed"]["42"]["required_chain_loss_pp"] == 0.0
    assert answer["answer_match_delta"]["macro"] == -0.0013531799729364006
    assert answer["answer_match_delta"]["ci95"] == [
        -0.0056179775280898875,
        0.002770083102493075,
    ]
