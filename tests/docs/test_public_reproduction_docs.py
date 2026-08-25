from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_public_reproduction_reading_paths_exist() -> None:
    expected = (
        "REPRODUCIBILITY_MAP.md",
        "experiments/README.md",
        "experiments/selector/README.md",
        "experiments/generator/README.md",
        "experiments/experiment04/README.md",
        "experiments/experiment05/README.md",
        "experiments/generator/frozen_provenance.json",
    )

    assert all((ROOT / path).is_file() for path in expected)


def test_reproducibility_map_preserves_negative_results_and_commands() -> None:
    text = (ROOT / "REPRODUCIBILITY_MAP.md").read_text(encoding="utf-8")

    for required in (
        "NOT_SUPPORTED",
        "KEEP_TOPK10",
        "results/selector/misleading_evidence_summary.json",
        "results/selector/blind_answer_gate.json",
        "results/experiment04/final_results.json",
        "results/experiment05/claim_labels.json",
        "experiments/experiment04/build_tables.py",
        "experiments/experiment05/build_tables.py",
        "research-archive-2026-08-25",
    ):
        assert required in text


def test_public_reproduction_docs_have_no_personal_hpc_paths() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "experiments").rglob("*")
        if path.is_file()
    )

    assert "/scratch" + "/" not in text
    assert "/user" + "/work/" not in text
