from __future__ import annotations

import json
from pathlib import Path

from evidence_rag.cli.selector_error_analysis import main as error_analysis_main
from evidence_rag.cli.selector_final_report import main as report_main
from evidence_rag.cli.selector_stability import main as stability_main
from evidence_rag.contracts.models import EvidenceCandidate, SelectedEvidenceSet


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def selected(query_id: str) -> SelectedEvidenceSet:
    return SelectedEvidenceSet(
        query_id=query_id,
        evidence=(
            EvidenceCandidate(
                evidence_id=f"e-{query_id}",
                document_id=f"d-{query_id}",
                chunk_id="0",
                text="fixture",
                source_uri="fixture://source",
                retrieval_score=1.0,
                retrieval_rank=1,
            ),
        ),
    )


def write_arm(root: Path, method: str, query_ids: tuple[str, ...]) -> None:
    directory = root / method
    write_json(
        directory / "run_manifest.json",
        {"status": "complete", "candidate_pool_sha256": "a" * 64},
    )
    (directory / "selected_evidence_sets.jsonl").write_text(
        "".join(selected(query_id).model_dump_json() + "\n" for query_id in query_ids),
        encoding="utf-8",
    )


def test_stability_report_requires_exact_ids_and_pool(tmp_path: Path) -> None:
    query_ids = tuple(f"q{index:02d}" for index in range(60))
    frozen_ids = tmp_path / "query_ids.txt"
    frozen_ids.write_text("\n".join(query_ids) + "\n", encoding="utf-8")
    for root in (tmp_path / "formal", tmp_path / "repeat"):
        for method in ("top-k", "reliability-mis"):
            write_arm(root, method, query_ids)

    output = tmp_path / "stability.json"
    assert (
        stability_main(
            (
                "--formal-arms",
                str(tmp_path / "formal"),
                "--repeat-arms",
                str(tmp_path / "repeat"),
                "--query-ids",
                str(frozen_ids),
                "--output",
                str(output),
            )
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["arms"]["reliability-mis"]["exact_agreement_count"] == 60


def test_final_tables_put_metrics_in_rows_and_methods_in_columns(tmp_path: Path) -> None:
    sealed_pool = {
        "candidate_pool": {
            "query_count": 600,
            "top_n": 20,
            "exact_top_n_rate": 1.0,
            "sha256": "a" * 64,
        },
        "audit": {
            "required_recall_at_top_n": 0.9,
            "harmful_pool_hit_rate": 0.5,
            "harmful_pool_hit_count": 300,
            "unresolved_parent_count": 0,
        },
    }
    twowiki_pool = {
        "candidate_pool": {
            "query_count": 2000,
            "top_n": 20,
            "exact_top_n_rate": 1.0,
            "sha256": "b" * 64,
        },
        "audit": {
            "required_recall_at_top_n": 0.8,
            "harmful_pool_hit_rate": None,
            "harmful_pool_hit_count": 0,
            "unresolved_parent_count": 0,
        },
    }
    arm = {
        "harmful_in_context_conditional": 0.2,
        "required_evidence_recall": 0.9,
        "evidence_precision": 0.3,
        "selected_evidence_count": 10.0,
    }
    paired_harm = {
        "delta_mis_minus_top_k": -0.1,
        "ci_low": -0.15,
        "ci_high": -0.05,
        "p_value": 0.01,
    }
    paired_recall = {
        "delta_mis_minus_top_k": 0.0,
        "ci_low": -0.005,
        "ci_high": 0.005,
        "p_value": 1.0,
    }
    sealed_summary = {
        "top_k": arm,
        "reliability_mis": {**arm, "harmful_in_context_conditional": 0.1},
        "paired_harm": paired_harm,
        "paired_required_recall": paired_recall,
        "decision": {
            "harm_reduction_pass": True,
            "required_recall_noninferiority_pass": True,
        },
    }
    twowiki_summary = {
        "top_k": arm,
        "reliability_mis": arm,
        "paired_harm": None,
        "paired_required_recall": paired_recall,
        "decision": {
            "harm_reduction_pass": None,
            "required_recall_noninferiority_pass": True,
        },
    }
    stability = {
        "pass": True,
        "arms": {"reliability-mis": {"exact_agreement_count": 60, "exact_agreement_rate": 1.0}},
    }
    inputs = {
        "sealed_pool.json": sealed_pool,
        "sealed_summary.json": sealed_summary,
        "twowiki_pool.json": twowiki_pool,
        "twowiki_summary.json": twowiki_summary,
        "stability.json": stability,
    }
    for name, value in inputs.items():
        write_json(tmp_path / name, value)
    output = tmp_path / "results"
    assert (
        report_main(
            (
                "--sealed-pool",
                str(tmp_path / "sealed_pool.json"),
                "--sealed-summary",
                str(tmp_path / "sealed_summary.json"),
                "--twowiki-pool",
                str(tmp_path / "twowiki_pool.json"),
                "--twowiki-summary",
                str(tmp_path / "twowiki_summary.json"),
                "--stability",
                str(tmp_path / "stability.json"),
                "--output",
                str(output),
            )
        )
        == 0
    )
    lines = (output / "primary_selector_table.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "Metric,TopK,Reliability-MIS"
    assert lines[1].startswith("Harmful-in-context (%)")
    assert json.loads((output / "primary_summary.json").read_text())["final_selector"] == (
        "reliability-mis"
    )


def test_error_analysis_uses_frozen_outcomes_without_retuning(tmp_path: Path) -> None:
    def arm(
        recall: float,
        *,
        harmful_selected: bool,
        selected_id: str,
    ) -> dict[str, object]:
        return {
            "metrics": {
                "selected_ids": [selected_id],
                "required_document_ids": ["required"],
                "harmful_document_id": "harmful",
                "harmful_pool_hit": True,
                "harmful_selected": harmful_selected,
                "required_evidence_recall": recall,
            }
        }

    comparison = tmp_path / "comparison.jsonl"
    comparison.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "query_id": "q-success",
                    "top_k": arm(1.0, harmful_selected=True, selected_id="harmful#0"),
                    "reliability_mis": arm(1.0, harmful_selected=False, selected_id="required#0"),
                },
                {
                    "query_id": "q-failure",
                    "top_k": arm(1.0, harmful_selected=True, selected_id="required#0"),
                    "reliability_mis": arm(0.0, harmful_selected=True, selected_id="harmful#0"),
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "query_id": "q-success",
                    "diagnostic": {"mode": "mis", "unresolved_parent_count": 0},
                },
                {
                    "query_id": "q-failure",
                    "diagnostic": {
                        "mode": "topk_backend_fallback",
                        "unresolved_parent_count": 0,
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "error-analysis"
    assert (
        error_analysis_main(
            (
                "--comparison",
                str(comparison),
                "--events",
                str(events),
                "--output",
                str(output),
                "--limit",
                "20",
            )
        )
        == 0
    )
    report = json.loads((output / "error_analysis.json").read_text(encoding="utf-8"))
    assert report["category_counts"] == {
        "backend_or_parent_failure": 1,
        "harmful_not_removed": 1,
        "harmful_removed_required_retained": 1,
        "required_dropped": 1,
    }
