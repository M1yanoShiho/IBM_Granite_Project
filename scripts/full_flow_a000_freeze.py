"""Materialise the A000 data/model freeze and paired-power audit.

This command is deliberately read-only with respect to every source experiment.
It consolidates their frozen manifests into the next full-flow route and makes
the final held-out power limitation explicit before any new model is trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

SCHEMA_VERSION = "full-flow-a000-freeze-v1"
ALPHA = 0.05
TARGET_POWER = 0.80
TARGET_EFFECT = 0.02
SERVER_AUDIT_SCHEMA_VERSION = "full-flow-a000-server-audit-v1"
SERVER_AUDIT_CHECKS = {
    "repo_sync",
    "candidate_pools",
    "model_snapshots",
    "selector_checkpoint",
    "system_heldout_boundary",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _server_verification(repo: Path, server_audit_path: Path | None) -> dict[str, Any]:
    if server_audit_path is None:
        return {
            "status": "PENDING",
            "required": sorted(SERVER_AUDIT_CHECKS),
        }

    audit = _read_json(server_audit_path)
    if audit.get("schema_version") != SERVER_AUDIT_SCHEMA_VERSION:
        raise ValueError("A000 server audit schema is not frozen v1")
    if audit.get("status") != "PASS":
        raise ValueError("A000 server audit has not passed")
    checks = audit.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("A000 server audit checks are missing")
    missing = SERVER_AUDIT_CHECKS.difference(checks)
    if missing:
        raise ValueError(f"A000 server audit checks are missing: {sorted(missing)}")
    failed = sorted(
        name
        for name in SERVER_AUDIT_CHECKS
        if not isinstance(checks[name], dict) or checks[name].get("status") != "PASS"
    )
    if failed:
        raise ValueError(f"A000 server audit checks have not passed: {failed}")
    return {
        "status": "PASS",
        "audit_schema_version": audit["schema_version"],
        "audit_path": str(server_audit_path.relative_to(repo)),
        "audit_sha256": _sha256(server_audit_path),
        "audited_at": audit["audited_at"],
        "server": audit["server"],
        "heldout_exposure_boundary": audit["checks"]["system_heldout_boundary"][
            "historical_exposure"
        ],
    }


def required_paired_n(
    *,
    discordance_rate: float,
    effect: float,
    alpha: float = ALPHA,
    power: float = TARGET_POWER,
    design_effect: float = 1.0,
) -> int:
    """Conservative normal approximation for a paired binary comparison.

    ``discordance_rate`` is P(0,1)+P(1,0), while ``effect`` is the desired
    absolute paired accuracy difference.  Exact McNemar and component-cluster
    bootstrap inference remain mandatory for the final report; this calculation
    is only the pre-run sample-size sensitivity analysis.
    """

    if not 0.0 < discordance_rate <= 1.0:
        raise ValueError("discordance_rate must be in (0, 1]")
    if not 0.0 < effect < 1.0:
        raise ValueError("effect must be in (0, 1)")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be in (0, 1)")
    if design_effect < 1.0:
        raise ValueError("design_effect must be at least 1")
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    estimate = design_effect * (z_alpha + z_power) ** 2 * discordance_rate / effect**2
    return math.ceil(estimate)


def minimum_detectable_effect(
    *,
    n: int,
    discordance_rate: float,
    alpha: float = ALPHA,
    power: float = TARGET_POWER,
    design_effect: float = 1.0,
) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    return math.sqrt(
        design_effect * (z_alpha + z_power) ** 2 * discordance_rate / n
    )


def _transition_summary(report: dict[str, Any], key: str) -> dict[str, Any]:
    comparison = report["comparisons"][key]
    transitions = comparison["transitions"]
    n = int(comparison["answer_match"]["n_paired"])
    wrong_to_right = int(transitions["wrong_to_right"])
    right_to_wrong = int(transitions["right_to_wrong"])
    discordant = wrong_to_right + right_to_wrong
    return {
        "n": n,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "discordant": discordant,
        "discordance_rate": discordant / n,
    }


def build_outputs(
    repo: Path, server_audit_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_route = repo / "docs/full-flow/experiments/01_selector_generator_bridge_2026-08-12"
    sources = {
        "niah_train_pool": repo / "results/selector-beam-v1/m0/pools/niah-train.json",
        "niah_dev_pool": repo / "results/selector-beam-v1/m0/pools/niah-dev.json",
        "sealed600_pool": repo / "results/selector-beam-v1/m0/pools/sealed600.json",
        "f000_input": old_route / "artifacts/F000_INPUT_MANIFEST.json",
        "f005_selection": old_route / "artifacts/F005/selection/selection_manifest.json",
        "f005_run": old_route / "artifacts/F005/eval/run_manifest.json",
        "f005_report": old_route / "artifacts/F005/eval/report.json",
        "f006_data": old_route / "artifacts/F006/data/manifest.json",
        "f006_clean": old_route / "artifacts/F006/clean/training_manifest.json",
        "f006_mixed": old_route / "artifacts/F006/mixed/training_manifest.json",
        "system_heldout_sample": repo / "configs/heldout-sample.json",
        "system_heldout_handover": repo / "docs/heldout-three-set-handover.md",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"A000 source files missing: {missing}")

    niah_train = _read_json(sources["niah_train_pool"])
    niah_dev = _read_json(sources["niah_dev_pool"])
    sealed = _read_json(sources["sealed600_pool"])
    f000 = _read_json(sources["f000_input"])
    selection = _read_json(sources["f005_selection"])
    f005_run = _read_json(sources["f005_run"])
    f005_report = _read_json(sources["f005_report"])
    f006_data = _read_json(sources["f006_data"])
    f006_clean = _read_json(sources["f006_clean"])
    f006_mixed = _read_json(sources["f006_mixed"])
    heldout = _read_json(sources["system_heldout_sample"])

    if f005_run.get("gold_loaded_at_runtime") is not False:
        raise ValueError("F005 runtime gold boundary is not frozen false")
    if f006_data.get("sealed_or_heldout_read") is not False:
        raise ValueError("F006 data manifest does not preserve held-out isolation")
    if f006_clean.get("model_snapshot") != f006_mixed.get("model_snapshot"):
        raise ValueError("F006 clean/mixed Granite snapshots differ")
    if sealed["candidate_pool"]["query_count"] != 600:
        raise ValueError("retired sealed set is not the frozen 600-query set")

    heldout_compact: dict[str, Any] = {}
    for name, entry in heldout["datasets"].items():
        heldout_compact[name] = {
            key: entry[key]
            for key in ("population", "sampled_ids", "sampled_records", "sha256")
            if key in entry
        }

    data_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "DEVELOPMENT_READY / FINAL_HELDOUT_RESERVED",
        "runtime_gold_boundary": {
            "generation_reads_gold": False,
            "utility_labelling_reads_gold": "post-generation offline scoring only",
            "formal_runtime": "Retriever -> Selector -> Generator -> one answer",
        },
        "datasets": {
            "niah_train": {
                "role": ["generator_train", "utility_labels", "selector_train"],
                "query_count": niah_train["candidate_pool"]["query_count"],
                "candidate_pool_sha256": niah_train["candidate_pool"]["sha256"],
                "source_parent_sha256": niah_train["source_parent"]["sha256"],
                "final_claim_allowed": False,
            },
            "niah_decision_dev": {
                "role": ["diagnosis", "method_selection", "ablation"],
                "raw_pool_queries": niah_dev["candidate_pool"]["query_count"],
                "full_flow_queries": f000["counts"]["queries"],
                "legacy_selector_changed": f000["counts"]["selector_changed_queries"],
                "runtime_input_sha256": f000["runtime_input_sha256"],
                "final_claim_allowed": False,
            },
            "sealed600": {
                "role": "RETIRED_READ_ONLY_HISTORY",
                "query_count": sealed["candidate_pool"]["query_count"],
                "candidate_pool_sha256": sealed["candidate_pool"]["sha256"],
                "dataset_signature": sealed["dataset"]["signature"],
                "new_training_or_selection_allowed": False,
            },
            "system_heldout": {
                "role": "ONE_SHOT_FINAL_CONFIRMATION_AFTER_METHOD_FREEZE",
                "status": "RESERVED / SCHEMA_AND_PIPELINE_DRYRUN_ONLY / NEVER_SCORED",
                "historical_exposure": {
                    "full_datasets_loaded_for": "schema and record-count checks",
                    "generated_records_per_dataset_per_arm": 3,
                    "arms": ["baseline", "verify-annotate-nogate"],
                    "answer_text_persisted": False,
                    "metric_or_scorer_run": False,
                    "final_sample_scores_observed": False,
                },
                "sample_seed": heldout["seed"],
                "datasets": heldout_compact,
                "headline_rules": {
                    "hotpotqa": "report separately",
                    "musique-full": "answerable is headline; unanswerable reports abstention",
                    "rgb": "report separately",
                    "rgb-counterfactual": "secondary analysis only",
                    "pooling": False,
                },
            },
        },
        "models": {
            "retriever": {
                "architecture": "Hybrid RRF: StrongBM25 + Granite dense",
                "embedding_model": niah_train["embedding"]["model_id"],
                "embedding_revision": niah_train["embedding"]["revision"],
                "parameters_sha256": niah_train["retriever"]["parameters_sha256"],
            },
            "selector_legacy": {
                "model_id": selection["selector_model_id"],
                "revision": selection["selector_revision"],
                "checkpoint_sha256": selection["selector_checkpoint_sha256"],
                "safe_threshold": selection["safe_threshold"],
                "cap": selection["cap"],
            },
            "generator": {
                "model": f000["generation"]["model"],
                "snapshot_path_recorded_by_f006": f006_clean["model_snapshot"],
                "snapshot_config_sha256": f006_clean["model_snapshot_config_sha256"],
                "allowed_variants": ["base", "clean_draft_lora", "mixed_draft_lora"],
                "other_model_family_allowed": False,
            },
            "claim_verifier": {
                "model_id": "google/t5_xxl_true_nli_mixture",
                "role": "frozen production verifier, excluded from judging",
            },
            "citation_evaluator": {
                "model_id": "lytang/MiniCheck-Flan-T5-Large",
                "revision": "96eafd01cee2d16cf81aaa2fb226b14f422a37b3",
                "role": "independent post-generation evaluation only",
            },
        },
        "legacy_results": {
            "f005_queries": f005_report["queries"],
            "f005_selector_deletion_precision": f005_report["selector_evidence"][
                "deletion_precision"
            ],
            "f005_full_system_delta": f005_report["comparisons"][
                "C_minus_A_full_system"
            ]["answer_match"]["delta"],
            "f006_train_queries": f006_data["train_queries"],
            "f006_validation_queries": f006_data["validation_queries"],
            "f006_adapter_scope": f005_run["mixed_adapter_scope"],
        },
        "source_files": {
            name: {"path": str(path.relative_to(repo)), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "server_entity_verification": _server_verification(repo, server_audit_path),
    }

    design_effect = f005_report["queries"] / f005_report["components"]
    observed = {
        "generator_only": _transition_summary(f005_report, "B_minus_A_generator_only"),
        "selector_only": _transition_summary(f005_report, "C_minus_B_selector_only"),
        "full_system": _transition_summary(f005_report, "C_minus_A_full_system"),
    }
    generic_rates = (0.05, 0.10, 0.15)
    effects = (0.01, 0.02, 0.03, 0.05)
    sensitivity = {
        f"discordance_{rate:.2f}": {
            f"effect_{effect:.2f}": required_paired_n(
                discordance_rate=rate,
                effect=effect,
                design_effect=design_effect,
            )
            for effect in effects
        }
        for rate in generic_rates
    }
    final_primary_n = {
        "hotpotqa": int(heldout["datasets"]["hotpotqa"]["sampled_records"]),
        "musique_answerable": int(heldout["datasets"]["musique-full"]["sampled_ids"]),
        "rgb": int(heldout["datasets"]["rgb"]["sampled_records"]),
    }
    mde = {
        name: {
            f"discordance_{rate:.2f}": minimum_detectable_effect(
                n=n,
                discordance_rate=rate,
                design_effect=design_effect,
            )
            for rate in generic_rates
        }
        for name, n in final_primary_n.items()
    }
    power_manifest = {
        "schema_version": "full-flow-a000-power-v1",
        "status": "FINAL_SAMPLE_UNDERPOWERED_FOR_2PP_PER_DATASET",
        "method": {
            "calculation": "two-sided normal approximation for paired binary superiority",
            "final_inference": "exact McNemar + paired component-cluster bootstrap",
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "design_effect": design_effect,
            "target_effect": TARGET_EFFECT,
        },
        "f005_observed_transitions": observed,
        "required_n_sensitivity": sensitivity,
        "reserved_final_sample_n": final_primary_n,
        "minimum_detectable_effect": mde,
        "frozen_decision": {
            "development": "point estimate + mechanism/stability gates select candidates",
            "final_superiority": "paired 95% CI lower bound > 0",
            "coverage_noninferiority_margin": -0.01,
            "citation_noninferiority_margin": -0.02,
            "interpretation": (
                "The reserved 300-400 query per-dataset samples can confirm only medium "
                "paired effects (roughly 3-6pp depending on discordance), not a 2pp effect. "
                "A null CI is inconclusive for small effects and must not be called equivalence."
            ),
        },
    }
    return data_manifest, power_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--server-audit",
        type=Path,
        help="completed A000 server audit; omit to materialise a PENDING manifest",
    )
    args = parser.parse_args()
    data_manifest, power_manifest = build_outputs(
        args.repo.resolve(),
        args.server_audit.resolve() if args.server_audit else None,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "A000_DATA_MODEL_MANIFEST.json": data_manifest,
        "A000_POWER.json": power_manifest,
    }
    for filename, value in outputs.items():
        path = args.out_dir / filename
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
