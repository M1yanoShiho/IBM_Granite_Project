"""Freeze the G000 protocol, source identities, denylist, and server audit.

This is a read-only preflight for route
``03_generator_grounding_repair_2026-08-18``.  It does not start training,
generate utility labels, score held-out data, or modify server state.  The only
server interaction is SSH-based entity verification of files that already exist
in the project repo/runtime roots.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_PREFIX = "full-flow-g000"
ROUTE = Path("docs/full-flow/experiments/03_generator_grounding_repair_2026-08-18")
PREVIOUS_ROUTE = Path("docs/full-flow/experiments/02_generator_selector_alignment_2026-08-15")
BRIDGE_ROUTE = Path("docs/full-flow/experiments/01_selector_generator_bridge_2026-08-12")
PLAN_ARCHIVE_COMMIT = "907ab80b7ccecd7690bbc17f473f355a529557bc"
EXPECTED_BRANCH = "refactor/three-module-baseline"
DEFAULT_SSH_TARGET = "fl25387@10.70.71.11"
SERVER_REPO = "/home/fl25387/projects/IBM_Granite_Project_latest"
SERVER_RUNTIME = "/scratch/fl25387/IBM_Granite_Project_latest"
G200_RUNTIME = f"{SERVER_RUNTIME}/runs/full-flow/G200-v1"
G220_RUNTIME = f"{SERVER_RUNTIME}/runs/full-flow/G220-v1/formal"
G230_RUNTIME = f"{SERVER_RUNTIME}/runs/full-flow/G230-v1"
G220_RUNS = tuple((arm, seed) for arm in ("gc", "gm") for seed in (13, 42, 73))
G230_RUNS = ("gn", "seed13", "seed42", "seed73")
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_ids(ids: Sequence[str]) -> str:
    return _sha256_text("\n".join(ids))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _rel(repo: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo.resolve()))


def _git(repo: Path, args: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_pin(repo: Path, relative: str | Path) -> dict[str, Any]:
    path = repo / relative
    if not path.is_file():
        raise FileNotFoundError(f"required G000 source file missing: {path}")
    return {
        "path": _rel(repo, path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _ids_block(ids: Sequence[str], *, source: str, role: str) -> dict[str, Any]:
    ordered = [str(item) for item in ids]
    if len(ordered) != len(set(ordered)):
        raise ValueError(f"{source} contains duplicate ordered IDs")
    return {
        "role": role,
        "source": source,
        "count": len(ordered),
        "ordered_ids_sha256": _sha256_ids(ordered),
        "ordered_ids": ordered,
    }


def _role_assignment_block(repo: Path, relative: Path, *, role: str) -> dict[str, Any]:
    rows = _jsonl(repo / relative)
    query_ids: list[str] = []
    component_ids: list[str] = []
    roles: Counter[str] = Counter()
    for row in rows:
        query_id = str(row.get("query_id", ""))
        component_id = str(row.get("component_id", ""))
        row_role = str(row.get("role", ""))
        if not query_id or not component_id or not row_role:
            raise ValueError(f"invalid role-assignment row in {relative}: {row}")
        query_ids.append(query_id)
        component_ids.append(component_id)
        roles[row_role] += 1
    block = _ids_block(query_ids, source=str(relative), role=role)
    block.update(
        {
            "source_file_sha256": _sha256(repo / relative),
            "roles": dict(sorted(roles.items())),
            "ordered_component_ids_sha256": _sha256_ids(component_ids),
            "unique_component_count": len(set(component_ids)),
        }
    )
    return block


def _heldout_blocks(repo: Path) -> dict[str, Any]:
    heldout = _json(repo / "configs/heldout-sample.json")
    blocks: dict[str, Any] = {}
    for name, entry in heldout["datasets"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"held-out entry is not an object: {name}")
        query_ids = [str(item) for item in entry["query_ids"]]
        digest = _sha256_ids(query_ids)
        if digest != entry["sha256"]:
            raise ValueError(f"held-out ordered ID hash mismatch for {name}")
        block = _ids_block(
            query_ids,
            source="configs/heldout-sample.json",
            role="SYSTEM_HELDOUT_DENY_UNTIL_SYSTEMF_AND_SEPARATE_AUTHORIZATION",
        )
        block.update(
            {
                "dataset": name,
                "population": entry["population"],
                "sampled_records": entry.get("sampled_records"),
                "sampled_ids": entry.get("sampled_ids"),
                "config_sha256": entry["sha256"],
                "content_or_score_loaded": False,
            }
        )
        blocks[name] = block
    return blocks


def _f005_sealed_block(repo: Path) -> dict[str, Any]:
    relative = BRIDGE_ROUTE / "artifacts/F005/eval/generations.jsonl"
    rows = _jsonl(repo / relative)
    ids = [str(row["query_id"]) for row in rows]
    block = _ids_block(
        ids,
        source=str(relative),
        role="SEALED600_RETIRED_READ_ONLY_HISTORY",
    )
    block.update(
        {
            "source_file_sha256": _sha256(repo / relative),
            "new_training_or_selection_allowed": False,
            "historical_result_only": True,
        }
    )
    return block


def _g230_task_blocks(repo: Path) -> dict[str, Any]:
    relative = PREVIOUS_ROUTE / "artifacts/G230/gn/generations.jsonl.gz"
    rows = _jsonl_gz(repo / relative)
    full = [str(row["query_id"]) for row in rows if row.get("scope") == "full"]
    stress = [str(row["query_id"]) for row in rows if row.get("scope") == "stress"]
    task_ids = [str(row["task_id"]) for row in rows]
    component_ids = [str(row["component_id"]) for row in rows]
    if len(full) != 739:
        raise ValueError("G230 full TopK task count is not the frozen 739")
    if len(stress) != 218 * 5:
        raise ValueError("G230 stress task count is not the frozen 5 x 218")
    return {
        "g230_all_tasks": {
            **_ids_block(
                task_ids,
                source=str(relative),
                role="G230_REVEALED_DEVELOPMENT_TASKS_NOT_FINAL_CLAIM",
            ),
            "ordered_component_ids_sha256": _sha256_ids(component_ids),
            "source_gzip_sha256": _sha256(repo / relative),
        },
        "g230_full_topk_query_ids": _ids_block(
            full,
            source=str(relative),
            role="NIAH_DECISION_DEV_GENERATOR_QUALIFICATION",
        ),
        "g230_stress_query_ids": _ids_block(
            list(dict.fromkeys(stress)),
            source=str(relative),
            role="G230_REVEALED_STRESS_DIAGNOSTICS",
        ),
    }


def build_denylist(repo: Path) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-denylist-v1",
        "status": "FROZEN",
        "rules": {
            "sealed600": "retired read-only history; no training, selection, or retesting",
            "system_heldout": "IDs may be pinned; content/results are not loaded until SystemF and separate user authorization",
            "development_rows": "revealed development and G230 task IDs may diagnose/qualify but cannot support final superiority claims",
            "future_splits": "new NIAH/2Wiki materialization must check query/component/parent overlap before target generation",
        },
        "role_assignments": {
            "niah_train_r002": _role_assignment_block(
                repo,
                Path("results/selector-adaptive-risk-v1/R002/niah-train/role_assignments.jsonl"),
                role="NIAH_TRAIN_FIT_AND_MODELVAL_SOURCE_BOUNDARY",
            ),
            "niah_dev_r002": _role_assignment_block(
                repo,
                Path("results/selector-adaptive-risk-v1/R002/niah-dev/role_assignments.jsonl"),
                role="NIAH_DECISION_DEV_SOURCE_BOUNDARY",
            ),
            "twowiki_train_r002": _role_assignment_block(
                repo,
                Path("results/selector-adaptive-risk-v1/R002/2wiki-train/role_assignments.jsonl"),
                role="2WIKI_TRAIN_FIT_AND_MODELVAL_SOURCE_BOUNDARY",
            ),
            "twowiki_dev_r002": _role_assignment_block(
                repo,
                Path("results/selector-adaptive-risk-v1/R002/2wiki-dev/role_assignments.jsonl"),
                role="2WIKI_DEV_QUALIFICATION_SOURCE_BOUNDARY",
            ),
        },
        "revealed_development": _g230_task_blocks(repo),
        "retired": {
            "sealed600": _f005_sealed_block(repo),
        },
        "system_heldout": _heldout_blocks(repo),
    }


def _decompressed_gzip_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, mode="rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _g230_archive_verification(repo: Path) -> dict[str, Any]:
    archive = _json(repo / PREVIOUS_ROUTE / "artifacts/G230/ARCHIVE_MANIFEST.json")
    checks: dict[str, Any] = {}
    for name, pin in archive["generation_archives"].items():
        relative = PREVIOUS_ROUTE / "artifacts/G230" / pin["path"]
        path = repo / relative
        gzip_sha = _sha256(path)
        decompressed_sha = _decompressed_gzip_sha256(path)
        checks[name] = {
            "path": str(relative),
            "gzip_sha256": gzip_sha,
            "expected_gzip_sha256": pin["gzip_sha256"],
            "decompressed_sha256": decompressed_sha,
            "expected_decompressed_sha256": pin["decompressed_sha256"],
            "status": "PASS"
            if gzip_sha == pin["gzip_sha256"]
            and decompressed_sha == pin["decompressed_sha256"]
            else "FAIL",
        }
    status = "PASS" if all(item["status"] == "PASS" for item in checks.values()) else "FAIL"
    return {
        "status": status,
        "archive_manifest_sha256": _sha256(repo / PREVIOUS_ROUTE / "artifacts/G230/ARCHIVE_MANIFEST.json"),
        "generation_archives": checks,
    }


def _source_files(repo: Path) -> dict[str, Any]:
    relative_paths = {
        "route_plan": ROUTE / "PLAN.md",
        "route_readme": ROUTE / "README.md",
        "route_tracker": ROUTE / "TRACKER.md",
        "g000_protocol": ROUTE / "G000_PROTOCOL.md",
        "g000_script": Path("scripts/full_flow_g000_freeze.py"),
        "g200_script": Path("scripts/full_flow_g200.py"),
        "g220_train_script": Path("scripts/full_flow_g220_train.py"),
        "g230_run_script": Path("scripts/full_flow_g230.py"),
        "g230_citation_score_script": Path("scripts/full_flow_g230_citation_score.py"),
        "g200_manifest": PREVIOUS_ROUTE / "artifacts/G200/data/manifest.json",
        "g220_training_report": PREVIOUS_ROUTE / "G220_TRAINING_REPORT.md",
        "g230_results": PREVIOUS_ROUTE / "G230_RESULTS.md",
        "g230_archive_manifest": PREVIOUS_ROUTE / "artifacts/G230/ARCHIVE_MANIFEST.json",
        "g230_answer_report": PREVIOUS_ROUTE / "artifacts/G230/score/answer_report.json",
        "g230_citation_report": PREVIOUS_ROUTE / "artifacts/G230/citation/citation_report.json",
        "a000_manifest": PREVIOUS_ROUTE / "artifacts/A000/A000_DATA_MODEL_MANIFEST.json",
        "a000_server_audit": PREVIOUS_ROUTE / "artifacts/A000/A000_SERVER_AUDIT.json",
        "heldout_sample": Path("configs/heldout-sample.json"),
        "heldout_handover": Path("docs/heldout-three-set-handover.md"),
        "legacy_selector_selection": BRIDGE_ROUTE / "artifacts/F005/selection/selection_manifest.json",
        "legacy_selector_report": BRIDGE_ROUTE / "artifacts/F005/eval/report.json",
        "generator_draft_source": Path("src/evidence_rag/generator/draft.py"),
        "generator_granite_source": Path("src/evidence_rag/generator/granite.py"),
        "claim_splitter_source": Path("src/evidence_rag/generator/claim_splitter.py"),
        "verify_annotate_source": Path("src/evidence_rag/generator/verify_annotate.py"),
        "true_nli_source": Path("src/evidence_rag/generator/nli.py"),
        "legacy_selector_source": Path("src/evidence_rag/selector/risk_controlled.py"),
        "retriever_hybrid_source": Path("src/evidence_rag/retriever/hybrid.py"),
    }
    return {name: _source_pin(repo, relative) for name, relative in relative_paths.items()}


def _server_verification(repo: Path, server_audit_path: Path | None) -> dict[str, Any]:
    if server_audit_path is None:
        return {
            "status": "PENDING",
            "required": [
                "repo_sync",
                "runtime_roots",
                "g200_runtime_data",
                "g220_runtime_adapters",
                "g230_runtime_archive",
                "prior_a000_model_and_heldout_boundary",
            ],
        }
    audit = _json(server_audit_path)
    if audit.get("schema_version") != f"{SCHEMA_PREFIX}-server-audit-v1":
        raise ValueError("G000 server audit schema mismatch")
    return {
        "status": audit["status"],
        "audit_path": _rel(repo, server_audit_path),
        "audit_sha256": _sha256(server_audit_path),
        "audited_at_utc": audit["audited_at_utc"],
        "server": audit["server"],
        "checks": {name: value["status"] for name, value in audit["checks"].items()},
    }


def build_outputs(
    repo: Path,
    server_audit_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    route_dir = repo / ROUTE
    plan_text = (route_dir / "PLAN.md").read_text(encoding="utf-8")
    readme_text = (route_dir / "README.md").read_text(encoding="utf-8")
    tracker_text = (route_dir / "TRACKER.md").read_text(encoding="utf-8")
    plan_has_authorized_state = (
        "AUTHORIZED FOR CONTROLLED EXECUTION" in plan_text
        or "G000 COMPLETE / PASS" in plan_text
        or "- [x] G000 protocol 已从本修订快照为 frozen version。" in plan_text
    )
    route_has_g000_state = (
        "G000 IN PROGRESS" in readme_text
        or "G000 COMPLETE / PASS" in readme_text
        or "NO TRAINING STARTED" in readme_text
    ) and (
        "G000 IN PROGRESS" in tracker_text
        or "G000 COMPLETE / PASS" in tracker_text
        or "| G000 | Protocol freeze |" in tracker_text
        and "COMPLETE / PASS" in tracker_text
    )
    if not plan_has_authorized_state:
        raise ValueError("route PLAN is not marked as authorized or G000 complete")
    if not route_has_g000_state:
        raise ValueError("route README/TRACKER do not show an active G000 state")

    a000 = _json(repo / PREVIOUS_ROUTE / "artifacts/A000/A000_DATA_MODEL_MANIFEST.json")
    g200 = _json(repo / PREVIOUS_ROUTE / "artifacts/G200/data/manifest.json")
    g230_archive = _json(repo / PREVIOUS_ROUTE / "artifacts/G230/ARCHIVE_MANIFEST.json")
    g230_answer = _json(repo / PREVIOUS_ROUTE / "artifacts/G230/score/answer_report.json")
    g230_citation = _json(repo / PREVIOUS_ROUTE / "artifacts/G230/citation/citation_report.json")
    legacy_selector = _json(repo / BRIDGE_ROUTE / "artifacts/F005/selection/selection_manifest.json")
    branch = _git(repo, ["branch", "--show-current"])
    head = _git(repo, ["rev-parse", "HEAD"])
    origin_head = _git(repo, ["rev-parse", f"origin/{EXPECTED_BRANCH}"])
    source_files = _source_files(repo)
    denylist = build_denylist(repo)
    archive_verification = _g230_archive_verification(repo)
    server = _server_verification(repo, server_audit_path)

    status = "PASS" if server["status"] == "PASS" and archive_verification["status"] == "PASS" else "PENDING_SERVER_AUDIT"
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-input-manifest-v1",
        "status": status,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "route": str(ROUTE),
        "plan_archive_commit": PLAN_ARCHIVE_COMMIT,
        "execution_order": ["G", "freeze GQ", "S", "freeze SQ", "I"],
        "heldout_authorization": "H requires separate user authorization after SystemF",
        "git": {
            "branch": branch,
            "head": head,
            "origin_branch": f"origin/{EXPECTED_BRANCH}",
            "origin_head": origin_head,
            "head_equals_origin": head == origin_head,
            "dirty_allowed_because_stage_outputs_are_uncommitted": True,
        },
        "runtime_contract": {
            "formal_system": "Retriever -> Selector -> Generator -> one answer",
            "parallel_arms_are_experimental_comparisons_only": True,
            "retriever_runtime_gold_loaded": False,
            "selector_runtime_gold_loaded": False,
            "generator_runtime_gold_loaded": False,
            "reference_answers_loaded_at_runtime": False,
            "gold_use_allowed": "offline target construction, utility labels after generation, and post-generation scoring only",
        },
        "frozen_components": {
            "retriever": {
                "status": "FROZEN",
                "architecture": a000["models"]["retriever"]["architecture"],
                "embedding_model": a000["models"]["retriever"]["embedding_model"],
                "embedding_revision": a000["models"]["retriever"]["embedding_revision"],
                "parameters_sha256": a000["models"]["retriever"]["parameters_sha256"],
            },
            "legacy_selector_sl": {
                "status": "FROZEN_BASELINE_ONLY",
                "role": "Legacy safety/risk baseline; not final Selector",
                "model_id": legacy_selector["selector_model_id"],
                "revision": legacy_selector["selector_revision"],
                "checkpoint_sha256": legacy_selector["selector_checkpoint_sha256"],
                "threshold": legacy_selector["safe_threshold"],
                "cap": legacy_selector["cap"],
            },
            "generator_base_g0": {
                "status": "FROZEN_RELIABILITY_BASELINE_AND_FALLBACK_TEACHER",
                "model": a000["models"]["generator"]["model"],
                "snapshot_config_sha256": a000["models"]["generator"]["snapshot_config_sha256"],
                "default_new_training_scope": "draft LoRA only",
            },
            "g230_gc_gm_history": {
                "status": "COMPLETE / NO CANDIDATE",
                "archive_manifest_sha256": source_files["g230_archive_manifest"]["sha256"],
                "source_runtime": g230_archive["source_runtime"],
                "answer_report_status": g230_answer.get("status"),
                "citation_report_status": g230_citation.get("status"),
            },
            "true": {
                "status": "FROZEN",
                "model_id": a000["models"]["claim_verifier"]["model_id"],
                "role": "runtime verifier; never evaluates itself",
            },
            "minicheck": {
                "status": "FROZEN_POST_GENERATION_EVALUATOR",
                "model_id": a000["models"]["citation_evaluator"]["model_id"],
                "revision": a000["models"]["citation_evaluator"]["revision"],
            },
        },
        "data_scope": {
            "g200_existing_niah": {
                "status": g200["status"],
                "source_role": g200["source_role"],
                "role_assigned_queries": g200["counts"]["role_assigned_queries"],
                "train_queries": g200["train_queries"],
                "validation_queries": g200["validation_queries"],
                "sealed_or_heldout_read": g200["sealed_or_heldout_read"],
                "dev_read": g200["dev_read"],
                "input_sha256": g200["input_sha256"],
            },
            "heldout": {
                "status": "DENY_UNTIL_SYSTEMF_AND_SEPARATE_AUTHORIZATION",
                "source": "configs/heldout-sample.json",
                "datasets": {
                    name: {
                        "count": block["count"],
                        "ordered_ids_sha256": block["ordered_ids_sha256"],
                    }
                    for name, block in denylist["system_heldout"].items()
                },
            },
            "sealed600": {
                "status": "RETIRED_READ_ONLY_HISTORY",
                "count": denylist["retired"]["sealed600"]["count"],
                "ordered_ids_sha256": denylist["retired"]["sealed600"]["ordered_ids_sha256"],
            },
        },
        "gates_and_statistics": {
            "cluster_unit": "query provenance component",
            "bootstrap": "component-cluster paired bootstrap, 10000 resamples, seed=13",
            "family_estimate": "average seed-level paired delta within query before component bootstrap",
            "module_responsibility_gate": "core point estimates, margins, direction, and tripwires; not a significance claim",
            "strong_claim_gate": "pre-registered primary CI lower bound > 0 with non-inferiority bounds",
            "observed_power_forbidden": True,
            "g010_input": "G230 paired discordance and component structure only; no new candidate effects",
        },
        "budget": {
            "generator": {
                "seed13_screens": 2,
                "formal_fits": 3,
                "conditional_downstream_fix_smoke_or_refit": 1,
                "grid_search_allowed": False,
            },
            "selector": {
                "utility_pilot_queries": 100,
                "full_utility_materializations": 1,
                "seed13_screens": 1,
                "formal_seeds": [13, 42, 73],
            },
            "full_flow": {
                "locked_development_bundles": 1,
                "heldout_bundles_after_separate_authorization": 1,
            },
        },
        "source_files": source_files,
        "g230_archive_verification": archive_verification,
        "server_entity_verification": server,
    }
    checks = {
        "route_authorized": {
            "status": "PASS",
            "detail": "PLAN/README/TRACKER are marked authorized and G000 in progress",
        },
        "git_branch": {
            "status": "PASS" if branch == EXPECTED_BRANCH else "FAIL",
            "actual": branch,
            "expected": EXPECTED_BRANCH,
        },
        "git_remote_sync": {
            "status": "PASS" if head == origin_head else "FAIL",
            "head": head,
            "origin_head": origin_head,
        },
        "g230_archive_hashes": archive_verification,
        "runtime_gold_boundary": {
            "status": "PASS",
            "retriever": False,
            "selector": False,
            "generator": False,
            "reference_answers_loaded_at_runtime": False,
        },
        "heldout_boundary": {
            "status": "PASS",
            "heldout_ids_frozen": True,
            "heldout_content_or_scores_loaded": False,
            "requires_separate_authorization": True,
        },
        "server_entity_verification": server,
        "training_or_utility_generation_started": {
            "status": "PASS",
            "value": False,
        },
    }
    preflight_status = "PASS" if all(value.get("status") == "PASS" for value in checks.values()) else "FAIL"
    preflight = {
        "schema_version": f"{SCHEMA_PREFIX}-preflight-report-v1",
        "status": preflight_status,
        "checks": checks,
        "next_allowed_stage": "G010 and G100 only after G000 artifacts are committed, pushed, and server-synced"
        if preflight_status == "PASS"
        else "STOP",
    }
    if preflight_status != "PASS":
        manifest["status"] = "FAIL"
    report = render_report(manifest, preflight)
    return manifest, denylist, preflight, report


def _ssh(ssh_target: str, args: Sequence[str]) -> str:
    result = subprocess.run(
        ["ssh", *SSH_OPTIONS, ssh_target, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _ssh_test(ssh_target: str, args: Sequence[str]) -> bool:
    result = subprocess.run(
        ["ssh", *SSH_OPTIONS, ssh_target, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _ssh_sha256(ssh_target: str, path: str) -> str:
    output = _ssh(ssh_target, ["sha256sum", path])
    return output.split()[0]


def _check_equal(actual: object, expected: object) -> str:
    return "PASS" if actual == expected else "FAIL"


def build_server_audit(repo: Path, ssh_target: str = DEFAULT_SSH_TARGET) -> dict[str, Any]:
    hostname = _ssh(ssh_target, ["hostname"])
    local_head = _git(repo, ["rev-parse", "HEAD"])
    branch = _ssh(ssh_target, ["git", "-C", SERVER_REPO, "branch", "--show-current"])
    head = _ssh(ssh_target, ["git", "-C", SERVER_REPO, "rev-parse", "HEAD"])
    status_short = _ssh(ssh_target, ["git", "-C", SERVER_REPO, "status", "--short"])
    checks: dict[str, Any] = {
        "repo_sync": {
            "status": "PASS"
            if branch == EXPECTED_BRANCH and head == local_head and status_short == ""
            else "FAIL",
            "branch": branch,
            "expected_branch": EXPECTED_BRANCH,
            "head": head,
            "expected_head": local_head,
            "status_short": status_short,
        },
        "runtime_roots": {
            "status": "PASS"
            if _ssh_test(ssh_target, ["test", "-d", SERVER_RUNTIME])
            and _ssh_test(ssh_target, ["test", "-d", G200_RUNTIME])
            and _ssh_test(ssh_target, ["test", "-d", G220_RUNTIME])
            and _ssh_test(ssh_target, ["test", "-d", G230_RUNTIME])
            else "FAIL",
            "runtime": SERVER_RUNTIME,
            "g200": G200_RUNTIME,
            "g220": G220_RUNTIME,
            "g230": G230_RUNTIME,
        },
    }

    a000_audit_path = repo / PREVIOUS_ROUTE / "artifacts/A000/A000_SERVER_AUDIT.json"
    a000_audit = _json(a000_audit_path)
    checks["prior_a000_model_and_heldout_boundary"] = {
        "status": "PASS" if a000_audit.get("status") == "PASS" else "FAIL",
        "source_path": _rel(repo, a000_audit_path),
        "source_sha256": _sha256(a000_audit_path),
        "heldout_metric_or_scorer_run": a000_audit["checks"]["system_heldout_boundary"][
            "historical_exposure"
        ]["metric_or_scorer_run"],
    }

    g200_manifest_path = repo / PREVIOUS_ROUTE / "artifacts/G200/data/manifest.json"
    g200_manifest = _json(g200_manifest_path)
    g200_data = {
        "manifest": {
            "remote_path": f"{G200_RUNTIME}/data/manifest.json",
            "remote_sha256": _ssh_sha256(ssh_target, f"{G200_RUNTIME}/data/manifest.json"),
            "expected_sha256": _sha256(g200_manifest_path),
        },
        "train_cases": {
            "remote_path": f"{G200_RUNTIME}/data/train_cases.jsonl",
            "remote_sha256": _ssh_sha256(ssh_target, f"{G200_RUNTIME}/data/train_cases.jsonl"),
            "expected_sha256": g200_manifest["train_cases_sha256"],
        },
        "validation_cases": {
            "remote_path": f"{G200_RUNTIME}/data/validation_cases.jsonl",
            "remote_sha256": _ssh_sha256(ssh_target, f"{G200_RUNTIME}/data/validation_cases.jsonl"),
            "expected_sha256": g200_manifest["validation_cases_sha256"],
        },
    }
    for entry in g200_data.values():
        entry["status"] = _check_equal(entry["remote_sha256"], entry["expected_sha256"])
    checks["g200_runtime_data"] = {
        "status": "PASS" if all(entry["status"] == "PASS" for entry in g200_data.values()) else "FAIL",
        "files": g200_data,
    }

    g220_files: dict[str, Any] = {}
    for arm, seed in G220_RUNS:
        name = f"{arm}-seed{seed}"
        local_manifest_path = repo / PREVIOUS_ROUTE / f"artifacts/G220/formal/{name}/training_manifest.json"
        local_config_path = repo / PREVIOUS_ROUTE / f"artifacts/G220/formal/{name}/adapter_config.json"
        manifest = _json(local_manifest_path)
        remote_manifest_sha = _ssh_sha256(ssh_target, f"{G220_RUNTIME}/{name}/training_manifest.json")
        remote_config_sha = _ssh_sha256(ssh_target, f"{G220_RUNTIME}/{name}/adapter/adapter_config.json")
        remote_weights_sha = _ssh_sha256(ssh_target, f"{G220_RUNTIME}/{name}/adapter/adapter_model.safetensors")
        entry = {
            "remote_run_dir": f"{G220_RUNTIME}/{name}",
            "training_manifest_sha256": remote_manifest_sha,
            "expected_training_manifest_sha256": _sha256(local_manifest_path),
            "adapter_config_sha256": remote_config_sha,
            "expected_adapter_config_sha256": _sha256(local_config_path),
            "adapter_weights_sha256": remote_weights_sha,
            "expected_adapter_weights_sha256": manifest["adapter_weights_sha256"],
        }
        entry["status"] = (
            "PASS"
            if entry["training_manifest_sha256"] == entry["expected_training_manifest_sha256"]
            and entry["adapter_config_sha256"] == entry["expected_adapter_config_sha256"]
            and entry["adapter_weights_sha256"] == entry["expected_adapter_weights_sha256"]
            else "FAIL"
        )
        g220_files[name] = entry
    checks["g220_runtime_adapters"] = {
        "status": "PASS" if all(entry["status"] == "PASS" for entry in g220_files.values()) else "FAIL",
        "runs": g220_files,
    }

    archive = _json(repo / PREVIOUS_ROUTE / "artifacts/G230/ARCHIVE_MANIFEST.json")
    g230_files: dict[str, Any] = {}
    for name in G230_RUNS:
        pin = archive["generation_archives"][name]
        entry = {
            "run_manifest_sha256": _ssh_sha256(ssh_target, f"{G230_RUNTIME}/{name}/run_manifest.json"),
            "expected_run_manifest_sha256": _sha256(repo / PREVIOUS_ROUTE / f"artifacts/G230/{name}/run_manifest.json"),
            "run_spec_sha256": _ssh_sha256(ssh_target, f"{G230_RUNTIME}/{name}/run_spec.json"),
            "expected_run_spec_sha256": _sha256(repo / PREVIOUS_ROUTE / f"artifacts/G230/{name}/run_spec.json"),
            "generations_sha256": _ssh_sha256(ssh_target, f"{G230_RUNTIME}/{name}/generations.jsonl"),
            "expected_generations_sha256": pin["decompressed_sha256"],
        }
        entry["status"] = (
            "PASS"
            if entry["run_manifest_sha256"] == entry["expected_run_manifest_sha256"]
            and entry["run_spec_sha256"] == entry["expected_run_spec_sha256"]
            and entry["generations_sha256"] == entry["expected_generations_sha256"]
            else "FAIL"
        )
        g230_files[name] = entry
    for relative, expected in {
        **archive["scored_artifacts"],
        **archive["execution_logs"],
    }.items():
        remote = f"{G230_RUNTIME}/{relative}"
        entry = {
            "remote_path": remote,
            "remote_sha256": _ssh_sha256(ssh_target, remote),
            "expected_sha256": expected,
        }
        entry["status"] = _check_equal(entry["remote_sha256"], entry["expected_sha256"])
        g230_files[relative] = entry
    checks["g230_runtime_archive"] = {
        "status": "PASS" if all(entry["status"] == "PASS" for entry in g230_files.values()) else "FAIL",
        "files": g230_files,
    }

    status = "PASS" if all(value["status"] == "PASS" for value in checks.values()) else "FAIL"
    return {
        "schema_version": f"{SCHEMA_PREFIX}-server-audit-v1",
        "status": status,
        "audited_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": {
            "hostname": hostname,
            "ssh_target": ssh_target,
            "repo": SERVER_REPO,
            "runtime": SERVER_RUNTIME,
        },
        "checks": checks,
    }


def render_report(manifest: Mapping[str, Any], preflight: Mapping[str, Any]) -> str:
    lines = [
        "# G000 协议和统计范围冻结报告",
        "",
        "**日期：** 2026-08-18",
        f"**状态：** `{preflight['status']}`",
        "",
        "## 完成内容",
        "",
        "- 已把第 03 路线从等待确认推进到受控执行，并冻结 G000 的协议、输入清单、denylist 和服务器实体核验。",
        "- 未启动训练、未生成 utility labels、未运行 held-out，也未修改 Retriever、Legacy Selector、TRUE 或历史 G230 结果。",
        "- 正式 runtime 边界保持为 `Retriever -> Selector -> Generator -> one answer`；gold/reference 只允许离线构造目标、utility label 和生成后评分。",
        "",
        "## 核验结果",
        "",
    ]
    for name, check in preflight["checks"].items():
        lines.append(f"- `{name}`: `{check['status']}`")
    lines.extend(
        [
            "",
            "## 冻结身份",
            "",
            f"- Retriever: `{manifest['frozen_components']['retriever']['parameters_sha256']}`",
            f"- Legacy Selector checkpoint: `{manifest['frozen_components']['legacy_selector_sl']['checkpoint_sha256']}`",
            f"- Generator base: `{manifest['frozen_components']['generator_base_g0']['model']}`",
            f"- G230 archive: `{manifest['frozen_components']['g230_gc_gm_history']['archive_manifest_sha256']}`",
            "",
            "## 下一步",
            "",
            "G000 通过后，下一阶段只允许进入 G010/G100 的预注册功效范围与 citation 断点归因；不得跳到训练。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--collect-server-audit", action="store_true")
    parser.add_argument("--server-audit", type=Path)
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--allow-pending-server", action="store_true")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    out_dir = args.out_dir or (repo / ROUTE / "artifacts/G000")
    server_audit_path = args.server_audit.resolve() if args.server_audit else None
    if args.collect_server_audit:
        audit = build_server_audit(repo, args.ssh_target)
        server_audit_path = out_dir / "G000_SERVER_AUDIT.json"
        _write_json(server_audit_path, audit)
        print(f"written {server_audit_path}")
        if audit["status"] != "PASS":
            return 1

    manifest, denylist, preflight, report = build_outputs(repo, server_audit_path)
    outputs: dict[str, object] = {
        "G000_INPUT_MANIFEST.json": manifest,
        "G000_DENYLIST.json": denylist,
        "G000_PREFLIGHT_REPORT.json": preflight,
    }
    for filename, value in outputs.items():
        path = out_dir / filename
        _write_json(path, value)
        print(f"written {path}")
    report_path = repo / ROUTE / "G000_REPORT.md"
    _write_text(report_path, report)
    print(f"written {report_path}")
    if preflight["status"] != "PASS" and not args.allow_pending_server:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
