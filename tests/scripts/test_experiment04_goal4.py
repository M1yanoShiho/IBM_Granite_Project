from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import experiment04_goal4 as g4  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _score(count: int, rar: float) -> dict[str, object]:
    return {
        "n_queries": count,
        "aggregate": {"ret": 1.0, "sel": 1.0, "ans": rar, "cit": rar, "rar": rar},
        "per_query": [
            {
                "query_id": f"q{index:04d}",
                "component_id": f"c{index // 2:04d}",
                "metrics": {"ret": 1.0, "sel": 1.0, "ans": rar, "cit": rar, "rar": rar},
                "failure_reason": None,
            }
            for index in range(count)
        ],
    }


def _write_goal4_dataset(root: Path, count: int) -> None:
    entries: dict[str, object] = {}
    for arm_index, arm in enumerate(g4.ABLATION_ARMS):
        path = root / "scores" / f"{arm}.json"
        _write_json(path, {"score": _score(count, float(arm_index == 0))})
        entries[arm] = {"score_sha256": g4._sha256(path)}
    _write_json(
        root / "score_manifest.json",
        {
            "schema_version": "experiment04.goal4_dataset_score_manifest.v1",
            "status": "PASS",
            "query_count": count,
            "ordered_ids_sha256": "a" * 64,
            "generation_manifest_sha256": "b" * 64,
            "full_regenerated": False,
            "arms": entries,
        },
    )


def _write_goal3_dataset(root: Path, count: int) -> None:
    generation = root / "generations" / "ours_seed13.jsonl"
    generation.parent.mkdir(parents=True, exist_ok=True)
    generation.write_text("synthetic-frozen-full\n", encoding="utf-8")
    score = root / "scores" / "ours_seed13.json"
    _write_json(score, {"score": _score(count, 1.0)})
    _write_json(
        root / "generation_manifest.json",
        {
            "bundle_status": "PASS",
            "arms": {
                "ours_seed13": {"file_sha256": g4._sha256(generation)},
            },
        },
    )
    _write_json(
        root / "score_manifest.json",
        {
            "status": "PASS",
            "query_count": count,
            "ordered_ids_sha256": "a" * 64,
            "arms": {"ours_seed13": {"score_sha256": g4._sha256(score)}},
        },
    )


def test_summarize_reuses_full_and_builds_exact_3300_new_output_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goal4_dirs: list[str] = []
    goal3_dirs: list[str] = []
    for dataset, count in g4.EXPECTED_COUNTS.items():
        goal4_dir = tmp_path / "goal4" / dataset
        goal3_dir = tmp_path / "goal3" / dataset
        _write_goal4_dataset(goal4_dir, count)
        _write_goal3_dataset(goal3_dir, count)
        goal4_dirs.append(f"{dataset}={goal4_dir}")
        goal3_dirs.append(f"{dataset}={goal3_dir}")
    monkeypatch.setattr(
        g4,
        "paired_component_cluster_bootstrap",
        lambda **kwargs: {
            "difference": 0.5,
            "ci95_low": 0.4,
            "ci95_high": 0.6,
            "resamples": kwargs["resamples"],
            "seed": kwargs["seed"],
        },
    )
    output = tmp_path / "summary"

    status = g4.summarize_stage(
        argparse.Namespace(
            goal4_dataset_dir=goal4_dirs,
            full_dataset_dir=goal3_dirs,
            output_dir=output,
        )
    )

    audit = json.loads((output / "goal4_audit.json").read_text(encoding="utf-8"))
    table = json.loads((output / "table2.json").read_text(encoding="utf-8"))
    assert status == 0
    assert audit["status"] == "PASS"
    assert audit["observed_new_outputs"] == 3_300
    assert audit["observed_reused_full_rows"] == 1_100
    assert audit["full_regenerated"] is False
    assert all(rows[0]["configuration"] == "Full" for rows in table["datasets"].values())


def test_slurm_hides_sidecar_and_has_no_full_generation_path() -> None:
    script = (ROOT / "scripts/run_experiment04_goal4_dataset.slurm").read_text(encoding="utf-8")

    unset = script.index("unset EXP04_SIDECAR")
    prepare = script.index("experiment04_goal4.py prepare")
    grounded = script.index("experiment04_goal4.py generate-grounded")
    direct = script.index("experiment04_goal4.py generate-direct")
    freeze = script.index("experiment04_goal4.py freeze")
    score = script.index("experiment04_goal4.py score")
    sidecar = script.index('--sidecar "$scorer_sidecar"')
    assert unset < prepare < grounded < direct < freeze < score < sidecar
    assert script.count('--sidecar "$scorer_sidecar"') == 1
    assert "generate-full" not in script
    assert "ours_seed13" not in script
    assert "/user/work/" not in script
    assert "EVIDENCE_RAG_ROOT" in script
    assert "EVIDENCE_RAG_VENV" in script
    assert "MODEL_CACHE_DIR" in script
