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

import experiment04_goal3 as g3  # noqa: E402


def _write_dataset(root: Path, dataset: str, count: int) -> None:
    score_dir = root / "scores"
    score_dir.mkdir(parents=True)
    arm_entries: dict[str, object] = {}
    for arm_index, arm in enumerate(g3.PRIMARY_ARMS):
        rar = 1.0 if arm.startswith("ours_") else float(arm_index == 1)
        per_query = [
            {
                "query_id": f"q{index:04d}",
                "component_id": f"c{index // 2:04d}",
                "metrics": {"ret": 1.0, "sel": 1.0, "ans": rar, "cit": rar, "rar": rar},
                "failure_reason": None,
            }
            for index in range(count)
        ]
        artifact = {
            "score": {
                "n_queries": count,
                "aggregate": {"ret": 1.0, "sel": 1.0, "ans": rar, "cit": rar, "rar": rar},
                "per_query": per_query,
            }
        }
        path = score_dir / f"{arm}.json"
        g3._write_json(path, artifact)
        arm_entries[arm] = {"score_sha256": g3._sha256(path)}
    g3._write_json(
        root / "score_manifest.json",
        {
            "schema_version": "experiment04.dataset_score_manifest.v1",
            "status": "PASS",
            "query_count": count,
            "generation_manifest_sha256": "0" * 64,
            "arms": arm_entries,
        },
    )


def test_summarize_builds_traceable_table_and_exact_7700_row_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dirs: list[str] = []
    for dataset, count in g3.EXPECTED_COUNTS.items():
        directory = tmp_path / dataset
        _write_dataset(directory, dataset, count)
        dataset_dirs.append(f"{dataset}={directory}")
    output = tmp_path / "summary"
    monkeypatch.setattr(
        g3,
        "paired_component_cluster_bootstrap",
        lambda **kwargs: {
            "difference": 1.0,
            "ci95_low": 1.0,
            "ci95_high": 1.0,
            "resamples": kwargs["resamples"],
            "seed": kwargs["seed"],
        },
    )

    status = g3.summarize_stage(
        argparse.Namespace(dataset_dir=dataset_dirs, output_dir=output)
    )

    audit = json.loads((output / "goal3_audit.json").read_text(encoding="utf-8"))
    table = json.loads((output / "table1.json").read_text(encoding="utf-8"))
    assert status == 0
    assert audit["status"] == "PASS"
    assert audit["observed_total_outputs"] == 7_700
    assert set(table["datasets"]) == set(g3.EXPECTED_COUNTS)
    for rows in table["datasets"].values():
        ours = rows[-1]
        assert ours["system"] == "Ours"
        assert ours["rar"] == {"mean": 1.0, "sample_sd": 0.0}


def test_slurm_hides_scorer_path_until_after_generation_freeze() -> None:
    script = (ROOT / "scripts/run_experiment04_goal3_dataset.slurm").read_text(encoding="utf-8")

    unset = script.index("unset EXP04_SIDECAR")
    prepare = script.index("experiment04_goal3.py prepare")
    generate = script.index("experiment04_goal3.py generate-direct")
    freeze = script.index("experiment04_goal3.py freeze")
    score = script.index("experiment04_goal3.py score")
    sidecar = script.index('--sidecar "$scorer_sidecar"')
    assert unset < prepare < generate < freeze < score < sidecar
    assert script.count('--sidecar "$scorer_sidecar"') == 1
