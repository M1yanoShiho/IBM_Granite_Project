import json
from pathlib import Path

import pytest

from evidence_rag.evaluation.paired_metric_cli import main

METRIC = "selector.core.conditional_document_recall"


def _report(
    path: Path,
    values: dict[str, float | None],
    *,
    dataset_signature: str = "dataset-signature",
    metric_registry_signature: str = "metric-registry-signature",
    direction: str = "higher",
) -> Path:
    per_case = [
        {"query_id": qid, "schema_version": "1.0",
         "metrics": {METRIC: {"reason": None, "value": value}}}
        for qid, value in values.items()
    ]
    scored = [value for value in values.values() if value is not None]
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "stage": "selector",
                "dataset_signature": dataset_signature,
                "metric_registry_signature": metric_registry_signature,
                "case_ids": list(values),
                "per_case": per_case,
                "aggregate": {
                    METRIC: {
                        "mean": sum(scored) / len(scored) if scored else None,
                        "n_scored": len(scored),
                        "n_total": len(values),
                    }
                },
                "directions": {METRIC: direction},
            }
        ),
        encoding="utf-8",
    )
    return path


def _component_map(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_cli_pairs_metric_and_prints_stats(tmp_path, capsys) -> None:
    on = _report(tmp_path / "on.json", {"q1": 1.0, "q2": 1.0, "q3": 0.0})
    off = _report(tmp_path / "off.json", {"q1": 0.0, "q2": 1.0, "q3": 0.0})

    exit_code = main(
        ["--on-report", str(on), "--off-report", str(off), "--metric", METRIC, "--iterations", "2000"]
    )

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metric"] == METRIC
    assert out["n_paired"] == 3
    assert out["n_queries"] == 3
    assert out["n_clusters"] == 3
    assert out["n_total"] == 3
    assert out["n_unscored"] == 0
    assert out["dataset_signature"] == "dataset-signature"
    assert out["metric_direction"] == "higher"
    assert abs(out["delta"] - 1 / 3) < 1e-9
    assert 0.0 <= out["p_value"] <= 1.0


def test_cli_uses_component_map_for_clustered_inference(tmp_path, capsys) -> None:
    on = _report(tmp_path / "on.json", {"q1": 1.0, "q2": 1.0, "q3": 0.0})
    off = _report(tmp_path / "off.json", {"q1": 0.0, "q2": 1.0, "q3": 0.0})
    component_map = _component_map(
        tmp_path / "components.jsonl",
        [
            {"query_id": "q1", "component_id": "component-a", "ignored": "allowed"},
            {"query_id": "q2", "component_id": "component-a"},
            {"query_id": "q3", "component_id": "component-b"},
        ],
    )

    exit_code = main(
        [
            "--on-report",
            str(on),
            "--off-report",
            str(off),
            "--metric",
            METRIC,
            "--component-map",
            str(component_map),
            "--iterations",
            "200",
        ]
    )

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_queries"] == 3
    assert out["n_clusters"] == 2


def test_cli_rejects_duplicate_query_in_component_map(tmp_path) -> None:
    on = _report(tmp_path / "on.json", {"q1": 1.0})
    off = _report(tmp_path / "off.json", {"q1": 0.0})
    component_map = _component_map(
        tmp_path / "components.jsonl",
        [
            {"query_id": "q1", "component_id": "component-a"},
            {"query_id": "q1", "component_id": "component-b"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate component-map query ID 'q1'"):
        main(
            [
                "--on-report",
                str(on),
                "--off-report",
                str(off),
                "--metric",
                METRIC,
                "--component-map",
                str(component_map),
            ]
        )


@pytest.mark.parametrize(
    "component_rows",
    [
        [
            {"query_id": "q1", "component_id": "component-a"},
            {"query_id": "q2", "component_id": "component-a"},
        ],
        [
            {"query_id": "q1", "component_id": "component-a"},
            {"query_id": "q2", "component_id": "component-a"},
            {"query_id": "q3", "component_id": "component-b"},
            {"query_id": "q-extra", "component_id": "component-c"},
        ],
    ],
    ids=["missing-query", "extra-query"],
)
def test_cli_rejects_component_map_with_different_query_keys(
    tmp_path,
    component_rows: list[dict[str, str]],
) -> None:
    on = _report(tmp_path / "on.json", {"q1": 1.0, "q2": 1.0, "q3": 0.0})
    off = _report(tmp_path / "off.json", {"q1": 0.0, "q2": 1.0, "q3": 0.0})
    component_map = _component_map(tmp_path / "components.jsonl", component_rows)

    with pytest.raises(ValueError, match="metric/component query keys differ"):
        main(
            [
                "--on-report",
                str(on),
                "--off-report",
                str(off),
                "--metric",
                METRIC,
                "--component-map",
                str(component_map),
            ]
        )


def test_cli_rejects_different_dataset_signatures(tmp_path) -> None:
    on = _report(tmp_path / "on.json", {"q1": 1.0}, dataset_signature="dataset-a")
    off = _report(tmp_path / "off.json", {"q1": 0.0}, dataset_signature="dataset-b")

    with pytest.raises(ValueError, match="different dataset signatures"):
        main(["--on-report", str(on), "--off-report", str(off), "--metric", METRIC])


def test_cli_rejects_different_metric_registry_signatures(tmp_path) -> None:
    on = _report(
        tmp_path / "on.json", {"q1": 1.0}, metric_registry_signature="registry-a"
    )
    off = _report(
        tmp_path / "off.json", {"q1": 0.0}, metric_registry_signature="registry-b"
    )

    with pytest.raises(ValueError, match="different metric registry signatures"):
        main(["--on-report", str(on), "--off-report", str(off), "--metric", METRIC])


def test_cli_rejects_different_metric_directions(tmp_path) -> None:
    on = _report(tmp_path / "on.json", {"q1": 1.0}, direction="higher")
    off = _report(tmp_path / "off.json", {"q1": 0.0}, direction="lower")

    with pytest.raises(ValueError, match="different metric registries or directions"):
        main(["--on-report", str(on), "--off-report", str(off), "--metric", METRIC])


def test_cli_rejects_duplicate_query_ids_in_a_report(tmp_path) -> None:
    on = _report(tmp_path / "on.json", {"q1": 1.0})
    payload = json.loads(on.read_text(encoding="utf-8"))
    payload["case_ids"].append("q1")
    payload["per_case"].append(payload["per_case"][0])
    payload["aggregate"][METRIC]["n_total"] = 2
    on.write_text(json.dumps(payload), encoding="utf-8")
    off = _report(tmp_path / "off.json", {"q1": 0.0})

    with pytest.raises(ValueError, match="case IDs must be unique"):
        main(["--on-report", str(on), "--off-report", str(off), "--metric", METRIC])


def test_cli_rejects_nonfinite_metric_value(tmp_path) -> None:
    on = _report(tmp_path / "on.json", {"q1": float("nan")})
    off = _report(tmp_path / "off.json", {"q1": 0.0})

    with pytest.raises(ValueError, match="invalid stage evaluation report"):
        main(["--on-report", str(on), "--off-report", str(off), "--metric", METRIC])
