from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import evidence_rag.cli.run_selector_lean as lean_cli
from evidence_rag.cli.run_selector_lean import (
    LeanTrainingRow,
    build_epoch_microbatches,
    calibrate_selector_lean,
    final_input_sha256,
    pair_digest,
    row_digest,
)
from evidence_rag.evaluation.selector_lean import (
    DevelopmentCandidateResult,
    FrozenPolicy,
    build_policy_candidates,
    thresholds_from_train_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "selector" / "lean_v3.toml"


def _row(
    dataset: str,
    query_id: str,
    evidence_id: str,
    *,
    protect: int | None = 1,
    harm: int | None = None,
) -> LeanTrainingRow:
    return LeanTrainingRow(
        dataset_kind=dataset,  # type: ignore[arg-type]
        query_id=query_id,
        evidence_id=evidence_id,
        retrieval_rank=1,
        question=f"question {query_id}",
        candidate_text=f"candidate {evidence_id}",
        protect_label=protect,
        protect_mask=protect is not None,
        harm_label=harm,
        harm_mask=harm is not None,
    )


def test_frozen_config_loads_and_rejects_optimizer_drift(tmp_path: Path) -> None:
    config = lean_cli._load_lean_config(CONFIG_PATH)
    assert config.learning_rate == pytest.approx(2e-5)
    assert config.expected_active_rows == {"niah": 5312, "2wiki": 5311}
    assert config.expected_strict_pairs == 870
    assert config.expected_microbatches_per_source == 1328
    assert config.expected_optimizer_steps_per_epoch == 664

    changed = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "learning_rate = 0.00002", "learning_rate = 0.00003"
    )
    changed_path = tmp_path / "changed.toml"
    changed_path.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="learning_rate"):
        lean_cli._load_lean_config(changed_path)


def test_pair_and_row_digests_match_the_frozen_utf8_payload() -> None:
    pair_payload = b"selector-lean-v3\n13\n2\nPAIR\nq1\nclean\ncf"
    assert (
        pair_digest(
            seed=13,
            epoch=2,
            query_id="q1",
            clean_evidence_id="clean",
            cf_evidence_id="cf",
        )
        == hashlib.sha256(pair_payload).hexdigest()
    )

    row = _row("2wiki", "q2", "ev2")
    row_payload = b"selector-lean-v3\n42\n3\n2wiki\nq2\nev2"
    assert row_digest(seed=42, epoch=3, row=row) == hashlib.sha256(row_payload).hexdigest()


def test_final_input_hash_changes_when_an_actual_raw_input_changes(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for name in (
        "niah_dataset",
        "niah_pool",
        "niah_components",
        "twowiki_dataset",
        "twowiki_pool",
        "twowiki_components",
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "payload.json").write_text(f'{{"name":"{name}"}}', encoding="utf-8")
        paths[name] = directory
    for name in ("niah_parent", "niah_assignment", "niah_provenance", "twowiki_parent"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(f'{{"name":"{name}"}}\n', encoding="utf-8")
        paths[name] = path

    kwargs = {
        "niah_dataset_manifest": paths["niah_dataset"] / "payload.json",
        "niah_source_parent": paths["niah_parent"],
        "niah_candidate_pool": paths["niah_pool"],
        "niah_components_directory": paths["niah_components"],
        "niah_assignment": paths["niah_assignment"],
        "niah_provenance": paths["niah_provenance"],
        "twowiki_dataset_manifest": paths["twowiki_dataset"] / "payload.json",
        "twowiki_source_parent": paths["twowiki_parent"],
        "twowiki_candidate_pool": paths["twowiki_pool"],
        "twowiki_components_directory": paths["twowiki_components"],
    }
    before = final_input_sha256(**kwargs)  # type: ignore[arg-type,unused-ignore]
    (paths["twowiki_pool"] / "payload.json").write_text('{"changed":true}', encoding="utf-8")
    after = final_input_sha256(**kwargs)  # type: ignore[arg-type,unused-ignore]
    assert before != after


def test_generator_snapshot_identity_binds_config_and_both_weight_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = lean_cli._load_lean_config(CONFIG_PATH)
    generator = config.raw["generator"]
    assert isinstance(generator, dict)
    snapshot_files = generator["snapshot_files"]
    assert isinstance(snapshot_files, dict)
    digests = {str(name): str(digest) for name, digest in snapshot_files.items()}
    digests.update(
        {
            "model-00001-of-00002.safetensors": next(
                digest for digest in lean_cli._GENERATOR_WEIGHT_SHA256 if digest.startswith("895")
            ),
            "model-00002-of-00002.safetensors": next(
                digest for digest in lean_cli._GENERATOR_WEIGHT_SHA256 if digest.startswith("de8")
            ),
        }
    )
    for filename in digests:
        (tmp_path / filename).write_bytes(b"fixture")
    monkeypatch.setattr(lean_cli, "_sha256_file", lambda path: digests[Path(path).name])

    identity = lean_cli._audit_generator_snapshot(tmp_path, config)
    assert len(identity) == 64

    digests["model-00002-of-00002.safetensors"] = "0" * 64
    with pytest.raises(ValueError, match="weight shards"):
        lean_cli._audit_generator_snapshot(tmp_path, config)


def test_full_frozen_schedule_is_balanced_pair_preserving_and_exact_once() -> None:
    niah: list[LeanTrainingRow] = []
    for index in range(870):
        query_id = f"pair-{index:04d}"
        niah.extend(
            (
                _row("niah", query_id, f"{query_id}-clean", protect=1, harm=0),
                _row("niah", query_id, f"{query_id}-cf", protect=0, harm=1),
            )
        )
    niah.extend(_row("niah", f"single-{index:04d}", f"niah-{index:04d}") for index in range(3572))
    twowiki = [_row("2wiki", f"tw-{index // 2:04d}", f"tw-{index:04d}") for index in range(5311)]

    schedule = build_epoch_microbatches(
        niah_rows=niah,
        twowiki_rows=twowiki,
        seed=13,
        epoch=1,
        expected_active_rows={"niah": 5312, "2wiki": 5311},
        expected_strict_pairs=870,
        expected_microbatches_per_source=1328,
    )

    assert len(schedule) == 2656
    assert len(schedule) // 4 == 664
    assert all(
        (schedule[index].dataset_kind, schedule[index + 1].dataset_kind) == ("niah", "2wiki")
        for index in range(0, len(schedule), 2)
    )
    paired = [batch for batch in schedule if batch.strict_pair_positions is not None]
    assert len(paired) == 870
    assert all(batch.strict_pair_positions == (0, 1) for batch in paired)
    scheduled = [row.identity for batch in schedule for row in batch.rows]
    assert len(scheduled) == 5312 + 5311
    assert len(set(scheduled)) == len(scheduled)


def test_ambiguous_pair_is_not_fabricated() -> None:
    niah = [
        _row("niah", "ambiguous", "clean-1", protect=1, harm=0),
        _row("niah", "ambiguous", "clean-2", protect=1, harm=0),
        _row("niah", "ambiguous", "cf", protect=0, harm=1),
        _row("niah", "single", "single"),
    ]
    twowiki = [_row("2wiki", f"q-{index}", f"ev-{index}") for index in range(4)]

    schedule = build_epoch_microbatches(
        niah_rows=niah,
        twowiki_rows=twowiki,
        seed=42,
        epoch=3,
        expected_strict_pairs=0,
    )
    assert all(batch.strict_pair_positions is None for batch in schedule)


def _development_result(
    *, quantile: float, cap: int, thresholds: tuple[tuple[int, float], ...]
) -> DevelopmentCandidateResult:
    harm = 0.03 if (quantile, cap) == (0.99, 1) else 0.02
    return DevelopmentCandidateResult(
        quantile=quantile,
        cap=cap,
        thresholds_by_seed=thresholds,
        mean_deletions_per_query=0.2,
        harmful_reduction_seed13=harm,
        harmful_reduction_seed42=harm,
        niah_recall_loss_seed13_pp=0.5,
        niah_recall_loss_seed42_pp=2.0,
        twowiki_recall_loss_seed13_pp=0.5,
        twowiki_recall_loss_seed42_pp=2.0,
        niah_chain_loss_seed13_pp=0.5,
        niah_chain_loss_seed42_pp=2.0,
        twowiki_chain_loss_seed13_pp=0.5,
        twowiki_chain_loss_seed42_pp=2.0,
        seed13_harm_beats_random=True,
        seed13_harm_beats_bottom=True,
        seed13_precision_beats_random=True,
        seed13_precision_beats_bottom=True,
    )


def test_calibrate_freezes_only_one_development_selected_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lean_cli, "_load_lean_config", lambda path: object())

    def fit_bundle(**kwargs: Any) -> Any:
        seed = kwargs["expected_seed"]
        return lean_cli._FitBundle(
            seed=seed,
            variant="NLI-base",
            checkpoint_sha256=("b" if seed == 13 else "c") * 64,
            safe_scores=(0.2 if seed == 13 else 0.4,) * 100,
        )

    monkeypatch.setattr(lean_cli, "_read_fit_bundle", fit_bundle)

    def development_results(*args: Any, **kwargs: Any) -> tuple[DevelopmentCandidateResult, ...]:
        del args, kwargs
        candidates = build_policy_candidates(
            thresholds_from_train_scores({13: [0.2] * 100, 42: [0.4] * 100})
        )
        return tuple(
            _development_result(
                quantile=candidate.quantile,
                cap=candidate.cap,
                thresholds=candidate.thresholds_by_seed,
            )
            for candidate in candidates
            if candidate.quantile is not None and candidate.cap is not None
        )

    monkeypatch.setattr(lean_cli, "_read_development_results", development_results)
    output = tmp_path / "frozen_policy.json"
    result = calibrate_selector_lean(
        config_path=tmp_path / "config.toml",
        variant="NLI-base",
        seed13_fit_directory=tmp_path / "seed13",
        seed42_fit_directory=tmp_path / "seed42",
        development_results_path=tmp_path / "development.json",
        code_commit="a" * 40,
        final_input_sha256="d" * 64,
        development_projection_sha256="e" * 64,
        generator_sha256="f" * 64,
        output_policy_path=output,
        base_stop_path=None,
    )

    policy = FrozenPolicy.from_dict(json.loads(output.read_text(encoding="utf-8")))
    assert result["action"] == "policy-frozen"
    assert (policy.quantile, policy.cap) == (0.99, 1)
    assert policy.development_projection_sha256 == "e" * 64


def test_development_results_are_bound_to_the_scoring_checkpoints(tmp_path: Path) -> None:
    path = tmp_path / "development.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "selector-lean-development-results-v1",
                "role": "crc-calibration",
                "variant": "NLI-base",
                "development_projection_sha256": "e" * 64,
                "checkpoint_sha256_by_seed": {"13": "b" * 64, "42": "c" * 64},
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    assert (
        lean_cli._read_development_results(
            path,
            expected_variant="NLI-base",
            development_projection_sha256="e" * 64,
            checkpoint_sha256_by_seed={13: "b" * 64, 42: "c" * 64},
        )
        == ()
    )
    with pytest.raises(ValueError, match="different checkpoints"):
        lean_cli._read_development_results(
            path,
            expected_variant="NLI-base",
            development_projection_sha256="e" * 64,
            checkpoint_sha256_by_seed={13: "b" * 64, 42: "d" * 64},
        )


def test_nli_pair_requires_a_matching_base_stop_before_calibration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lean_cli, "_load_lean_config", lambda path: object())
    with pytest.raises(ValueError, match="prior NLI-base STOP"):
        calibrate_selector_lean(
            config_path=tmp_path / "config.toml",
            variant="NLI-pair",
            seed13_fit_directory=tmp_path / "seed13",
            seed42_fit_directory=tmp_path / "seed42",
            development_results_path=tmp_path / "development.json",
            code_commit="a" * 40,
            final_input_sha256="d" * 64,
            development_projection_sha256="e" * 64,
            generator_sha256="f" * 64,
            output_policy_path=tmp_path / "policy.json",
        )


def test_cli_has_only_three_stages_and_final_has_no_threshold_override() -> None:
    parser = lean_cli._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == {"fit", "calibrate", "final-evaluate"}
    final_destinations = {action.dest for action in subparsers.choices["final-evaluate"]._actions}
    assert "quantile" not in final_destinations
    assert "cap" not in final_destinations
