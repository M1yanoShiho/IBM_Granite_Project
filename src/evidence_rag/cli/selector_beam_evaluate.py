"""Run Beam Selector M2/M3 development checks and M4 frozen formal evaluation."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet, Query
from evidence_rag.evaluation.selector_beam_metrics import (
    choose_threshold,
    compare_selector_arms,
)
from evidence_rag.evaluation.selector_experiment import (
    align_inputs,
    read_jsonl,
    run_selector_arm,
    sha256_file,
    validate_pool_manifest,
)
from evidence_rag.infrastructure.datasets import DatasetBundle, GoldCase, JsonlDatasetAdapter
from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.materializer.selector_beam_split import NiahSelectorAssignment
from evidence_rag.selector.beam_three_class import (
    BeamSelectorEvent,
    BeamTextScorer,
    FirstHopCachingScorer,
    ThreeClassBeamSelector,
)
from evidence_rag.selector.beam_torch import TorchBeamNetwork, TorchBeamTextScorer
from evidence_rag.selector.top_k import TopKSelector


@dataclass(frozen=True)
class DatasetArguments:
    name: str
    manifest: Path
    candidates: Path
    pool_manifest: Path
    source_parent: Path
    provenance: Path | None
    assignments: Path | None


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    bundle: DatasetBundle
    aligned: tuple[tuple[Query, CandidateSet, GoldCase], ...]
    harm_by_query: dict[str, str] | None
    candidate_sha256: str
    source_parent_sha256: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Beam Selector checkpoint")
    parser.add_argument("--stage", choices=("dev", "formal"), required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--m0-report", required=True, type=Path)
    parser.add_argument("--train-report", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir")
    parser.add_argument("--threshold-selection", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--git-root", type=Path, default=Path.cwd())
    for prefix in ("niah", "twowiki"):
        parser.add_argument(f"--{prefix}-manifest", required=True, type=Path)
        parser.add_argument(f"--{prefix}-candidates", required=True, type=Path)
        parser.add_argument(f"--{prefix}-pool-manifest", required=True, type=Path)
        parser.add_argument(f"--{prefix}-source-parent", required=True, type=Path)
    parser.add_argument("--niah-provenance", required=True, type=Path)
    parser.add_argument("--niah-assignments", type=Path)
    return parser


def _json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _section(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = value.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"missing object {name}")
    return section


def _text(value: Mapping[str, object], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{name} must be non-empty text")
    return item


def _integer(value: Mapping[str, object], name: str) -> int:
    item = value.get(name)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"{name} must be an integer")
    return item


def _number(value: Mapping[str, object], name: str) -> float:
    item = value.get(name)
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise ValueError(f"{name} must be numeric")
    return float(item)


def _number_list(value: Mapping[str, object], name: str) -> tuple[float, ...]:
    item = value.get(name)
    if not isinstance(item, list) or any(
        not isinstance(entry, (int, float)) or isinstance(entry, bool) for entry in item
    ):
        raise ValueError(f"{name} must be a numeric list")
    return tuple(float(entry) for entry in item)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assignment_ids(path: Path) -> tuple[str, ...]:
    values = tuple(
        NiahSelectorAssignment.model_validate_json(line).query_id
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(values) != len(set(values)):
        raise ValueError("NIAH assignment file contains duplicate query IDs")
    return values


def _load_dataset(
    arguments: DatasetArguments,
    *,
    expected_signature: str,
    expected_pool_sha256: str,
) -> LoadedDataset:
    pool_manifest = _json(arguments.pool_manifest)
    candidate_sha = validate_pool_manifest(
        pool_manifest,
        candidate_pool_path=arguments.candidates,
        source_parent_path=arguments.source_parent,
        expected_top_n=20,
    )
    if candidate_sha != expected_pool_sha256:
        raise ValueError(f"{arguments.name} is not the candidate pool frozen by M0")
    bundle = JsonlDatasetAdapter.load(arguments.manifest)
    if bundle.dataset_signature != expected_signature:
        raise ValueError(f"{arguments.name} dataset content differs from M0")
    aligned = list(align_inputs(bundle, read_jsonl(arguments.candidates, CandidateSet)))
    if arguments.assignments is not None:
        ids = _assignment_ids(arguments.assignments)
        by_query = {row[0].query_id: row for row in aligned}
        missing = set(ids) - set(by_query)
        if missing:
            raise ValueError(f"{arguments.name} assignments are absent from its candidate pool")
        aligned = [by_query[query_id] for query_id in ids]
    harm = (
        None
        if arguments.provenance is None
        else {
            item.query_id: item.counterfactual_document_id
            for item in read_provenance(arguments.provenance)
        }
    )
    if harm is not None:
        harm = {row[0].query_id: harm[row[0].query_id] for row in aligned}
    return LoadedDataset(
        name=arguments.name,
        bundle=bundle,
        aligned=tuple(aligned),
        harm_by_query=harm,
        candidate_sha256=candidate_sha,
        source_parent_sha256=sha256_file(arguments.source_parent),
    )


def _run_topk(
    dataset: LoadedDataset,
    *,
    output: Path,
    seed: int,
    resume: bool,
    git_root: Path,
) -> list[dict[str, object]]:
    rows, _selections, _selected = run_selector_arm(
        "top-k",
        TopKSelector(),
        dataset.aligned,
        output_directory=output,
        candidate_pool_sha256=dataset.candidate_sha256,
        source_parent_sha256=dataset.source_parent_sha256,
        max_selected=10,
        run_seed=seed,
        dataset_signature=dataset.bundle.dataset_signature,
        harm_by_query=dataset.harm_by_query,
        resume=resume,
        git_root=git_root,
    )
    return rows


def _run_beam(
    dataset: LoadedDataset,
    scorer: BeamTextScorer,
    *,
    required_threshold: float,
    reject_threshold: float,
    beam_size: int,
    output: Path,
    seed: int,
    resume: bool,
    git_root: Path,
) -> list[dict[str, object]]:
    events: dict[str, BeamSelectorEvent] = {}

    def record(event: BeamSelectorEvent) -> None:
        events[event.query_id] = event

    selector = ThreeClassBeamSelector(
        scorer,
        beam_size=beam_size,
        required_threshold=required_threshold,
        reject_threshold=reject_threshold,
        on_event=record,
    )

    def event_lookup(query_id: str) -> object | None:
        event = events.pop(query_id, None)
        return None if event is None else asdict(event)

    rows, _selections, _selected = run_selector_arm(
        "three-class-beam",
        selector,
        dataset.aligned,
        output_directory=output,
        candidate_pool_sha256=dataset.candidate_sha256,
        source_parent_sha256=dataset.source_parent_sha256,
        max_selected=10,
        run_seed=seed,
        dataset_signature=dataset.bundle.dataset_signature,
        harm_by_query=dataset.harm_by_query,
        event_lookup=event_lookup,
        resume=resume,
        git_root=git_root,
    )
    return rows


def _thresholds(path: Path) -> tuple[float, float]:
    decision = _json(path)
    selected = decision.get("selected")
    if not isinstance(selected, Mapping):
        nested = decision.get("decision")
        if isinstance(nested, Mapping):
            selected = nested.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("threshold selection contains no selected configuration")
    return _number(selected, "required_threshold"), _number(selected, "reject_threshold")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = tomllib.loads(arguments.config.read_text(encoding="utf-8"))
    model_config = _section(config, "model")
    training_config = _section(config, "training")
    selection_config = _section(config, "selection")
    evaluation_config = _section(config, "evaluation")
    raw_seeds = training_config.get("seeds")
    if not isinstance(raw_seeds, list) or arguments.seed not in raw_seeds:
        raise ValueError("seed is not frozen in the Selector config")
    if arguments.stage == "formal" and arguments.threshold_selection is None:
        raise ValueError("formal evaluation requires --threshold-selection")
    if arguments.stage == "dev" and arguments.niah_assignments is None:
        raise ValueError("development evaluation requires --niah-assignments")
    if arguments.stage == "formal" and arguments.niah_assignments is not None:
        raise ValueError("formal evaluation must use the complete sealed NIAH split")

    m0 = _json(arguments.m0_report)
    if m0.get("status") != "PASS":
        raise ValueError("evaluation requires a passing M0 report")
    if _section(m0, "model").get("config_sha256") != sha256_file(arguments.config):
        raise ValueError("Selector config differs from M0")
    train_report = _json(arguments.train_report)
    if train_report.get("status") != "PASS" or train_report.get("seed") != arguments.seed:
        raise ValueError("evaluation requires the matching completed training report")
    checkpoint_record = _section(train_report, "checkpoint")
    if checkpoint_record.get("sha256") != sha256_file(arguments.checkpoint):
        raise ValueError("checkpoint differs from its training report")

    niah_split = "dev" if arguments.stage == "dev" else "sealed"
    twowiki_split = "dev" if arguments.stage == "dev" else "heldout"
    niah_pool_name = "niah-dev" if arguments.stage == "dev" else "sealed600"
    twowiki_pool_name = "2wiki-dev" if arguments.stage == "dev" else "2wiki-heldout"
    niah_datasets = _section(_section(m0, "niah"), "datasets")
    if arguments.stage == "dev":
        assignments = _section(_section(m0, "niah"), "assignments")
        dev_assignment = _section(assignments, "dev")
        if dev_assignment.get("sha256") != sha256_file(arguments.niah_assignments):
            raise ValueError("NIAH dev assignments are not the file frozen by M0")
    twowiki_datasets = _section(m0, "twowiki")
    pools = _section(m0, "candidate_pools")

    niah = _load_dataset(
        DatasetArguments(
            name=niah_pool_name,
            manifest=arguments.niah_manifest,
            candidates=arguments.niah_candidates,
            pool_manifest=arguments.niah_pool_manifest,
            source_parent=arguments.niah_source_parent,
            provenance=arguments.niah_provenance,
            assignments=arguments.niah_assignments,
        ),
        expected_signature=_text(_section(niah_datasets, niah_split), "dataset_signature"),
        expected_pool_sha256=_text(_section(pools, niah_pool_name), "sha256"),
    )
    twowiki = _load_dataset(
        DatasetArguments(
            name=twowiki_pool_name,
            manifest=arguments.twowiki_manifest,
            candidates=arguments.twowiki_candidates,
            pool_manifest=arguments.twowiki_pool_manifest,
            source_parent=arguments.twowiki_source_parent,
            provenance=None,
            assignments=None,
        ),
        expected_signature=_text(_section(twowiki_datasets, twowiki_split), "dataset_signature"),
        expected_pool_sha256=_text(_section(pools, twowiki_pool_name), "sha256"),
    )

    network = TorchBeamNetwork(
        model_id=_text(model_config, "model_id"),
        revision=_text(model_config, "revision"),
        device=arguments.device,
        cache_dir=arguments.cache_dir,
        seed=arguments.seed,
    )
    network.load(arguments.checkpoint)
    max_length = _integer(model_config, "max_length")
    batch_size = _integer(training_config, "batch_size")
    beam_size = _integer(selection_config, "beam_size")
    stats_seed = _integer(evaluation_config, "bootstrap_seed")
    iterations = _integer(evaluation_config, "bootstrap_samples")
    scorer: BeamTextScorer = TorchBeamTextScorer(
        network, max_length=max_length, batch_size=batch_size
    )
    if arguments.threshold_selection is None:
        scorer = FirstHopCachingScorer(scorer)
    top_rows = {
        "niah": _run_topk(
            niah,
            output=arguments.output / "niah/top-k",
            seed=arguments.seed,
            resume=arguments.resume,
            git_root=arguments.git_root,
        ),
        "twowiki": _run_topk(
            twowiki,
            output=arguments.output / "twowiki/top-k",
            seed=arguments.seed,
            resume=arguments.resume,
            git_root=arguments.git_root,
        ),
    }

    if arguments.threshold_selection is None:
        threshold_pairs = tuple(
            (required, reject)
            for required in _number_list(selection_config, "required_thresholds")
            for reject in _number_list(selection_config, "reject_thresholds")
        )
    else:
        threshold_pairs = (_thresholds(arguments.threshold_selection),)

    grid: list[dict[str, object]] = []
    for required, reject in threshold_pairs:
        label = f"required-{required:.2f}_reject-{reject:.2f}"
        niah_rows = _run_beam(
            niah,
            scorer,
            required_threshold=required,
            reject_threshold=reject,
            beam_size=beam_size,
            output=arguments.output / f"niah/{label}",
            seed=arguments.seed,
            resume=arguments.resume,
            git_root=arguments.git_root,
        )
        twowiki_rows = _run_beam(
            twowiki,
            scorer,
            required_threshold=required,
            reject_threshold=reject,
            beam_size=beam_size,
            output=arguments.output / f"twowiki/{label}",
            seed=arguments.seed,
            resume=arguments.resume,
            git_root=arguments.git_root,
        )
        niah_comparison = compare_selector_arms(
            top_rows["niah"], niah_rows, seed=stats_seed, iterations=iterations
        )
        twowiki_comparison = compare_selector_arms(
            top_rows["twowiki"], twowiki_rows, seed=stats_seed, iterations=iterations
        )
        niah_paired = _section(niah_comparison, "paired")
        twowiki_paired = _section(twowiki_comparison, "paired")
        harm = _section(niah_paired, "harmful_in_context")
        niah_recall = _section(niah_paired, "required_recall")
        twowiki_recall = _section(twowiki_paired, "required_recall")
        record = {
            "required_threshold": required,
            "reject_threshold": reject,
            "harm_delta": _number(harm, "delta_beam_minus_top_k"),
            "harm_ci_high": _number(harm, "ci_high"),
            "niah_recall_loss": -_number(niah_recall, "delta_beam_minus_top_k"),
            "twowiki_recall_loss": -_number(twowiki_recall, "delta_beam_minus_top_k"),
            "niah": niah_comparison,
            "twowiki": twowiki_comparison,
        }
        grid.append(record)
        _write_json(arguments.output / f"comparisons/{label}.json", record)

    decision = (
        choose_threshold(grid)
        if len(grid) > 1
        else {
            "status": "FIXED",
            "selected": {
                key: grid[0][key]
                for key in (
                    "required_threshold",
                    "reject_threshold",
                    "harm_delta",
                    "harm_ci_high",
                    "niah_recall_loss",
                    "twowiki_recall_loss",
                )
            },
        }
    )
    report = {
        "schema_version": "1.0",
        "stage": arguments.stage,
        "seed": arguments.seed,
        "checkpoint_sha256": sha256_file(arguments.checkpoint),
        "threshold_policy": selection_config.get("threshold_policy"),
        "grid": grid,
        "decision": decision,
    }
    _write_json(arguments.output / "EVALUATION_REPORT.json", report)
    print(
        json.dumps(
            {"report": str(arguments.output / "EVALUATION_REPORT.json"), "decision": decision},
            sort_keys=True,
        )
    )
    return 2 if decision.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
