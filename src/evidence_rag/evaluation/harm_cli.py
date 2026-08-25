"""CLI: offline harmful-in-context comparison of two selector runs (spec §7)."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet, SelectedEvidenceSet
from evidence_rag.evaluation.harm import (
    compare_harm,
    counterfactual_pool_hit_rate,
    evaluate_selector_harm,
    provenance_harm_map,
)
from evidence_rag.materializer.provenance import read_provenance


def _read_selected(path: Path) -> tuple[SelectedEvidenceSet, ...]:
    text = Path(path).read_text(encoding="utf-8")
    return tuple(
        SelectedEvidenceSet.model_validate_json(line)
        for line in text.splitlines()
        if line.strip()
    )


def _read_candidates(path: Path) -> tuple[CandidateSet, ...]:
    text = Path(path).read_text(encoding="utf-8")
    return tuple(
        CandidateSet.model_validate_json(line) for line in text.splitlines() if line.strip()
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline harmful-in-context gate-on/off report")
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--selected-on", required=True, type=Path)
    parser.add_argument("--selected-off", required=True, type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--dataset-signature", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--iterations", type=int, default=10000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    harm_map = provenance_harm_map(read_provenance(arguments.provenance))
    on_report = evaluate_selector_harm(
        _read_selected(arguments.selected_on),
        harm_map,
        dataset_signature=arguments.dataset_signature,
    )
    off_report = evaluate_selector_harm(
        _read_selected(arguments.selected_off),
        harm_map,
        dataset_signature=arguments.dataset_signature,
    )
    comparison = compare_harm(
        on_report, off_report, seed=arguments.seed, iterations=arguments.iterations
    )
    pool_hit = (
        counterfactual_pool_hit_rate(_read_candidates(arguments.candidates), harm_map)
        if arguments.candidates
        else None
    )
    print(
        json.dumps(
            {
                "harm_on": comparison.harm_on,
                "harm_off": comparison.harm_off,
                "delta": comparison.delta,
                "p_value": comparison.p_value,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "n_paired": comparison.n_paired,
                "pool_hit_rate": pool_hit,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
