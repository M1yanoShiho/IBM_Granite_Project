"""G010 simulation-based power/scope sensitivity from frozen G230 rows.

This command uses only archived G230 paired discordance and frozen sample sizes.
It reports minimum detectable effects for plausible and observed discordance
rates.  It deliberately does not compute observed power, does not inspect system
held-out content, and does not change any gate or margin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

ROUTE = Path("docs/full-flow/experiments/03_generator_grounding_repair_2026-08-18")
PREVIOUS_ROUTE = Path("docs/full-flow/experiments/02_generator_selector_alignment_2026-08-15")
SCHEMA_PREFIX = "full-flow-g010"
ALPHA = 0.05
TARGET_POWER = 0.80
GENERIC_DISCORDANCE_RATES = (0.05, 0.10, 0.15, 0.20)
EFFECT_GRID = (0.01, 0.02, 0.03, 0.05)
CONFIGS = ("GN", "GC13", "GC42", "GC73", "GM13", "GM42", "GM73")
BINARY_METRICS = ("answer_match", "citation_all_supported", "correct_and_cited")


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _git(repo: Path, args: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_pin(repo: Path, relative: Path) -> dict[str, Any]:
    path = repo / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing G010 source file: {path}")
    return {
        "path": str(path.relative_to(repo)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


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
    if not 0.0 < discordance_rate <= 1.0:
        raise ValueError("discordance_rate must be in (0, 1]")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be in (0, 1)")
    if design_effect < 1.0:
        raise ValueError("design_effect must be at least 1")
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    return math.sqrt(design_effect * (z_alpha + z_power) ** 2 * discordance_rate / n)


def required_paired_n(
    *,
    effect: float,
    discordance_rate: float,
    alpha: float = ALPHA,
    power: float = TARGET_POWER,
    design_effect: float = 1.0,
) -> int:
    if not 0.0 < effect < 1.0:
        raise ValueError("effect must be in (0, 1)")
    mde = minimum_detectable_effect(
        n=1,
        discordance_rate=discordance_rate,
        alpha=alpha,
        power=power,
        design_effect=design_effect,
    )
    return math.ceil((mde / effect) ** 2)


def _citation_supported(config: Mapping[str, Any]) -> float:
    precision = config.get("citation_precision")
    recall = config.get("citation_recall")
    return float(precision == 1.0 and recall == 1.0)


def _joined_binary_rows(
    answer_rows: Sequence[Mapping[str, Any]],
    citation_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    citations = {str(row["task_id"]): row for row in citation_rows}
    if set(citations) != {str(row["task_id"]) for row in answer_rows}:
        raise ValueError("G230 answer and citation case task IDs differ")
    joined: list[dict[str, Any]] = []
    for answer_row in answer_rows:
        task_id = str(answer_row["task_id"])
        citation_row = citations[task_id]
        if answer_row["component_id"] != citation_row["component_id"]:
            raise ValueError(f"component mismatch for task {task_id}")
        configs: dict[str, Any] = {}
        for name, answer_config in answer_row["configs"].items():
            citation_config = citation_row["configs"][name]
            answer_match = float(answer_config["answer_match"])
            citation_supported = _citation_supported(citation_config)
            configs[name] = {
                "answer_match": answer_match,
                "citation_all_supported": citation_supported,
                "correct_and_cited": float(answer_match == 1.0 and citation_supported == 1.0),
            }
        joined.append(
            {
                "task_id": task_id,
                "scope": answer_row["scope"],
                "context": answer_row["context"],
                "query_id": answer_row["query_id"],
                "component_id": answer_row["component_id"],
                "configs": configs,
            }
        )
    return joined


def _discordance(
    rows: Sequence[Mapping[str, Any]], *, config: str, baseline: str, metric: str
) -> dict[str, Any]:
    before_positive = after_positive = wrong_to_right = right_to_wrong = 0
    for row in rows:
        before = float(row["configs"][baseline][metric])
        after = float(row["configs"][config][metric])
        before_positive += int(before == 1.0)
        after_positive += int(after == 1.0)
        wrong_to_right += int(before == 0.0 and after == 1.0)
        right_to_wrong += int(before == 1.0 and after == 0.0)
    n = len(rows)
    discordant = wrong_to_right + right_to_wrong
    return {
        "n": n,
        "baseline_positive": before_positive,
        "candidate_positive": after_positive,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "paired_delta": (after_positive - before_positive) / n,
        "discordant": discordant,
        "discordance_rate": discordant / n,
    }


def _context_rows(rows: Sequence[Mapping[str, Any]], context: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["context"] == context]


def build_report(repo: Path) -> tuple[dict[str, Any], str]:
    route = repo / ROUTE
    g000_manifest = _json(route / "artifacts/G000/G000_INPUT_MANIFEST.json")
    if g000_manifest.get("status") != "PASS":
        raise ValueError("G010 requires G000 PASS manifest")
    answer_path = PREVIOUS_ROUTE / "artifacts/G230/score/scored_cases.jsonl"
    citation_path = PREVIOUS_ROUTE / "artifacts/G230/citation/citation_cases.jsonl"
    joined = _joined_binary_rows(_jsonl(repo / answer_path), _jsonl(repo / citation_path))
    full_rows = _context_rows(joined, "K_topk")
    components = {str(row["component_id"]) for row in full_rows}
    if len(full_rows) != 739:
        raise ValueError("G010 expected the frozen 739 full TopK rows")
    design_effect = len(full_rows) / len(components)

    discordance: dict[str, Any] = {}
    for config in CONFIGS:
        discordance[config] = {
            metric: _discordance(full_rows, config=config, baseline="G0", metric=metric)
            for metric in BINARY_METRICS
        }

    by_family: dict[str, Any] = {}
    for family in ("GC", "GM"):
        family_configs = [name for name in CONFIGS if name.startswith(family)]
        by_family[family] = {}
        for metric in BINARY_METRICS:
            rates = [
                discordance[name][metric]["discordance_rate"]
                for name in family_configs
            ]
            by_family[family][metric] = {
                "seed_configs": family_configs,
                "min_discordance_rate": min(rates),
                "mean_discordance_rate": sum(rates) / len(rates),
                "max_discordance_rate": max(rates),
                "mde_at_mean_discordance": minimum_detectable_effect(
                    n=len(full_rows),
                    discordance_rate=max(sum(rates) / len(rates), 1 / len(full_rows)),
                    design_effect=design_effect,
                ),
            }

    sample_sizes = {
        "niah_g230_full_topk": len(full_rows),
        "niah_g230_components": len(components),
        "twowiki_dev_planned": 2000,
        "hotpotqa_heldout_reserved": g000_manifest["data_scope"]["heldout"]["datasets"][
            "hotpotqa"
        ]["count"],
        "musique_full_heldout_records_reserved": g000_manifest["data_scope"]["heldout"][
            "datasets"
        ]["musique-full"]["count"],
        "rgb_heldout_reserved": g000_manifest["data_scope"]["heldout"]["datasets"]["rgb"][
            "count"
        ],
    }
    generic_mde = {
        name: {
            f"discordance_{rate:.2f}": minimum_detectable_effect(
                n=n,
                discordance_rate=rate,
                design_effect=design_effect,
            )
            for rate in GENERIC_DISCORDANCE_RATES
        }
        for name, n in sample_sizes.items()
        if name != "niah_g230_components"
    }
    required_n = {
        f"discordance_{rate:.2f}": {
            f"effect_{effect:.2f}": required_paired_n(
                effect=effect,
                discordance_rate=rate,
                design_effect=design_effect,
            )
            for effect in EFFECT_GRID
        }
        for rate in GENERIC_DISCORDANCE_RATES
    }
    contexts = Counter(str(row["context"]) for row in joined)
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-power-scope-v1",
        "status": "PASS",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "head": _git(repo, ["rev-parse", "HEAD"]),
            "branch": _git(repo, ["branch", "--show-current"]),
        },
        "method": {
            "calculation": "simulation-style paired-binary sensitivity using G230 discordance and fixed n",
            "observed_power_computed": False,
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "design_effect_source": "G230 full TopK query/component ratio",
            "design_effect": design_effect,
            "final_inference_remains": "exact McNemar + paired component-cluster bootstrap",
            "gate_or_margin_changed": False,
        },
        "source_files": {
            "g000_manifest": _source_pin(repo, ROUTE / "artifacts/G000/G000_INPUT_MANIFEST.json"),
            "g000_denylist": _source_pin(repo, ROUTE / "artifacts/G000/G000_DENYLIST.json"),
            "g230_scored_cases": _source_pin(repo, answer_path),
            "g230_citation_cases": _source_pin(repo, citation_path),
            "g010_script": _source_pin(repo, Path("scripts/full_flow_g010_power_scope.py")),
        },
        "sample_sizes": sample_sizes,
        "contexts": dict(sorted(contexts.items())),
        "g230_full_topk_discordance": discordance,
        "family_sensitivity_from_g230": by_family,
        "generic_minimum_detectable_effect": generic_mde,
        "required_n_sensitivity": required_n,
        "interpretation": {
            "small_effects": "2pp effects are underpowered for 300-400 query held-out sets unless discordance is very low",
            "null_ci": "CI crossing 0 means insufficient evidence for small effects, not equivalence",
            "allowed_next": "G100 citation attribution may proceed; training remains blocked by G110/G210/G300 sequence",
        },
    }
    return manifest, render_markdown(manifest)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# G010 Power/Scope Sensitivity",
        "",
        "**日期：** 2026-08-18",
        f"**状态：** `{report['status']}`",
        "",
        "## Scope",
        "",
        f"- G230 full TopK rows: `{report['sample_sizes']['niah_g230_full_topk']}`",
        f"- G230 components: `{report['sample_sizes']['niah_g230_components']}`",
        f"- Design effect: `{report['method']['design_effect']:.4f}`",
        "- Observed power was not computed.",
        "",
        "## Mean G230 Discordance MDE",
        "",
        "| Family | Metric | Mean discordance | MDE |",
        "|---|---|---:|---:|",
    ]
    for family, metrics in report["family_sensitivity_from_g230"].items():
        for metric, item in metrics.items():
            lines.append(
                f"| {family} | {metric} | {_pct(item['mean_discordance_rate'])} | "
                f"{_pct(item['mde_at_mean_discordance'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- G010 does not relax any gate or margin.",
            "- A null or wide CI in later stages must be reported as insufficient evidence for small effects, not equivalence.",
            "- Next allowed work is G100 citation attribution; training remains blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    out_dir = args.out_dir or (repo / ROUTE / "artifacts/G010")
    report, markdown = build_report(repo)
    _write_json(out_dir / "G010_POWER_SCOPE.json", report)
    _write_text(repo / ROUTE / "G010_POWER_SCOPE.md", markdown)
    print(f"written {out_dir / 'G010_POWER_SCOPE.json'}")
    print(f"written {repo / ROUTE / 'G010_POWER_SCOPE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
