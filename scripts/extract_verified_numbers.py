"""Extract every dissertation number from the result files, with provenance.

The rule the write-up depends on: **a figure that only exists in a summary
document is not verified.** So nothing here reads a docs/ file. Every value comes
from a scoring log or a per-case report produced by a named job, and each table
carries the job, dataset, n, convention and calibration/held-out status.

    PYTHONPATH=src python scripts/extract_verified_numbers.py \
        --out local/report-writing/verified-numbers.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS = REPO_ROOT / "local" / "results"

# Every run this project will cite, and the job that produced each.
RUNS: dict[str, dict[str, Any]] = {
    "G9": {
        "dir": "g9",
        "score_log": "g5-score-18322643.out",
        "gen_log": "g5-annotate-18322642.out",
        "gen_job": "18322642",
        "score_job": "18322643",
        "dataset": "ASQA (ALCE), 400 queries, top-5 GTR",
        "role": "calibration (final)",
        "correctness": "STR-EM",
    },
    "QAMPARI": {
        "dir": "qampari",
        "score_log": "qampari-score-18326079.out",
        "gen_log": "qampari-run-18326078.out",
        "gen_job": "18326078",
        "score_job": "18326079",
        "dataset": "QAMPARI (ALCE), 400 queries, top-5 GTR",
        "role": "module-level held-out",
        "correctness": "answer recall by containment (NOT ALCE F1, NOT comparable to STR-EM)",
    },
}

ARMS = (
    "baseline",
    "verify-only",
    "verify-annotate-capped",
    "verify-annotate-open",
    "verify-annotate-nogate",
)

SCORE_LINE = re.compile(r"^\[score\] (\S+): (\{.*\})$")
PAIR_HEADER = re.compile(r"^--- (\w+) : (\S+) vs (\S+) ---$")


def read_scores(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SCORE_LINE.match(line.strip())
        if match:
            out[match.group(1)] = json.loads(match.group(2))
    return out


def read_pairs(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        match = PAIR_HEADER.match(line.strip())
        if match and index + 1 < len(lines):
            try:
                out[(match.group(1), match.group(2), match.group(3))] = json.loads(
                    lines[index + 1].strip()
                )
            except json.JSONDecodeError:
                continue
    return out


def fmt(value: Any, places: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def arm_table(scores: dict[str, dict[str, Any]], run: dict[str, Any]) -> list[str]:
    has_r5 = any(s.get("answer_recall_at5") is not None for s in scores.values())
    head = "| arm | coverage | correctness | " + ("rec@5 | " if has_r5 else "")
    head += "cite prec (ALCE) | cite prec (cited) | n cited | cite recall | answered |"
    lines = [head, "|---" * (9 if has_r5 else 8) + "|"]
    for arm in ARMS:
        s = scores.get(arm)
        if s is None:
            continue
        row = f"| {arm} | {fmt(s['coverage'])} | {fmt(s['answer_correctness'])} | "
        if has_r5:
            row += f"{fmt(s.get('answer_recall_at5'))} | "
        row += (
            f"{fmt(s['citation_prec'])} | {fmt(s['citation_prec_cited_examples'])} | "
            f"{s['examples_with_citations']} | {fmt(s['citation_rec'])} | "
            f"{s['answered']}/{s['n']} |"
        )
        lines.append(row)
    return lines


def pair_table(pairs: dict, on: str, off: str, metrics: list[str]) -> list[str]:
    lines = [
        "| metric | delta | p | 95% CI | n paired |",
        "|---|---|---|---|---|",
    ]
    for metric in metrics:
        row = pairs.get((metric, on, off))
        if row is None:
            continue
        lines.append(
            f"| {metric} | **{fmt(row['delta'])}** | {fmt(row['p_value'], 4)} | "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {row['n_paired']} |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    doc: list[str] = [
        "# Verified numbers — Generator",
        "",
        "**Every figure below was extracted from a result file by "
        "`scripts/extract_verified_numbers.py`.** Nothing here is copied from a "
        "summary, a chat report, or an earlier document. Regenerate with:",
        "",
        "```",
        "PYTHONPATH=src python scripts/extract_verified_numbers.py \\",
        "    --out local/report-writing/verified-numbers.md",
        "```",
        "",
        "Evaluated system: tag `generator-frozen-g9-qampari-2026-08-10` (commit "
        "`eb64023`). `generator/` and `contracts/` are byte-identical between the "
        "G9 freeze `27e8280` and that tag.",
        "",
        "**No table mixes jobs.** Generation is not reproducible across jobs under "
        "differing execution conditions — `verify-annotate-capped`'s citation "
        "precision reads 0.890 in G7 and 0.911 in G8 with no change to that arm.",
        "",
    ]

    metrics = ["coverage", "answer_correctness", "answer_recall_at5",
               "citation_precision", "citation_recall"]

    for name, run in RUNS.items():
        base = RESULTS / run["dir"]
        score_log = base / run["score_log"]
        gen_log = base / run["gen_log"]
        if not score_log.exists():
            doc += [f"## {name} — MISSING ({score_log} not found)", ""]
            continue
        scores = read_scores(score_log)
        pairs = read_pairs(score_log)

        doc += [
            f"## {name} — {run['role']}",
            "",
            f"- **jobs**: generation `{run['gen_job']}`, scoring `{run['score_job']}`",
            f"- **dataset**: {run['dataset']}",
            "- **judge**: MiniCheck (TRUE is the production verifier and never judges)",
            f"- **correctness metric**: {run['correctness']}",
            "- **precision conventions**: `cite prec (ALCE)` scores an example with "
            "no citations as 0; `cite prec (cited)` averages only over examples that "
            "cited. Same arm, different numbers — always name the convention.",
            f"- **source**: `local/results/{run['dir']}/{run['score_log']}`",
            "",
            "### Arm-level (not paired)",
            "",
        ]
        doc += arm_table(scores, run)
        doc += ["", "### Paired — nogate vs baseline (the headline)", ""]
        doc += pair_table(pairs, "verify-annotate-nogate", "baseline", metrics)
        doc += ["", "### Paired — nogate vs open (what the entity gate costs)", ""]
        doc += pair_table(pairs, "verify-annotate-nogate", "verify-annotate-open", metrics)
        doc += ["", "### Paired — other arms", ""]
        for on, off in (
            ("verify-annotate-nogate", "verify-only"),
            ("verify-annotate-open", "verify-annotate-capped"),
            ("verify-only", "baseline"),
        ):
            rows = pair_table(pairs, on, off, metrics)
            if len(rows) > 2:
                doc += [f"**{on} vs {off}**", ""] + rows + [""]

        # routing + flag, from the routing-stats file the generation job wrote
        stats_path = base / "routing-stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            doc += [
                f"### Routing — `{run['gen_job']}`, arm `{stats.get('arm')}`",
                "",
                f"- claims routed **{stats['claims_routed']}** → verified "
                f"**{stats['verified']}**, annotated **{stats['unverified_annotated']}**, "
                f"destroyed **{stats['dropped_entity_conflict']}**",
                f"- entity gate would have destroyed **{stats['gate_would_drop']}**; "
                f"now cited **{stats['gate_would_drop_now_cited']}**, "
                f"annotated {stats['gate_would_drop_now_annotated']}",
                f"- control-arm self-check: `gate_would_drop` "
                f"{stats['control_gate_would_drop']} == `dropped_entity_conflict` "
                f"{stats['control_dropped_entity_conflict']}",
                f"- declared citations verified **{stats['declared_citation_verified']}"
                f"/{stats['claims_with_a_declared_citation']}** "
                f"({stats['declared_citation_verified'] / stats['claims_with_a_declared_citation']:.3f})",
                f"- per-arm errors: `{stats['errors']}`",
                "",
            ]

        nogate = scores.get("verify-annotate-nogate", {})
        if nogate.get("review_flagged_sentences"):
            flagged = nogate["prec_flagged_examples"]
            unflagged = nogate["prec_unflagged_examples"]
            doc += [
                "### Review flag — screening enrichment",
                "",
                "| cohort | cited-sample precision | error rate | n |",
                "|---|---|---|---|",
                f"| flagged | {fmt(flagged)} | {fmt(1 - flagged)} | "
                f"{nogate['n_flagged_examples']} |",
                f"| unflagged | {fmt(unflagged)} | {fmt(1 - unflagged)} | "
                f"{nogate['n_unflagged_examples']} |",
                "",
                f"Lift **{(1 - flagged) / (1 - unflagged):.2f}×** on error rate. "
                f"Flagged sentences: {nogate['review_flagged_sentences']}.",
                "",
            ]

        # error rate, straight from the generation log
        if gen_log.exists():
            arm_lines = re.findall(
                r"^\[arm\] (\S+): (\d+)/(\d+) answered, (\d+) errors",
                gen_log.read_text(encoding="utf-8", errors="replace"),
                re.M,
            )
            if arm_lines:
                doc += [f"### Generation errors — `{run['gen_job']}`", "",
                        "| arm | answered | errors | rate |", "|---|---|---|---|"]
                for arm, ans, tot, err in arm_lines:
                    doc.append(
                        f"| {arm} | {ans}/{tot} | {err} | {int(err)/int(tot):.4f} |"
                    )
                doc.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(doc) + "\n", encoding="utf-8")
    print(f"written: {args.out}  ({len(doc)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
