"""Read R20's two repeats of R11's decompose arm against the pre-registered criterion.

R11 concluded that the decomposition gain on NQ total recall replicates 4/4 across corpus
sizes (+0.0105 to +0.0145, every point significant). The value of that entry is precisely
that it *rules out a fluke* -- and it was measured once. The decompose arm builds its
sub-queries with `GraniteLLMClient` on the GPU; R11's pre-registration waved LLM variance
away on the grounds that greedy decoding yields identical sub-queries, but greedy is
deterministic in arithmetic only, not in GPU reduction order, and R7->R8 measured the
resulting drift on this very arm at -0.0031 MRR -- the same order as the effect's leading
digit.

`retriever_significance.sh` cannot do this reading: it hardcodes `runs/retr-<ds>-<arm>/`,
while repeat 1 has been archived elsewhere so repeat 2 could reuse the canonical path. It
also sweeps 17 pairs and four metrics when the pre-registered quantity is one pair and one
metric. So this script targets that pair directly, and shells out to the same
`paired_metric_cli` the other script uses rather than reimplementing the statistics.

The criterion, written before the repeats were read:

  * every size significant and positive in BOTH repeats -> R11's "4/4" stands and finally
    has a variance behind it; R12, which is a re-read of R11's artifacts, is carried with it;
  * any size losing significance in either repeat -> R11's "4/4" must be rewritten to the
    count actually achieved, and R12's three rows are downgraded with it.

Only the two new repeats are compared against each other. R11's 2026-08-13 numbers ran at a
different source tree and the difference against them mixes in code drift; see R20.

    python scripts/r20_repeat_readout.py --rep1-root runs/_r20-rep1 --rep2-root runs
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

METRIC = "retriever.core.document_recall"
ALPHA = 0.05
# (label, decompose dir suffix, strong-bm25 dir suffix). The 100k point is the original
# R11 arm and carries no size tag, which is why this is a table rather than a format string.
POINTS = [
    ("25k", "retr-nq25k-decompose-orig", "retr-nq25k-strong-bm25"),
    ("50k", "retr-nq50k-decompose-orig", "retr-nq50k-strong-bm25"),
    ("100k", "retr-nq-decompose-orig", "retr-nq-strong-bm25"),
    ("200k", "retr-nq200k-decompose-orig", "retr-nq200k-strong-bm25"),
]


def _signature(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return manifest["source_tree_signature"]


def _paired(on_report: Path, off_report: Path) -> dict:
    result = subprocess.run(
        [
            sys.executable, "-m", "evidence_rag.evaluation.paired_metric_cli",
            "--on-report", str(on_report),
            "--off-report", str(off_report),
            "--metric", METRIC,
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep1-root", required=True, type=Path)
    parser.add_argument("--rep2-root", required=True, type=Path)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="where the strong-bm25 arms live; defaults to --rep2-root. That arm is pure-CPU "
             "and bit-reproducible (R7xR8 matched it to the digit), so it is not repeated.",
    )
    arguments = parser.parse_args(argv)
    baseline_root = arguments.baseline_root or arguments.rep2_root

    # The self-check comes first: if the two repeats ran at different source trees they are
    # not the same experiment and no number below means anything (R20, echoing R18's A).
    signatures: dict[str, set[str]] = {"rep1": set(), "rep2": set()}
    for label, decompose, _ in POINTS:
        signatures["rep1"].add(_signature(arguments.rep1_root / decompose))
        signatures["rep2"].add(_signature(arguments.rep2_root / decompose))
    every = signatures["rep1"] | signatures["rep2"]
    print("== source_tree_signature self-check")
    for name, found in signatures.items():
        print(f"  {name}: " + ", ".join(sorted(s[:16] for s in found)))
    if len(every) != 1:
        print("\nSTOP: the repeats did not run at the same source tree. Per R20 the two rounds "
              "are not the same experiment; every number below would be code drift mixed with "
              "run-to-run variance. Investigate before reading.")
        return 1
    print(f"  identical across both repeats: {next(iter(every))[:16]}\n")

    print(f"== delta {METRIC} (decompose-orig - strong-bm25), by corpus size")
    print(f"{'size':>6} {'rep':>4} {'delta':>9} {'p':>8} {'95% CI':>22} {'n':>6}")
    outcomes = []
    for label, decompose, baseline in POINTS:
        off_report = baseline_root / baseline / "retriever_report.json"
        deltas = []
        for rep, root in (("1", arguments.rep1_root), ("2", arguments.rep2_root)):
            result = _paired(root / decompose / "retriever_report.json", off_report)
            deltas.append(result["delta"])
            significant = result["p_value"] < ALPHA and result["delta"] > 0
            outcomes.append((label, rep, significant))
            ci = f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]"
            print(f"{label:>6} {rep:>4} {result['delta']:>+9.4f} {result['p_value']:>8.4f} "
                  f"{ci:>22} {result['n_paired']:>6}"
                  + ("" if significant else "   <- NOT significant-positive"))
        print(f"{'':>6} {'spread':>4} {abs(deltas[0] - deltas[1]):>9.4f}"
              "   <- run-to-run, at fixed source tree")
    print()

    held = [o for o in outcomes if o[2]]
    print(f"== verdict: {len(held)}/{len(outcomes)} arm-repeats significant and positive")
    if len(held) == len(outcomes):
        print("R11's 4/4 replication stands and now has a variance behind it: every corpus "
              "size clears significance in both repeats, so the 'rules out a fluke' role that "
              "entry plays is finally closed. R12, a re-read of these artifacts, is carried "
              "with it.")
    else:
        lost = ", ".join(f"{label}(rep{rep})" for label, rep, ok in outcomes if not ok)
        print(f"Per R20's criterion R11 must be rewritten: {lost} did not clear "
              "significance. The entry may no longer claim 4/4, and R12's three rows are "
              "downgraded with it -- R12 is a re-read of these same artifacts and inherits "
              "the variance rather than having its own.")
    print("\nReminder (R20, 'the reading is pinned'): compare the two repeats against each "
          "other only. A gap against R11's 2026-08-13 values would mix in code drift and is "
          "a separate finding, not a failure of this entry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
