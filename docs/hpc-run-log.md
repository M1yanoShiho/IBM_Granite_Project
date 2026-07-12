# HPC Run Ledger

One entry per HPC run (or tightly-coupled run group). **PURPOSE section is written BEFORE
submitting** (pre-registration); RESULT section after pulling the data back. Raw per-query
CSVs/dumps are committed with `git add -f results/...` — a run whose raw data is not in git
is not done. Format and workflow: `.claude/skills/running-hpc-experiments/SKILL.md`.

Entry template:

```
## <date> — <short name>  [PENDING|DONE|INVALID]
- Purpose / hypothesis: <what question, expected metric + direction>
- Command: <exact sbatch line(s)>
- Code state: <git commit hash the job runs>
- Job id(s): <filled at submit>
- Raw data: <files + where committed>
- Result: <headline numbers + significance; link to results-summary finding>
```

---

## 2026-07-05 — Corroboration α-sweep certification (nq300cert) [DONE — backfilled example]

- Purpose / hypothesis: does ANY relevance/corroboration blend α beat pure q2d
  needle-found@10 on the frozen 300q task? Expected: yes, small positive at mid α.
- Command: `sbatch scripts/run_tune_corroboration.slurm results/niah_nq300_frozen.json nq300cert 0.1`
- Job id: 18025180
- Raw data: α-curve + per-query CSVs committed; runs dump on HPC
  (`results/` on bp1); nested-CV re-analysis CSVs
  (`results/corroboration_nested_cv_per_query.csv`, `_mrr.csv`) committed.
- Result: α\*=0.6, q2d_corroborate 0.610 vs q2d 0.570, +0.040 (p=0.022); later
  de-caveated by nested 5-fold CV → +0.037 out-of-fold (p=0.026/0.036) =
  results-summary **finding 15** (and Table 4d / finding 18 via `eval/compare_rules.py`).

## 2026-07-12 — WS-0 item 3: full-ranking + answers re-dump (nq300) [PENDING — submit when bp1 returns]

- Purpose / hypothesis: materialise the WS-0 raw material — dense + q2d FULL top-100
  rankings (`run_niah --dump-runs`, new in d0d18af) and a corroboration dump WITH raw
  answer strings (`tune_corroboration --dump-runs`, answers persist since edd54da) on the
  frozen 300q task. Consistency check expected: dense needle-found@10 ≈ 0.493, q2d ≈ 0.563
  (no metric change — this run only adds dumped artifacts). Unblocks WS-2/3/4/5 offline
  analyses + the V1 any-gold re-score.
- Command (NOTE: `scripts/run_niah.slurm` needs a one-line `--dump-runs` passthrough on its
  python line first — verify args against the script's usage header before submitting):
  - `sbatch scripts/run_niah.slurm results/niah_nq300_frozen.json nq300dump 10 granite_dense q2d_granite`
  - `sbatch scripts/run_tune_corroboration.slurm results/niah_nq300_frozen.json nq300ansdump 0.1`
- Code state: d0d18af (item 2) + edd54da (item 1).
- Job id(s): —
- Raw data (expected): `results/niah_nq300dump_runs_granite_dense.json`,
  `..._q2d_granite.json`, `results/corroboration_runs_nq300ansdump.json` → `git add -f`.
- Result: —
