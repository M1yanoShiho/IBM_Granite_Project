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

## 2026-07-12 — WS-0 item 3: full-ranking + answers re-dump (nq300) [SUPERSEDED]

Superseded by the Validity-suite entry below: `scripts/run_validity_suite.slurm` produces
the same two dumps (run_niah `--dump-runs` + tune_corroboration answers dump) AND runs the
V1/V2/V3 analyses on them in the same job — one submit instead of two, no separate
`run_niah.slurm` passthrough edit needed.

## 2026-07-12 — [superseded original entry kept for the record]

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

## 2026-07-12 — Validity suite V1+V2+V3 (nq300v) [PENDING — submit when bp1 returns]

- Purpose / hypothesis (pre-registered, all outcomes reported):
  - **Consistency gates:** dense needle-found@10 ≈ 0.493, q2d ≈ 0.563, α\*=0.6 blend
    ≈ 0.61 must reproduce (same frozen task, cached index) — deviations invalidate the run.
  - **V1 any-gold:** absolute levels rise (other golds also surface); the story SURVIVES
    if the q2d-vs-dense and blend-vs-q2d deltas keep sign (and ideally significance)
    under the any-gold rule. If the deltas vanish, the metric critique was substantive —
    report that and reframe.
  - **V2 injection removal:** splits buried needles into injection-caused vs natural.
    Either outcome is reportable: injection-dominated = the constructed task is doing the
    stress-testing (by design); natural-dominated = the pathology exists in the wild and
    the motivation strengthens.
  - **V3 natural conflict rate:** expected > 0 (literature: version drift / ambiguity /
    misinfo occur naturally); point estimate is the deliverable. Caveat to carry into the
    report: natural pool = injected-corpus top-20 minus injections (a prefix of the true
    natural top-20) → slightly conservative.
- Command: `mkdir -p logs results && sbatch scripts/run_validity_suite.slurm`
  (defaults: task `results/niah_nq300_frozen.json`, tag `nq300v`)
- Code state: 309a779 (V1-V3 modules + slurm) on top of d0d18af/edd54da (WS-0 dumps).
- Job id(s): —
- Raw data (expected): `results/niah_nq300v_runs_granite_dense.json`,
  `results/niah_nq300v_runs_q2d_granite.json`, `results/corroboration_runs_nq300v.json`
  (first dump WITH raw answer strings), `results/anygold_nq300v_*.csv`,
  `results/injection_nq300v_*.csv`, curve/perq CSVs → `git add -f results/*nq300v*`.
- Result: —

## 2026-07-12 — V4 RAMDocs external validation (frozen α=0.6) [PENDING — submit when bp1 returns]

- Purpose / hypothesis (pre-registered): does the certified corroboration blend help on a
  conflicting-evidence benchmark WE did not construct? α is FROZEN at 0.6 — no tuning on
  RAMDocs, whatever the outcome. Expected direction: blend ≥ relevance-only on
  correct-doc@1 and MRR-of-first-correct, blend ≤ relevance on misinfo@1. If null/negative:
  corroboration is honestly re-scoped as "effective in our constructed regime; did not
  transfer to RAMDocs' ambiguity-heavy mix" — still a reportable finding.
- One-time login-node prep (compute nodes offline):
  `mkdir -p /user/work/$USER/ramdocs && wget https://raw.githubusercontent.com/HanNight/RAMDocs/main/RAMDocs_test.jsonl -O /user/work/$USER/ramdocs/RAMDocs_test.jsonl`
- Command: smoke first, then full —
  `sbatch scripts/run_ramdocs_corroboration.slurm /user/work/$USER/ramdocs/RAMDocs_test.jsonl smoke 0.6 20`
  `sbatch scripts/run_ramdocs_corroboration.slurm`
- Code state: 4df6bdb.
- Job id(s): —
- Raw data (expected): `results/ramdocs_ramdocs_{correct1,correct3,mrr_correct,misinfo1}.csv`
  (+ `ramdocs_smoke_*` from the smoke) → `git add -f results/ramdocs_*`.
- Result: —
