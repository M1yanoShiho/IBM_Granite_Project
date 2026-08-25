# Experiment 04 Goal 3 prompt-budget repair

**State:** `PASS / READY FOR FRESH FULL RERUN`

**Held-out content, answers, or support labels inspected:** no  
**Scorer-only sidecar used:** no

## Invalid first attempt

The first formal attempt (`attempt-20260822a`) exposed a generation-only runtime defect. Exact
chat-template token counts showed that every observed `generation_valueerror` corresponded to a
prompt exceeding the frozen `2,304` input-token budget, with no error without an overflow and no
overflow without an error. This diagnosis used only counts, states, IDs, and aggregate token
statistics; it did not print or inspect held-out text, answers, or labels.

HotpotQA job `18673236` and MuSiQue job `18673237` were cancelled before freeze. RGB job
`18673238` completed all seven generation arms, correctly marked itself
`INVALID_REQUIRES_FULL_RERUN`, and exited before scoring. No output from this attempt is eligible
for Table 1, and no cross-job or cross-attempt mixing is allowed.

## Frozen repair

The input limit remains `2,304`; model settings, retriever/reranker/selector weights, and scoring
are unchanged. Before generation, every arm now applies the same exact tokenizer/chat-template
count and retains the rank-ordered maximal prefix of complete evidence units that fits. It never
skips a middle unit or truncates the text of an evidence unit. The output records the actual
budgeted source and unit IDs, so downstream selection and citation scoring use the evidence that
was genuinely presented to the generator.

## Validation

- Local and BluePebble targeted suites: `19 passed` on each environment.
- Goal 3-targeted Ruff: `PASS`; `mypy src`: `PASS` across 137 source files.
- Local full regression suite, excluding the already recorded stale Experiment 03 state
  assertion: `PASS`.
- Repository-wide Ruff still reports 26 pre-existing unrelated Experiment 03/script findings;
  no unrelated user work was changed.
- A sealed, gold-free audit covered all `7,700` arm-query contexts. It shortened 602 contexts,
  left zero prompts over budget, and introduced no empty contexts in non-Provence arms. Maximum
  post-repair exact chat prompt sizes were 2,303 (HotpotQA), 2,300 (MuSiQue), and 2,303 (RGB).

The repair therefore passes its implementation and isolation gate. Goal 3 itself remains active
until fresh full seven-arm bundles complete, freeze, score, produce Table 1 and paired RAR CIs,
and pass the final `7,700`-row audit.

Machine-readable evidence: `artifacts/goal3_prompt_budget_repair_manifest.json`.
