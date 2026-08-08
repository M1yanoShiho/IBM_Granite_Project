# Held-out pre-registration — Generator

**Committed before any held-out data is loaded.** Promoting `verify-annotate-nogate`
to the main method was a choice made *after* seeing calibration results. Held-out
is only a genuine test if every choice that could be tuned to it is fixed first,
so this document exists to be checked against the run that follows, not revised
by it.

Status at the time of writing: calibration is frozen at G8. No system change
follows from anything in the held-out results.

## Method under test

| | |
|---|---|
| **Main method** | `verify-annotate-nogate` — draft, verify every claim against the selected evidence, cite what entails, label what does not, destroy nothing |
| **Baseline** | generation-time citation (`GraniteGenerator`) |
| **Ablations, reported not compared as claims** | `verify-only` (the published delete filter), `verify-annotate-capped` (pre-lift abstention), `verify-annotate-open` (entity gate engaged) |

The entity layer runs observe-only. Flagged claims stay cited and carry
`[may warrant review]`; the label changes no routing decision and is stripped
before scoring.

## Operating point — frozen

| | |
|---|---|
| Verifier | `google/t5_xxl_true_nli_mixture` (TRUE), **threshold 0.50** |
| Entailment direction | premise = evidence text, hypothesis = claim text |
| Entity gate | **off** (observe-only) |
| Generator | `ibm-granite/granite-4.1-3b` |
| Retrieval / selection | as in the frozen reference config; top-k 5 |
| Seed | 13 |

No prompt, threshold, or routing change is permitted after this commit.

## Metrics

- **ALCE sentence-level citation precision and recall**, judge **MiniCheck**
  (`lytang/MiniCheck-Flan-T5-Large`).
- **The primary precision figure is the ALCE-standard convention**, under which an
  example that produced no citations scores 0. Cited-sample precision is reported
  alongside, never instead.
- Coverage (answered / total) and correctness (STR-EM against gold short answers).
- **TRUE is the production verifier and never judges.** Using it to score its own
  filtering decisions would be circular.
- Annotation labels are stripped before any text reaches a judge or a scorer.

## Success criterion

A pre-registered criterion has now triggered and proved mis-specified in two
consecutive rounds. G6's compared against a reference arm that had changed
underneath it. G7's compared arm-level means across arms answering different
query sets, which structurally penalises any coverage gain — the very thing the
method was built to deliver. Each explanation was legitimate on its own, and a
third instance would rightly read as reinterpreting the rules whenever the result
is inconvenient. Both lessons are encoded directly in the wording below:

> **The calibration result replicates if, on queries answered by both arms,
> nogate's citation precision exceeds baseline's at p < 0.05, and nogate's
> coverage deficit against baseline is not significantly worse than the
> calibration value of −0.018.**

Two properties this wording has that the previous two did not: it is **paired on
commonly-answered queries**, so neither arm is charged for answering more; and it
names a **fixed numeric reference** (−0.018) rather than another arm's
simultaneously-moving mean.

Tests: paired randomization, 10 000 iterations, seed 13, with bootstrap CI —
the same protocol used throughout calibration.

## Calibration values this is measured against

From G8 (final calibration), for reference at read-out time:

| | nogate | baseline |
|---|---|---|
| coverage | see G8 | see G8 |
| correctness (STR-EM) | see G8 | see G8 |
| citation precision (ALCE) | see G8 | see G8 |
| citation recall | see G8 | see G8 |

The G7 values the criterion's −0.018 comes from: coverage deficit −0.018
(p = 0.297), correctness +0.001 (p = 0.888), citation precision +0.191 (p = 0.0),
citation recall +0.071 (p = 0.009).

## Procedure

1. The three held-out sets — **HotpotQA, RGB, MuSiQue-Full** — are loaded **once**,
   and only after this document is committed. They have never been loaded during
   any calibration round.
2. All five arms run in one job, as in calibration, so cross-run comparability is
   not reintroduced as a confound.
3. **Results are reported whatever they show.** A failure to replicate is a
   reportable finding about the method's generality, not a prompt to adjust the
   method, the metric, or this criterion.
4. **No system change follows from seeing held-out results.** Anything discovered
   afterwards goes into limitations, unless it makes a reported number *wrong*
   rather than *suboptimal* — in which case the correction is made, disclosed, and
   the affected numbers are re-reported as corrected.

### If a job fails partway

Decided now, because deciding it after a crash — with results possibly
half-written — is not a decision anyone should trust.

- **Infrastructure failure before any results exist may be re-run.** Out of
  memory, walltime exceeded, node failure, a model that will not load, a crash in
  the generation stage. Nothing was observed, so nothing is contaminated. The
  re-run is a first run.
- **Once results exist, they stand.** No re-run, whatever they show. "Results
  exist" means the scoring stage produced a metric for that dataset — not that
  generation completed, and not that a partial file is on disk.
- **The boundary is the scorer, not the eye.** A generation job that finished but
  was never scored may be re-run; whether anyone happened to look at a log is not
  the test, because a test that depends on what a person remembers seeing is not
  a test.
- A dataset whose generation completes but whose scoring fails is re-scored, not
  re-generated: scoring is deterministic given the same inputs and the same judge,
  so re-scoring observes nothing new.
- Every re-run, and its reason, is recorded in `docs/hpc-run-log.md` at the time
  it happens.

The one case this deliberately does not permit: re-running because the numbers
look wrong. That is the failure mode the whole document exists to prevent.

**Per-dataset independence.** Each dataset is one job containing all five arms
(cross-job non-determinism was measured at 11% answer churn under fixed seed,
code, model and node, so no arm may be split across jobs). A failure on one
dataset does not invalidate another that already produced results.

## Known limitations, recorded in advance

Stated now so they cannot be presented later as though they were anticipated only
once convenient:

- **`count_sentences` and the scorer share one abbreviation list.** It is small
  and English-only. A missed abbreviation splits one sentence into two; a wrong
  entry merges two real ones.
- **The claim splitter over-segments.** Fragments such as "*. in 2010.*" are
  emitted as separate sentences, each carrying its own label. They are counted as
  separate sentences because that is what the answer contains.
- **Per-sentence citation mapping is exact for the verify arms and flat for the
  baseline**, which has no per-sentence record. The baseline is therefore scored
  under the more generous convention.
- **The entity layer's benign-data cost is measured; its adversarial benefit is
  measured on one slice only** (G1 counterfactual, 0.963 → 1.000).
- **Human adjudication covered the entity-conflict path only**, 40 items across
  two blind rounds. No blind audit exists for the annotate path.
