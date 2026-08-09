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

### Across three datasets

The criterion above is evaluated **per dataset**. How the three combine is fixed
here, because deciding it after seeing two passes and one failure would look like
picking whichever aggregation was convenient:

> **Full replication requires the criterion to hold on all three datasets.** If it
> holds on some but not all, the result is reported as **partial replication**,
> with the specific pattern stated per dataset, and **no aggregate-level claim is
> made** — no pooling, no averaging, no "two of three" framing offered as though
> it were the headline.

This sets no impossible bar: partial replication is a reportable, publishable
outcome. What it forecloses is choosing the unit of analysis after the fact.

For MuSiQue-Full the criterion is evaluated on the **answerable** subset, per the
addendum below.

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

## Addendum — MuSiQue-Full subsets

*Added after the plumbing dry run, **before any held-out result exists**. It is a
reporting refinement forced by the dataset's structure, not a reaction to any
number, and it is recorded separately from the original text so the distinction
stays visible.*

The dry run established, by counting rather than assumption, that MuSiQue-Full's
dev split is **4834 records = 2417 ids appearing twice**, once answerable and
once not — and that **both variants carry a non-empty gold answer string**.

For an unanswerable item the supporting paragraphs have been removed, so the
evidence does not support that gold answer. String-match correctness would
therefore **reward a system for asserting the answer anyway, and penalise
abstention or an unverified label** — the precise inversion of what this project
claims to be good at. A single blended STR-EM over MuSiQue-Full would flatter the
baseline and punish the main method for behaving correctly, and would be
misleading in either direction.

Accordingly, for MuSiQue-Full only:

1. The two variants are given distinct query IDs (`{id}#ans`, `{id}#unans`); the
   raw id is not unique and would collide in every id-keyed structure, the paired
   tests included.
2. **Coverage, correctness, citation precision and citation recall are reported
   separately for the answerable and unanswerable subsets**, and the pre-registered
   criterion is evaluated on the **answerable** subset, which is the one where
   correctness means what the criterion assumes.
3. On the unanswerable subset, **STR-EM is a negative indicator and is read as
   one.** That subset is a ready-made test of honest abstention, and the two
   quantities point in opposite directions:

   | quantity | direction | what it means |
   |---|---|---|
   | abstention / `[unverified]` rate | **higher is better** | the system correctly recognised that the evidence does not support an answer |
   | STR-EM | **higher is worse** | parametric knowledge leaked; the system asserted a fact its evidence does not carry |

   This is registered now, before any result exists, precisely because it inverts
   the usual reading. A high STR-EM here is not partial credit — the gold string
   is unreachable from the given evidence, so producing it is evidence of
   ungrounded assertion, which is the failure mode this whole method exists to
   prevent. The figure is reported under the label **"ungrounded assertion rate"**,
   never as correctness.
4. The union figure is also reported, so nothing is hidden by the split.

HotpotQA and RGB are unaffected: neither has an unanswerable partition.

## Addendum — RGB composition, and the counterfactual sub-test

*Added with the addendum above, before any held-out result exists.*

**Composition, established by fetching all four files and counting** — the 300
records are *not* spread across the four sub-tests. RGB ships each sub-test as its
own file:

| file | records | sub-test |
|---|---|---|
| `en.json` | **300** | noise robustness / negative rejection |
| `en_int.json` | 100 | information integration |
| `en_fact.json` | **100** | counterfactual robustness |
| `en_refine.json` | 300 | refine |

So "RGB, 300 records" is one sub-test at full size. The pre-registered RGB result
uses `en.json` and nothing else, and is labelled with the sub-test it is.

### Secondary, pre-registered analysis: counterfactual robustness

`en_fact.json` is **the only real adversarial data available to this project.**
The threat-model claim currently rests on G1's entity-substitution slice, which
was synthetic and constructed by us. This is a published benchmark's adversarial
subset, so it is the strongest available test of whether the entity layer earns
its keep — and it is registered as a **named secondary analysis**, never folded
into a headline number.

Each item carries a true `answer`, a planted `fakeanswer`, three `positive`
documents stating the truth, and three `positive_wrong` documents stating the
falsehood. **The evidence pool is built true-documents-first, then wrong ones**,
so a top-k cut leaves both present. That is the whole point: it creates a pool
carrying two competing values in the same role, which is exactly the condition
the entity layer claims to detect. A pool of only-wrong documents would test
nothing about it.

**What is measured** (all on `verify-annotate-nogate`, gate observe-only):

1. **Flag rate** on this subset against the flag rate on `en.json` — does the
   signal fire more under adversarial conditions than benign ones?
2. **Flag rate on sentences asserting `fakeanswer`** against sentences asserting
   `answer` — does it fire *selectively*, or merely more?
3. Cited-sample citation precision, flagged against unflagged, as elsewhere.
4. The observe-only counters: how many claims the gate would have destroyed here.

**What would support the claim:** the flag fires substantially more often on this
subset than on `en.json`, and preferentially on sentences carrying the planted
falsehood.

**What would refute it:** the flag fires at a similar rate to benign data, or
fires on truth and falsehood indiscriminately. Either is reported as such.

**A limitation stated in advance, because it will otherwise look like a defect.**
RGB's own counterfactual metric is *error detection rate*: whether the model
notices, from its own parametric knowledge, that a document is factually wrong.
**This system deliberately does not do that.** The Generator sees only the
selected evidence and never consults world knowledge, by contract. RGB's headline
metric therefore measures a capability the method intentionally lacks, and it is
**not** reported as a score of this system. The quantities above are reported
instead, and this distinction is stated wherever the subset appears — a system
faithful to wrong evidence is behaving correctly under this method's definition,
and that definition is itself part of what is being evaluated.

## Sampling — drawn and committed before the run

The samples are drawn by `scripts/heldout_sample.py`, seed 13, and committed as
`configs/heldout-sample.json` **before the run**, listing every evaluated
`query_id` with a SHA-256 over the ordered list. A sample that is fixed and
auditable cannot be re-drawn after seeing anything.

| dataset | population | drawn |
|---|---|---|
| hotpotqa | 7405 | **400 queries**, uniform — matching the calibration sample size |
| musique-full | 4834 | **400 ids × both variants = 800 records** |
| rgb | 300 | all 300 |
| rgb-counterfactual | 100 | all 100 |

MuSiQue samples **ids, not records**. Drawing 400 records would break the pair
structure the set exists to provide — the same question with and without its
support — and would leave the answerable and unanswerable subsets as two
unrelated samples rather than a matched comparison. Drawing 400 ids gives 400 in
each subset, matched.

Full-set runs are not feasible and are not the reason for sampling being 400:
at the calibration rate of 13.5–31 s/case across five arms, HotpotQA in full
would take 28–64 hours against a 12-hour walltime. 400 is chosen to match
calibration's statistical power, and the constraint is noted so the choice is not
mistaken for one made purely on cost.

## Addendum — claim-splitter failure rate

*Added with the addenda above, before any held-out result exists.*

> **If claim-splitter failure on any held-out set is materially above the
> calibration rate of roughly 0.5%, coverage comparisons on that set are reported
> as unreliable, and the failure rate is stated with the results.**

The splitter is the **single point of total failure**. It sits upstream of every
verification arm, so one failure removes the same query from all of them while
leaving the baseline — which does not use it — scored on the full set. At
calibration this cost 2 queries in 400 (0.5%) and shifted arm-level coverage by
+0.003 in the verify arms' favour, which is negligible. On an unseen distribution
the rate is unknown.

`--max-error-rate 0.10` is a **wide** guard: at a 9% failure rate the job
completes, writes every file, and produces a report that looks entirely
plausible. This criterion is the narrow one, and it is about *interpretation*
rather than about aborting — the run still completes and is still reported, with
the caveat attached where a reader will see it.

The rate is reported per dataset regardless of whether it clears the bar.

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
