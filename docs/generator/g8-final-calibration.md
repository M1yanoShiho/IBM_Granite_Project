# G8 — final calibration

Jobs `18307720` (generation, 01:31:24) + `18307721` (scoring, 00:16:24), five arms
in one job, both exit 0. Judge **MiniCheck**; TRUE is the production verifier and
never judges. Pre-registered in `docs/hpc-run-log.md` §G8 before the run, together
with a quantified neutrality threshold.

Calibration is frozen here. The held-out protocol is pre-registered separately in
`heldout-preregistration.md`.

## Five arms

| arm | coverage | correctness (STR-EM) | cite prec (ALCE) | cite prec (cited) | cite recall | answered |
|---|---|---|---|---|---|---|
| baseline | 0.933 | 0.273 | 0.613 | 0.613 (373) | 0.647 | 373/400 |
| verify-only | 0.633 | 0.197 | 0.871 | 0.871 (252) | 0.871 | 252/398 |
| verify-annotate-capped | 0.633 | 0.196 | 0.911 | 0.911 (252) | 0.885 | 252/398 |
| verify-annotate-open | 0.789 | 0.219 | 0.731 | 0.911 (252) | 0.710 | 314/398 |
| **verify-annotate-nogate** | **0.917** | **0.267** | 0.731 | 0.871 (306) | 0.712 | 365/398 |

### The headline replicates

| nogate vs baseline | G8 | G7 |
|---|---|---|
| coverage | −0.015 (**p = 0.352**) | −0.018 (p = 0.297) |
| correctness | −0.006 (**p = 0.572**) | +0.001 (p = 0.888) |
| citation precision | **+0.203 (p = 0.0)** | +0.191 (p = 0.0) |
| citation recall | **+0.079 (p = 0.002)** | +0.071 (p = 0.009) |

Statistically indistinguishable from the baseline on coverage and correctness,
while carrying +0.20 citation precision and +0.08 citation recall. Both null
results are still null and both gains are still significant, on an independently
generated set of answers.

`nogate vs open` also replicates: coverage +0.128 (p = 0.0), correctness +0.048
(p = 0.0), citation precision −0.014 (p = 0.145), citation recall −0.002
(p = 0.924). The entity gate still costs ~13 coverage points and ~5 correctness
points for no measurable citation-quality gain.

Incidentally, G7's failure criterion — cited-sample precision below verify-only's
— is no longer breached: 0.8715 against 0.8710. That margin is 0.0005 and should
be read as "these are the same number", not as a pass. The criterion was
mis-specified either way, which is why the held-out one is written differently.

## Task 1 — the sentence fix, verified on the query it destroyed

G7 lost query `-5608871660568079389` in both annotate arms. In G8 it survives:

```
'Matt Kuchar won the 2018 U.S. Open golf championship. [unverified]'
   sentences=1  markers=1  -> accepted
```

`U.S. Open` — exactly the pattern. `ValidationError` does not appear anywhere in
the G8 log; per-arm errors fall from 5–6 to **2**, and both remaining ones are
`LLM output must be valid JSON`, unrelated to this.

The rule only suppresses terminators after abbreviations and initials. The
period-then-lowercase pattern (`World Cup. in 2010.`), 30–36 per arm, is the claim
splitter's genuine output — separate sentences each carrying their own label — and
is deliberately left alone.

## Task 2 — the review flag

**Wording, verbatim: `[may warrant review]`**

66 claims flagged in nogate, **0 in every other arm** — with the gate engaged a
would-drop claim is dropped, so an ablation arm cannot produce the label, and the
ablations are untouched by construction rather than by inspection.

### Enrichment on this run

| | precision | error rate | n |
|---|---|---|---|
| flagged examples | **0.763** | 0.237 | 59 |
| unflagged examples | **0.897** | 0.103 | 247 |

**Lift: 2.3× the error rate.** Lower than G7's 2.9× (0.678 / 0.888), and this is
the operative figure — it is measured on the run being reported rather than
inherited from the round that motivated the flag. The screening claim holds:
roughly one in four flagged sentences has a citation the judge rejects, against
one in ten elsewhere.

## Task 3 — metrics-neutrality, proved rather than inferred

The guide's check was "compare G8 to G7 and see whether any arm moves more than
the `count_sentences` fix predicts". **That test turned out to be unusable, and a
stronger one was available.**

### Why the cross-run test does not work

Generation does **not** reproduce across jobs. Comparing G8 to G7 answer by
answer:

| arm | identical answers |
|---|---|
| baseline | 356/400 (0.890) |
| verify-only | 345/395 (0.873) |
| verify-annotate-capped | 347/395 (0.878) |
| verify-annotate-open | 337/394 (0.855) |
| verify-annotate-nogate | 331/394 (0.840, ignoring the flag) |

**The baseline arm contains none of G8's changes and still moves on 11% of
queries.** Decoding is greedy (`temperature=0.0` → `do_sample=False`), the seed,
the code, the models and even the node (`bp1-gpu035`) are identical across G7 and
G8; the run was nevertheless 2.3× faster (13.5 s/case against 31.0). Greedy
decoding is deterministic in arithmetic but not bitwise across differing GPU
kernel selection and reduction order, and near-ties in the logits then flip.

The baseline's own G7→G8 movement is therefore the noise floor: coverage +0.008,
correctness +0.007, precision +0.016, recall +0.012. Every G7→G8 movement in
nogate sits inside it — coverage +0.006, correctness −0.001, precision +0.013,
recall +0.009 — so nothing there is attributable in either direction. The
predicted −7 to −12 sentence deltas are simply not measurable against an 11%
answer-level confound.

**This vindicates the all-arms-in-one-job discipline and extends it: within-job
comparisons are sound, cross-job ones are not, even holding seed, code and
hardware fixed.** Every claim in this document is within-job.

### The proof that replaces it

Neutrality can be settled exactly, without a run. Rebuilding nogate's scoring
input from G8's own records, once as generated and once with every review label
stripped:

```
examples: with flag 365, without 365
ScoredExample sequences identical: True
kept sentences: 451 vs 451     annotated: 80 vs 80
coverage identical: True       STR-EM identical: True
```

The `ScoredExample` objects handed to the judge are **byte-identical** with and
without the label. The flag cannot move a metric, because by the time any metric
is computed it is not there. That is stronger than any cross-run comparison would
have been.

## Sentence composition

| arm | verified & cited | annotated unverified | of which flagged (cited) | total kept |
|---|---|---|---|---|
| baseline | 384 | 0 | 0 | 384 |
| verify-only | 257 | 0 | 0 | 257 |
| verify-annotate-capped | 290 | 15 | 0 | 305 |
| verify-annotate-open | 290 | 82 | 0 | 372 |
| **verify-annotate-nogate** | **358** | 80 | **66** | 438 |

Routing for nogate: 439 claims, 359 verified, 80 annotated, **0 dropped**. The
gate would have destroyed **66**; all 66 are cited instead, none fell through to
annotation, and the control-arm self-check is exact (`gate_would_drop` = 66 =
`dropped_entity_conflict`).

## Status

Calibration frozen. No further design change. Held-out runs next, under
`heldout-preregistration.md`, whose criterion is paired on commonly-answered
queries against a fixed reference — written that way because a pre-registered
criterion has been mis-specified twice, and the cross-run result above shows a
third way it could have been.
