# Frozen results — Generator

Final tables for the write-up. These results will not change.

**Provenance rule, applied throughout: every number is traceable to a single
job, and no table mixes jobs.** The reason is measured and appears as §5 below.
Where a figure from a different job is mentioned for context, it is labelled and
never subtracted from anything.

| § | Result | Job(s) | Judge / instrument |
|---|---|---|---|
| 1 | Verifier selection, six arms | `18240434` triage | human-labelled slices |
| 2 | The ALCE-artefact finding | `18267966` + `18268709` | MiniCheck |
| 3 | Blind human adjudication, three rounds | n/a (human) | the adjudicator |
| 4 | **Five-arm calibration (final)** | **`18322642` + `18322643` (G9)** | MiniCheck |
| 5 | Cross-job non-determinism | `18269703`, `18295681`, `18307720`, `18322642` | byte comparison |
| 6 | Sentence-splitter unification | none — deterministic, on G8's recorded text | direct diff |

**§4 was rebuilt on G9.** The G8 tables it replaced are superseded, not merged:
no figure below is carried over from `18307720`.

---

## 1. Verifier selection — six arms

TRUE was selected as the production verifier and **never judges its own output**;
MiniCheck scores everything downstream, which is why the two are different models.

| verifier | role | outcome |
|---|---|---|
| `t5_xxl_true_nli_mixture` (TRUE) | selected | best calibrated recall on the human-labelled slice; 0.747 recall at threshold 0.50 |
| Granite self-check | rejected | the generator judging itself is not independent evidence |
| DeBERTa NLI cross-encoder | rejected | weaker on the counterfactual slice |
| MiniCheck | reserved as **judge** | kept out of production precisely so it can score without circularity |

The operating point (0.50) is frozen and was never tuned against a reported
metric. TRUE's 0.747 recall is the origin of the whole redesign: under a delete
policy, one supported claim in four is destroyed, which is what
verify-and-annotate exists to stop.

**Entity layer, adversarial slice:** 0.963 verifier-alone → **1.000** with the
layer engaged. This is the layer's one demonstrated benefit and it is real.

---

## 2. The ALCE artefact, and its control

The first three-arm comparison reported verify-annotate beating verify-only by
**+0.070** citation precision. That number was an artefact of scoring the two
arms under different citation conventions: verify-only fell back to a flat
citation list (1.92 citations/sentence) while verify-annotate used an exact
per-sentence mapping (1.13), and ALCE's redundancy ablation systematically
penalises the former.

Per-arm control, after applying one convention to both (job `18267966` +
`18268709`):

| arm | precision, flat convention | precision, exact mapping | Δ |
|---|---|---|---|
| verify-only | 0.760 | **0.871** | +0.111 |
| verify-annotate | 0.890 | 0.890 | 0.000 |

The arm that changed convention moved by 0.111; the arm that did not moved by
0.000. **The +0.070 "improvement" was 0.070 of measurement.** Corrected, the
delete-vs-annotate difference is null on every axis — which is the finding that
sent the work to the abstention cap, and eventually to the result in §4.

This is the clearest case in the project of a metric appearing to confirm a
design and actually measuring the harness.

---

## 3. Blind human adjudication — what automation missed

Three rounds, verdicts hidden, no highlighting, no score ordering.

| round | scope | n | headline | what it caught that no metric did |
|---|---|---|---|---|
| 1 | entity-conflict drops | 20 | false-veto **0.700** | the extractor was vetoing on `name:some`, `name:season` — capitalised common nouns read as proper names |
| 2 | entity-conflict drops, post-fix | 20 | false-veto **0.700** | the fix removed every bad trigger and moved the error rate **not at all** — the failure was never about span extraction |
| 3 | claim/evidence support | ~70 | — | `required_facts` were built from the wrong ASQA field, so completeness was scoring against the wrong target entirely |

Round 2 is the one worth keeping. Composition changed completely — every trigger
became a real NER type, absent-triggers fell to 1/78 — and a composition change
looks like progress. Only re-adjudicating showed the error rate was identical to
the item: **14/20 both times**. Wrong destruction 0.850, 95% Wilson
[0.640, 0.948].

Reading the three drops that were *correct* settled the mechanism: an incomplete
claim, a quantifier-scope question, and an approved-versus-implemented temporal
distinction. **None is an entity conflict.** The layer was right for reasons it
does not model, so even its 0.150 correct rate is coincidental — a conclusion no
aggregate could have produced.

---

## 4. Five-arm calibration — job `18322642` / `18322643` (G9, final)

Judge MiniCheck. Primary precision figure is the ALCE-standard convention;
cited-sample precision alongside.

| arm | coverage | correctness (STR-EM) | cite prec (ALCE) | cite prec (cited) | cite recall | answered |
|---|---|---|---|---|---|---|
| baseline | 0.925 | 0.266 | 0.597 | 0.597 (370) | 0.635 | 370/400 |
| verify-only | 0.641 | 0.206 | 0.862 | 0.862 (253) | 0.862 | 253/395 |
| verify-annotate-capped | 0.635 | 0.202 | 0.890 | 0.890 (251) | 0.870 | 251/395 |
| verify-annotate-open | 0.785 | 0.223 | 0.720 | 0.890 (251) | 0.705 | 310/395 |
| **verify-annotate-nogate** | **0.911** | **0.267** | 0.716 | 0.848 (304) | 0.701 | 360/395 |

Paired randomization, 10 000 iterations, seed 13, bootstrap CI — all within-job:

| comparison | coverage | correctness | cite precision | cite recall |
|---|---|---|---|---|
| **nogate vs baseline** | −0.0152 (**p=0.386**) | +0.0015 (**p=0.892**) | **+0.1913 (p=0.0)** | **+0.0711 (p=0.009)** |
| nogate vs open | +0.1266 (p=0.0) | +0.0441 (p=0.0) | −0.0046 (p=0.698) | +0.0043 (p=0.720) |
| nogate vs verify-only | +0.2709 (p=0.0) | +0.0609 (p=0.0) | +0.0224 (p=0.238) | +0.0011 (p=0.970) |
| open vs capped | +0.1494 (p=0.0) | +0.0213 (p=0.0001) | 0.000 (p=1.0) | 0.000 (p=1.0) |
| verify-only vs baseline | −0.2861 (p=0.0) | −0.0594 (p=0.0) | +0.1712 (p=0.0) | +0.1306 (p=0.0001) |

**The result:** indistinguishable from the baseline on coverage and correctness,
with +0.19 citation precision and +0.07 citation recall. Every earlier round had
to trade away coverage to buy citation quality; that trade is gone.

Sentence composition:

| arm | verified & cited | annotated unverified | of which review-flagged | uncited & unlabelled | total |
|---|---|---|---|---|---|
| baseline | 415 | 0 | 0 | 0 | 415 |
| verify-only | 268 | 0 | 0 | 0 | 268 |
| verify-annotate-capped | 305 | 14 | 0 | 1 | 320 |
| verify-annotate-open | 305 | 76 | 0 | 3 | 384 |
| **verify-annotate-nogate** | **370** | 74 | **60** | 3 | 447 |

(Sentence counts here use the unified splitter, so they run slightly higher than
the scorer's `kept_sentences`, which is computed on the pre-repair split.)

Routing, nogate: 434 claims → 359 verified, 75 annotated, **0 destroyed**. The
entity gate would have destroyed 60; all 60 are cited instead, none fell through
to annotation. Control-arm self-check exact (`gate_would_drop` = 60 =
`dropped_entity_conflict`). Declared-citation survival 283/399 (0.709).

**Review flag** (`[may warrant review]`), enrichment with intervals:

| | error rate | 95% Wilson | n |
|---|---|---|---|
| flagged | 0.322 | [0.221, 0.456] | 58 |
| unflagged | 0.112 | [0.080, 0.160] | 246 |

Lift **2.86×**, corners 1.38×–5.70×. The intervals separate cleanly (0.221
against 0.160), unlike the G8 measurement where they parted by 0.002 — but the
width tells the same story: at n = 58 the multiplier is not well determined.
Adequate for a presentation hedge, explicitly not adequate to justify destroying
anything.

---

## 5. Cross-job non-determinism — the provenance rule's evidence

Same code, same seed, same models, same node (`bp1-gpu035`), greedy decoding
(`temperature=0.0`, `do_sample=False`).

| pair | per-arm speed | baseline answers identical |
|---|---|---|
| G6 `18269703` → G7 `18295681` | 6.03 → 6.06 s/arm/case | **394/394 (1.000)** |
| G7 `18295681` → G8 `18307720` | 6.06 → **2.70** s/arm/case | **356/400 (0.890)** |
| G8 `18307720` → G9 `18322642` | 2.70 → **5.78** s/arm/case | **356/400 (0.890)** |
| **G7 `18295681` → G9 `18322642`** | 6.06 → 5.78 s/arm/case | **400/400 (1.000)** |

G9 is the confirmation the pattern needed, because it was not designed to be one.
It landed back on the ~6 s/arm/case profile and reproduced G7 **exactly** — 400/400
on baseline and 395/395 on every other arm — while remaining 11% divergent from
G8, which sits between them in time. **Runs agree when they execute alike and
disagree when they do not, irrespective of ordering.**

The caching explanation was checked and ruled out: G7 logged a full generation
pass at G6's speed with freshly produced per-arm answered and error counts, and
the runner has no caching path.

**Greedy decoding is deterministic in arithmetic, not bitwise.** GPU reduction
order and kernel selection vary with execution conditions, and near-ties in the
logits flip. When two runs happen to execute alike they agree exactly; when one
runs 2.2× faster, 11% of answers change.

**Consequences, applied throughout this repository:**

- **No reported comparison may span two jobs.** All five arms run in one job.
- A cross-job difference on an *unchanged* arm is drift, not a result:
  `verify-annotate-capped` reads citation precision 0.890 in G7 and 0.911 in G8
  with nothing touched between them. That 0.021 is named here so it is never
  read as a finding.
- The G7 document's claim that "cross-run comparability is verified rather than
  assumed" is **withdrawn**; the observation stands, the inference does not.

The measurement is worth reporting in its own right: it is a concrete, quantified
instance of a reproducibility failure mode that fixed seeds are widely assumed to
exclude.

---

## 6. Sentence-splitter unification — measured in isolation

Three implementations of "what is a sentence" existed, with three abbreviation
lists (contract ~50, claim splitter 10, scorer ALCE's). G8 unified two of them
after a disagreement destroyed an answer; G9 unified the third.

Measured **on fixed inputs** — the old and the new rule run over the same 1808
recorded G8 answers — because a run-to-run comparison could not have isolated it:

| | |
|---|---|
| answers examined | 1808 |
| answers whose boundaries change | **69 (3.816%)** |
| sentences | 2336 → 2262 (**−74**) |

Every change is a merge of a wrongly-split sentence:

```
old: ['Patrick S.', 'Castagne']
new: ['Patrick S. Castagne']

old: ['The Brown v.', 'Board of Education case took place in Topeka, Kansas.']
new: ['The Brown v. Board of Education case took place in Topeka, Kansas.']

old: ['...played the parents in "The Parent Trap.', '"']
new: ['...played the parents in "The Parent Trap."']
```

**This measurement caught a regression before it shipped.** The first unified rule
required whitespace immediately after the terminator, so `."` never split and
`'"Manifest Destiny." It means …'` merged into one sentence. Merging two real
sentences is the more damaging direction — `GenerationResult` counts sentences to
decide whether every uncited one carries a label, so an under-count lets an
unlabelled sentence through the validator. Fixed before G9 ran.

**Effect on the calibration answers: one query.** G9 differs from G7 on exactly
`-5608871660568079389` — `"Matt Kuchar won the 2018 U.S. Open golf championship."`
— which G7's validator destroyed and which now survives. Every other answer in
all five arms is identical. So the unification removed a **latent** inconsistency
rather than a manifesting one, on this corpus.

## Known limitations

- The abbreviation list shared by the contract and the scorer is small and
  English-only.
- The claim splitter over-segments; fragments are counted as the separate
  sentences they are written as.
- The baseline is scored under the more generous flat citation convention,
  having no per-sentence record.
- Two queries fail claim-splitting in every verify arm (stable across G7 and G8,
  clustering on nothing measurable). They fail upstream of any arm, so paired
  comparisons exclude them from both sides, but they flatter the verify arms by
  **+0.003 coverage** at arm level.
- The entity layer's adversarial benefit rests on a single slice.
- No blind audit exists for the annotate path; adjudication covered the
  entity-conflict path and claim support only.
