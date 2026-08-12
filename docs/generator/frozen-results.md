# Frozen results — Generator

Final tables for the write-up. These results will not change.

**The evaluated system is tagged `generator-frozen-g9-qampari-2026-08-10`**
(commit `eb64023`) — the commit the cluster checked out for both the G9
calibration and the QAMPARI held-out. `generator/` and `contracts/` are
byte-identical between the G9 freeze (`27e8280`) and that tag, so both runs
measured the same object. Anything above the tag is post-freeze engineering and
changes no number here.

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
| 7 | **Module-level held-out (QAMPARI)** | **`18326078` + `18326079`** | MiniCheck |

**§4 was rebuilt on G9.** The G8 tables it replaced are superseded, not merged:
no figure below is carried over from `18307720`.

---

## 1. Verifier selection

**Recomputed from the raw scores** (`results/verifier-triage/scores.jsonl`, 1189
rows) by `scripts/g1_verifier_metrics.py`, with every definition stated. Nothing
below is carried over from the earlier triage tables. Full per-slice output:
`local/report-writing/g1-verifier-metrics.md`.

**Definitions.** A verifier *fires* when `p_entail ≥ 0.50`.
`recall = TP / (gold-entailment rows)`; `FP rate = FP / (gold-neutral rows)`;
`precision = TP / (TP + FP)`, so precision is only defined on a slice carrying
both polarities. Of the seven cells, **only `asqa` does** — the `2wiki-*` cells
are entailment-only apart from `2wiki-neutral`, and `counterfactual` is
neutral-only.

### `asqa` — n = 300 (150 gold-entailment, 150 gold-neutral)

| backend | recall | FP rate | precision | TP | FP |
|---|---|---|---|---|---|
| **TRUE** (`t5_xxl_true_nli_mixture`) | **0.7467** | **0.0067** | **0.9912** | 112 | 1 |
| Granite-3B self-check | 0.7667 | 0.0667 | 0.9200 | 115 | 10 |
| Granite-8B self-check | 0.9000 | 0.2400 | 0.7895 | 135 | 36 |

### `counterfactual` — n = 109, all gold-neutral (entity-substituted)

Rejection rate = 1 − FP rate.

| backend | rejection | FP |
|---|---|---|
| **TRUE** | **0.9633** | 4 |
| Granite-3B | 0.9450 | 6 |
| Granite-8B | 0.8716 | 14 |

### Two corrections this recomputation forces

**The published derived citation precision of 0.966 does not reproduce.** On the
`asqa` slice the formula gives **0.9912**; on `asqa + 2wiki-neutral` it is also
0.9912; across all cells, 0.9874. No principled slice yields 0.966. **0.9912 on
`asqa` is the figure to use**, with its slice and formula named.

**TRUE's 0.747 recall is ASQA-specific, not global.** On the 2WikiMultihop
entailment cells (n = 540) its recall is **0.5204**, and across all
gold-entailment rows (n = 690) it is **0.5696**. The 0.747 figure should always
carry its slice.

### Why TRUE, and why MiniCheck is not in production

TRUE is selected on the **precision/FP** axis, not on recall: Granite-8B has
higher recall (0.9000) but a 0.2400 false-positive rate, so it would attach
citations that do not hold. Granite self-check is additionally rejected on
principle — the generator grading its own output is not independent evidence.
MiniCheck is deliberately reserved as the downstream **judge**, so the production
verifier never scores its own decisions.

The operating point (0.50) is frozen and was never tuned against a reported
metric. TRUE's 0.7467 ASQA recall is the origin of the redesign: under a delete
policy, roughly one supported claim in four is destroyed, which is what
verify-and-annotate exists to stop.

### The entity layer

The entity check targets a real and documented attack surface: **evidence that
supports a claim in wording while the entity has been substituted**, to which a
general-purpose entailment model is blind.

Two independent blind adjudications put its false-veto rate on natural data at
**70%**, so it was removed from the citation decision. Removing it cost **nothing
on either citation axis** (precision −0.0046, p = 0.698; recall +0.0043,
p = 0.720) and recovered **12.7 points of coverage and 4.4 points of correctness**
(G9, nogate vs open). It is retained in **observe-only** mode.

Its only positive measurement comes from a synthetic slice constructed in its
favour, using a component version later shown to be inaccurate, and **cannot be
reproduced from the surviving artefacts** — the raw file carries the three NLI
backends only, with no entity-layer column. **It is therefore not relied upon.**

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

## 7. Module-level held-out — QAMPARI, jobs `18326078` / `18326079`

400 queries, seed 13, top-5 GTR passages — the same evidence setting as
calibration. Pre-registered at `c559d5c`, 2026-08-09 15:57:20 +0100, **before any
QAMPARI byte was read**. Loaded once; no system change followed.

Correctness here is **answer recall by containment**, the pre-registered metric.
It is **not** ALCE's official QAMPARI F1 and **not comparable to ASQA's STR-EM**;
it is recall-only. Citation metrics are the identical measurement on both sets,
which is why the generalisation test is strongest on the axis the claim rests on.

| arm | coverage | recall | rec@5 | cite prec (ALCE) | cite prec (cited) | cite recall | answered |
|---|---|---|---|---|---|---|---|
| baseline | 0.808 | 0.075 | 0.121 | 0.535 | 0.535 (323) | 0.578 | 323/400 |
| verify-only | 0.597 | 0.055 | 0.086 | 0.802 | 0.802 (228) | 0.802 | 228/382 |
| verify-annotate-capped | 0.579 | 0.052 | 0.084 | 0.844 | 0.848 (220) | 0.828 | 221/382 |
| verify-annotate-open | 0.720 | 0.058 | 0.096 | 0.679 | 0.848 (220) | 0.665 | 275/382 |
| **verify-annotate-nogate** | **0.785** | **0.065** | **0.107** | 0.696 | **0.848 (246)** | 0.684 | 300/382 |

Paired, within-job:

| comparison | coverage | correctness | rec@5 | cite precision | cite recall |
|---|---|---|---|---|---|
| **nogate vs baseline** | −0.0157 (p=0.469) | −0.0105 (p=0.025) | −0.0141 (p=0.101) | **+0.2685 (p=0.0)** | **+0.0929 (p=0.010)** |
| nogate vs open | +0.0654 (p=0.0) | +0.0068 (p=0.0) | +0.0110 (p=0.0) | −0.0066 (p=0.350) | −0.0010 (p=0.874) |

### The pre-registered criterion

| conjunct | result |
|---|---|
| citation precision > baseline at p<0.05 | **+0.2685, p = 0.0**, CI [0.198, 0.336], n=233 — **met** |
| coverage deficit not worse than −0.0152 | −0.0157, p=0.469 — **untestable, see below** |

### Why the coverage conjunct is untestable, not passed

The registered claim-splitter clause fired: failure was **18/400 = 4.50%** against
calibration's ~0.5%, so *"coverage comparisons are reported as unreliable and the
failure rate is stated with the results."*

The bias is measured, not assumed. The splitter sits upstream of every verify arm,
so a failure removes the query from all of them while the **baseline** keeps it —
and the baseline answered **17/18 (0.944)** of the failed queries against **0.801**
of the rest. Pairing correctly drops them from both sides, which removes exactly
the ground the baseline was winning on.

| treatment | nogate | baseline | deficit |
|---|---|---|---|
| paired, failures excluded (reported) | 300/382 = 0.785 | 306/382 = 0.801 | **−0.0157** |
| failures counted as unanswered (bound) | 300/400 = 0.750 | 323/400 = 0.808 | **−0.0575** |

−0.0157 passes; −0.0575 would not. The true value is between them and this run
cannot locate it. **Reported as untested, not as passed.**

The failures are not concentrated on many-answer questions (median gold answers
7.5 for failures against 8.0 for the rest), so this is a higher base rate of
malformed JSON on a new distribution.

### What replicates

| claim | ASQA (G9) | QAMPARI | |
|---|---|---|---|
| citation precision > baseline | +0.1913 (p=0.0) | **+0.2685 (p=0.0)** | **replicates, larger** |
| citation recall > baseline | +0.0711 (p=0.009) | **+0.0929 (p=0.010)** | **replicates, larger** |
| gate costs coverage, buys no citation quality | +0.1266 cov, citation ns | +0.0654 cov, citation ns | **replicates** |
| nothing destroyed | 0 drops | 0 drops | **holds** |
| coverage ≈ baseline | −0.0152 (p=0.386) | −0.0157 (p=0.469) | **untestable here** |
| correctness ≈ baseline | +0.0015 (p=0.892) | **−0.0105 (p=0.025)** | **does not replicate** |
| review-flag enrichment | 2.86× | **1.28×** | **does not replicate** |

Routing: 441 claims → 364 verified, 77 annotated, **0 destroyed**. The gate would
have destroyed 48; all 48 cited instead. Control self-check exact (48 = 48).
Declared-citation survival 310/400 = 0.775.

**Review flag on QAMPARI**: flagged error rate 0.184 (n=44) against unflagged
0.145 (n=202) — **lift 1.28×** against calibration's 2.86×. The screening claim is
**calibration-specific** and must not be stated as a general property of the
entity layer.

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
- **The claim splitter is a single point of total failure.** It sits upstream of
  every verification arm, so one malformed response removes a query from all of
  them while the baseline keeps it. Its rate moved nine-fold between two datasets
  in the same benchmark family (0.5% → 4.5%), which is what made QAMPARI's
  coverage comparison untestable.
- **Correctness on QAMPARI is recall-only**, so an answer that enumerates from
  parametric knowledge is rewarded and never penalised for over-generation.
