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
| 4 | Five-arm calibration | `18307720` + `18307721` | MiniCheck |
| 5 | Cross-job non-determinism | `18269703`, `18295681`, `18307720` | byte comparison |

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

## 4. Five-arm calibration — job `18307720` / `18307721`

Judge MiniCheck. Primary precision figure is the ALCE-standard convention;
cited-sample precision alongside.

| arm | coverage | correctness (STR-EM) | cite prec (ALCE) | cite prec (cited) | cite recall | answered |
|---|---|---|---|---|---|---|
| baseline | 0.933 | 0.273 | 0.613 | 0.613 (373) | 0.647 | 373/400 |
| verify-only | 0.633 | 0.197 | 0.871 | 0.871 (252) | 0.871 | 252/398 |
| verify-annotate-capped | 0.633 | 0.196 | 0.911 | 0.911 (252) | 0.885 | 252/398 |
| verify-annotate-open | 0.789 | 0.219 | 0.731 | 0.911 (252) | 0.710 | 314/398 |
| **verify-annotate-nogate** | **0.917** | **0.267** | 0.731 | 0.871 (306) | 0.712 | 365/398 |

Paired randomization, 10 000 iterations, seed 13, bootstrap CI — all within-job:

| comparison | coverage | correctness | cite precision | cite recall |
|---|---|---|---|---|
| **nogate vs baseline** | −0.015 (**p=0.352**) | −0.006 (**p=0.572**) | **+0.203 (p=0.0)** | **+0.079 (p=0.002)** |
| nogate vs open | +0.128 (p=0.0) | +0.048 (p=0.0) | −0.014 (p=0.145) | −0.002 (p=0.924) |
| nogate vs verify-only | +0.284 (p=0.0) | +0.070 (p=0.0) | +0.024 (p=0.232) | −0.002 (p=0.954) |
| open vs capped | +0.156 (p=0.0) | +0.023 (p=0.0) | 0.000 (p=1.0) | 0.000 (p=1.0) |

**The result:** indistinguishable from the baseline on coverage and correctness,
with +0.20 citation precision and +0.08 citation recall. Every earlier round had
to trade away coverage to buy citation quality; that trade is gone.

Sentence composition:

| arm | verified & cited | annotated unverified | of which review-flagged | total |
|---|---|---|---|---|
| baseline | 384 | 0 | 0 | 384 |
| verify-only | 257 | 0 | 0 | 257 |
| verify-annotate-capped | 290 | 15 | 0 | 305 |
| verify-annotate-open | 290 | 82 | 0 | 372 |
| **verify-annotate-nogate** | **358** | 80 | **66** | 438 |

Routing, nogate: 439 claims → 359 verified, 80 annotated, **0 destroyed**. The
entity gate would have destroyed 66; all 66 are cited instead. Control-arm
self-check exact (`gate_would_drop` = 66 = `dropped_entity_conflict`).

**Review flag** (`[may warrant review]`), enrichment with intervals:

| | error rate | 95% Wilson | n |
|---|---|---|---|
| flagged | 0.237 | [0.147, 0.360] | 59 |
| unflagged | 0.103 | [0.070, 0.145] | 247 |

Lift 2.3×, intervals separating by 0.002; the lift is compatible with 1.01×–5.18×.
Directionally sound, poorly determined at n = 59 — adequate for a presentation
hedge, and explicitly not adequate to justify destroying anything.

---

## 5. Cross-job non-determinism — the provenance rule's evidence

Same code, same seed, same models, same node (`bp1-gpu035`), greedy decoding
(`temperature=0.0`, `do_sample=False`).

| pair | per-arm speed | baseline answers identical |
|---|---|---|
| G6 `18269703` → G7 `18295681` | 6.03 → 6.06 s/arm/case | **394/394 (1.000)** |
| G7 `18295681` → G8 `18307720` | 6.06 → **2.70** s/arm/case | **356/400 (0.890)** |

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
