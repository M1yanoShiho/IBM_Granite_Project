# NLI calibration results (Generator Part B, Half 1.5)

Measurement only -- no production code changed. Data: canonical SciFact release (SUPPORT/CONTRADICT rationale sentences + full abstracts; hard neutrals = claim x a cited-but-non-evidence abstract). Premise = evidence, hypothesis = claim. Sentence-cell aggregation: max over sentences, precedence entailment > contradiction > neutral. CPU.

Pairs per gold label: 50.

## id2label per checkpoint

- `cross-encoder/nli-deberta-v3-base`: `{"0": "contradiction", "1": "entailment", "2": "neutral"}`
- `cross-encoder/nli-deberta-v3-large`: `{"0": "contradiction", "1": "entailment", "2": "neutral"}`

## Headline decision table

| cell | entailment recall | contradiction FP rate | ms/pair |
|---|---|---|---|
| base / sentence | 0.220 | 0.480 | 276.3 |
| base / abstract | 0.100 | 0.080 | 237.0 |
| large / sentence | 0.260 | 0.320 | 1110.0 |
| large / abstract | 0.340 | 0.120 | 863.9 |

## Confusion matrices (gold rows x predicted cols)

### base / sentence (n=150)

| gold \ pred | entailment | contradiction | neutral |
|---|---|---|---|
| **entailment** | 11 | 2 | 37 |
| **contradiction** | 2 | 35 | 13 |
| **neutral** | 1 | 24 | 25 |

### base / abstract (n=150)

| gold \ pred | entailment | contradiction | neutral |
|---|---|---|---|
| **entailment** | 5 | 2 | 43 |
| **contradiction** | 3 | 28 | 19 |
| **neutral** | 1 | 4 | 45 |

### large / sentence (n=150)

| gold \ pred | entailment | contradiction | neutral |
|---|---|---|---|
| **entailment** | 13 | 4 | 33 |
| **contradiction** | 2 | 30 | 18 |
| **neutral** | 9 | 16 | 25 |

### large / abstract (n=150)

| gold \ pred | entailment | contradiction | neutral |
|---|---|---|---|
| **entailment** | 17 | 4 | 29 |
| **contradiction** | 2 | 31 | 17 |
| **neutral** | 3 | 6 | 41 |

## Reading (measurement, not a production change)

The three questions the pass exists to answer:

**1. base vs large -> large, but neither is good.** On the metric that matters
most (entailment recall), large/abstract (0.34) is 3.4x base/abstract (0.10) for
~3.6x the CPU cost (864 vs 237 ms/pair), so large earns its cost. But even the
best cell recalls only **34% of genuinely-supported claims** -- two-thirds of true
support gets no citation. This is the damaging false-negative mode the guide
flagged (risk #2): a missed entailment strips a legitimate citation and can drive
A3/A4 to delete a true claim. Off-the-shelf MNLI-style DeBERTa is known to degrade
on scientific paraphrase (SciFact's own baselines use fine-tuned verifiers), and
these numbers confirm it on our target domain.

**2. sentence-split vs whole-abstract -> whole-abstract. Do not split.**
Sentence-level (max over sentences) does **not** buy recall (base 0.22 vs 0.10 up,
but large 0.26 vs 0.34 down -- no consistent gain) and it **multiplies
contradiction false positives**: contradiction-FP jumps from 0.08/0.12 (abstract)
to 0.48/0.32 (sentence). Cause is mechanical -- every extra sentence is another
chance to fire a spurious contradiction, and one poisons the whole chunk under max
aggregation. Feed the whole chunk.

**3. `contradicted` signal -> demote to diagnostic-only, do not report as a
result.** Even in the best cell (large/abstract) the contradiction FP rate on
hard neutrals is 0.12. In real RAG most selected evidence is neutral w.r.t. any
one claim, so with a 12% FP the absolute count of false contradictions will swamp
true ones. At sentence level (0.32-0.48) it is unusable. Keep `contradicted` for
inspection; do not surface it as a user-facing "evidence contradicts the answer"
result on this model.

**Was the hard-neutral pool genuinely hard? Yes.** It is not a falsely-reassuring
near-zero: FP reaches 0.12 (large/abstract) and 0.32-0.48 (sentence), and even at
abstract granularity only 45/50 (base) and 41/50 (large) neutrals are labeled
neutral. The cited-but-non-evidence construction produces genuinely on-topic,
entity-sharing negatives (they were cited by the same claim's authors), which is
what makes the FP number meaningful.

**id2label normalization on large: no problem.** Both checkpoints report the
identical mapping `{0: contradiction, 1: entailment, 2: neutral}`, so
`normalize_nli_label` handled large with no special-casing. No production code was
touched.

**Lever for the decision phase (NOT done here):** predictions use `argmax`. Many
true entailments land as `neutral` with entailment as the likely runner-up, so a
softmax threshold (accept entailment when `P(entail)` exceeds a tuned cutoff)
could recover recall at some precision cost -- a concrete next experiment. Per the
guide this pass changes no thresholds; flagging only.

**Bottom line for the next decision:** if staying on off-the-shelf NLI, use
`large` + whole-abstract and treat it as a low-recall precision filter (it will
under-cite); the 34% recall ceiling is the real risk to the attribution approach
and argues for either a probability threshold or a domain-adapted/fine-tuned
verifier before this is relied on.

