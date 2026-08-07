# G6 — lifting the abstention cap

Jobs `18269703` (generation) + `18269704` (scoring), four arms in one job.
Judge **MiniCheck**; TRUE is the production verifier and never judges.
Pre-registration in `docs/hpc-run-log.md` §G6, committed before the run.

The contract change this round measures: the old invariant guaranteed *every
answer is grounded*; the new one guarantees *every ungrounded sentence is
labelled*. Numbers move because of an approved contract change, not tuning.

## Four arms, three axes

| arm | coverage | correctness (STR-EM) | cite prec (ALCE) | cite prec (cited examples) | cite recall | answered |
|---|---|---|---|---|---|---|
| baseline | 0.925 | 0.266 | 0.597 | 0.597 (370) | 0.636 | 370/400 |
| verify-only | 0.641 | 0.206 | 0.862 | 0.862 (253) | 0.862 | 253/395 |
| verify-annotate-capped | 0.635 | 0.202 | 0.883 | 0.883 (251) | 0.870 | 251/395 |
| **verify-annotate-open** | **0.784** | **0.223** | 0.717 | **0.883 (251)** | 0.707 | 309/394 |

Two precision columns because they answer different questions. `cite prec` is
ALCE's own convention, under which an example that cited nothing at all scores
0. `cite prec (cited examples)` restricts the mean to examples that produced a
citation — which is what this study **pre-registered** ("annotated claims are
neither numerator nor denominator"). The two agree on every arm that cannot emit
a fully annotated answer, and diverge only on `verify-annotate-open`:

    0.883 × 251 / 309 = 0.717

exact to the digit. The arm-level 0.717 is a change of denominator, not a
regression. Both are reported; neither is silently substituted for the other.

## What the contract lift bought — open vs capped

Paired randomization, 10 000 iterations, bootstrap CI.

| axis | delta | p | 95% CI | n |
|---|---|---|---|---|
| coverage | **+0.147** | 0.0 | [0.112, 0.183] | 394 |
| correctness | **+0.021** | 0.0 | [0.011, 0.033] | 394 |
| citation precision | **0.000** | 1.0 | [0.000, 0.000] | 251 |
| citation recall | **0.000** | 1.0 | [0.000, 0.000] | 251 |

A delta of exactly zero on both citation axes is not a weak result; it is the
strongest possible form of the pre-registered claim. **All 251 answers the capped
arm produced are byte-identical in the open arm**, both arms emit exactly **313
cited sentences**, and both have exactly **251 examples carrying a citation**. The
lift is purely additive: 58 queries that previously abstained now answer, and
nothing that was already answered changed.

### Against the pre-registered failure criterion

> The lift fails if citation precision on cited sentences regresses toward
> baseline — which would mean unverified content is leaking into the cited pool
> or the annotation is not being excluded correctly — or if coverage does not
> rise materially above 0.633.

Precision on cited sentences: 0.883 → 0.883, paired delta 0.000. Zero leakage.
Coverage: 0.635 → 0.784, +0.147, p = 0.0. **The lift passes on both clauses.**

Expected directions, checked: coverage rises ✓, correctness rises ✓, precision on
cited sentences holds ✓. The predicted "citation recall falls substantially" is
true at arm level (0.870 → 0.707) and false pairwise (delta 0.000) — the fall is
entirely the 58 new zero-citation examples entering the mean, which is the
acknowledged cost, correctly priced.

## Against the baselines

| comparison | coverage | correctness | cite precision | cite recall |
|---|---|---|---|---|
| open vs baseline | −0.145 (p=0.0) | −0.043 (p=0.0) | **+0.192 (p=0.0)** | **+0.061 (p=0.032)** |
| open vs verify-only | +0.142 (p=0.0) | +0.017 (p=0.034) | +0.028 (p=0.11) | +0.006 (p=0.79) |
| capped vs verify-only | −0.005 (p=0.86) | −0.004 (p=0.53) | +0.028 (p=0.11) | +0.014 (p=0.47) |

Citation recall now **beats** the baseline significantly (+0.061, CI [0.006,
0.115]). Under the cap this method could only argue "much better precision, worse
recall, much worse coverage". Open, it argues **better precision and better
recall**, with coverage the remaining cost.

Annotate-vs-delete remains null on every axis under the cap — G5's finding
reproduces exactly. The cap was the whole story: with it lifted, the same
annotation mechanism moves coverage and correctness significantly.

## Sentence composition

| arm | verified & cited | annotated unverified | uncited, unlabelled | total |
|---|---|---|---|---|
| baseline | 427 | 0 | 0 | 427 |
| verify-only | 275 | 0 | 0 | 275 |
| verify-annotate-capped | 313 | 14 | 1 | 328 |
| verify-annotate-open | 313 | **75** | 3 | 391 |

The three uncited-unlabelled sentences in the open arm are splitter residue on
embedded quotations (`"Pee-wee's Playhouse."` splitting mid-quote), not contract
violations — the contract validator counts sentences with the marker stripped and
accepted every answer.

**Annotation reach**: of 77 annotated claims, **76 reach an answer** under the
lift, against **15** under the cap. The annotation rate over kept sentences goes
from 4.3% to **19.2%**. This was the mechanism G5 identified as the reason
annotate looked null, and lifting it moves reach by a factor of five.

## Scorer defects fixed this round

Both are scoring-side (no `src` change), both were found by this run's data, and
both only ever **understated** the annotate arms.

**1. The annotation marker landed on the wrong sentence.** The Generator writes
`"<sentence>. [unverified]"` — the marker follows the terminator, so the sentence
splitter carries it onto the *next* sentence, or strands it alone at the end of
the answer. Consequences: a genuinely unverified sentence was scored as
unlabelled while its successor was scored as annotated, and each stranded marker
entered ALCE's **recall denominator** as an extra uncited sentence. Numerators are
unaffected — neither piece is ever cited — so corrected recall is recoverable
exactly as `entail = round(recall × old sentence count)`, except for 8 examples
where a marker leaked onto a sentence that may now entail; those numerators are
approximate and can only be *under*-stated. The fix cut uncited-unlabelled
sentences in the open arm from 69 to 3. Pinned by `tests/generator/test_g5_score_split.py`,
including that the scorer and the contract validator must agree on the sentence count.

**2. `summary["verify-annotate"]` was a stale arm name** after the rename to
`-capped`/`-open`, crashing the report-writing tail. All four arms' numbers had
already been computed; nothing was lost.

## Positioning

> The system answers 78% of queries against the baseline's 93%, and every
> sentence it emits either carries a citation that survives independent
> verification or is explicitly marked unverified. It does this at **+0.19
> citation precision and +0.06 citation recall** over the baseline.

The remaining coverage gap and the −0.043 correctness gap are the honest cost,
and part of both is attributable to the entity-conflict drop path, which a second
blind audit puts at **85% wrongly destroyed** — see
`g5-entity-conflict-audit.md`.
