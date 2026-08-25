# G5 — verify-and-annotate, three-arm calibration

ALCE sentence-level citation metrics, judge **MiniCheck** (TRUE is the production verifier and never judges). Annotated sentences carry no citation, so they are outside the precision denominator and inside the recall denominator -- the intended, visible cost of the redesign.

| arm | coverage | correctness (STR-EM) | cite precision | cite recall | answered |
|---|---|---|---|---|---|
| baseline | 0.932 | 0.273 | 0.613 | 0.647 | 373/400 |
| verify-only | 0.633 | 0.197 | 0.871 | 0.871 | 252/398 |
| verify-annotate | 0.633 | 0.196 | 0.890 | 0.868 | 252/398 |

Annotation rate: **15/313** (0.048) of kept sentences carry the unverified marker.


## Paired comparisons — both arms under identical citation conventions

Round 2. `verify-only` now records its own per-sentence citation mapping, so the
flat-list convention no longer depresses it. Randomization + bootstrap CI.

| comparison | metric | delta | p | 95% CI | n |
|---|---|---|---|---|---|
| **verify-annotate vs verify-only** | cite precision | **+0.024** | **0.27** | [−0.017, +0.065] | 237 |
| verify-annotate vs verify-only | cite recall | +0.001 | 0.98 | [−0.043, +0.046] | 237 |
| verify-annotate vs verify-only | correctness | −0.001 | 0.89 | [−0.014, +0.011] | 398 |
| verify-annotate vs verify-only | coverage | 0.000 | 1.00 | — | 398 |
| verify-annotate vs baseline | cite precision | **+0.188** | **<0.0001** | [+0.131, +0.244] | 248 |
| verify-only vs baseline | cite precision | **+0.173** | **<0.0001** | [+0.109, +0.237] | 247 |
| verify-annotate vs baseline | cite recall | **+0.133** | **<0.0001** | [+0.078, +0.188] | 248 |
| verify-only vs baseline | cite recall | **+0.138** | **<0.0001** | [+0.075, +0.202] | 247 |
| verify-annotate vs baseline | correctness | −0.077 | <0.0001 | [−0.103, −0.052] | 398 |
| verify-only vs baseline | correctness | −0.076 | <0.0001 | [−0.101, −0.051] | 398 |

### What the convention correction did

Equalising the conventions moved `verify-only` from 0.760 to **0.871**: the
flat-list fallback had been charging it for citations it never attached to a
given sentence, through ALCE's redundancy ablation. The round-1 gap of +0.070
over verify-only was therefore **substantially a scoring artifact**, exactly as
anticipated. The gap against the baseline did not shrink, because both verified
arms rose once their citations were attached correctly.

### The redesign makes no measurable difference

`verify-annotate` is statistically indistinguishable from `verify-only` on
**every axis** — precision p=0.27, recall p=0.98, correctness p=0.89, coverage
identical. Annotate-versus-delete, measured properly, buys nothing here.

The mechanism is structural, and it is the more useful finding: **82 claims were
annotated, but only 15 reach an answer.** A case with zero verified claims
abstains outright, so its annotated content is discarded with it. Annotation can
therefore only add content to cases that *already* had a verified claim — which
are exactly the cases that were already answered. The zero-verified abstention
rule, which exists to keep the `GenerationResult` contract, caps the redesign's
reach at 4.8% of kept sentences.

### What both verified arms do buy

Against generation-time citation, post-hoc verification raises citation precision
by **+0.17 to +0.19** and recall by **+0.13**, both p<0.0001, at a cost of
**−0.30 coverage** and **−0.077 correctness**. That trade is the real result;
the delete-vs-annotate refinement is not.

## Entity layer after the fix

Drops did **not** become rare: **66 of 439 routed claims (15.0%)**, against
72/490 (14.7%) before. The composition changed completely, though — every trigger
is now a real NER type, where the old extractor was firing on `name:some` and
`name:season`:

| type | organization | location | person | product | number | date |
|---|---|---|---|---|---|---|
| triggers | 29 | 18 | 13 | 12 | 4 | 3 |

**Aliasing residue (as requested):** of 78 value-conflicts, **9 (11.5%)** share a
token with an evidence value — the morphological/alias shape the fix cannot reach
without an alias table ("west germany" against "federal republic of germany").
The other **69 (88.5%)** are fully disjoint from every same-role evidence value,
which is the shape of a genuine swap.

**Caveat, stated plainly:** disjointness is a *lexical* proxy for "genuine
conflict", not a verdict. The 0.700 false-veto rate came from adjudicating the
*old* extractor's population, which is no longer representative. Whether the
false-veto rate actually fell needs a fresh audit of this population; it should
not be assumed from the composition change alone.
