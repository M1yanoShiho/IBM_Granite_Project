# QAMPARI — module-level held-out results

Jobs `18325888` (dry run) + `18326078` (generation, 01:44:26) + `18326079`
(scoring, 00:27:26), five arms in one job, all exit 0. 400 queries, seed 13,
top-5 GTR passages. Judge MiniCheck; TRUE is the production verifier and never
judges.

Pre-registered in `heldout-preregistration.md` at commit `c559d5c`,
**2026-08-09 15:57:20 +0100**, before any QAMPARI byte was read. Loaded once. No
system change follows.

---

## The pre-registered criterion, first

> The calibration result replicates if, on queries answered by both arms,
> nogate's citation precision exceeds baseline's at p < 0.05, **and** nogate's
> coverage deficit against baseline is not significantly worse than the
> calibration value of −0.0152.

| conjunct | result | verdict |
|---|---|---|
| citation precision > baseline, p < 0.05 | **+0.2685** (0.8499 vs 0.5813), **p = 0.0**, CI [0.198, 0.336], n = 233 | **met, decisively** |
| coverage deficit not worse than −0.0152 | **−0.0157**, p = 0.469, CI [−0.050, +0.018], n = 382 | met as measured — **but see below** |

**The precision conjunct replicates and is not in doubt.** The calibration gain
of +0.19 becomes **+0.27** on an untouched dataset with a different task shape.

**The coverage conjunct cannot be settled on this set**, and the
pre-registration says so in advance rather than in hindsight.

## The claim-splitter clause fires

> If claim-splitter failure on the held-out set is materially above the
> calibration rate of roughly 0.5%, coverage comparisons are reported as
> unreliable and the failure rate is stated with the results.

**Failure rate: 18/400 = 4.50%**, against 0.5% (G8) and 1.25% (G9) — nine times
calibration. All are `LLM output must be valid JSON` from the splitter. So
**coverage comparisons on QAMPARI are reported as unreliable.**

The clause was right to fire, and the reason is measurable rather than
precautionary:

| | |
|---|---|
| baseline answered **17/18 (0.944)** of the failed queries | |
| baseline answered **306/382 (0.801)** of everything else | |

The splitter sits upstream of every verify arm, so its failures remove a query
from all of them while leaving the **baseline** scored on it. Those queries are
ones the baseline handles *better* than average, so excluding them — which is what
pairing correctly does — removes ground where the baseline was winning.

| treatment | nogate | baseline | deficit |
|---|---|---|---|
| paired, failures excluded (reported) | 300/382 = 0.785 | 306/382 = 0.801 | **−0.0157** |
| failures counted as unanswered (bound) | 300/400 = 0.750 | 323/400 = 0.808 | **−0.0575** |

−0.0157 passes the criterion; −0.0575 would not. **The true value lies between
them and this run cannot say where.** The honest reading is that the coverage
claim is untested here, not that it passed.

The failures are **not** concentrated on many-answer questions — median gold
answers 7.5 for the failures against 8.0 for the rest — so this is a higher base
rate of malformed JSON on a new distribution, not the entity-list hypothesis the
dry run raised.

---

## Five arms

Correctness is **answer recall by containment**, the pre-registered metric. It is
**not comparable to ASQA's STR-EM** and is recall-only.

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

**Citation recall also replicates**: +0.093 against calibration's +0.071.

**Correctness does not.** On ASQA nogate was indistinguishable from baseline
(+0.0015, p=0.892); here it is **behind** by −0.0105 (p=0.025), and rec@5 by
−0.014 (p=0.101, not significant). Reported as it stands. Two things bear on it,
neither offered as an explanation that removes the result:

- the metric is recall-only, so a verbose enumerating answer is rewarded and
  never penalised, which favours the arm that asserts most;
- absolute correctness is very low for every arm (0.05–0.12) because five
  passages cannot contain a median of eight gold entities, so the whole axis is
  compressed.

The gate still costs coverage for nothing: nogate over open is +0.065 coverage
and +0.007 correctness with citation metrics unchanged (p = 0.35, p = 0.87) —
the same shape as calibration.

## Routing and the review flag

441 claims → 364 verified, 77 annotated, **0 destroyed**. The gate would have
destroyed **48**; all 48 are cited instead. Control self-check exact
(`gate_would_drop` = 48 = `dropped_entity_conflict`). Declared-citation survival
310/400 = 0.775, higher than calibration's 0.709.

**The review flag does not generalise.**

| | error rate | n | |
|---|---|---|---|
| flagged | 0.184 | 44 | |
| unflagged | 0.145 | 202 | |

**Lift 1.28×**, against **2.86×** on calibration. On QAMPARI the flagged cohort is
barely distinguishable from the rest. The screening claim is calibration-specific
and must not be stated as a general property of the entity layer.

## What replicates and what does not

| claim | ASQA (calibration) | QAMPARI (held-out) | |
|---|---|---|---|
| citation precision > baseline | +0.191 (p=0.0) | **+0.269 (p=0.0)** | **replicates** |
| citation recall > baseline | +0.071 (p=0.009) | **+0.093 (p=0.010)** | **replicates** |
| gate costs coverage, buys no citation quality | +0.127 cov, ns citation | +0.065 cov, ns citation | **replicates** |
| nothing destroyed | 0 | 0 | **holds** |
| coverage ≈ baseline | −0.0152 (p=0.386) | −0.0157 (p=0.469) | **untestable here** — splitter failure 4.5% |
| correctness ≈ baseline | +0.0015 (p=0.892) | −0.0105 (p=0.025) | **does not replicate** |
| review-flag enrichment | 2.86× | 1.28× | **does not replicate** |

The headline the project rests on — **citations that survive independent
verification, at materially higher precision and recall than generation-time
citation** — replicates on an untouched dataset with a different task shape, and
by a larger margin than on calibration.

The two things that do not replicate are reported as not replicating.
