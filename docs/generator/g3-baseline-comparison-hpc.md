# G3 baseline comparison — verified vs generation-time citation

Tests the method's central claim: **citations produced by post-hoc verification
are more faithful than the citations the model declares while generating.** Three
arms on the same 400 ASQA queries, same Granite-4.1-3b, same selected evidence;
only the post-generation treatment varies. Judge: **MiniCheck** (not TRUE, not
TRUE-derived) — the verified arms' citations were selected by TRUE, so scoring
with TRUE would be circular. Job 18238790, seed 13.

**Hold-out respected:** ALCE/ASQA only. HotpotQA, RGB, MuSiQue-Full never loaded.

## Three axes (reported together)

| arm | coverage | answer correctness (STR-EM) | cite precision (answered) | cite recall (answered) | answered |
|---|---|---|---|---|---|
| baseline (`GraniteGenerator`) | **0.932** | **0.273** | 0.602 | 0.646 | 373/400 |
| verify-only (draft→verify→repair) | 0.552 | 0.189 | **0.762** | **0.862** | 218/395 |
| verified-full (+ completeness/recheck) | 0.331 | 0.128 | 0.575 | 0.736 | 125/378 |

Baseline is not always-answer: its own `_is_unknown_answer` path abstains on
**6.8%** of queries. Citation numbers are on answered queries only, so the
head-to-head is the paired both-answered subset below — a single precision number
across different answer sets would be a misrepresentation.

## Paired, both-answered subset (MiniCheck, paired randomization, bootstrap CI)

`evidence_rag.evaluation.paired_metric_cli`, seed 13; `citation_*` is None on
abstained queries so the pairing restricts to both-answered automatically.

| comparison | metric | mean (on) | mean (off) | delta | p | 95% CI | n |
|---|---|---|---|---|---|---|---|
| **verify-only vs baseline** | citation precision | **0.764** | 0.674 | **+0.090** | **0.010** | [+0.021, +0.158] | 215 |
| verify-only vs baseline | citation recall | 0.860 | 0.730 | +0.130 | 0.0002 | [+0.065, +0.195] | 215 |
| **verified-full vs baseline** | citation precision | 0.578 | 0.685 | **−0.107** | **0.034** | [−0.207, −0.009] | 123 |
| verified-full vs baseline | citation recall | 0.732 | 0.748 | −0.016 | 0.88 | [−0.114, +0.081] | 123 |
| verified-full vs verify-only | citation precision | 0.594 | 0.823 | −0.228 | 0.0 | [−0.313, −0.148] | 94 |
| verified-full vs verify-only | citation recall | 0.798 | 0.947 | −0.149 | 0.0006 | [−0.234, −0.074] | 94 |
| verified-full vs verify-only | coverage | 0.331 | 0.542 | −0.212 | 0.0 | [−0.270, −0.153] | 378 |
| verified-full vs verify-only | answer correctness | 0.128 | 0.185 | −0.057 | 0.0 | [−0.080, −0.034] | 378 |

## Reading

### The central claim holds — but for the faithfulness half only

**verify-only (verify + repair) significantly beats generation-time citation**:
citation precision **0.764 vs 0.674, +0.090, p=0.010, CI[+0.021, +0.158]**, and
recall +0.130, p=0.0002, on the both-answered subset. Post-hoc verification *does*
produce more faithful citations than the model declares while generating — the
mechanism works.

### …and the full system fails the pre-registered criterion

**verified-full does NOT beat baseline** — citation precision **0.578 vs 0.685,
−0.107, p=0.034**. This meets the failure criterion registered in the ledger: the
full method's citations are, if anything, *less* precise than the baseline's. The
method as a whole is not supported; the *faithfulness component* is.

### The ablation earns its keep: the completeness/recheck loop is a net negative

Adding completeness/recheck to verify-only loses on **every axis** — citation
precision −0.228 (p=0), recall −0.149 (p=0.0006), coverage −0.212 (p=0), answer
correctness −0.057 (p=0). The recheck step appends answer fragments whose
citations MiniCheck does not support, diluting precision, while the
missing-required-fact abstention roughly halves coverage. Without this ablation
the −0.107 vs baseline would have been blamed on "verification"; it is actually
the completeness loop. **Disable or redesign the completeness/recheck loop**; the
value is in verify-only.

### The precision gain is bought with coverage and correctness

verify-only's +0.090 citation precision comes at a large cost: coverage 0.552 vs
baseline 0.932, correctness 0.189 vs 0.273. This is the honest trade-off — the
verified path answers fewer questions and gets fewer of them right, in exchange
for more faithful citations when it does answer. Presented alone, either number
misleads; the method's real position is "more faithful citations on a smaller,
self-selected set of answers."

### entity_check fix (report item 6)

Run on the fixed `entity_check` (tolerant name matching). Validated against the 15
labelled stratum-B audit items: **8/9 false vetoes now pass entity**, the true
swap (Sobers≠Gooch) still vetoes, one borderline flip (Victoria, evidence does
name her). Without this fix the verify arms would have dropped many more supported
claims and understated the method.

## Caveats

- MiniCheck is weaker than TRUE (G1 recall 0.620 vs 0.747), so absolute citation
  levels are a conservative floor; it is applied identically to all arms, so the
  *comparisons* are fair. A ~40-item human-adjudicated subsample
  (`results/g3/human-subsample.jsonl`, blind across baseline/verified-full) is to
  be scored against these MiniCheck numbers; if they disagree materially the human
  number governs.
- ASQA calibration only; the held-out confirmation runs once after the system is
  frozen.
- Answer-level citation metrics (each cited chunk vs the whole answer), not
  ALCE per-sentence, because arms attach citations at answer granularity.
