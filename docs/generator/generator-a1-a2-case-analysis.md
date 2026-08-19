# Generator A1/A2 case-level paired analysis and failure taxonomy

> Analysis date: 2026-08-19  
> Source: frozen `annotation_unblinded.csv`, 639/639 validated rows  
> Status: **exploratory post-hoc analysis**; this document supplements rather than replaces
> `generator-a1-a2-ablation-results.md`

## 1. Why this analysis was added

The frozen G-A12 report uses claim- or sentence-weighted proportions. Those are the correct headline
results for the preregistered descriptive evaluation, but they do not show whether a change was spread
across many questions or concentrated in a few. This follow-up therefore gives every eligible query equal
weight, counts improved/unchanged/regressed cases, and audits the annotated failure types.

The analysis does not introduce a new primary hypothesis. Query-level bootstrap intervals use 10,000
resamples with replacement and seed `20260819`; they are uncertainty diagnostics, not preregistered
significance tests.

## 2. Pairing and metrics

- A1 compares old and new answers by `query_id`.
- A2 compares `old_old` with `old_new`, and `new_old` with `new_new`. Each pair receives the same A1
  answer, so the splitter remains the only intended treatment difference.
- For sentence/claim metrics, a per-query proportion is computed after excluding `NA`; only queries with a
  defined value in both arms enter that metric.
- Coverage retains its original `none/partial/complete` transition and additionally uses an exploratory
  ordinal score: `none=0`, `partial=0.5`, `complete=1`. `NA` pairs are excluded.
- A negative delta is an improvement only for `unresolved_pronoun`; higher is better elsewhere.

## 3. Case-level paired results

### 3.1 A1 prompt

| Metric | Paired n | Mean case delta | Exploratory 95% bootstrap interval | Improved / unchanged / regressed |
|---|---:|---:|---:|---:|
| Atomic | 43 | +7.8 pp | [−0.8, +16.3] | 8 / 34 / 1 |
| Self-contained | 50 | +28.5 pp | [+20.2, +37.2] | 25 / 24 / 1 |
| Independently verifiable | 50 | +20.5 pp | [+12.9, +28.1] | 24 / 23 / 3 |
| Unresolved pronoun ↓ | 49 | −8.4 pp | [−13.4, −4.1] | 11 / 38 / 0 |

This strengthens the original interpretation of A1. The self-containment and verifiability gains were not
created by one or two unusually long answers: approximately half of the questions improved and very few
regressed. The atomicity change is less certain and is dominated by unchanged cases.

The human failure labels tell the same story at sentence level, while requiring caution because old A1
produced more sentences than new A1:

| Error type | Old A1 | New A1 |
|---|---:|---:|
| Any annotated error | 58/103 = 56.3% | 15/60 = 25.0% |
| Citation-only fragment | 23/103 = 22.3% | 6/60 = 10.0% |
| Meta/non-claim | 15/103 = 14.6% | 7/60 = 11.7% |
| Multi-fact sentence | 9/103 = 8.7% | 2/60 = 3.3% |
| Unresolved reference | 15/103 = 14.6% | 0/60 = 0.0% |
| Vague/unverifiable | 2/103 = 1.9% | 0/60 = 0.0% |

The residual A1 failures show that a prompt is a behavioural constraint rather than a guarantee. For
example, the new answer to “Movie with the song sunshine on my shoulders?” combined the broadcast year
and song-use fact in one sentence, so it was still labelled `multi_fact`.

### 3.2 A2 splitter with old A1 fixed

| Metric | Paired n | Mean case delta | Exploratory 95% bootstrap interval | Improved / unchanged / regressed |
|---|---:|---:|---:|---:|
| Atomic | 42 | −1.6 pp | [−7.1, +2.4] | 1 / 40 / 1 |
| Self-contained | 43 | −1.6 pp | [−8.5, +5.4] | 1 / 39 / 3 |
| Span aligned | 43 | +2.9 pp | [−2.5, +9.7] | 4 / 37 / 2 |
| Rewrite faithful | 43 | +5.2 pp | [−1.6, +13.4] | 5 / 36 / 2 |
| Coverage ordinal | 45 | +3.3 pp | [0.0, +7.8] | 3 / 42 / 0 |

Coverage improved only through three `partial → complete` transitions; 42/45 eligible cases were unchanged
and none regressed. This supports a narrow statement: the new splitter recovered omitted source facts in a
small subset, rather than improving every query.

For example, in “When does Disney's food and wine festival end?”, old A2 represented only the November 12
fact, whereas new A2 preserved both November 13 and November 12 facts from the same frozen answer.

### 3.3 A2 splitter with new A1 fixed

| Metric | Paired n | Mean case delta | Exploratory 95% bootstrap interval | Improved / unchanged / regressed |
|---|---:|---:|---:|---:|
| Atomic | 48 | 0.0 pp | [0.0, 0.0] | 0 / 48 / 0 |
| Self-contained | 49 | +0.7 pp | [0.0, +2.0] | 1 / 48 / 0 |
| Span aligned | 49 | −1.4 pp | [−6.1, +2.0] | 1 / 47 / 1 |
| Rewrite faithful | 49 | −3.4 pp | [−10.2, +1.4] | 1 / 46 / 2 |
| Coverage ordinal | 48 | +1.0 pp | [0.0, +3.1] | 1 / 47 / 0 |

The coverage change is one `partial → complete` transition, for the adipose-tissue question. All other
eligible coverage cases were unchanged. The claim metrics are almost entirely ties. This is consistent with
the interpretation that the verification-oriented A1 already produces a splitter-friendly answer shape,
leaving limited headroom for A2.

## 4. Macro versus micro aggregation

The frozen report pools all annotated sentences or claims, so a query with more claims receives more weight.
This follow-up first computes a proportion inside each query and then averages queries equally. The two views
answer different questions:

- micro: “What fraction of all annotated units passed?”
- case-level macro: “How much did the average eligible query change?”

The distinction matters most for A2 on new-A1 answers. The frozen claim-weighted result reported span
alignment `+1.2 pp` and rewrite faithfulness `−0.4 pp`; the case-weighted results are `−1.4 pp` and
`−3.4 pp`. This is not a calculation contradiction. It means queries produce unequal numbers of claims and
the few changed queries receive different weights. Neither view supports a broad A2 win; both support the
original “partial descriptive support” conclusion.

## 5. Failure taxonomy

The taxonomy is multi-label: a row can be both `span_misaligned` and `meaning_changed`, so category counts
must not be added to obtain a total.

### 5.1 Definitions

| Failure type | Operational meaning in this annotation set |
|---|---|
| `citation_only` | A standalone citation marker was emitted as a sentence/unit. |
| `meta_nonclaim` | Text discusses evidence, sources or the answer rather than stating the answer fact. |
| `multi_fact` | One sentence/claim combines separable facts. |
| `unresolved_reference` | A pronoun or generic noun does not identify its referent when read alone. |
| `vague_unverifiable` | The statement is too vague to test independently. |
| `span_misaligned` | The recorded source span is insufficient or not the correct source for the claim. |
| `meaning_changed` | The rewritten claim adds, removes or changes meaning relative to the displayed span. |
| `fact_omission` | The A2 output does not completely cover the source facts used by the case annotation. |

### 5.2 Distribution

- A1 error-labelled sentence rows fell from 56.3% to 25.0%, especially through removing unresolved
  references and reducing citation-only fragments.
- On old-A1 answers, A2 rows with any error fell from 32/74 (43.2%) to 30/77 (39.0%); fact-omission cases
  fell from 12/50 (24.0%) to 9/50 (18.0%).
- On new-A1 answers, A2 rows with any error were essentially unchanged: 17/61 (27.9%) versus 18/64
  (28.1%); fact omissions fell from 7/50 (14.0%) to 6/50 (12.0%).
- `meaning_changed` and `span_misaligned` remain the dominant A2 labels. Their frequent co-occurrence shows
  why faithful rewriting and span anchoring should be analysed together rather than treated as independent
  failures.

The row-rate comparison is descriptive because the arms contain different claim counts. The paired table is
the better source for causal interpretation.

## 6. Identical visible-input consistency audit

Many A2 outputs were exactly identical across old/new splitter arms. For every exact match on question,
claim text and displayed source span, the human task was visibly identical; any label difference therefore
cannot be an A2 effect.

| Fixed A1 answer | Metric | Identical visible items | Agreement |
|---|---|---:|---:|
| Old A1 | Atomic | 71 | 69/71 = 97.2% |
| Old A1 | Self-contained | 72 | 68/72 = 94.4% |
| Old A1 | Span aligned | 72 | 69/72 = 95.8% |
| Old A1 | Rewrite faithful | 72 | 68/72 = 94.4% |
| New A1 | Atomic | 59 | 59/59 = 100.0% |
| New A1 | Self-contained | 61 | 60/61 = 98.4% |
| New A1 | Span aligned | 61 | 59/61 = 96.7% |
| New A1 | Rewrite faithful | 61 | 58/61 = 95.1% |

Agreement is high, but the 2.8–5.6% disagreement range on several metrics is comparable to the small A2
effect sizes. Examples include identical Jason Gideon text/span changing from failed to passed span alignment,
and identical Jacob text/span changing in the opposite direction. These should not be narrated as splitter
successes or failures.

This audit makes the existing limitation more precise: a second annotator is still needed for true
inter-annotator agreement, while the duplicate audit reveals non-zero repeat inconsistency within the current
single-annotator data. The correct response is not to relabel after seeing the arms, but to keep the frozen
results and narrow the A2 claim.

## 7. Paper-ready interpretation

> A query-level paired analysis supported the stability of the A1 result: self-containment improved in 25 of
> 50 questions and independent verifiability in 24 of 50, with only one and three regressions respectively.
> A2 changes were much more local. With old A1 answers fixed, span alignment improved in four cases and
> regressed in two, while source-fact coverage improved through three partial-to-complete transitions and did
> not regress. With new A1 answers fixed, almost all claim-level cases were unchanged and only one coverage
> case improved. Exact duplicate-input auditing found 94.4–100% repeat agreement depending on metric and
> condition; the remaining annotation instability was similar in scale to several A2 deltas. Accordingly, the
> follow-up strengthens the conclusion that A1 produced a broad improvement, while A2 delivered targeted
> fixes whose overall magnitude remains small and uncertain.

## 8. Reporting limits

1. This is a post-hoc exploratory analysis of the same 50 ASQA calibration cases, not a new held-out test.
2. Bootstrap intervals quantify case-sampling uncertainty but do not address single-annotator measurement
   error and must not be called preregistered significance tests.
3. Coverage scoring imposes an ordinal spacing between `none`, `partial` and `complete`; transition counts
   should accompany the mean score.
4. Failure rows and claims are not independent observations, and failure labels are multi-label.
5. The analysis evaluates A1/A2 structure, not downstream NLI, citation precision, answer correctness,
   abstention or end-to-end RAG performance.

## 9. Artifacts

The derived files are under `results/generator-a1-a2-ablation/case-analysis/`:

- `a1_a2_case_analysis.xlsx`: formatted, auditable workbook;
- `case_level_pairs.csv`: one row per eligible query/metric comparison;
- `paired_summary.csv`: case-level mean deltas, intervals and direction counts;
- `failure_taxonomy.csv`: multi-label error counts and denominators;
- `duplicate_input_audit.csv` and `duplicate_input_summary.csv`: identical-input consistency audit;
- `representative_cases.csv`: automatically selected candidates for manual narrative review;
- `analysis_manifest.json`: analysis parameters and row counts.

The reproducible analysis entry point is `scripts/analyze_a1_a2_cases.py`.
