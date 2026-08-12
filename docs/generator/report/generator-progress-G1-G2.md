# Generator — Progress Summary, G1–G2 (PARTLY INVALIDATED)

> **⚠ Every completeness figure here is invalid, including the 0.786 own-fact
> coverage below.**
>
> `required_facts` was built from the wrong ASQA field — background provenance
> snippets rather than the facts the gold answer states — so the completeness
> checker was scored against a target that was never the right one. That
> invalidates own-fact coverage, every false-gap rate derived from it, and
> `verified-full`'s coverage read as a completeness measure.
>
> Found by **human blind adjudication**, not by any metric: the numbers looked
> plausible throughout. The completeness loop was subsequently retired as a
> runtime mechanism (G4) and never returned.
>
> **What survives:** the G1 verifier selection (TRUE at threshold 0.50) and the
> adversarial entity-substitution slice, both of which are re-stated with
> provenance in `frozen-results.md` §1.
>
> Current numbers live in `local/report-writing/verified-numbers.md`.

Notice: G1, G2, etc. are task numbers

---

## G1 - Verifier selection (job 18200000, A100, ~3.4h, 1189 pairs/arm)

Six arms, identical pair set, calibration data only (ALCE/ASQA + 2WikiMultihopQA).

| Arm | Recall | Hard-neutral → entail FP | Derived citation precision |
|---|---|---|---|
| deberta-base | 0.467 | 0.007 | 0.946 |
| deberta-large | 0.540 | 0.073 | 0.648 |
| MiniCheck | 0.620 | 0.020 | 0.886 |
| **TRUE (11B)** | **0.747** | **0.007** | **0.966** |
| Granite-3B judge | 0.767 | 0.067 | 0.742 |
| Granite-8B judge | 0.900 | 0.240 | 0.484 |

Precision = `recall / (recall + FP×4)`, assuming 5 selected chunks with 1 genuine supporter.

**Conclusions**

1. **TRUE wins on both axes** — highest recall among usable arms and the cleanest false-positive rate. Its threshold sweep is usable (recall 0.593–0.880 vs FP 0.000–0.053); Granite's degenerates to near 0/1 under one-word output, so no threshold rescue is possible.
2. **The 8B control was decisive.** Scaling 3B→8B made FP *worse* (0.067 → 0.240), buying recall with looseness. This separates capacity from approach: the failure is hard-label LLM-as-a-judge, not model size.
3. **The earlier SciFact result (0.34 recall) was a model-family artefact, not a domain limit** — the same deberta-large scores 0.540 on ASQA.
4. Counterfactual slice, verifier alone: TRUE 0.963 > 3B 0.945 > 8B 0.872; all reach 1.000 with `entity_check`.

**Caveat:** the `ms/pair` column is not comparable across arms — CPU arms were carried over from the CPU round, GPU arms measured on A100. TRUE's real cost is deployment footprint (~21GB, GPU-only), not latency.

**Decision: TRUE adopted as main-path verifier at threshold 0.50. MiniCheck retained as CPU fallback.**

---

## G2 - TRUE adoption + first real-data run (job 18206324)

### Co-residency: proven, not assumed

The chain interleaves Granite (draft / completeness / recheck) and TRUE (attribution) **per query**, so it cannot be split into a generate-then-verify sequence. Both models resident peaked at **27.8GB of a 39.2GB MIG slice** — ~11GB headroom.

### completeness — the strongest component

| Metric | Value |
|---|---|
| Own-fact coverage | 0.786 |
| Foreign-fact correctly uncovered | 1.000 |
| Gap-question vagueness | 0.067 |
| Constraint-year survival | 0.423 |

The checker is **conservatively biased**: it never falsely claims coverage (1.000), but over-reports gaps (21% false gaps). That is the safer direction — a false gap wastes a recheck; a missed gap silently drops a required fact.

93% of gap questions are concrete and evidence-answerable, e.g. *"What is the make, model, and color of the car driven by Grace Kelly?"* — this was the component most at risk and it held up.

Constraint-year survival (0.423) is a lower bound depressed by synthetic construction, but needs re-checking against real checklists.

### full chain (49/60 completed, 11 errored)

| Diagnostic | Value |
|---|---|
| Supported / faithful claims | 23/69 (0.333) |
| Unsupported dropped by repair | 46/69 (0.667) |
| Entity mismatches caught | 47 |
| Gaps found / patched | 88 / 42 (0.477) |
| Repair fired | 33/49 (0.673) |
| Honest abstention | 28/49 (0.571) |
| `GenerationResult` contract | 49/49 |
| Latency | 9.6 s/example |

`contradicted` is 0 as expected under a binary backend; documented as *not computed* in `docs/generator/verifier-backends.md` per the agreed annotation.

**Repair fires on two-thirds of queries** — the earlier concern that it would rarely trigger on real data is resolved in the opposite direction.

---

## The open question

**0.333 support and 57% abstention have two opposite readings, and G2 cannot distinguish them:**

- **The method is working** — the draft genuinely contained that much unsupported content, and the verifier caught it.
- **The method is over-rejecting** — TRUE is dropping claims the evidence does support, deleting true statements and abstaining unnecessarily.

A sanity check: if ~80% of draft claims were genuinely supported, TRUE's calibrated recall of 0.747 predicts ~60% observed support. The measured 33% is well below that, so either draft quality is poor or in-chain recall is materially worse than the calibration figure. G1 pairs used human-written claims against annotated supporting passages; in-chain, claims are model-generated and evidence is whatever the Selector returned — different distributions.

This blocks the next decision: loosen the threshold, or keep the strict operating point and present high abstention as an intentional trade.

---

## Current work — blind manual audit

A ~40-item adjudication packet, human-judged:

- **Blind** — no verdicts, no supporting-evidence ids, evidence not score-ordered (ordering leaks the verdict).
- **Stratified** — dropped claims split by failure mode: NLI no-entailment vs entity-layer veto. Different causes, different remedies, reported separately.
- **With controls** — ~10 system-supported claims mixed in, yielding **in-chain citation precision measured rather than extrapolated** from G1's 0.966.
- **Three-way** — supported / not supported / unclear, plus a per-item reason. The `unclear` bucket captures claims that are neither hallucinated nor single-chunk verifiable, which is direct evidence about decomposition quality.

Adjudication is human by design: using a model to check a model's verdicts would not be independent and could not go in the report.

**Prerequisite:** confirm `results/verified-generator/cases.jsonl` from job 18206324 has been pushed back from bp1 — `results/` currently shows only `e1-cluster-eval`.

---

## Next steps

1. **Run the audit**  → decide whether to loosen the operating point or defend strict abstention.
2. **Fix `claim_splitter`'s verbatim-substring contract** — it requires `source_text` to appear verbatim in `answer_text`, but Granite paraphrases. Accounts for 7 of 11 chain errors. Note the 11 failures are *not* random: they skew toward heavier paraphrasing, so the completed 49 may be systematically easier.
3. **Run the baseline comparison** — generation-time citation on the same queries. Without it, 57% abstention reads as failure; with it, the trade (fewer answers, far better citations) becomes presentable. **This is the headline experiment and it has not been run.**
