# Experiment 04 — Final report

**Final decision:** `FINAL PASS`  
**Date:** 2026-08-22

## Executive conclusion

The frozen three-module evaluation is technically valid and fully traceable. Goal 3
produced 7,700 formal answers and Goal 4 produced 3,300 new ablation answers, for 11,000
formal answer generations in total. Table 2 reuses 1,100 Goal 3 Full seed13 rows and does
not count them as new generations.

The primary scientific claim that Ours improves RAR is not supported. Ours has higher
citation scores than the deterministic baselines, but lower answer performance and lower
RAR. The ablation result localizes the principal observed weakness to the frozen GR-C
seed13 Generator: replacing it with frozen Direct Granite improves RAR by 3.25 percentage
points on HotpotQA (Full-minus-Direct 95% CI [-5.25, -1.50]) and by 1.00 point on MuSiQue
([-2.00, -0.25]). The RGB direction is similar but its CI includes zero. Dense Retriever
and Top-10 substitutions do not change RAR in these held-out samples.

These conclusions describe the frozen systems and datasets only. They do not authorize
post-held-out tuning or a revised method claim.

## Execution validity

- All six formal dataset jobs across Goals 3 and 4 completed with exit code 0.
- All generation bundles and scorer manifests are PASS.
- Every frozen dataset-arm query set is complete and uses the common denominator.
- Generation was gold-free; scorer-only sidecars were read only after generation hashes
  were frozen.
- The single HotpotQA Goal 4 Dense-Retriever runtime failure remains in the denominator
  (0.25%, below the preregistered 1% invalidation guard).
- No seed was selected: Table 1 uses all three independent GR-C training seeds 13/42/73.
- Goal 4 Full is byte-equivalent at the metric-row level to frozen Goal 3 Ours seed13 and
  was not regenerated.

## Statistical interpretation

Table 1 reports mean and sample SD across the three GR-C training seeds only for Ours
Ans./Cit./RAR; fixed upstream Ret./Sel. and deterministic baselines remain point estimates.
All RAR comparisons use the paired component-cluster percentile bootstrap with 10,000
resamples and seed 13. Goal 3 differences are Ours three-seed per-query mean minus the
baseline; Goal 4 differences are Full minus the named ablation.

The valid negative findings are:

1. Ours does not exceed any Table 1 baseline on RAR in the frozen evaluation.
2. Removing the NLI Selector (Top-10) does not alter RAR on any of the three datasets.
3. Replacing Hybrid retrieval with Dense retrieval does not alter RAR on any dataset.
4. Direct Granite exceeds Full GR-C seed13 RAR on HotpotQA and MuSiQue; RGB is only a
   directional signal because its interval includes zero.

## Final artifacts

- Frozen report tables: [`final_tables.md`](../../results/experiment04/final_tables.md)
- Unified long-form metrics: [`summary_metrics.csv`](../../results/experiment04/summary_metrics.csv)
- Combined machine result: [`final_results.json`](../../results/experiment04/final_results.json)
- LaTeX tables: [`table1.tex`](../../results/experiment04/table1.tex),
  [`table2.tex`](../../results/experiment04/table2.tex)
- Final machine audit: [`final_audit.json`](../../results/experiment04/final_audit.json)
- Goal 3 evidence: [`experiment04-main-results.md`](experiment04-main-results.md)
- Goal 4 evidence: [`experiment04-ablation-results.md`](experiment04-ablation-results.md)

## Final boundary

Experiment 04 is complete. No new held-out answers, tuning, seed selection, or method
changes were performed in Goal 5.
