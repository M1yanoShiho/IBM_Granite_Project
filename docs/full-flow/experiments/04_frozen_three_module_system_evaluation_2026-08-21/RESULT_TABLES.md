# Report-ready result tables

The main report contains only these two tables. Metric names stay short; definitions appear in the notes.

## Table 1. End-to-end performance on three held-out datasets

*Values are percentages. Ours reports mean ± sample SD across three independently trained Generator checkpoints (seeds 13, 42, and 73) for Ans., Cit., and RAR; Ret. and Sel. are fixed single values. Deterministic baselines are run once. Paired 95% CIs for ΔRAR use 10,000 component-cluster bootstrap resamples.*

### (a) HotpotQA (n = 400)

| System | Ret. ↑ | Sel. ↑ | Ans. ↑ | Cit. ↑ | RAR ↑ |
|---|---:|---:|---:|---:|---:|
| Dense RAG | 100.00 | 100.00 | 26.77 | 46.25 | 3.00 |
| Hybrid RAG | 100.00 | 100.00 | 26.01 | 45.21 | 3.50 |
| Granite Rerank RAG | 100.00 | 100.00 | 27.12 | 46.95 | 4.00 |
| Provence RAG | 100.00 | 61.68 | 24.76 | 47.62 | 2.25 |
| **Ours** | 100.00 | 99.88 | 13.86 ± 0.36 | 65.70 ± 4.61 | 0.25 ± 0.00 |

### (b) MuSiQue answerable (n = 400)

| System | Ret. ↑ | Sel. ↑ | Ans. ↑ | Cit. ↑ | RAR ↑ |
|---|---:|---:|---:|---:|---:|
| Dense RAG | 87.98 | 87.90 | 15.89 | 27.88 | 1.00 |
| Hybrid RAG | 84.92 | 84.92 | 15.81 | 27.83 | 1.00 |
| Granite Rerank RAG | 85.75 | 85.75 | 17.29 | 33.12 | 1.00 |
| Provence RAG | 84.92 | 41.94 | 14.85 | 38.21 | 0.25 |
| **Ours** | 84.92 | 84.92 | 5.70 ± 1.08 | 38.03 ± 12.69 | 0.00 ± 0.00 |

### (c) RGB noise (n = 300)

| System | Ret. ↑ | Sel. ↑ | Ans. ↑ | Cit. ↑ | RAR ↑ |
|---|---:|---:|---:|---:|---:|
| Dense RAG | 46.67 | 46.25 | 0.00 | 80.46 | 0.00 |
| Hybrid RAG | 47.85 | 46.27 | 0.67 | 79.11 | 0.67 |
| Granite Rerank RAG | 52.58 | 49.14 | 0.33 | 81.76 | 0.33 |
| Provence RAG | 47.85 | 46.49 | 0.00 | 73.46 | 0.00 |
| **Ours** | 47.85 | 46.25 | 0.00 ± 0.00 | 90.69 ± 0.22 | 0.00 ± 0.00 |

*Ret.: gold support recalled in the retrieved Top-10. Sel.: gold support retained in the final context. Ans.: token F1 for HotpotQA/MuSiQue and accuracy for RGB. Cit.: macro MiniCheck citation F1. RAR: correct answers with fully supporting valid citations.*

## Table 2. Component ablation across three held-out datasets

*Full and the Retriever/Selector ablations use the canonical GR-C checkpoint (seed 13); the Direct Generator ablation uses frozen base Granite. Full reuses the seed-13 run from Table 1. Values are percentages.*

### (a) HotpotQA (n = 400)

| Configuration | Ret. ↑ | Sel. ↑ | Ans. ↑ | Cit. ↑ | RAR ↑ |
|---|---:|---:|---:|---:|---:|
| **Full** | 100.00 | 99.88 | 13.45 | 63.17 | 0.25 |
| w/ Dense Retriever | 100.00 | 99.88 | 13.92 | 66.47 | 0.25 |
| w/ Top-10 | 100.00 | 100.00 | 13.53 | 63.17 | 0.25 |
| w/ Direct Generator | 100.00 | 99.88 | 25.76 | 45.29 | 3.50 |

### (b) MuSiQue answerable (n = 400)

| Configuration | Ret. ↑ | Sel. ↑ | Ans. ↑ | Cit. ↑ | RAR ↑ |
|---|---:|---:|---:|---:|---:|
| **Full** | 84.92 | 84.92 | 6.60 | 41.53 | 0.00 |
| w/ Dense Retriever | 87.98 | 87.90 | 7.05 | 41.34 | 0.00 |
| w/ Top-10 | 84.92 | 84.92 | 6.60 | 41.53 | 0.00 |
| w/ Direct Generator | 84.92 | 84.92 | 15.81 | 27.83 | 1.00 |

### (c) RGB noise (n = 300)

| Configuration | Ret. ↑ | Sel. ↑ | Ans. ↑ | Cit. ↑ | RAR ↑ |
|---|---:|---:|---:|---:|---:|
| **Full** | 47.85 | 46.25 | 0.00 | 90.56 | 0.00 |
| w/ Dense Retriever | 46.67 | 46.22 | 0.00 | 89.47 | 0.00 |
| w/ Top-10 | 47.85 | 46.27 | 0.00 | 90.56 | 0.00 |
| w/ Direct Generator | 47.85 | 46.25 | 0.67 | 79.11 | 0.67 |

*Metrics follow Table 1. Each non-Full row replaces only the named module; all other components remain unchanged.*

## Machine-readable files

Use these files as the numerical source:

| File | Purpose |
|---|---|
| `results/per_query_metrics.csv` | Paired per-query scores |
| `results/table1.json` | Machine-readable Table 1 values |
| `results/TABLE1.md` | Frozen rendered Table 1 |
| `results/per_query_goal4_metrics.csv` | Goal 4 Table 2 paired per-query scores |
| `results/table2.json` | Machine-readable Table 2 values |
| `results/TABLE2.md` | Frozen rendered Table 2 |
| `results/summary_metrics.csv` | Table 1 and Table 2 values |
| `results/bootstrap_ci.json` | Paired RAR differences and 95% CIs |
| `results/bootstrap_goal4_ci.json` | Full-minus-ablation paired RAR differences and 95% CIs |
| `results/goal3_audit.json` | Goal 3 coverage and hash audit |
| `results/goal4_audit.json` | Goal 4 coverage, reuse, and hash audit |
| `results/final_results.json` | Combined frozen tables, bootstrap results, hashes, and claim decisions |
| `results/final_audit.json` | Final cross-artifact statistical and hash audit |

CSV rules: store proportions as 0--1; render percentages only in Markdown/LaTeX; keep `mean`, `std`, `variance`, `seed`, `dataset`, `system`, `metric`, `n_queries` and `missing_reason` as separate fields.
