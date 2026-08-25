# Report-ready result tables

The main report contains only these two tables. Metric names stay short; definitions appear in the notes.

## Table 1. End-to-end performance on three held-out datasets

*Values are percentages. Ours reports mean ± sample SD across the three frozen Generator checkpoints (seeds 13, 42, and 73) for Ans., Cit., and RAR; Ret. and Sel. are fixed single values. Baselines are run once.*

### (a) HotpotQA (n = 400)

| System | Ret. ↑ | Sel. ↑ | Ans. ↑ | Cit. ↑ | RAR ↑ |
|---|---:|---:|---:|---:|---:|
| Vanilla Granite RAG | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| Granite + Provence | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| **Ours** | XX.XX | XX.XX | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX |

### (b) MuSiQue answerable (n = 400)

| System | Ret. ↑ | Sel. ↑ | Ans. ↑ | Cit. ↑ | RAR ↑ |
|---|---:|---:|---:|---:|---:|
| Vanilla Granite RAG | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| Granite + Provence | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| **Ours** | XX.XX | XX.XX | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX |

### (c) RGB noise (n = 300)

| System | Ret. ↑ | Sel. ↑ | Ans. ↑ | Cit. ↑ | RAR ↑ |
|---|---:|---:|---:|---:|---:|
| Vanilla Granite RAG | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| Granite + Provence | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| **Ours** | XX.XX | XX.XX | XX.XX ± X.XX | XX.XX ± X.XX | XX.XX ± X.XX |

*Ret.: gold support recalled in the retrieved Top-10. Sel.: gold support retained in the final context. Ans.: token F1 for HotpotQA/MuSiQue and accuracy for RGB. Cit.: macro MiniCheck citation F1. RAR: correct answers with fully supporting valid citations.*

## Table 2. Component ablation across three held-out datasets

*All rows use the canonical Generator checkpoint (seed 13) and report one value; Full reuses the seed-13 run from Table 1. Values are percentages.*

### (a) HotpotQA (n = 400)

| Configuration | Ret. ↑ | Sel. ↑ | Ans. ↑ | Cit. ↑ | RAR ↑ |
|---|---:|---:|---:|---:|---:|
| **Full** | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| w/ Dense Retriever | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| w/ Top-10 | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| w/ Direct Generator | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |

### (b) MuSiQue answerable (n = 400)

| Configuration | Ret. ↑ | Sel. ↑ | Ans. ↑ | Cit. ↑ | RAR ↑ |
|---|---:|---:|---:|---:|---:|
| **Full** | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| w/ Dense Retriever | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| w/ Top-10 | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| w/ Direct Generator | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |

### (c) RGB noise (n = 300)

| Configuration | Ret. ↑ | Sel. ↑ | Ans. ↑ | Cit. ↑ | RAR ↑ |
|---|---:|---:|---:|---:|---:|
| **Full** | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| w/ Dense Retriever | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| w/ Top-10 | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| w/ Direct Generator | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |

*Metrics follow Table 1. Each non-Full row replaces only the named module; all other components remain unchanged.*

## Machine-readable files

Use these files as the numerical source:

| File | Purpose |
|---|---|
| `results/per_query_metrics.csv` | Paired per-query scores |
| `results/summary_metrics.csv` | Table 1 and Table 2 values |
| `results/bootstrap_ci.json` | Paired RAR differences and 95% CIs |

CSV rules: store proportions as 0--1; render percentages only in Markdown/LaTeX; keep `mean`, `std`, `variance`, `seed`, `dataset`, `system`, `metric`, `n_queries` and `missing_reason` as separate fields.
