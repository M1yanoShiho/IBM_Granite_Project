# Experiment 04 — Final frozen tables

## hotpotqa

| System | Ret. | Sel. | Ans. | Cit. | RAR |
|---|---:|---:|---:|---:|---:|
| Dense RAG | 100.00 | 100.00 | 26.77 | 46.25 | 3.00 |
| Hybrid RAG | 100.00 | 100.00 | 26.01 | 45.21 | 3.50 |
| Granite Rerank RAG | 100.00 | 100.00 | 27.12 | 46.95 | 4.00 |
| Provence RAG | 100.00 | 61.68 | 24.76 | 47.62 | 2.25 |
| Ours | 100.00 | 99.88 | 13.86 ± 0.36 | 65.70 ± 4.61 | 0.25 ± 0.00 |

## musique-answerable

| System | Ret. | Sel. | Ans. | Cit. | RAR |
|---|---:|---:|---:|---:|---:|
| Dense RAG | 87.98 | 87.90 | 15.89 | 27.88 | 1.00 |
| Hybrid RAG | 84.92 | 84.92 | 15.81 | 27.83 | 1.00 |
| Granite Rerank RAG | 85.75 | 85.75 | 17.29 | 33.12 | 1.00 |
| Provence RAG | 84.92 | 41.94 | 14.85 | 38.21 | 0.25 |
| Ours | 84.92 | 84.92 | 5.70 ± 1.08 | 38.03 ± 12.69 | 0.00 ± 0.00 |

## rgb-noise

| System | Ret. | Sel. | Ans. | Cit. | RAR |
|---|---:|---:|---:|---:|---:|
| Dense RAG | 46.67 | 46.25 | 0.00 | 80.46 | 0.00 |
| Hybrid RAG | 47.85 | 46.27 | 0.67 | 79.11 | 0.67 |
| Granite Rerank RAG | 52.58 | 49.14 | 0.33 | 81.76 | 0.33 |
| Provence RAG | 47.85 | 46.49 | 0.00 | 73.46 | 0.00 |
| Ours | 47.85 | 46.25 | 0.00 ± 0.00 | 90.69 ± 0.22 | 0.00 ± 0.00 |


## hotpotqa

| Configuration | Ret. | Sel. | Ans. | Cit. | RAR |
|---|---:|---:|---:|---:|---:|
| Full | 100.00 | 99.88 | 13.45 | 63.17 | 0.25 |
| w/ Dense Retriever | 100.00 | 99.88 | 13.92 | 66.47 | 0.25 |
| w/ Top-10 | 100.00 | 100.00 | 13.53 | 63.17 | 0.25 |
| w/ Direct Generator | 100.00 | 99.88 | 25.76 | 45.29 | 3.50 |

## musique-answerable

| Configuration | Ret. | Sel. | Ans. | Cit. | RAR |
|---|---:|---:|---:|---:|---:|
| Full | 84.92 | 84.92 | 6.60 | 41.53 | 0.00 |
| w/ Dense Retriever | 87.98 | 87.90 | 7.05 | 41.34 | 0.00 |
| w/ Top-10 | 84.92 | 84.92 | 6.60 | 41.53 | 0.00 |
| w/ Direct Generator | 84.92 | 84.92 | 15.81 | 27.83 | 1.00 |

## rgb-noise

| Configuration | Ret. | Sel. | Ans. | Cit. | RAR |
|---|---:|---:|---:|---:|---:|
| Full | 47.85 | 46.25 | 0.00 | 90.56 | 0.00 |
| w/ Dense Retriever | 46.67 | 46.22 | 0.00 | 89.47 | 0.00 |
| w/ Top-10 | 47.85 | 46.27 | 0.00 | 90.56 | 0.00 |
| w/ Direct Generator | 47.85 | 46.25 | 0.67 | 79.11 | 0.67 |

Full is the frozen Goal 3 Ours seed13 artifact; each non-Full row replaces only the named module.
