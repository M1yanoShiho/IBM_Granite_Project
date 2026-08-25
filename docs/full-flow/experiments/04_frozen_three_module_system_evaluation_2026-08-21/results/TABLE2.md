# Experiment 04 — Table 2

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
