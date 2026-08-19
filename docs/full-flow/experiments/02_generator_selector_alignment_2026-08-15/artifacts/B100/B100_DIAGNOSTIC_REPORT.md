# B100 Controlled Context Diagnostic

**Status:** `COMPLETE`

## All matched diagnostic queries

| Arm | Answer | Coverage | Gold-doc citation precision | Visible-support citation recall | Reference visible but wrong |
|---|---:|---:|---:|---:|---:|
| K_topk | 60.09% | 89.91% | 82.57% | 31.76% | 87 |
| S_legacy_selected | 58.72% | 88.07% | 87.72% | 33.41% | 90 |
| O_support_only | 68.81% | 90.37% | 100.00% | 40.86% | 68 |
| OB_support_benign | 67.43% | 89.91% | 97.43% | 39.85% | 71 |
| OH_support_harmful | 65.14% | 88.53% | 98.20% | 38.45% | 76 |
| OP_support_last | 63.30% | 89.45% | 79.60% | 32.96% | 80 |

## Paired answer comparisons

| Comparison | n | Delta | 95% CI | Wrong to right | Right to wrong | Exact text equal |
|---|---:|---:|---:|---:|---:|---:|
| S_minus_K | 218 | -1.38% | [-4.31%, 1.45%] | 4 | 7 | 176/218 |
| O_minus_K | 218 | 8.72% | [3.10%, 14.69%] | 30 | 11 | 91/218 |
| OB_minus_O | 218 | -1.38% | [-5.68%, 2.83%] | 11 | 14 | 137/218 |
| OH_minus_O | 218 | -3.67% | [-8.37%, 0.88%] | 9 | 17 | 126/218 |
| OH_minus_OB | 212 | -2.36% | [-7.41%, 2.53%] | 11 | 16 | 112/212 |
| OP_minus_K | 218 | 3.21% | [-2.15%, 8.70%] | 21 | 14 | 98/218 |

This is an offline diagnostic matrix. Oracle arms are not deployable, and the six
answers are parallel experiment outputs rather than repeated answers in the formal system.
