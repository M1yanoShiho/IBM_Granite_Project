# F001 现有三模块四臂联合实验

**状态：** `COMPLETE`

## 总体结果

| 范围 | 组别 | Answer match | Coverage | Gold-doc citation precision | Gold-doc citation recall |
|---|---|---:|---:|---:|---:|
| 全部 decision-dev (739) | A_topk_basic | 65.90% | 91.20% | 80.09% | 32.03% |
| 全部 decision-dev (739) | B_selector_basic | 65.49% | 90.53% | 81.89% | 31.97% |
| 全部 decision-dev (739) | C_topk_verify | 64.01% | 87.96% | 87.01% | 22.21% |
| 全部 decision-dev (739) | D_selector_verify | 63.60% | 87.42% | 88.63% | 22.70% |
| Selector 改变上下文 (109) | A_topk_basic | 59.63% | 91.74% | 70.75% | 38.54% |
| Selector 改变上下文 (109) | B_selector_basic | 56.88% | 87.16% | 82.91% | 38.17% |
| Selector 改变上下文 (109) | C_topk_verify | 58.72% | 88.07% | 79.31% | 27.51% |
| Selector 改变上下文 (109) | D_selector_verify | 55.96% | 84.40% | 89.87% | 30.86% |

## Selector 对答案的影响

| 范围 | 比较 | Delta | 95% CI | wrong→right | right→wrong |
|---|---|---:|---:|---:|---:|
| 全部 | B_minus_A | -0.406 pp | [-1.233, +0.389] pp | 3 | 6 |
| 全部 | D_minus_C | -0.406 pp | [-1.307, +0.435] pp | 4 | 7 |
| 全部 | interaction | +0.000 pp | [-1.157, +1.221] pp | — | — |
| 改动题 | B_minus_A | -2.752 pp | [-8.333, +2.703] pp | 3 | 6 |
| 改动题 | D_minus_C | -2.752 pp | [-8.621, +2.970] pp | 4 | 7 |
| 改动题 | interaction | +0.000 pp | [-7.843, +8.333] pp | — | — |

Gold-doc citation precision/recall 是生成后使用官方相关文档计算的诊断指标，
不是逐句 NLI 引用指标，也没有进入 Generator 运行时。
