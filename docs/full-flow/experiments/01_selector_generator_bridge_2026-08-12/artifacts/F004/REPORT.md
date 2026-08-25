# F004：G0/G1/G2 关键事实笔记消融

| 组 | Answer match | Coverage | 有笔记题数 | 错误 |
|---|---:|---:|---:|---:|
| G0_current | 55.96% | 84.40% | 0 | 0 |
| G1_notes | 57.80% | 84.40% | 37 | 0 |
| G2_guided_notes | 57.80% | 84.40% | 38 | 0 |

| 比较 | Answer delta | 95% CI | wrong→right | right→wrong |
|---|---:|---:|---:|---:|
| G1_minus_G0 | +1.83 pp | [-2.61, +6.42] | 4 | 2 |
| G2_minus_G0 | +1.83 pp | [-2.48, +6.48] | 4 | 2 |
| G2_minus_G1 | +0.00 pp | [-3.57, +3.77] | 2 | 2 |

G0 复用 F001 的冻结 D 输出；G1/G2 在同一作业共享 Granite 和 TRUE。
只有 G2 明确优于 G1，才支持 Selector 信号具有独立作用。
