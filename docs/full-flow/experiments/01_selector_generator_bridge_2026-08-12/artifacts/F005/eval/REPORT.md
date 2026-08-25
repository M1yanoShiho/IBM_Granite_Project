# F005 独立最终确认结果

| 组 | Answer match | Coverage |
|---|---:|---:|
| A_TopK_Base | 63.17% | 90.00% |
| B_TopK_Mixed | 63.33% | 88.00% |
| C_Selector_Mixed | 63.17% | 87.50% |

| 比较 | Answer delta | 95% CI | 错→对 | 对→错 |
|---|---:|---:|---:|---:|
| B_minus_A_generator_only | +0.17 pp | [-2.68, +3.15] | 42 | 41 |
| C_minus_B_selector_only | -0.17 pp | [-0.84, +0.50] | 2 | 3 |
| C_minus_A_full_system | +0.00 pp | [-2.84, +2.86] | 40 | 40 |

- Selector 改变问题：81 / 600
- 删除证据：82 条
- 删除已知有害证据：74 / 524
- 删除 gold 相关文档：3 条

**点估计目标：** FAIL
**统计确认：** NOT CONFIRMED
