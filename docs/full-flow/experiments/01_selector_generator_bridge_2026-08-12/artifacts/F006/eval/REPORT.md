# F006 LoRA 开发集结果

| 组 | Answer match | Coverage | 有笔记题数 |
|---|---:|---:|---:|
| TopK_frozen | 58.72% | 88.07% | 0 |
| Base_notes_frozen | 57.80% | 84.40% | 37 |
| Clean_LoRA | 58.72% | 85.32% | 99 |
| Mixed_LoRA | 61.47% | 85.32% | 100 |

| 比较 | Answer delta | 95% CI | wrong→right | right→wrong |
|---|---:|---:|---:|---:|
| Clean_minus_Base | +0.92 pp | [-6.96, +9.26] | 9 | 8 |
| Mixed_minus_Base | +3.67 pp | [-3.67, +11.54] | 10 | 6 |
| Mixed_minus_Clean | +2.75 pp | [-2.00, +8.18] | 5 | 2 |
| Mixed_minus_TopK | +2.75 pp | [-6.03, +12.17] | 13 | 10 |

**F005 gate：** PASS
