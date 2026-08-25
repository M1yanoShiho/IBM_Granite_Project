# F001 独立逐句引用评分

**裁判：** MiniCheck；系统内部 TRUE 不参与评分。

| 范围 | 组别 | Answered | Citation precision | Precision (实际有引用) | Citation recall |
|---|---|---:|---:|---:|---:|
| 全部 (739) | A_topk_basic | 674 | 66.95% | 66.95% | 73.27% |
| 全部 (739) | B_selector_basic | 669 | 67.51% | 67.51% | 73.64% |
| 全部 (739) | C_topk_verify | 650 | 75.17% | 86.94% | 74.28% |
| 全部 (739) | D_selector_verify | 646 | 74.78% | 87.20% | 73.97% |
| Selector 改动题 (109) | A_topk_basic | 100 | 70.12% | 70.12% | 74.70% |
| Selector 改动题 (109) | B_selector_basic | 95 | 74.25% | 74.25% | 77.37% |
| Selector 改动题 (109) | C_topk_verify | 96 | 82.81% | 91.38% | 82.29% |
| Selector 改动题 (109) | D_selector_verify | 92 | 80.43% | 93.67% | 80.43% |

基础组只有答案级引用，因此把全部引用宽松地分给每句话；高级组使用运行时保存的精确句子—引用对应。
ALCE precision 将“回答了但没有任何引用”的样本记为 0；实际有引用样本的 precision 同时单独报告。
