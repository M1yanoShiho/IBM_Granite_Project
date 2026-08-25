# 状态记录

当时没有单独 tracker；计划本身定义 M0–M5。最终状态如下：

| 阶段 | 状态 | 证据 |
|---|---|---|
| M0 数据与代码冻结 | PASS | `results/selector-beam-v1/m0/` |
| M1 32题正确性 | PASS | `results/selector-beam-v1/m1/` |
| M2 seed13 开发评估 | FAIL | 16 个 threshold cell 合格数 0 |
| M3 三 seed | CUT / NOT RUN | M2 stop gate |
| M4 sealed600/2Wiki 正式测试 | CUT / NOT RUN | M2 stop gate |
| M5 cutover | COMPLETE | 失败代码退役，TopK 保留 |

M1 的高训练准确率只证明管线能学习 sanity 样本，不等于 Selector 有效。
