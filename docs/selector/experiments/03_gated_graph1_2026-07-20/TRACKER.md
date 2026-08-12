# 状态对照

> 这是根据 S1–S6 记录整理的归档对照；当时没有一份单独覆盖整个 Graph 1.0 路线的最终 tracker。

| 阶段 | 状态 | 关键结论 |
|---|---|---|
| Gated hard-drop | COMPLETE | harmful-in-context 约 `−11.2pp` |
| Required recall 保护门 | FAIL | required recall 约 `−4.8pp`，超过允许的 `−1pp` |
| Lenient clustering | PARTIAL RECOVERY | 约回收 `+1.2pp` recall，仍无法闭合保护门 |
| S4/S5 抽取探针 | COMPLETE | 发现部分早期归因错误并主动推翻 |
| S6 真实池级联 | NEGATIVE | false-conflict 明显恶化，强抽取方案不采用 |
| 生产注册 | NOT ALLOWED | TopK 保留 |

最终状态：`COMPLETE / SAFETY GATE FAIL / RETIRED`。
