# 状态对照

> 当时没有独立 tracker；这是根据 execution notes 和最终报告形成的归档对照，不冒充 contemporaneous preregistration。

| 阶段 | 状态 | 结果 |
|---|---|---|
| 代码实施与回归 | COMPLETE | 相关测试、类型与代码检查通过 |
| sealed600 候选池与审计 | COMPLETE | 600 题，571 题包含 harmful 候选 |
| 2Wiki 候选池与审计 | COMPLETE | 2,000 题 |
| Reliability-MIS 正式比较 | COMPLETE / FAIL | harmful 保留下降，但 required recall `−37.17pp` |
| 2Wiki 保护门 | FAIL | supporting recall `−11.19pp` |
| 稳定性 | PASS AS REPRODUCIBILITY | 60/60 重复一致，只证明失败可复现 |
| 生产注册 | REJECTED | TopK 保留 |
