# R005A/R005B v2 纠错与重新审计报告

**日期：** 2026-08-12

**当前结论：** `PASS（设计审计 P0=0，P1=0，P2=0） / WAITING FOR USER APPROVAL / NOT IMPLEMENTED / NOT RUN`

## 1. 为什么需要 v2

在把计划逐条映射到现有代码时，两名独立只读 reviewer 发现：此前 v1 草案的最终审计漏掉了四项实现阻断。为了不让一个已知自相矛盾的方案进入 A001，v1 被保留为 `SUPERSEDED DRAFT / NEVER APPROVED / NEVER RUN`，没有回写成“通过”。

| v1 遗漏 | 风险 | v2 修正 |
|---|---|---|
| `FIT_TERMINAL=COMPLETE` 写在 checkpoint reload/threshold freeze 前 | terminal 无法绑定尚未生成的 artifact | 唯一顺序改为 checkpoint→strict reload→fit scores→threshold table/trace→gate→closure→terminal；终态后只准零变更 verify-only |
| 四个 ordered projection 只有名称，没有固定路径与逐文件 schema | 实现者可用不同字段或目录内容拼出“等价”输入 | A/B 各固定 full+四projection共五个 absolute literal path、object envelope、manifest type、exact fields、projection rule与三处绑定 |
| B-fit 未反向绑定 `R005A_TO_R005B.json` SHA | B job 可能只引用 A bundle而绕过正式授权事件 | B anchor/claim/STARTED都直接绑定相同authorization path/SHA、A attempt与V*；A-lock→B-job-lock后才可STARTED |
| A001 要求重跑旧真实 runner `--verify-only` | 旧 runner在verify分支前读取含train-modelval的旧输入 | A001只跑synthetic/fixture旧finalizer零写回归；历史正式复验引用现有attestation，真实runner不得在A001重跑 |

这些都是未批准草案中的协议缺陷，不是新实验科学结果失败；没有代码、样本或训练需要回滚，也没有 held-out 被揭示。

## 2. 当前稳定快照

| 文件 | Git blob |
|---|---|
| `EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v2.md` | `f6adbb7d2c4a9e8fcd91e800fcb6f7d3f43d4eab` |
| `EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v2.json` | `a1691e886b91a57d2b8e304711b255070c9aaa7d` |
| `EXPERIMENT_TRACKER_AMENDMENT_2026-08-12_R005AB_v2.md` | `6b5e0228062d3b6532cbd8329b9c7f171e72d476` |

v2 没有改变四角色 `64/96/64/128`、预冻结 assignment hashes、V0→V1→V2、61/64、92/96、122/128、两 seed one-shot confirm、最多5 jobs/150 epochs/2 GPU-hours或TopK10默认。它只收紧可实现性与防绕过合同。

## 3. 两路独立复审

### 状态机对抗复审

- 结论：`P0=0，P1=0，P2=0`；
- 确认 terminal 顺序、五个 ordered manifests、B authorization/锁序、旧回归边界、recovery/veto与五件链均闭合；
- trace：`.aris/traces/experiment-audit/2026-08-12_r005ab_v2_state_machine/`。

### 跨文档一致性复审

- 结论：`P0=0，P1=0，P2=0`；
- 确认 MD/JSON/tracker/固定入口的数值、状态、路径与链接一致，v1 不再是当前执行入口；
- trace：`.aris/traces/experiment-audit/2026-08-12_r005ab_v2_consistency/`。

审计 PASS 只表示 v2 可以提交用户批准，不表示 A001 已实现，更不表示模型会通过 R005A/R005B。

## 4. A001 代码映射

[`R005AB_A001_IMPLEMENTATION_MAP_2026-08-12.md`](R005AB_A001_IMPLEMENTATION_MAP_2026-08-12.md) 已把 v2 逐条落到现有仓库：旧四文件保持 Git blob 不变；新建独立 versioned model、materializer、durable state、bundle与semantic verifier模块；冻结 assignment/candidate/strict-pair/batch/score schemas；列出 synthetic、防泄漏、并发、崩溃与mutation测试矩阵。

## 5. 现在的许可边界

当前唯一合法下一步仍是 A000：用户明确批准 v2 amendment。批准前：

- 不实现 scorer/state/materializer；
- 不同步服务器工作树到新实现提交；
- 不运行 A002；
- 不物化新样本；
- 不训练；
- 不读取 A-screen/B-confirm。

批准后只先执行 A001；A001 通过、clean commit推送且本地/GitHub/服务器同commit以前，A002仍不获授权。
