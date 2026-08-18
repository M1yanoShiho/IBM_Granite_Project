# Generator grounding repair 执行跟踪表

**日期：** 2026-08-18
**计划：** [PLAN.md](PLAN.md)
**当前状态：** `PLANNED / WAITING FOR PLAN REVIEW`

计划写成不等于已经授权训练。只有 R000 冻结并通过后，才允许开始新的实现或服务器运行。

| ID | 阶段 | 目的 | 主要输入 | 必需产物 | 状态 |
|---|---|---|---|---|---|
| R000 | Protocol freeze | 冻结边界、比较、数据角色、指标和预算 | G230、A000、现有 manifests | frozen protocol、input manifest、held-out denylist | TODO |
| R010 | Power/scope audit | 计算现有样本可确认的 MDE，冻结主张强度 | G230 discordance、各 split 数量 | power/MDE report | TODO |
| R100 | Citation attribution | 用已有 G230 trace 定位 citation 损失最早发生在哪一步 | G230 generations、routing、MiniCheck rows | claim-level attribution table | TODO |
| R110 | Independent audit | 复核自动归因和 MiniCheck 误判风险 | R100 分层样本 | audit report、route decision | TODO |
| R120 | Splitter repair | 仅当 R110 判定 downstream splitter 为主责任时修复 | R100/R110 | tests、implementation report | CONDITIONAL |
| R130 | Routing repair | 仅当 R110 判定 TRUE routing/final attachment 为主责任时修复 | R100/R110 | tests、implementation report | CONDITIONAL |
| R200 | Data materialization | 构造跨 NIAH/2Wiki 的 claim-citation 与 unsupported 训练组 | allowed train splits | train/model-val cases、manifest | BLOCKED BY R110 |
| R210 | Target audit | 审计每个事实、引用和 split，冻结最终训练数据 | R200 | target audit、leakage report、data freeze | BLOCKED BY R200 |
| R300 | Training implementation | 实现 query-group loss、citation token weighting 和 warm start | frozen R210 data | tests、smoke manifest | BLOCKED BY R210 |
| R310 | Seed-13 screen | 比较 fresh joint training 与 G230 continuation | R210、Granite base、GM13 | two adapters、model-val report | BLOCKED BY R300 |
| R320 | Recipe freeze | 在未打开 qualification bundle 前选择唯一训练配方 | R310 model-val only | frozen recipe、tie-break trace | BLOCKED BY R310 |
| R330 | Three-seed fit | 使用唯一配方训练 seeds 13/42/73 | R320 | 3 adapters、training manifests | BLOCKED BY R320 |
| R400 | Locked dev generation | 一次运行 NIAH qualification 与 stress matrix | R330、G0 frozen inputs | generations、answer/coverage report | BLOCKED BY R330 |
| R410 | Cross-dataset gate | 一次运行 2Wiki 和已揭示 citation regressions | R330、frozen cross-data inputs | per-dataset reports | BLOCKED BY R330 |
| R420 | Independent citation/statistics | MiniCheck、paired CI、McNemar 和 family-level gate | R400/R410 | citation rows、statistics、gate decision | BLOCKED BY R400/R410 |
| R430 | Generator freeze | 通过全部门才安装唯一 `G*`；否则记录 no candidate | R420 | G* manifest 或 STOP report | BLOCKED BY R420 |
| R500 | Existing-plan handoff | `G*` 通过后解锁第 02 路线 S300；不在本路线重写 Selector | R430 G* | handoff manifest | CONDITIONAL |

## 当前禁止事项

- [ ] 不使用 sealed600。
- [ ] 不读取 HotpotQA、MuSiQue-Full、RGB 的逐题内容、答案或分数。
- [ ] 不在 qualification bundle 上比较多个新配方。
- [ ] 不挑单个最好 seed。
- [ ] 不把重复上下文行当成独立样本扩大统计量。
- [ ] 不同时改变 Retriever、Selector、draft、splitter 和 TRUE。
- [ ] 不删除、还原或提交用户无关文件。

## Git/GitHub 同步纪律

每个可独立复核的阶段完成后：

1. 先检查远端分支是否有团队新提交；如有，正常拉取并合并，不建立临时 worktree，不 reset/stash 覆盖用户文件；
2. 只提交本阶段明确列出的代码、协议和归档产物；
3. 运行相应测试和 artifact/hash 检查；
4. 推送 `origin/refactor/three-module-baseline`；
5. 在本 tracker 记录 commit、服务器 run root、产物 SHA256 和阶段判定。
