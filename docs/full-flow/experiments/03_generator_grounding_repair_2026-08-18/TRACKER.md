# Generator 修复与 Selector 分阶段协同跟踪表

**日期：** 2026-08-18
**计划：** [PLAN.md](PLAN.md)
**当前状态：** `G130 COMPLETE / PASS / G200 READY / NO TRAINING STARTED`

用户已确认本路线的边界和协同顺序。G130 已通过；G200/G210 通过前，不允许启动新训练、utility generation 或 held-out。

## 阶段 G：Generator

| ID | 阶段 | 目的 | 必需产物 | 状态 |
|---|---|---|---|---|
| G000 | Protocol freeze | 冻结数据、职责门、强结论门、预算和 fallback | frozen protocol、input manifest、denylist | COMPLETE / PASS |
| G010 | Power/scope | 计算样本可分辨效应，不用 observed power | MDE/sensitivity report | COMPLETE / PASS |
| G100 | Citation attribution | 定位 G230 citation 最早失败阶段 | claim-level attribution rows | COMPLETE / PASS |
| G110 | Independent audit | 复核归因和 MiniCheck disagreement | audit report、route decision | COMPLETE / PASS |
| G120 | Splitter repair | 仅当 G110 证明 splitter 为主要断点 | tests、implementation report | NOT ACTIVATED |
| G130 | Routing repair | 仅当 G110 证明 routing/attachment 为主要断点 | tests、implementation report | COMPLETE / PASS |
| G200 | Data materialization | 构造 NIAH/2Wiki 原子 claim-citation 与 unsupported groups | train/model-val cases、manifest | READY / NOT RUN |
| G210 | Target audit | 审计 support、citation、split、人工样本并冻结数据 | audit、leakage report、hashes | BLOCKED BY G200 |
| G300 | Training implementation | query-group loss、citation weighting、fresh/continuation | tests、smoke manifest | BLOCKED BY G210 |
| G310 | Seed13 screen | 比较 GR-F 与 GR-C | two adapters、model-val report | BLOCKED BY G300 |
| G320 | Recipe freeze | 按 maximin 冻结唯一 Generator 配方 | recipe、tie-break trace | BLOCKED BY G310 |
| G330 | Three-seed fit | 唯一配方训练 seeds 13/42/73 | adapters、training manifests | BLOCKED BY G320 |
| G400 | Locked NIAH qualification | 生成 NIAH full/stress/unsupported | generations、answer report | BLOCKED BY G330 |
| G410 | Cross-data qualification | 生成 2Wiki 和已揭示 citation regression | per-dataset outputs | BLOCKED BY G330 |
| G420 | Generator gate | 技术门、职责门、tripwire 与强结论分开判定 | statistics、gate report | BLOCKED BY G400/G410 |
| G430 | Teacher freeze | 新候选通过则 GQ=new；否则 GQ=G0 fallback | teacher manifest | BLOCKED BY G420 |

## 阶段 S：Utility Selector

| ID | 阶段 | 目的 | 必需产物 | 状态 |
|---|---|---|---|---|
| S100 | Utility pilot | 用冻结 GQ 做 100 题 full/leave-one-out 一致性检查 | pilot labels、budget report | BLOCKED BY G430 |
| S110 | Utility materialization | 扩展 NIAH/2Wiki MUST_KEEP/SAFE_DROP/NEUTRAL | utility dataset、manifest | BLOCKED BY S100 PASS |
| S200 | Utility implementation | legacy safety + generator utility，cap 保持 2 | code、tests、smoke | BLOCKED BY S110 |
| S210 | Seed13 screen | 只用 model-val 冻结唯一 utility 配方/阈值 | screen report、recipe | BLOCKED BY S200 |
| S220 | Three-seed fit | 训练/复现 SU seeds 13/42/73 | checkpoints、manifests | BLOCKED BY S210 |
| S300 | Selector qualification | 同一 GQ 下比较 TopK、SL、SU | evidence/answer/citation gate report | BLOCKED BY S220 |
| S310 | Selector freeze | 主目标 SQ=SU；仅 SL 单独通过时 SQ=SL fallback；否则 STOP | selector manifest 或 no-candidate report | BLOCKED BY S300 |

## 阶段 I/H：完整系统

| ID | 阶段 | 目的 | 必需产物 | 状态 |
|---|---|---|---|---|
| I100 | Retriever input freeze | 从真实 frozen Retriever 入口产生共同候选 | pool manifest、visibility report | BLOCKED BY S310 SQ |
| I200 | Locked full-flow dev | A=TopK+G0、B=TopK+GQ、C=SL+GQ、D=SQ+GQ；重复臂复用 | generations、scores | BLOCKED BY I100 |
| I210 | System responsibility gate | 检查 D-A 总作用、D-B Selector 净作用和安全 | gate report | BLOCKED BY I200 |
| I220 | System freeze | 冻结 Retriever+SQ+GQ 为 SystemF | SystemF manifest、all hashes | BLOCKED BY I210 PASS |
| H100 | One-time heldout | 分别运行 HotpotQA、MuSiQue-Full、RGB | per-dataset results、claim matrix | CONDITIONAL / REQUIRES USER AUTHORIZATION |

## 当前禁止事项

- [ ] 不使用 sealed600。
- [ ] 不在 SystemF 冻结前读取或评分 HotpotQA、MuSiQue-Full、RGB。
- [ ] 不同时更新 Generator 和 Selector。
- [ ] utility labels 生成后不修改 GQ。
- [ ] 不把 Legacy Selector 的 NIAH harmful precision 写成跨数据能力。
- [ ] 不把模块职责通过写成统计显著优越。
- [ ] 不要求每个次级 slice 的 CI 都显著才能进入下一阶段。
- [ ] 不挑单个最好 seed。
- [ ] 不把 context/leave-one-out rows 当作独立问题。
- [ ] 不删除、还原或提交用户无关文件。

## Git/GitHub 同步纪律

每个可独立复核阶段完成后：

1. fetch 并核对远端；团队有新提交时正常 pull/merge；
2. 不建立临时 worktree，不 reset/stash 覆盖用户文件；
3. 只提交该阶段代码、协议和归档产物；
4. tests、hash、报告一致后 push；
5. 在本 tracker 记录 commit、server run root、SHA256 和阶段判定。

## 已完成记录

- G000：协议、统计范围、预算、fallback、runtime gold 边界和 denylist 已冻结；服务器 `it097952` 上 repo/runtime/G200/G220/G230 实体核验 PASS；G230 gzip 归档解压 SHA256 与 runtime manifest 一致；sealed600 标记为 retired read-only，HotpotQA/MuSiQue-Full/RGB/RGB-counterfactual 只保留 ordered IDs/hash 且仍需 SystemF 后单独授权；未启动训练、utility labels 或 held-out。产物位于 `artifacts/G000/`，报告见 `G000_REPORT.md`；本阶段提交为包含本记录的 Git commit，push 后以 GitHub 历史为准。
- G010：只使用 G230 archived answer/citation rows 和 G000 frozen sample sizes 做 simulation-style sensitivity；未计算 observed power，未放宽任何 gate/margin，未读取 held-out 内容。G230 full TopK 为 739 rows / 534 components，design effect 为 1.3839；按 G230 family mean discordance，`correct_and_cited` MDE 约 GC 6.15pp、GM 6.33pp。产物位于 `artifacts/G010/G010_POWER_SCOPE.json`，报告见 `G010_POWER_SCOPE.md`。
- G100：只使用 G230 archived traces、answer rows 和 MiniCheck citation rows，生成自动 claim-level attribution candidates；共同 answered full TopK tasks 为 596，claim-level regression rows 为 501。自动分布为 TRUE/routing/attachment 268、MiniCheck evaluator disagreement 219、draft citation missing/wrong 8、splitter boundary/rewrite 6、unsupported draft 0；这不是最终修复决策，G110 必须独立审计后才能激活 G120/G130。产物位于 `artifacts/G100/`，报告见 `G100_ATTRIBUTION_REPORT.md`。
- G110：按固定分层样本完成 120 条主审和 24 条双审；双审一致 22/24，agreement rate 91.67%。最终标签为 TRUE/routing/attachment 44、MiniCheck evaluator disagreement 37、splitter boundary/rewrite 22、draft citation missing/wrong 8、unsupported draft 9；因此条件激活 G130 routing/attachment repair，G120 不激活。G110 未启动训练、utility labels 或 held-out；产物位于 `artifacts/G110/`，报告见 `G110_AUDIT_REPORT.md`。
- G130：完成一次共享 deterministic routing/attachment 修复；TRUE hypothesis 现在使用最终展示给用户的 citation-stripped sentence，trace 和 G230 routing export 显式记录 `routing_hypothesis`、`declared_verified`、`rescued_by_scan`、`review_flagged` 和 `attachment_verified`。G110 revealed diagnostic 中 TRUE/routing 44 条拆分为 33 条真无附件、11 条已有附件但带 observe-only gate warning；修复不改变 TRUE 模型/阈值、Retriever、Selector、gold/reference runtime 边界、held-out 或训练。产物位于 `artifacts/G130/G130_ROUTING_REPAIR_SUMMARY.json`，报告见 `G130_ROUTING_REPAIR_REPORT.md`。
