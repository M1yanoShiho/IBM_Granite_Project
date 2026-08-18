# Generator 修复与 Selector 分阶段协同跟踪表

**日期：** 2026-08-18
**计划：** [PLAN.md](PLAN.md)
**当前状态：** `G000 COMPLETE / PASS / G010-G100 READY / NO TRAINING STARTED`

用户已确认本路线的边界和协同顺序。G000 冻结通过前，不允许启动新训练、utility generation 或 held-out。

## 阶段 G：Generator

| ID | 阶段 | 目的 | 必需产物 | 状态 |
|---|---|---|---|---|
| G000 | Protocol freeze | 冻结数据、职责门、强结论门、预算和 fallback | frozen protocol、input manifest、denylist | COMPLETE / PASS |
| G010 | Power/scope | 计算样本可分辨效应，不用 observed power | MDE/sensitivity report | READY / NOT RUN |
| G100 | Citation attribution | 定位 G230 citation 最早失败阶段 | claim-level attribution rows | READY / NOT RUN |
| G110 | Independent audit | 复核归因和 MiniCheck disagreement | audit report、route decision | BLOCKED BY G100 |
| G120 | Splitter repair | 仅当 G110 证明 splitter 为主要断点 | tests、implementation report | CONDITIONAL |
| G130 | Routing repair | 仅当 G110 证明 routing/attachment 为主要断点 | tests、implementation report | CONDITIONAL |
| G200 | Data materialization | 构造 NIAH/2Wiki 原子 claim-citation 与 unsupported groups | train/model-val cases、manifest | BLOCKED BY G110 |
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
