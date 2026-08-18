# Generator 修复与 Selector 分阶段协同跟踪表

**日期：** 2026-08-18
**计划：** [PLAN.md](PLAN.md)
**当前状态：** `G216 PRE-MANUAL PASS / G212R2 REQUIRED / NO TRAINING STARTED`

用户已确认本路线的边界和协同顺序。G210 hard data gate 已失败；用户已进一步授权按积极信号原则修订计划。G215 将失败解释为“原数据冻结失败但路线可修”；G200R2 已完成 answer-alias-preserving materialization；G210R2 已完成 structural、TRUE 和 pre-manual finalize，自动数据门通过；G212 length gate 发现唯一 overlength train group；G214 已成组排除该 train group 并保持数据门通过；G212R length gate 已通过；G212M 固定 100 条 sample review/adjudication 为 92 PASS / 8 FAIL / 0 UNCERTAIN；G216 已排除失败样本对应 11 个 case，预冻结数据门仍通过。下一步必须执行 G212R2 revised length/sample review；仍不允许在修复后 freeze readiness 通过前启动训练、utility generation 或 held-out。

## 阶段 G：Generator

| ID | 阶段 | 目的 | 必需产物 | 状态 |
|---|---|---|---|---|
| G000 | Protocol freeze | 冻结数据、职责门、强结论门、预算和 fallback | frozen protocol、input manifest、denylist | COMPLETE / PASS |
| G010 | Power/scope | 计算样本可分辨效应，不用 observed power | MDE/sensitivity report | COMPLETE / PASS |
| G100 | Citation attribution | 定位 G230 citation 最早失败阶段 | claim-level attribution rows | COMPLETE / PASS |
| G110 | Independent audit | 复核归因和 MiniCheck disagreement | audit report、route decision | COMPLETE / PASS |
| G120 | Splitter repair | 仅当 G110 证明 splitter 为主要断点 | tests、implementation report | NOT ACTIVATED |
| G130 | Routing repair | 仅当 G110 证明 routing/attachment 为主要断点 | tests、implementation report | COMPLETE / PASS |
| G200 | Data materialization | 构造 NIAH/2Wiki 原子 claim-citation 与 unsupported groups | train/model-val cases、manifest | COMPLETE / PRE-AUDIT PASS |
| G210 | Target audit | 审计 support、citation、split、人工样本并冻结数据 | audit、leakage report、hashes | FAIL / HARD DATA GATE |
| G215 | Gate/data amendment | 解释硬门依据，按积极信号授权受控数据修订 | amendment report、manifest | COMPLETE / PASS |
| G200R | Revised data materialization | 修订 2Wiki target construction/audit candidate，重建数据 | train/model-val cases、manifest | COMPLETE / PRE-AUDIT PASS |
| G210R-v1 | Revised target audit | 对 G200R-v1 重新执行 structural 审计 | structural rows、failure summary | FAIL / STRUCTURAL ANSWER_ALIAS |
| G200R2 | Revised data materialization | 过滤 answer alias 不保留的 2Wiki support-sentence targets | train/model-val cases、manifest | COMPLETE / PRE-AUDIT PASS |
| G210R2 | Revised target audit | 对 G200R2 重新执行 structural/TRUE/pre-manual finalize | audit、leakage report、hashes | COMPLETE / PRE-MANUAL PASS |
| G212 | Sample/length audit | 对 G210R2 pre-manual 数据做固定样本、长度/truncation 和最终冻结前检查 | sample packet、length report、freeze readiness manifest | FAIL / LENGTH |
| G214 | Controlled length repair | 成组排除 G212 发现的唯一 overlength 2Wiki train group 及对应 unsupported case | revised pre-manual bundle、manifest、ordered IDs | COMPLETE / PRE-MANUAL PASS |
| G212R | Revised sample/length audit | 对 G214 bundle 重跑长度审计和 fixed sample packet/review | length report、sample packet、freeze readiness manifest | LENGTH PASS / SAMPLE PENDING |
| G212M | Sample review/adjudication | 对 G212R 100 条 fixed sample 做判定并写 freeze readiness | review rows、adjudication、freeze readiness manifest | FAIL / SAMPLE REVIEW |
| G216 | Controlled sample-review repair | 修复或过滤 G212M 暴露的 2Wiki 自洽性问题与 NIAH QA2D 错配 | revised bundle、manifest、rerun audit | COMPLETE / PRE-MANUAL PASS |
| G212R2 | Revised length/sample review | 对 G216 bundle 重跑长度审计并准备/判定固定样本 | length report、sample review、freeze readiness | NEXT / NO TRAINING |
| G300 | Training implementation | query-group loss、citation weighting、fresh/continuation | tests、smoke manifest | BLOCKED BY G212R2 PASS |
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
- [ ] 不把 G210 的 76/100 改写为通过；只能作为 G200R/G210R 的诊断输入。

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
- G200：完成新版数据预物化但仍为 pre-audit。复用旧 G200 已审计 NIAH train 515 groups；在旧 1,023 题之外按 parent-disjoint 规则构造 NIAH 新 model-val，310 个候选经冻结 QA2D 后 307 个 answer-preserved；2Wiki official train 物化 1,075 train / 136 model-val answerable groups；从 train split 生成 1,075 个 unsupported updates，比例 10.1703%。train/model-val group overlap=0、component overlap=0；完整 cases 留在服务器 `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G200-v2/data`，Git 归档小产物位于 `artifacts/G200/`，报告见 `G200_DATA_MATERIALIZATION_REPORT.md`。G200 未启动训练、utility labels 或 held-out；G210 仍必须审计 TRUE/minimal support/citation/manual/length 后才能冻结数据。
- G210：structural audit 3,108/3,108 case 通过，TRUE worklist 2,758 rows 中 2,140 entailed、618 not entailed。按 TRUE 过滤后，NIAH train/model-val 为 515/215，2Wiki train/model-val 为 683/76，unsupported update ratio 为 12.4855%，split overlap 仍为 0；但 2Wiki model-val answerable groups 低于最低门 100，因此 hard data gate failed。人工抽样审计未启动，因为硬门已失败；G300 和任何训练均 blocked。产物位于 `artifacts/G210/`，报告见 `G210_TARGET_AUDIT_REPORT.md`。
- G210 failure triage：只读 G210 structural/TRUE rows 做诊断，不改变 gate、不重跑 TRUE、不训练。2Wiki model-val 136 个 case 中 76 个整链 TRUE 通过、60 个未通过；失败 pair 主要集中在 `country` 16、`publication date` 16、`country of citizenship` 14、`country of origin` 6 等 relation 模板。诊断产物位于 `artifacts/G210/failure-triage/`，报告见 `G210_FAILURE_TRIAGE_REPORT.md`；G210 hard fail 结论不变。
- G210 stop packet：归档当前停止边界、禁止事项和任何恢复推进所需的单独授权条件。该 stop packet 不授权新数据修订，不改变 G210 hard fail，也不解锁 G300/S/I/H；报告见 `G210_STOP_PACKET.md`。
- G215 gate/data amendment：用户要求区分“未达原硬门”和“路线无意义”，并授权适当修订计划。G215 保留 G210 hard fail，不把 76/100 改成通过；但根据 structural 3,108/3,108、split overlap=0、NIAH 515/215、2Wiki train 683、unsupported ratio 12.4855%、held-out/dev 未读取、训练未启动、失败集中于 2Wiki relation 模板等积极信号，授权回到 G200R/G210R 做受控数据修订。该授权后来推进到 G210R2 自动复核通过；当前 G300、Generator 训练、utility labels、S/I/H 仍 blocked until repaired freeze readiness PASS。报告见 `G215_GATE_AND_DATA_REVISION_AMENDMENT.md`，manifest 见 `artifacts/G215/gate_amendment_manifest.json`。
- G200R revised data materialization：2Wiki target construction 改为 `support_sentence_aligned_v1`；NIAH train/model-val 为 515/307，2Wiki train/model-val 为 1,075/136，unsupported groups 为 1,075，unsupported update ratio 为 10.1703%，split group/component overlap=0；pre-audit gates 全部通过。完整 train/validation cases 留在服务器 `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G200R-v1/data`，Git 归档小产物位于 `artifacts/G200R/`，报告见 `G200R_DATA_MATERIALIZATION_REPORT.md`。G200R 未启动训练、utility labels 或 held-out；G210R 仍必须审计 TRUE/manual/length 后才能冻结数据。
- G210R-v1 structural failure：对 G200R-v1 运行 structural audit，3,108 cases 中 25 个 2Wiki answerable case 因 `literal_answer_missing` 失败；citation remap 和 split overlap 均无问题。TRUE audit、finalize、manual audit、训练和 held-out 均未启动。产物位于 `artifacts/G210R-v1/`，报告见 `G210R_STRUCTURAL_FAILURE_REPORT.md`。下一步为 G200R2，在物化时过滤 answer alias 不保留的 support-sentence targets。
- G200R2 revised data materialization：在 2Wiki support-sentence target construction 后新增 answer-alias preservation 过滤。NIAH train/model-val 为 515/307，2Wiki train/model-val 为 1,053/133，unsupported groups 为 1,053，unsupported update ratio 为 10.0881%，split group/component overlap=0；pre-audit gates 全部通过。完整 train/validation cases 留在服务器 `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G200R-v2/data`，Git 归档小产物位于 `artifacts/G200R2/`，报告见 `G200R2_DATA_MATERIALIZATION_REPORT.md`。G200R2 未启动训练、utility labels 或 held-out；G210R2 仍必须审计 TRUE/manual/length 后才能冻结数据。
- G210R2 revised target audit：对 G200R2 执行 structural audit、TRUE audit 和 pre-manual finalize。Structural 3,061 cases 全部通过，TRUE worklist 2,708 rows 中 2,338 entailed、370 not entailed；过滤后 NIAH train/model-val 为 515/215，2Wiki train/model-val 为 828/106，unsupported groups 为 1,053，unsupported update ratio 为 11.3068%，split group/component overlap=0，自动数据门全部通过。完整 train/validation cases 留在服务器 `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G210R-v2/pre-manual`，Git 归档小产物位于 `artifacts/G210R2/`，报告见 `G210R2_TARGET_AUDIT_REPORT.md`。G210R2 未启动训练、utility labels 或 held-out；manual audit 和 length/truncation audit 仍必须在 G212 完成后才能冻结数据。
- G212 length/manual audit：新增 `scripts/full_flow_g212_manual_length_audit.py` 并在服务器 G212-v1 runtime 执行 prepare。长度审计覆盖 2,717 cases / 11,348 examples，`max_length=2304`；4 个 examples 超长，全部来自 `2wiki::b779ecdc08c411ebbd8eac1f6bf848b6` 的 train answerable variants，最大 2,528。manual sample 按 5 个 stratum 各 20 条生成，但未进入 reviewer 判定。G212 状态为 `FAIL / LENGTH / MANUAL REVIEW NOT STARTED`；产物位于 `artifacts/G212/`，报告见 `G212_LENGTH_AUDIT_REPORT.md`。下一步为 G214，且仍不得训练、生成 utility labels 或读取 held-out。
- G214 controlled length repair：新增 `scripts/full_flow_g214_length_repair.py`，只根据 G212 length rows 成组排除 `group_id=b779ecdc08c411ebbd8eac1f6bf848b6` 的 1 个 answerable train case 和 1 个 unsupported counterpart。修订后 train/validation cases 为 2,394/321；NIAH train/model-val 为 515/215，2Wiki train/model-val 为 827/106，unsupported groups 为 1,052，unsupported update ratio 为 11.3033%，split group/component overlap=0，所有 gates 仍通过。完整 revised cases 留在服务器 `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G214-v1/data`，Git 小产物位于 `artifacts/G214/`，报告见 `G214_LENGTH_REPAIR_REPORT.md`。G214 未启动训练、utility labels 或 held-out；G212R 仍必须通过后才能冻结数据。
- G212R revised length/manual audit：对 G214 bundle 重跑 G212 prepare。长度审计覆盖 2,715 cases / 11,342 examples，`max_length=2304`，over max length 为 0，最大长度 2,120；fixed sample 为 5 个 stratum 各 20 条，但全部 `review_decision=PENDING`，未进入 adjudication。产物位于 `artifacts/G212R/`，报告见 `G212R_LENGTH_AUDIT_REPORT.md`。G212R 未启动训练、utility labels 或 held-out；G212M 通过前仍不能进入 G300。
- G212M sample review/adjudication：新增 `scripts/full_flow_g212m_sample_adjudication.py` 并对 G212R 固定 100 条样本完成判定。结果为 92 PASS / 8 FAIL / 0 UNCERTAIN；失败集中在 2Wiki answerable target 的自洽 relation chain 和 NIAH 新 model-val 的少数 QA2D 语义错配。freeze readiness 为 `NOT_FREEZE_READY_SAMPLE_REVIEW_FAILED`，G300 仍 locked；下一步为 G216 controlled sample-review repair。产物位于 `artifacts/G212M/`，报告见 `G212M_SAMPLE_REVIEW_REPORT.md`。
- G216 controlled sample-review repair：新增 `scripts/full_flow_g216_sample_review_repair.py`，只排除 G212M 失败样本对应 case；其中 2Wiki train answerable 失败 case 同步排除 unsupported counterpart。共排除 11 个 case，修订后 train/validation cases 为 2,388/316；NIAH train/model-val 为 515/213，2Wiki train/model-val 为 824/103，unsupported groups 为 1,049，unsupported update ratio 为 11.2929%，split group/component overlap=0，所有 pre-manual gates 仍通过。完整 revised cases 留在服务器 `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G216-v1/data`，Git 小产物位于 `artifacts/G216/`，报告见 `G216_SAMPLE_REVIEW_REPAIR_REPORT.md`。G216 未启动训练、utility labels 或 held-out；下一步为 G212R2 revised length/sample review。
