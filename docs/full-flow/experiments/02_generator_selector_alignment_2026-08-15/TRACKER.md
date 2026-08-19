# Generator-first、Selector-aligned 执行跟踪表

**对应计划：** [PLAN.md](PLAN.md)
**当前状态：** `A000-G230 COMPLETE; G230 NO CANDIDATE; S300 NOT ACTIVATED`
**更新规则：** 未实际执行不得填写 PASS；每个 COMPLETE 项必须附 commit、input hash、run manifest 和结果路径。

| Run | Milestone | 目的 | 主要输入 | 必须产物 | 优先级 | 状态 |
|---|---|---|---|---|---|---|
| A000 | 协议冻结 | 冻结数据、Granite-centered 模型栈、主比较、统计门和 gold 边界 | 旧 query/provenance、system held-out、模型 snapshots | `A000_PROTOCOL.md`、data/model/power/server-audit manifest | MUST | COMPLETE / PASS |
| A001 | Trace schema | 记录 draft→claims→faithfulness→routing 全链路 | 当前 Generator 代码 | schema、单测、逐题 trace | MUST | COMPLETE / PASS |
| A002 | 基线/方差 | 复现基线并测同输入生成波动 | 已揭示 dev | baseline report、repeat report | MUST | COMPLETE / PASS |
| B100 | Context matrix | 测 support-only、noise、position、Selected | changed + matched unchanged dev | cases JSONL、diagnostic report | MUST | COMPLETE / DIAGNOSTIC PASS |
| B110 | Bottleneck decision | 将问题路由到 Retriever/Generator/Selector | B100 产物 | decision record | MUST | COMPLETE |
| G200 | Draft data | 构造可溯源的 evidence→draft targets | NIAH train、provenance groups | train/val cases、manifest | CONDITIONAL | COMPLETE / PASS |
| G210 | Split fallback | 只在 splitter 归因为主时增加降级路径 | A001/B100 trace | tests、implementation report | CONDITIONAL | NOT ACTIVATED BY B110 |
| G220-S | Training smoke | 验证 draft adapter、loss mask、长度和显存 | 小样本 train cases | smoke manifest | CONDITIONAL | COMPLETE / PASS |
| G220-GC | Clean draft LoRA | 普通 draft 微调控制组 | clean/support-only contexts | 3-seed checkpoints/manifests | CONDITIONAL | COMPLETE / TRAINING PASS |
| G220-GM | Mixed draft LoRA | 训练 TopK/Selected/noise/position 鲁棒性 | matched mixed contexts | 3-seed checkpoints/manifests | CONDITIONAL | COMPLETE / TRAINING PASS |
| G230 | Generator dev gate | 选择或拒绝 G* | G0/GN/GC/GM | paired report、citation report | MUST AFTER TRAIN | COMPLETE / NO CANDIDATE |
| S300 | Freeze G* | 冻结 utility-label 教师 | G230 通过版本 | model/runtime manifest | CONDITIONAL | NOT ACTIVATED / NO G* |
| S310-P | Utility pilot | 验证 leave-one-out 标签产出与稳定性 | 100 train queries、G* | utility cases、yield report | CONDITIONAL | NOT ACTIVATED / NO G* |
| S310-F | Utility full | 构造 Selector utility 训练集 | NIAH train、G* | utility dataset、manifest | CONDITIONAL | BLOCKED BY PILOT |
| S320 | Utility Selector | 训练 legacy safety + utility 目标 | S310-F dataset | checkpoints、training report | CONDITIONAL | BLOCKED BY S310-F |
| S330 | Selector dev gate | 验证 SU 是否提高同一 G* 的答案 | TopK/SL/SU + G* | answer/evidence/citation report | MUST AFTER TRAIN | BLOCKED BY S320 |
| I400 | Retriever freeze | 用真实 Hybrid RRF 生成并冻结候选 | dev/new-held-out corpus/index | candidate pools、manifest | MUST | BLOCKED BY A000 |
| I410 | 五臂联合 dev | 估计 Generator、Selector 和交互作用 | A/B/C/D/H | paired report、trace、CI | MUST | BLOCKED BY G230/S330/I400 |
| I500 | System held-out | HotpotQA/MuSiQue-Full/RGB 一次最终确认 | 冻结系统、冻结 query IDs | final report、all manifests | MUST | BLOCKED BY I410 |

## 每个 Run 的完成清单

- [ ] 输入路径、角色和 SHA256 已记录
- [ ] runtime gold boundary 已审计
- [ ] 模型、adapter、prompt、decode 和软件环境已记录
- [ ] 主 Generator 仍为冻结的 Granite 4.1-3B 基座，非 Granite 模型只承担 Selector/Verifier/Evaluator 的既定角色
- [ ] 错误数和缺失输出已报告
- [ ] 完整分布为主结果，changed subset 仅作诊断
- [ ] 逐题产物和聚合报告同时保存
- [ ] 配对置信区间与 W→R/R→W 已报告
- [ ] 未读取或调试新 held-out 逐题结果
- [ ] 结果为负时按停止规则归档，没有继续在同一数据上选方法

## 当前禁止项

- sealed600 再利用
- Selector threshold/cap 扫描
- 新方法在 held-out 上调参
- runtime 多次 Generator 调用或读取 reference answer
- 同时改动 Generator、Selector 和 Retriever 后只做单一前后比较
- 未完成 A001/B100 就启动新 LoRA

## 已完成记录

- A000：服务器 `it097952` 实体审计 PASS；数据、模型、Selector checkpoint 与历史 manifest 的 SHA256 一致；system held-out 的准确边界为“schema/count + 每数据集每臂 3 条 pipeline dry-run，未保存答案、未评分”；产物位于 `artifacts/A000/`。
- A001：默认关闭的逐题 Generator trace 已覆盖 raw draft、split/faithfulness、TRUE routing、claim disposition 与 final-empty reason；trace on/off 行为等价；全项目 `1646 passed, 20 skipped`；schema 位于 `artifacts/A001/`。
- A002：739 题 Base Verify 基线逐题复现 F001；K=64.01%、L=63.60%，L-K=-0.41pp，CI [-1.31,+0.44]pp；74 题每臂同输入重复与 630 个 unchanged 平行调用均逐题完全一致；两臂各 813 次调用、0 错误、trace 零缺失；产物位于 `artifacts/A002/`。
- B100：218 题六臂 context matrix 共 1308 次调用、0 错误、trace 零缺失；O support-only=68.81%，K=60.09%，O-K=+8.72pp，CI [+3.10,+14.69]pp；O 仍有 68 题失败，噪声与位置导致大量逐题翻转；产物位于 `artifacts/B100/`。
- B110：主路线判定为 Generator evidence utilization/context robustness；全 dev 有 12/739 Retriever support-absent；7 个 K 对/S 错题的 8 条删除均为 harmful；G210 不激活，G200/G220 解锁，S300 继续等待 G230；产物位于 `artifacts/B110/`。
- G200：从 1,023 个 NIAH train role-assigned 问题构造实际 `DRAFT_PROMPT` 数据；严格 answer-token、support/benign/harmful、Legacy Selector support 和 TRUE entailment 审计后，最终得到 515 train + 62 model-val 问题，每题 8 个上下文，GC/GM 各 4,120 个等量样本；train/model-val component、decision-dev query 和 sealed600 query overlap 均为 0；产物位于 `artifacts/G200/`，完整 cases 保存在服务器 runtime。
- G220-S：全量长度审计后冻结 `max_length=2304`（GM max=2120，零截断）；GC/GM 均在相同 4 个最长 query 上完成 seed-13 smoke，各 32 样本、4 updates，峰值显存 9.18/10.09 GB，保存后的 adapter 均通过 fresh-base reload；正式配置冻结为 515 queries、4,120 examples、515 updates/arm、seeds 13/42/73；产物位于 `artifacts/G220/smoke/`。
- G220-GC/GM：两臂均完成 seeds 13/42/73；每个 run 为 515 queries、4,120 examples、515 updates、0 truncation，六份 persisted adapter 均通过 fresh-base reload；adapter 只作用于 draft call，decision-dev/sealed/held-out 未读取。该阶段只判定 TRAINING PASS，不判定方法有效；报告见 `G220_TRAINING_REPORT.md`，manifest/config 位于 `artifacts/G220/formal/`。
- G230：GN、GC/GM seeds 13/42/73 均完成 1,829 tasks（739 TopK + 5×218 diagnostics），所有运行错误和 trace 缺失为 0，生成 runtime 未加载 gold。GC/GM 均通过 citation 前 family gate；GM mixed-context robustness 未通过，回退 GC。GC 三个 seed 的 MiniCheck citation precision/recall 配对 CI 下界全部低于 -2pp，最终 `NO CANDIDATE`，S300 未激活。报告见 `G230_RESULTS.md`，完整归档位于 `artifacts/G230/`。

## 已测吞吐

- A002：1626 次调用、1892 个 trace claims，调用计时合计 8057.50 秒，平均 4.96 秒/调用；
- B100：1308 次调用、1487 个 trace claims，调用计时合计 6384.87 秒，平均 4.88 秒/调用；
- 环境：两张 RTX A4000 同时承载 Granite 4.1-3B 与 TRUE，单进程顺序执行各臂；上述计时用于后续预算，不外推到训练 steps。
