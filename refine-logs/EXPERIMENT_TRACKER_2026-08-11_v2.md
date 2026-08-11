# 自适应保守 Selector 实验跟踪表（v2）

**对应计划：** `EXPERIMENT_PLAN_2026-08-11_v2.md`

**状态规则：** `TODO / RUNNING / PASS / FAIL / BLOCKED / CUT`

**禁止：** 未运行前预填结果；失败产物不得删除；sealed/heldout 不得用于调参。

## Run 队列

| Run | Milestone | 目的 | 系统/数据 | 决定性指标 | 优先级 | 状态 | 结果路径/备注 |
|---|---|---|---|---|---|---|---|
| R001 | M0 | A盘点→B恢复/重建→C实现严格 Hybrid-v2 manifest | NIAH/2Wiki train/dev；sealed 只审计 hash | 完整性、retriever/pool SHA、连通分量 overlap | MUST | RUNNING | R001A/B 与 R001C pool freeze PASS；component/crossing artifact 随 R002 完成；不训练 |
| R002 | M0 | 冻结指标、cluster CI、expected-risk CRC 协议 | toy + simulated losses | document-ID、conditional chain、component representative、`(ΣL+1)/(n+1)` | MUST | RUNNING | CRC 与 95% CI 分开；同时封存 component/crossing artifact |
| R003 | M0 | 数量基线并冻结等量删除对照生成器 | TopK10/9/8/7；random/bottom-rank 协议 | harm、recall、chain、selected count、seed derivation | MUST | TODO | 实际对照等 Selector trace 产生后生成 |
| R004 | M1 | 标签审计与 200q 资源预检 | train-modelval 子集 | label/mask、truncation、吞吐 | MUST | TODO |  |
| R005 | M1 | 双头 sanity | 小样本 protect/harm | overfit、held-out safe corner、fallback | MUST | TODO |  |
| R006 | M2 | 全量训练 seed13，只冻结 checkpoint/候选分位点 | train-fit → train-modelval | dual scores、checkpoint rule | MUST | TODO | 不提前冻结 P0–P6 |
| R007 | M2 | 方法选择与决定性消融 | grouped OOF/train-modelval | 0–cap1/2/3、`ε_harm`、复杂度序；冻结唯一 family/cap/分位点 | MUST | TODO | 不看 calibration/decision |
| R008 | M2 | CRC calibration seed13 | CRC-calibration | P0–P6、recall/chain risk bound | MUST | TODO | 非零策略全未通过、只能结构回退 P0 则 CUT |
| R009 | M3 | 单种子主结果 | decision-dev | harm CI lower>0；recall target/hard gate；等量删除对照 | MUST | TODO | 一次性 decision；对照来自真实 trace |
| R010 | M4 | 训练并校准 seed42 | train-fit → modelval → calibration | 同 R006/R008 | MUST if R009 PASS | TODO |  |
| R011 | M4 | 训练并校准 seed73 | train-fit → modelval → calibration | 同 R006/R008 | MUST if R009 PASS | TODO |  |
| R012 | M4 | 三种子 decision-dev | decision-dev | 均值、每 seed、CI、最差 recall、各 seed 等量对照 | MUST | TODO |  |
| R013 | M5 | 正式冻结 | model/config/CRC/source/data | manifest 完整性 | MUST | TODO |  |
| R014 | M5 | NIAH sealed600 | sealed600 | harm CI、required recall、chain、正式 trace 对照 | MUST | TODO | 一次正式批次 |
| R015 | M5 | 2Wiki heldout | heldout | supporting recall、chain、正式 trace 对照 | MUST | TODO | 与 R014 间不得调参 |

### R001 分阶段状态

| 子阶段 | 状态 | 已验证结论 | 证据/下一步 |
|---|---|---|---|
| R001A inventory | PASS | 本地与原 HPC 的 pool、dataset/gold/provenance/source-parent、run/index manifest、Beam 模型资产已盘点 | [`R001_RECOVERY_AUDIT.md`](../results/selector-adaptive-risk-v1/R001/R001_RECOVERY_AUDIT.md) |
| R001B recover-or-rebuild | PASS | 六个历史 Hybrid RRF Top20 pool 的字节级 SHA-256 全部命中；决策为 `EXACT_RECOVERY / REPACKAGE`，不重新检索 | 同上 |
| R001C Hybrid-v2 pool integrity | PASS | 六个真实 pool 均完成 write-once freeze 与独立 verify-only；逐 query hash、run/index/metadata、query/corpus 对齐全部通过 | [`R001C_GATE_EVIDENCE.md`](../results/selector-adaptive-risk-v1/R001/R001C_GATE_EVIDENCE.md)；BM25-only pin 未修改 |

## Gate 0 — 代码和资产现实

- [x] 实际 commit/branch 已记录：本地与原 HPC 均为 `refactor/three-module-baseline@74026c0`，远端工作树核验时 clean。
- [x] 六个历史 Hybrid RRF Top20 pool 已恢复并逐文件匹配历史 SHA-256；决策是 exact recovery/repackage，不是 rebuild。
- [x] Hybrid RRF Top20 使用独立 `SelectorCandidatePoolManifestV2`；现有 BM25-only pin 未被放宽或误用。
- [x] retriever name/version/params hash、`top_n=20`、query/data signature 和逐题 pool hash 已冻结。
- [x] candidate/gold/provenance/source-parent 的 query 和 document 映射严格一致；candidate 另已逐字段匹配 signed corpus chunk。
- [ ] NIAH query/gold-parent/family component 与同一 source split 内的整组分配已完成，并通过预注册 fingerprint 审计。
- [ ] 2Wiki component 只使用 query 与 official supporting/gold document-title parent；没有把所有 Top20 candidate parent 当作关系边。
- [x] R001A 已只读量化 2Wiki official-support component 与跨官方 split overlap；query overlap 三对均为0，supporting-parent overlap 明确非0。
- [ ] R001C/R002 正式 artifact 已复算并冻结上述计数；同一 source split 内的派生角色 query/component crossing=0。
- [ ] 2Wiki heldout 成员保持官方 split 不变；正式 cluster bootstrap 与 parent-seen/unseen 敏感性协议已冻结。
- [ ] 每项 CRC 风险的 component representative 选择规则/seed/hash 已冻结；每 component 最多一个代表 query。
- [x] R001A inventory 有本地/远端只读证据。
- [x] R001B 已冻结六池 `EXACT_RECOVERY / REPACKAGE` 决策。
- [x] R001C manifest/tests 有完整 PASS 证据；Gate 0 前未训练 scorer。
- [x] sealed/heldout 只做存在性、hash 与结构核验，未用于 Selector 效果检查或调参。
- [x] 当前 `top-k` 默认行为未改变。

**状态：** RUNNING（R001A/B 与 R001C pool freeze PASS；component/crossing/CRC representative artifact 等待 R002）

**证据：**

- 恢复审计：[`results/selector-adaptive-risk-v1/R001/R001_RECOVERY_AUDIT.md`](../results/selector-adaptive-risk-v1/R001/R001_RECOVERY_AUDIT.md)。
- R001C 机器可读报告：[`results/selector-adaptive-risk-v1/R001/R001C_VALIDATION_REPORT.json`](../results/selector-adaptive-risk-v1/R001/R001C_VALIDATION_REPORT.json)；六份逐 query v2 manifest 位于同目录 `manifests/`。
- 六池均为 `hybrid/hybrid-v1`、RRF `k=60`、`strong-bm25(k1=0.9,b=0.4)+granite-dense`、直接运行 `top_k=20`，且每题 20 条/rank `1..20` 完整。
- 2Wiki official-support components：train `2,324`（max `20`）、dev `1,732`（max `8`）、heldout `1,692`（max `13`）；误用全部 Top20 candidate parent 时每个 split 都塌成单一 component。
- 2Wiki 跨官方 split：query overlap 三对均 `0`；supporting-parent overlap train–dev `449`、train–heldout `422`、dev–heldout `434`。
- 远端 Beam seed-13 checkpoint 与 DeBERTa base snapshot 均命中历史 SHA-256。
- R001C 已在提交 `f05060a` 完成；本地最新远端合并后 `1179 passed`，服务器新增测试 `21 passed`，六池 freeze 与 verify-only 均为 `6/6 PASS`。

## Gate 1 — 指标与基线

- [ ] `harm_reduction = TopK−Selector`，正数为好。
- [ ] `recall_loss = TopK−Selector`，正数为坏。
- [ ] recall 和 complete-chain 只比较 `document_id` 集合，不混用 evidence ID/Candidate。
- [ ] paired 比较拒绝 query-set 不一致；共享 parent/family 时使用 paired cluster bootstrap/sign-flip。
- [ ] 2Wiki 的 cluster 单位来自 official supporting/gold parent；跨官方 split overlap 另作敏感性分层，不通过重分 heldout 消除。
- [ ] p-value 使用 plus-one；不出现 `p=0.0`。
- [ ] Top20 pool-conditional、TopK10 baseline-exposed、unconditional 三种 harm 分母均保存。
- [ ] TopK10/9/8/7 可复算；count-matched 生成器、100 repeats 和 seed derivation 已冻结。
- [ ] chain loss 只在 TopK10 chain-eligible query 上计算，分母没有被全部 query 稀释。
- [ ] CRC 对每个风险使用 `（ΣL+1）/(n+1)≤0.01`，P0–P6 selected sets 嵌套且 loss 单调，四风险最终取最保守策略。
- [ ] 四项必需风险各自有效 component `n≥99`，才允许进入 R004/scorer 训练；否则 Gate 1 标记 `BLOCKED/CUT`，先扩充 calibration 或提交事前论证的 amendment。
- [ ] calibration 与未来 component 的可交换性假设和分布审计已记录；无法支持时未声称 CRC 理论保证。
- [ ] CRC toy/simulation 在无安全策略时选择 P0；bootstrap 95% CI 没有被写成 CRC 保证。

**状态：** TODO

**证据：**

## Gate 2 — Scorer 可行性

- [ ] protect/harm 是两个独立输出，不是 softmax 互斥类。
- [ ] NIAH counterfactual 没有仅因“相关”被标成 protect positive。
- [ ] 2Wiki unjudged 使用 mask，不是 negative。
- [ ] provenance/source group 没进入文本推理特征。
- [ ] modelval 至少有一个非零策略 harm point 改善、两数据 recall loss ≤3 pp。
- [ ] deletion precision 优于逐题 count-matched random。

**状态：** TODO

**证据：**

## Gate 3 — Seed 13 主结果

- [ ] NIAH Top20 pool-conditional harm reduction 95% CI lower >0。
- [ ] unconditional harm 同方向并明确分母。
- [ ] NIAH required recall loss point ≤1 pp、CI upper ≤3 pp。
- [ ] 2Wiki supporting recall loss point ≤1 pp、CI upper ≤3 pp。
- [ ] TopK10-chain-eligible conditional chain loss point ≤1 pp、CI upper ≤3 pp。
- [ ] random/bottom-rank 按本次真实 trace 的逐题删除数量生成，并且 Selector 优于二者。
- [ ] 至少有部分 query 实际 `DROP_HARM`。
- [ ] decision-dev 后没有修改阈值或策略梯子。

**状态：** TODO

**证据：**

## Gate 4 — 简洁性与三种子

- [ ] 0–cap1/2/3 与固定输出 TopK9/8/7 已公平比较。
- [ ] harm-only、无 protect、无 CRC 已消融。
- [ ] R007 已冻结唯一 `policy_family` 与最终 cap；R008 只在其中选择 threshold 强度。
- [ ] R007 使用 `ε_harm=max(0.5 pp,100/n_harm_eligible pp)` 与预注册复杂度顺序，没有事后解释“明显更好”。
- [ ] 如果 cap1 位于冻结 `ε_harm` 最优范围内，最终方案已降级为自适应 0–1。
- [ ] seeds 13/42/73 全部报告。
- [ ] 每个 seed 的 CRC policy 都只由同一 calibration 角色和同一冻结规则产生。
- [ ] 三个 seed 的 harmful point reduction 均为正，cluster-bootstrap seed-mean CI lower >0。
- [ ] 任一 seed recall loss point 均未超过3 pp。
- [ ] source grouping 无独立作用时已降级，不强留复杂组件。

**状态：** TODO

**证据：**

## Gate 5 — 正式确认

- [ ] 升版 `selector_experiment_manifest.json` 已保存 model/config/CRC/source/label/data/pool 内容 hash 和 schema version。
- [ ] 独立 experiment runner 同时写出 SelectionResult 与 decision trace，生产 Selector protocol 未改变。
- [ ] NIAH sealed600 满足 Gate 3 的 harm/recall/chain 门槛。
- [ ] 2Wiki heldout 满足 Gate 3 的 recall/chain 门槛。
- [ ] R014 与 R015 之间未改变任何参数。
- [ ] 失败时 `top-k` 仍是唯一生产默认。
- [ ] 通过后才创建 production registration 提案。

**状态：** TODO

**证据：**

## 每次 Run 必填模板

```text
Run ID:
开始/结束时间:
Git commit / branch:
配置路径与 SHA-256:
Selector experiment manifest schema/version/path/SHA-256:
Retriever name/version/params SHA-256/top_n:
Selector-v2 candidate pool manifest SHA-256:
数据、candidate pool、gold、provenance、source sidecar SHA-256:
连通分量 map / CRC representative list / selection seed / SHA-256:
2Wiki official-support component schema / cross-official-split parent overlap audit / SHA-256:
模型 ID/revision/checkpoint SHA-256:
CRC artifact SHA-256:
数据角色:
Seed:
top_k / baseline_k / min_keep / final cap:
policy_family / final cap / 分位点级别 / 本 seed 绝对 thresholds / 策略梯子版本:
四风险各自有效 component n、经验风险、CRC corrected risk、最强合格策略:
四风险取最保守后的最终策略:
命令或入口:
设备、wall time、峰值显存:
逐 candidate score 路径:
decision trace 路径:
selected sets 路径:
count-matched 对照 seed/100 repeats/输出路径:
聚合指标与 paired CI 路径:
0/1/2/3 删除分布:
对应 Gate:
判定与理由:
是否允许下一 Run:
```

## 决策日志

| 日期 | 决策 | 理由 | 影响 |
|---|---|---|---|
| 2026-08-11 | 不再把 `DEFER` 既保留又计为 harmful 改善 | 保留证据仍会暴露给 Generator | 全部 Run |
| 2026-08-11 | `ABSTAIN_KEEP` 等同保留且不记功 | 指标语义唯一 | R002 以后 |
| 2026-08-11 | utility 改名 protect，表示 official required/supporting 保护价值 | 相关不等于正确有用 | R004 以后 |
| 2026-08-11 | 不固定每题最多删1条；比较 0–cap1/2/3，R007 冻结最终 cap | 最终可能为0–1、0–2或0–3，不预设复杂方案胜出 | R003、R007–R009 |
| 2026-08-11 | TopK10 候选 Selector 专用 min_keep=7、max_delete≤3 | 保险丝，不影响仓库其他小 `max_selected` 配置 | 新 Selector only |
| 2026-08-11 | harmful 最低门槛改为“正且 CI lower>0”，不再强制3 pp | 项目目标是比 TopK 可信改善一点 | Gate 3/5 |
| 2026-08-11 | recall 目标收紧到 point≤1 pp，hard CI upper≤3 pp | 优先保护正确证据 | Gate 3/5 |
| 2026-08-11 | formal 前不注册新 Selector | 当前生产只允许 top-k，安全回退 | R001–R015 |
| 2026-08-11 | Hybrid pool 使用独立 v2 manifest，不复用 BM25-only candidate pin | 当前代码的 sealed pin 明确只接受 BM25 | R001/Gate 0 |
| 2026-08-11 | recall/chain 统一映射到 document ID | official qrels 与 Candidate/evidence ID 类型不同 | R002 以后 |
| 2026-08-11 | split 改为 query/parent/family 多 key 连通分量 | 简单拼接 key 仍会让共享 parent/family 跨 split | R001/R004 |
| 2026-08-11 | CRC 冻结为 expected-risk `（ΣL+1）/(n+1)`；四风险取最保守策略 | 区分期望风险控制与最终 95% CI，避免任意 δ 分配 | R002/R008 以后 |
| 2026-08-11 | count-matched 对照在真实 decision trace 产生后生成 | R003 尚不知道每题实际删除数 | R003/R009/R012/R014/R015 |
| 2026-08-11 | 新增独立 experiment runner 与升版 manifest | 当前 SelectionResult/RunManifest 无法完整保存动作和依赖 hash | M0/M1 |
| 2026-08-11 | chain risk 采用 TopK10 完整链条件口径 | 防止 baseline-ineligible query 的 0 稀释断链风险 | R002 以后 |
| 2026-08-11 | CI 按连通分量 cluster bootstrap；CRC 每 component 预选一个代表 query | 共享 parent/family 的 query 不能重复当独立样本 | R001/R002 以后 |
| 2026-08-11 | P0 是结构性 fallback，不写成通过修正经验风险检验 | `n<99` 时非零策略无法在 α=1% 下被选中 | R008 以后 |
| 2026-08-11 | R006 不冻结梯子；R007 用数值化 `ε_harm`/复杂度序冻结 family、cap、分位点 | 避免 R006/R007 重复选择和事后解释“明显更好” | R006/R007 |
| 2026-08-11 | seeds 42/73 复用 family/cap/分位点级别，但从各自 modelval 分数映射绝对 threshold | 分数尺度可随 seed 变化，同时不允许重新选方法 | R010/R011 |
| 2026-08-11 | 四项必需风险有效 component 均须 `n≥99` 才进入训练 | α=1% 的 CRC 修正项给出的数学前置条件 | Gate 1/R002 |
| 2026-08-11 | 六个历史 Hybrid RRF Top20 pool 采用 `EXACT_RECOVERY / REPACKAGE` | 原 HPC 文件的 bytes、整文件 hash、run/index manifest 与历史 M0 全部匹配 | R001A/B PASS；R001C 继续 |
| 2026-08-11 | 六池独立 Hybrid-v2 manifest 已 write-once freeze 并二次 verify | 逐题 canonical hash、角色锁、run/index/metadata/query/corpus 联合验证全部通过 | R001C pool integrity PASS；R002 继续 component/CRC |
| 2026-08-11 | 2Wiki component 只用 official supporting/gold parent，不用所有 Top20 candidate parent | distractor parent 不是正确证据关系，会制造伪相关 component | R001C/R002/R004 |
| 2026-08-11 | 2Wiki 跨官方 split supporting-parent overlap 如实报告，不为追求0而重分 heldout | 官方 split 必须保持；overlap 是敏感性变量，不是可删掉的数据 | R002/R013/R015 |
