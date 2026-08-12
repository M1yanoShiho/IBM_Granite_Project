# 自适应保守 Selector 实验跟踪表（v2）

**对应计划：** `EXPERIMENT_PLAN_2026-08-11_v2.md`

**状态规则：** `TODO / RUNNING / PASS / FAIL / BLOCKED / CUT`

**禁止：** 未运行前预填结果；失败产物不得删除；sealed/heldout 不得用于调参。

## Run 队列

| Run | Milestone | 目的 | 系统/数据 | 决定性指标 | 优先级 | 状态 | 结果路径/备注 |
|---|---|---|---|---|---|---|---|
| R001 | M0 | A盘点→B恢复/重建→C实现严格 Hybrid-v2 manifest | NIAH/2Wiki train/dev；sealed 只审计 hash | 完整性、retriever/pool SHA、连通分量 overlap | MUST | PASS | 六池 exact recovery 与 Hybrid-v2 pool freeze PASS；component/crossing 由 R002 封存；未训练 |
| R002 | M0 | 冻结指标、cluster CI、expected-risk CRC 协议 | toy + simulated losses | document-ID、conditional chain、component representative、`(ΣL+1)/(n+1)` | MUST | PASS | `COMPLETE / SAMPLE-SIZE GO`；四风险 n 均≥99；不代表真实 Selector/策略已通过 CRC |
| R003 | M0 | 数量基线并冻结等量删除对照生成器 | TopK10/9/8/7；random/bottom-rank 协议 | harm、recall、chain、selected count、seed derivation | MUST | PASS | `BASELINE-PROTOCOL PASS`；生成/verify-only `5/5`，本地/服务器 `14/14` hash match；固定 TopK9 不满足保守目标；非 Selector PASS |
| R004 | M1 | 标签审计与 200q 资源预检 | source-train 全量标签；train-modelval 固定 200q | label/mask、truncation、吞吐 | MUST | PASS | `RESOURCE-PREFLIGHT PASS`；80,460 条标签零语义违规；4,000 pair 前向 145.431 pair/s；无截断/OOM/NaN；未写 checkpoint、未执行删除 |
| R005 | M1 | 双头 sanity | 固定 train-fit 16+16 小样本；全部 403q train-modelval | 分头/分类准确率、配对方向、safe corner、等量随机对照、fallback | MUST | RUNNING | 实现、完整本地回归与运行前独立审计已通过（P0/P1=0）；等待固定 commit 的服务器 formal，尚无实验结果 |
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

- [x] 实际 commit/branch 已记录：R001 恢复盘点基线为 `refactor/three-module-baseline@74026c0`；R002 正式生成与服务器核验使用 `ffe2d27411e6c4848debcb2887405c045cea0541`；R003 正式生成使用 `c23c4df71ba61c95b1acb72411684e6a62ff3e77`；R004 正式资源预检使用 clean commit `3170d154351cc620e9cee5cefbdf49cae7e831a2`。
- [x] 六个历史 Hybrid RRF Top20 pool 已恢复并逐文件匹配历史 SHA-256；决策是 exact recovery/repackage，不是 rebuild。
- [x] Hybrid RRF Top20 使用独立 `SelectorCandidatePoolManifestV2`；现有 BM25-only pin 未被放宽或误用。
- [x] retriever name/version/params hash、`top_n=20`、query/data signature 和逐题 pool hash 已冻结。
- [x] candidate/gold/provenance/source-parent 的 query 和 document 映射严格一致；candidate 另已逐字段匹配 signed corpus chunk。
- [x] NIAH query/assignment-required-source-parent/family component 与同一 source split 内的整组分配已完成，并通过预注册 fingerprint 审计。
- [x] 2Wiki component 只使用 query 与 official supporting/gold document-title parent；没有把所有 Top20 candidate parent 当作关系边。
- [x] R001A 已只读量化 2Wiki official-support component 与跨官方 split overlap；query overlap 三对均为0，supporting-parent overlap 明确非0。
- [x] R001C/R002 正式 artifact 已复算并冻结上述计数；同一 source split 内的派生角色 query/component crossing=0。
- [x] 2Wiki heldout 成员保持官方 split 不变；正式 cluster bootstrap 与 parent-seen/unseen 敏感性协议已冻结。
- [x] 每项 CRC 风险的 component representative 选择规则/seed/hash 已冻结；每 component 最多一个代表 query。
- [x] R001A inventory 有本地/远端只读证据。
- [x] R001B 已冻结六池 `EXACT_RECOVERY / REPACKAGE` 决策。
- [x] R001C manifest/tests 有完整 PASS 证据；Gate 0 前未训练 scorer。
- [x] sealed/heldout 只做存在性、hash 与结构核验，未用于 Selector 效果检查或调参。
- [x] 当前 `top-k` 默认行为未改变。

**状态：** PASS（R001 六池与 R002 component/role/representative artifact 均已冻结并独立复验；未训练 scorer，未读取 sealed/heldout Selector 效果）

**证据：**

- 恢复审计：[`results/selector-adaptive-risk-v1/R001/R001_RECOVERY_AUDIT.md`](../results/selector-adaptive-risk-v1/R001/R001_RECOVERY_AUDIT.md)。
- R001C 机器可读报告：[`results/selector-adaptive-risk-v1/R001/R001C_VALIDATION_REPORT.json`](../results/selector-adaptive-risk-v1/R001/R001C_VALIDATION_REPORT.json)；六份逐 query v2 manifest 位于同目录 `manifests/`。
- R002 人类可读报告：[`results/selector-adaptive-risk-v1/R002/R002_PROTOCOL_REPORT.md`](../results/selector-adaptive-risk-v1/R002/R002_PROTOCOL_REPORT.md)；机器可读验证：[`R002_VALIDATION_REPORT.json`](../results/selector-adaptive-risk-v1/R002/R002_VALIDATION_REPORT.json)。
- 四套 R002 artifact 独立 `--verify-only` 为 `4/4 PASS`；本地/服务器 24 文件 `24/24 MATCH`；10 项 canonical projection fingerprint `10/10 MATCH`；全部派生角色 query/component crossing=`0`。
- 六池均为 `hybrid/hybrid-v1`、RRF `k=60`、`strong-bm25(k1=0.9,b=0.4)+granite-dense`、直接运行 `top_k=20`，且每题 20 条/rank `1..20` 完整。
- 2Wiki official-support components：train `2,324`（max `20`）、dev `1,732`（max `8`）、heldout `1,692`（max `13`）；误用全部 Top20 candidate parent 时每个 split 都塌成单一 component。
- 2Wiki 跨官方 split：query overlap 三对均 `0`；supporting-parent overlap train–dev `449`、train–heldout `422`、dev–heldout `434`。
- 远端 Beam seed-13 checkpoint 与 DeBERTa base snapshot 均命中历史 SHA-256。
- R001C 已在提交 `f05060a` 完成；本地最新远端合并后 `1179 passed`，服务器新增测试 `21 passed`，六池 freeze 与 verify-only 均为 `6/6 PASS`。

## Gate 1 — 指标与基线

- [x] `harm_reduction = TopK−Selector`，正数为好。
- [x] `recall_loss = TopK−Selector`，正数为坏。
- [x] recall 和 complete-chain 只比较 `document_id` 集合，不混用 evidence ID/Candidate。
- [x] paired 比较拒绝 query-set 不一致；共享 parent/family 时使用 paired cluster bootstrap/sign-flip。
- [x] 2Wiki 的 cluster 单位来自 official supporting/gold parent；跨官方 split overlap 另作敏感性分层，不通过重分 heldout 消除。
- [x] p-value 使用 plus-one；不出现 `p=0.0`。
- [x] Top20 pool-conditional、TopK10 baseline-exposed、unconditional 三种 harm 分母均保存。
- [x] TopK10/9/8/7 可复算；count-matched 生成器、100 repeats 和 seed derivation 已冻结；真实对照结果按协议等待实际 Selector trace，不伪造。
- [x] chain loss 只在 TopK10 chain-eligible query 上计算，分母没有被全部 query 稀释。
- [x] CRC 对每个风险使用 `（ΣL+1）/(n+1)≤0.01`，P0–P6 selected sets 嵌套且 loss 单调，四风险最终取最保守策略。
- [x] 四项必需风险各自有效 component `n≥99`，才允许进入 R004/scorer 训练；否则 Gate 1 标记 `BLOCKED/CUT`，先扩充 calibration 或提交事前论证的 amendment。
- [x] calibration 与未来 component 的可交换性假设和分布审计已记录；无法支持时未声称 CRC 理论保证。
- [x] CRC toy/simulation 在无安全策略时选择 P0；bootstrap 95% CI 没有被写成 CRC 保证。

**状态：** PASS（R002 的指标/CRC 样本量协议与 R003 的 TopK 数量基线、三种 harm 分母、count-matched 生成协议均已冻结并独立复验；R004 资源前置检查也已通过，下一步 R005）

**证据：**

- R002 协议报告与机器可读验证同 Gate 0 链接；指标/CRC 实现提交：`ffe2d27411e6c4848debcb2887405c045cea0541`。
- 四项 calibration representative component 数分别为 NIAH recall `523`、NIAH conditional chain `402`、2Wiki recall `866`、2Wiki conditional chain `468`，均满足 `n≥99`。
- R002 判定只为 `SAMPLE-SIZE GO`：当前没有 scorer 或真实策略 loss，不能写成 Selector、阈值或非零策略 CRC PASS。
- R003 零基础报告：[`R003_PROTOCOL_REPORT.md`](../results/selector-adaptive-risk-v1/R003/R003_PROTOCOL_REPORT.md)；机器可读验证：[`R003_VALIDATION_REPORT.json`](../results/selector-adaptive-risk-v1/R003/R003_VALIDATION_REPORT.json)；run-level 依赖与缺失状态：[`selector_experiment_manifest.json`](../results/selector-adaptive-risk-v1/R003/selector_experiment_manifest.json)。
- R003 正式生成与独立 verify-only 均为 `5/5 PASS`，本地/服务器 14 个原始文件 `14/14 SHA-256 MATCH`，服务器上游输入 pin 直接重算 `54/54 MATCH`；独立复算 `30,008` 行 trace 的嵌套/删除/R002 绑定异常均为 `0`；服务器相关 `60 passed`，本地全仓 `1268 passed`，Ruff/mypy PASS。
- NIAH dev 固定 TopK9 相对 TopK10 的 Top20 pool-conditional harmful reduction 约 `+2.04 pp`，但 recall loss 约 `1.90 pp`、conditional chain loss 约 `4.47 pp`；因此数量缩减本身不是可接受 Selector，后续必须允许逐题 0 删除。
- count-matched random/bottom-rank 目前只有冻结协议，数值结果明确为 `DEFERRED_UNTIL_REAL_SELECTOR_TRACE`；R003 PASS 不是 Selector 或策略 PASS。

## Gate 2 — Scorer 可行性

- [x] protect/harm 是两个独立 sigmoid 输出，不是 softmax 互斥类；R004 对两头原始输出差值做了非退化审计。
- [x] NIAH counterfactual 没有仅因“相关”被标成 protect positive；965 条 own counterfactual 全部为 `(protect=0,harm=1)`。
- [x] 2Wiki unjudged 使用 mask，不是 negative；除 5,922 条 official supporting protect positive 外，其余 54,078 条均为双 mask。
- [x] provenance/source group 没进入文本推理特征；模型可见输入严格只有 `question + candidate_text`。
- [ ] modelval 至少有一个非零策略 harm point 改善、两数据 recall loss ≤3 pp。
- [ ] deletion precision 优于逐题 count-matched random。

**状态：** RUNNING（R004 `RESOURCE-PREFLIGHT PASS`；前四项结构/标签/输入条件已证实；最后两项必须由 R005 及其后续真实 Selector trace 回答）

**证据：**

- R004 顶层封存：[`selector_experiment_manifest.json`](../results/selector-adaptive-risk-v1/R004/selector_experiment_manifest.json) 与 [`CHECKSUMS.sha256`](../results/selector-adaptive-risk-v1/R004/CHECKSUMS.sha256)；原子改名前后 verify-only 均 PASS；独立 post-run 审计逐行复算后 PASS、无 P0/P1。
- 标签审计：NIAH `20,460` 条、2Wiki `60,000` 条；候选文本语义违规 `0`，未判断 2Wiki 候选没有被伪造为 negative。
- 资源预检：[`resource_preflight_report.json`](../results/selector-adaptive-risk-v1/R004/preflight/resource_preflight_report.json)；固定 `100+100` query 共 `4,000` pair 前向为 `145.431 pair/s`，全部 `8,060` pair 最大 token 长度 `298<512`、截断 `0`，12 个临时训练 micro-batch 无 OOM/NaN。
- seed-13 全量训练估计为 `0.2142 GPU-hour`，p95 保守估计 `0.2818 GPU-hour`；这只是资源可行性，不是效果 PASS。R004 没有 checkpoint、策略、删除 trace、count-matched 结果或 sealed/heldout 效果。

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
| 2026-08-11 | R002 判定为 `COMPLETE / SAMPLE-SIZE GO`，不判 Selector 或非零策略 PASS | 四项代表 component n 均≥99，只证明当前 α=1% 路线具备继续实验的样本量；尚无真实策略 loss | Gate 0 PASS；Gate 1 继续 R003 |
| 2026-08-12 | R003 判定为 `COMPLETE / BASELINE-PROTOCOL PASS`，Gate 1 PASS | TopK10/9/8/7 与三种 harm 分母已复算；固定 TopK9 虽改善约2.04 pp harmful，却损失约1.90 pp recall、4.47 pp 完整链，不能替代自适应 Selector | R004 NEXT；后续策略必须允许逐题删0条 |
| 2026-08-12 | count-matched 只冻结生成器与100个重复，结果延后到真实 Selector trace | 公平对照必须逐题复制实际删除数；R003 没有 Selector trace，提前填结果属于伪造 | R009/R012/R014/R015 生成真实对照；R003 manifest 标记 `DEFERRED` |
| 2026-08-12 | run-level artifact contract 显式区分 `NOT_APPLICABLE` 与 `DEFERRED` | 基线/协议阶段没有 learned scores 或 Selector trace；伪造空占位文件比明确状态更易误读 | R003 起每阶段 manifest 声明适用性；进入 scorer/Selector 阶段后仍必须产出当阶段必需文件 |
| 2026-08-12 | R004 的 200q 固定为两数据各100个 train-modelval query，按预注册 SHA-256 顺序抽样；每题前向 Top20 | 不按标签/长度/结果挑样本，同时用最大候选负载测 scorer；后续删除动作仍只在 TopK10 | R004 RUNNING；另对全部403q/8,060 pair做 tokenizer-only 截断审计 |
| 2026-08-12 | R004 允许最多12个不落盘的训练 micro-batch 资源探针 | 只有 forward 吞吐不能可信估计反向传播、optimizer 与显存成本；该探针不保存 checkpoint、不用于选模型 | R004 只估资源；R005 才开始小样本 sanity |
| 2026-08-12 | R004/R005 不再复用旧 Beam 的“每题4个 hard negative”；未判断候选数量固定为0 | 当前没有 audited irrelevant sidecar，非 gold 不等于负例；沿用旧开关会把已识别的监督错误重新引入 | 只训练 active-mask 标签；某 source/head/class 频数为0时不造样本、不除零、不计 loss |
| 2026-08-12 | R004 判定为 `COMPLETE / RESOURCE-PREFLIGHT PASS`，Gate 2 仅完成前四项 | 全量标签、输入隔离、独立双头、长度、GPU 前向/短反向探针与封存复验均通过；但没有训练 checkpoint 或真实删除效果 | 只允许进入 R005 小样本 sanity；不得声称 Selector、非零策略或 harmful reduction 已通过 |
| 2026-08-12 | R005 在运行前冻结为 train-fit 哈希抽样 16+16、最终 epoch checkpoint、训练分数分位点阈值、全部 403q modelval 一次性验收 | 原计划未说明 16 题如何选、95% 按什么分母、阈值从哪里来；先补规则可阻止看过 modelval 后换样本/阈值 | R005 仅诊断 0–cap1；checkpoint/cap/threshold 不进入 R006/R007，失败按 FAIL/CUT 停止 |
| 2026-08-12 | R005 加权 BCE 固定按每个 head 的 active 样本数归一，不按当前 micro-batch 的 weight sum 再归一 | 后一种写法会在单类 micro-batch 中把 class weight 自身抵消，使已冻结的少数类加权名存实亡；该问题在任何 R005 模型结果产生前发现 | R004 默认 loss 语义不变；R005 显式使用 `active-count-per-head` |
| 2026-08-12 | R005 的固定样本覆盖不足、训练 CUDA OOM 或 NaN/Inf 必须封存为可复验 FAIL | 直接异常退出会丢掉“为什么没有进入 modelval”的证据；但把任意代码/输入错误包装成实验失败也会掩盖实现缺陷 | 覆盖不足保存 epoch-0 初始化 checkpoint；训练异常回滚最后完整 epoch；两者都短路阈值/modelval，普通实现或 pin 错误继续硬失败 |
| 2026-08-12 | R005 将 base snapshot identity 与 trained checkpoint fingerprint 分开，并移除 mtime 证据 | 通用 `ModelPin` 要能跨 R004/R005 比较同一基座；checkpoint 状态另需独立绑定，mtime 不能跨 Git/SSH 稳定保存 | inner/top-level manifest 固定基座 identity；checkpoint manifest 固定加载后 state fingerprint；同一 pinned server 环境做语义重算，跨机只核内容 hash；staging 完整复验后原子发布 |
| 2026-08-12 | R005 仅把样本覆盖失败与 epoch 1–30 训练循环内 OOM/NaN 封存为实验 FAIL；其余执行异常保持 hard error | 实验失败回答“冻结方法不满足门”，而初始化、checkpoint、最终评分或 modelval 无法执行只说明该次运行未完成；混在一起会把工程故障误写成科学结论 | hard error 不发布正式 R005、不改阈值或样本；修复执行问题后按同一冻结协议重跑 |
| 2026-08-12 | R005 使用独立 `adaptive_risk_r005_sanity.toml`，恢复且不再修改 R004 已 pin 的 `adaptive_risk_v1.toml` | 首次 formal 在训练前复验 R004 时，被配置内容 hash/size 保护拦截；共享可变配置会破坏上游证据链 | 该次未创建 staging、未读取模型结果；修复后按同一预注册样本与阈值运行 |
