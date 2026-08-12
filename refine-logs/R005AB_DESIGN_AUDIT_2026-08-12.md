# R005A/R005B 修订方案独立设计与批准前终审

> **后续纠正（2026-08-12）：** 本文件只记录从未获批、从未执行的 v1 草案审计快照。A001 代码映射复核后来发现四项遗漏，因此这里的“最终 P0/P1/P2=0”属于已被推翻的 false-negative 结论，不能再支持批准或实施。当前权威结论与 v2 稳定 blobs 见 [`R005AB_CORRECTIVE_REAUDIT_2026-08-12.md`](R005AB_CORRECTIVE_REAUDIT_2026-08-12.md)。保留本文件是为了让审计失误也可追踪，不是继续沿用旧结论。

**日期：** 2026-08-12  
**对象：** R005A/R005B 的数据划分、统计门和模型/损失实现设计  
**方式：** 三个领域设计审计 + 两个最终稳定快照终审，全部独立只读；审计员未编辑项目文件  
**历史结果：** v1 当时被误判为 `P0=0, P1=0, P2=0`；该结论现已 superseded，v1 从未获批、实现或运行

三个领域 reviewer 的 request、output 与 metadata 分别保存在 `.aris/traces/experiment-audit/2026-08-12_r005ab_{sample,stats,code}/`；最终状态机与跨文档一致性终审保存在 `2026-08-12_r005ab_final_{state_machine,consistency}/`。最终两路共同审计的稳定 blob 为 amendment MD `a107362`、JSON `d52e76c`、tracker `f97f895`，不是只保留“做过审计”的总结声明。

## 1. 总结

三路审计共同支持：新路线在现有数据和 RTX A4000 资源上可执行，且能比“继续调阈值”更明确地区分失败原因。它们同时要求更严格的防泄漏措施：

1. 三个 A-fit variant 必须在第一次看 A-screen 前全部训练、复验并冻结；
2. 两个 B-fit seed 必须在第一次看 B-confirm 前全部训练、复验并冻结；
3. B-confirm 必须一个命令一次性评分两个 seed，四个 seed×dataset 格子全部通过；
4. 统计单位必须是每 component 一个代表 query 的复合成功；
5. 三个 variant 必须共用 pair-preserving batch manifest；
6. 2Wiki 资格只要求 TopK10 至少一条官方支持证据，避免错误的检索难度筛选。

这些要求已经完整写入 [`EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB.md`](EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB.md)。

批准前对抗复核进一步把以下要求写死：A001 必须先冻结、A002 held-out 密封；五个 formal-fit job 使用全局 literal registry/claim/lock/互斥终态；screen/confirm 使用固定 namespace、split anchor/veto、全临界区 owner lock 和 no-replace closed-world bundle；verifier 从 strict checkpoint 重算 raw scores，并逐层重算 threshold、qualification、query composite、exact CP 与最终 gate。两个最终 reviewer 在同一稳定快照上均给出 `P0=0, P1=0, P2=0`。这只说明协议可进入实现，不是对尚未实现的代码或尚未运行的实验作通过保证。

## 2. 样本与隔离审计

### 复算结果

| 数据 | 条件 | query | component |
|---|---|---:|---:|
| NIAH | 排除旧 R005 component 后 | 893 | 801 |
| NIAH | strict clean/cf pair 均在 Top20 | 846 | 766 |
| NIAH | strict clean/cf pair 均在 TopK10 | 778 | 719 |
| 2Wiki | 排除旧 R005 component；TopK10 至少一条 support | 2,677 | 2,078 |

四角色 `64/96/64/128` 每数据源均可分配，角色间 query/component 与完整 text-pair hash overlap 为 0。

### 被修正的问题

- 初稿的 2Wiki “全部 active supporting chunk 都在 TopK10”会过滤 179 题、让 136 个 component 消失，却没有提高真实 gold chain complete 比例；计划已改为“至少一条 TopK10 support 入选，保护全部 TopK10 support”。
- raw document/evidence/text 会因共享语料在不同 query 间复用，不能声称它们全局零重叠；计划只要求复合 `(dataset,query,evidence)`、完整模型输入 hash 和 active-supervised content 隔离。
- NIAH 个别 partition 内存在重复 text-pair row，所以所有门与 CI 均按 query/component，而不是 evidence row。

### Hash 复核

独立审计给出了四个角色 assignment、Top20、Top10 的 canonical 字节定义和 SHA-256。计划已固定所有完整值；批准后必须正式物化并 verify-only 复算，任何不一致都在训练前停止。

## 3. 统计设计审计

### 推荐主门

- `NIAH_success(q)`：canonical 四个类别、三个严格方向和 TopK10 所有 active required 保护全部正确；
- `2Wiki_success(q)`：TopK10 至少一条 official support，且全部 TopK10 support 被保护。

这比把同题多个 candidate/label/direction 当独立样本更诚实。

### 精确边界复算

| n | observed≥.95 且 one-sided exact 95% lower≥.90 的最少成功数 |
|---:|---:|
| 64 | 62；但 fit 不作 CI，因此只需 61/64 |
| 96 | 92/96，lower=`0.907188` |
| 128 | 122/128，lower=`0.909583` |

### 被写入计划的阻断规则

- 三个 variant 的代码/config/loss/threshold/gate 必须在 screen 前同一 commit 冻结；
- 三个 A-fit checkpoint 必须在 screen 首读前全部冻结；
- 两个 B seed 的 checkpoint/threshold/hash 必须在 confirm 首读前全部冻结；
- 两个 seed × 两数据集四格 all-must-pass，不得合并 n、挑 seed 或平均；
- 任一 confirm 失败后，同一 confirm 集永久不能用于换 variant/阈值重试；
- 每条 exact bound 只能称 marginal 95% lower bound，不能称四条同时 95% CI。

审计还指出：该门很保守，真实成功率恰为 95% 时也不保证通过。项目优先保护正确证据，因此计划明确接受这种停止风险。

## 4. 代码与模型审计

### 已核实模型事实

- snapshot 为 `DebertaV2ForSequenceClassification`；标签 0/1/2 对应 contradiction/entailment/neutral；
- `AutoModelForSequenceClassification` 可无 missing/unexpected key 严格加载；
- 原 `AutoModel` 路径明确忽略 pooler/classifier；
- 新 wrapper 可注册 shared encoder、ContextPooler、一次 shared classification dropout 和两份独立 3-logit classifier；strict state reload 可行。

### 精确初始化

线性“目标 classifier 行减其余行平均”不能精确复现 one-vs-rest softmax。计划已采用：

```text
protect_logit = z_entail - logsumexp(z_contrad, z_neutral)
harm_logit    = z_contrad - logsumexp(z_entail, z_neutral)
```

初始化 sigmoid 与原模型对应 softmax 概率的服务器数值误差约 `1e-11`。

### Pairwise loss 与公平 batching

拟议 pairwise logistic 公式和梯度方向正确；zero margin 时 raw `L_pair=log(2)`。但旧 batch 会拆散 pair，所以审计要求：

- V0/V1/V2 共用同一 pair-preserving batch manifest；
- V2 复用同一次 forward，不额外获取 dropout 视图；
- V0/V1 pair weight 为 0，V2 为 0.5；
- 新旧架构使用 versioned builder/fingerprint，旧 R005 verifier 不变。

计划已加入对应 artifact 和测试门。

## 5. 最终批准前边界

五路审计只能说明“当前稳定计划在运行前没有已知 P0/P1/P2 缺口”，不能保证实现会自动正确，也不能保证模型一定通过。下一步仍必须先完成 A001 实现、故障注入测试和旧 R005 回归；R005A/B 仍可能诚实失败。即使 PASS，也只允许提出 R006A 协议，不证明 Selector 优于 TopK10 或现实世界 misinformation 检测有效。
