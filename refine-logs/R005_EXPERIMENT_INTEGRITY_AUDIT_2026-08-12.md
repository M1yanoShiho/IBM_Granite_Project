# R005 实验完整性独立审计

**日期：** 2026-08-12  
**对象：** 正式 R005 bundle、对应代码、执行报告、计划与 tracker  
**审计方式：** 独立只读复核  
**审计模型：** GPT-5.6 Terra，xhigh reasoning（原先希望使用的历史 GPT-5.5 reviewer 在当前环境不可用，已按预先说明采用可用的独立高推理 reviewer）  
**总判定：** `WARN`，但正式结论 `R005 = COMPLETE / FAIL` 与 fail-closed 停止决定成立；未发现 P0 完整性问题

## 1. 一句话结论

R005 的文件、数值、标签掩码、模型输入边界和短路逻辑足以支持以下结论：训练确实完成，预注册逐类门槛确实失败，程序确实没有进入 modelval、阈值和删除效果阶段。因此 R005 必须保持 `FAIL`，R006 之前停止是正确的。

审计的主要警告不是实验结果本身，而是旧报告中“历史两次 verify-only 和三次 post-run audit 均已通过”的事件性说法，没有把当时的独立 stdout/JSON attestation 作为不可变文件保留下来。可重复验证 bundle 是成立的；但历史发生次数不能只靠总结文字证明。

## 2. 六项检查

| 检查 | 判定 | 结论 |
|---|---|---|
| A. Ground-truth provenance | WARN | NIAH harm 是 provenance 严格核验的确定性合成 counterfactual 代理，不是开放世界 misinformation 真值；2Wiki 只有 official supporting 的 protect-positive，harm 与未判断项均 mask。现有 R005 报告已披露该边界。 |
| B. Score normalization | PASS | 决定性准确率按原始 `score >= 0.5` 与冻结标签计算，分母为 active label 总数；没有用预测结果改变分母或做自我归一化。 |
| C. Result existence | WARN | 训练轮数、步数、loss、16/16 配对方向、五项逐类准确率、640 条 train-fit 分数及所有空的下游文件均与正式 bundle 一致；历史复验/审计次数缺少独立 attestation。 |
| D. Dead code / invocation | WARN | 训练门在正式运行中确实执行；失败分支确实只保存 sanity 分数并短路。policy metrics 与 safe-corner 函数在本次运行中未执行，这是正确行为，因此不能把它们当作 R005 效果证据。 |
| E. Scope | PASS | 计划、tracker 与报告均把范围限定为 seed 13、NIAH 16题 + 2Wiki 16题、train-fit sanity；没有声称 modelval 或 Selector 增益。 |
| F. Evaluation classification | PASS | NIAH 被正确描述为确定性合成代理的 train-fit overfit sanity；2Wiki `32/32` 是官方正类 sensitivity，不是完整二分类准确率；selector effect、CRC、sealed/heldout 均为未评估。 |

## 3. 数值与 artifact 对账

- `30/30` epochs、`360` optimizer steps、三项 active source/head loss 下降、两头参数变化、2Wiki harm masked gradient 为 0，与 `sanity/training_trace.jsonl` 一致。
- NIAH protect、harm、safe 三种配对方向均为 `16/16`，与正式报告一致。
- 五项逐类结果为：2Wiki protect-positive `32/32`；NIAH protect-negative `16/16`、protect-positive `71/79`、harm-negative `12/16`、harm-positive `14/16`。
- `sanity/candidate_scores.jsonl` 有 640 行，全部属于固定 32 个 train-fit query；没有 train-modelval 分数。
- `decision_trace.jsonl`、`selection_results.jsonl`、`selected_sets.jsonl`、`count_matched_controls.jsonl` 均为空；`quantile_policies.json` 为 `NOT_EVALUATED_TRAINING_GATE_FAIL`。
- `CHECKSUMS.sha256` 中 14 个条目全部可核验；正式根目录有且只有 16 个普通文件、无 symlink。
- 顶层 manifest、sanity manifest、checkpoint manifest 与 735,356,896-byte checkpoint 的记录互相一致。

## 4. 泄漏检查

- R005 manifest 没有 pinned sealed/heldout 结果路径；runner 与 finalizer 都拒绝 sealed/heldout 路径。
- 实际 R005 在训练门失败后没有进入 modelval。正常 PASS 路径的代码顺序也是先保存 checkpoint、冻结 train-fit 阈值，再允许 modelval。
- scorer 的模型输入只有 `question + candidate_text`；provenance、query/evidence/document ID、rank、role、component 和标签均被禁止进入编码输入。
- provenance 与 source grouping 用于构造和核验标签/角色，这属于监督 sidecar，不是模型输入特征；研究结论仍然以这些冻结代理标签为条件。

## 5. 问题等级与处理

### P0

无。

### P1-1：旧的历史复验/审计次数没有独立 attestation

旧报告、计划和 tracker 直接写成 runner/finalizer 复验通过且三次审计 `P0=0, P1=0`，但正式 R005 的 exact-file-set bundle 按设计不包含这些外部复验 stdout 或 reviewer 输出。因此：

- 可以保留“bundle 目前可按哈希和语义重新核验”；
- 应把没有留存原始凭据的历史次数降级为“先前工作记录曾报告”；
- 从本次起，把新的 verifier 输出和本独立审计以带日期文件长期保存，并记录命令、commit、时间、退出码、身份和哈希。

### P2 边界

1. 任何摘要都必须保留“确定性合成 counterfactual 代理”和“2Wiki 官方 protect 正类”限定，不能写成自然世界 misinformation accuracy。
2. 2Wiki `32/32` 只能称 protect-positive sensitivity/正类准确率。
3. policy metrics/safe-corner 代码虽有测试，但没有在 R005 实际效果路径运行，不能称为本次已验证效果。

## 6. 审计后的可用结论

在完成 P1 文档修正和当前复验 attestation 后，可以准确声称：

> 正式 R005 bundle 可重新验证且支持 `COMPLETE / TRAINING-GATE FAIL`；没有证据泄漏到 modelval、sealed 或 heldout，也没有执行删除效果评估。一个新的独立完整性审计未发现 P0，但指出旧的历史复验次数没有单独保存原始 attestation，现已按新的长期留证规则修正。

该审计不允许把 R005 改判 PASS，也不允许直接运行 R006。
