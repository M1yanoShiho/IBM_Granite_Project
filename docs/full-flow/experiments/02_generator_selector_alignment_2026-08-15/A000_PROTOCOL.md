# A000 数据、模型与统计协议冻结

**日期：** 2026-08-15
**状态：** `COMPLETE / PASS`
**机器产物：** [`artifacts/A000/A000_DATA_MODEL_MANIFEST.json`](artifacts/A000/A000_DATA_MODEL_MANIFEST.json)、[`artifacts/A000/A000_POWER.json`](artifacts/A000/A000_POWER.json)、[`artifacts/A000/A000_SERVER_AUDIT.json`](artifacts/A000/A000_SERVER_AUDIT.json)

## 1. 冻结决定

### 数据角色

- NIAH train：只用于 Generator 训练、离线 utility labels 和 Selector 训练；
- NIAH decision-dev 739：只用于诊断、方法选择和消融；
- sealed600：永久退休，只读历史结果；
- HotpotQA 400、MuSiQue-Full 400 对 answerable/unanswerable、RGB 300：项目已经预留且从未评分的系统级最终集；历史 dry-run 载入全量数据检查 schema/count，并让每个数据集、每个实验臂各 3 条通过 Generator，但未保存答案、未运行 scorer；
- RGB-counterfactual 100：预注册的次级分析；
- ASQA/QAMPARI：只作已经揭示的 Generator/citation regression。

系统 held-out 继续使用 `configs/heldout-sample.json` 中 seed 13 已冻结的 query IDs，不重抽、不查看答案结果。三个主数据集分别报告，不 pooling；MuSiQue-Full 以 answerable 400 为答案主分析，unanswerable 400 报无依据断言和 abstention。

### 模型栈

- Retriever：StrongBM25 + `ibm-granite/granite-embedding-english-r2@47ea694...` Hybrid RRF；
- Legacy Selector：`cross-encoder/nli-deberta-v3-base@6c749ce...`，checkpoint `86622bd...`；
- Generator：`ibm-granite/granite-4.1-3b@c065040...`；
- 新 Generator 只允许同一 Granite 基座的 base、clean draft LoRA、mixed draft LoRA；
- Claim verifier：TRUE `google/t5_xxl_true_nli_mixture`，冻结且不作裁判；
- Citation evaluator：`lytang/MiniCheck-Flan-T5-Large@96eafd...`，只在生成后独立评分。

### Runtime 边界

正式系统固定为：

```text
Retriever → Selector → Generator → one answer
```

Reference answer、official supporting IDs、gold chain 和 leave-one-out 分数不能进入 runtime。Utility 构造允许多次生成，但 gold 只能在每次生成完成后由离线 scorer 读取。

## 2. 统计冻结

最终主要比较：

- `D-A`：完整新系统相对默认系统；
- `D-C`：相同新 Generator 下 Utility Selector 的净作用；
- `D-H`：Utility Selector 相对 Legacy Selector。

最终 inference 使用 exact McNemar 和 paired component-cluster bootstrap 95% CI。Answer superiority 要求预注册比较的 CI lower bound `>0`；coverage non-inferiority margin 为 `-1pp`；citation precision/recall non-inferiority margin 为 `-2pp`。

F005 的 component design effect 为 `600/588=1.0204`。按双侧 α=.05、power=.80 的保守 paired-binary 近似：

| 假设 discordance | 检出 +2pp 所需 n | 检出 +3pp 所需 n | 检出 +5pp 所需 n |
|---:|---:|---:|---:|
| 5% | 1,002 | 445 | 161 |
| 10% | 2,003 | 890 | 321 |
| 15% | 3,004 | 1,335 | 481 |

当前 HotpotQA/MuSiQue-answerable 各 n=400，RGB n=300。因此每数据集大致只能稳定识别 3–6pp 的中等效应，不能可靠确认 2pp 小效应。最终 CI 跨0时必须写“对小效应证据不足”，不能写等价或无效。

开发集只用于候选选择：要求完整分布点估计、coverage、空答案、上下文稳定性和多 seed 方向共同通过，不用 dev p-value 宣布正式提升。

## 3. Server 核验结果

2026-08-15 已在 `it097952` 完成实体核验并通过：

1. 服务器原有 23 个 dirty 文件先逐一与团队远端 blob 比较，内容全部一致；随后建立保护提交并直接合并团队提交，最终工作区干净；
2. NIAH train/dev 和退休 sealed600 candidate pool 的实体 SHA256 全部与冻结 manifest 一致；
3. Granite Generator、Granite embedding、Selector base、TRUE 和 MiniCheck snapshot/revision/config hash 均存在并一致；
4. seed-13 Selector checkpoint 实体 SHA256 为 `86622bd...72bf`，与 F005 一致；
5. 当前服务器可访问的 `/home/fl25387` 与 `/scratch/fl25387` 中没有 HotpotQA/RGB/MuSiQue system-heldout cache 或评分产物；历史 `/user/work/fl25387` 路径在当前服务器不存在，因此历史暴露边界以已归档 dry-run 代码和 handover 为准；
6. 历史 dry-run 确实执行过小切片生成，因此 A000 冻结语义是 `SCHEMA_AND_PIPELINE_DRYRUN_ONLY / NEVER_SCORED`，不是“从未进入 Generator”。

完整路径、字节数、revision 和 hash 保存在 `A000_SERVER_AUDIT.json`。A000 已满足进入 A001 的工程前置条件；本阶段没有启动 GPU 训练，也没有读取 held-out 分数。
