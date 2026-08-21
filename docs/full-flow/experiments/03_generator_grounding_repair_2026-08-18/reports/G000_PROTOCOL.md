# G000 协议和统计范围冻结

**日期：** 2026-08-18
**状态：** `FROZEN BY G000 / NO TRAINING`
**路线：** `03_generator_grounding_repair_2026-08-18`

## 1. 冻结目的

G000 只确认新版路线已经获得用户授权，并在任何新训练、utility label 生成或 held-out 运行前，固定输入身份、运行边界、统计门、预算、denylist 和服务器实体。G000 本身不训练模型、不运行 GPU 生成、不评分 system held-out。

正式系统保持：

```text
Retriever -> Selector -> Generator -> one answer
```

TopK、Legacy Selector、Utility Selector 和不同 Generator 只作为平行实验臂比较，不是运行时多次回答。

## 2. 运行时边界

- Retriever runtime 不读取 gold/reference；
- Selector runtime 不读取 gold/reference；
- Generator runtime 不读取 gold/reference；
- reference answer、official supporting facts、provenance、utility labels 和 scorer 只能在离线 target 构造、utility label 构造或生成后评分中读取；
- 所有后续 manifest 必须显式记录 `gold_loaded_at_runtime=false` 和 `reference_answers_loaded_at_runtime=false`。

## 3. 冻结组件

- Retriever：当前 Hybrid RRF/default、索引、语料身份、模型和参数全部冻结；
- Legacy Selector `SL`：只作为有限范围 safety/risk baseline，冻结 `cross-encoder/nli-deberta-v3-base`、seed-13 checkpoint、threshold `0.9212157130241394`、cap=2；
- Generator：IBM Granite 4.1-3B 继续作为主 base，阶段 G 默认只训练 draft LoRA；
- TRUE：冻结 runtime verifier，不作为自身输出的裁判；
- MiniCheck：只作为生成后的独立 citation judge；
- G0/GC/GM/G230：只读历史；G230 结论仍是 `COMPLETE / NO CANDIDATE`。

## 4. 冻结数据边界

- sealed600 永久退休，只读历史，不再训练、筛选、重测或调参；
- HotpotQA、MuSiQue-Full、RGB 和 RGB-counterfactual 只保留 frozen ordered IDs/hash；SystemF 冻结且用户单独授权前不读取内容、不生成、不评分；
- NIAH/2Wiki train 可用于阶段 G/S 的训练或 utility，但必须先通过 split、component、parent 和 held-out denylist 检查；
- NIAH decision-dev 和 2Wiki dev 可用于模块资格和完整系统开发，不形成最终 superiority 结论；
- ASQA/QAMPARI 只作已揭示 citation catastrophe guard。

## 5. 统计冻结

- 主要独立单位为 query provenance component；
- 同一 query 的 context variants、leave-one-out variants 和重复 rows 是 repeated measurements；
- seeds 是方法重复，不扩充样本量；
- family estimate 先在 query 内平均 seed-level paired delta，再按 component cluster bootstrap；
- bootstrap 固定为 10,000 次、seed=13；
- G010 只能使用 G230 paired discordance 和 component 结构做 simulation-based MDE/sensitivity，不使用 observed power；
- 模块职责门、强统计结论门和完整系统门分开报告。

## 6. 固定预算

- Generator：2 个 seed13 screen fits、3 个 formal fits、最多 1 个 G110 激活后的 downstream smoke/refit；不做 rank/alpha/dropout/lr/prompt 网格；
- Selector：100-query utility pilot、1 次完整 utility materialization、1 个 seed13 recipe/threshold screen、3 个 formal seeds；
- 完整系统：1 次 locked full-flow development bundle；
- held-out：SystemF 后且用户单独授权时最多 1 次 bundle。

## 7. 停止与 fallback

- G000 任一实体核验失败：停止，不进入 G010/G100；
- 新 Generator 失败：记录 `NO NEW CANDIDATE`，可按计划使用 `GQ=G0` 作为冻结教师，但不能声称 Generator 修复成功；
- utility pilot 标签不足或不稳定：停止 Utility Selector，不强造训练集；
- Selector 无同 GQ 下端到端正作用：不冻结三模块新方法；
- 完整系统职责门失败：不运行 held-out；
- held-out H 永远需要用户单独授权。
