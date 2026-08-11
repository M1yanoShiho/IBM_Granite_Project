# R001A/B Recovery Audit

**日期：** 2026-08-11
**项目：** Adaptive Conservative Selector v2
**审计范围：** R001A inventory + R001B recover-or-rebuild decision
**代码快照：** `refactor/three-module-baseline@74026c09d3b2a7fe0496eed3c9b4044ddccb67da`
**状态：** R001A `PASS`；R001B `PASS`；R001C `RUNNING`

## 1. 决策

六个历史 Hybrid RRF Top20 candidate pool 均在原 HPC 存储中找到，重新计算的文件 SHA-256 与冻结的 Beam/MIS M0 记录逐字节一致。因此 R001B 的决策是：

```text
EXACT_RECOVERY / REPACKAGE
```

这表示：

- 不需要重新运行 retrieval；
- 可在相同旧池上重新计算 TopK10 和后续 Selector 结果；
- 旧 pool 可以作为 `SelectorCandidatePoolManifestV2` 的输入。

这**不表示** R001 已全部完成。历史 v1 manifest 只冻结整文件身份，尚未提供 v2 要求的逐 query canonical hash、严格 candidate/corpus 对齐、component map 和独立验证测试。R001C 完成前不得训练 scorer。

## 2. 审计边界

本次核验是只读资产审计：

- 本地工作树与原 HPC 的 HEAD 都是 `74026c0`；远端核验时工作树 clean；
- 对正式 pool 重新计算 bytes、query-window 数和 SHA-256；
- 解析 pool、run manifest、index manifest 和 candidate metadata 的结构字段；
- 检查 retriever provenance、Top20 窗口和 rank 完整性；
- 检查 dataset/gold/provenance/source-parent 的存在性与冻结身份；
- 没有重新运行 retrieval，没有改写或复制远端正式文件；
- sealed600 与 2Wiki heldout 只检查身份与结构，没有运行或查看新 Selector 效果；
- 远端文件 mtime 未纳入本次冻结记录，不能从本报告推断 mtime。

## 3. 六个 candidate pool

| 角色 | 原 HPC 路径 | Windows | Bytes | 重新计算的 SHA-256 | 冻结记录 | 判定 |
|---|---|---:|---:|---|---|---|
| NIAH dev | `/home/fl25387/projects/IBM_Granite_Project_latest/runs/selector-beam-v1/pools/niah-dev/candidate_sets.jsonl` | 2,000 | 36,265,904 | `89ede8249e0682568ea0a85d00eac4881323c992d8d16362016810bd8cb96ee3` | 同值 | EXACT |
| NIAH train | `/home/fl25387/projects/IBM_Granite_Project_latest/runs/selector-beam-v1/pools/niah-train/candidate_sets.jsonl` | 2,000 | 36,283,237 | `09e8c9b4972f48a67661c8b06dcf220f178528156699da671a16dd8d08fb908f` | 同值 | EXACT |
| NIAH sealed600 | `/home/fl25387/projects/IBM_Granite_Project_latest/runs/selector-reliability-mis/sealed600/pool/candidate_sets.jsonl` | 600 | 10,889,278 | `777391fac4854448a47a7cdc9543cd77710b17f4e960084a26f4640f340ae590` | 同值 | EXACT |
| 2Wiki dev | `/home/fl25387/projects/IBM_Granite_Project_latest/runs/selector-reliability-mis/2wiki/pool/candidate_sets.jsonl` | 2,000 | 27,222,255 | `26442003e230c93e53fe71f4b9a16d269fc5b02674ba6137f6cfceabb721607f` | 同值 | EXACT |
| 2Wiki train | `/home/fl25387/projects/IBM_Granite_Project_latest/runs/selector-beam-v1/pools/2wiki-train/candidate_sets.jsonl` | 3,000 | 38,645,090 | `0ab0fc92f95add1c5d514f531e7b567c4e67d1e5430c401f37ec6f719dbc8887` | 同值 | EXACT |
| 2Wiki heldout | `/home/fl25387/projects/IBM_Granite_Project_latest/runs/selector-beam-v1/pools/2wiki-heldout/candidate_sets.jsonl` | 2,000 | 27,249,204 | `fcca691ed0867bfdc8c6491452d82b4a5701218c8bac7ead0bb486a8469ee219` | 同值 | EXACT |

### 3.1 Retriever 身份与窗口结构

六池共同核验结果：

- candidate provenance 唯一为 `hybrid / hybrid-v1`；
- retriever 参数 SHA-256 为 `67333c6f3fcf6567756b382048961bc150da0ee26af4f9cecc59f58f30b5d78d`；
- fusion 为 RRF，`k=60`，`pool_size=null`；
- sparse arm 为 `strong-bm25`，`k1=0.9`、`b=0.4`；
- dense arm 为 `ibm-granite/granite-embedding-english-r2`；
- run manifest 的直接请求值为 `top_k=20`；candidate metadata 的 execution 参数同样为 `top_k=20`；
- 这六池不是从当前常见的 Top50 run 事后截断所得；
- 每个 query 恰好有 20 条 candidate，retrieval rank 完整覆盖 `1..20`；
- 每个 pool 对应的 run manifest 与 index manifest 均存在。

## 4. 数据、Gold、Provenance 与 ParentIndex

### 4.1 NIAH 结构计数

| 角色 | 数据目录 | Queries | Gold | MutationRecord | ParentIndex rows | Parent unresolved |
|---|---|---:|---:|---:|---:|---:|
| dev | `/home/fl25387/projects/IBM_Granite_Project_latest/runs/niah-injected` | 2,000 | 2,000 | 1,479 | 101,479 | 0 |
| train | `/home/fl25387/projects/IBM_Granite_Project_latest/runs/niah-train-injected` | 2,000 | 2,000 | 1,472 | 101,472 | 0 |
| sealed600 | `/home/fl25387/projects/IBM_Granite_Project_latest/runs/niah-sealed600` | 600 | 600 | 600 | 100,600 | 0 |

冻结标签 hash：

| 角色 | `gold_cases.jsonl` SHA-256 | `provenance.jsonl` SHA-256 | `source_parent.jsonl` SHA-256 |
|---|---|---|---|
| dev | `500115399a04405c71ebf8effdd8eb5aa63a3ce74476f7be60e79ff4747d22e7` | `dc8c0b720cc1a01f1da9fe5a2571988bd89ee0ebe7b7e91493c4def9d1830953` | `b3d4e5c58d79ae2b56fe5842229b8135483efa699bd9ab9ad283ab998a230378` |
| train | `91af72dba9e8ccc97286f5933475bd21e43701a16727d0dfe0c442c55be90699` | `50dbbb3941a65e47363f1aa1acb8bdff7eb755bba9e22000a485d1f233769898` | `721c1b23668c5576d60cf80db592dcf3d6b4bc60ab2b282acce08a737f9582d5` |
| sealed600 | `b1a3fcf131247a5f9a11916accbb495280f8e65acdd00f218d2bedd988aabab4` | `6b328fc3c62702abf1dedaff3c7fe0108e2f67f3b4ff8de690c23150d0769685` | `e5a20e0f99ccff6697aec921dcfe74fc79f7962dd8c03a6f1ea42ef33117e766` |

本地仍保存 NIAH-train 的 `source_parent.jsonl`：7,662,163 bytes、101,472 行，重新计算 SHA-256 为 `721c1b...82d5`，与远端及冻结记录一致；`document_id` 无重复，共 91,595 个规范化 parent。

### 4.2 2Wiki 结构核验

| 角色 | 数据目录 | Queries/Gold | Gold SHA-256 | Source-parent SHA-256 | Candidate→gold parent mapping |
|---|---|---:|---|---|---|
| dev | `/home/fl25387/projects/IBM_Granite_Project_latest/runs/twowiki` | 2,000 | `fba533ebfd77e5c35dee179efab2fb84c6be7ce8c7202d19210d5dc291a5b7c9` | `482ccd4c28fc8312282cb551898beb12bd4b26efbbf210c4ec5fbe5f5510ee34` | missing 0；conflict 0 |
| train | `/home/fl25387/projects/IBM_Granite_Project_latest/runs/twowiki-train` | 3,000 | `b62cc139f390067322b0337431dc962d4befca77026667f1f347e1f09380165b` | `1af322fb6bf749ce5f7897ced3674758b100dfd2c5f7ab80b7021d07a5b7d76e` | missing 0；conflict 0 |
| heldout | `/home/fl25387/projects/IBM_Granite_Project_latest/runs/twowiki-heldout` | 2,000 | `829f1d05de6b4457d6c4d5c410b4179ef01359109919e7ca8724fc07682a816f` | `171c5b5091ea3ddd18af537bc8de9400e83f2b4ed76ed926992158a2bcf5a549` | missing 0；conflict 0 |

这里的 `missing 0 / conflict 0` 只说明 official supporting document 可以稳定映射到 parent。R001A 进一步做了只读 component/overlap 审计：

| Official split | Query 数 | Official-support components | 最大 component（queries） | 若误用全部 Top20 candidate parent |
|---|---:|---:|---:|---|
| train | 3,000 | 2,324 | 20 | 塌成 1 个 component |
| dev | 2,000 | 1,732 | 8 | 塌成 1 个 component |
| heldout | 2,000 | 1,692 | 13 | 塌成 1 个 component |

跨官方 split 的实测 overlap：

| Split pair | Query ID overlap | Distinct official supporting-parent overlap |
|---|---:|---:|
| train–dev | 0 | 449 |
| train–heldout | 0 | 422 |
| dev–heldout | 0 | 434 |

所以“query overlap 为0”不能被写成“supporting-parent overlap 也为0”。把所有 Top20 candidate parent 当作 component edge 会让每个 split 各自成为一个不可用的巨型簇，验证了计划为何必须只使用 official supporting/gold parent。heldout 在本次审计中没有被重分；主结果继续使用完整官方 heldout，parent-seen/unseen 只作为预注册敏感性分层。

## 5. 模型资产

### 5.1 Beam seed-13 checkpoint

- 路径：`/scratch/fl25387/IBM_Granite_Project_latest/runs/selector-beam-v1/models/seed-13/model.pt`
- 大小：735,412,418 bytes
- 重新计算 SHA-256：`d882b90fd98c1c11ebf1251f0fc9b1830a320a60f25bef200e41219a7950e755`
- 与 `M2_TRAIN_REPORT.json` 和 `M2_EVALUATION_REPORT.json` 一致。

### 5.2 DeBERTa base snapshot

- revision 目录：`/scratch/fl25387/IBM_Granite_Project_latest/hf-cache/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7`
- `model.safetensors` SHA-256：`d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`
- 与 `configs/selector/beam_v1.toml` 和 M0 模型记录一致。

这些文件允许复查历史 Beam，但新 dual-head scorer 不会直接把旧 Beam checkpoint 当作正式新模型。

## 6. 配置可恢复性

六份 pool TOML 已从当前工作树退休，但仍能从 Git 历史精确读取：

- Beam pool configs：commit `f84fa72c49a98e80029a1ec60ded405f6715aed1`；
- MIS pool configs：commit `eda6ef1306f02ac9b74f0f989a9e09ba25d81cc5`。

`git show` 得到的配置内容 hash 与历史 pool manifest 逐一匹配：

| 配置 | SHA-256 |
|---|---|
| `selector_beam_niah_dev_pool.toml` | `88cfc8b0a611bb386906a5bb3b42f8311ccdfb4b0abec7c56c961788d395179e` |
| `selector_beam_niah_train_pool.toml` | `6d0201393f61bbe399233790fbe4c233f0e9d641a40d05e422e61696e9cfab0d` |
| `selector_mis_sealed600_pool.toml` | `a080af3a10cb08bd27181363e493a5aa7208cc225c13124f744470389096ef26` |
| `selector_mis_2wiki_pool.toml` | `216934422f9136cc52652856443d45e471916707bc9bbf5dccdc740d5fa244de` |
| `selector_beam_2wiki_train_pool.toml` | `1606b5c63ffa21884d4b3382bdd83cb9bd583bfc83cbe3e7552415c1a3580ad7` |
| `selector_beam_2wiki_heldout_pool.toml` | `544ab4cd300f6c89a14026f4ee0518bc5d9fed1dd22112daf91f748b6e194bdf` |

## 7. 旧结果记录的可用边界

- Beam M0/M1/M2 报告仍由 Git 跟踪；M2 的两个文件通过已有 checksum 清单。
- 本地 MIS 目录的 20 个文件全部通过其 `checksums.sha256`。
- MIS 逐题结果包含 query ID、20 个 candidate ID/rank、required/harmful document ID 和 TopK/MIS selected ID，可用于恢复后的交叉核验。
- MIS 逐题结果不含 candidate text 和完整 evidence→document 映射，不能替代正式 candidate pool 或训练数据。
- 历史 Beam/MIS 指标是失败边界与回归参照，不是新 v2 实验结果。

## 8. R001C 尚未完成的事项

在把 Gate 0 标记为 PASS 前，仍需完成：

1. 实现独立 `SelectorCandidatePoolManifestV2` 和只读 pin/validation CLI；
2. 冻结 pool/run/index/dataset/query/corpus signatures 与内容 hash；
3. 为每个 query 保存完整 canonical `CandidateSet` hash；
4. 验证 query 集合严格一致、每题 20 条、rank `1..20`、candidate 与 corpus chunk 内容一致；
5. 保持现有 BM25-only `CandidateFreeze` 与 `pin_candidates.py` 原样不动；
6. NIAH component 使用 query/gold-parent/family；2Wiki component 只使用 official supporting/gold parent，不使用全部 Top20 candidate parent；
7. 在 R001C/R002 正式 artifact 中复算并冻结已观察到的 2Wiki supporting-parent overlap 与 component 计数；heldout 不重分；
8. 通过对应单元测试和现有回归测试；
9. 在以上事项完成前不运行 scorer 训练。

## 9. 尚未由本审计证明的内容

本报告没有证明：

- 新 Selector 能减少 harmful evidence；
- recall 或 chain loss 满足门槛；
- 2Wiki parent-seen/unseen 分层对新 Selector 效果的影响；本审计只证明 overlap 明确非0并冻结了结构计数；
- 四项 CRC 风险的有效 component 数达到 99；
- v2 manifest 和实验 runner 已完成；
- sealed600/2Wiki heldout 上的新方法有效。

这些都必须由后续预注册 Run 提供新证据，不能从“文件恢复成功”推导出来。
