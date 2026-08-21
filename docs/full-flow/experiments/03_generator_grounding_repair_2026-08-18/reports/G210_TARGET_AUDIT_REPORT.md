# G210 target audit 报告

**日期：** 2026-08-18  
**状态：** `FAIL / HARD DATA GATE / STOP BEFORE MANUAL AUDIT AND TRAINING`  
**服务器 runtime：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G210-v1`

## 本阶段做了什么

G210 对 G200 v2 数据做了三步审计：

1. structural audit：检查 hash、split overlap、answer alias、citation remap、unsupported support removal，并生成 TRUE worklist；
2. TRUE audit：用冻结 `google/t5_xxl_true_nli_mixture` 判断每个新增 target 是否由其绑定 evidence 蕴含；
3. pre-manual finalize：剔除 structural/TRUE 不通过的 case 后，重新计算最低数据门。

本阶段没有训练模型，没有启动 utility labels，没有读取 sealed600 或 system held-out。因为 hard data gate 已失败，未进入人工抽样审计。

## Structural audit

| 项目 | 数量 |
|---|---:|
| total cases | 3,108 |
| structural pass | 3,108 |
| structural fail | 0 |
| TRUE worklist rows | 2,758 |
| split group overlap | 0 |
| split component overlap | 0 |

结构性检查通过：所有 answerable variant 的 citation 都能映射到对应 support evidence；所有 unsupported row 都没有 citation，且被移除的 support 不在可见 evidence 中。

## TRUE audit

| 项目 | 数量 |
|---|---:|
| TRUE worklist rows | 2,758 |
| entailed | 2,140 |
| not entailed | 618 |
| threshold | 0.50 |

旧 G200 已审计 NIAH train-fit targets 按既有 G200 TRUE audit 复用，不进入新增 TRUE worklist。新增 NIAH model-val 与 2Wiki evidence-chain targets 均重新评分。

## 过滤后结果

| 项目 | 过滤后数量 |
|---|---:|
| NIAH train answerable groups | 515 |
| NIAH model-val answerable groups | 215 |
| 2Wiki train answerable groups | 683 |
| 2Wiki model-val answerable groups | 76 |
| 2Wiki unsupported groups | 1,075 |
| train cases | 2,273 |
| validation cases | 291 |
| unsupported update ratio | 12.4855% |
| split group overlap | 0 |
| split component overlap | 0 |

失败门：

```text
2Wiki model-val answerable groups = 76 < required 100
```

因此 G210 不能冻结训练数据，G300 不能启动。

## 关键 SHA256

| 产物 | SHA256 |
|---|---|
| structural rows | `cc777f8759d45e555c290072183f845d73e22974bd0b73b0f35e6794859a96a9` |
| TRUE worklist | `feaa287acee6c65f8f29e7b641862cc4eaad5c48785ca19ae986aaf4c68691bc` |
| TRUE audit rows | `fa930e221d738605312c7623c59d9c520cbb98b3fa43dceebb04845bfc4a5075` |
| pre-manual manifest | `fcbf810cab2ae976f642676990acf7425711b07e9847bed3b7dedec25f9ae4b0` |
| pre-manual ordered IDs | `a2aa08cd7f323818a10524ab8030fa8f560712c1e79eb8df54cf483d9f0376d6` |
| G210 execution audit | `683800a4d88559588bec7b114fd03fea481d22daedd7ee85170b87270e7ad041` |

## 阶段判定

G210 hard data gate failed. 按计划，不能降低 TRUE 门、不能改 split、不能换 seed、不能用最终测试集调参，也不能启动 G300 training implementation。当前路线必须停在 G210 failure，除非用户明确批准新的数据修订计划。
