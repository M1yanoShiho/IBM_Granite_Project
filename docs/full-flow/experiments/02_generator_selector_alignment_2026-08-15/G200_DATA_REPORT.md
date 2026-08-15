# G200 evidence-to-draft 训练数据报告

**日期：** 2026-08-15  
**状态：** `COMPLETE / DATA CONSTRUCTION PASS`  
**正式服务器路径：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G200-v1`

## 1. 本阶段完成了什么

G200 不再训练 F006 的五类 key-fact notes。它构造的是实际 Granite draft call 使用的：

```text
question + evidence context -> self-contained answer sentence with citation index
```

每个合格问题固定八种上下文：

1. `support_only`；
2. `topk`；
3. `legacy_selected`；
4. `support_benign`；
5. `support_harmful`；
6. `support_first`；
7. `support_middle`；
8. `support_last`。

八种上下文使用同一个语义答案，只根据当前 evidence 顺序重映射 citation index。prompt 是生产 draft 路径的 `DRAFT_PROMPT`，SHA256 为 `50acd572...bd8f3`。

## 2. 数据和模型边界

- 数据只来自 NIAH train 的冻结候选池和既有 `train-fit/train-modelval` role assignments；
- split 继续沿用 provenance component 分组，不按单独 query 随机重切；
- 冻结 Legacy Selector 只读取 question、TopK10 和 role IDs，没有读取 gold/reference；
- QA2D `MarkS/bart-base-qa2d@94f286a...` 只用于离线把短 reference 改写成自包含目标句，不是运行时系统模块；
- 冻结 TRUE `aa6cfe1...` 只用于确认 support evidence 蕴含目标句；
- 下一步要训练和运行的主答案模型仍是 `ibm-granite/granite-4.1-3b@c0650403...`；
- 训练与目标筛选没有使用 decision-dev 的答案或分数；数据完成后只用冻结的 decision-dev/sealed600 query IDs 做 overlap audit；没有读取 system held-out。

## 3. 实际筛选结果

| 项目 | 数量 |
|---|---:|
| role-assigned NIAH train 问题 | 1,023 |
| QA2D 保留完整 answer token sequence | 997 |
| QA2D 未保留 | 26 |
| 无 support / harmful / benign | 12 / 111 / 55 |
| reference 为 `unknown` | 1 |
| Legacy Selector 删除唯一答案 support | 1 |
| 进入 TRUE target audit | 817 |
| TRUE entailment 通过 | 577 |
| TRUE entailment 未通过并排除 | 240 |
| 最终 train / model-val 问题 | 515 / 62 |
| 每题上下文变体 | 8 |
| GC / GM 训练样本数 | 4,120 / 4,120 |

最终 577 题的 TRUE 分数最小值为 0.5117，中位数为 0.9888，阈值固定为 0.50。

## 4. Selector train-context 结果

冻结 Selector 在全部 1,023 个 role-assigned 问题中改变 336 题、删除 367 条证据。最终 577 个合格问题中改变 217 题、删除 235 条。

这里的 Selector 结果只负责形成 Generator 训练时会遇到的 `legacy_selected` 上下文，不是新的 Selector 训练，也不形成 Selector 效果结论。

## 5. 隔离与一致性审计

- train/model-val provenance component overlap：0；
- 最终 query 与 739 题 decision-dev overlap：0；
- 最终 query 与 sealed600 overlap：0；
- 577/577 case 都有八种变体；
- 577/577 semantic target SHA256 可复算一致；
- 三个 position 变体保持同一个 TopK evidence set；
- Selector selection manifest 明确记录 `gold_loaded_at_runtime=false` 和 `reference_answers_loaded_at_runtime=false`。

## 6. 产物

Git 归档保存 selection trace、QA2D targets、TRUE target audit 和各阶段 manifest：

- `artifacts/G200/selection/`；
- `artifacts/G200/qa2d/`；
- `artifacts/G200/target-audit/`；
- `artifacts/G200/data/manifest.json`；
- `artifacts/G200/G200_EXECUTION_AUDIT.json`。

完整 train/model-val cases 共约 31.7 MB，只保存在正式服务器 runtime；其 SHA256 已写入 data manifest。旧的宽松匹配和 pre-audit 数据也保留在服务器带版本后缀的目录中，不用于训练。

## 7. 阶段判定

G200 的通过含义仅为：存在可追溯、component 隔离、目标受支持且直接训练 Granite draft call 的 GC/GM 等量数据。

本阶段没有训练 LoRA，也没有测量 Generator answer/coverage/citation 提升。G220-S 因此解锁，G220-GC/GM 仍需先通过 smoke。
