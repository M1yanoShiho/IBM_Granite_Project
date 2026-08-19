# G210R2 target audit report

**日期：** 2026-08-18  
**状态：** `COMPLETE / PRE-MANUAL PASS / MANUAL+LENGTH PENDING / NO TRAINING STARTED`  
**服务器 runtime：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G210R-v2`

## 本阶段做了什么

G210R2 对 G200R2 的 answer-alias-preserving 数据重新执行训练前自动审计：

1. structural audit：检查 schema、answer alias、citation remap、unsupported support removal、split leakage，并生成 TRUE worklist；
2. TRUE audit：使用冻结 `google/t5_xxl_true_nli_mixture` snapshot 判断 evidence 是否蕴含新增 target；
3. pre-manual finalize：剔除 TRUE 不通过的 case 后，重新计算最低数据门、ordered IDs 和 SHA256。

本阶段没有训练 Generator，没有生成 utility labels，没有读取 sealed600、HotpotQA、MuSiQue-Full、RGB 或 official dev。人工抽样审计和长度/truncation 审计仍未完成，因此 G300 仍不能启动。

## Structural audit

| 项目 | 数量 |
|---|---:|
| total cases | 3,061 |
| NIAH train pass | 515 |
| NIAH validation pass | 307 |
| 2Wiki train answerable pass | 1,053 |
| 2Wiki validation answerable pass | 133 |
| 2Wiki unsupported pass | 1,053 |
| TRUE worklist rows | 2,708 |
| split group overlap | 0 |
| split component overlap | 0 |

结构性检查通过：G210R-v1 中的 `literal_answer_missing` 问题已经被 G200R2 的 answer-alias preservation 过滤消除；citation remap 和 split leakage 检查均通过。

## TRUE audit

| 项目 | 数量 |
|---|---:|
| TRUE worklist rows | 2,708 |
| entailed rows | 2,338 |
| not entailed rows | 370 |
| threshold | 0.50 |

TRUE 审计完成。`not entailed rows` 是 evidence-target pair 层面的数量；pre-manual finalize 按 case 过滤后排除了 344 个 `true_fail` case。

## 过滤后结果

| 项目 | 过滤后数量 |
|---|---:|
| NIAH train answerable groups | 515 |
| NIAH model-val answerable groups | 215 |
| 2Wiki train answerable groups | 828 |
| 2Wiki model-val answerable groups | 106 |
| 2Wiki unsupported groups | 1,053 |
| train cases | 2,396 |
| validation cases | 321 |
| answerable train updates | 8,260 |
| unsupported updates | 1,053 |
| unsupported update ratio | 11.3068% |
| split group overlap | 0 |
| split component overlap | 0 |

自动数据门均通过，包括：

```text
2Wiki model-val answerable groups = 106 >= required 100
```

这说明 G215 授权的数据修订产生了积极结果：原 G210 的 `76/100` 没被改写成通过，而是通过 support-sentence alignment 加 answer-alias preservation 重新构造并重新审计，得到一批通过自动门的数据。

## 关键 SHA256

| 产物 | SHA256 |
|---|---|
| G210R2 execution audit | `17ec634b32865a16228a07d710987465be16fef0ccf0234ffa5d9e142f734552` |
| structural summary | `e22f1a6e10e799ba35df95ec64df1425fab31f6cd3fcecec0fbf980c954399d7` |
| structural rows | `0c003c18aa5d7922917a7d37473a007b87e8c21b5aff13a6ea61ee7e030bc7be` |
| TRUE worklist | `e058e24229561442c5a8752093cacaceece15b972da50b840c009be98e6eea81` |
| TRUE audit manifest | `b24bdbbc4adcade0c59f962a4bcd88796a463ca7d6360e097310829bfbf6bade` |
| TRUE audit rows | `8fb179a6615fe7a5ca48c86a0c7ccb3fbc42aaeda1f14dbde0399a75d1d2fedf` |
| pre-manual manifest | `49a181f53a5f425ce104515462111fa2752f6bfa65c80fbe15b76686cef3284b` |
| pre-manual ordered IDs | `31e64be349709b3004d476f18776dc714e488406879c08c2f088956e503022af` |
| runtime train cases | `ffcf925178f5992401340da44310eb1e38e7b15081a7208efe246c22878d39fd` |
| runtime validation cases | `cd2ba2309349ce99665701ac05077ed522c5cdbe1a1660dfd742a3d6a292a6e1` |

## 归档产物

Git 归档的小型可审计产物位于：

```text
artifacts/G210R2/
```

完整 `train_cases.jsonl` 和 `validation_cases.jsonl` 保留在服务器 runtime，不进入 Git：

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G210R-v2/pre-manual/train_cases.jsonl
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G210R-v2/pre-manual/validation_cases.jsonl
```

## 阶段判定

G210R2 自动复核完成并达到 `PRE-MANUAL PASS`。这足以证明 G215 数据修订有积极信号，路线应继续到下一项必要检查；但它还不足以冻结训练数据。

下一阶段是 G212 manual/length audit。只有 G212 通过后，G300 才能启动；如果 G212 失败，必须按失败类型停下或再次要求用户授权修订，不能为了训练跳过人工或长度门。
