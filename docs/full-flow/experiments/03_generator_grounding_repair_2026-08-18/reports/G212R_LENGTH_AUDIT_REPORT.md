# G212R revised length/manual audit report

**日期：** 2026-08-18  
**状态：** `LENGTH PASS / MANUAL REVIEW PENDING / NO TRAINING STARTED`  
**服务器 runtime：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G212R-v1`

## 本阶段做了什么

G212R 对 G214 revised pre-manual bundle 重新运行训练前长度审计，并重新固定 manual review packet：

1. 使用冻结 Granite 4.1-3B tokenizer；
2. 保持 `max_length=2304`；
3. 覆盖 revised train/validation 的所有 prompt + target examples；
4. 按同一固定分层规则准备 100 条 manual sample。

本阶段没有训练 Generator，没有生成 utility labels，没有读取 sealed600、HotpotQA、MuSiQue-Full、RGB 或 official dev。

## 长度审计结果

| 项目 | 数量 |
|---|---:|
| cases | 2,715 |
| examples | 11,342 |
| max_length | 2,304 |
| over max_length examples | 0 |
| min length | 309 |
| p50 | 1,344 |
| p90 | 1,841 |
| p95 | 1,882 |
| p99 | 1,966 |
| max | 2,120 |
| truncation rate if encoded at 2304 | 0.0000% |

G214 排除唯一超长 train group 后，长度门已通过。

## Stratum 结果

| Stratum | examples | p95 | max | over max_length |
|---|---:|---:|---:|---:|
| 2wiki train answerable | 4,135 | 1,645 | 2,012 | 0 |
| 2wiki train unsupported | 1,052 | 1,359 | 1,667 | 0 |
| 2wiki model-val answerable | 530 | 1,619 | 1,943 | 0 |
| NIAH train answerable | 4,120 | 1,906 | 2,120 | 0 |
| NIAH model-val answerable | 1,505 | 1,918 | 2,030 | 0 |

## Manual packet

G212R manual sample 已固定，但 reviewer 判定仍未完成：

| Stratum | rows |
|---|---:|
| 2wiki train answerable | 20 |
| 2wiki train unsupported | 20 |
| 2wiki model-val answerable | 20 |
| NIAH train answerable | 20 |
| NIAH model-val answerable | 20 |

所有 sample rows 当前仍为：

```text
review_decision = PENDING
```

因此 G212R 不能写成 `PASS`，也不能解锁 G300。

## 关键 SHA256

| 产物 | SHA256 |
|---|---|
| runtime G212R script | `18330af6b71c68b62550df437dc086bded6649a15ba1350654079ed49d02da2c` |
| G212R execution audit | `c8936074098ba27c0d86445e9ec00c98576b2f02a7413bac32b89fe6aa2fabf1` |
| prepare manifest | `fa683cd0e25b1f190b9197a2743de630085f1bec748053aaa5594cda47a69485` |
| length audit | `0946f082118dbb5d8eb12cf2ed3ef3b8f1b51c76cc5f41b1aa303d372b1b7873` |
| length rows | `67dfeaa393f9fbce33350a0e5d8bcd4b96712db90b1f41f7fc5dbf79b499f988` |
| manual sample summary | `fd58aa15b79d0090d155f7273e44387a9f269c7ae7137e9d14f88ff85f35b1a7` |
| manual sample | `9434da2434ea10dad5e1fe7548ab54745e11820b4f325f5b1728bf0b4921bd58` |

## 阶段判定

G212R length gate passed, but manual review is still pending. G300 remains blocked.

下一步只能是 G212M manual review/adjudication：对 `artifacts/G212R/prepare/manual_sample.jsonl` 的 100 条样本给出 reviewer 判定，保存 review/adjudication 产物，并在全部通过后再写 freeze readiness manifest。
