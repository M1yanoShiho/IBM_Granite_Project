# G212 length/manual audit report

**日期：** 2026-08-18  
**状态：** `FAIL / LENGTH / MANUAL REVIEW NOT STARTED / NO TRAINING STARTED`  
**服务器 runtime：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G212-v1`

## 本阶段做了什么

G212 对 G210R2 pre-manual 数据做训练前长度审计，并准备固定分层人工审计样本：

1. 使用冻结 Granite 4.1-3B tokenizer 计算每个 prompt + target 的训练长度；
2. 固定 `max_length=2304`，检查是否存在会被截断的训练/验证例子；
3. 按 5 个 stratum 各抽 20 条，生成 100 条 manual review packet；
4. 记录所有输入、脚本、模型 snapshot 和输出 SHA256。

本阶段没有训练 Generator，没有生成 utility labels，没有读取 sealed600、HotpotQA、MuSiQue-Full、RGB 或 official dev。由于长度审计失败，manual review 未启动，G300 仍 blocked。

## 长度审计结果

| 项目 | 数量 |
|---|---:|
| cases | 2,717 |
| examples | 11,348 |
| max_length | 2,304 |
| over max_length examples | 4 |
| affected cases | 1 |
| min length | 309 |
| p50 | 1,345 |
| p90 | 1,842 |
| p95 | 1,882 |
| p99 | 1,966 |
| max | 2,528 |
| truncation rate if encoded at 2304 | 0.03525% |

超过 2,304 token 的 4 个 examples 全部来自同一个 2Wiki train answerable case：

```text
case_id = 2wiki::b779ecdc08c411ebbd8eac1f6bf848b6
variants = support_first, support_last, support_middle, topk
max observed length = 2528
```

validation split 没有超长；NIAH 没有超长；unsupported examples 没有超长。

## Stratum 结果

| Stratum | examples | p95 | max | over max_length |
|---|---:|---:|---:|---:|
| 2wiki train answerable | 4,140 | 1,645 | 2,528 | 4 |
| 2wiki train unsupported | 1,053 | 1,360 | 1,927 | 0 |
| 2wiki model-val answerable | 530 | 1,619 | 1,943 | 0 |
| NIAH train answerable | 4,120 | 1,906 | 2,120 | 0 |
| NIAH model-val answerable | 1,505 | 1,918 | 2,030 | 0 |

## Manual packet

G212 已生成人工审计样本包，但由于长度审计失败，没有进入 reviewer 判定：

| Stratum | rows |
|---|---:|
| 2wiki train answerable | 20 |
| 2wiki train unsupported | 20 |
| 2wiki model-val answerable | 20 |
| NIAH train answerable | 20 |
| NIAH model-val answerable | 20 |

所有 sample rows 当前都保留 `review_decision=PENDING`。这些样本不能被当作 manual audit pass。

## 关键 SHA256

| 产物 | SHA256 |
|---|---|
| runtime G212 script | `70fd47fa81b04bbe2c90770eb5080d869761f5fa77aa7bb53f2ae2e4e86aafe5` |
| G212 execution audit | `d5c067a0388c73a194ca15b702a833ac9376667504453ecf5b6742c3dd2a4e9d` |
| prepare manifest | `d703b2a455f1a5eb914af2ba0d1b6bd31f504b04b2c22bc7599fc8da35ae12fc` |
| length audit | `48ed8aa209e01111789ffca1af65bc4eed73a2271b5838df351ec9089f8ef1b6` |
| length rows | `153b624301e947c6b8bf5cdd48b85f68be0ad614f25250f7c44cbae4dc8fcbb9` |
| manual sample summary | `fd58aa15b79d0090d155f7273e44387a9f269c7ae7137e9d14f88ff85f35b1a7` |
| manual sample | `9434da2434ea10dad5e1fe7548ab54745e11820b4f325f5b1728bf0b4921bd58` |

## 阶段判定

G212 failed the length gate. 这不能解锁 G300，也不能通过提高 `max_length` 或忽略 4 条超长 examples 来继续训练。

但失败非常集中：只有 1 个 2Wiki train group 导致超长，且该 group 有对应 unsupported case。按原计划“审计不通过就排除并记录”的原则，下一步允许执行 G214 controlled length repair：成组排除这个 overlength train group 及其 unsupported counterpart，重新生成 revised pre-manual bundle，并再次运行 G212R length/manual audit。

如果 G214/G212R 后仍有超长或 manual review 不通过，G300 继续 blocked。
