# G214 controlled length repair report

**日期：** 2026-08-18  
**状态：** `COMPLETE / PRE-MANUAL PASS / G212R REQUIRED / NO TRAINING STARTED`  
**服务器 runtime：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G214-v1`

## 本阶段做了什么

G214 只针对 G212 length gate 发现的唯一超长 train group 做受控修订：

```text
group_id = b779ecdc08c411ebbd8eac1f6bf848b6
```

成组排除的 train cases：

| case_id | 类型 | variants |
|---|---|---:|
| `2wiki::b779ecdc08c411ebbd8eac1f6bf848b6` | answerable | 5 |
| `unsupported::2wiki::b779ecdc08c411ebbd8eac1f6bf848b6` | unsupported counterpart | 1 |

本阶段没有提高 `max_length=2304`，没有改变 TRUE checkpoint/threshold，没有改 manual sample 规则，没有训练 Generator，没有生成 utility labels，也没有读取 sealed600、held-out 或 official dev。

## 修订后数据门

| 项目 | 数量 |
|---|---:|
| train cases | 2,394 |
| validation cases | 321 |
| NIAH train answerable groups | 515 |
| NIAH model-val answerable groups | 215 |
| 2Wiki train answerable groups | 827 |
| 2Wiki model-val answerable groups | 106 |
| 2Wiki unsupported groups | 1,052 |
| answerable train updates | 8,255 |
| unsupported updates | 1,052 |
| unsupported update ratio | 11.3033% |
| split group overlap | 0 |
| split component overlap | 0 |

所有 revised pre-manual gates 仍通过。2Wiki model-val 保持 `106 >= 100`，因此这次排除没有破坏验证稳定性门。

## 关键 SHA256

| 产物 | SHA256 |
|---|---|
| runtime G214 script | `fb9a7c731fad72be4d2821719bfef066fe619d65e05cc283928a92cf7afcf2e0` |
| G214 execution audit | `548bbe515e37c776eb5bdbefd40802ab0a7b04f4b01f84266d589a32b53a5c20` |
| revised manifest | `d9b59e1ab660d65643979bda29bcf694a2004954bee9240bd36a906324eb1ffb` |
| revised ordered IDs | `23aba0e3e78a0b4947de3fbfa24e8e735acdc147693325555b6fcca5414dbb84` |
| runtime revised train cases | `a387baa78f05c45f5952e1b695b587ea91736d041d384ce80f689f6ae056a9f2` |
| runtime revised validation cases | `cd2ba2309349ce99665701ac05077ed522c5cdbe1a1660dfd742a3d6a292a6e1` |

## 归档产物

Git 归档的小型可审计产物：

```text
artifacts/G214/
```

完整 revised train/validation cases 只保留在服务器 runtime：

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G214-v1/data/train_cases.jsonl
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G214-v1/data/validation_cases.jsonl
```

## 阶段判定

G214 完成并得到 revised `PRE_MANUAL_PASS` bundle，但它本身不等于数据冻结。下一步必须对 G214 bundle 执行 G212R revised manual/length audit。

只有 G212R 的 length gate 通过，并且 manual review 完整通过后，G300 才能启动。
