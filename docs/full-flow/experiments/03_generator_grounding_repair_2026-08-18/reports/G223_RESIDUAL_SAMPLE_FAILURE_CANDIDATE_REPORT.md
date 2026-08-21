# G223 residual sample-failure candidate report

**日期：** 2026-08-18
**状态：** `CONTROLLED CONTINUATION READY / G300 LIMITED DRAFT ENTRY / NO TRAINING STARTED`
**输入状态：** `G222 CONTROLLED CONTINUATION`

## 做了什么

G223 只处理 G212M5 暴露的 3 条剩余失败。三条失败的证据都不能直接支持缺失关系，因此本阶段不重写 target，而是做 deletion-only residual quarantine。

隔离规则：

- 失败的 2Wiki train answerable case 直接隔离；
- 对应的 unsupported train counterpart 同步隔离；
- 失败的 2Wiki model-val answerable case 直接隔离；
- 不改 Retriever、TRUE、Selector、Generator runtime；
- 不启动训练、不生成 utility labels、不读取 held-out/sealed/dev。

第一次正式运行因为缺少 `PYTHONPATH` 失败，错误为 `ModuleNotFoundError: full_flow_g216_sample_review_repair`；该次没有写出 G223 产物。补齐脚本路径后正式运行成功。

## 隔离内容

| source sample_index | case_id | action |
|---:|---|---|
| 14 | `2wiki::7e38489c0bda11eba7f7acde48001122` | quarantine train answerable |
| 14 | `unsupported::2wiki::7e38489c0bda11eba7f7acde48001122` | quarantine unsupported counterpart |
| 17 | `2wiki::c083071908cf11ebbd95ac1f6bf848b6` | quarantine train answerable |
| 17 | `unsupported::2wiki::c083071908cf11ebbd95ac1f6bf848b6` | quarantine unsupported counterpart |
| 56 | `2wiki::23257d6e087c11ebbd69ac1f6bf848b6` | quarantine model-val answerable |

## 结果

| 项 | 数值 |
|---|---:|
| train cases | 2,370 |
| validation cases | 302 |
| NIAH train answerable groups | 511 |
| NIAH model-val answerable groups | 207 |
| 2Wiki train answerable groups | 815 |
| 2Wiki model-val answerable groups | 95 |
| 2Wiki unsupported train groups | 1,044 |
| unsupported update ratio | 0.113392 |
| split group overlap | 0 |
| split component overlap | 0 |

G223 按 G222 修订后的 controlled continuation gate 通过：

| gate | status |
|---|---|
| NIAH train >= 400 | PASS |
| NIAH model-val >= 100 | PASS |
| 2Wiki train >= 400 | PASS |
| 2Wiki model-val >= 95 | PASS |
| unsupported ratio in [0.10, 0.15] | PASS |
| split group overlap = 0 | PASS |
| split component overlap = 0 | PASS |

## 判定

G223 不是 clean freeze：固定样本并不是 100/100 clean pass，2Wiki model-val screen 也从 96 进一步降到 95。

但 G223 是 controlled continuation ready：已知 3 条失败被隔离，NIAH、unsupported、split 和长度风险没有新增问题，且没有读取 held-out。下一步可以进入 G300 的 limited draft entry，但报告必须持续声明：

- 这不是 clean 100/100 freeze；
- 2Wiki model-val screen 是 95，内部验证方差风险更高；
- 不能把 G300 后续结果写成强统计结论；
- 最终 held-out 仍需要 SystemF 冻结后单独授权。

## 验证

服务器测试通过：

```text
2 passed in 0.02s
```

测试范围：

- G223 会隔离残余失败和 train unsupported counterpart；
- G223 拒绝没有 G222 controlled continuation 授权的输入。

## 产物

Runtime:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G223-v1/data
```

Git-archived artifacts:

| 产物 | SHA256 |
|---|---|
| `scripts/full_flow_g223_residual_sample_failure_candidate.py` | `957a1067134aa0e5a1a1ef6dc8a73397be232218836bc20e8f78378941ac3050` |
| `tests/scripts/test_full_flow_g223_residual_sample_failure_candidate.py` | `0354004874793749d1e7f639bcd29d20fcd8771baa95a05093bd101fd92e67ee` |
| `artifacts/G223/data/manifest.json` | `9213eb32f1234290ac5d25c66c51756927adaff58ff4cd4cf2f70ece2312b30b` |
| `artifacts/G223/data/ordered_ids.json` | `b3bd73290f9753ada6b6f79b7d6758d755b6eec1e42035936b76ceee90b5b283` |
| runtime `train_cases.jsonl` | `88e5592ed796e756fb836c9faee22393a0fa56c57b50eae69c449bd78c438635` |
| runtime `validation_cases.jsonl` | `f896e8a92fe5e36757249f1a22d3c55f579233879beca229341833137f95cb67` |
| `artifacts/G223/G223_EXECUTION_AUDIT.json` | `4c9c2f9040ed558371dd9ffdad310bee765f1a3aa7bfca15d13a77b125a4d6ca` |

## 下一步

下一步是 G300 limited draft entry：只允许按 G223 manifest 使用 controlled-continuation candidate 启动 Generator draft LoRA 训练实现/烟测。不得改 final held-out 边界，也不得把该入口写成 clean freeze。
