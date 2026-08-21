# G218 systematic target repair report

**日期：** 2026-08-18  
**状态：** `COMPLETE / PRE-SAMPLE PASS / NO TRAINING STARTED`  
**输入状态：** `G216 PRE-SAMPLE PASS + G212M2 SAMPLE REVIEW FAIL`

## 做了什么

G218 不继续只删固定样本失败项，而是系统修复 target construction：

- 2Wiki answerable target 改为 deterministic title/subject anchoring；
- NIAH 只过滤可检测的 QA2D title truncation；
- 因为 2Wiki target text 被改写，重新执行 structural 和 frozen TRUE；
- 不启动 Generator 训练、不生成 utility labels、不读取 held-out/dev、不改变 TRUE checkpoint 或 threshold。

新增代码：

```text
scripts/full_flow_g218_target_repair.py
tests/scripts/test_full_flow_g218_target_repair.py
```

服务器测试：

```text
5 passed
```

## 结果

### Materialization

| 项 | 数值 |
|---|---:|
| train cases | 2,388 |
| validation cases | 315 |
| 2Wiki train/model-val answerable groups | 824 / 103 |
| NIAH train/model-val answerable groups | 515 / 212 |
| unsupported groups | 1,049 |
| unsupported update ratio | 0.1129292712 |
| split group/component overlap | 0 / 0 |
| removed NIAH QA2D cases | 1 |
| changed 2Wiki cases | 553 |
| unchanged 2Wiki cases | 374 |

Removed case:

```text
niah-new-modelval::1015  acted_on_title_truncation
```

### Structural + TRUE

| 检查 | 结果 |
|---|---:|
| structural cases | 2,703 / 2,703 PASS |
| TRUE worklist rows | 2,078 |
| TRUE entailed | 2,074 |
| TRUE not entailed | 4 |

TRUE not-entailed cases were all 2Wiki train-fit cases:

```text
2wiki::300945840bdc11eba7f7acde48001122
2wiki::6783469a0bdc11eba7f7acde48001122
2wiki::88142b760bdd11eba7f7acde48001122
2wiki::914b34940bda11eba7f7acde48001122
```

### Finalize

| 项 | 数值 |
|---|---:|
| final train cases | 2,384 |
| final validation cases | 315 |
| final 2Wiki train/model-val answerable groups | 820 / 103 |
| final NIAH train/model-val answerable groups | 515 / 212 |
| final unsupported groups | 1,049 |
| final unsupported update ratio | 0.1131729421 |
| final split group/component overlap | 0 / 0 |

## 判定

G218 通过自动数据门，且有明确积极信号：系统性 target repair 后，TRUE 只剔除 4 个 train case，2Wiki model-val 仍保持 `103 >= 100`。

但 G218 仍不是训练许可。它只产生新的 pre-sample bundle；G300 继续 blocked。下一步必须执行 G212R3 length/sample review。

## 产物

Runtime:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G218-v1
```

Git-archived artifacts:

| 产物 | SHA256 |
|---|---|
| `artifacts/G218/data/manifest.json` | `d9bcc40b346ec89c934bbc2f954230b06a89da3b98f79c5581147ad2b883ffc5` |
| `artifacts/G218/data/ordered_ids.json` | `804b785455f7a69d6c5671107fc7b878d2c42dd5995b3cdd0222e17d171f3d9a` |
| `artifacts/G218/structural/structural_summary.json` | `01fb842e4b8477631a2047aa7b4735a94035291d233b3ad39e5596f0bd5d3d1b` |
| `artifacts/G218/true/true_audit_manifest.json` | `4b39d73a32d10dc37745075b552613aef38c9f4c8742efe1b4f10442e1cffdde` |
| `artifacts/G218/pre-sample/manifest.json` | `a5c291e005ef1a9aa4d333079f583235e7a5c5f3ab4a87b9d073a59accdde2ca` |
| `artifacts/G218/pre-sample/ordered_ids.json` | `7175aa296596e604b99293e698a44af8ba90ce27a57789b2c069ffa5749f09d4` |
| `artifacts/G218/G218_EXECUTION_AUDIT.json` | `ffbb9f18bd4ae6a60a631f72ad3cd61d27ccd28cea3cc14d5790ff3bd066a941` |

Code hashes:

```text
8333346a07a41744353f07a8788f468f7e42c8c3e9a8901439c1ce346f2cb13b  scripts/full_flow_g218_target_repair.py
1ff4408beb81f4052b35a6a8e959073bb764a6e16a2e4fd2e00d7ae3cc2bc30c  tests/scripts/test_full_flow_g218_target_repair.py
```
