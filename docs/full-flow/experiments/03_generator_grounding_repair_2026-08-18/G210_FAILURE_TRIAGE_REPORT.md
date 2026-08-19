# G210 failure triage 报告

**日期：** 2026-08-18  
**状态：** `COMPLETE / DIAGNOSTIC ONLY / G210 STILL FAILS`  
**服务器 runtime：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G210-v1/failure-triage`

## 目的

本报告只解释 G210 为什么失败，不改变任何 gate，不重跑 TRUE，不重切数据，不启动人工审计或训练。

## 结论

G210 的失败原因保持不变：

```text
2Wiki model-val answerable groups = 76 < required 100
```

更细地看，2Wiki model-val 在 TRUE 过滤前有 136 个 case；过滤后：

| 项目 | 数量 |
|---|---:|
| case-level entailed | 76 |
| case-level not entailed | 60 |
| pair-level entailed | 203 |
| pair-level not entailed | 71 |

NIAH 新 model-val 也有 TRUE 不通过，但过滤后仍有 215 个 case，高于最低门 100，因此没有触发硬门失败。

## 主要失败类型

2Wiki model-val 的失败 pair 主要集中在：

| relation | failed pairs |
|---|---:|
| country | 16 |
| publication date | 16 |
| country of citizenship | 14 |
| country of origin | 6 |
| date of birth | 4 |
| date of death | 4 |
| performer | 3 |
| spouse | 3 |

这说明失败主要来自“official triple -> templated atomic fact -> TRUE entailment”的目标构造/判定匹配，而不是 split overlap、citation remap 或 unsupported removal；那些结构性检查已经全部通过。

## 归档

| 产物 | SHA256 |
|---|---|
| `artifacts/G210/failure-triage/triage_summary.json` | `273ea6cc288a43f01be1168d08f365fed1d934733122955cbe529af05bbf0a48` |
| `artifacts/G210/failure-triage/failure_samples.jsonl` | `3f628092d2f38db4d1a4b1357be32f08b1f618017450223bb22242da95d4d333` |

## 阶段影响

G210 仍然是 hard data gate failure。按当前计划，G300、Generator 训练、utility labels、S 阶段和 I 阶段都不能继续。下一步如果要推进，必须由用户明确批准新的数据修订计划，例如修改 2Wiki target construction/audit protocol 后重新从 G200/G210 做受控归档；不能在当前失败数据上训练。
