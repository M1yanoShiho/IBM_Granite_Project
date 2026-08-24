# G219 controlled target repair failure report

**日期：** 2026-08-18
**状态：** `FINALIZE FAIL / 2WIKI MODEL-VAL FLOOR FAILED / NO TRAINING STARTED`
**输入状态：** `G212M3 FAIL / 91 PASS / 9 FAIL / 0 UNCERTAIN`

## 做了什么

G219 尝试把 G212M3 暴露的问题系统化修复：

- 2Wiki answerable target 中，对 `country of origin`、`country of citizenship`、`place of birth` 关系写成更明确的 claim；
- NIAH 中，过滤 G212M3 暴露的 4 条明确 QA2D malformed case；
- 重跑 structural、TRUE 和 finalize。

这一步没有训练模型、没有生成 utility labels、没有读取 held-out/sealed/dev，也没有改变 Retriever 或 TRUE 阈值。

## 结果

| 阶段 | 结果 |
|---|---|
| G219 materialization | `PRE_AUDIT` |
| structural | `PASS`，2,695/2,695 cases 通过 |
| TRUE | 2,006 entailed / 63 not entailed |
| finalize | `FAIL` |

Finalize 后数据规模：

| 项 | 数值 |
|---|---:|
| train cases | 2,332 |
| validation cases | 305 |
| NIAH train/model-val | 512 / 211 |
| 2Wiki train/model-val | 771 / 94 |
| unsupported groups | 1,049 |
| unsupported update ratio | 11.6556% |
| split group/component overlap | 0 / 0 |

## 判定

G219 没有通过。失败门是 `twowiki_modelval_groups=false`：2Wiki model-val 从 103 降到 94，低于当前最低 100。

失败归因显示，TRUE 不通过集中在 G219 的 relation-explicit 目标句上，而不是 NIAH 或 split 问题。也就是说，G219 的修复方向抓到了样本问题，但修复方式过强：把关系写得更直白以后，有些 claim 不再被 frozen TRUE 认为由 evidence 直接支持。

这说明路线仍有积极信号，但不能进入 G300，也不能简单把 100 改成 94 来通过。下一步必须执行 G220 conservative target repair review，重新处理 TRUE 支持性与 sample 自洽性的冲突。

## 产物

Runtime:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G219-v1
```

Git-archived artifacts:

| 产物 | SHA256 |
|---|---|
| `scripts/full_flow_g219_target_repair.py` | `ca30abe6212dba99ca1f0a9154e7145aa8b3ce9b8ffc0aa579d142f439705159` |
| `tests/scripts/test_full_flow_g219_target_repair.py` | `f76a67449e9bd0add47e3f8eee5b9b37e6a27903bf6d40771747f89f04116dbc` |
| `artifacts/G219/data/manifest.json` | `85d1d4006ab6e1f6e40e1c4349fa754569a39a1948bc1dc992654bd1852cf71e` |
| `artifacts/G219/data/ordered_ids.json` | `598c89cf21b94d9141b3ebaa529b8e2ede83a97dffba3c2efd2d9424ab5ec831` |
| `artifacts/G219/structural/structural_summary.json` | `9bf6e836292d840a16814c7955f850417400b9276def4011bc2a7223bff146c2` |
| `artifacts/G219/true/true_audit_manifest.json` | `7898ea90433b47754ea90ee29d171916e684375763e4ff8ae9972c65c10b19e8` |
| `artifacts/G219/pre-sample/manifest.json` | `76b24405f675d1ea1fae188a4adabf08552e3ba29974524408aed6f8dbd770e1` |
| `artifacts/G219/pre-sample/ordered_ids.json` | `c9bc8b68c333d6338b738a3c3177bb46c16065ea0d11492b217355d0adaf8b09` |
| `artifacts/G219/triage/failure_triage.json` | `1168227240ea634aea70c98e2d13b659d7d4bc238fb74c5778c5d184fb4d45f6` |
| `artifacts/G219/G219_EXECUTION_AUDIT.json` | `956d6ac702ed00b3ac4d609c2031ad388d477a638e6dee99a95590ec9d22ec52` |

## 下一步

G220 不能直接训练。G220 必须先选择更保守的处理方式，例如只隔离不可同时满足 TRUE 与 sample 自洽的 target，或把 model-val floor 的统计含义重新论证清楚。任何门槛调整都必须作为独立计划修订归档，不能作为“凑通过”的临时改动。
