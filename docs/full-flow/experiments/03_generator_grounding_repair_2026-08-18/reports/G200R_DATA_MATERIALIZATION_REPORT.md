# G200R 数据修订物化报告

**日期：** 2026-08-18  
**状态：** `COMPLETE / PRE-AUDIT PASS / G210R READY / NO TRAINING STARTED`  
**服务器 runtime：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G200R-v1`

## 本阶段做了什么

G200R 是 G215 授权后的数据修订阶段，不训练 Generator，不生成 utility labels，也不读取 sealed600、system held-out 或 2Wiki official dev。

本阶段只修 2Wiki target construction：

```text
旧 G200/G210:
official triple -> "X's relation is Y."

G200R:
official supporting fact sentence -> cited target sentence
```

也就是说，2Wiki 的可训练 atomic target 现在使用 official `supporting_facts` 指向的原 support sentence，并用 `target_construction=support_sentence_aligned_v1` 记录。`official_evidences` 仍保留为审计元数据，但不再强行把每个 relation triple 改写成固定英文模板。

这样做的目的不是降低 TRUE 门槛，而是消除 G210 triage 指出的模板/审计错配：`country`、`publication date`、`country of citizenship` 等 relation 用旧模板时，经常和原句表达不贴合。

## 边界

- `training_started=false`；
- `sealed_or_heldout_read=false`；
- `dev_read=false`；
- gold/reference/provenance 只用于离线 target construction；
- runtime prompt 仍只包含问题和 evidence text；
- TRUE checkpoint、TRUE threshold 和 judge 角色未改变；
- G210R structural/TRUE/manual/length audit 尚未执行；
- 因此本阶段只解锁 G210R，不解锁 G300 训练。

## 结果

| 项目 | 数量 |
|---|---:|
| NIAH train answerable groups | 515 |
| NIAH 新 model-val groups | 307 |
| 2Wiki train answerable groups | 1,075 |
| 2Wiki model-val answerable groups | 136 |
| 2Wiki train unsupported groups | 1,075 |
| train cases | 2,665 |
| validation cases | 443 |
| answerable train updates | 9,495 |
| unsupported updates | 1,075 |
| unsupported update ratio | 10.1703% |
| split group overlap | 0 |
| split component overlap | 0 |
| 2Wiki target construction | `support_sentence_aligned_v1` |

G200R 预物化硬门全部通过：NIAH train >=400、NIAH 新 model-val >=100、2Wiki train >=400、2Wiki model-val >=100、unsupported update ratio 在 10%-15%、split leakage 为 0。

抽查显示 1,211 个 2Wiki answerable case 全部带有 `semantic_sentences` 和 `target_construction=support_sentence_aligned_v1`。

## 关键 SHA256

| 产物 | SHA256 |
|---|---|
| runtime `train_cases.jsonl` | `239f274b362ac8467c865a6f16eeb14b032a37ea539ea465e636ea7aa11d8db8` |
| runtime `validation_cases.jsonl` | `24e1a618a7a4c667bfcd1392d767033e107894285a09eb4e52d9efad021b9e9f` |
| `artifacts/G200R/data/manifest.json` | `0121e897431ba7e87b4fa4b1918fa6d357ec3341c18201a2271dafbb97a8d564` |
| `artifacts/G200R/data/ordered_ids.json` | `a2813b7f456ac3fcc32209b69871249b470e864c24ca8d53970abf6bc38a78bc` |
| `artifacts/G200R/G200R_EXECUTION_AUDIT.json` | `f430a7ca7318cc4dc4fa1e5ff994230ea2ecec64937ad1c7521997166e77d274` |
| runtime code `full_flow_g200_v2.py` | `58748b56040f3b97dd63011b2606153f61387e3265d4b01269fadc11f0c44597` |

## 命令与错误记录

- 第一次尝试使用服务器默认 `/usr/bin/python3`，失败原因是 Python 3.8 环境缺少当前项目需要的 `typing.Annotated`；
- 数据目录当时仍为空；
- 随后使用 `/scratch/fl25387/IBM_Granite_Project_latest/envs/selector_mis_py311/bin/python` 成功完成物化；
- 本阶段没有模型调用，没有训练。

## 验证

- 本地语法检查：`scripts/full_flow_g200_v2.py`、`scripts/full_flow_g210_audit_v2.py` pass；
- 本地单元测试：`tests/scripts/test_full_flow_g200_v2.py` 与 `tests/scripts/test_full_flow_g210_audit_v2.py` 共 4 个测试 pass；
- 服务器 runtime SHA256 与 Git 归档小产物一致；
- 服务器 runtime 中 `train_cases.jsonl` 为 2,665 行，`validation_cases.jsonl` 为 443 行。

## 阶段判定

G200R 通过 pre-audit，可以进入 G210R。G210R 必须重新执行 structural audit、TRUE audit、finalize、人工抽样、minimal support 和 length/truncation 审计。G210R 通过前，仍不允许启动 G300、Generator 训练、utility labels、S/I/H 或 held-out。
