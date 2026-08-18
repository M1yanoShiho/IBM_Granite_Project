# G200 数据物化报告

**日期：** 2026-08-18  
**状态：** `COMPLETE / PRE-AUDIT PASS / G210 READY / NO TRAINING STARTED`  
**服务器 runtime：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G200-v2`

## 本阶段做了什么

G200 v2 只做训练数据预物化，不训练 Generator，不生成 utility labels，也不读取 sealed600 或 system held-out。

本阶段构造了四类数据：

1. 复用旧 G200 已审计的 NIAH train-fit 515 题；
2. 在旧 NIAH 1,023 题之外，按 parent-disjoint 规则筛出新 NIAH model-val，并用冻结 QA2D 生成离线目标；
3. 从 2Wiki official train 中，用 `answer`、`evidences`、`supporting_facts` 和 `context` 生成多跳原子事实链；
4. 从 2Wiki train split 中移除完成答案必需的 support，生成 target 固定为 `I don't know.` 的 unsupported rows。

完整 `train_cases.jsonl` 和 `validation_cases.jsonl` 分别为 58MB 和 17MB，只保存在服务器 runtime；Git 归档保存脚本、测试、manifest、ordered IDs、NIAH 新 model-val role/component、QA2D targets 和执行审计。

## 边界

- `training_started=false`；
- `sealed_or_heldout_read=false`；
- `dev_read=false`；
- gold/reference/provenance 只用于离线 target construction，不进入 runtime prompt；
- TRUE target audit、minimal support、人工抽样和长度/truncation 审计都属于 G210/G300，尚未通过；
- 因此本阶段只解锁 G210，不解锁训练。

## 结果

| 项目 | 数量 |
|---|---:|
| NIAH train answerable groups | 515 |
| NIAH 新 model-val candidates | 310 |
| NIAH 新 model-val QA2D answer-preserved | 307 |
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

全部 G200 预物化硬门通过：NIAH train >=400、NIAH 新 model-val >=100、2Wiki train >=400、2Wiki model-val >=100、unsupported update ratio 在 10%-15%、split leakage 为 0。

## 关键 SHA256

| 产物 | SHA256 |
|---|---|
| runtime `train_cases.jsonl` | `03b53fb049680691d7bf6f2da3d8e5ff77c461381474831085091c921c3ce87c` |
| runtime `validation_cases.jsonl` | `fa4db7e3c9a2c1674ae762ea7b9b9a7e300f46e46d366a5dff395e29a6ea5bd0` |
| `artifacts/G200/data/manifest.json` | `bb659070f99588312872b30ded5fb8adfd95a1449887ef6e5b5b054cfa2801c1` |
| `artifacts/G200/data/ordered_ids.json` | `a2813b7f456ac3fcc32209b69871249b470e864c24ca8d53970abf6bc38a78bc` |
| NIAH new model-val roles | `8c565a9b56d840b72af9ab572b0c66746426dcca9bb9ccda052391f677fd0da2` |
| NIAH new model-val component map | `e4aae598d9c83c5dcee2c0988466aeccef3c202aca785b4d730b77f220bd7cbc` |
| NIAH new model-val QA2D targets | `b3cd30a922f9ae9e00d96c8c033b865f47c7aedef9e977bfd22e70fa283f7ea9` |
| `artifacts/G200/G200_EXECUTION_AUDIT.json` | `2fb2b08a10e4fea360d7c6e6320027bd944f907b8dec8f045f00ab6ccfd708af` |

## 阶段判定

G200 数据预物化通过，可以进入 G210。G210 必须继续审计 support、citation remap、TRUE entailment、minimal support、unsupported support absence、split leakage 和人工样本。G210 通过前，不允许启动 G300 training implementation、Generator 训练、utility labels 或 held-out。
