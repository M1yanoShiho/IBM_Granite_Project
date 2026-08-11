# Selector 风险控制实验跟踪表（v1）

**对应计划：** [`EXPERIMENT_PLAN_2026-08-11.md`](EXPERIMENT_PLAN_2026-08-11.md)

**冻结日期：** 2026-08-11

**规则：** 先登记、后运行；结果栏只能填写真实产物，不得预填预期结果。

## 1. 状态定义

- `TODO`：尚未开始。
- `RUNNING`：正在运行，必须填写 run id 和开始时间。
- `PASS`：全部退出门槛通过。
- `FAIL`：完成但未通过门槛；保留失败产物。
- `BLOCKED`：依赖或数据问题阻塞，不等同于模型失败。
- `CUT`：依据预注册停止条件主动停止。

## 2. 里程碑总表

| 阶段 | 目标 | 状态 | 进入条件 | 退出门槛 | 结果目录 |
|---|---|---|---|---|---|
| M0 | 协议、拆分、指标、hash 冻结 | TODO | 无 | G1 完整性部分 | `results/selector-crc-dual-head-v1/m0_protocol/` |
| M1 | TopK 与 count-matched 基线 | TODO | M0 PASS | 分母和逐问题结果可追溯 | `results/selector-crc-dual-head-v1/m1_baselines/` |
| M2 | scorer 可分性与 200q 资源基准 | TODO | M1 PASS | G2 | `results/selector-crc-dual-head-v1/m2_scorer/` |
| M3 | query-level CRC 主实验，seed 13 | TODO | M2 PASS | G3 | `results/selector-crc-dual-head-v1/m3_crc/` |
| M4 | source 消融与 seeds 42/73 | TODO | M3 PASS | G4-source + 多种子门槛 | `results/selector-crc-dual-head-v1/m4_source_and_seeds/` |
| M5 | sealed600 / 2Wiki heldout 一次性确认 | TODO | M4 PASS + 配置冻结 | G5 | `results/selector-crc-dual-head-v1/m5_formal/` |
| M6 | 生成端 defer/查证闭环 | TODO | M5 PASS | 单独方案 | `results/selector-crc-dual-head-v1/m6_e2e/` |

## 3. Run 队列

| Run ID | 阶段 | 内容 | 数据角色 | Seed | 预算 | 状态 | 结果/备注 |
|---|---|---|---|---:|---:|---|---|
| R001 | M0 | protocol-freeze：group split、overlap audit、metric schema、hash | 全部只读；不解封测试标签用于调参 | 20260811 | 0 GPU | TODO |  |
| R002 | M1 | TopK10 / TopK5 / count-matched random | calibration + decision-dev | 20260811 | 0 GPU | TODO |  |
| R003 | M2 | 200-query scorer feasibility 与资源基准 | train-modelval 子集 | 13 | ≤小样本推理/拟合 | TODO |  |
| R004 | M2 | dual-head seed 13 全量训练与 scorer 诊断 | train-fit + train-modelval | 13 | ≤1 训练单位 | TODO | 仅 R003 有可行迹象时 |
| R005 | M3 | fixed threshold / budget / marginal CP / query CRC | calibration → decision-dev | 13 | 复用 R004 | TODO | 不新增训练 |
| R006 | M4 | flat / dedup / source-group 消融 | decision-dev | 13 | 复用 R004 | TODO | 仅 R005 PASS |
| R007 | M4 | dual-head 训练 | train-fit + train-modelval | 42 | 1 训练单位 | TODO | 仅 R005 PASS |
| R008 | M4 | dual-head 训练 | train-fit + train-modelval | 73 | 1 训练单位 | TODO | 仅 R005 PASS |
| R009 | M4 | 三种子冻结评测 | calibration → decision-dev | 13/42/73 | 推理 | TODO | 阈值规则在运行前冻结 |
| R010 | M4 | 现实 source-dependent 100-case 标注 pilot | 新建真实多视图集 | n/a | 人工 | TODO | 双标 + 仲裁；只设计协议与功效 |
| R011 | M4 | 冻结规模的 source benchmark 正式双标与盲评 | 新建真实多视图集 | n/a | 人工 | TODO | 不与 pilot 数字混合 |
| R012 | M5 | NIAH sealed600 正式评测 | sealed600 | 13/42/73 | 推理 | TODO | 只允许一次正式运行 |
| R013 | M5 | 2Wiki heldout 正式评测 | heldout | 13/42/73 | 推理 | TODO | 只允许一次正式运行 |

## 4. Gate 核对表

### G1 — 完整性

- [ ] TopK10 与归档报告在舍入误差内一致。
- [ ] query/source parent/synthetic family 跨 split 交集为 0。
- [ ] conditional 与 unconditional harmful 分母均明确。
- [ ] 每个方法都有逐 query、逐 candidate 预测。
- [ ] count-matched random 按每个 query 的实际保留数匹配。
- [ ] p-value 已使用 plus-one 修正，无 `p=0.0`。
- [ ] Git commit、数据/model/config hash 已记录。

**G1 状态：** TODO

**证据路径：**

**判定人/日期：**

### G2 — Scorer 可行性（只看 train-modelval）

- [ ] NIAH conditional harmful reduction ≥3 pp。
- [ ] NIAH required recall loss ≤5 pp。
- [ ] 2Wiki supporting recall loss ≤5 pp。
- [ ] PR-AUC、Brier、ECE、rank/hop 分层、截断率已报告。
- [ ] 如果 G2 失败，已停止自动 truth-filtering，而非继续扫阈值。

**G2 状态：** TODO

**证据路径：**

**判定人/日期：**

### G3 — 单种子 query-level 风险控制（decision-dev）

- [ ] NIAH harmful reduction ≥3 pp，paired 95% CI upper `<0`。
- [ ] NIAH recall loss ≤3 pp。
- [ ] 2Wiki recall loss ≤3 pp。
- [ ] 两数据集 recall 差的 95% CI lower 均 ≥−5 pp。
- [ ] chain failure point ≤3%，95% CI upper ≤5%。
- [ ] 优于逐 query count-matched random。
- [ ] calibration 与 decision-dev 完全隔离。

**G3 状态：** TODO

**证据路径：**

**判定人/日期：**

### G4 — 来源机制与三种子

- [ ] source-group 优于 flat。
- [ ] source-group 优于纯文本 dedup；否则撤销 C2。
- [ ] false corroboration rate 显著下降。
- [ ] 两数据集三种子平均 recall loss ≤3 pp，单种子 ≤5 pp。
- [ ] 平均 harmful reduction ≥3 pp，所有种子方向一致。
- [ ] chain failure 平均 ≤3%，单种子 ≤5%。

**G4 状态：** TODO

**证据路径：**

**判定人/日期：**

### G5 — 正式确认

- [ ] 配置、checkpoint 和规则在解封前已 hash 冻结。
- [ ] sealed600 满足 G3。
- [ ] 2Wiki heldout 满足 G3。
- [ ] 未在正式结果后调阈值或重新定义分母。
- [ ] 失败时生产默认保持 TopK10。

**G5 状态：** TODO

**证据路径：**

**判定人/日期：**

## 5. 每次运行的必填记录模板

复制本节，为每个 Run 建一个记录文件：

```text
Run ID:
开始/结束时间:
负责人:
Git commit:
工作树是否干净及例外:
配置文件与 SHA-256:
数据 manifest 与 SHA-256:
模型名称/revision/checkpoint hash:
数据角色（train-fit/modelval/calibration/decision/sealed）:
Seed:
命令或入口:
GPU/CPU/内存:
Wall time / 峰值显存:
逐问题预测路径:
聚合指标路径:
CI/统计检验路径:
对应 Gate:
Gate 结果（PASS/FAIL/BLOCKED/CUT）:
失败分析:
是否允许下一 Run（是/否及理由）:
```

## 6. 决策日志

| 日期 | 决策 | 依据 | 影响的 Run/主张 |
|---|---|---|---|
| 2026-08-11 | 第一版只从 TopK10 删除，不从 11–20 补位 | 避免把删除与二次检索混为一个变量 | R003–R013 / C1 |
| 2026-08-11 | harmful 与 utility 改为双头；高高组合进入 DEFER | 旧三分类不能表达“相关但错误” | R003–R009 / C1 |
| 2026-08-11 | 用 query-level recall 与 chain loss 做 CRC | 单片段阈值不能保护多跳集合 | R005–R013 / C1 |
| 2026-08-11 | 保留 count-matched random 为强制基线 | 排除“只是删得更多”的解释 | R002 以后 / C1、C2 |
| 2026-08-11 | sealed/heldout 只正式运行一次 | 防止测试集变成调参集 | R012–R013 |

后续任何门槛、数据角色或主要指标变更，必须先在本表新增带日期的决策，再执行受影响实验；不得覆盖旧记录。
