# Goal 1 Handoff — 数据与评分准备

**执行状态：** `COMPLETE / PASS / STOPPED 2026-08-22`  
**唯一目标：** 在不生成或评分正式 held-out 答案的前提下，完成数据隔离与五指标评分器准备，并给出 `PASS` 或 `FAIL`。

**实际交付：** [GOAL1_DATA_SCORER_READINESS.md](reports/GOAL1_DATA_SCORER_READINESS.md)；
最终结果为 `PASS`，未启动 Goal 2。

## 先读这些文件

1. `docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21/PLAN.md`
2. `docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21/RESULT_TABLES.md`
3. `docs/heldout-three-set-handover.md`
4. `docs/generator/heldout-preregistration.md`
5. `scripts/heldout_data.py`
6. `configs/heldout-sample.json`

第 3–6 项只用于核对数据来源、schema 和冻结信息，其中若包含旧版 generation、scoring、dry-run 或多系统运行步骤，**一律不得在 Goal 1 执行**。任何冲突都以本交接和 v4 `PLAN.md` 的 Goal 1 边界为准。

## 本目标必须完成

1. 核对 HotpotQA 400、MuSiQue answerable 400、RGB noise 300 的 ordered IDs、顺序与 hash；
2. 确保每个 query 只使用自己的候选池；
3. 将 runtime 输入与 scorer-only gold sidecar 物理分离；
4. sidecar 保留 gold answer、最小可评分 support unit 和 `component_id`；
5. 在 synthetic 或已揭示 development fixtures 上验证 `Ret./Sel./Ans./Cit./RAR`；
6. 验证空答案、非法引用、生成/评分失败仍留在共同分母并按协议记 0；
7. 写出数据 manifest、隔离说明、scorer validation report 和最终 `PASS/FAIL`。

## held-out 边界

允许确定性地物化正式 runtime bundle、scorer sidecar 和 manifest，但只能由程序核对 count、schema、ordered IDs 与 hash。禁止：

- 打印、浏览或抽查 held-out 问题、答案、support labels；
- 在 held-out 上运行 Retriever、Selector、Generator 或 MiniCheck；
- 根据 held-out 内容或结果修改任何方法；
- 接入四个 baseline；
- 启动 Goal 2。

日志不得包含 held-out 正文、答案或 gold labels。

## 必须提交的结果

建议将小型可审计产物保存在本实验目录：

- `artifacts/goal1_data_manifest.json`
- `reports/GOAL1_DATA_SCORER_READINESS.md`
- 与实现对应的自动测试

报告必须明确列出：三个数据集的 count/hash 是否一致、runtime 是否无 gold、候选池是否逐题隔离、五指标测试是否通过、失败分母规则是否通过，以及最终 `PASS` 或 `FAIL`。

## PASS 条件

- 三套 frozen IDs 分别为 400/400/300，且 ordered hash 与预注册一致；
- runtime schema 不含 gold answer/support/component 字段；
- runtime 与 scorer sidecar 只能通过 `query_id` 对齐，系统运行入口不能访问 sidecar；
- 每题候选池隔离测试通过；
- 五项指标及错误样本规则在 development fixtures 上均通过；
- 无 held-out generation、scoring 或内容暴露。

任一条件未满足即为 `FAIL`，留在 Goal 1 内修复和复测。

## 强制停止点

Goal 1 达到 `PASS` 或确认 `FAIL` 后立即结束任务，向用户交付报告。即使时间和算力仍有剩余，也不得开始 baseline 接线、smoke check 或正式实验。Goal 2 必须由用户在新的独立目标中启动。
