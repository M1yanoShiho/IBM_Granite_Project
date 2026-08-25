# 冻结三模块系统分目标跟踪表（v4）

**状态：** `MASTER ROADMAP FROZEN / GOALS 1–5 COMPLETE / FINAL PASS`

## 总规则

每一行是一个防偏移执行目标，同时只能有一个 active goal。按用户
2026-08-22 的最新指令，当前目标 `PASS` 后立即冻结交付物、建立下一 active
goal 并继续；`FAIL` 留在当前目标内修复。

| Goal | 独立目标 | 核心交付物 | 当前状态 |
|---:|---|---|---|
| 1 | 数据与评分准备 | data manifest、runtime/gold 隔离、scorer validation、PASS/FAIL | **COMPLETE / PASS / STOPPED 2026-08-22** |
| 2 | 五系统接线与开发集检查 | 10 configs、model manifest、smoke report、PASS (= READY)/FAIL | **COMPLETE / PASS 2026-08-22** |
| 3 | 主系统比较 | 7,700 outputs、Table 1、paired CI、PASS/FAIL | **COMPLETE / PASS 2026-08-22** |
| 4 | 模块消融 | 3,300 outputs、Table 2、paired CI、PASS/FAIL | **COMPLETE / PASS 2026-08-22** |
| 5 | 最终统计与核验 | final tables、CSV/JSON、audit、FINAL PASS/FAIL | **COMPLETE / FINAL PASS 2026-08-22** |

## 已完成目标：Goal 1

Goal 1 只处理数据结构和评分器：逐题候选池、runtime/scorer 物理隔离、固定 IDs/hashes、五指标 development validation 和失败规则。

明确禁止：运行正式答案生成、查看或评分 held-out 结果、接入 baseline、启动 Goal 2。

Goal 1 的完整交接见 [GOAL_1_HANDOFF.md](GOAL_1_HANDOFF.md)。

实际结果为 **PASS**，见 [数据与评分 readiness 报告](reports/GOAL1_DATA_SCORER_READINESS.md) 和
[机器可读 data manifest](artifacts/goal1_data_manifest.json)。Goal 2 已由用户启动并完成。

## 已完成目标：Goal 2

Goal 2 冻结并验证了完整 10-arm 矩阵、模型/权重身份、共同运行预算、统一
runner 和 scorer。确定性 revealed-development smoke 为 10/10 PASS；真实模型
GPU smoke 为四个 baseline 4/4 PASS，Ours 三个 seed 复用既有 reload 与锁定
qualification 证据。正式 held-out 输出目录在 Goal 2 全程为空。

完整交付见 [Goal 2 readiness 报告](reports/GOAL2_SYSTEM_WIRING_READINESS.md)、
[10-arm smoke manifest](artifacts/goal2_smoke_manifest.json) 和
[真实 baseline smoke](artifacts/goal2_real_baseline_smoke.json)。

## 已完成目标：Goal 3

Goal 3 的七臂正式 runner、冻结后 MiniCheck 评分、五指标、Table 1 和 paired
component-cluster bootstrap 已实现并通过本地与服务器自动测试。三套 sealed
runtime 与 scorer-only sidecar 已按物理目录隔离并完成六个文件的哈希复核；公开
Selector backbone 与 TRUE 也已缓存并通过哈希。

VPN 恢复后，原教师服务器 `it097952` 上的四组冻结自定义模型实体被找回；七个
文件的源端和 BluePebble 目标端 SHA-256 均与预注册清单匹配，文件权限为 `0600`。
HotpotQA、MuSiQue answerable、RGB noise 的正式 preflight 均 PASS。首轮单作业七臂
bundle (`18673236`、`18673237`、`18673238`) 揭示了统一的 prompt-budget 运行时错误；
该轮已按整包规则作废，且 scorer 从未启动。修复保持冻结的 2,304-token 上限，统一
采用按排名的最大完整 evidence 前缀；7,700 个 sealed arm-query 上的纯机器 token
审计为零 overflow。attempt B 因提交时把 Selector 目录而非具体 checkpoint 文件传给
preflight，在 6 秒内、数据读取前统一退出。attempt C 的 runtime 参数预检查又发现
指向目录而非数据集文件，因此三个排队作业在 0 秒时被主动取消。最终参数的独立
preflight 已全部 PASS（400/400/300），新鲜完整 attempt D 已作为 jobs `18674672`、
`18674673`、`18674674` 启动并在同一 A100 节点运行；旧输出不复用、不跨作业混合。证据见
[prompt-budget repair](reports/GOAL3_PROMPT_BUDGET_REPAIR.md)、
[Goal 3 execution readiness](reports/GOAL3_EXECUTION_READINESS.md) 和
[机器审计](artifacts/goal3_execution_readiness.json)。该段记录的是 Goal 3 正式
attempt D 启动前的历史状态；Goal 3 完成后已按接力规则进入 Goal 4。

最终 fresh attempt D 三个 Slurm jobs `18674672`、`18674673`、`18674674` 均
`COMPLETED 0:0`。21 个 dataset-arm 单元覆盖全部 frozen IDs，共 7,700 条输出，
每臂 runtime errors 为 0；三份 generation manifest 和三份 score manifest 均 PASS。
Table 1、三 seed mean ± sample SD、12 个 paired RAR bootstrap CI 和最终逐题/hash
审计均 PASS。执行有效性为 PASS，但科学结论是不支持 Ours 的 RAR 优越性。详见
[Goal 3 results](reports/GOAL3_MAIN_SYSTEM_RESULTS.md) 和
[Goal 3 PASS manifest](artifacts/goal3_pass_manifest.json)。

## 已完成目标：Goal 4

Goal 4 的三个正式 jobs `18674854`、`18674855`、`18674856` 均 `COMPLETED 0:0`。
三个消融产生精确 3,300 个新答案；Table 2 的 Full 行复用 Goal 3 冻结 Ours seed13，
没有重新生成。三份 generation manifest 与三份 score manifest 全部 PASS；一个
HotpotQA Dense-Retriever 生成错误保留在共同分母（0.25%，低于 1% guard），其余
dataset-arm 单元零 runtime errors。4,400 行逐题表、9 个 paired RAR CI、聚合重算
和 hash 审计全部 PASS。科学结论是 Retriever/Selector 替换不改变 RAR，而 Direct
Generator 在 HotpotQA 和 MuSiQue 上优于 Full。详见
[Goal 4 results](reports/GOAL4_MODULE_ABLATION_RESULTS.md) 和
[Goal 4 PASS manifest](artifacts/goal4_pass_manifest.json)。

## 已完成目标：Goal 5

Goal 5 没有产生新答案或调用 scorer。它从冻结逐题文件独立重算 Table 1 的 mean 与
sample SD、Table 1/2 全部聚合值和 21 个 paired bootstrap 单元，并验证 1,100 行
Goal 4 Full 与 Goal 3 seed13 指标完全一致。最终生成 135 行 `summary_metrics.csv`、
统一 JSON、Markdown/LaTeX 两表、最终报告和跨 artifact hash 审计；编译器连续两次
返回 `FINAL PASS`，Goal 5 的 4 项交付测试通过。详见
[FINAL_REPORT.md](FINAL_REPORT.md)、[final audit](results/final_audit.json) 和
[Goal 5 PASS manifest](artifacts/goal5_final_pass_manifest.json)。

Experiment 04 的五个目标现已全部完成，当前无 active 后续目标。

## 冻结运行量

| Formal group | Outputs |
|---|---:|
| Goal 3：四个 baseline | 4,400 |
| Goal 3：Ours 三个 Generator seeds | 3,300 |
| Goal 4：三个单模块消融 | 3,300 |
| **Total** | **11,000** |

`Full` seed 13 在 Table 1 与 Table 2 之间复用。

## 时间预算

| Goal | Estimate |
|---:|---:|
| 1 | 2–4 h |
| 2 | 4–8 h |
| 3 | 18–30 h |
| 4 | 10–18 h |
| 5 | 2–4 h |

两张 A4000 可持续使用、依赖顺利且无需整包重跑时，整体目标约 2–3 个自然日；安全排期为 3–4 个自然日。
