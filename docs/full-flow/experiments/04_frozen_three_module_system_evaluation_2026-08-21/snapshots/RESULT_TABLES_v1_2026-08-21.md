# 最终结果表格格式（v1，已被 2026-08-22 版本取代）

这些模板提前固定结果怎样存，避免结果出来后只挑好看的指标。

## 表 1：三个完整系统的主结果

每个数据集单独一组；MuSiQue answerable 与 unanswerable 分开。

| 数据 | 系统 | correct_and_cited | answer | coverage | citation precision | citation recall |
|---|---|---:|---:|---:|---:|---:|
| HotpotQA | 共同基线 | — | — | — | — | — |
| HotpotQA | Generator 升级版（3 seed mean±SD） | — | — | — | — | — |
| HotpotQA | 最终三模块系统（3 seed mean±SD） | — | — | — | — | — |

数值源：`results/main_system_table.csv`。

## 表 2：每个模块带来的差值

| 数据 | 比较 | mean delta | seed SD | 95% query-cluster CI | 三个 seed 方向 |
|---|---|---:|---:|---:|---|
| — | Generator 升级版 − 共同基线 | — | — | — | — |
| — | 最终系统 − Generator 升级版 | — | — | — | — |
| — | 最终系统 − 共同基线 | — | — | — | — |

数值源：`results/module_effects_table.csv`。

## 表 3：Selector 是否真的在做有用的事

| 数据 | changed queries | 删除条数 | 删除 gold support | 错→对 | 对→错 | correct_and_cited 净变化 |
|---|---:|---:|---:|---:|---:|---:|
| — | — | — | — | — | — | — |

RGB counterfactual 另加：true-document drop、wrong-document drop、true-answer assertion、fake-answer assertion。

数值源：`results/selector_safety_table.csv`。

## 表 4：特殊安全结果

| 数据/子集 | 系统 | 无依据断言率 | 克制或 unverified | fake answer assertion | splitter degraded/failure |
|---|---|---:|---:|---:|---:|
| MuSiQue unanswerable | — | — | — | 不适用 | — |
| RGB counterfactual | — | — | — | — | — |

## 表 5：运行成本

| 数据 | 系统/seed | queries | 新生成 | 复用输出 | error | GPU hours | wall hours |
|---|---|---:|---:|---:|---:|---:|---:|
| — | — | — | — | — | — | — | — |

数值源：`results/runtime_table.csv`。

## 每个 CSV 的共同要求

- 一个单元格只存一个数，不把 `mean±SD` 字符串当机器数值；分别保存 `mean`、`std`、`variance`；
- 比例统一存 0–1，Markdown 渲染时转成百分比；
- dataset、subset、system_id、generator_seed、metric、n_queries、n_components 都有独立列；
- 缺失值写空并附 `missing_reason`，不写成 0；
- 每个 CSV 都能由 `aggregate_results.json` 重新生成；
- 每一行都能追溯到 dataset bundle manifest 和 raw file SHA-256。
