# 发现与决策：正式发布版仓库

## 需求

- 在同一个 GitHub 仓库中整理一个可以提交毕设并正式公开的干净版本。
- 不永久删除旧代码、结果或过程文档。
- 保留团队合作和研究开发历史，旧版本公开可见没有问题。
- 整理后的正式版本最终接入默认 `main`。
- 计划必须可拆成独立 Goal 分步执行，每步有验收门。
- 用户已确认目标结构并授权 G1–G8 在各自验收通过后按顺序接替执行。
- 完成前必须实际运行安装、CPU smoke、测试、静态检查、构建、结果一致性及 clean-clone 验证。

## 本地仓库发现

- 仓库当前为公开仓库，默认分支是 `main`。
- `origin/main` 是当前开发线的祖先，当前开发线相对 main 约领先 898 个提交。
- 正式三模块 runtime 已在 `refactor/three-module-baseline` 接线并推送。
- 当前工作区仍有大量 Experiment 04/05 modified/untracked 工作，整理前必须先安全冻结。
- 当前 tracked tree 约 301.3 MiB。
- `src` 约 1.5 MiB、`tests` 约 1.4 MiB、`scripts` 约 1.6 MiB。
- `docs` 约 111.8 MiB、`results` 约 176.9 MiB、`runs` 约 7.3 MiB。
- 约 98% 的 tracked tree 是文档、结果和运行产物，而非正式源码。
- 仓库约有 563 个文档和 138 个脚本；包括大量 artifacts、snapshots、plans、trackers 和阶段脚本。
- 约 242 个 tracked 文件含个人/HPC 绝对路径或账号标识；运行配置范围内只发现一处旧 CLI 路径引用。
- 启发式扫描未发现明显已提交密钥，但正式 release 仍需独立 secret scan。
- 当前仓库没有 LICENSE，GitHub description/topics 为空。
- 根 README 过短，且旧统一说明仍以 TopK/Corroboration 为主线，与最终 trained NLI Selector 不一致。
- 当前 Python package 提供三模块 pipeline，但没有 HTTP API。
- GitHub 全量 CI 最近为 1755 passed、22 skipped、1 failed；失败来自旧 G000 README/TRACKER 文本状态验证。

## G1 冻结审计

- 当前是普通 checkout（`.git` 与 common dir 相同），分支为 `refactor/three-module-baseline`，不是 linked worktree 或 submodule。
- 当前 HEAD `7d4f578` 与 `origin/refactor/three-module-baseline` 对齐；相对 `origin/main` 为 0 behind / 898 ahead。
- `archive/full-research-history-2026-08-25`、`release/dissertation-v1`、`research-archive-2026-08-25` 和 `v1.0.0-dissertation` 当前均不存在，没有命名冲突。
- 当前用户已通过 GitHub CLI 登录 `jiaxiyou-ctrl`，令牌具备 `repo` 和 `workflow` scopes；实际 push 权限仍以推送结果为准。
- 已跟踪但修改的文件只有 5 个：4 个 full-flow 索引/状态文档和 `pyproject.toml` 的 Experiment 05 data-prep 依赖。
- 未跟踪内容约 3,087,808 bytes，均被 `file` 识别为文本/JSON/CSV/Python/TOML/LaTeX 等；未发现模型权重或二进制缓存。
- 未跟踪文件按职责可分为 Experiment 04 配置/代码/测试/报告结果、Experiment 05 代码/测试/报告结果，以及本次正式发布规划文件。
- 最大的新文件是 Exp04 `per_query_metrics.csv`（约 1.1 MiB），其次是 `per_query_goal4_metrics.csv`（约 736 KiB）；它们可进入完整 archive，但 G2 必须从 release 默认树外置。
- 当前旧 full-flow README/索引中的“尚未运行 held-out”“Goal 3 active”等状态与同一工作区新增的 Exp04/05 final results 处于不同时间点；G1 应原样保存开发历史，G6 再以冻结最终报告重写公开说明。
- Exp05 冻结的 claim label 精确值是 `NOT SUPPORTED`（空格），不是计划摘要中使用的 `NOT_SUPPORTED`（下划线）；两者科学含义一致，机器验证必须使用冻结 JSON 的实际 schema 值。
- 当前全量 pytest 的唯一失败仍是历史 G000 文档状态测试：`tests/scripts/test_full_flow_g000_freeze.py::test_g000_without_server_audit_stops_before_next_stage` 要求旧 README/TRACKER 包含 active G000 文本，但该历史路线已经推进到后续状态。它与本次 Exp04/05 代码无关；G1 原样记录，G2/G7 在正式树移出旧 tracker 后重新验证公开 CI。
- Exp04/05 冻结 CSV 使用 CRLF，若按普通 text whitespace 检查会把每行的 `\r` 报为 trailing whitespace；部分 Markdown 也用两个尾随空格表示硬换行。archive 必须保留这些已登记哈希的原字节，release 中的小型公开表格再由 G4 规范化生成。

### G1 mypy 根因分析

- `experiment05_data.py` 的 pyarrow 是函数内可选依赖；库已安装但不提供 `py.typed`，错误来自第三方类型元数据而不是本项目数据逻辑。
- `experiment05_index.py:212` 直接返回 `json.loads` 的 `Any`；函数虽做字段校验，类型检查器仍无法推断为 `dict[str, Any]`。
- `experiment05_index.py` 用参数名 `faiss_module` 作为动态 import alias，既触发重定义，也阻止 `Any | None` 在后续调用前可靠收窄；sentence-transformers/FAISS 同时属于可选服务器依赖，当前 dev 环境没有类型实现。
- `experiment05_scorer.py:301` 在异常恢复分支把 `data` 暂时设为 `None`；运行时的 raise 已保护，但 mypy 未把跨 try/except 的值收窄到 dict。
- `experiment05_scorer.py:610` 的 dict comprehension 被推断为 `dict[str, float]`，随后合法的 undefined UCR `None` 与该局部推断冲突。
- `experiment05_generation.py:109` 把两个不同 Pydantic record tuple 展开后，mypy 将公共上界推断成只有配置的 `_FrozenModel`，因此看不到两个子类共有的 `text`/`text_sha256` 字段。
- 修复假设：精确标注无类型可选 import、显式验证/cast JSON mapping、把动态 FAISS import 赋给不同局部名并断言非空、显式收窄 scorer data/metric union、分别遍历两种 evidence record；这些修改只改变静态表达，不改变计算、阈值、结果或 I/O schema。
- 实施验证支持该假设：第一次最小修正将 17 个错误降到 1 个；余下错误来自两个连续循环复用变量名造成的 mypy 类型绑定，使用独立变量名后 strict mypy 全绿。

## 最终方法与研究发现

- 最终 runtime：Hybrid Retriever → trained NLI Selector → grounded GR-C Generator。
- Hybrid Retriever：Strong BM25 + Granite dense + RRF。
- 最终 Selector：seed-13 NLI risk-controlled selector。
- 最终 Generator：Granite base + GR-C adapter + grounded verification/annotation。
- TopK、Direct Granite、Granite rerank、Provence、threshold-only 是最终实验需要的基线/消融，不是冗余。
- Experiment 04 技术执行有效，但主要 whole-system RAR superiority claim 未获得支持。
- Experiment 04 的主要弱点定位到 frozen GR-C Generator；Direct Granite 在部分数据上 RAR 更高。
- Experiment 05 的更强 whole-system Claim A/B 均为 NOT SUPPORTED。
- Selector 在 misleading-evidence 专项压力测试中有条件性 evidence-level 收益。
- Selector dedicated blind answer gate 未显示支持的答案提升。
- ordinary Experiment 05 数据上 frozen Selector 大多未触发，因此不能声称普通数据整体提升。
- `FINAL PASS` 表示实验协议/执行完成，不表示科学 superiority 成立。

## 外部资产发现

- Selector seed-13 checkpoint 约 704 MiB。
- GR-C seed-13 adapter 约 60 MiB。
- GR-C seed-42/73 adapters 需在 G5 重新核对大小与发布权限。
- base model snapshots、数据集缓存、indexes 和 raw generation bundles 不适合进入 Git。
- 公开发布前需要确认第三方模型/数据许可证及衍生权重再分发权限。

## 技术决策

| 决策 | 理由 |
|---|---|
| 建立 archive branch/tag | 不删除旧历史，保证任意研究材料可恢复。 |
| 从 archive commit 建立普通 release branch | 保持完整 Git 历史连通，不使用 orphan/history rewrite。 |
| release 验收后接入 main | GitHub 默认首页展示正式版本。 |
| 每次只执行一个 Goal | 限定授权范围，降低混合工作区和远端写入风险。 |
| 最终代码与研究复现分层 | 同时满足可用性和研究完整性。 |
| raw assets 外置并用 manifest 引用 | 控制 Git 体积并处理许可证/访问边界。 |
| cleanup manifest 只做文件决策 | 执行状态统一由 `task_plan.md` 管理。 |
| 正式 main 采用运行、配置、实验、测试、文档、结果、示例七层结构 | 同时服务答辩评审、普通使用、前端接入和研究复现。 |
| 前端通过最小 HTTP API 接入三模块后端 | 前端无需获得个人 HPC 账号，也不直接加载 checkpoint。 |
| 目标树优先保留当前已测试实现并增加薄入口 | 避免仅为目录美观重写冻结算法。 |
| 每个外部资产由单一 manifest 登记 | Git 树中不放大权重，但安装和运行不会依赖口头路径。 |

## 未决问题

| 问题 | 影响 | 解决 Goal |
|---|---|---|
| 代码许可证尚未确定 | 阻塞正式开源声明和 release metadata | G6 |
| 前端是否属于同仓库交付范围 | 决定是否必须实现 HTTP API | G3 |
| 模型 adapters 是否允许公开再分发 | 阻塞公众真实 runtime 下载 | G5 |
| raw outputs 是否允许按数据许可归档 | 决定 DOI archive 内容 | G5 |
| 仓库所有者/用户权限范围 | 可能阻塞 branch/tag/main/Release 操作 | G1/G8 |
| 文档 ingestion 是否属于最终论文交付主张 | 决定 `loaders/` 和 `cli/ingest.py` 的最小保留范围 | G3 |

## 规划验证发现

- P0/P1 complete；G1 in progress；G2–G8 pending。
- G1–G8 均包含目的、前置条件、允许动作、产物、验收门和禁止事项。
- P0 初版缺少部分同名结构标题，已在自检后补齐。
- cleanup inventory 已链接到 `task_plan.md`，执行状态不会在两份文件中重复维护。
- 四份规划/清单文件未包含个人账号或 HPC 绝对路径。
- 正式结构预览将默认 `main` 定义为运行代码、冻结配置、复现实验、测试、公开文档、最终聚合结果和示例七层。
- 结构预览包含建议新增的最小 HTTP API；这是后续 G3 的实现/验证项，不是对当前仓库已有能力的陈述。
- 当前实际 loader 和部分 Generator/Selector 辅助文件名已对照预览修正；最终删除边界仍须由 G3/G4 的依赖闭包和测试决定。
- G1 必须先在当前开发工作区冻结尚未提交的 Experiment 04/05；G2 起采用隔离 worktree，避免正式树整理污染归档现场。

## 资源

- `docs/PUBLIC_RELEASE_CLEANUP_MANIFEST.md`
- `docs/PROPOSED_PUBLIC_REPOSITORY_STRUCTURE.md`
- `docs/three-module-runtime-handoff.md`
- `configs/models/three_module_seed13.json`
- `configs/experiments/systemf_three_module_smoke_seed13.toml`
- Experiment 04 final report and result tables
- Experiment 05 findings, final report, tables, and integrity audit

## 视觉/浏览器发现

- 本次计划编写未新增视觉或浏览器发现。

---
*执行每个 Goal 时更新事实、决策和未决问题。*
