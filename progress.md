# 进度日志：正式发布版仓库

## 会话：2026-08-25

### P0：计划与清单冻结

- **状态：** complete
- 执行的操作：
  - 读取并遵循文件化规划技能。
  - 确认项目中原先不存在 `task_plan.md`、`findings.md` 和 `progress.md`。
  - 读取规划模板。
  - 将现有 cleanup inventory 转换为 8 个可单独启动和验收的执行 Goal。
  - 记录固定决策、仓库审计事实、风险、未决问题和执行边界。
- 创建/修改的文件：
  - `task_plan.md`（新建）
  - `findings.md`（新建）
  - `progress.md`（新建）
  - `docs/PUBLIC_RELEASE_CLEANUP_MANIFEST.md`（将链接到本计划）
- 未执行：
  - 未创建系统 Goal。
  - 未创建、修改或切换 Git 分支。
  - 未提交或推送。
  - 未整理、移动或删除仓库文件。

### P1：正式仓库结构预览与确认

- **状态：** complete
- 执行的操作：
  - 增加执行前 P1 阶段，阻止在用户确认目标形态前启动 G1。
  - 设计 archive、release、main 和两个标签的最终关系。
  - 编写正式 `main` 的目标目录树和逐文件/文件组职责说明。
  - 补充毕设评审、普通运行、前端接入和实验复现四条使用路径。
  - 明确模型、数据、索引、原始结果和历史材料不进入正式树的边界。
  - 对照当前 `src/evidence_rag` 与 Experiment 04 配置，修正目标树中的辅助文件和 loader 命名。
- 创建/修改的文件：
  - `docs/PROPOSED_PUBLIC_REPOSITORY_STRUCTURE.md`（新建）
  - `task_plan.md`（增加 P1）
  - `findings.md`（记录目标结构决策和未决边界）
  - `progress.md`（本记录）
- 未执行：
  - 未创建、修改或切换 Git 分支。
  - 未提交或推送。
  - 未整理、移动、重命名或删除项目文件。
  - 未启动 G1–G8。

### 执行授权与总目标

- 用户确认目标结构，并授权 G1–G8 在各自验收通过后按顺序接替执行。
- 已建立系统总目标：保留完整研究历史，形成、验证并发布 `v1.0.0-dissertation`。
- 执行策略：G1 在当前开发工作区冻结研究现场；G2 起使用隔离 worktree。
- 最终验证范围：clean install、pytest、ruff、mypy、package build、CPU smoke、文档/manifest/路径/密钥/大文件扫描、结果一致性；HPC/GPU 验证按可用环境如实记录。

### G1：冻结完整研究历史

- **状态：** complete
- 已满足开始条件：P0/P1 complete，用户已授权顺序执行。
- 已执行：
  - 确认当前分支、HEAD、远端关系、普通 checkout 状态和 GitHub 登录状态。
  - 确认计划中的 archive/release 分支及两个 tag 尚不存在。
  - 审计全部 tracked modifications 和 untracked files。
  - 计算未跟踪内容总体积并检查文件类型；未发现模型权重或二进制缓存。
  - 将工作区分为 Exp04、Exp05、full-flow 索引更新和正式发布规划四组。
- 下一步：运行 focused tests、静态检查和结果一致性验证，再按精确路径分组提交。
- 环境发现：系统默认 `python3` 为 3.14.6 且没有 pytest；仓库已有 `.venv/bin/pytest`，本机也有符合项目约束的 `python3.11`。后续验证改用仓库 `.venv`。
- focused tests 在 `.venv`（Python 3.11.14）中退出码 0；ruff 对 `src`、`tests` 和新增 Exp04/05 脚本检查通过。
- strict mypy 可稳定复现 17 个错误，集中在 Exp05 的 `experiment05_data.py`、`experiment05_index.py`、`experiment05_scorer.py` 和 `experiment05_generation.py`；提交前进入根因分析。
- 已完成 mypy Phase 1/2 根因与模式分析：17 个错误归为第三方可选依赖类型元数据、动态模块 Optional 收窄、JSON `Any`、局部 union 推断和异类 tuple 公共上界五类；尚未改代码。
- 已用失败的 strict mypy 作为 RED，实施不改变算法的最小类型修正；第一次重跑剩 1 个循环变量类型绑定错误，改用独立变量后 GREEN：147 source files 无错误。
- 类型修正后的四文件 ruff 通过，Exp05 focused tests 再次退出码 0。
- Exp04 已在临时副本重建出 `FINAL PASS`（11,000 generations、135 summary rows、21 bootstrap cells），且与当前冻结目录逐字一致。
- 首次联合结果校验在 Exp05 label 字符串假设处停止：实际 schema 值为 `NOT SUPPORTED`，不是摘要写法 `NOT_SUPPORTED`；9 个 artifact 哈希尚未在该次脚本中完成逐项断言。
- 更正 label 后结果完整性通过：Exp04 hashes、Exp05 9 个 artifact hashes/claims 和 38 个 JSON 全部验证；遗留临时目录已精确清理。
- 全量 pytest：1877 passed、20 skipped、1 failed。唯一失败是已知旧 G000 README/TRACKER 文本状态测试，与新增 Exp04/05 运行代码无关。
- 对全部 changed/untracked 文件运行常见私钥、GitHub/AWS/Google/OpenAI token 样式扫描：无匹配；未跟踪文件中没有超过 10 MiB 的文件。
- Exp04 第一组 43 个代码/配置/测试文件已精确 staged；报告生成器的 Markdown 双空格改为显式字符串且生成内容逐字不变，staged whitespace 检查和受影响的 Goal5 tests 通过。提交命令尚未成功执行。
- 已创建 G1 科学工作提交：
  - `cdf2186` — `feat(evaluation): add frozen Experiment 04 implementation`
  - `84a90de` — `feat(evaluation): add general RAG Experiment 05 implementation`
  - `421f193` — `docs(results): record final experiments and audits`
- 报告提交前重新验证 Exp04/05 audit hashes。冻结 Markdown/CRLF CSV 保留原字节，因此报告组不以普通 `git diff --check` 作为通过条件；该例外只适用于 archive 研究证据。
- 已创建 `ARCHIVE_INDEX.md`，覆盖 Selector 8 条路线、Full-flow 5 条路线、最终 runtime、Exp04/05、外部模型哈希和恢复方法。
- 已推送开发分支和 `archive/full-research-history-2026-08-25`。
- 已创建并推送 annotated tag `research-archive-2026-08-25`。
- 远端 branch/tag 均指向 `ea4d617753aff868fff8f846964f7cfb050414bb`；通过 tag 抽查 Exp05 audit、scorer 源码和 archive index 均可恢复。

### G2：建立干净 release 文件树

- **状态：** in_progress
- 开始条件已满足：G1 complete，archive branch/tag 已在远端验证。
- 下一步：建立隔离 worktree，从 archive commit 创建 `release/dissertation-v1`，形成受控保留树和归档映射。
- worktree 预检确认当前是普通 checkout，项目原先没有 `.worktrees/` 且未忽略该目录；按隔离执行规则在开发分支增加 `/.worktrees/` 防护。

### G3–G8

- **状态：** pending
- 按 `task_plan.md` 顺序和验收门执行。

## 测试结果

| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|---|---|---|---|---|
| 规划文件存在性 | `task_plan.md`, `findings.md`, `progress.md` | 三个文件均存在 | 三个文件和 cleanup inventory 均存在，链接有效 | PASS |
| 计划结构检查 | Goal P0/G1–G8 | 每个 Goal 有目的、条件、动作、产物、验收门和禁止事项 | P0、G1–G8 六类结构字段全部通过机器检查 | PASS |
| Markdown whitespace | 四个规划/清单文件 | 无格式错误 | 四个文件 whitespace 检查通过 | PASS |
| 安全边界 | Git status/diff | 仅新增/修改规划文件 | 仅四份规划/清单文件为本任务新增或修改 | PASS |
| P1 文件存在性 | 目标结构稿和四份规划文件 | 文件存在且非空 | 目标结构稿 717 行；五份规划/结构文件均存在 | PASS |
| P1 结构覆盖 | 目标结构稿 | 包含分支关系、目标树、文件职责、排除项、外部资产和读者路径 | 七类关键章节全部存在 | PASS |
| P1 Markdown whitespace | 五份规划/结构文件 | 无行尾空格 | 扫描通过 | PASS |
| P1 信息边界 | 五份规划/结构文件 | 不含账号、个人绝对路径或常见密钥样式 | 扫描通过 | PASS |
| G1 工作区文件类型 | 全部 untracked files | 不含模型权重、缓存或未知二进制 | 约 3.09 MB，均为文本/JSON/CSV/Python/TOML/LaTeX 等 | PASS |
| G1 tracked diff whitespace | 5 个 tracked modifications | 无 whitespace error | `git diff --check` 无输出 | PASS |
| G1 focused pytest | Exp04/Exp05 + rerank/Provence/threshold-only 新测试 | 退出码 0 | Python 3.11.14 下全部完成，退出码 0 | PASS |
| G1 ruff | `src`、`tests`、新增 Exp04/05 Python 脚本 | 0 errors | `All checks passed!` | PASS |
| G1 strict mypy（首次） | `src` + `tests/typecheck.py` | 0 errors | 4 个 Exp05 文件共 17 errors | FAIL |
| G1 strict mypy（修正后） | `src` + `tests/typecheck.py` | 0 errors | `Success: no issues found in 147 source files` | PASS |
| G1 Exp05 回归 pytest | Exp05 evaluation + scripts focused tests | 退出码 0 | 68 tests（进度点）全部完成，退出码 0 | PASS |
| G1 结果一致性 | Exp04 临时重建；Exp04/05 audits/hashes/JSON | 冻结结果可重建或哈希一致 | Exp04 重建逐字一致；Exp05 9 hashes/claims 一致；38 JSON 可解析 | PASS |
| G1 全量 pytest | 整个仓库 | 记录真实基线 | 1877 passed、20 skipped、1 个已知历史文档状态失败 | KNOWN-FAIL |
| G1 changed-file secret scan | 全部 modified/untracked files | 无常见密钥样式 | 无匹配 | PASS |
| G1 untracked large-file scan | 全部 untracked files | 无意外 >10 MiB 文件 | 0 files | PASS |
| G1 Exp04 staged boundary | 43 files | 只含 Exp04/config/runtime baseline/tests | allowlist 检查通过；提交 `cdf2186` | PASS |
| G1 Exp05 staged boundary | 39 files | 只含 Exp05/pyproject/tests | allowlist、ruff、mypy、focused pytest 通过；提交 `84a90de` | PASS |
| G1 research evidence boundary | 81 files | 只含 full-flow docs/results | allowlist 和冻结 hashes 通过；提交 `421f193` | PASS |
| G1 remote archive refs | archive branch + annotated tag | 远端解析到同一冻结提交 | 两者均为 `ea4d617753aff868fff8f846964f7cfb050414bb` | PASS |
| G1 tag recovery sample | Exp05 audit、scorer、archive index | 可从 tag 读取 | 三个 `git cat-file -e` 均通过 | PASS |

## 错误日志

| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|---|---|---:|---|
| 2026-08-25 | P0 初版缺少与执行 Goal 一致的前置条件、产物和禁止事项标题 | 1 | 补齐标题并重新运行结构检查。 |
| 2026-08-25 | 三个新规划文件 EOF 多余空行 | 1 | 移除多余空行并重新运行 whitespace 检查。 |
| 2026-08-25 | 首次同步 P1/G1 状态时补丁上下文与文件中的空格不一致 | 1 | 使用文件中的精确上下文重新应用补丁。 |
| 2026-08-25 | 默认 Python 3.14.6 无 pytest，首次 focused test 未启动 | 1 | 发现仓库已有 `.venv/bin/pytest` 和本机 Python 3.11，改用项目虚拟环境。 |
| 2026-08-25 | strict mypy 在 4 个新 Exp05 文件报告 17 个错误 | 1 | 按系统化调试流程分析可选依赖、`Any` 返回、Optional 收窄和 protocol 字段根因后再最小修正。 |
| 2026-08-25 | 第一次类型修正后两个连续循环复用变量名，mypy 剩 1 个 assignment error | 1 | 第二个循环改用独立变量名；重跑后 147 source files 无错误。 |
| 2026-08-25 | 结果校验脚本把 Exp05 claim label 假设为 `NOT_SUPPORTED` | 1 | 读取冻结 audit，确认实际值为 `NOT SUPPORTED`；改用实际 schema 值并重新验证，失败遗留临时目录已精确定位。 |
| 2026-08-25 | 首次 secret-scan shell 使用 zsh 只读变量名 `status` | 1 | 改用任务专用变量 `scan_result` 并添加精确临时文件退出清理；扫描通过。 |
| 2026-08-25 | staged allowlist 循环使用 zsh 特殊变量 `path`，覆盖命令搜索路径并导致 `git` command not found | 1 | 暂存区保持完整；改用 `staged_file` 并通过 `/usr/bin/git` 执行提交。 |
| 2026-08-25 | 首次同步 G1 complete/G2 in-progress 补丁包含已过期的 `progress.md` 邻接上下文 | 1 | 读取当前段落并拆成精确的小补丁。 |

## 五问重启检查

| 问题 | 答案 |
|---|---|
| 我在哪里？ | P0/P1/G1 已完成；G2 正在建立干净 release 文件树。 |
| 我要去哪里？ | 完成 G2 release 文件边界，然后依次完成 G3–G8。 |
| 目标是什么？ | 同仓库内形成可提交毕设的干净可用正式 main，同时保留完整研究历史。 |
| 我学到了什么？ | 见 `findings.md`。 |
| 我做了什么？ | 已将整理清单转换成 Goal 计划，并完成正式 main 的结构与逐文件职责预览。 |

---
*每个 Goal 完成后或遇到错误时更新此文件。*
