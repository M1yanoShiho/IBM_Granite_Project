# 任务计划：毕设与研究项目正式发布版仓库

## 总目标

在不丢失现有开发历史、实验结果和团队协作记录的前提下，在同一个 GitHub 仓库内形成一个
干净、可安装、可运行、可验证、可引用的正式发布版本，并最终接入默认 `main` 分支，发布
`v1.0.0-dissertation`。

完整研究历史保留在独立归档分支和不可变标签中；正式 `main` 只展示最终三模块系统、必要的
训练/评测复现代码、测试、配置、最终聚合结果和公开文档。

## 当前阶段

G1 已冻结远端 archive；G2 已完成 release 边界；G3 已冻结最终 runtime/API；G4 正在整理研究复现包。

## 固定决策与边界

1. 使用同一个仓库，不新建正式代码仓库。
2. 不重写 Git 历史，不永久删除旧研究材料。
3. 旧版本通过 `archive/full-research-history-2026-08-25` 和
   `research-archive-2026-08-25` 保留。
4. 整理工作在 `release/dissertation-v1` 进行，验收后接入 `main`。
5. 从 release/main 当前文件树移除旧文件，不等于从归档分支或 Git 历史中删除。
6. 模型权重、数据集缓存、索引、逐题输出和原始日志不进入正式 Git 文件树。
7. 必须保留复现论文最终方法、基线、消融、评分和表格所需的代码。
8. 必须公开报告有效的负面结果，不因清理仓库而改变或省略科学结论。
9. 每次只允许一个执行 Goal 处于 `in_progress`。
10. 用户已授权 G1–G8 按顺序接替执行；每个 Goal 达到验收门后才自动启动下一个 Goal。
11. 遇到许可证、模型/数据公开权、远端权限、测试失败或不可逆外部动作等真实阻塞时停止并记录，不猜测或伪造通过。
12. G1 在当前开发工作区冻结未提交研究现场；G2 起使用隔离 worktree 整理 release，保护归档现场。

## Goal 启动协议

以后用户可以用以下方式启动：

```text
执行 Goal 1
继续 Goal 4
检查 Goal 7 是否通过
```

启动一个 Goal 时：

1. 重新读取 `task_plan.md`、`findings.md` 和 `progress.md`。
2. 确认前置 Goal 已达到验收门。
3. 将该 Goal 标为 `in_progress`，其余执行 Goal 保持 `pending`。
4. 只执行该 Goal 授权的范围。
5. 记录文件变化、测试结果、异常和恢复方式。
6. 达到验收门后标为 `complete`；依据本次全局授权自动启动下一 Goal。
7. 若下一 Goal 的外部前置条件尚未满足，则停止在该验收门并向用户报告具体 blocker。

远端写入、提交、推送、创建/修改分支和 GitHub 设置必须与被启动 Goal 的授权范围一致。

## Goal 总览

| Goal | 名称 | 核心产物 | 状态 |
|---|---|---|---|
| P0 | 计划与清单冻结 | 本计划、发现记录、进度日志、整理清单 | complete |
| P1 | 正式仓库结构预览与确认 | 目标目录树、逐文件职责、用户视角和边界说明 | complete |
| G1 | 冻结完整研究历史 | 完整提交、归档分支、归档标签、归档索引 | complete |
| G2 | 建立干净 release 文件树 | `release/dissertation-v1` 和受控保留树 | complete |
| G3 | 冻结可运行的最终三模块系统 | 最终运行入口、配置、API 边界、核心测试 | complete |
| G4 | 整理研究复现包 | 训练、基线、消融、评分和表格复现入口 | in_progress |
| G5 | 发布外部模型与数据资产清单 | 公开链接/DOI、许可证、版本和 SHA-256 | pending |
| G6 | 完成公开文档与项目元数据 | README、架构、复现、结果、限制、引用 | pending |
| G7 | 独立验证正式候选版本 | clean clone、CI、CPU/HPC smoke、审计报告 | pending |
| G8 | 接入 main 并发布毕设版本 | `main`、release tag、GitHub Release、交付记录 | pending |

## P0：计划与清单冻结

### 目的

把仓库审计结论转换成可持续恢复、可分阶段执行的文件化计划。

### 前置条件

- 用户要求本次只写可分阶段执行的计划。
- 现有仓库扫描和 cleanup inventory 可用。

### 已完成

- [x] 扫描当前分支、默认分支、工作区和仓库体积。
- [x] 区分最终运行代码、研究复现代码、最终结果、外部资产和历史档案。
- [x] 追踪最终 Retriever、Selector、Generator 与 Experiment 04/05 主要依赖。
- [x] 创建 `docs/PUBLIC_RELEASE_CLEANUP_MANIFEST.md`。
- [x] 创建本计划、`findings.md` 和 `progress.md`。

### 产物

- `task_plan.md`
- `findings.md`
- `progress.md`
- 更新后的 `docs/PUBLIC_RELEASE_CLEANUP_MANIFEST.md`

### 验收门

- [x] 计划明确同仓库、归档分支、release 分支和 main 的关系。
- [x] 每个执行 Goal 有前置条件、动作、产物、验收门和禁止事项。
- [x] 当前没有执行清理、分支、提交或推送。

### 禁止事项

- 不启动 G1–G8。
- 不创建执行型系统 Goal。
- 不修改 Git 分支、提交、远端或仓库文件边界。

**状态：** complete

## P1：正式仓库结构预览与确认

### 目的

在任何 Git 或文件整理动作发生前，把整理完成后的正式 `main` 文件树、每个文件的职责、
不同使用者的入口以及不会进入正式树的内容完整展示给用户确认。

### 前置条件

- P0 complete。
- 用户要求先看到整理完成后的仓库形态。

### 允许动作

1. 基于现有审计和最终三模块方法设计目标文件树。
2. 说明每个拟保留文件或文件组的职责与来源。
3. 标注前端 API、数据接入和外部模型资产等需要在后续 Goal 验证的边界。
4. 更新计划、发现记录和进度日志。

### 产物

- `docs/PROPOSED_PUBLIC_REPOSITORY_STRUCTURE.md`
- 更新后的 `task_plan.md`
- 更新后的 `findings.md`
- 更新后的 `progress.md`

### 验收门

- [x] 同仓库中 archive、release 和 main 的最终关系清楚。
- [x] 正式 `main` 的目标目录树完整可读。
- [x] 树中每个命名文件或文件组都有明确职责。
- [x] 普通使用、前端接入、实验复现和论文审阅路径均已说明。
- [x] 模型、数据、索引、日志和历史材料的存放边界清楚。
- [x] 用户可以据此确认或修改期望，再决定是否启动 G1。
- [x] 用户明确确认目标结构，或提出需要写回计划的修改。

### 禁止事项

- 不创建、切换或修改 Git 分支。
- 不提交、推送、移动、删除或重命名项目文件。
- 不启动 G1–G8。

**状态：** complete

## G1：冻结完整研究历史

### 目的

在任何 release 文件树整理前，把当前完整研究状态安全保存，使后续从 main 移出的文件仍可永久查找。

### 前置条件

- P0、P1 complete。
- 用户明确授权“执行 Goal 1”。
- 确认当前工作区中 Experiment 04/05、测试、配置和报告的归属。

### 允许动作

1. 审计所有 modified/untracked 文件并分组。
2. 运行 Experiment 04/05 相关测试、lint、类型和结果一致性检查。
3. 用精确路径分批提交，不使用 `git add .` 或 `git add -A`。
4. 将规划文件记录为 archive-only 工作记录，避免混入科学代码提交。
5. 推送完整研究提交到当前开发分支。
6. 在同一完整提交上创建：
   - `archive/full-research-history-2026-08-25`
   - `research-archive-2026-08-25`
7. 创建 `ARCHIVE_INDEX.md`，索引 Experiment 01–05、Selector、Generator、结果和外部 HPC 资产。

### 建议提交分组

1. `feat(evaluation): add frozen Experiment 04 implementation`
2. `feat(evaluation): add general RAG Experiment 05 implementation`
3. `docs(results): record final experiments and audits`
4. `docs(archive): index complete research history`

### 产物

- 完整、可检出的研究状态提交。
- 归档分支和归档标签，二者指向已验证的完整提交。
- `ARCHIVE_INDEX.md`。
- G1 验证记录和提交哈希。

### 验收门

- [x] 所有属于最终研究工作的文件均已分类并安全保存。
- [x] Experiment 04/05 相关测试通过，或已记录与本次代码无关的已知失败。
- [x] 归档分支和标签解析到相同、正确的提交。
- [x] 远端可以读取归档分支和标签。
- [x] 从归档引用能够恢复任意后续从 main 移出的文件。
- [x] 当前用户拥有或已获得所需远端权限。

### 禁止事项

- 不整理或移除旧文件。
- 不修改 `main`。
- 不创建 release 版科学结论。
- 不把模型、数据缓存或原始 HPC 运行目录提交进 Git。

### 恢复方式

归档分支和标签是后续全部 Goal 的不可变恢复点。若任意整理结果不正确，release 分支可重新从该提交创建。

**状态：** complete

## G2：建立干净 release 文件树

### 目的

从完整归档提交建立 `release/dissertation-v1`，按整理清单形成面向正式发布的最小必要文件树。

### 前置条件

- G1 complete。
- 归档分支和标签已验证。
- 用户明确授权“执行 Goal 2”。

### 允许动作

1. 从归档提交创建 `release/dissertation-v1`，不使用 orphan 分支。
2. 按 `docs/PUBLIC_RELEASE_CLEANUP_MANIFEST.md` 应用：
   - `KEEP-CORE`
   - `KEEP-REPRO`
   - `KEEP-RESULT`
   - `CONSOLIDATE`
   - `EXTERNAL`
   - `ARCHIVE`
   - `REMOVE`
   - `HOLD`
3. 从 release 当前文件树移出 `.aris/`、`refine-logs/`、内部工具计划、旧 snapshots、原始 runs 和大部分 raw results。
4. 将 `HOLD` 源码保留到依赖审计证明其未使用。
5. 为每批移出记录归档恢复路径。
6. 只整理文件边界，不在本 Goal 改变算法、评分器或科学结果。

### 建议提交分组

1. `release: establish dissertation repository skeleton`
2. `release: externalize raw artifacts and historical records`
3. `release: retain final runtime and reproduction surface`

### 产物

- `release/dissertation-v1`。
- release 文件树清单及归档映射。
- 体积、文件数、绝对路径和大文件审计结果。

### 验收门

- [x] release 分支来自已冻结归档提交。
- [x] 所有移出内容仍可从归档分支或标签读取。
- [x] release 文件树不含模型权重、数据缓存、索引、逐题输出或原始运行日志。
- [x] release 文件树不含个人账号名和绝对 HPC 路径。
- [x] 普通被跟踪文件不超过 1 MiB，除非有明确最终交付理由。
- [x] checked-out release 树目标体积不超过 25 MiB。
- [x] `HOLD` 文件未经依赖验证不会被移出。
- [x] 此 Goal 没有改变任何 frozen metric、claim label 或 checkpoint checksum。

### 禁止事项

- 不修改归档分支和归档标签。
- 不合并到 `main`。
- 不为了减小体积删除复现最终论文结果必需的代码。
- 不重写 Git 历史。

**状态：** complete

## G3：冻结可运行的最终三模块系统

### 目的

确保公开 release 能清楚、稳定地运行最终 Hybrid Retriever → trained NLI Selector → grounded GR-C Generator。

### 前置条件

- G2 complete。
- 用户明确授权“执行 Goal 3”。

### 允许动作

1. 收敛并记录最终运行模块：
   - Retriever：Strong BM25 + Granite dense + RRF hybrid。
   - Selector：seed-13 NLI risk-controlled selector。
   - Generator：Granite + frozen GR-C adapter + grounded verification/annotation。
2. 保留 TopK、Direct Granite、Granite rerank、Provence 和 threshold-only 作为有标签的基线/消融，不作为主方法。
3. 整理 `composition.py`、模块注册、环境变量和模型 manifest。
4. 将所有存储路径改为环境变量或 CLI 参数。
5. 提供：
   - 无 GPU/权重的 CPU smoke；
   - 有外部权重的真实三模块命令；
   - 可审计的候选、选择、答案和引用 trace。
6. 决定前端交付边界：
   - 若前端属于毕设提交系统，加入最小 HTTP API 和冻结 response schema；
   - 若前端独立提交，至少提供 Python API、mock response 和接口文档。

### 产物

- 唯一正式 runtime config。
- 唯一正式模型 manifest。
- CPU smoke fixture。
- 真实三模块启动命令。
- 接口/API schema 和集成测试。

### 验收门

- [x] README 之外存在可执行的 smoke 命令。
- [x] CPU smoke 不访问 HPC、网络或真实权重。
- [x] 真实 config 缺少环境变量时快速、明确失败。
- [x] 模型 revision 和 SHA-256 与冻结 manifest 一致。
- [x] Generator 不能引用 Selector 未选择的证据。
- [x] final runtime 测试、架构测试、lint 和 mypy 通过。
- [x] 前端边界已有明确决定和可验证产物。

### 禁止事项

- 不重新训练或调参。
- 不读取 held-out 数据做开发决策。
- 不把基线描述成最终 trained Selector。
- 不改变冻结 checkpoint 或最终实验配置。

**状态：** complete

## G4：整理研究复现包

### 目的

保留一个研究项目和毕设答辩所需的训练、基线、消融、评分、统计和表格复现能力，同时移除重复版本。

### 前置条件

- G3 complete。
- 用户明确授权“执行 Goal 4”。

### 允许动作

1. Selector 复现：
   - `lean_v3.toml`
   - `run_selector_lean.py`
   - dual-head/NLI runtime/risk policy
   - 必要标签与投影 materialisation
   - focused tests
2. Generator 复现：
   - G300 training
   - G310 seed screen
   - G400 NIAH qualification
   - G410 cross-data qualification
   - G320/G330 frozen recipe/provenance
3. Experiment 04：保留三数据集、三 Generator seeds、正式基线、三模块消融、统一 scorer 和 final table build。
4. Experiment 05：保留三数据集、五系统、三模块消融、统一评分、bootstrap、claim labels 和 integrity audit。
5. 保留 misleading-evidence Selector 专项评测，支撑条件性模块贡献。
6. 将多阶段脚本收敛成少量 documented commands，但只能包装/迁移已测试冻结逻辑。
7. 生成“论文主张 → 代码 → 配置 → 输入 manifest → 结果表 → 外部 artifact”的复现映射。

### 产物

- `experiments/` 或等价的正式复现目录。
- Selector 与 Generator 最终训练命令。
- Experiment 04/05 端到端评测和表格构建命令。
- `REPRODUCIBILITY_MAP.md`。
- 最终基线与消融说明。

### 验收门

- [ ] 每个论文主要数值都能映射到代码、配置、输入和结果。
- [ ] 最终表格能从小型聚合输入或外部归档输入重建。
- [ ] scorer 与 claim label 逻辑没有 post-hoc 改写。
- [ ] 被移出的旧脚本没有被保留命令或测试引用。
- [ ] 训练数据准备链要么可公开重建，要么有不可变外部 artifact 和限制说明。
- [ ] Experiment 04/05 focused tests、lint 和类型检查通过。

### 禁止事项

- 不只保留正向结果。
- 不删除产生最终 checkpoint 所必需且尚未被替代的数据准备步骤。
- 不将 `FINAL PASS` 改写为科学 superiority 成立。
- 不使用 held-out 结果重新选择模型、seed 或阈值。

**状态：** in_progress

## G5：发布外部模型与数据资产清单

### 目的

让公开代码在不把大文件放进 Git 的情况下，可以定位、验证和在许可允许时下载正式模型与研究数据。

### 前置条件

- G4 complete。
- 团队/导师确认模型、数据和代码的发布权限。
- 用户明确授权“执行 Goal 5”。

### 允许动作

1. 审核并记录：
   - Selector seed-13 checkpoint；
   - GR-C seed-13/42/73 adapters；
   - base model IDs/revisions；
   - final evaluation raw bundles；
   - per-query metrics、traces 和 indexes。
2. 确认第三方模型与数据集许可证和再分发边界。
3. 选择 Hugging Face、data.bris、Zenodo 或 GitHub Release。
4. 发布或准备外部 artifact，并生成 DOI/URL、版本、字节数和 SHA-256。
5. 提供下载与校验脚本，不在 Git 中放真实权重。

### 产物

- `ARTIFACT_MANIFEST.json`。
- `docs/models-and-data.md`。
- 稳定下载 URL/DOI 和校验命令。
- 无法公开资产的明确原因与复现替代方案。

### 验收门

- [ ] 每个正式 checkpoint/adapter 都有版本、大小和 SHA-256。
- [ ] 每个公开资产有明确许可证或使用边界。
- [ ] 外部链接在无 HPC 账号的环境可访问，或文档明确说明限制。
- [ ] 下载后校验值与冻结 manifest 一致。
- [ ] Git tree 不含模型权重、原始数据集和大运行目录。

### 禁止事项

- 未确认许可证前不公开上传第三方或衍生权重。
- 不公开密钥、HPC 主机信息、个人路径或受限数据。
- 不把“团队可访问”写成“公众可复现”。

**状态：** pending

## G6：完成公开文档与项目元数据

### 目的

让不了解开发历史的读者能从 GitHub 首页理解、安装、运行、复现和正确引用项目。

### 前置条件

- G3–G5 complete，或所有未决外部链接均有明确 placeholder 和 blocker。
- 用户明确授权“执行 Goal 6”。

### 允许动作

1. 重写根 README：贡献、系统图、安装、quickstart、模型下载、复现、结果、限制、引用。
2. 创建/收敛：
   - `docs/architecture.md`
   - `docs/setup.md`
   - `docs/reproduction.md`
   - `docs/models-and-data.md`
   - `docs/results.md`
   - `docs/limitations.md`
   - Selector 与 Generator model cards
3. 添加：
   - `LICENSE`
   - `CITATION.cff`
   - `CHANGELOG.md`
   - 作者与贡献说明
4. 更新 `pyproject.toml` 的 README、作者、许可证、URLs 和 classifiers。
5. 统一英文为公开主文档；中文可作为单独翻译。
6. 明确写出科学边界：
   - Selector 在 misleading-evidence stress 下有条件性 evidence-level 收益；
   - dedicated blind answer gate 未显示支持的答案提升；
   - ordinary Experiment 05 数据上 Selector 大多未触发；
   - whole-system superiority claims 未通过；
   - `FINAL PASS` 只表示执行/协议完成。

### 产物

- 完整公开 README 和文档导航。
- 软件许可证和 citation metadata。
- model/data cards、results、limitations。
- 无失效相对链接的文档树。

### 验收门

- [ ] 新读者可以只从 README 完成安装和 CPU smoke。
- [ ] 所有命令、文件路径和链接在 release tree 中有效。
- [ ] 不使用旧 TopK/Corroboration 说明代替最终 trained NLI Selector。
- [ ] 正向和负向结论同时呈现且与冻结报告一致。
- [ ] 许可证已由有权人员确认，而不是自动选择。
- [ ] `CITATION.cff` 可被解析。
- [ ] 文档不含个人/HPC 绝对路径。

### 禁止事项

- 不为了宣传效果改变结果表述。
- 不把内部 tracker 作为公开主文档。
- 不复制大量旧计划或运行流水账到正式 README。

**状态：** pending

## G7：独立验证正式候选版本

### 目的

证明 release candidate 在干净环境可安装、可运行、可测试、可构建，并与冻结研究资产一致。

### 前置条件

- G2–G6 complete。
- 用户明确授权“执行 Goal 7”。

### 允许动作

1. 从远端 release 分支进行全新 clone，不依赖当前工作区缓存。
2. 验证：
   - 安装与依赖；
   - 全量 pytest；
   - ruff；
   - strict mypy；
   - package build；
   - CLI/CPU smoke；
   - 文档链接；
   - manifest/schema；
   - secret、个人路径和大文件扫描。
3. 在 HPC 使用冻结外部资产运行真实三模块 smoke。
4. 对关键聚合结果和 model artifact 重算 SHA-256。
5. 生成 `RELEASE_VALIDATION_REPORT.md`。

### 产物

- clean-clone 测试日志。
- CI 绿灯。
- HPC 三模块 smoke 报告。
- `RELEASE_VALIDATION_REPORT.md`。
- 完整 blocker/known limitations 列表。

### 验收门

- [ ] fresh clone 安装、pytest、ruff、mypy、build、CPU smoke 全部通过。
- [ ] GitHub CI 全绿。
- [ ] HPC 真实三模块 smoke 通过且 checkpoint hashes 一致。
- [ ] release tree 个人路径/明显密钥扫描为零。
- [ ] release tree 不含非豁免大文件。
- [ ] README 命令与实际执行一致。
- [ ] Experiment 04/05 关键表格与冻结结果一致。
- [ ] 所有已知限制已写入公开文档。

### 三次失败协议

1. 第一次：诊断根因并记录。
2. 第二次：采用不同修复或验证路径。
3. 第三次：停止 Goal，记录 blocker，请求用户决定。

### 禁止事项

- 不通过跳过测试或删除失败测试伪造绿灯。
- 不使用当前脏工作区代替 fresh clone。
- 不在验证阶段改变科学方法或冻结评分定义。

**状态：** pending

## G8：接入 main 并发布毕设版本

### 目的

将已验证的 release candidate 作为同仓库正式首页发布，同时保持完整研究历史可见、可恢复。

### 前置条件

- G7 complete。
- 用户明确授权“执行 Goal 8”。
- 具备 push/merge、main 保护规则和 GitHub Release 所需权限。

### 允许动作

1. 最终比较 archive、release 和 main 的 refs、提交和文件树。
2. 通过 fast-forward 或受保护 PR 将 `release/dissertation-v1` 接入 `main`。
3. 不强推、不重写归档历史。
4. 确认 GitHub 默认分支为 `main`，README/CI/许可证/引用正确显示。
5. 创建 `v1.0.0-dissertation` annotated tag。
6. 创建 GitHub Release，链接 DOI、模型、数据和论文/毕设信息。
7. 验证 archive branch/tag 仍可见且内容完整。
8. release 分支在确认后可删除；archive 分支和标签永久保留。
9. 将 `task_plan.md`、`findings.md`、`progress.md` 和内部 cleanup manifest 保留在 archive，
   正式 main 可移除或将必要流程改写为简洁 release-maintenance 文档。

### 产物

- 干净的默认 `main`。
- `v1.0.0-dissertation`。
- GitHub Release 和外部 DOI/asset 链接。
- 可访问的完整 archive branch/tag。
- 最终交付摘要。

### 验收门

- [ ] `main` HEAD 与已验证 release candidate 完全一致。
- [ ] GitHub 首页只展示正式发布文件树。
- [ ] CI 在 main push 上通过。
- [ ] release tag 指向 main 的正确提交。
- [ ] GitHub Release 可下载、可引用、链接有效。
- [ ] archive branch/tag 未改变且可恢复旧文件。
- [ ] 没有 force push 或历史重写。
- [ ] 毕设中使用的仓库 URL、tag、DOI 和 commit 已冻结记录。

### 禁止事项

- G7 未通过时不得接入 main。
- 不删除 archive branch/tag。
- 不将未审计的本地修改混入发布提交。

**状态：** pending

## 跨 Goal 关键问题

1. 团队/导师选择并批准哪一种代码许可证？
2. 前端是否属于本仓库正式交付范围；若属于，采用何种最小 HTTP API？
3. Selector 与三个 Generator adapters 是否允许公开再分发？
4. raw generations/per-query metrics 的第三方数据许可是否允许外部归档？
5. data.bris、Zenodo、Hugging Face 或 GitHub Release 分别承载哪些资产？
6. 当前仓库所有者是否能执行 main、默认分支、archive 和 Release 设置？

这些问题不阻止 G1–G2 的本地准备，但可能阻塞 G5、G6 或 G8 的验收。

## 已做决策

| 决策 | 理由 |
|---|---|
| 同仓库归档 + release + main | 保留团队历史，同时提供干净正式首页。 |
| 普通分支，不使用 orphan | 保留开发历史连通性，main 可以正常接入 release。 |
| 归档优先于整理 | 任意后续文件移出都可恢复。 |
| 每次执行一个 Goal | 限制风险，便于验收和用户控制。 |
| raw assets 外置 | 避免 Git 体积、许可和个人路径问题。 |
| 保留负面结果 | 研究诚信与毕设结论完整性要求。 |

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| 当前远端 CI 存在一个旧 G000 文档状态测试失败 | 1 | 在 G2/G7 中分离历史文档验证和公开核心 CI，不通过简单删测试处理。 |

## 规划文件说明

- `task_plan.md`：唯一执行计划和 Goal 状态来源。
- `findings.md`：仓库事实、审计发现、未决问题和技术决策。
- `progress.md`：每次 Goal 的操作、文件、提交、测试和错误日志。
- `docs/PUBLIC_RELEASE_CLEANUP_MANIFEST.md`：文件保留/归档/外置决策清单，不负责运行状态。

在所有 Goal 完成前保留这些文件。正式 release 是否包含它们由 G8 决定；完整副本始终保留在 archive。
