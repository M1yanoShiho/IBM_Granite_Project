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

- **状态：** complete
- 开始条件已满足：G1 complete，archive branch/tag 已在远端验证。
- 下一步：建立隔离 worktree，从 archive commit 创建 `release/dissertation-v1`，形成受控保留树和归档映射。
- worktree 预检确认当前是普通 checkout，项目原先没有 `.worktrees/` 且未忽略该目录；按隔离执行规则在开发分支增加 `/.worktrees/` 防护。
- 已从 `research-archive-2026-08-25` 创建隔离 worktree 和 `release/dissertation-v1`，再带入 G1/G2 状态与 worktree ignore 两个运维提交；archive commit 仍是 release 的祖先。
- 已创建独立 Python 3.11 `.venv` 并成功安装 editable `dev + data-prep` 环境。
- 清理前基线：pytest 1877 passed/20 skipped/1 known historical failure；strict mypy 147 files PASS；全 scripts ruff 29 errors（历史脚本范围）。
- 清理前 tracked 体积审计：docs 119.6 MB、results 185.5 MB、runs 7.66 MB；src/tests/scripts 合计约 5.38 MB。最大 tracked 文件为 34.15 MB Selector labels。
- 第一批 release 边界已应用：从当前树移出 `.aris/`、`refine-logs/`、`runs/`、历史 docs 和 raw/intermediate results；保留文档、Exp04/05 最终报告及最终聚合结果正在迁移到公开路径。
- 第一批迁移在加入新的 `results/README.md` 时被旧 `/.gitignore` 的整目录规则中断；此前的精确删除/迁移仍完整保留在暂存区，没有回滚。修正方案是只放行公开的 `results/README.md`、`experiment04/` 和 `experiment05/`，其余运行结果继续忽略。
- 已修正公开结果目录层级和 ignore allowlist；Exp04/05 共 21 个文件均为 100% byte-preserving rename。当前 tracked tree 约 5.75 MB、没有超过 1 MiB 的文件。
- 第一批后个人/HPC 路径匹配只剩 `scripts/` 中 23 个文件；这些脚本将按 runtime/复现依赖闭包在下一批精简，而不是直接改写历史路径。
- 第二批移出 21 个带个人服务器地址且不再属于最终入口的旧启动/冻结脚本及对应历史测试；两个仍被 Exp04 复现测试引用的 dataset launcher 改为要求 `EVIDENCE_RAG_ROOT`、`EVIDENCE_RAG_VENV` 和 `MODEL_CACHE_DIR`，不再内嵌账号或绝对存储路径。
- 第二批首次全量 pytest 出现 16 个失败，均为 release 已移出文档/artifact 的路径依赖：2 个旧文档检查、5 个 Exp04 中间 artifact 检查、4 个 Exp04 最终结果旧路径检查、5 个旧 full-flow artifact 检查。处理边界：最终结果测试迁移到 canonical `results/`；只验证 archive-only 材料的测试从 release 移出。
- 16 个路径型失败已分类处理：Exp04 最终 audit/summary/表格测试改指 canonical 结果；其余 real-history artifact 断言移出，同时保留不依赖历史文件的 checker/unit tests。受影响测试 26 项通过，随后全量 pytest 退出码 0。
- 最终验收全部通过：release 来自 archive；远端 archive branch/tag 未变；896 个移出/迁移源路径可恢复；tracked tree 为 682 files/5,684,279 bytes；无 >1 MiB 文件、模型/缓存/index/raw output/个人路径；frozen source/config 未改。
- G2 commits：`396c08a`（外置历史记录和 raw artifacts）、`6ac02a5`（移出账号绑定启动器并迁移结果测试）。

### G3：冻结可运行的最终三模块系统

- **状态：** complete
- 开始条件已满足：G2 complete；用户已授权顺序接替执行。
- 下一步：重新审计最终 composition/config/model manifest、CPU smoke、真实权重失败边界与前端 API 交付面。
- 初步审计完成：composition 已包含 final class/factory 与模型哈希校验，但 development config、baseline-only smoke 和缺失 HTTP API 尚未达到 G3 产物要求。
- 已决定沿用计划中的前端边界：本仓库提供最小后端 HTTP API、冻结 response schema 和 mock；前端不加载 checkpoint、不共享 HPC 账号。
- G3 config TDD：新增 dataset/output `${ENV}` 展开与缺失变量测试，首次按预期 2 failed；实现集中环境变量展开后 2 tests 和 ruff 通过。正式 config 因此可用环境变量声明外部 dataset/output，而不是写个人路径。
- G3 runtime-config TDD：新 canonical final/smoke 路径测试首次按预期 2 failed（文件尚不存在）；迁移配置后组合测试又暴露两个旧常量仍指向原路径，已拆分为 CPU smoke config 与 final config 常量。
- G3 API TDD 首次按预期在 collection 阶段失败：`evidence_rag.api` 尚不存在。同时最新 Starlette 报告旧 `httpx` TestClient fallback 已弃用，开发依赖已改为其明确要求的 `httpx2 2.12`。
- 已复核 API RED 测试的冻结契约：health 必须保持 lazy，连续查询只允许加载一次 pipeline；response 必须同时交付 candidates、selected evidence、citations 和固定三模块 diagnostics。下一步按该测试实现 API package 与 serve CLI。
- API package、versioned schema、lazy service、FastAPI app 与 serve CLI 已实现；首轮 API/runtime focused pytest 45 项全部通过。strict mypy 随后发现真实 loader 把 `chunker_config` 误当作 `CorpusBuilder.build` 参数，现按已有 `build_chunker` factory 修正并增加 loader 回归测试。
- 修正后 focused pytest 46 项、ruff 和 strict mypy（152 source files）全部通过；API、TestClient 与其传递依赖的精确版本已写入开发 lock，供 clean CI/clone 复现。
- 已用 CPU 三模块执行生成前端响应，并据此新增 versioned mock、标准库调用示例和前端接口文档；`.env.example` 已改指 canonical final config，不再要求共享个人 HPC 目录。
- 空环境变量边界测试先按预期失败（未抛异常）；环境展开与模型预检现统一把缺失或纯空白变量视为未配置。CPU smoke 测试同时封锁 socket 连接，防止离线 smoke 意外联网。
- requirements lock 复装、`pip check`、公开 CPU smoke 命令、JSON 解析和 serve CLI help 均通过；安装过程发现历史锁定的 `build 1.5.1` 已被上游撤回，已改锁当前未撤回的 1.5.0，待复装确认。
- `build 1.5.0` 已复装且无 broken requirements；随后全量 pytest 退出码 0。G3 最终模型审计发现 Selector base `config.json` hash 尚未进入 runtime gate，下一步先加失败测试再补校验。
- Selector base hash 测试先按预期因 `model_config_sha256` 未注册而失败；factory 现要求并验证 snapshot `config.json`，formal TOML 与 model manifest 的 hash 已对齐。
- 首次 Selector focused 回归有 1 个旧 missing-env fixture 未提供新增必需 hash，因此提前失败在参数完整性检查；fixture 已补通用 hash，占位仍不会越过缺失 checkpoint 环境变量边界。
- 修正后 Selector/runtime focused tests、ruff 和 strict mypy 通过。进一步扩展 manifest consistency 与 generic 503 隐私测试后，G3 API/runtime/architecture focused 共 66 项通过；全 `src/tests` ruff 和 152-source strict mypy 均通过。
- G3 末轮全量 pytest：1878 passed、20 skipped；secret/weight/large-file scan 无命中且没有 >1 MiB 文件。全树路径扫描仍显示 legacy Slurm 脚本的 `/user/work/$USER`，已分类为 G4 复现脚本收敛输入，不属于 final runtime surface。
- G3 final surface 审计：恰好 1 个 `final_*.toml` 和 1 个 `final_*.json`；新增/修改 runtime/API/docs/examples 无个人路径；公开 smoke trace 再次证明 citations 是 selected evidence 子集且 poison 在 Selector 后消失。人工复核 API schema/service/app/serve 未发现新的阻塞问题。
- G3 runtime 首次精确暂存未发生：`git add` 对已重命名且工作树中不存在的两个旧路径报 pathspec 不匹配。后续只对已审计的 `configs/experiments`、`configs/models` 使用 index update，再显式加入新路径和其余文件。
- G3 commits：`9a79129`（portable seed-13 runtime/config）、`34b3dff`（frontend HTTP API）、`dbd023c`（frontend contract/mock）。
- 提交后再次验证：1878 passed、20 skipped；全 `src/tests` ruff、152-source strict mypy、CPU smoke JSON、`pip check` 和 whitespace 全部通过。G3 验收门全部满足。

### G4：整理研究复现包

- **状态：** complete
- 开始条件已满足：G3 complete；用户已授权顺序接替执行。
- 已知输入：legacy Slurm 路径、92 个旧 Selector sweep 配置和历史阶段脚本需要按最终论文主张/结果依赖闭包分类，而不能直接批量保留。
- 下一步：重读 G4 计划，盘点 Selector/Generator/Experiment 04/05 的最终命令、配置、源码、测试和聚合结果映射。
- 已完成第一轮树盘点：正式结果端已收敛到 5 份研究报告与 Exp04/05 共 20 个聚合结果文件；代码端仍有约 150 个顶层 scripts 和约百个旧 sweep configs。下一步从冻结报告/审计反向追踪最终命令与依赖，而不是按文件名猜测保留。
- 已对照目标公开树与最终脚本 docstring，初步冻结四个复现入口及 Exp04/05 阶段链；下一步生成实际 import/file-reference 依赖图，并确认 Generator G300/G310/G400/G410 与 Selector lean v3 的训练输入链边界。
- Selector 审计确认 final lean v3 已有包内 CLI/config/focused tests；Generator 资格链仍直接导入 B100/G230/joint legacy helpers。下一步用源码 import graph 得到各根入口的最小递归闭包，再决定迁移还是抽取 helper。
- 递归 import graph 已完成：Generator roots 的最小 closure 为 8 个 scripts；Exp04 Goal5 closure 为 Goal3/4/5 三文件，Goal1 另依赖 heldout_data。已确认当前 Exp04 defaults 指向已归档 full-flow 目录，G4 必须用明确 external input contract 替代这些隐式默认。
- 小型结果结构审计完成：Exp04 final JSON 和 Exp05 table/claim/bootstrap JSON 足以作为 table-only rebuild 输入，但现有正式 compilers 只支持 raw runroot。下一步以 committed outputs 为 golden tests，先 RED 再实现公开 table rebuild。
- 发现 Exp05 `final_tables.md` 还混入独立 Selector stress-test/贡献表并引用 archive-only 路径；先从 archive 恢复 Selector 两项最终聚合证据，再设计可追踪的公开表格 rebuild，避免把人工表当成新的数据源。
- 已从 immutable archive 定位 R005AB Lean v3 `L003_FINAL_REPORT.json`，确认它同时记录 evidence PASS 与 answer/overall FAIL。下一步基于该单一 source hash 生成两份公开聚合 JSON，并加一致性测试阻止选择性报告。
- 权威 L003 JSON 已用 Python 流式计算 SHA-256 为 `b031b6f2…2495c`；macOS `shasum` 再次因无效 locale 崩溃，该轮输出已明确作废。
- Public table-rebuild 测试已先建立并按预期在 collection 阶段失败：`evidence_rag.evaluation.public_tables` 尚不存在。已确认 Exp04 schema/metric order 与 Exp05 frozen dataset/metric order，下一步实现最小纯渲染模块。
- 首次写入纯渲染模块后，pytest 在导入阶段发现 6 处 LaTeX 行尾反斜杠字符串未闭合；模块尚未执行、冻结结果未改变。现只修复字符串转义并重跑黄金文件测试。
- 修复转义后，公开表与 Selector 结果共 4 个 focused tests 通过；重建产物与冻结文件逐字节一致，ruff、strict mypy 与 whitespace 检查通过。下一步建立四个读者可见的 `experiments/` 入口并收敛脚本依赖闭包。
- 公开入口 TDD 首轮按预期 3 failed（Selector console command 与两个 table builder 尚不存在）；实现薄包装后 7 tests 和 ruff 通过。strict mypy 随后识别两个同名 `build_tables.py` 尚无 package 边界，现增加最小 `__init__.py` 后复验。
- 增加 package 边界后，7 个公开入口/重建/Selector 结果测试、ruff、strict mypy 全部通过。开始根据冻结报告和 audit 编写论文主张到证据的逐项映射。
- 四个实验 README、Generator 三 seed provenance 和根级复现映射已建立。文档契约首轮发现 Exp04/05 两个 aggregate 名称未写完整目录前缀，已补成可直接定位的 canonical paths 后复验。
- G4 allowlist 外置批次完成：265 个候选全部先通过 archive recovery；release 现只保留 29 个 Python research scripts、2 个 Exp04 Slurm 和 18 个配置。删除后 0 个失效 script imports，37 个 focused tests 通过。下一步运行全量 pytest/ruff/mypy 并处理真实回归。
- 精简后全量 pytest 首轮退出码 0。全范围 ruff 随后暴露 3 个保留 Generator 资格脚本的 9 个历史 lint 项；5 个由 formatter 机械修正，余下 3 个 lambda 和 1 个未使用循环变量做等价改写后复验。
- Generator focused 回归捕获 G410 两个相邻循环的机械改名落在了实际读取路径的第一个循环；未提交。现恢复第一个 `path`，仅把第二个真正未使用变量改为 `_path` 后重跑。
- 修正后 7 个 Generator 资格回归、全 `src/tests/scripts/experiments` ruff、153-source strict mypy、两个 Slurm 语法和 whitespace 全部通过。为取得 pytest 数量而清空 `addopts` 的一次复验同时误删了必需的 `--import-mode=importlib`，导致同名测试 collection 冲突；该调用无效，恢复正常 pytest 配置复验。
- 正常 importlib 模式的精简后全量回归为 1639 passed、20 skipped。删除引用扫描只剩两个无运行依赖的测试文字命中（历史 preflight 注释、函数名中的 `topk_baseline`），已改成不冒充文件引用的表述；个人/HPC 路径扫描为 0。
- G4 commits：`e398bd4`（公开复现入口、结果映射和 Selector 双门结果）、`b5e5fe6`（外置 265 个旧阶段文件并收敛 lint 面）。
- G4 提交后最终验证：1639 passed、20 skipped；全范围 ruff、153-source strict mypy、Slurm 语法、公开 JSON、安装后 Selector CLI、`pip check` 和个人路径扫描全部通过；265 个删除项全部可从 archive tag 恢复。

### G5：发布外部模型与数据资产清单

- **状态：** complete
- G4 验收门已满足；用户已授权顺序接替执行。
- 权限边界：尚无团队/导师对衍生权重和 raw bundle 公开上传的明确确认，因此 G5 可先完成只读许可证/哈希/大小/可用性审计，不会上传受限资产或把团队访问写成公众访问。
- 本地模型文件扫描为 0；无法从当前工作区重新读取 HPC 权重，因此使用冻结 archive manifest、已验证的用户记录和公开上游 model card 三方交叉核对。
- 已区分 Selector 基础模型 snapshot（737,726,552 bytes）与训练后 seed-13 checkpoint（737,731,768 bytes），避免资产清单混淆两者。
- 已提取 G330 三 seed 的实际 weights/config SHA、Granite revision、LoRA recipe 和训练版本；冻结 manifest 未包含 adapter 字节数，不能用参数量伪造大小。
- 已用上游固定 revision API 复核 7 个基础模型的许可证、精确 weights bytes 和 SHA；全部 frozen hash 对齐。Provence 授权文本互相矛盾，采用“不镜像、仅链接、非商业/无衍生再分发”的保守边界。
- 已核对 Exp04/Exp05 数据边界；所有 raw bundles/indexes/generations 继续外置，公开仓库只提供上游来源、冻结哈希和聚合结果。
- 已只读登录 BluePebble 复核资产：旧 `/scratch` checkpoint 是计算节点临时资产，但独立保留副本仍在共享项目存储。三个 GR-C adapter weights 均为 62,332,992 bytes、configs 均为 1,274 bytes，四类 derived assets 的现场 SHA 全部与 frozen manifest 一致。
- 资产清单/CLI TDD：首次 collection 按预期因 `evidence_rag.cli.artifacts` 不存在而失败；实现 manifest、原子下载、fail-closed 校验与文档后 6 tests 通过。取得保留副本精确大小后，测试先按预期拒绝旧 `null` 值，再更新正式清单。
- 清单更新后 6 tests、focused ruff、单模块 strict mypy、JSON parse 与 19 项 CLI list 全部通过；27 个唯一上游 source/download URL 逐一 HEAD 验证均返回 HTTP 200。
- 增加 final seed13 runtime identity 与资产 manifest 一致性契约后，focused suite 为 7 passed；Exp04 三个源 URL 也已从移动的 main/master 改为 frozen revision，更新后的全部链接仍返回 HTTP 200。
- G5 最终验收：真实 HotpotQA 27,452,575-byte 下载与 SHA 校验 PASS；全量 1646 passed、20 skipped；全范围 ruff、154-source strict mypy、editable install、安装后 CLI、`pip check` 和 whitespace 全部通过。
- 未经团队/导师明确授权，四个 derived weights 继续保持 `restricted-not-published`；这是清楚记录的发布边界，不冒充公众可复现。

### G6：完成公开文档与项目元数据

- **状态：** complete
- G5 验收门已满足；开始审计 README、文档导航、LICENSE/CITATION/CHANGELOG、作者贡献和 model cards。
- 已完成公开文档盘点：根 README 与旧 docs 导航均不足或过时；setup/architecture/reproduction/results/limitations、两个 model cards、citation/changelog/contribution/author 元数据尚缺。
- 已确认文档会使用最终 Hybrid Retriever → trained NLI Selector → grounded GR-C Generator，并把正向 evidence gate 与负向 answer/whole-system gates 同时呈现。
- 已按文档规范确定无 inline style、含可访问性描述的 Mermaid flowchart；接下来先写公开文档契约测试，再实现文档。
- 许可证仍等待项目所有者明确选择，不会自动写入开源授权文本；其余 G6 工作不因此暂停。
- G6 文档契约已先写测试并取得预期 RED：9 项全部失败，分别锁定缺失文件、README 主入口、系统图、科学边界、CFF、package metadata、相对链接、路径隐私和许可证确认。
- 已在当前 release worktree 实际复跑 CPU smoke 并确认输出契约，且从归档 Git history 提取仅含 commit identity 的 contributor 清单；两者将分别用于 quick start 与 AUTHORS，不引入个人路径或邮箱。
- 已从 CFF 1.2.0 官方 schema 确认 entity author 和根必需字段，可用标准 JSON/YAML 1.2 语法生成无需新增依赖且不虚构个人作者顺序的 `CITATION.cff`。
- 已重写根 README 与文档导航，新增 architecture/setup/reproduction/results/limitations、Selector/Generator model cards、CITATION、CHANGELOG、CONTRIBUTING 和 AUTHORS，并更新 package metadata 到 1.0.0。
- 当前文档契约 8 passed、1 failed；唯一失败是 owner-selected license。`LICENSE` 目前是明确不授予许可的临时保护文本，避免在所有者确认前自动开放权限。
- CFF 已通过标准库 JSON、Ruby YAML 和官方 CFF 1.2.0 JSON Schema 验证；Schema 验证需把 `.cff` 临时复制为 `.json`，因为 AJV CLI 按扩展名误解析 `.cff`。
- Package 1.0.0 已成功构建 sdist/wheel、重新 editable install，`pip check` 与安装后 CPU smoke 均通过；文档标题契约首次运行发现测试自己的批准 emoji 集漏列官方 `💾`，已补全测试词表。
- 首轮 G6 全量回归除许可证外还发现 1 个旧测试仍强制从已重写的 `docs/README.md` 查找 reference baseline 命令；已把这份仍有效的运行说明迁移到正式 reproduction 文档，并同步测试读取新的权威位置。
- baseline 文档契约修正后，全量非许可证回归为 1655 passed、20 skipped、1 deselected；ruff、154-source strict mypy、两个 Slurm shell 语法、package build/install、pip check、CPU smoke、CFF schema、内部链接和 whitespace 全部通过。
- G6 七项验收门已有六项通过；唯一未通过的是项目所有者尚未确认 MIT、Apache-2.0 或保留 All Rights Reserved。G6 保持 in_progress，G7 不提前启动。
- 自动 continuation 重新读取计划并复核工作区：当前确为 linked worktree `release/dissertation-v1`，不是 submodule；归档现场仍与 release 修改隔离。计划仍只允许 G6 in_progress，许可证未确认前不启动 G7。
- continuation 恢复检查未报告未同步上下文；磁盘上的 `task_plan.md`、`findings.md`、`progress.md` 与 Git 工作树是本轮继续验证的权威状态。
- 当前 G6 diff 边界已复核：只含公开 Markdown、CFF/LICENSE/作者/变更记录、pyproject 和相关文档契约；未改冻结结果、模型 manifest、runtime 源码或 archive refs。下一步用 Node 临时工具实际渲染两份 Mermaid 图。
- Mermaid CLI 11.16.0 已实际渲染 README 和 architecture 各 1 张图，两次均报告成功；生成物仅用于临时验证，不进入 release tree。
- 公开文档新增的 14 个外部 URL 全部实时返回 HTTP 200。GitHub 只读权限审计确认仓库 public/default main，当前登录者为 WRITE（非 ADMIN）；历史与源码没有可继承的项目 LICENSE/header，因此仍需 owner/团队明确选择。
- 已同步内部计划的跨 Goal 问题：前端、衍生资产、raw outputs 和外部承载边界均标为已保守解决；许可证保持唯一 G6 open 项，G8 权限保留到实际操作验证。旧 config 资源名也更新到 final canonical 路径。
- Wheel 内容审计发现 AUTHORS 被构建工具误归为 license file；按 TDD 先新增 metadata 契约并取得缺少 `license-files` 的预期 RED，再加入精确 `license-files = ["LICENSE"]`。focused test GREEN，新 sdist/wheel 构建通过，wheel 只含 LICENSE。
- 全读者文档链接审计稳定复现 16 个旧路径。根因是文件已在 G2/G4 迁移到 canonical `results/`，而早期 research/final_tables 文档和窄范围契约未同步。扩展 link test 后先取得 16-link RED，再把 12 个 Exp04 与 4 个 Exp05 链接映射到正式结果/报告或 archive boundary；GREEN 且独立 30-document scanner 为 missing 0。
- 链接与 wheel metadata 修正后的全量非许可证回归再次通过：1655 passed、20 skipped、1 license test deselected；全范围 ruff、154-source strict mypy、两个 Slurm 语法、pip check 和 whitespace 均 PASS。冻结表格/结果契约包含在该全量测试中且未出现回归。
- 本轮结束前单独复跑许可证契约，仍按设计 1 failed：临时 `LICENSE` 不含 MIT/Apache/All Rights Reserved 任一 owner-selected policy。工作树未暂存、未提交、未推送；等待 owner/团队确认后才能完成 G6。
- 用户已明确回复 `MIT`，解除唯一许可证阻塞；`LICENSE` 已替换为标准 MIT 文本，版权主体与 CFF/AUTHORS 的团队实体保持一致。
- MIT 精确许可证契约通过；随后完整回归为 1656 passed、20 skipped，ruff、154-source strict mypy、两个 Slurm 语法、package 1.0.0 sdist/wheel、`pip check`、CFF JSON、CPU smoke 和 whitespace 全部退出码 0。
- G6 diff 边界与大文件/密钥扫描通过：只包含公开文档、metadata、文档契约及规划记录，没有新模型权重、冻结算法或聚合结果数据改写。

### G7–G8

- **状态：** pending
- 在 G6 验收通过后按 `task_plan.md` 顺序接替执行。

## 测试结果

| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|---|---|---|---|---|
| 规划文件存在性 | `task_plan.md`, `findings.md`, `progress.md` | 三个文件均存在 | 三个文件和 cleanup inventory 均存在，链接有效 | PASS |
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
| G2 isolated install | Python 3.11 + `.[dev,data-prep]` | editable install 成功 | build/install exit 0 | PASS |
| G2 pre-cleanup pytest | archive-derived release tree | 与 G1 基线一致 | 1877 passed、20 skipped、1 known G000 failure | KNOWN-FAIL |
| G2 pre-cleanup ruff | `src tests scripts` | 记录清理前 lint 债务 | 历史 scripts 29 errors；正式新增范围此前通过 | BASELINE |
| G2 pre-cleanup mypy | `src` + `tests/typecheck.py` | 0 errors | 147 source files 无错误 | PASS |
| G2 archive recovery | 第一批移出的 872 个源路径 | archive tag 全部可读取 | `git cat-file -e` 全部通过 | PASS |
| G2 public result relocation | 5 个 final docs + 20 个 final result/audit 文件 | 与 archive 源文件逐字节一致 | 25 个 `git hash-object` 比较一致 | PASS |
| G2 portable launchers | 两个保留的 Exp04 Slurm 文件 | shell 语法、阶段顺序、环境变量边界正确 | `bash -n` 和 4 个 focused tests 通过 | PASS |
| G2 personal path scan | release tracked tree（规划/迁移记录除外） | 无账号和个人绝对路径 | 无匹配 | PASS |
| G2 post-boundary pytest | 当前 release tree | 0 failed | 全量退出码 0 | PASS |
| G3 focused runtime/API | final runtime、API、architecture、config | 0 failed | 66 项通过 | PASS |
| G3 full pytest | 提交后的 release tree | 0 failed | 1878 passed、20 skipped | PASS |
| G3 ruff | `src`、`tests`、公开 Python example | 0 errors | All checks passed | PASS |
| G3 strict mypy | `src/evidence_rag` + typecheck contract | 0 errors | 152 source files 无错误 | PASS |
| G3 CPU smoke | final classes + offline doubles + socket denial | 无网络/权重/GPU且 trace 合法 | poison 被删除；citations 是 selected subset | PASS |
| G3 model manifest | final TOML 与 model JSON | IDs/revisions/hashes 一致 | Retriever/Selector/Generator/TRUE 全部对齐 | PASS |
| G3 dependency lock | `requirements-dev.lock` | 可复装且无 broken requirements | build 1.5.0/API 依赖复装；`pip check` 通过 | PASS |
| G4 public table rebuild | Exp04/Exp05 small aggregates | 表格与冻结产物逐字一致 | Exp04 6 files、Exp05 3 tables + report 全部一致 | PASS |
| G4 scientific result boundary | Selector/Exp04/Exp05 public results | 正负结果与 claim labels 同时保留 | evidence PASS + answer FAIL；Exp04 superiority/Exp05 Claim A/B 未支持 | PASS |
| G4 archive recovery | cleanup commit 265 deletions | 所有路径可从 archive tag 读取 | archive missing 0 | PASS |
| G4 final pytest | 精简后的正式树 | 0 failed | 1639 passed、20 skipped | PASS |
| G4 final static checks | `src tests scripts experiments` | ruff/mypy/shell/whitespace 全部通过 | ruff PASS；strict mypy 153 files；两个 Slurm `bash -n` PASS | PASS |
| G4 install/CLI | editable install + public commands | 安装后命令可发现 | Selector 和两个 table builders help；`pip check` PASS | PASS |
| G5 artifact manifest | final models/datasets/bundles | 版本、bytes、SHA、license、availability 完整 | 19 assets；四个 derived assets 和七个 base models 均完整登记 | PASS |
| G5 live derived assets | 共享存储保留副本 | bytes/SHA 与 frozen manifests 一致 | Selector 737,731,768 bytes；三 adapters 各 62,332,992 bytes；全部 SHA 一致 | PASS |
| G5 external links | 27 个唯一 source/download URL | 无 HPC 账号可访问 | 全部 HTTP 200；restricted assets 明确无 URL | PASS |
| G5 real download | pinned HotpotQA parquet | 原子下载后 bytes/SHA 一致 | 27,452,575 bytes，SHA `c20b638c…f7c6`，PASS | PASS |
| G5 full regression | release tree | 0 failed | 1646 passed、20 skipped；ruff/mypy/pip check/CLI 均 PASS | PASS |
| G6 documentation TDD RED | 9 项公开文档契约 | 在实现前因缺失/过时内容失败 | 9 failed，失败面与计划产物一致 | EXPECTED-FAIL |
| G6 documentation partial GREEN | 9 项公开文档契约 | 除许可证确认外均通过 | 8 passed、1 expected license blocker | PARTIAL |
| G6 CFF parse/schema | `CITATION.cff` | JSON、YAML 与 CFF 1.2 schema 有效 | 三种验证均 PASS | PASS |
| G6 package build/install | package 1.0.0 + API/dev extras | sdist/wheel、editable install、pip check、smoke 通过 | 全部 PASS | PASS |
| G6 wheel license boundary | wheel metadata/files | 只把 LICENSE 标为 license file | RED→GREEN；新 wheel 仅 `dist-info/licenses/LICENSE` | PASS |
| G6 full regression excluding license blocker | release tree | 0 unexpected failures | 1655 passed、20 skipped、1 license test deselected；ruff/mypy/shell/pip 全 PASS | PASS |
| G6 post-link full regression | release tree after full-link and wheel fixes | 0 unexpected failures | 1655 passed、20 skipped、1 license test deselected；all static gates PASS | PASS |
| G6 Mermaid rendering | README + architecture | 两张图可由当前 Mermaid CLI 渲染 | 2/2 charts rendered successfully | PASS |
| G6 public documentation URLs | 14 unique external URLs | 当前无 HPC 账号环境可访问 | 14/14 HTTP 200 | PASS |
| G6 full public relative links | 30 reader-visible Markdown files | 所有相对链接目标存在 | RED 16 missing → GREEN 0 missing | PASS |
| G6 license owner confirmation | `LICENSE` + exact public-release contract | 由仓库 owner/团队明确选择许可证 | 用户明确选择 MIT；exact contract 1 passed | PASS |
| G6 final regression | MIT + complete public documentation tree | 0 failures，构建/静态检查/smoke 完整 | 1656 passed、20 skipped；ruff/mypy/build/pip/shell/CFF/smoke/whitespace 全 PASS | PASS |

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
| 2026-08-25 | `git check-ignore -q .worktrees` 对尚不存在的空目录未命中 | 1 | 改为验证 `.worktrees/release-placeholder`，确认 `/.worktrees/` 规则生效。 |
| 2026-08-25 | G2 第一批迁移加入 `results/README.md` 时命中原有 `/.gitignore` 的 `/results/` 整目录规则 | 1 | 保留已暂存的精确迁移；把规则收窄为默认忽略 `results/*`，仅放行最终 Exp04/05 聚合结果和说明文件，并修正多余目录层级。 |
| 2026-08-25 | 第一轮迁移 SHA-256 比对中的 macOS `shasum` 因无效 `C.UTF-8` locale 多次 panic，导致该轮计数不可作为有效证据 | 1 | 不采信该轮哈希结果；改用 `git hash-object` 对 archive blob 流和工作树文件逐字节比较。 |
| 2026-08-25 | 第二批路径扫描命中新加入的 launcher 测试，因为测试本身写入了旧账号字面量 | 1 | 删除账号字面量；保留对 `/user/work/` 绝对路径和三个必需环境变量的通用断言，再重跑扫描与测试。 |
| 2026-08-25 | 第一批 docs/results 外置后全量 pytest 有 16 个测试读取已归档的旧路径而失败 | 1 | 分类为最终结果契约与纯历史 artifact 契约；前者改指 canonical `results/experiment04/`，后者连同其 archive-only 前置材料一并从 release 测试面移出。 |
| 2026-08-25 | 第二批提交前 `git diff --cached --check` 发现 4 个删减后的测试文件末尾多余空行 | 1 | 提交被安全中止；用精确补丁移除四个 EOF 空行并重复格式检查。 |
| 2026-08-25 | G3 引用扫描把 ripgrep 的排除参数误写成 Git pathspec，输出两个 “No such file” | 1 | 不复用该写法；后续 `rg` 使用标准 `-g '!pattern'`，Git 内容扫描再使用 `:(exclude)` pathspec。 |
| 2026-08-25 | G3 环境预检补丁因函数返回签名上下文过宽，首次插入 `build_baseline_from_corpus` 而非正式 config factory | 1 | RED 测试仍先调用 Retriever 并暴露错误；将预检精确移到 `build_pipeline_from_config` 的函数体首行。 |
| 2026-08-25 | CPU smoke 首轮替换补丁对同一路径同时使用 Delete/Add，编辑工具拒绝执行 | 1 | 文件未改变；改用单一 Update File 补丁原地替换实现。 |
| 2026-08-25 | CPU smoke 功能测试通过后 strict mypy 发现离线 NLI double 返回普通 `str`，不满足三标签 Literal protocol | 1 | 将 double 的签名和返回值收窄为正式 `NLIModel` protocol；不改 smoke 行为。 |
| 2026-08-25 | G3 契约复核命令引用了不存在的单文件路径 `src/evidence_rag/pipeline.py` | 1 | 项目使用 `pipeline/` package；停止猜测路径，后续先列出 package 文件再读取准确的 `service.py` 与 contracts。 |
| 2026-08-25 | G3 API 功能测试通过，但 strict mypy 发现真实 loader 向 `CorpusBuilder.build` 传入不存在的 `chunker_config` 关键字 | 1 | 改为使用现有 `build_chunker` 创建实例并传给 `CorpusBuilder` 构造器；新增不加载外部模型的真实 loader 回归测试。 |
| 2026-08-25 | G3 文档补丁在同一次 `apply_patch` 中对 handoff 文件同时 Delete/Add，被编辑器拒绝 | 1 | 文件未改变；拆分为原地 Update 与独立新增文件，不再对同一路径使用同批 Delete/Add。 |
| 2026-08-25 | requirements lock 安装提示 `build==1.5.1` 是上游 yanked release | 1 | 查询包索引确认当前稳定最新版为 1.5.0；将 lock 改为 `build==1.5.0` 并重新安装验证。 |
| 2026-08-25 | Selector base hash gate 加入后，一个 missing-checkpoint 测试先因缺少新增 `model_config_sha256` 而失败 | 1 | 为旧 fixture 补必需 hash 占位，使测试继续验证原本的 checkpoint 环境变量失败顺序。 |
| 2026-08-25 | G3 runtime 暂存命令把已不存在的 rename 源路径作为普通 `git add` pathspec，Git 拒绝且未创建提交 | 1 | 只在两个已审计 config 目录运行 tracked index update，再显式 `git add` 新目标和 runtime allowlist。 |
| 2026-08-25 | G4 计算 archive Selector JSON SHA-256 时再次调用了已知受 locale 影响的 macOS `shasum`，进程 panic | 1 | 不采信该输出；改用 Python `hashlib.sha256` 对 Git blob stream 计算并得到完整 digest。后续不再使用 `shasum`。 |
| 2026-08-25 | G4 公开表重建模块首次写入时，LaTeX 行尾反斜杠在 6 个字符串中转义不足，pytest collection 报 `SyntaxError` | 1 | 模块未执行且结果未改变；把行尾改为合法的双反斜杠字符串并重新运行黄金文件测试。 |
| 2026-08-25 | G4 两个公开 table builder 同名且目录无 Python package 标记，strict mypy 报 duplicate module | 1 | 为 `experiments/` 和两个子目录增加最小 `__init__.py`，保持脚本入口不变并重新类型检查。 |
| 2026-08-25 | G4 结果摘要只读命令误用系统 `python`，当前 shell 没有该别名 | 1 | 未发生写入；后续统一使用 release worktree 的 `.venv/bin/python`。 |
| 2026-08-25 | G4 精简后首次全范围 ruff 在 G310/G400/G410 报 9 个旧风格问题 | 1 | 仅做 import 排序、移除未使用 import、命名未使用变量，并把 lambda 赋值改为等价局部函数；随后运行对应回归与全量静态检查。 |
| 2026-08-25 | G410 有两个相邻 `for seed, path` 循环，首次 lint 修正误把实际读取文件的第一个变量改成 `_path`，focused test 报 `UnboundLocalError` | 1 | 测试在提交前捕获；恢复第一个 `path`，只改第二个未使用变量并重跑全部相关测试。 |
| 2026-08-25 | 为显示 pytest 总数而使用 `-o addopts=''`，意外移除了项目必需的 `--import-mode=importlib`，8 个同名 test module collection 冲突 | 1 | 不是代码回归；改用 `-o addopts='--import-mode=importlib' -q` 保留导入模式并显示单次 summary。 |
| 2026-08-25 | G5 在整个 archive 上用宽泛数值正则搜索 adapter 大小时命中大体积逐题 JSON，输出被截断且不能作为资产证据 | 1 | 放弃该输出；后续只枚举冻结 manifest/audit 文件，并用 `jq` 按字段路径提取大小、哈希和模型身份。 |
| 2026-08-25 | G5 首次数据盘点假设存在 `configs/experiments/experiment05/` 和 `scripts/experiment05_data.py`，两个路径已在 G4 收敛后不存在 | 1 | 不猜路径；改为枚举正式 `scripts/experiment05_*` 入口并从 `experiment05_goal1.py` 与公开 audit 提取数据边界。 |
| 2026-08-25 | G5 按 archive 旧 `/scratch` 路径查询四个权重时全部不存在；该位置属于计算节点临时空间，不是登录节点共享目录 | 1 | 未修改远端文件；转而检查共享项目目录与本地恢复源，并把私有 adapter 的大小标为必须由保留副本重新 `stat`，不猜测。 |
| 2026-08-25 | G5 远端全工作区 `du` 与宽层级查找超过 30 秒，只返回登录目录与缺失 scratch 信息 | 1 | 停止扫描大目录；改用已知共享项目根的精确存在性/文件名查询，确认其中没有正式权重。 |
| 2026-08-25 | G5 首次 URL 审计命令包含临时文件 `rm -f` 清理，被安全策略拒绝且未执行 | 1 | 改用 shell process substitution，不创建临时文件；27 个 URL 全部完成 HTTP 200 验证。 |
| 2026-08-25 | G5 正式配置一致性检查沿用旧计划名，引用了不存在的 `three_module_seed13.json` 和 `final_three_module.toml` | 1 | 先枚举当前 `configs/`，改用 G3 已定稿的 `final_seed13.json` 与 `final_seed13.toml`；七项 runtime identity 比对全部通过。 |
| 2026-08-25 | G5 提交前 whitespace 检查发现三个新 Python 文件末尾多余空行 | 1 | 提交被安全中止；用精确补丁移除空行、重新暂存并验证后提交。 |
| 2026-08-25 | Ruby 版本没有 `YAML.safe_load_file`，首次 CFF YAML 命令报 `NoMethodError` | 1 | 改用 `YAML.safe_load(File.read(...))`，解析通过。 |
| 2026-08-25 | 首次 AJV CFF schema 验证直接读取 `.cff` 报 `Unexpected token ':'`；带 `rm -f` 的临时清理命令又被安全策略拒绝 | 2 | 不删除或覆盖项目文件；把 CFF 精确复制到 `/tmp` 的 `.json` 路径再对官方 schema 验证，结果 valid。 |
| 2026-08-25 | 新增 H2 可扫描性契约漏列 Mermaid 规范允许的 `💾`，导致 architecture 文档误报 1 项失败 | 1 | 文档用法正确；补齐测试中的批准 emoji 集并重新运行。 |
| 2026-08-25 | G6 首轮全量 pytest 有 1 个非许可证失败：历史测试仍把旧 `docs/README.md` 当 reference baseline 命令说明；同时 shell 语法检查猜错两个 Slurm 路径 | 1 | 将仍有效的 baseline/索引说明迁移到 `docs/reproduction.md` 并更新契约；枚举后改用 `scripts/run_experiment04_goal3_dataset.slurm` 与 `scripts/run_experiment04_goal4_dataset.slurm`。 |
| 2026-08-25 | 迁移后的 baseline 文档把契约短语 `live Pipeline` 在 Markdown 源码中换行，focused test 仍失败 | 1 | 合并该句源码行，不改变渲染内容；重新运行精确契约。 |
| 2026-08-25 | Mermaid CLI 输出把 SVG 的相对显示写成 `./...`，误以为文件落在 worktree 根并尝试移动时报告不存在 | 1 | `git status` 与根目录精确查找确认没有生成物进入 worktree；不重复移动，后续只检查 `/tmp` 输出目录。 |
| 2026-08-25 | 扩大 G6 文档扫描后发现 16 个相对链接仍指向已迁移/归档路径 | 1 | 追踪到 G2/G4 文件迁移与测试覆盖不足；扩展契约范围，链接 canonical aggregates/reports，并把 archive-only 文件改为明确 archive ref。 |
| 2026-08-25 | 用户恢复被阻塞目标后，`create_goal` 拒绝新建，因为原目标仍被系统视为 unfinished | 1 | 不创建重复目标；沿用原 G1–G8 目标继续执行，并只在全部验收后更新为 complete。 |

## 五问重启检查

| 问题 | 答案 |
|---|---|
| 我在哪里？ | P0/P1/G1–G6 已完成；正式候选版本准备进入 G7。 |
| 我要去哪里？ | 执行 G7 clean-clone/CI/HPC 验证，再执行 G8 正式发布。 |
| 目标是什么？ | 同仓库内形成可提交毕设的干净可用正式 main，同时保留完整研究历史。 |
| 我学到了什么？ | 见 `findings.md`。 |
| 我做了什么？ | 已冻结历史、建立干净 release，交付三模块 runtime/API、复现包、资产清单和公开文档；现按用户决定写入 MIT。 |

---
*每个 Goal 完成后或遇到错误时更新此文件。*
