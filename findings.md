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

## G2 隔离基线

- Release worktree 位于项目忽略的 `.worktrees/dissertation-v1`，分支 `release/dissertation-v1`，并以 `research-archive-2026-08-25` 为祖先。
- 隔离 Python 3.11 环境已通过 `pip install -e '.[dev,data-prep]'` 安装成功。
- 清理前全量 pytest 复现 archive 基线：1877 passed、20 skipped、1 个旧 G000 README/TRACKER 状态失败。
- 清理前 strict mypy 仍为 147 source files 无错误。
- 清理前全 `scripts/` ruff 有 29 errors，集中在历史分析/Full-flow 阶段脚本；新增 Exp04/05 和 `src/tests` 的 focused ruff 已通过。G2/G4 应仅保留正式依赖脚本，使最终 ruff 作用面与公开树一致，而不是批量格式化 archive-only 脚本。
- Release worktree 的 tracked 内容约为：`docs` 119.6 MB/642 files、`results` 185.5 MB/170 files、`runs` 7.66 MB、`scripts` 2.01 MB/171 files、`src` 1.82 MB/146 files、`tests` 1.55 MB/234 files。
- `results`、历史 `docs` artifacts 和 `runs` 是 >25 MiB release gate 的主要来源；最大 tracked 文件为 34.15 MB Selector labels。所有这些大文件都能从 archive tag 恢复，不需要留在 release tree。
- G2 不应按体积直接删除 `src/tests/scripts`：它们虽然小，但需要 G3/G4 dependency closure 后才可精简；本阶段先移出明确的 `.aris/refine-logs/runs/results`、历史 docs/artifacts、配置 sweep 和无争议旧脚本组。
- G2 第一批外置后，tracked tree 已从约 301 MB 降到约 5.75 MB；`docs` 约 99 KB、最终 `results` 约 93 KB，且没有 tracked 文件超过 1 MiB。
- Exp04/05 的 21 个公开结果文件都由 Git 检测为 100% rename，说明迁移没有改变文件字节；其中包含冻结 audit、claim labels、bootstrap 和最终表格，不包含逐题输出或原始 generations。
- 原有 `/.gitignore` 将 `results/` 整体排除；正式树应采用 deny-by-default 并只放行 `results/README.md`、`experiment04/` 和 `experiment05/`，避免今后把 raw results 误提交。
- 清理后个人/HPC 路径启发式匹配已从约 242 个 tracked 文件降到 23 个，全部位于仍待依赖审计的历史 `scripts/`；G2 的下一精简批次应以正式 runtime/复现入口的依赖闭包决定保留项。
- 23 个剩余路径匹配中，21 个是 archive-only 的旧 server/Slurm/freeze 入口；另外两个是最终 Exp04 dataset launcher。前者可安全移出，后者可只把 workspace、venv 和 cache 变为必需环境变量，不改变实验阶段顺序或 scorer 隔离边界。
- 第一批外置后的 16 个 pytest 失败不是 runtime 回归：它们全部读取已移出的历史文档或中间 artifacts。将三项最终 Exp04 聚合契约改指 `results/experiment04/`，并移除纯 archive artifact 集成断言后，全量 release 测试重新通过。
- G2 最终验收：release tag ancestry、远端 immutable archive refs、896 个移出/迁移路径恢复、frozen source/config diff、artifact/path scan、5.68 MB 体积门和 clean worktree 全部通过。
- 92 个旧 sweep/adaptive Selector 配置被识别为后续 archive 候选；其中 15 个仍由 legacy launchers/Selector tests 直接引用。它们连同相关源码/测试应在 G3/G4 依赖闭包中成组移出，避免 G2 只删配置却留下失效入口。

## G3 runtime 审计

- `composition.py` 已能装配 Hybrid Retriever、trained `nli-risk-controlled` Selector 和 `grounded-grc` Generator，并对 Selector checkpoint、GR-C adapter、Granite/TRUE config 执行 SHA-256 校验；这部分应作为冻结算法核心保留。
- 当前 `configs/experiments/systemf_three_module_smoke_seed13.toml` 仍是 development smoke：dataset 指向 `tests/fixtures`，output 指向 `runs/`，不适合作为唯一正式 runtime config。
- 当前 `evidence-rag-smoke` 使用 BM25 + TopK + Extractive baseline，只验证最小三接口，不能证明 final trained Selector 与 grounded GR-C 的组合边界。G3 需要用实际 final class/factory 加可注入离线 doubles 的 CPU smoke。
- `configs/models/three_module_seed13.json` 已冻结 Retriever/Selector/Generator seed-13 revisions 与 checksum，但文件名和 schema 仍需与唯一正式 runtime config 对齐；all-seed 研究 manifest 属 G4/G5，不应让默认 runtime 混用 seed42/73。
- 仓库没有 `src/evidence_rag/api/` 或 serve CLI；`docs/three-module-runtime-handoff.md` 明确将 HTTP adapter 留作未来任务。根据已确认目标结构和前端协作需求，G3 应提供 `/health`、`/v1/query`、冻结 schema、mock response 和模型单次加载服务边界。
- 2026-08-25 可用的 FastAPI/Uvicorn 版本为 0.141.1/0.52.4；Starlette 1.6 的 `TestClient` 已明确要求 `httpx2`，旧 `httpx` fallback 会发弃用警告。因此 API test extra 固定到 `httpx2>=2.12,<2.13`。
- G3 前端接口采用延迟且单次加载：`GET /health` 不触碰模型，第一次 `POST /v1/query` 才装载正式 pipeline，后续请求复用同一实例。公开响应明确区分 Retriever candidates、Selector 保留证据和 Generator citations，前端因此不需要模型权重、HPC 账号或服务器文件权限。
- CPU 三模块运行生成的完整响应验证了公开 contract：10 个 Retriever candidates 中 poison 位于 rank 1，Selector 保留 9 个且排除 poison，Generator 最终只引用被保留的 clean evidence。公开 mock 使用同一 schema 的精简三候选版本，测试会验证计数和 citation subset 不变量。
- 开发依赖锁首次复装没有 broken requirements，但上游已撤回原锁中的 `build==1.5.1`；包索引当前稳定最新版为 1.5.0，因此正式 lock 回退到 `build==1.5.0`，不改变项目的 `build>=1.2,<2` 约束。
- 最终模型一致性复核发现一个 G3 缺口：Selector checkpoint 本身会校验 SHA-256，且 model ID/revision 已冻结，但 NLI base snapshot 的 `config.json` 哈希只存在 model manifest，正式 TOML/factory 尚未验证。Retriever、Granite Generator 和 TRUE verifier 已有同类 config hash gate；Selector 也应补齐同一防漂移边界。
- G3 末轮全树路径扫描仍命中一批 legacy Slurm/研究脚本中的通用 `/user/work/$USER`；它们不是 final runtime/API 依赖，但属于 G4 复现入口收敛的明确清理对象。G3 新增/修改的 runtime、API、config、examples 和 handoff 范围无个人账号或绝对 HPC 路径。

## G4 复现包审计

- G4 启动时 release 仍有约 150 个顶层研究/诊断/Slurm 脚本；其中大量 `full_flow_*`、G3/G5 旧诊断和通用集群 launcher 是开发过程阶段，不应作为正式公开入口并列展示。
- `configs/experiments/` 仍包含约百个 chunk/retriever sweep 配置，正式论文主方法实际只需 Experiment 04 的 10 个冻结系统配置；Selector 正式训练配置是 `configs/selector/lean_v3.toml`，其余 adaptive/beam 配置需要按测试依赖成组审计。
- 当前可公开的小型研究证据已经集中为 5 份 `docs/research/` 报告、Experiment 04 的 8 个 final 聚合文件和 Experiment 05 的 12 个 final 聚合/audit 文件；这些是复现映射的结果端，不应被历史 raw bundle 路径替代。
- 目标结构已确认采用 `experiments/selector`、`experiments/generator`、`experiments/experiment04`、`experiments/experiment05` 四个读者入口；现有已测试脚本可迁移/薄封装到这些入口，算法实现继续复用 `src/evidence_rag`，不为目录美观重写冻结逻辑。
- Exp04 的冻结链是 Goal 1 sealed/scorer validation → Goal 2 ten-arm development wiring → Goal 3 七臂/三 seed formal execution → Goal 4 三项模块消融 → Goal 5 compile/audit；正式公开入口需要表达这条链及可单独运行的 table rebuild，而不是让读者理解内部 Goal 编号。
- Exp05 的冻结链分为 data inventory/index/retrieval、prepare arms、generate、score、compile/bootstrap/claim labels、independent audit；需要保留负面 Claim A/B 结论和 3 datasets × 10 arms 的冻结边界。
- Selector 最终训练实现已经在包内 `src/evidence_rag/cli/run_selector_lean.py`，并由 `configs/selector/lean_v3.toml` 冻结模型、数据角色、训练日程、阈值选择、统计与 Generator blind gate；正式入口应复用它而不是保留旧 adaptive/beam launchers。
- Generator G300/G310/G400/G410 并非四个完全独立脚本：G400 直接依赖旧 B100、G230、G310 和 joint helpers，G410 又依赖 G310/G400。清理前必须递归计算实际 helper 闭包，并区分“方法依赖”与只为历史 Goal 命名/manifest schema 服务的耦合。
- 递归 import closure 显示 Generator 四入口合计只需 8 个现有 script modules：G300、G310、G400、G410、`alce_metrics`、B100、G230、`full_flow_joint`；因此无需保留整个 Full-flow 脚本史。G400/G410 的 seed42/73 输入来自已验证的 G330 external run dirs，G320/G330 在当前树中没有独立脚本。
- Exp04 Goal 3 脚本本身构造正式四基线和三 seed Ours；Goal 4 复用 Goal 3 prepared/Full seed13 并生成三个 ablation；Goal 5 compile 同时依赖 Goal 3/4 helpers。Goal 2 real-model baseline smoke 可作为模型装载预检，但不产生最终表中数值。
- 多个 Exp04 脚本仍把已归档的 `docs/full-flow/...` 目录写成默认 experiment root。正式复现入口必须要求 CLI/environment 指向外部 raw bundle，不能让缺失的 archive-only 默认路径看起来可直接运行。
- Exp04 `final_results.json` 已内含 Table 1/2 canonical rows、paired bootstrap、claim decisions 和生成文件 SHA；公开 `table1/2.json` 只是带 schema 的同一 rows wrapper。因此可以从这个 35KB 小型聚合输入精确重建 JSON/CSV/LaTeX/Markdown，而不需要 11,000 条 raw generations。
- Exp05 `table1.json`、`table2.json`、`bootstrap.json` 和 `claim_labels.json` 已是小型聚合输入；现有 compile CLI 仍强制读取完整 runroot/Goal1 bundle。G4 需要增加只读 table-rebuild 入口，并用 byte-for-byte 测试证明没有 post-hoc 改写 scorer 或 claim labels。
- `docs/research/experiment05.md` 是冻结 compiler 的原始 FINAL_REPORT，可由 table1/table2/claim labels 精确重建；`results/experiment05/final_tables.md` 是后来面向论文的三表排版，额外嵌入 module contribution 与 Selector stress-test 数值，且尾部仍引用已归档相对路径。G4 应先恢复正式 Selector 聚合 JSON，再让论文表从这些公开输入生成。
- 当前正式树缺少目标结构要求的 `results/selector/misleading_evidence_summary.json` 和 `blind_answer_gate.json`，虽然论文版 Experiment 05 表已经引用相应数值。需要从 immutable archive 中定位冻结 source artifacts、验证哈希/结论后以小型聚合形式恢复。
- Immutable archive 中的权威 Selector source 是 R005AB Lean v3 `L003_FINAL_REPORT.json`：evidence gate 为 PASS（seed13/42 harmful reduction 12.9518%/13.1024%，required recall/chain loss 均 0），answer gate 为 FAIL（macro answer delta −0.1353 pp，95% CI −0.5618 至 +0.2770 pp），overall 决策是 KEEP TOPK10。公开结果必须同时保留这两部分，不能只摘正向 evidence gate。
- Archive 里另有 2026-08-10 的 Beam Selector 最终报告，但它是被 Lean v3 后续实验取代的旧路线；正式研究结论可以在限制文档中提及，不应让它与 R005AB final artifacts 并列为当前复现输入。
- R005AB Lean v3 权威 JSON 的 archive byte SHA-256 为 `b031b6f29051fed94a76220ac829ebe2acffe086e253dec9f3b24c090252495c`；两份公开 Selector 聚合文件将共同登记这个 source hash 和 archive path，便于恢复完整逐项 inference。
- 纯渲染模块现已证明：Exp04 的 6 个公开表文件可只由 `final_results.json` 逐字节重建；Exp05 的 CSV、LaTeX 和冻结最终报告可只由公开 `table1.json`、`table2.json`、`claim_labels.json` 重建。正式 release 不需要保留逐题 outputs 或 raw runroot 才能验证论文表格。
- Selector 两份公开摘要由同一个 archive source SHA-256 约束，并同时固定 evidence gate `PASS` 与 answer/overall gate `FAIL`；任何公开说明都不能只写前者或把该模型描述成最终答案指标已有提升。
- `run_selector_lean.py` 已有稳定的 `main(argv)` 与 `fit`、`calibrate`、`final-evaluate` 三个子命令，可直接注册为安装后的公开命令，无需复制训练逻辑。
- Generator G300/G310/G400/G410 同样已有 argparse 入口；正式 `experiments/generator/` 只需解释阶段关系、外部输入与冻结选择，底层仍引用这 8 个已测试模块，避免第二份实现。
- `configs/experiments/` 现有约百个文件几乎全是早期 retriever/chunk sweeps，而 Exp04/05 正式链由各自 sealed manifest/runtime bundle 驱动；这些 sweep configs 不应继续与唯一 runtime config 和最终研究配置并列展示。
- Exp04 冻结 audit 明确登记 11,000 次新生成、1,100 行复用、21 个 paired-bootstrap cells 和所有源/生成哈希；小型 release 只保留 aggregate，但 `research-archive-2026-08-25` 仍可恢复逐题输入用于完全重算。
- Exp05 冻结 audit 明确登记 3 数据集 × 10 arms × 400 = 12,000 generation outputs / query scores、0 scorer errors、10,000 bootstrap resamples（seed 13）以及两个 `NOT SUPPORTED` claim labels；技术 `FINAL PASS` 不能被写成科学 superiority 通过。
- Exp04 的科学结论同样是负向/混合：Ours 的 RAR 未超过基线，Top-10 和 Dense 替换不改变冻结样本 RAR，Direct Generator 在 HotpotQA/MuSiQue 优于 GR-C seed13；公开映射必须保留这些结论，而不仅列系统表格。
- Archive 含 G320 recipe freeze 与 G330 三 seed aggregate manifest；三份逐 seed training manifests 各约 128 KB，主要体积来自选中 case ID 清单，并含原 HPC 运行路径。正式树应发布一个无个人路径的派生 provenance summary（保留 source archive path/SHA、recipe、数据哈希、三 seed adapter 哈希），完整原 manifest 继续由 archive ref 恢复。
- G330 aggregate 固定 adapter weights SHA：seed13 `492d336c…0707`、seed42 `96d8087e…383c`、seed73 `d5f90954…2431`；seed13 与 G3 final runtime manifest 完全一致。
- 实际 AST 闭包把 119 个顶层 Python scripts 收敛到 29 个：Generator 8 个、Exp04 7 个、Exp05 14 个（其中 Exp05 import/freeze helpers 又引用 `g3_baseline_comparison` 与 `verifier_triage`）。其余 90 个是早期诊断/阶段脚本，可连同仅测试这些脚本的历史测试从 release tree 移出。
- 旧脚本引用扫描未发现正式 runtime 或新 `experiments/` 入口依赖这 90 个候选；命中项都是对应的历史 tests/docs contracts。删除批次需要同步移出这些测试，随后做全树 basename/reference scan 与全量 pytest。
- 除规划/清单外，剩余个人/HPC 路径集中在 36 个 legacy Slurm/utility 文件、`build_sealed600.py`、两个 Exp04 Slurm 测试和本次明确断言无路径的 provenance 测试。G4 应移出 legacy launchers，并把保留的两个 Exp04 launchers 改成 workspace/env contract；`build_sealed600.py` 需先查 final Selector 数据链依赖。
- 两个保留的 Exp04 Slurm 已改为 `EVIDENCE_RAG_ROOT`/`EVIDENCE_RAG_VENV`/`MODEL_CACHE_DIR` 环境契约，本体没有个人路径；路径扫描命中它们是 tests 内的通用禁止字面量。其余约 30 个非 Python launcher 都属于旧实验路线，可从 release 外置。
- `build_sealed600.py` 是旧 M0 sealed-600 构建器；Lean v3 配置明确把 sealed600 列为 `unused_unread_roles`，最终 Selector 训练不读取它。源码仍被旧 materializer/relations 测试覆盖，可保留实现但把 docstring 的 `/user/work/$USER` 示例改成通用路径，不把它描述为最终训练链。
- 最终配置闭包为：10 个 Exp04 system configs、`reference_baseline.toml`（仍由 package evaluation test 使用）、`heldout-sample.json`（Exp04 Goal1）、ingestion/runtime/model configs 与唯一 `selector/lean_v3.toml`。其余约百个 retriever/chunk sweep 和 3 个旧 Selector configs 仅由旧 launcher/tests 引用，可从 release 外置。
- 对 90 个待移出脚本的反向文本命中中，`src/evidence_rag/cli/gate0b.py` 与 `relations/training.py` 只在注释中提到早期测量脚本，没有运行时 import；`test_selector_risk.py` 的命中也只是测试名中的 `topk_baseline`。这些 package/test 文件不应因字符串命中误删。
- 正式 Generator helper `full_flow_b100`、`full_flow_g230`、`full_flow_joint` 仍有直接 focused tests，应保留这些测试；旧 Full-flow tests 只在 AST 确认导入已移出脚本时才随脚本外置。
- AST 精确识别出 51 个 tests 直接导入 90 个旧 scripts；另有两个 tests 只服务已淘汰的 adaptive Selector configs。其余文本命中（如 gate0b 测试中的历史注释、Selector risk 测试名）不构成依赖并继续保留。
- 删除批次将由 allowlist 反向生成：保留 29 个 Python research scripts、两个 Exp04 Slurm、最终配置闭包和未导入旧脚本的 tests；每个候选先用 archive tag 做 `git cat-file` 恢复检查，再通过删除后引用扫描验证闭包。
- 265 个外置候选（119 scripts/launchers、95 configs、51 tests）全部可从 `research-archive-2026-08-25` 读取。应用后正式树恰好剩 29 个 Python research scripts、2 个 Exp04 Slurm、18 个配置和 164 个 Python tests。
- 删除后 AST 扫描发现 0 个失效的旧 script imports；Generator/Exp04/Exp05/公开表/Selector focused suite 共 37 项通过。剩余路径扫描命中仅是测试中的禁止字符串本身，已改用拼接断言以便全树审计真正零命中。

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
- G5 本地工作区扫描未发现任何 `model.safetensors` 或 `adapter_model.safetensors` 实体；正式权重仍只在 HPC/外部存储，不能从本地文件重新测量大小。
- Archive 的 Selector base snapshot 记录的是 `cross-encoder/nli-deberta-v3-base` 基础模型文件（737,726,552 bytes，SHA-256 `d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa`），不能把它误写成训练后的 seed-13 Selector checkpoint（737,731,768 bytes，SHA-256 `86622bd9ab6391c9eb560133b01b0cf3744c3706638ff8b0bd38925b84bf72bf`）。
- Archive 中可定位 G330 三个逐 seed training manifest、三 seed aggregate manifest 和 Experiment 04 `goal2_model_config_manifest.json`；后续应只从这些冻结记录提取公开 provenance，避免把原 HPC 路径复制进 release。
- G330 三个正式 GR-C adapters 共享 Granite 4.1 3B revision `c0650403e44e78ec0262dab1c90914c65b196c4e`、15,564,800 个可训练参数、同一 LoRA r=8/alpha=16/七类 target modules 和 PEFT 0.20.0；逐 seed frozen manifests 记录实际权重/config SHA，但没有记录文件字节数。
- 训练脚本以 bfloat16 加载基础模型并调用 PEFT `save_pretrained(..., safe_serialization=True)`；三个 adapter 的张量结构相同，但不能仅凭参数数目猜测实际 safetensors 文件大小。正式清单需把大小来源写明为实际 HPC `stat`，或在未取回文件时明确标为待核验。
- Experiment 04 的三份原始来源已由冻结 Goal 1 manifest 固定：HotpotQA distractor validation 27,452,575 bytes / SHA `c20b638c…f7c6`，MuSiQue full dev 59,422,562 bytes / SHA `8cab31d5…7a4a`，RGB English noise 10,808,210 bytes / SHA `872fc551…5285`。Git 只保留 400/400/300 条的 ordered-ID manifest 和聚合结果，不应再分发 runtime/scorer raw bundles。
- Experiment 05 正式数据范围是 KILT-NQ、KILT-TriviaQA 和 ALCE-ASQA；其 raw runroot、KILT/DPR corpus snapshots、indexes、12,000 generations/query scores 与 claim traces 均已外置，Git 仅保留冻结的 table/bootstrap/claim/audit aggregates。
- Hugging Face 官方 revision API 与 frozen manifest 的全部基础模型 SHA 对齐，并补齐精确权重大小：Granite embedding 298,041,696 bytes；Granite reranker 598,436,708；Granite 4.1 3B 两 shard 共 6,805,714,792；NLI DeBERTa 737,726,552；Provence 1,740,308,732；TRUE 五 shard 共 45,492,657,356；MiniCheck 3,132,786,242。
- 上游模型许可证边界：IBM Granite、NLI DeBERTa 与 TRUE model card 标为 Apache-2.0；MiniCheck 标为 MIT。Provence 的 Hub metadata 标为 CC-BY-NC-ND-4.0，而 README/license 文件内部又混用 CC-BY-NC、CC-BY-NC-SA 与额外限制，正式发布必须采用最严格边界：仅提供上游固定 revision 链接，不镜像权重、不声明商业可用。
- 数据许可证边界：HotpotQA 为 CC-BY-SA-4.0，MuSiQue 为 CC-BY-4.0，RGB 明确为 CC-BY-NC-SA-4.0 non-commercial；KILT 与 ALCE 仓库代码为 MIT，但聚合数据还继承 Wikipedia/NQ/TriviaQA/ASQA 等来源条款，不能把仓库代码许可证当成数据再分发许可。
- BluePebble 登录节点只读检查确认旧 `/scratch` 权重位置是计算节点临时空间，但正式 checkpoint 的独立保留副本仍在共享项目存储。三个 GR-C adapter 权重均为 62,332,992 bytes、config 均为 1,274 bytes；现场 SHA-256 与 G330 frozen manifests 全部一致。Selector 保留副本也再次核验为 737,731,768 bytes 且 hash 一致。
- 三个 Exp04 上游源已进一步固定到不可变 revision：HotpotQA mirror `1908d6af…10ab`、MuSiQue mirror `22873a40…d2df`、RGB repo `65ec39e4…6615`；这些 revision 的远端 bytes/SHA 与冻结 Goal 1 manifest 完全一致。
- 共享保留副本的权限不完全一致：seed13 adapter 与 Selector 为 owner/group-readable，seed42/73 仅 owner-readable。这个内部团队访问状态不等于公共发布；G5 不在未确认授权时扩大权限或上传权重。

## G6 公开文档审计

- 根 `README.md` 只有数行，无法支持安装、CPU smoke、架构、复现、结果边界和引用；必须以最终三模块方法重写。
- 旧 `docs/README.md` 仍把 Query2Doc、TopK 和 Corroboration 写成主线，并引用已外置的开发材料；它应改为英文公开文档导航，而不是继续承担历史 tracker 职责。
- 正式树已有 runtime handoff、frontend integration、models/data 和研究结果文档，但缺少公开的 setup、architecture、reproduction、results、limitations 以及 Selector/Generator model cards。
- G6 开始时正式树尚无 `LICENSE`、`CITATION.cff`、`CHANGELOG.md`、`CONTRIBUTING.md` 和 `AUTHORS.md`；这些文件现已建立。许可证没有自动选择，用户于 2026-08-25 明确确认采用 MIT。
- Git 历史可作为贡献事实来源，但不能据此推断团队成员角色、单位或论文作者顺序；公开作者页只应记录 commit identity 或使用团队实体名称。
- 本地没有 PyYAML；`CITATION.cff` 可采用兼容 YAML 1.2 的 JSON 语法，并用标准库 `json` 做解析契约，避免为元数据验证引入运行依赖。
- CFF 1.2.0 官方 JSON Schema 已复核：根字段必须包含 authors、cff-version、message 和 title；entity author 只要求 name，因此可合法使用不推断个人顺序的 `Evidence RAG project contributors` 团队实体。
- 公开结果说明必须直接依据冻结聚合 JSON：misleading-evidence evidence gate PASS、blind answer gate FAIL、ordinary Experiment 05 Selector 大多未触发、Exp04 superiority 与 Exp05 Claim A/B 均未获支持。
- README 的系统图采用无主题指令、无 inline style、含 `accTitle`/`accDescr` 的 Mermaid flowchart，并控制在十个节点内。
- 当前离线 `evidence-rag-smoke` 已实际运行成功：使用最终模块类和确定性 CPU doubles，候选 10 条，NLI Selector 删除 poison evidence，Generator 返回并只引用 clean evidence；公开 quick start 可以据此写出可验证预期。
- 归档 commit history 显示 9 个 contributor identities；公开 AUTHORS 可按 commit identity 列出，但不附邮箱、不猜角色、不推断论文作者顺序。
- G6 当前所有变更都局限在公开文档、package metadata、文档契约测试和三份内部规划记录；没有模型、结果 JSON、runtime 算法或归档引用被修改。
- 本机没有预装 `mmdc`，但有 Node 20/npm；Mermaid 实际渲染可用临时 `npx @mermaid-js/mermaid-cli` 验证，不需要把生成 SVG/PNG 提交进仓库。
- npm 当前 Mermaid CLI 为 11.16.0；该版本已分别解析并实际渲染 README 与 architecture 中的两张图，证明 `accTitle`、`accDescr`、节点、边和 classDef 语法可执行。
- G6 新增的 14 个唯一外部文档 URL（仓库、issues、docs、5 个数据源和 5 个固定模型 revision）均从当前无 HPC 依赖环境返回 HTTP 200。
- GitHub 当前状态：仓库仍为 public、默认分支为 main，当前登录者权限是 WRITE 而非 ADMIN；仓库 owner 是 `M1yanoShiho`。正式代码许可证应由仓库 owner/团队确认，当前登录权限本身不证明可代表全部贡献者授权。
- 全 Git 历史没有既有 LICENSE/COPYING/NOTICE 提交；正式源码范围也没有 SPDX/copyright/license header 可继承。因此不存在一个可自动沿用的项目代码许可证。
- 首次 wheel 内容审计发现 Hatchling 的默认 license-file glob 会把 `AUTHORS.md` 与 `LICENSE` 一起放入 `dist-info/licenses/`。显式设置 `license-files = ["LICENSE"]` 后，新构建 wheel 只含 LICENSE 且 METADATA 只有一条 `License-File: LICENSE`。
- 扩大链接审计到全部 30 份读者可见 Markdown 后，发现 16 个 G2/G4 迁移遗留的旧相对链接；根因是先前契约只扫描核心 G6 文档，没有覆盖 `docs/research/` 和 `results/`。这些链接现已指向 canonical aggregate/report，archive-only 原始 manifest 则改为明确 archive ref。
- G7 第一次从远端 clone `20c9717` 后发现 README 末尾仍保留许可证 pending 句子；`LICENSE` 和 wheel metadata 已是 MIT，因此这是发布说明一致性缺口，不是许可证文件或代码错误。

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
| 代码许可证 | 用户已明确选择 MIT；G6 使用集体主体 `Evidence RAG project contributors` | G6 resolved |
| 前端是否属于同仓库交付范围 | 决定是否必须实现 HTTP API | G3 |
| 模型 adapters 是否允许公开再分发 | 阻塞公众真实 runtime 下载 | G5 |
| raw outputs 是否允许按数据许可归档 | 决定 DOI archive 内容 | G5 |
| 仓库所有者/用户权限范围 | 可能阻塞 branch/tag/main/Release 操作 | G1/G8 |
| 文档 ingestion 是否属于最终论文交付主张 | 决定 `loaders/` 和 `cli/ingest.py` 的最小保留范围 | G3 |

## 规划验证发现

- P0/P1/G1–G6 complete；MIT、公开文档、package metadata 和完整回归均已通过；G7–G8 pending。
- G1–G8 均包含目的、前置条件、允许动作、产物、验收门和禁止事项。
- P0 初版缺少部分同名结构标题，已在自检后补齐。
- cleanup inventory 已链接到 `task_plan.md`，执行状态不会在两份文件中重复维护。
- 四份规划/清单文件未包含个人账号或 HPC 绝对路径。
- 正式结构预览将默认 `main` 定义为运行代码、冻结配置、复现实验、测试、公开文档、最终聚合结果和示例七层。
- 结构预览包含建议新增的最小 HTTP API；这是后续 G3 的实现/验证项，不是对当前仓库已有能力的陈述。
- 当前实际 loader 和部分 Generator/Selector 辅助文件名已对照预览修正；最终删除边界仍须由 G3/G4 的依赖闭包和测试决定。
- G1 必须先在当前开发工作区冻结尚未提交的 Experiment 04/05；G2 起采用隔离 worktree，避免正式树整理污染归档现场。
- G1 的远端 archive branch 和 annotated tag 均解析到 `ea4d617753aff868fff8f846964f7cfb050414bb`；tag 已验证可直接读取 Exp05 final audit、scorer 源码和 `ARCHIVE_INDEX.md`。

## 资源

- `docs/PUBLIC_RELEASE_CLEANUP_MANIFEST.md`
- `docs/PROPOSED_PUBLIC_REPOSITORY_STRUCTURE.md`
- `docs/three-module-runtime-handoff.md`
- `configs/models/final_seed13.json`
- `configs/runtime/final_seed13.toml`
- `ARTIFACT_MANIFEST.json`
- `REPRODUCIBILITY_MAP.md`
- Experiment 04 final report and result tables
- Experiment 05 findings, final report, tables, and integrity audit

## 视觉/浏览器发现

- 本次计划编写未新增视觉或浏览器发现。

---
*执行每个 Goal 时更新事实、决策和未决问题。*
