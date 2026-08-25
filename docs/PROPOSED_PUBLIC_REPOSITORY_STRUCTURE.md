# 正式发布版仓库结构预览

> 状态：执行前设计稿。本文描述整理完成后默认 `main` 应呈现的样子；它不代表这些文件已经移动、重命名或提交。

## 1. 整理完成后的整体形态

同一个 GitHub 仓库承担两种职责，但由不同引用隔开：

```text
archive/full-research-history-2026-08-25
└── 保存完整开发现场、旧版本、过程文档、原始实验结果和协作历史

release/dissertation-v1
└── 从归档快照整理出的正式候选版本，用于逐步验收

main  ←  release/dissertation-v1 验收后接入
└── GitHub 默认首页：干净、可安装、可运行、可复现、可引用

research-archive-2026-08-25
└── 指向完整研究快照的不可变标签

v1.0.0-dissertation
└── 指向最终正式版本的发布标签
```

因此，整理不是把旧材料销毁，而是让默认首页只展示论文最终系统。需要查开发过程时，仍能切换到归档分支或标签。

## 2. 默认 `main` 的目标目录树

下面是建议的最终形态。它以“一个陌生人克隆后能理解并运行”为标准，而不是照搬当前开发目录。

```text
IBM_Granite_Project_latest/
├── README.md
├── LICENSE
├── CITATION.cff
├── AUTHORS.md
├── CHANGELOG.md
├── REPRODUCIBILITY_MAP.md
├── ARTIFACT_MANIFEST.json
├── pyproject.toml
├── requirements-dev.lock
├── .python-version
├── .env.example
├── .gitignore
├── .gitattributes
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
│
├── src/evidence_rag/
│   ├── __init__.py
│   ├── composition.py
│   ├── query_analysis.py
│   │
│   ├── contracts/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── protocols.py
│   │   └── validation.py
│   │
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── bm25.py
│   │   ├── strong_bm25.py
│   │   ├── granite.py
│   │   ├── hybrid.py
│   │   ├── fusion.py
│   │   ├── chunking.py
│   │   ├── indexing.py
│   │   └── rerank.py
│   │
│   ├── selector/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── answer_norm.py
│   │   ├── guidance.py
│   │   ├── dual_head.py
│   │   ├── nli_dual_head.py
│   │   ├── nli_runtime.py
│   │   ├── risk_controlled.py
│   │   ├── top_k.py
│   │   ├── threshold_only.py
│   │   └── provence.py
│   │
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── granite.py
│   │   ├── nli.py
│   │   ├── draft.py
│   │   ├── claim_splitter.py
│   │   ├── attribution.py
│   │   ├── completeness.py
│   │   ├── verify_annotate.py
│   │   ├── verifier.py
│   │   ├── verified.py
│   │   ├── evidence_recheck.py
│   │   ├── repair.py
│   │   ├── entity_check.py
│   │   ├── key_facts.py
│   │   ├── json_parsing.py
│   │   ├── trace.py
│   │   └── extractive.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── service.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── service.py
│   │   └── app.py
│   │
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── corpus.py
│   │   ├── datasets.py
│   │   ├── benchmarks.py
│   │   └── artifacts.py
│   │
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── dispatch.py
│   │   ├── cache.py
│   │   ├── text_loader.py
│   │   ├── image_loader.py
│   │   ├── pdf_loader.py
│   │   └── office_loader.py
│   │
│   ├── materializer/
│   │   ├── __init__.py
│   │   ├── provenance.py
│   │   ├── source_parent.py
│   │   ├── selector_pool.py
│   │   └── selector_labels.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── scoring.py
│   │   ├── paired_metric.py
│   │   ├── citation_metrics.py
│   │   ├── system_scorer.py
│   │   ├── harm.py
│   │   ├── selector_artifacts.py
│   │   ├── selector_lean.py
│   │   ├── selector_sanity.py
│   │   ├── sealed_runtime.py
│   │   ├── experiment04_goal3.py
│   │   ├── experiment04_goal4.py
│   │   ├── experiment04_runner.py
│   │   ├── experiment05_data.py
│   │   ├── experiment05_index.py
│   │   ├── experiment05_io.py
│   │   ├── experiment05_retrieval.py
│   │   ├── experiment05_generation.py
│   │   ├── experiment05_runtime.py
│   │   └── experiment05_scorer.py
│   │
│   └── cli/
│       ├── __init__.py
│       ├── smoke.py
│       ├── serve.py
│       ├── ingest.py
│       ├── experiment.py
│       ├── build_selector_components.py
│       ├── build_selector_labels.py
│       └── run_selector_lean.py
│
├── configs/
│   ├── models/
│   │   └── three_module_all_seeds.json
│   ├── runtime/
│   │   ├── final_seed13.toml
│   │   └── cpu_smoke.toml
│   ├── training/
│   │   ├── selector_lean_v3.toml
│   │   └── generator_grc.toml
│   └── evaluation/
│       ├── experiment04/
│       │   ├── ours_seed13.toml
│       │   ├── ours_seed42.toml
│       │   ├── ours_seed73.toml
│       │   ├── hybrid_rag.toml
│       │   ├── dense_rag.toml
│       │   ├── granite_rerank_rag.toml
│       │   ├── provence_rag.toml
│       │   ├── ablation_dense_retriever.toml
│       │   ├── ablation_top10.toml
│       │   └── ablation_direct_generator.toml
│       └── experiment05/
│           ├── ours_seed13.toml
│           ├── ours_seed42.toml
│           ├── ours_seed73.toml
│           ├── bm25_rag.toml
│           ├── hybrid_rag.toml
│           ├── granite_rerank_rag.toml
│           ├── provence_rag.toml
│           ├── ablation_bm25_retriever.toml
│           ├── ablation_no_selector.toml
│           └── ablation_direct_generator.toml
│
├── experiments/
│   ├── README.md
│   ├── selector/
│   │   ├── README.md
│   │   ├── train.py
│   │   └── evaluate_misleading_evidence.py
│   ├── generator/
│   │   ├── README.md
│   │   ├── train_grc.py
│   │   ├── screen_seed.py
│   │   ├── qualify_niah.py
│   │   └── qualify_cross_data.py
│   ├── experiment04/
│   │   ├── README.md
│   │   ├── prepare.py
│   │   ├── run_baselines.py
│   │   ├── run_main.py
│   │   ├── run_ablations.py
│   │   ├── build_tables.py
│   │   └── hpc/run.slurm
│   └── experiment05/
│       ├── README.md
│       ├── prepare_data.py
│       ├── build_index.py
│       ├── retrieve.py
│       ├── generate.py
│       ├── score.py
│       ├── audit.py
│       ├── build_tables.py
│       └── hpc/run.slurm
│
├── tests/
│   ├── contracts/
│   ├── retriever/
│   ├── selector/
│   ├── generator/
│   ├── pipeline/
│   ├── api/
│   ├── evaluation/
│   ├── architecture/
│   ├── cli/
│   ├── fixtures/
│   │   └── three_module_smoke_dataset/
│   └── typecheck.py
│
├── docs/
│   ├── README.md
│   ├── architecture.md
│   ├── setup.md
│   ├── frontend-integration.md
│   ├── reproduction.md
│   ├── models-and-data.md
│   ├── results.md
│   ├── limitations.md
│   ├── models/
│   │   ├── selector.md
│   │   └── generator.md
│   └── research/
│       ├── selector-findings.md
│       ├── experiment04.md
│       └── experiment05.md
│
├── results/
│   ├── README.md
│   ├── selector/
│   │   ├── misleading_evidence_summary.json
│   │   └── blind_answer_gate.json
│   ├── experiment04/
│   │   ├── final_tables.md
│   │   ├── summary_metrics.csv
│   │   ├── final_results.json
│   │   ├── final_audit.json
│   │   └── tables.tex
│   └── experiment05/
│       ├── final_report.md
│       ├── table1.csv
│       ├── table1.json
│       ├── table2.csv
│       ├── table2.json
│       ├── claim_labels.json
│       ├── bootstrap.json
│       ├── final_audit.json
│       └── tables.tex
│
└── examples/
    ├── quickstart.py
    ├── api_request.py
    └── mock_frontend_response.json
```

## 3. 根目录文件分别做什么

| 文件 | 用途 | 为什么正式版需要 |
|---|---|---|
| `README.md` | 项目首页；一句话问题定义、三模块架构、安装、快速运行、主要结果、模型下载和文档入口 | 评审者和新用户首先看到的文件 |
| `LICENSE` | 明确代码可如何使用、修改和再发布 | 没有许可证的公开代码默认并不等于可自由使用；具体许可证须由团队确认 |
| `CITATION.cff` | 项目名称、作者、版本、年份和推荐引用方式 | GitHub 可以直接显示 “Cite this repository” |
| `AUTHORS.md` | 列出成员及 Retriever、Selector、Generator、Frontend、Evaluation 等贡献 | 避免贡献归属含糊 |
| `CHANGELOG.md` | 记录 `v1.0.0-dissertation` 的正式内容及后续版本变化 | 让发布版本和开发历史分开表达 |
| `REPRODUCIBILITY_MAP.md` | 把论文主张/表格映射到配置、命令、结果文件和代码入口 | 答辩和复核时可以快速追踪证据 |
| `ARTIFACT_MANIFEST.json` | 记录模型、数据和索引的名称、版本、外部地址、大小、许可证、SHA-256 及预期本地位置 | Git 不保存大文件，但仍能自动下载和校验正确资产 |
| `pyproject.toml` | Python 包、依赖、可选 API/HPC 依赖、测试/格式/类型配置以及命令行入口 | 作为唯一权威安装配置，替代散乱环境说明 |
| `requirements-dev.lock` | 锁定测试和开发环境的精确依赖版本 | 提高毕业提交后的可复现性 |
| `.python-version` | 声明支持并验证过的 Python 版本 | 减少环境不一致 |
| `.env.example` | 只给出变量名称和示例占位符，如模型根目录和缓存目录 | 说明运行配置，但绝不保存密码、令牌或个人路径 |
| `.gitignore` | 排除模型、缓存、索引、原始数据、日志、临时输出和本地环境 | 防止再次把 HPC/本机产物误提交 |
| `.gitattributes` | 统一文本换行；对结果表和必要二进制样例声明 Git 行为 | 减少跨平台差异，也可阻止不合适文件进入普通 Git |

## 4. 自动检查文件

| 文件 | 用途 |
|---|---|
| `.github/workflows/ci.yml` | 每次提交/PR 安装最小环境，运行格式、类型、单元、架构和 CPU smoke 测试 |
| `.github/workflows/release.yml` | 对正式标签重复 clean-install 验证，生成发布包和校验摘要；若不做自动发布，可在 G8 省略 |

## 5. `src/evidence_rag`：正式系统代码

### 5.1 包入口和统一组装

| 文件 | 用途 |
|---|---|
| `__init__.py` | 暴露公开版本号和稳定的顶层接口，不暴露内部实验细节 |
| `composition.py` | 按配置组装 Retriever → Selector → Generator；所有 CLI、API 和评测共用同一个组装入口 |
| `query_analysis.py` | 提取查询特征或类型，供检索/选择策略使用；只保留最终管线实际依赖的行为 |

### 5.2 `contracts`：三个模块之间的数据协议

| 文件 | 用途 |
|---|---|
| `contracts/__init__.py` | 导出稳定的数据类型和模块接口 |
| `contracts/models.py` | 定义 Query、Document、Chunk、RetrievedCandidate、SelectedEvidence、Answer、Trace 等结构 |
| `contracts/protocols.py` | 定义 Retriever、Selector、Generator 必须实现的方法，避免模块互相依赖具体实现 |
| `contracts/validation.py` | 在模块交界检查 ID、分数、引用、来源和空输入等不变量 |

### 5.3 `retriever`：候选证据检索

| 文件 | 用途 |
|---|---|
| `retriever/__init__.py` | 只导出正式 Retriever 接口和实现 |
| `retriever/bm25.py` | 标准 BM25 稀疏检索基线 |
| `retriever/strong_bm25.py` | 最终系统使用的 Strong BM25 配置和稳定实现 |
| `retriever/granite.py` | Granite embedding 稠密检索器及其模型加载逻辑 |
| `retriever/hybrid.py` | 同时调用 Strong BM25 和 Granite dense，并产生最终候选池 |
| `retriever/fusion.py` | 用 RRF 等冻结规则融合不同检索排名 |
| `retriever/chunking.py` | 把文档切成带稳定 ID 和来源信息的段落 |
| `retriever/indexing.py` | 构建、保存、加载稀疏/稠密索引和版本元数据 |
| `retriever/rerank.py` | Granite rerank 基线；不是最终 Selector，但必须用于论文对照实验 |

### 5.4 `selector`：经过训练的证据选择器

| 文件 | 用途 |
|---|---|
| `selector/__init__.py` | 导出训练后 NLI Selector 和论文基线 |
| `selector/models.py` | 定义 Selector 输入、逐证据分数、决策和诊断信息 |
| `selector/answer_norm.py` | 统一答案文本规范化，避免训练/评测的字符串差异影响标签或 gate |
| `selector/guidance.py` | 为选择决策准备最终管线仍依赖的指导信号；若依赖审计证明仅属旧实验则移到 archive |
| `selector/dual_head.py` | 双头 Selector 的通用网络/推理接口 |
| `selector/nli_dual_head.py` | NLI 风险与有用性双头模型结构，和训练 checkpoint 对齐 |
| `selector/nli_runtime.py` | 从外部 manifest 找到 checkpoint，加载 tokenizer/权重并执行批量推理 |
| `selector/risk_controlled.py` | 最终风险控制选择策略：依据训练分数、阈值和预算选证据 |
| `selector/top_k.py` | 固定返回检索排名前 K 条的无训练基线 |
| `selector/threshold_only.py` | 只使用阈值、不使用完整风险策略的消融 |
| `selector/provence.py` | Provence 选择基线的适配层 |

这里会明确：Top-K 是基线；论文最终 Selector 是训练后的 NLI risk-controlled Selector。两者不能在 README 或 API 中混为一谈。

### 5.5 `generator`：GR-C 生成和验证

| 文件 | 用途 |
|---|---|
| `generator/__init__.py` | 导出正式 GR-C Generator 和 Direct Granite 基线 |
| `generator/models.py` | 定义草稿、claim、验证结果、引用和最终回答结构 |
| `generator/granite.py` | 加载 Granite 基座和外部 GR-C adapter，或运行 Direct Granite 基线 |
| `generator/nli.py` | 生成阶段的 NLI 支持度检查 |
| `generator/draft.py` | 根据选择后的证据产生初稿 |
| `generator/claim_splitter.py` | 把回答拆成可独立验证的 claim |
| `generator/attribution.py` | 把 claim 与证据/引用建立可追踪对应关系 |
| `generator/completeness.py` | 检查回答是否遗漏所选证据中的关键支持事实 |
| `generator/verify_annotate.py` | 核验 claim 并把支持证据/风险标注回回答 |
| `generator/verifier.py` | 组合 NLI、实体和引用检查，给出统一验证决策 |
| `generator/verified.py` | 表示已通过或带保留意见的验证后回答结构/流程 |
| `generator/evidence_recheck.py` | 对低置信或冲突 claim 重新检查候选证据 |
| `generator/repair.py` | 对未获支持的草稿执行受约束修复，而非自由重写 |
| `generator/entity_check.py` | 检查实体是否能在证据中得到支持，降低错配 |
| `generator/key_facts.py` | 提取生成与验证阶段需要保持的关键事实 |
| `generator/json_parsing.py` | 稳健解析模型结构化输出并处理失败情况 |
| `generator/trace.py` | 记录可公开调试的生成步骤和引用来源，不记录敏感原始环境信息 |
| `generator/extractive.py` | 无生成模型时的抽取式 fallback/smoke 实现；不能伪装成主系统结果 |

### 5.6 `pipeline`：端到端系统入口

| 文件 | 用途 |
|---|---|
| `pipeline/__init__.py` | 导出统一的 `EvidenceRAGPipeline` |
| `pipeline/service.py` | 接受 query/corpus，依次运行三模块，返回 answer、citations、selected evidence 和 trace |

### 5.7 `api`：给前端使用的最小后端接口

| 文件 | 用途 |
|---|---|
| `api/__init__.py` | 标记 API 包并暴露应用工厂 |
| `api/schemas.py` | 固定前端请求/响应格式，例如 query、session、answer、citations、module diagnostics |
| `api/service.py` | 管理模型只加载一次、请求调用 pipeline、错误转换和健康状态 |
| `api/app.py` | 提供 `/health`、`/v1/query` 等 HTTP 端点；前端不直接读取 HPC 文件或加载模型 |

这是推荐纳入正式版的新边界，因为现有仓库只有 Python pipeline，没有稳定 HTTP API。前端代码可以在另一个仓库，但它只需要知道 API 契约，不应获得任何人的 HPC 账号。

### 5.8 `infrastructure`：环境与外部资产

| 文件 | 用途 |
|---|---|
| `infrastructure/__init__.py` | 导出基础设施公共接口 |
| `infrastructure/config.py` | 读取 TOML/环境变量，解析相对路径，禁止把个人绝对路径写进代码 |
| `infrastructure/corpus.py` | 加载正式 corpus 描述和文档 ID 映射 |
| `infrastructure/datasets.py` | 下载/读取论文允许公开的数据集及固定 split |
| `infrastructure/benchmarks.py` | 登记 benchmark 名称、split 和加载规则，避免实验脚本各自硬编码 |
| `infrastructure/artifacts.py` | 按 manifest 下载、定位并做 SHA-256 校验；缺失时给出可操作错误 |

### 5.9 `loaders`：文档接入

| 文件 | 用途 |
|---|---|
| `loaders/__init__.py` | 暴露正式支持的文档加载入口 |
| `loaders/dispatch.py` | 按文件类型选择加载器，并统一不支持格式的错误处理 |
| `loaders/cache.py` | 缓存解析后的本地文档，同时记录源文件版本 |
| `loaders/text_loader.py` | 读取纯文本和 Markdown |
| `loaders/image_loader.py` | 读取项目最终确认支持的图片/OCR 输入；若无系统主张则归档 |
| `loaders/pdf_loader.py` | 读取 PDF，同时保留页码供引用 |
| `loaders/office_loader.py` | 读取正式支持的 DOCX/PPTX 等 Office 文档 |

若最终论文只评测固定 benchmark、没有“用户上传文档”这一系统主张，G3 可以进一步把非必要 loader 移到归档；反之这些文件应保留。

### 5.10 `materializer`：训练/评测数据的可追踪构造

| 文件 | 用途 |
|---|---|
| `materializer/__init__.py` | 导出数据物化入口 |
| `materializer/provenance.py` | 给每条样本记录来源、版本和转换步骤 |
| `materializer/source_parent.py` | 保持原文档、父段落和切片之间的关系 |
| `materializer/selector_pool.py` | 构建 Selector 的候选证据池 |
| `materializer/selector_labels.py` | 按冻结规则生成/读取 Selector 监督标签 |

这些不是在线请求必需代码，但如果训练复现仍直接依赖它们，就必须和训练入口一起保留。

### 5.11 `evaluation`：论文评分与审计实现

| 文件 | 用途 |
|---|---|
| `evaluation/__init__.py` | 导出正式评分接口 |
| `evaluation/metrics.py` | 通用指标定义和聚合 |
| `evaluation/scoring.py` | 统一逐样本评分和结果 schema |
| `evaluation/paired_metric.py` | 配对比较、差值和置信区间所需计算 |
| `evaluation/citation_metrics.py` | 引用正确性、覆盖率和相关指标 |
| `evaluation/system_scorer.py` | 把 RAR/答案质量/引用/模块诊断合并为系统级结果 |
| `evaluation/harm.py` | misleading-evidence 等压力测试中的 harm 计算 |
| `evaluation/selector_artifacts.py` | 校验 Selector checkpoint、manifest 和评测产物 |
| `evaluation/selector_lean.py` | Selector Lean v3 训练/评测共享逻辑 |
| `evaluation/selector_sanity.py` | 标签、分数、阈值和样本泄漏的 sanity checks |
| `evaluation/sealed_runtime.py` | 保证最终评测只能使用冻结配置和已登记模型 |
| `evaluation/experiment04_goal3.py` | Experiment 04 主三模块运行与统计逻辑 |
| `evaluation/experiment04_goal4.py` | Experiment 04 基线/消融对比逻辑 |
| `evaluation/experiment04_runner.py` | Experiment 04 统一运行器和恢复机制 |
| `evaluation/experiment05_data.py` | Experiment 05 固定数据 schema 和 split |
| `evaluation/experiment05_index.py` | Experiment 05 索引构建与校验 |
| `evaluation/experiment05_io.py` | Experiment 05 产物命名、读取和原子写入协议 |
| `evaluation/experiment05_retrieval.py` | Experiment 05 各检索系统运行逻辑 |
| `evaluation/experiment05_generation.py` | Experiment 05 生成运行逻辑 |
| `evaluation/experiment05_runtime.py` | Experiment 05 配置、断点续跑和系统编排 |
| `evaluation/experiment05_scorer.py` | Experiment 05 最终打分、bootstrap 和 claim 判定 |

这里保留的是论文最终实验真正调用的共享实现。过时 scorer、一次性 debug 脚本和未进入论文结论的实验版本不出现在正式树中。

### 5.12 `cli`：用户可执行命令

| 文件 | 用途 |
|---|---|
| `cli/__init__.py` | 命令行包入口 |
| `cli/smoke.py` | 使用小型 fixture 在 CPU 上验证三模块接口和输出结构 |
| `cli/serve.py` | 启动给前端使用的 HTTP 服务 |
| `cli/ingest.py` | 把本地文档转成 corpus 和索引；仅在保留文档接入能力时存在 |
| `cli/experiment.py` | 读取冻结配置并调用正式实验运行器 |
| `cli/build_selector_components.py` | 构造 Selector 训练所需中间组件 |
| `cli/build_selector_labels.py` | 生成并验证 Selector 标签 |
| `cli/run_selector_lean.py` | 训练/评测最终 Selector Lean v3 |

## 6. `configs`：方法和实验的机器可读定义

### 6.1 模型与运行配置

| 文件 | 用途 |
|---|---|
| `models/three_module_all_seeds.json` | 登记 seed13/42/73 Selector、Generator adapter 和基础模型；只写逻辑 ID、相对缓存位置与 manifest 键 |
| `runtime/final_seed13.toml` | README 默认演示的完整系统配置：Hybrid Retriever + seed13 trained Selector + seed13 GR-C Generator |
| `runtime/cpu_smoke.toml` | 小模型或 mock 组件的离线 CPU 健康检查配置，不用于论文主结果 |
| `training/selector_lean_v3.toml` | 最终 Selector 数据、结构、seed、优化器、阈值和选择准则 |
| `training/generator_grc.toml` | 最终 GR-C adapter 训练超参数和数据引用 |

### 6.2 Experiment 04 配置

`evaluation/experiment04/` 中每个 TOML 对应一个冻结实验臂：

| 文件 | 系统/消融 |
|---|---|
| `ours_seed13.toml` | 完整三模块系统，seed13 |
| `ours_seed42.toml` | 完整三模块系统，seed42 |
| `ours_seed73.toml` | 完整三模块系统，seed73 |
| `hybrid_rag.toml` | Hybrid retrieval + 无训练选择器的基线 |
| `dense_rag.toml` | Dense-only 检索基线 |
| `granite_rerank_rag.toml` | Granite rerank 基线 |
| `provence_rag.toml` | Provence Selector 基线 |
| `ablation_dense_retriever.toml` | 去掉 BM25/融合的 Retriever 消融 |
| `ablation_top10.toml` | 用固定 Top-10 替代训练 Selector |
| `ablation_direct_generator.toml` | 用 Direct Granite 替代 GR-C Generator |

### 6.3 Experiment 05 配置

`evaluation/experiment05/` 同样是一配置一实验臂：三个完整系统 seed，四个外部/传统系统，以及三个模块消融。这样运行记录不会靠脚本参数猜测系统身份。

## 7. `experiments`：复现入口，不是另一份核心实现

原则是：这里的脚本只负责准备输入、调用 `src/evidence_rag`、保存产物，不复制模型算法。

| 文件 | 用途 |
|---|---|
| `experiments/README.md` | 全部正式实验的顺序、资源需求、预计产物和论文对应关系 |
| `selector/README.md` | Selector 数据、训练、选 seed 和压力测试说明 |
| `selector/train.py` | 调用冻结 Selector Lean v3 训练入口 |
| `selector/evaluate_misleading_evidence.py` | 复现 Selector 在误导证据压力测试中的条件性收益 |
| `generator/README.md` | GR-C 训练、seed screening 和 qualification 说明 |
| `generator/train_grc.py` | 训练最终 Generator adapter |
| `generator/screen_seed.py` | 用冻结准则筛选 seed，不接触最终测试结论 |
| `generator/qualify_niah.py` | 运行 NIAH/能力资格检查 |
| `generator/qualify_cross_data.py` | 跨数据资格检查 |
| `experiment04/README.md` | Exp04 的目标、10 个臂、资源和最终输出 |
| `experiment04/prepare.py` | 校验数据、模型和冻结配置 |
| `experiment04/run_baselines.py` | 运行对照系统 |
| `experiment04/run_main.py` | 运行三个 seed 的完整系统 |
| `experiment04/run_ablations.py` | 运行三个模块消融 |
| `experiment04/build_tables.py` | 从机器可读结果重建论文表格 |
| `experiment04/hpc/run.slurm` | 不带个人账号/绝对路径的 HPC 作业模板 |
| `experiment05/README.md` | Exp05 的数据、10 臂、恢复顺序和结论边界 |
| `experiment05/prepare_data.py` | 固定数据和 split，生成 provenance |
| `experiment05/build_index.py` | 构建或验证各检索索引 |
| `experiment05/retrieve.py` | 运行全部检索臂 |
| `experiment05/generate.py` | 基于冻结检索输出运行 Selector/Generator |
| `experiment05/score.py` | 运行最终 scorer 和 bootstrap |
| `experiment05/audit.py` | 检查 completeness、泄漏、manifest 和结果一致性 |
| `experiment05/build_tables.py` | 重建 Table 1、Table 2 和 LaTeX 表 |
| `experiment05/hpc/run.slurm` | 可由其他学校账号填写环境变量后使用的 HPC 模板 |

这些较整齐的文件名可以由现有测试过的脚本加薄包装器形成，不会为了“好看”重写已经冻结的算法。

## 8. `tests`：证明正式树不是展示品

| 路径 | 用途 |
|---|---|
| `tests/contracts/` | 检查三模块输入输出协议和边界条件 |
| `tests/retriever/` | BM25、dense、hybrid、RRF、索引和 rerank 测试 |
| `tests/selector/` | 训练 Selector、Top-K/threshold/Provence 基线、checkpoint 加载和选择决策测试 |
| `tests/generator/` | draft、claim split、NLI 核验、实体、引用和 adapter 加载测试 |
| `tests/pipeline/` | 完整三模块顺序、失败恢复和 trace 测试 |
| `tests/api/` | `/health`、`/v1/query`、schema 和模型单次加载测试 |
| `tests/evaluation/` | 指标、Experiment 04/05、bootstrap、审计和结果 schema 测试 |
| `tests/architecture/` | 禁止循环依赖、个人绝对路径和实验脚本反向侵入核心包 |
| `tests/cli/` | 安装后的命令入口和错误信息测试 |
| `tests/fixtures/three_module_smoke_dataset/` | 小型、可公开、非论文测试集的离线样例 |
| `tests/typecheck.py` | 对核心公共接口执行静态类型检查 |

最终不会照搬当前所有历史测试文件；G2–G4 会按保留代码的依赖闭包挑选并重组。任何移出 main 的旧测试仍在 archive 分支。

## 9. `docs`：给四类读者的正式说明

| 文件 | 用途 |
|---|---|
| `docs/README.md` | 文档导航，告诉读者按“使用、前端、复现、审阅”选择入口 |
| `architecture.md` | 三模块数据流、模块协议、模型边界和部署结构 |
| `setup.md` | 本地 CPU smoke、GPU 推理、HPC 评测三套安装方式 |
| `frontend-integration.md` | API 请求/响应、错误码、启动方式、模型资产位置和前端联调示例 |
| `reproduction.md` | 从环境、模型、数据到 Exp04/05 表格的分层复现步骤 |
| `models-and-data.md` | 所有外部资产的来源、用途、大小、许可证、下载和校验方式 |
| `results.md` | 只呈现最终冻结结果，并明确什么是技术通过、什么是科学支持 |
| `limitations.md` | Generator 弱点、Selector 生效条件、普通数据上接近不触发等限制 |
| `models/selector.md` | Selector model card：训练目标、数据、seed、适用范围和误用风险 |
| `models/generator.md` | GR-C adapter model card：基座、训练、qualification、弱点和许可证 |
| `research/selector-findings.md` | Selector 误导证据与 blind answer gate 的完整研究结论 |
| `research/experiment04.md` | Exp04 设计、系统臂、统计和结论边界 |
| `research/experiment05.md` | Exp05 设计、Claim A/B 未获支持及系统级解释 |

## 10. `results`：只保留最终、小型、可审计结果

| 文件 | 用途 |
|---|---|
| `results/README.md` | 结果 schema、生成命令、哪些是预计算结果、哪些需重新运行 |
| `selector/misleading_evidence_summary.json` | Selector 在误导证据压力测试上的最终聚合指标 |
| `selector/blind_answer_gate.json` | blind answer gate 未显示支持性提升的冻结结果 |
| `experiment04/final_tables.md` | 人可读的 Exp04 最终表 |
| `experiment04/summary_metrics.csv` | 便于论文/分析工具读取的聚合值 |
| `experiment04/final_results.json` | 带 schema、配置哈希和 seed 的机器可读结果 |
| `experiment04/final_audit.json` | 完整性、模型/数据哈希和协议检查 |
| `experiment04/tables.tex` | 可直接放入论文的冻结 LaTeX 表 |
| `experiment05/final_report.md` | Exp05 最终报告和 Claim A/B 判定 |
| `experiment05/table1.csv/.json` | Table 1 的人机可读双格式 |
| `experiment05/table2.csv/.json` | Table 2 的人机可读双格式 |
| `experiment05/claim_labels.json` | 每个预注册 claim 的支持/不支持标签及依据 |
| `experiment05/bootstrap.json` | bootstrap 差值和置信区间 |
| `experiment05/final_audit.json` | 运行完整性和协议审计 |
| `experiment05/tables.tex` | 论文使用的冻结表格 |

不保留逐题完整 prompt/response、调试 dump、重复中间快照和几百 MB 的原始运行目录。需要审计这些内容时去 archive 分支或外部研究存储。

## 11. `examples`：最快理解实际用法

| 文件 | 用途 |
|---|---|
| `examples/quickstart.py` | 用 Python API 加载最终配置并完成一次查询 |
| `examples/api_request.py` | 调用本地 HTTP API，展示前端实际收到的数据 |
| `examples/mock_frontend_response.json` | 无 GPU/模型时供前端开发的固定响应样例 |

## 12. 不进入正式 `main` 的内容

以下内容不会丢失，但默认首页不再展示：

- `.aris/`、refine logs、Codex/代理工作记录和本次 `task_plan.md`、`findings.md`、`progress.md`。
- 旧版 Full-Flow 文档、过期 tracker、计划草稿、阶段快照和大量内部 handoff。
- 没有进入论文最终方法的 Retriever/Selector/Generator 历史版本。
- 重复、一次性、仅修复某次 HPC 作业的脚本和配置 sweep。
- 原始 `runs/`、大部分 `results/` 中间快照、逐题 generation、cache、index 和日志。
- Selector/Generator 模型权重、基础模型缓存和第三方数据副本。
- 个人路径、HPC 账号、临时目录、令牌、内部 endpoint 和机器相关配置。
- 旧演示文稿、录屏和不参与复现的图片；必要架构图可重新以小型可编辑格式加入正式文档。

这些材料继续存在于 `archive/full-research-history-2026-08-25` 和 `research-archive-2026-08-25`，也仍可从 Git 历史查看。

## 13. 外部保存但由仓库登记的资产

| 资产 | 预计处理 |
|---|---|
| Selector Lean v3 seed13 `model.safetensors`（当前约 704 MiB） | 放在合规的外部模型存储；manifest 记录 URL、SHA-256 和加载位置 |
| Selector seed42/73 checkpoints | 核实大小、最终实验实际使用和发布许可后登记 |
| GR-C Generator seed13 adapter（当前约 60 MiB） | 外部模型存储；和基础 Granite 模型分开登记 |
| Generator seed42/73 adapters | 按最终复现要求和许可登记 |
| 基础 Granite、NLI、Provence 等第三方模型 | 只记录官方模型 ID、revision 和许可证，不复制进 Git |
| 数据集与固定 split | 公开数据记录原始来源和下载脚本；受限数据只给申请/放置说明 |
| 检索索引 | 默认由数据重建；如重建成本过高，可另发带 checksum 的归档 |
| 原始 generations、HPC logs、cache | 研究归档或学校存储；不作为使用正式系统的前置条件 |

前端同学不需要模型文件出现在自己的电脑或 GitHub 工作区。合理部署是后端所在机器拥有这些资产，前端通过 API 请求；只有负责运行后端的人才需要下载 checkpoint。

## 14. 四类人打开仓库后分别怎么走

### 毕设评审者

`README.md` → `docs/architecture.md` → `docs/results.md` → `docs/limitations.md` → `REPRODUCIBILITY_MAP.md`

他能看懂方法、结果和限制，不需要翻开发历史。

### 想运行系统的人

`README.md` → `docs/setup.md` → `ARTIFACT_MANIFEST.json` → `configs/runtime/final_seed13.toml` → `examples/quickstart.py`

他不需要修改源码里的个人路径。

### 前端同学

`docs/frontend-integration.md` → `examples/mock_frontend_response.json` → `examples/api_request.py`

先用 mock 开发页面；部署时把 API 地址换成运行三模块模型的后端，不共享个人 HPC 账号。

### 想复现实验的人

`REPRODUCIBILITY_MAP.md` → `docs/reproduction.md` → `experiments/README.md` → 对应配置 → `results/`

他可以选择只重建表格、运行 CPU smoke，或在 GPU/HPC 上完全复现。

## 15. 与当前仓库相比，核心变化是什么

| 当前状态 | 正式版状态 |
|---|---|
| 根 README 短且与最终方法不完全一致 | README 只讲冻结后的三模块方法和真实结果 |
| 代码、过期文档、过程脚本和大结果并列 | 默认树按运行、复现、结果和文档分层 |
| 多处个人/HPC 绝对路径 | 环境变量 + manifest + 相对路径 |
| 模型存在个人账号，别人不知道如何取得 | 外部资产有版本、链接、许可、大小和 checksum |
| Top-K 和 trained Selector 容易混淆 | 主配置明确使用 trained Selector，Top-K 仅作为 baseline/ablation |
| 只有 Python pipeline，前端边界不稳定 | 增加最小 HTTP API 和 mock response |
| Exp04/05 大量中间产物 | Git 只保留最终聚合结果与审计，原始产物留在归档/外部存储 |
| “PASS” 容易被误解为方法更好 | 文档区分协议执行通过和科学主张是否获支持 |

## 16. 后续 Goal 如何把它实现出来

1. **G1：先冻结完整现场。** 当前所有有效开发和实验材料进入归档引用。
2. **G2：建立正式候选树。** 从归档创建 release 分支，按本设计先做保留/归档，不改算法。
3. **G3：冻结实际可运行系统。** 验证依赖闭包、统一配置、补最小 API 和 CPU smoke。
4. **G4：整理复现包。** 把已验证脚本收束成上述 experiments/configs/results 结构。
5. **G5：登记外部资产。** 发布或登记模型、数据、索引及 checksum。
6. **G6：完成公开文档。** 用最终事实填写 README、model cards、结果和限制。
7. **G7：独立 clean-clone 验证。** 证明目录不是只在原作者电脑上可用。
8. **G8：接入 main 并发布。** 验收后的 release 接到默认首页并打正式标签。

## 17. 执行前仍需确认的边界

这份结构采用以下推荐假设；若与你预期不同，应在 G1 前修改计划：

1. **前端代码不强行搬进本仓库。** 本仓库提供稳定后端 API、mock 和集成文档，并在 README 链接前端仓库。若前端本来就在同一仓库，则需要另加 `frontend/` 并纳入 CI。
2. **保留最小文档 ingestion 能力。** 如果论文只需要固定 benchmark，可以删减 `loaders/` 和 `cli/ingest.py`。
3. **公开的是聚合结果，不是全部逐题输出。** 如学校要求完整逐题附件，应作为独立 release asset/数据存档登记。
4. **外部模型能否公开取决于许可证和项目同意。** 若不能公开，仓库会提供受限获取说明和 mock，不会伪造公开下载链接。
5. **当前命名是目标信息架构。** 后续若某个现有文件被多个测试依赖，优先保留经过验证的实现，并用包装器提供整齐入口，而不是冒险重写。

只要这五点符合预期，这份树就可以作为 G1–G8 的目标结构合同；后续每个 Goal 的验收都以它为参照。
目标结构不构成按文件名盲删的授权：G3/G4 必须先用 import、测试和实验入口确认依赖闭包，任何仍被最终系统调用的现有辅助文件都先保留，再决定是否合并或改名。
