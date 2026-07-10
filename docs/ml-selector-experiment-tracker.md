# ML Evidence Selector 实验执行 Tracker

**分支：** `week5_MLSelector`
**计划：** `docs/ml-selector-experiment-plan.md`
**当前状态：** 计划已修订，实验尚未开始

| Run ID | 阶段 | 任务 | 数据 | 关键产出 | 通过检查 | 优先级 | 状态 |
|---|---|---|---|---|---|---|---|
| P001 | M0 | 冻结标签 schema、主比较、指标和 Gate | 全部 | `protocol_manifest.json` | 团队确认 | MUST | TODO |
| D001 | M1 | 建立 dataset manifest 和许可记录 | 全部 | `dataset_manifest.json` | 字段与官方说明一致 | MUST | TODO |
| D002 | M1 | 建立父页面和 synthetic-family split | 新 NIAH | `split_manifest.json` | 父页面/family 无交叉 | MUST | TODO |
| D003 | M1 | 验证官方 contract split | ContractNLI | contract split | contract 无交叉 | MUST | TODO |
| D004 | M1 | 建立 company nested 5-fold | FinanceBench | company folds | company 无交叉 | MUST | TODO |
| D005 | M1 | 准备 official 与 adapted 两套协议 | RAMDocs | RAMDocs / RAMDocs-20 manifest | 不使用测试标签挖负例 | MUST | TODO |
| L001 | M2 | 实现多维标签 schema | 全部 | `label_schema.json` | 字段含义固定 | MUST | TODO |
| L002 | M2 | 派生五级 utility 和 harmful flag | 全部 | labeled candidates | 映射规则可复现 | MUST | TODO |
| L003 | M2 | 标注和复核至少 80 个 query group | 分层样本 | `label_audit.csv` | weighted kappa ≥ 0.70 | MUST | TODO |
| C001 | M3 | 生成训练和开发 top-20 | 新 NIAH | candidate cache | 父文档去重正确 | MUST | TODO |
| C002 | M3 | 生成 sealed NIAH top-20 | 新 NIAH test | sealed cache | 协议冻结后运行 | MUST | TODO |
| C003 | M3 | 合同内部生成 top-20 span | ContractNLI | candidate cache | 不跨合同检索 | MUST | TODO |
| C004 | M3 | company-fold 财报候选 | FinanceBench | candidate cache | test company 未参与训练 | MUST | TODO |
| C005 | M3 | 保留官方候选并补足至 20 | RAMDocs | official/adapted caches | 两套结果分开 | MUST | TODO |
| F001 | M3 | 生成 QA 结构化抽取缓存 | NIAH、Finance、RAMDocs | QA features | prompt/hash 固定 | MUST | TODO |
| F002 | M3 | 生成 NLI stance 缓存 | ContractNLI | NLI features | entail/contradict/unknown | MUST | TODO |
| F003 | M3 | 生成 support/conflict/condition 特征 | 全部 | Full features | 标签来源独立 | MUST | TODO |
| F004 | M3 | 检查候选、token budget 和 tie-break | 全部 | fairness audit | 各方法完全一致 | MUST | TODO |
| B001 | M4 | 复现旧 q2d 和 fixed 结果 | 旧 NIAH 300 | legacy metrics | 仅作 replication | MUST | TODO |
| B002 | M4 | mixed-dev 选择 `alpha*` | dev only | alpha curve | test 未参与选择 | MUST | TODO |
| B003 | M4 | source-deduplicated fixed baseline | dev/test | diagnostic metrics | source_parent 生效 | MUST | TODO |
| B004 | M4 | Oracle@20 | 全部 | headroom table | top-20 上限明确 | MUST | TODO |
| S001 | M5 | shuffled-label negative control | mixed train/dev | negative-control metrics | 接近随机排序 | MUST | TODO |
| S002 | M5 | feature-to-dataset probe | mixed features | leakage report | 域指纹可解释 | MUST | TODO |
| S003 | M5 | support-score-only baseline | mixed dev | diagnostic metrics | 隔离 judge 贡献 | MUST | TODO |
| M001 | M5 | Logistic/linear ranker sanity | mixed train/dev | sanity model | 流程可复现 | MUST | TODO |
| M002 | M6 | 训练 LightGBM Core | mixed train/dev | Core model | 3 seeds | MUST | TODO |
| M003 | M6 | 训练 LightGBM Full | mixed train/dev | Full model | 3 seeds | MUST | TODO |
| E001 | M7 | Legacy replication | 旧 NIAH 300 | legacy result | 不作盲测结论 | MUST | TODO |
| E002 | M7 | Blind controlled test | sealed NIAH | main result | Gate 1 | MUST | TODO |
| E003 | M7 | 合同证据测试 | ContractNLI test | evidence/NLI result | contract cluster stats | MUST | TODO |
| E004 | M7 | 财报 OOF 测试 | FinanceBench | company-level OOF | company cluster stats | MUST | TODO |
| E005 | M7 | 官方冲突诊断 | RAMDocs official | top-3 result | 与 adapted 分开 | MUST | TODO |
| E006 | M7 | 外部 20→10 测试 | Adapted RAMDocs-20 | main external result | Gate 2 | MUST | TODO |
| G001 | M7 | NQ+Finance→Contract | LODO | generalisation result | 目标域未参与训练 | MUST | TODO |
| G002 | M7 | NQ+Contract→Finance | LODO | generalisation result | 目标公司未参与训练 | MUST | TODO |
| G003 | M7 | 全部非 RAMDocs→RAMDocs | LODO | generalisation result | RAMDocs 零接触 | MUST | TODO |
| A001 | M7 | relevance/rank/corroboration 消融 | 全部 | ablation table | 固定候选 | MUST | TODO |
| A002 | M7 | parametric vote/source-dedup 消融 | 全部 | ablation table | 投票来源明确 | MUST | TODO |
| A003 | M7 | support/conflict/metadata 消融 | 全部 | ablation table | Full 贡献明确 | MUST | TODO |
| A004 | M7 | metadata-free/missingness-masked | 全部 | leakage ablation | 域指纹受控 | MUST | TODO |
| A005 | M7 | 三级 vs 五级标签 | 全部 | label ablation | 标签粒度价值明确 | MUST | TODO |
| R001 | M8 | Blind NIAH RAG | sealed NIAH | Answer F1 | Gate 3 | MUST | TODO |
| R002 | M8 | FinanceBench RAG | FinanceBench OOF | numeric accuracy/citation | Gate 3 | MUST | TODO |
| R003 | M8 | RAMDocs RAG | official/adapted | strict EM | Gate 3 | MUST | TODO |
| I001 | M9 | 注册 `q2d_ml_selector_core/full` | 项目代码 | retriever arms | top-20 重排、tail 保序 | MUST | TODO |
| I002 | M9 | 离线与在线排序一致性 | 固定 fixtures | integration tests | 排名逐 query 一致 | MUST | TODO |
| V001 | M9 | 生成主表和五张核心图 | 最终结果 | report assets | 数字可追溯 | MUST | TODO |
| X001 | 后续 | set-aware evidence selection | 待定 | extension | Gate 1–2 后评估 | LATER | BLOCKED |
| X002 | 后续 | Granite selector 微调 | 待定 | extension | 轻量模型达到上限后评估 | LATER | BLOCKED |

## Gate 记录

| Gate | 条件 | 状态 | 证据文件 | 结论 |
|---|---|---|---|---|
| Gate 0 | 无 split/feature 泄漏，weighted kappa ≥ 0.70 | TODO |  |  |
| Gate 1 | sealed NIAH：NDCG +0.02、Harmful -0.02、Recall 非劣 | TODO |  |  |
| Gate 2 | 三个真实/外部域至少两个正向，LODO 至少两域为正 | TODO |  |  |
| Gate 3 | 至少两个答案主指标改善，所有域通过非劣检验 | TODO |  |  |
