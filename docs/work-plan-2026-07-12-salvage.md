# Work Plan Salvage — 2026-07-12（从 work-plan-2026-07-09.md 提取仍有效的部分）

状态：工作草稿，供重新分工讨论或个人独立执行；不替代站会决定。
来源：`docs/work-plan-2026-07-09.md`（旧分工，已失效）＋ `docs/项目完整说明.md`（新定位：Retriever → Selector → Generator）。
原则：**只有分工作废，硬日期、纪律和未完成的实验本身都没作废。**

## 0. 不变的硬约束（原样保留）

- 实验冻结 **2026-08-03**（未认证结果只进 Future Work）· 群报告 **2026-09-04** · 展示 2026-09-11。
- Definition of done：per-query CSV 提交（`git add -f results/*.csv`）＋ `python -m eval.significance` ＋ results-summary 一段 finding。
- GPU：A100 slice `gpu:3g.40gb:1`；复用 `--cache-dir` 索引；spine 实验优先于背景任务。
- Report first：工作流超窗就砍掉写成 limitation。报告占 50% 分数。
- 引用卫生：每个 arXiv ID 写进报告前打开 abs 页核实（Bristol integrity）。
- 个人报告钩子：每个工作项保持"目标→方法→证据→局限"自包含结构，per-query CSV 和显著性输出留档。

## 1. 已完成 / 已被吸收——不要重做

| 旧条目 | 状态 | 证据 |
|---|---|---|
| WS-10 主体（top-k 4 vs 10 + Astute on NQ） | DONE | finding 16/17；`results/astute_nq300_*.csv` |
| WS-11 一半（q2d 接入 demo） | DONE | 07-05 合入 |
| Stretch #1（RAMDocs 外部验证） | DONE（被 selector B5/B6 吸收） | `docs/data/ml_selector_validation/ramdocs/` |
| Stretch #4（held-out distractor family） | 大部分吸收（sealed NIAH 测试集换了反事实模板） | `eval/prepare_niah_selector_split.py` |
| WS-12 部分（scale 曲线、selector 三图） | PARTIAL | `results/niah_*`、`docs/data/ml_selector_validation/*.png` |
| WS-13 部分（related-work 草稿） | PARTIAL | `docs/related-work-draft-2026-07-11.pdf` |
| Generator 侧引用指标（新增，不在旧计划内） | DONE | 43e214e：`score_citation_*` + `eval/citation_eval.py` |
| ML selector 验证轮（新增，不在旧计划内） | DONE（Gate 1/2 FAIL，按协议停机） | `docs/ml-selector-experiment-tracker.md` |

## 2. 存活工作项 → 新三模块的映射

优先级：P0 = 报告必需 · P1 = 高价值 · P2 = stretch。
"独立可做" = 无 GPU 或仅提交 slurm 等结果，一人可完成。

### Keystone（先做，30 行代码解锁 4 个分析）

| 项 | 旧编号 | 内容 | 成本 | 独立可做 |
|---|---|---|---|---|
| **Runs-dump 扩展** | WS-0 | `eval/run_niah.py` 加 `--dump-runs`（存 dense/q2d 完整 top-100 排名）＋ 冻结 300q 重 dump 一次 | 无 GPU（重 dump 用缓存索引，轻 GPU） | 是 |

仍然卡着 WS-2 / WS-3 / WS-5。`tune_corroboration` 的 dump 只有 top-20 q2d 池，不够。

### Retriever 组（"找全"）

| 项 | 旧编号 | 新定位 | P | GPU | 独立可做 |
|---|---|---|---|---|---|
| TriviaQA haystack 复制 | WS-1a | Retriever 结论的稳健性（回应"单任务实例"批评），仍是全项目单个最高价值实验 | P0 | 是 | 是（build 离线 + 一次 slurm） |
| Fresh-seed NQ 确认（α 冻结） | WS-1b | 预注册复现，消掉最后的调参 caveat | P1 | 是 | 是 |
| Oracle headroom 分解（R@100 / oracle routing / pool-union） | WS-5 | Retriever 模块评估 = "该找的进池没有"；同时 gate WS-14 | P0 | 否（需 WS-0） | 是 |
| Hybrid arm 进 NIAH stack | WS-8 | 攻 13% unreachable（recall headroom）；生产系统默认 hybrid | P1 | 是 | 是 |
| IVFPQ 10M（21M stretch） | WS-9 | "enterprise 规模"主张；先做 100k 吞吐探针再定目标；embedding 是真实成本，用 faiss reconstruct 回收已有 5M 向量 | P2 | 重 | 勉强（后台断点续跑） |
| Embedder 硬负例微调 | WS-14 | **2026-07-13 团队决策：正式立项为 freeze 前唯一新赌注。** spec 已用本轮文献补强（`docs/superpowers/specs/2026-07-08-embedder-hardneg-finetune-design.md` §12）——"query-blindness"论证证明它是唯一不受"判别实体不在 query 里"这一结构性盲区限制的方法类别，唯一可能移动 MRR（不只 @10 边界）。**仍然 gated on WS-5**：立项不等于解除排序前提，WS-0→WS-5→WS-14 顺序不变。建议先做 GRADA 离线消融（spec §12，零 GPU/零 LLM，1-2 天）作为 WS-14 前置的免费验证 | **P0（团队唯一下注）** | 重 | 否（建议结对） |

### Selector 组（"选准、找齐、去干扰"）

| 项 | 旧编号 | 新定位 | P | GPU | 独立可做 |
|---|---|---|---|---|---|
| Burial attribution（谁把 needle 压下去） | WS-2 | Selector 失败机理证据；预测 corroboration 收益集中在 cf-buried query | P0 | 否（需 WS-0） | 是 |
| Rank migration + found@k + k 敏感性 | WS-3 | 拆掉"@10 边界伪影"批评；直接喂 metrics bridge | P0 | 否（需 WS-0） | 是 |
| 置信度门控级联（dense→q2d→corroborate） | WS-4 | 成本故事 **兼** 新定位 §3.2"动态补充检索停止条件"的第一个可测实例——包装价值升级 | P1 | 否 | 是 |
| Selector Gate-0 人工双标 | selector 下一轮 | 决定已有 NDCG/Recall 正向结果能否解释；需要第二标注人（非作者） | P0 | 否 | 否（需两人） |
| 语义投票匹配 | WS-7 | 方法升级，保留 Step-0 kill-switch（先测 vote-splitting 频率，不显著就一句话收尾）；timebox 1 周 | P2 | 否 | 是 |
| 自然管线迁移（NQ/TriviaQA 上跑 q2d_corroborate） | WS-6 | 外部效度："为冲突证据域设计，无冲突域安全" | P1 | 是 | 是 |
| 实体/时间/条件/冲突特征 + harmful penalty | selector 下一轮 | **只在有确定性 metadata 的域做**（FinanceBench 公司/周期、ContractNLI 条件）——NIAH 上 Granite judge kappa 0.005–0.08 已证明 LLM 造不出这些特征 | P2（freeze 后/个人报告） | 部分 | 是 |

### Generator 组（"按证据回答、引用、拒答"）

| 项 | 旧编号 | 新定位 | P | GPU | 独立可做 |
|---|---|---|---|---|---|
| 引用质量 A/B（用已有 dumps 跑 citation P/R/F1） | 新（43e214e 之后的自然下一步） | Generator 模块评估主指标之一 | P1 | 否/轻 | 是 |
| Astute over NIAH haystack | WS-10 stretch | Astute 的本命场景（冲突域）还没测——finding 16 只测了无冲突 NQ | P2 | 是 | 是 |
| Demo：corroborate arm 接入 | WS-11 剩余 | 展示"降级一个 counterfactual"+ 引用 + 拒答 | P1 | 本地 | 是 |
| 证据不足拒答评估（sufficient-context 风格） | 新定位 §2.3 | 用现有 abstention 字段 + 官方证据集合做 selective-accuracy 曲线；文献锚点 arXiv:2411.06037 | P2 | 否 | 是 |

### 报告 / 集成（跨组）

| 项 | 旧编号 | 内容 | P |
|---|---|---|---|
| Metrics bridge 一节 | 07-11 brief §6 | needle-found@10 ↔ utility-NDCG/Harmful@10 是同一条链路相邻环节；一段讲清，两套数字不打架 | P0 |
| 三模块 ↔ 已认证 findings 映射表 | 新 | Retriever=findings 1–11+12/14/17；Selector=13/15/18+selector 验证轮；Generator=16/17+citation 指标。新叙事必须吸收全部旧结果，不许作废任何 certified finding | P0 |
| Future work + related work（引用核实） | WS-13 | 草稿已有；CAR/MADAM-RAG 为主邻居，VOTE-RAG 有 text-overlap flag 只顺带提 | P0 |
| 图表 + 统一效率表 | WS-12 | scale 曲线（+10M/21M 点若落地）、migration 热图、burial 堆叠条、级联成本曲线、效率表 | P0 |
| Brief→deliverable 映射表 | WS-13 | 对 IBM brief 的可追溯性 + 诚实的范围裁剪 | P1 |

## 3. 明确丢弃（写进报告 limitation / future work，不再排期）

- 旧的按人分工表（人员重排后无效）。
- Selector 在 NIAH 上继续加训练轮次——07-11 brief §3：同一堵墙（固定池 + 相关性系特征无区分度），边际收益低于以上任何 P0/P1。
- 动态补充检索的完整实现——新定位文档自己已说"第一版固定 top-20，动态检索是后续增强"；WS-4 级联是它 freeze 前唯一可测的代理。
- set-aware 组合选择、Granite selector 微调、私有企业数据（selector 计划 §18 原样保留 deferred）。

## 4. 最小可行收尾（如果只剩一个人 + 有限 GPU）

顺序执行，全部独立可做：

```text
WS-0（解锁）→ WS-2 + WS-3 + WS-5（三个离线分析，出机理+天花板）
→ WS-1a（一次 GPU 复制实验）→ WS-4（离线级联）
→ 引用质量 A/B（离线）→ 报告冲刺（metrics bridge + 映射表优先）
```

这条线零重 GPU 风险，回应全部三个批评（单实例、边界伪影、成本），并给新三模块叙事每个模块至少一个已认证证据。
