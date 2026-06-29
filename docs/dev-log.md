# 开发日志 / Dev Log

> **这份文档记录"谁实际做了什么"** —— 和 [team-roles.md](team-roles.md)(谁*负责*什么,
> 计划)互补。用途:个人报告找素材、团队对齐进度、向 IBM 汇报贡献时有据可查。
>
> **怎么维护(每个人自己更新):**
> 1. 干完一块活,在下面**你的小节**里加一行(做了什么 / 改了哪些文件 / 状态)。
> 2. 顺手在底部 **Changelog** 加一行(日期 / 区域 / 简述 / 文件 / 谁)。
> 3. 状态用:`☐ 未开始` · `◐ 进行中` · `✅ 完成(测试绿)`。
> 4. 改了跨模块接口?先看 [interfaces.md](interfaces.md) 的变更规则,别私自改契约。

配套:[team-roles.md](team-roles.md) · [interfaces.md](interfaces.md) · [work-plan-no-gpu.md](work-plan-no-gpu.md) · [meeting-1-questions.md](meeting-1-questions.md)

---

## 项目状态快照(2026-06-12)

- **范围已定(Meeting 1 确认):** **A+B = 检索 + RAG** 为 headline;NIAH 降为次要诊断;
  数据**只用公开数据集**(不碰企业数据/不换商业协议);交付物是**可用的 MVP**,不止 research report。
- **框架地基:✅ 已就位** —— repo、架构、4 个契约、所有 `@dataclass`、RAG glue 都立住。
- **基建地基:✅ 生成层已验证** —— BluePebble(HPC)已接通,环境跑通,**granite-4.1-3b 在 GPU 节点成功生成**(`Smoke test OK`)。可用组合:**python 3.12 + torch 2.5.1+cu121 + transformers 4.57**(HPC 用 4.x,见 [hpc-deployment.md](hpc-deployment.md))。脚本:`scripts/smoke_8b.slurm` / `run_rag.slurm`。待全队需 GPU 者复用 + 共享模型缓存。
- **核心逻辑:☐ 大部分待建** —— 检索核心(P4)、各指标、loader 实现仍是 skeleton。
- **测试基线:** `pytest -q` = 6 passed / 2 xfailed。
- **下一步:** 给 Bharat 发**架构图**(CPU/GPU 分工 + 数据如何喂进 RAG);跑通 CPU 最小闭环(loader→ir_metrics→bm25→run_benchmark)出第一张表。

---

## 个人工作记录

> 每人维护自己的小节。"已完成"只写真正做出来的;计划中的写进"下一步"。

### P1 — Benchmark 数据加载 · 许展瑜
- **负责:** `eval/benchmarks/loader.py` → `BenchmarkData`
- **已完成:** loader拉取scifact/test并返回corpus/queries/qrels
- **进行中:** 
- **下一步:** A+B 下需支持**带 gold answer** 的 QA 集(已为此加了可选字段 `BenchmarkData.answers`,见 Changelog 2026-06-12)
- **备注:** 这是所有评测的基石,优先做。

### P2 — IR 指标 · Frida
- **负责:** `eval/ir_metrics.py` → `Run`/`Qrels`
- **已完成:** ◐ `ir_metrics` 草稿(precision/recall/nDCG/MRR via ranx,commit 93640f8)—— _本人确认是否已测绿_
- **进行中:** _(待本人填)_
- **下一步:** RAG 答案质量指标搬到了新文件 **`eval/rag_metrics.py`**(不是 `metrics.py`),若由你实现 context_precision / faithfulness,请写在那里(见 Changelog)。
- **备注:** spike 阶段还跑过 BM25/SciFact(ndcg@10≈0.56)+ NIAH + Ollama 后端,均为参考,以仓库 canonical 文件为准。

### P3 — BM25 基线 · 尤佳希 + 魏铭
- **负责:** `src/retrieval/bm25_baseline.py`
- **已完成:** ✅ `BM25Retriever` 已实现:使用 `rank_bm25.BM25Okapi` 建关键词检索索引,`retrieve(query)` 返回符合契约 1 的 `List[RetrievedChunk]`;已覆盖 top-k、排序、空 query、空 corpus、corpus/doc_ids 长度校验。✅ 已通过 SciFact smoke:`load_benchmark("scifact")` + BM25 top-5 检索可运行。
- **进行中:** ☐ 等 P6 `run_benchmark.py` 串联后,在 SciFact 上跑完整 BM25 baseline 指标。
- **下一步:** 配合 P6 将 BM25 输出整理成 `Run = {query_id: {doc_id: score}}`,用于 precision/recall/nDCG/MRR。

### P4 — Embedding + 稠密检索器(核心) · 尤佳希 + 魏铭
- **负责:** `src/retrieval/embedder.py`, `src/retrieval/retriever.py`
- **已完成:** ✅ `Embedder` 已实现:支持 `granite` 与 `sentence-transformers` 两个 backend,模型 id 从参数或环境变量读取,输出 `List[List[float]]` / `List[float]`;单元测试使用 fake model,避免下载真实模型。✅ `DenseRetriever` adapter 已实现:复用契约 1,query 经 embedder 编码后调用 `index.search(query_vector, top_k)`,返回 `List[RetrievedChunk]`。
- **进行中:** ☐ 等 P5 `VectorIndexer` 提供 FAISS index wrapper 后,做真实 dense retrieval 集成测试。
- **下一步:** 和 P5/P6 正式确认 dense index 接口 `index.search(query_vector, top_k) -> List[RetrievedChunk]` 以及相似度策略(normalized embeddings + inner product 或 raw embeddings + L2 distance)。

### P5 — Ingestion 流水线 · 吴泽楠
- **负责:** `src/ingestion/{loaders,chunker,indexer}.py`
- **已完成:** _(待本人填)_
- **进行中:** ☐ `chunker`(token 切块,可独立先做)
- **下一步:** `indexer` 建 + 存 FAISS(需 P4 的 embedder);和 P4 对齐 `Chunk`/索引交接格式。

### P6 — 集成 + explainability + 接口负责人 · 毛威凯
> 详细个人开发记录见 **[dev-log-p6.md](dev-log-p6.md)**。
- **负责:** `eval/run_benchmark.py`, `src/explainability/citations.py`,接口 owner
- **已完成:**
  - ✅ 4 个跨模块契约(`src/retrieval/base.py` + `interfaces.md`):Retriever / Run / max-pool / **RAG I/O(新增)**
  - ✅ **A+B RAG 改造**(2026-06-12):`rag_pipeline` 改为复用 `Retriever`、新增 `run_rag.py` + `rag_metrics.py`、`BenchmarkData` 加 `answers`、app 改 RAG 口径(详见 Changelog)
  - ✅ **HPC 部署**(2026-06-12):BluePebble 生成层跑通(granite-4.1-3b 验证);产出 `scripts/smoke_8b.*` / `run_rag.slurm` + `docs/hpc-deployment.md`;repo 设为 public(已扫无密钥)
  - ✅ Meeting 1 准备 + 文档对齐(`meeting-1-questions.md` Q1/Q2/Q5)
  - ✅ **`run_benchmark` 检索编排**(2026-06-18):`build_run`/`evaluate_one`/`write_results_csv`/`_build_retrievers`/索引缓存/CLI,TDD;第一张 SciFact 表(granite_dense nDCG@10 0.767 > BM25 0.636 ≈ ST 0.641)
  - ✅ **A0 检索 ablation 配置打通**(2026-06-19,`week3` 未提交):chunk/pooling/model 旋钮接 CLI、配置列 CSV + append、修索引缓存"串味"bug;9 新测试,套件 49 passed/1 xfailed(详见 [dev-log-p6.md](dev-log-p6.md) §5)
  - ✅ **RAG 评测线 review + 标准化**(2026-06-29):合入 `week3_b1_c1`/`week3_b2` 到 `week3` 后,把 RAG 指标重写为学界标准口径——SQuAD 归一化 **EM + token-F1**(多 gold 取 max)、context precision 改 **qrels precision@k**(修了"把 qid 当问题文本"的真 bug)、faithfulness 简化为 token 覆盖;`_tokenize/_jaccard/_split_sentences` 抽到 `src/text_utils.py`;loader 加 **qrels 感知子集抽样**(`max_queries/max_docs`)、`answers` 改 `List[str]`(NQ 多别名);`run_rag` 加 CLI + `--per-query-out`(逐查询分数→显著性);TDD,套件 **218 passed/1 xfailed**(详见 Changelog)
- **进行中:** NQ 子集 RAG run(granite/gte/bm25,500q/500k,`scripts/run_rag.slurm`)待上 HPC;先在登录节点 verify `dpr-w100` 的 qrels/answers + 预取模型
- **下一步:** 跑 NQ 子集出第一组答案质量数(EM/F1)+ 配对显著性(`eval/significance.py`);合理则放大子集 / 上 HNSW 全量;给 Bharat 的架构图。

### P7 — 可视化 + demo + 结果 · _(待填)_
- **负责:** `notebooks/`, `app/main.py`
- **已完成:** _(待本人填)_
- **进行中:** ☐ Streamlit demo(已改成 RAG 口径:检索→生成→展示引用)+ 对比图(对 mock `results.csv`)
- **下一步:** A+B 下 demo 要展示完整 retrieve-then-generate-with-citations 流程。

---

## Changelog(倒序,最新在上)

| 日期                   | 区域               | 改动                                                                                                                                            | 文件 | 谁  |
|----------------------|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------| --- |----|
| 2026-06-29           | P6 RAG评测线 | **RAG 评测线打通(review→标准化→可跑)**:合入 b1_c1(B1指标+C1引用)/b2(B2 runner)到 `week3`;review 后把 RAG 指标重写为学界标准口径——SQuAD 归一化 EM+token-F1(多 gold 取 max)、context precision 改 qrels precision@k(**修了把 qid 当问题文本的 bug**)、faithfulness 简化为 token 覆盖;`_tokenize/_jaccard/_split_sentences` 抽到 `src/text_utils.py`(去重复+修句切漂移);loader 加 qrels 感知子集抽样(`max_queries/max_docs`)、`answers` 改 `List[str]`;`run_rag` 加 CLI + `--per-query-out`;`run_rag.slurm` 改 NQ 子集 granite/gte/bm25 三路。TDD,套件 218 passed/1 xfailed | `eval/rag_metrics.py`, `src/explainability/citations.py`, `src/text_utils.py`, `eval/run_rag.py`, `eval/benchmarks/loader.py`, `eval/metrics.py`, `scripts/run_rag.slurm`, `tests/` | P6 |
| 2026-06-23           | P6 生成层/MVP | **B3:8B 冒烟通过**(granite-4.1-8b 在 BluePebble RTX 3090 加载+生成,`Smoke test OK.`,job 17874131;3B/8B 两模型均验证)。MVP 接缝:`build_dense_retriever_from_text` + `build_rag_pipeline_from_text`,`app/main.py` 由 mock 改接真 RAG 管线(TDD,9 新测试;app 待本机 3B 跑通确认) | `scripts/smoke_8b.slurm`, `src/retrieval/factory.py`, `src/rag_app.py`, `app/main.py`, `tests/` | P6 |
| 2026-06-22           | P1 loader | A3 beir下多数据集、B4 NQ带答案完成 | `eval/benchmarks/loader.py`, `tests/test_benchmarks_loader.py.py` | P1许展瑜 |
| 2026-06-19           | P6 ablation | **A0:检索 ablation 配置打通** —— `build_run` 加 `pooling`(max/mean)、chunk/model 旋钮接 CLI、`write_results_csv` 配置列 + append、新增 `_cache_key` 修索引缓存"串味"bug;test-first 9 新测试,套件 49 passed/1 xfailed(`week3` 未提交) | `eval/run_benchmark.py`, `tests/test_run_benchmark.py`, `docs/interfaces.md` | P6 |
| 2026-06-18           | P6 检索编排 | **`run_benchmark` 检索线完成** —— `build_run`(chunk→doc max-pool)/`evaluate_one`/`write_results_csv`/`_build_retrievers`/索引缓存/CLI,TDD;第一张 SciFact 表(granite_dense nDCG@10 0.767 > BM25 0.636 ≈ ST 0.641) | `eval/run_benchmark.py`, `tests/test_run_benchmark.py` | P6 |
| 2026-06-16           | P4 检索核心 | 实现 embedding wrapper 与 DenseRetriever adapter,统一输出 `RetrievedChunk`;P5/P6 仍需确认 `index.search(query_vector, top_k)` 的正式交接接口与相似度策略 | `src/retrieval/embedder.py`, `src/retrieval/retriever.py`, `tests/test_retrieval_embedder.py`, `tests/test_retrieval_dense.py` | P3/P4 尤佳希 + 魏铭 |
| 2026-06-16           | P3 BM25 | 实现 BM25 baseline,支持 top-k 检索、稳定排序、空输入处理和契约测试;新增 SciFact smoke 脚本并验证 P1 loader + P3 BM25 可联通 | `src/retrieval/bm25_baseline.py`, `tests/test_retrieval_bm25.py`, `scripts/smoke_bm25_scifact.py` | P3/P4 尤佳希 + 魏铭 |
| 2026-06-13           | 数据加载 | 完成benchmark数据加载                                                                                                                                   | `loader.py` | P1 |
| 2026-06-12           | HPC              | **BluePebble 生成层跑通**(granite-4.1-3b 验证);Slurm 脚本 + 部署文档;HPC 版本修复(transformers 4.x);repo 设 public                                              | `scripts/smoke_8b.*`, `scripts/run_rag.slurm`, `docs/hpc-deployment.md`, `.gitattributes` | P6 |
| 2026-06-12           | 文档               | 新建本开发日志 + 个人开发文档                                                                                                                              | `docs/dev-log.md`, `docs/dev-log-p6.md` | P6 |
| 2026-06-12           | 范围/RAG           | **RAG 提为 A+B co-headline 的仓库改造**:RAGPipeline 复用 Retriever(不再自建检索栈)、新增 RAG 主线评测入口、RAG 指标独立成模块、BenchmarkData 加可选 `answers`、新增契约 4、demo 改 RAG 口径 | `src/rag_pipeline.py`, `eval/run_rag.py`(新), `eval/rag_metrics.py`(新), `eval/metrics.py`, `eval/benchmarks/loader.py`, `docs/interfaces.md`, `app/main.py`, `tests/test_data_structures.py` | P6 |
| 2026-06-12           | 文档               | Meeting 1 问题对齐 A+B 立场(Q1 推荐 A+B、Q2 system 含 RAG、Q5 数据需 gold answer、NIAH finding/取舍)                                                           | `docs/meeting-1-questions.md` | P6 |
| 2026-06-11           | 契约               | 锁定共享检索契约(Retriever Protocol + RetrievedChunk)                                                                                                 | `src/retrieval/base.py`, `docs/interfaces.md` | P6 |
| _(往前的提交见 `git log`)_ |                  |                                                                                                                                               | |    |

---

## Meeting 决议存档

- **Meeting 1(2026-06-11/12,IBM = Bharat):** A+B 确认 ✅;交付物 = 超越 report 的 MVP ✅;
  数据 = 公开数据集、不碰企业数据/法律 ✅;NIAH = 次要(RAG 是 foundation)✅;
  **行动项:** 给 Bharat 发架构图(infra + architecture 两视角)、发会议 transcript、一周内全队对齐算力环境。
