# 个人开发文档 — P6 毛威凯(Weikai Mao)

> 我个人的开发记录,服务于个人报告(individual report)。团队层面的进度看
> [dev-log.md](dev-log.md);这里按 **做了什么 / 为什么这么做 / 遇到的问题怎么解决**
> 展开,记录决策与踩坑,方便后面写 critical analysis。
>
> **角色:** 集成(integration)+ explainability + **接口负责人(interface owner)**。
> **负责模块:** `eval/run_benchmark.py`、`src/explainability/citations.py`、跨模块契约。

---

## 1. 接口契约与架构(interface owner)

作为接口负责人,我先把**跨模块的"合同"锁进代码**,让 7 个人能对着稳定接口并行开发、最后拼得上。

- **契约 1 — 统一检索器接口**:`src/retrieval/base.py` 的 `Retriever` Protocol + `RetrievedChunk` dataclass。三种检索器(Granite dense / BM25 / sentence-transformers baseline)都实现 `retrieve(query)->List[RetrievedChunk]`,评测框架可无差别替换。
- **契约 2 — Run 格式**:`Run = {qid:{doc_id:score}}` / `Qrels = {qid:{doc_id:rel}}`,作为"检索结果 → 指标"的桥梁。
- **契约 3 — chunk→doc 聚合**:max-pool(一个 doc 的分 = 命中 chunk 的最高分),因为检索是 chunk 级、qrels 是 doc 级。
- **契约 4 — RAG I/O(后加)**:`RAGPipeline(retriever, llm)` → `RAGResult(answer, retrieved_chunks)`。

**设计理由:** 7 人协作最大的风险是各写各的、最后无法集成。用结构化 `Protocol` + 类型别名把接口固化在代码里(而非只写文档),配 `@runtime_checkable` 可在测试里断言实现,做到"契约即代码"。文档侧在 `docs/interfaces.md` 维护导航 + 变更规则。

---

## 2. RAG 提升为 A+B co-headline 的架构改造(2026-06-12)

Meeting 1 与 IBM 确认把 **RAG 从次要提升为与检索并列(A+B)**,以忠于 brief(brief 把 semantic search 与 RAG 并列)。我据此重构了 RAG 这条线,**没有引入外部框架、没有新建 repo**,只在契约体系内调整:

| 改动 | 内容 |
| --- | --- |
| `src/rag_pipeline.py` 重写 | `RAGPipeline` 改为**复用契约 1 的 `Retriever`**(retrieve→拼 prompt→`LLMClient.generate`),不再自带重复的检索栈;`RAGResult.retrieved_chunks` 升级为 `List[RetrievedChunk]`(带 doc_id/score,供 citations 与 context-precision)。glue 已实现并用 mock 验证。 |
| 新增 `eval/run_rag.py` | RAG 主线评测入口,对标 `run_benchmark.py`。 |
| 新增 `eval/rag_metrics.py` | RAG 答案质量指标的权威家(answer correctness / context precision / faithfulness),从 NIAH 的 `metrics.py` 中独立出来,后者 re-export 保后向兼容。 |
| `eval/benchmarks/loader.py` | `BenchmarkData` 加可选字段 `answers`(RAG 打分需 gold answer,SciFact 无,关联 Q5)。 |
| `docs/interfaces.md` | 新增契约 4(RAG I/O)。 |
| `app/main.py` | demo 文案改为 RAG 口径(检索→生成→展示引用)。 |

**关键决策:不引入外部 RAG 框架(LangChain / LlamaIndex 等)。** 理由:现有契约化架构刚立起来,外部框架自带一套抽象,引入等于推翻契约、打乱并行分工,且对 MSc 评分无增益(novelty 来自地基之上的分析,不是底层用了谁家的库)。验证:`pytest -q` 保持 6 passed / 2 xfailed,改动未破坏基线。

---

## 3. HPC 部署:在 BluePebble 上跑通 Granite 生成层(2026-06-12)

RAG/生成层需要 GPU。我负责把 Granite 生成模型在学校超算 **BluePebble** 上从零跑通,并沉淀成可复现的脚本 + 文档,供需要 GPU 的组员复用。

**交付物:**
- `scripts/smoke_8b.py` + `scripts/smoke_8b.slurm` — 冒烟脚本/作业(验证 `llm_client` 对真实模型)。
- `scripts/run_rag.slurm` — 正式 RAG 作业(8B + RTX 3090)。
- `docs/hpc-deployment.md` — 连接 / VPN / 环境 / 下载 / 提交 全流程文档。
- 把 repo 设为 public(契合 Apache-2.0 + ILO 开源协议),公开前扫描确认无密钥泄露(`.env` 未跟踪、历史无真实 token)。

**确认的关键配置:** account `coms039904`、QOS `normal`、partition `gpu`/`gpu_short`;GPU 有 RTX 2080 Ti(11G)/3090(24G)/V100/A100-MIG(40G);驱动 CUDA 12.4。

**结果:** granite-4.1-3b 在 GPU 节点加载并生成出正确答案(`Smoke test OK.`),生成层打通。

### 踩坑与解决(report 素材)
1. **交互式 `srun --pty` 被禁**("launch failed: Unspecified error")→ 改用 `sbatch` 批处理提交。
2. **登录节点无 `python`** → `module load languages/python/3.12.3`(避开默认 3.14,wheel 太新不兼容),并把 module load 写进 Slurm 脚本(计算节点同样需要)。
3. **`ModuleNotFoundError: No module named 'src'`** → 脚本以路径方式运行时根目录不在搜索路径;`PYTHONPATH=.` 修复。
4. **`torch.float8_e8m0fnu` AttributeError → 无法加载 GraniteForCausalLM**:`requirements.txt` 钉的 `transformers 5.x` 需要 torch ≥2.7(仅 cu126/cu128,超过 12.4 驱动),而 cu121 的 torch 封顶 2.5.1。**修法:HPC 上用 `transformers 4.x`(验证组合 torch 2.5.1+cu121 + transformers 4.57)**;不改共享 `requirements.txt`(笔记本 torch 2.12+cpu 有该 dtype,5.x 正常),HPC 专用修复记入 `hpc-deployment.md`。
5. **`huggingface-cli` 已废弃** → 改用 `hf download`。

**结论性认知:** torch 必须装 CUDA 版(别被 `requirements.lock` 的 `+cpu` 覆盖);权重在登录节点预下载、作业内 `HF_HUB_OFFLINE=1`(计算节点无网);缓存 `HF_HOME` 放 `/user/work`;模型权重应**全队共享一份缓存**,不重复下载。

---

## 4. Meeting 1 准备与 scope 对齐

- 重读 brief,发现 proposal 的"检索为主、RAG 次要"是团队的收窄而非 brief 原意;推动把 RAG 提为 A+B,并在 `meeting-1-questions.md` 重写 Q1/Q2/Q5,把"要 IBM 拍板"的点(headline、数据需 gold answer、NIAH 取舍)显式化。
- 会上确认:A+B ✅、交付物=超越 report 的 MVP ✅、数据=公开数据集(不碰企业数据/法律)✅、NIAH=次要(RAG 是 foundation)✅。

---

## 5. 检索 ablation 的配置打通(A0,2026-06-19)

检索主线的编排(`run_benchmark`:`build_run` max-pool 聚合、`evaluate_one`、`write_results_csv`、`_build_retrievers`、索引缓存、CLI)此前已实现并出了第一张 SciFact 表(granite_dense nDCG@10 0.767 > BM25 0.636 ≈ ST 0.641)。但那只是"一个模型、一套配置、一个数据集、一次 run"。要把单点结果做成稳健结论,得在**系统层**做 ablation——把切分 / 聚合 / 模型这些旋钮接到 CLI,让队友不改代码就能扫,且结果能汇总分析。这就是 A0,也是后续 A 线(尤佳希 A1 扫 Granite 变体、吴泽楠 A2 扫 chunk/pooling)的前置缝。

### 做了什么(`eval/run_benchmark.py`,test-first,9 个新测试,全套 49 passed / 1 xfailed)
- `BenchmarkConfig` 新增 `chunk_size` / `chunk_overlap` / `pooling` / `embedding_model_id` / `append`,默认值全部还原既有行为。
- `build_run(..., pooling)`:`max`(契约 3)或 `mean`(对一个 doc 被检索到的 chunk 取均值),透传 `evaluate_one` → `run`。
- `_build_retrievers`:把 chunk 参数传进 `chunk_document`,把模型覆盖传进 `Embedder`。
- 新增 `_cache_key(config, name)`:把 chunk size/overlap + 模型折进索引缓存键。
- `write_results_csv(..., config_columns, append)`:给行打配置标签 + 追加写;`run` 在 `--append`(sweep)模式下给每行带上本次配置,累积进一张主 CSV。
- CLI 新增 `--chunk-size --overlap --pooling --embedding-model-id --append`;`docs/interfaces.md` 补 Contract 6b(ablation 列,给 P7)。

### 为什么这么做
- **加法式、向后兼容**:每个旋钮默认还原现状,不带 `--append` 的普通 run 写出的 CSV 和原来逐字节一致——headline 表和 P7 不受影响,新功能 opt-in。
- **一张主 CSV + 配置列 + append**:sweep 就是循环换 config 反复调 `run()`,每行自带配置,P7 一张表按 `chunk_size` / `pooling` / `model` 分组就能分析,而不是几十个零散文件。

### 踩坑 / 关键决策(report 素材)
1. **索引缓存键的"串味"bug(设计期发现并修)**:原缓存键只按 `dataset__retriever`。一旦带 `--cache-dir` 做 chunk-size sweep,第二个配置会命中第一个配置缓存下来的索引——静默用错向量,整轮 ablation 作废。修法:把 chunk size/overlap + 模型 slug 折进缓存键(抽成纯函数 `_cache_key`),并单测其"不同配置→不同键、相同配置→同键"。是评测严谨性的一个典型点。
2. **mean-pool 的语义**:只对一个 doc *被检索到的* chunk 取均值(没被检索的 chunk 看不到分数,无法纳入)。这个 ablation 的作用大概率是反过来证明 max-pool 更好,从而给当初选 max(契约 3)提供实证。
3. **用行为而非 mock 验证接线**:构造一个 doc,使它的 max-pool 与 mean-pool 分数会与竞争 doc 排名翻转,于是 `run()` 写出的 mrr 在两种 pooling 下不同(1.0 vs 0.5)——用**指标本身**证明旋钮被正确透传 run→evaluate_one→build_run,而不是断言某个 mock 被调用过。

### 状态
代码 + 测试 + 文档完成,在 `week3` 分支未提交(按团队流程应走 `week3-p6-ablation-config` → PR → 集成分支)。

---

## 6. 当前状态与下一步

- **已完成:** 4 契约;A+B 架构改造;HPC 生成层打通(3B + **8B 验证**,8B 冒烟 2026-06-23 `Smoke test OK.`);Meeting 1 准备;`run_benchmark` 检索编排 + 第一张 SciFact 表;A0 检索 ablation 配置打通;**MVP 接缝**(`retrieval/factory.py` + `rag_app.py`,`app/main.py` 接真 RAG 管线,TDD)。
- **进行中 / 下一步:** B3 剩 `run_rag.slurm` 全量 RAG(待 `run_rag.py` 实现);app 本机 3B 跑通确认;`citations.py` 的 `attribute_answer`(Phase 2);给 Bharat 的架构图(CPU/GPU 分工 + 数据如何喂进 RAG)。

> 后续个人报告可深挖的分析点:chunk→doc 聚合策略(max-pool vs mean)对指标的影响;索引缓存键的串味 bug 与修复(评测严谨性);集成层如何用 mock(FakeRetriever + synthetic benchmark)解耦并行开发;契约优先 + 依赖注入如何支撑 7 人并行。
