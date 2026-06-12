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

## 5. 当前状态与下一步

- **已完成:** 4 契约;A+B 架构改造;HPC 生成层打通(3B 验证);Meeting 1 准备;HPC 脚本 + 部署文档。
- **进行中:** `run_benchmark` 编排(对 mock retriever + synthetic mini-benchmark 出 CSV)。
- **下一步:** `build_run` + chunk→doc 聚合;给 Bharat 的架构图(CPU/GPU 分工 + 数据如何喂进 RAG);8B 冒烟验证;`citations.py`。

> 后续个人报告可深挖的分析点:chunk→doc 聚合策略(max-pool vs mean)对指标的影响;集成层如何用 mock(FakeRetriever + synthetic benchmark)解耦并行开发。
