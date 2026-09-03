# Evidence RAG Full-Stack Demo / 全栈演示

## 中文

这是一个带证据展示的本地问答演示。固定流程为：

```text
Hybrid RRF → NLI 接口（TopK10 fallback）→ Ollama Granite 4.1 3B
```

当前演示不加载 NLI 权重，也不生成虚假的 NLI 分数；页面中的 `P / H / Risk` 显示为
`N/A`。问题、答案和引用证据保存在浏览器本地历史记录中。

### 技术栈与功能

- 后端：Python、FastAPI、SSE 流式响应。
- 前端：Next.js、React、TypeScript、Tailwind CSS。
- 检索：BM25 与 Granite Dense Embedding，通过 RRF 融合排名。
- 生成：Ollama 托管的 IBM Granite 4.1 3B。
- 功能：本地文档问答、流式回答、引用跳转、证据面板和浏览器本地历史。

### 为什么实验中的 Generator 没有全部出现在用户层

研究代码包含 Direct Granite、GR-C LoRA、验证/修复流程和多个消融组。这些组件用于受控
实验比较，不等于都适合成为用户可切换的产品选项。部分方法需要未公开的 LoRA adapter、
TRUE T5-XXL 等大型权重、冻结数据和 HPC 环境；实验也没有证明 GR-C 在最终答案指标上稳定
优于 Direct Generator。因此，用户层固定使用较轻的 Ollama Granite，隐藏实验开关，保留
引用和证据展示等可解释功能。研究实现仍保留在代码中，供复现与后续部署使用。

### TopK 与 NLI 的关系

TopK 是无模型基线：按 Retriever 排名保留前 10 条证据。NLI Protect–Harm 则在 TopK10
候选上计算 protect/harm 风险，最多删除两条高风险证据。实验中 NLI 降低了特定压力测试的
有害证据，但没有证明最终答案质量显著提升，因此部署决策为 `KEEP_TOPK10`。

当前 Header 保留 `NLI`，表示接口和系统设计仍被保留；实际运行的是 TopK10 fallback。
只有配置完整的 DeBERTa 基础模型和训练后的 Selector checkpoint 后，才会产生真实的
`P / H / Risk` 分数。不得把 fallback 的 `N/A` 当作 NLI 推理结果。

### 权重、HPC 与本地延迟

- Hybrid Retriever 首次运行需要下载 Granite Embedding 权重。
- Ollama 需要单独拉取 `ibm/granite4.1:3b`。
- 真实 NLI 还需要 DeBERTa 基础权重和约 738 MB 的训练后 Selector checkpoint；当前 demo
  不下载也不使用该 checkpoint。
- 完整实验（真实 NLI、GR-C、TRUE verifier、批量评测）推荐在 HPC/GPU 环境运行，模型和
  数据留在 HPC 存储，只同步代码与允许发布的小型结果。
- 本地第一次查询包含下载、模型加载和索引构建，会明显较慢；后续查询会复用缓存，但仍受
  本地硬件性能、内存、文档数量和生成长度影响。页面出现短暂停顿不代表 NLI 正在运行。

### 启动

先启动 Ollama 并准备 Granite：

```bash
ollama serve
ollama pull ibm/granite4.1:3b
```

在另一个终端运行：

```bash
cd IBM_Granite_Project

export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=ibm/granite4.1:3b
export EVIDENCE_RAG_CORPUS=runs/reference-baseline
export EVIDENCE_RAG_DEMO_NLI_FALLBACK=1

.venv-stable/bin/python -m evidence_rag.ui.server
```

打开 <http://127.0.0.1:8000>。

测试问题：

```text
What percentage of the factory's electricity is supplied by Northstar's rooftop solar panels?
```

预期答案为 `40%`，并引用右侧证据。

## English

This is a local question-answering demo with visible supporting evidence. The fixed pipeline is:

```text
Hybrid RRF → NLI interface (TopK10 fallback) → Ollama Granite 4.1 3B
```

The demo does not load NLI weights or fabricate NLI scores. The UI reports `P / H / Risk` as
`N/A`. Questions, answers, and cited evidence are stored in browser-local history.

### Technology and features

- Backend: Python, FastAPI, and Server-Sent Events (SSE).
- Frontend: Next.js, React, TypeScript, and Tailwind CSS.
- Retrieval: BM25 plus Granite dense embeddings, fused with RRF.
- Generation: IBM Granite 4.1 3B served by Ollama.
- Features: local-document QA, streaming answers, clickable citations, an evidence panel, and
  browser-local conversation history.

### Why many experimental Generators are not exposed in the UI

The research code includes Direct Granite, GR-C LoRA, verification/repair pipelines, and multiple
ablation arms. These are controlled experimental conditions, not necessarily suitable user-facing
options. Some require unpublished LoRA adapters, the large TRUE T5-XXL model, frozen datasets, and
HPC infrastructure. The experiments also did not establish that GR-C consistently improves final
answer quality over the Direct Generator. The UI therefore uses a fixed, lightweight Ollama
Granite path while retaining citations and evidence visibility. Research implementations remain in
the repository for reproduction and future deployment.

### Relationship between TopK and NLI

TopK is the model-free baseline: it keeps the ten highest-ranked Retriever results. NLI
Protect–Harm operates on those TopK10 candidates, estimates protect/harm risk, and may remove at
most two high-risk items. NLI reduced harmful evidence in a dedicated stress test, but did not
demonstrate a significant final-answer improvement; the deployment decision was `KEEP_TOPK10`.

The Header retains `NLI` to show that the interface and system design remain available. This demo
actually runs the explicit TopK10 fallback. Real `P / H / Risk` values require both the DeBERTa
base model and the trained Selector checkpoint. `N/A` fallback values must not be interpreted as
NLI inference.

### Weights, HPC, and local latency

- Hybrid retrieval downloads Granite Embedding weights on first use.
- Ollama separately requires `ibm/granite4.1:3b`.
- Real NLI additionally requires the DeBERTa base weights and the approximately 738 MB trained
  Selector checkpoint; this demo does not download or use that checkpoint.
- Full experiments with real NLI, GR-C, TRUE verification, and batch evaluation are best run on an
  HPC/GPU system. Keep models and datasets in HPC storage and synchronize only code and permitted
  compact results.
- The first local query is slower because of downloads, model loading, and index construction.
  Later queries reuse caches, but latency still depends on hardware, memory, corpus size, and answer
  length. A local pause does not mean NLI inference is running.

### Run

Start Ollama and prepare Granite:

```bash
ollama serve
ollama pull ibm/granite4.1:3b
```

In another terminal:

```bash
cd IBM_Granite_Project

export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=ibm/granite4.1:3b
export EVIDENCE_RAG_CORPUS=runs/reference-baseline
export EVIDENCE_RAG_DEMO_NLI_FALLBACK=1

.venv-stable/bin/python -m evidence_rag.ui.server
```

Open <http://127.0.0.1:8000>.

Sample question:

```text
What percentage of the factory's electricity is supplied by Northstar's rooftop solar panels?
```

The expected answer is `40%`, with a citation linked to the evidence panel.
