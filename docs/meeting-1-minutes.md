# Meeting 1 — Minutes (IBM × Team)

- **日期:** 2026-06-12
- **与会:** IBM(Bharat 等) · 项目组(毛威凯 P6 主讲,Frida 及其他成员)
- **目的:** 介绍项目进展,敲定核心范围/交付物/数据等开放决策
- **配套:** [meeting-1-questions.md](meeting-1-questions.md)(会前问题) · [project-brief.md](project-brief.md)

---

## 一、决议(Decisions)

| # | 决议 | 说明 |
| --- | --- | --- |
| D1 | **核心方向 = A+B(检索 + RAG)** ✅ | Bharat 明确 "go ahead with A plus B"。忠于 brief(semantic search 与 RAG 并列)。 |
| D2 | **交付物 = 超越 research report 的可用 MVP** ✅ | 不只是研究报告/热力图;要做成能用的系统/工具。到 9/4 交付一个 MVP,作为后续扩展的 foundation。 |
| D3 | **数据 = 公开数据集,不碰企业数据** ✅ | 已签的 open-source 协议够用,**不换商业协议**。企业数据有法律/审批负担("I don't want this group to get into legality")。先用公开数据证明 novel 的点,之后作为 pilot 再推向企业。 |
| D4 | **两个目标不要混(research vs enterprise delivery)** ✅ | 先在公开数据上做出 robust + novel 的成果;企业落地是后续,别在现阶段交叉消耗时间。 |
| D5 | **数据格式/PDF 不是问题** ✅ | 应有一条 data transformation pipeline 把任意格式转成 RAG 可吃的;RAG 本身不挑格式。不必纠结 PDF。 |
| D6 | **NIAH = 次要诊断,RAG 是 foundation** ✅ | Bharat 框架:整个数据集 = haystack,建好 RAG 就是"为将来找针打地基";"needle" 是 marketing/大图。RAG 主、NIAH 次。 |
| D7 | **Foundation 已部分就位** ✅ | 两类 foundation:① 代码/框架(GitHub repo,已有)② 基建(GPU/HPC,刚批)。9/4 交付的 MVP 将成为扩展(企业或论文)的 foundation。 |

## 二、讨论摘要

- **范围:** proposal 原把检索设为主、RAG 次要;重读 brief 后提出提为 A+B,IBM 认可。NIAH 因 granite-4.1-8B 在 32k 无退化(此前"失败"是配置问题),作为次要诊断。
- **交付形态:** IBM 要"most viable product"思路——先展示大图,落地最小可用产品;9/4 的 MVP 是后续 scaling 的地基。
- **数据之争:** 组里曾考虑换商业协议拿企业数据(担心当前 RAG 是 "toy")。IBM 明确否决:用公开数据(如 SciFact + 金融数据集),避开法律,先证明研究价值。
- **算力:** 可选 Blue Pebble(Bristol HPC,本周批)/ RunPod(商用)/ Kaggle 免费 GPU(Bharat 推荐,覆盖优于 Colab)。

## 三、行动项(Action Items)

| # | 行动 | 负责 | 期限 |
| --- | --- | --- | --- |
| A1 | **给 Bharat 发架构图** —— 两个视角:① infrastructure(CPU/GPU 如何分工、评测在 GPU 上怎么跑)② architecture(数据如何喂进 RAG) | P6 | 下次会前/尽快 |
| A2 | 发会议 **transcript** 给 Bharat | P6 | 会后 |
| A3 | **一周内全队对齐** HF / GPU / Ollama\|vLLM 环境 | 全队 | ~2026-06-19 |
| A4 | 之后按 component-by-component 细化实施计划 | 全队 | 随后 |

---

> 下一次与 IBM 的沟通围绕 A1 的架构图展开,逐组件确认实现计划。
