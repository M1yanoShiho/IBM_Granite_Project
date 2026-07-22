# Selector 相关工作对比 — 报告素材

**日期:** 2026-07-20(导师会后反馈的六篇文献,已逐篇核实真实存在)

**用途边界:** 本文件是报告写作的**素材与对比要点**,不是可直接粘贴的报告文本。
按 Bristol AI-use policy,个人报告必须是本人行文;此处内容仅供组织论证结构与核对事实。
所有 arXiv ID 已于 2026-07-20 在线核实。

## 我们的定位(一句话)

在 retrieval 与 generation 之间做 pre-generation 证据仲裁,但把踢留判据约束为
**池内相对对照**(整数票差 + 竞争簇存在性),不使用任何绝对信誉分数阈值;
孤源结构性不可踢,平局双留如实暴露,失效方向为闭嘴。

## 1. ArbGraph(arXiv 2604.18362, 2026-04)——必须正面对比

**它做什么:** 检索文档原子化为 claims,建 conflict-aware evidence graph
(support/contradiction 边,LLM 仲裁器判边,confidence 过 τ_conf 才建边);
intensity 驱动的迭代仲裁传播 credibility(logit 空间,冲突胜者 +η、败者 −η,η=0.8);
最终按 **σ(s) ≥ τ_accept=0.3 的全局绝对阈值**过滤 claim。长文生成任务
(LongFact、RAGChecker),指标为 Fact Recall / Information Density / Faithfulness /
Noise Sensitivity。

**撞点:** 同一管道槽位(pre-generation arbitration)、同一图词汇
(claim 节点 + support/contradiction 边)、同一目标(压制不可靠证据)。

**差异(报告对比轴):**

| 维度 | ArbGraph | 本工作 |
|---|---|---|
| 踢留判据 | 连续 credibility 分 + 全局绝对阈值 τ_accept | 池内对照条件:竞争簇存在 + 整数票差 ≥ margin;门内无分数 |
| 跨域校准 | τ_conf/τ_accept 在其评测分布上定,迁移未验证 | 判据为同池无量纲整数差,不携带跨域刻度(动机:V1 在 ContractNLI 迁移 −0.134) |
| 孤源保护 | 无显式机制 | 结构性:无竞争簇则条件不成立,不可踢 |
| 平局 | 无显式处理 | 显式双留,冲突以原文暴露,拒答归 generator |
| 失效方向 | 阈值错位→过杀/漏杀均可能 | 投票碎裂→票差缩小→门闭嘴(退化为纯重排) |
| 边检测评估 | 200 对人工标注,96%(Table 4) | official aliases + deterministic mutation log,零人工标注(spec §12;动机:judge-kappa≈0) |
| 任务形态 | long-form 生成 | single-answer factoid needle-finding + 预注册双 Gate(harm↓ + recall 非劣 −0.01) |

**引用姿势:** 最近邻;承认其先行提出同槽位图仲裁,差异化落在判据类型
(绝对 vs 相对)、安全性质(孤源/平局/失效方向)与评估协议(预注册 Gate、零人工标注)。

## 2. RA-RAG(arXiv 2410.22954)

**它做什么:** 跨 query 离线迭代估计**来源级**可靠性先验,检索时取 top-κ 可靠来源,
答案聚合用加权多数投票(WMV)。

**差异:** 它是持久的 source-level 信誉档案 + 答案级聚合;我们是 query-local 池内
passage 级对照、无来源档案。其 source reliability 即我们"信号毕业制"(spec §8)中
排队的"来源权威性"信号的一种实现——未来毕业候选的参考,而非竞争者。
另:其设定要求语料具有来源身份结构;我们的 pre-chunked 语料无此结构。

## 3. MADAM-RAG / RAMDocs(arXiv 2504.13079)

**它做什么:** 冲突在**生成时**经多智能体辩论解决(每个 agent 代表一份文档,
聚合器裁决);发布 RAMDocs(ambiguity + misinformation + noise)。

**差异:** 我们在**选择时**用确定性规则,LLM-light;其强项是合法多答案歧义,
而我们的门在多答案形态明确关闭(spec §1/§13)——互补边界,报告可写成
"selection-time 确定性仲裁 vs generation-time 辩论仲裁"。RAMDocs 已列入
V2 计划 Block 4 的外部方向验证集。

## 4. HiREC(arXiv 2505.20368, ACL Findings 2025)

**它做什么:** 金融 SEC 文件的层级检索 + evidence curation(去无关、必要时生成
补充查询补检索);对付标准化文件的 boilerplate 近重复;LOFin benchmark。

**差异:** 其 dedup 是**文本近重复**检测,目的是省预算防混淆;我们的独立性修正是
**provenance 去重**(document_id/SAME_SOURCE),目的是防互证票虚增。
其 curation 以 relevance 判无关并过滤;我们按排除清单结论拒绝以相关性做剔除判据,
无关项只沉底不硬踢。金融域按团队约定不进 Selector 工作。

## 5. kapa.ai 生产剪枝博客(How we taught a small LLM to throw away 68% of our RAG context)

**它做什么:** 生产环境小模型 listwise 上下文剪枝:砍 68% chunk、保 96% recall、
省 34% 单查询成本;剪枝器成本从节省中自付。

**差异(脚注级引用):** 目标函数是成本,风险预算是 4% 的题丢失所需 chunk——
该误杀率在我们的 Required Recall 非劣门(−0.01)下直接不通过。
作为"成本导向剪枝 vs 安全导向门"的风险预算对照,说明为什么 harm-targeting
过滤需要不同的设计与协议。

## 6. CUE-R(arXiv 2604.05467, 2026-04)

**它做什么:** 对单条证据做 REMOVE/REPLACE/DUPLICATE 干预,沿 correctness /
grounding / confidence 三轴 + trace divergence 测量每条证据的操作性效用;
提出证据角色分类学(HotpotQA/2Wiki)。

**引用姿势(采纳式,不主张 novelty):** 我们的三检查点验证中"答案是否真依赖证据"
的干预检查与 CUE-R 思想同构,报告表述为采纳 CUE-R 式干预、嵌入对抗性
needle 场景作为系统验收协议的一部分。novelty 主张不落在验证方法学上,
落在门的判据设计上。

## 报告落点备忘

1. Related work 单独一段正面对比 ArbGraph(导师明确要求),用上表六轴;
2. 组件评估(spec §12 增补)执行后,结果与 ArbGraph Table 4 并排呈现方法学对照;
3. 六篇全部入引用;CUE-R 采纳式,kapa.ai 脚注级;
4. 叙事连续性:v1 门用图词汇呈现(答案簇=claim 节点,跨簇分歧=conflict 边的
   确定性代理,document_id 合票=duplicate 边最小实现),保住 slide 3 → 实现 → V2
   Graph 的一条线。
