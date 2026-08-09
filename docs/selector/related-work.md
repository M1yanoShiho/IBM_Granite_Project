# Selector 相关工作与备选路径 — 2026-08-09 检索

**这份文件的用途:** (1) 报告 Related Work 一节的引文来源;(2) **若 §3.8 微调不过 .85 的备选路径清单**,
写在任何训练读数存在之前,故属预注册材料而非事后找补;(3) 定位主张的外部核对。

**本次检索由四个并行 agent 执行,覆盖:验证器模型 SOTA、知识冲突/RAG 投毒防御、
GitHub 与 HF 可用产物、以及针对本项目已诊断失败模式的训练技术。**

## 使用规则(照抄本项目既有纪律)

- **每条标注核验状态。** `[已核验]` = 本次实际 fetch 过页面;`[未核验]` = 仅出现在搜索结果或摘要片段。
  **`[未核验]` 的条目在引用前必须自行 fetch** —— 按 WS-13 规则,2026 年的 arXiv ID 无法凭记忆确认,
  **编造的引文比没有引文更糟**。
- 数字一律连同其基准名一起记;跨基准比较必须显式说明。
- 本文件由 AI 辅助检索生成。**按 Bristol AI 使用政策,报告正文须本人行文;本文件是证据索引,不是可粘贴的文本。**

---

## 0. 三条改变判断的发现(先读这个)

**(1) 「检索阶段跨源答案一致性」这个槽位已被占据。** RADAR(2026-05)与 ReliabilityRAG(NeurIPS 2025)
都在做逐文档抽答案 + 跨文档一致性选择。**不引用并区分,审稿人一定会点。**

**(2) 但「按独立来源(parent page)计票」无人占据。** 四路检索均未找到。上述所有方法都把检索到的文档
当作**可交换的票**,因此都能被"把同一反事实复制到同一母页面的 k 个 chunk"平凡击破。
**我们的 `collision_rate 0.931`(R001b(a))正是这个缺陷普遍存在的实测证据。**
⇒ **定位应从"投票"改为"独立来源计票"。**

**(3) 现代验证器在最小编辑负例上从未被测过,而有证据表明它们会崩。** 见 §3。
这既解释了我们的 .794,也使我们的探针结果本身成为一个文献空白里的新数据点。

---

## 1. 占据同一槽位的工作(必须引用并区分)

| 方法 | 标识 | 阶段 | 机制 | 与本项目的差异 |
|---|---|---|---|---|
| **RADAR** | arXiv **2605.22041** [已核验] 2026-05-21 | 选择 | 逐文档孤立生成原子答案 → 答案间 NLI 蕴含/矛盾矩阵 → 特征向量中心性找共识集 → Max-Flow Min-Cut 解可靠子集 | **最接近的邻居。**聚合数学不同(中心性+最小割 vs 聚类计票);**不做来源去重**(已专门核查) |
| **ReliabilityRAG** | arXiv **2509.23519** [已核验] NeurIPS 2025 | 选择 | 文档图,边=NLI 判定的矛盾;加权**最大独立集**取"一致多数" | 带**可证明鲁棒性**:对手污染 ≤ k/5 时 (1−e^−O(k))-robust。可靠性权重用检索排名——**换成 parent-page 独立性正是我们的 delta** |
| **RA-RAG** | arXiv **2410.22954** [已核验] EMNLP 2025 | 源选择+聚合 | 跨源交叉核对估计**来源可靠性**,加权多数投票 | **源级而非文档级**;假定来源集已知且可靠性稳定。是"佐证投票作为防御信号"的引用先例 |
| **Astute RAG** | arXiv **2410.07176** [已核验] ACL 2025 | 生成 | 提示词内做答案一致性聚类("一致的聚在一起,冲突的分开"),再比较组可靠性 | 靠 LLM 提示而非关系模型;**不计来源数** |
| **RobustRAG** | arXiv **2405.15556** [已核验] ICML 2024 | 生成 | isolate-then-aggregate,逐段独立回答后安全聚合;可证明鲁棒 | 整个"逐文档再聚合"家族的祖先,**必引** |
| **ArgRAG** | arXiv **2508.20131** [已核验] NeSy 2025 | 选择/推理 | 文档标 support/contradict/irrelevant,构建定量双极论辩框架 | **等权闭类下退化为多数投票** —— 我们的计票规则是其特例。若平权计票不够,这是自然的推广方向 |

**RADAR 的一个消融对我们价值最高:** 它换了 DeBERTa-v3 / BART / ModernBERT 三个 NLI 骨干,
报告 Acc 与 ASR 只有 "minor changes",称对 NLI 选择"low sensitivity"。
⇒ **在继续投资微调之前,先在我们自己的数据上复现这个消融**(我们已有三臂 dump,成本极低):
- 若成立 ⇒ 关系模型从来不是承重墙,瓶颈在别处(答案抽取或聚类)。
- 若不成立 ⇒ **该差异本身是发现**:最小编辑冲突比 RADAR 测过的冲突更难,这为 §3.8 的微调路线提供正当性。

**两个方向都出结论,这是目前性价比最高的单个实验。**

---

## 2. 对 1-token 编辑构造性失明的防御(支撑我们的定位)

**没有任何论文写下这条批评** —— 这是可主张的观察。以下方法各自依赖一个最小编辑反事实**不具备**的性质。

| 方法 | 标识 | 依赖的信号 | 为何漏掉 1-token 编辑 |
|---|---|---|---|
| **TrustRAG** | arXiv 2501.00879 [已核验] | embedding 上 K-means 找可疑稠密簇 | 反事实与真文档落在**同一个簇**,同时标记或同时放过 |
| **RAGDefender** | arXiv 2511.01268 [已核验] ACSAC 2025 | 层次凝聚聚类 + TF-IDF 共享关键词 | 同上。其自身消融:凝聚 ASR 0.06 / k-means 0.16 / DBSCAN 0.44——**全部在 embedding 空间** |
| **PRA-RAG** | arXiv 2607.00012 [已核验] | embedding 空间几何结构找鲁棒子集 | 同一失明。报告 ASR→1%,accuracy 71% |
| **RAGuard**(检测框架) | arXiv 2510.25025 [已核验] IEEE BigData 2025 | chunk 级困惑度过滤 | 人写文档改一个实体,**困惑度正常** |
| **GMTP** | arXiv 2507.18202 [已核验] ACL Findings 2025 | 掩码后 MLM 概率异常低 = 注入 token | 换成的**合理**实体 MLM 概率天然高 |
| **ProGRank** | arXiv 2603.22934 [已核验] ECML PKDD 2026 | 扰动下的表示一致性/离散度 | 流畅的反事实在扰动下**是稳定的**,看起来干净 |
| **FilterRAG** | arXiv 2508.02835 [已核验(题录);机制仅据摘要] | query-answer 词共现密度 | 反事实不含注入的 query 文本 |

⚠ **命名陷阱:三个不同系统都叫 "RAGuard"** —— (a) 基准 arXiv 2502.16101;(b) 检测框架 2510.25025;
(c) 分层防御 2607.26339。**不加限定地写 "RAGuard" 会被读成另一个。**

---

## 3. 最小编辑负例上的证据:regime 决定一切

**SummEdits**(arXiv **2305.14540** [已核验],EMNLP 2023 `2023.emnlp-main.600`)是文献中最接近我们孪生探针的构造
—— 6348 样本,编辑类型 **实体修改 78%**、反义替换 48%、幻觉事实 22%、否定 18%(多标签)。

| 模型 | SummEdits BAcc | LLM-AggreFact 上的地位 |
|---|---:|---|
| QAFactEval | 65.7 | — |
| SummaC | 58.8 | — |
| DAE | 55.7 | — |
| ChatGPT | 71.3 | — |
| **GPT-4** | **82.4** | 与小专用模型仅差 1–2pp |
| 人类 | 90.9 | — |

**这个排序与 LLM-AggreFact 完全相反。** 在通用榜上能追平 GPT-4 的 sub-1B 专用检测器,
在最小编辑上塌到 56–66%,而 GPT-4 守住 82.4。SummEdits 编辑的是摘要侧、我们编辑的是段落侧,
但模型面对的判别问题同构。

⇒ **我们的 .794 大概率不是配置问题,是这一族在这个 regime 的真实水位。**
⇒ **且没有任何论文在 SummEdits 上评测过 MiniCheck / Bespoke-MiniCheck / AlignScore / FactCG /
Granite Guardian**(已专门多路搜索)。**现代验证器这一代从未在最小编辑负例上被测过。**

佐证:
- **SummExecEdit** arXiv **2412.13378** [已核验]:最佳模型 Claude 3 Opus 检测率 0.67;20+ LLM 中过半在该基准上失败 >30%。
- **arXiv 2604.10990** [已核验] (2026-04-13):现有基准"通过扰动单个显著元素构造不可行断言",
  模型靠**显著约束检查捷径**通过。⇒ 按其分类法,我们的孪生属**较易**的 regime,
  **失败因此是对被测模型的强负面信号,而非探针不公平**。
- **FaithBench** arXiv 2410.13210 [已核验] NAACL 2025:"即便最好的幻觉检测模型也只有接近 50% 的准确率"。
- **HallDetect** arXiv **2608.05823** [已核验]:紧凑编码器对分解后的原子断言做蕴含,
  **非对称聚合——一条被高置信否定的断言即否定整体**。该聚合形状比任何单阈值分类器更适合孪生门,
  且与模型选择正交。**值得作为决策规则借鉴。**
- **PARALLAX** arXiv **2605.17028** [已核验]:22 种检测方法 × 12 模型 × 6 语料;
  6 个语料中 4 个把答案泄漏进提示,文本相似度基线近乎满分,"多数既有基线在受控条件下接近随机"。
  *注意:该文针对隐状态幻觉检测,与文档-断言验证相邻但不同。*

---

## 4. 备选模型(若微调失败)—— 全部核验过 HF 文件清单

**硬约束:`torch<2.6`(CVE-2025-32434)⇒ 仅 `.bin` 的 repo 不可加载。以下 safetensors 一列逐个查过 API。**

### 干净可用(未在 VitaminC 上训练,可作评测臂)

| HF id | safetensors | 规模 | 许可 | 训练数据 | VitC / FEVER |
|---|:---:|---|---|---|---|
| `yaxili96/FactCG-DeBERTa-v3-Large` | **YES** | 0.4B | **MIT** | MiniCheck ANLI 子集 + C2D/D2C + CG2C 合成 | **干净 / 干净** |
| `cross-encoder/nli-deberta-v3-large` | **YES** | 435M | Apache-2.0 | SNLI+MNLI | 干净 / 干净 |
| `cross-encoder/nli-deberta-v3-base`(现基座) | **YES** | 184M | Apache-2.0 | SNLI+MNLI | 干净 / 干净 |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | **YES** | 435M | MIT | MNLI/ANLI/FEVER/LingNLI/WANLI | 干净 / **FEVER** |
| `bespokelabs/Bespoke-MiniCheck-7B` | YES(4 分片) | 7B | **CC-BY-NC-4.0** | MiniCheck 谱系 | 干净 / 干净 |
| `ibm-granite/granite-guardian-3.3-8b` | YES | 8B | **Apache-2.0** | IBM guardrail mix | 未核验 |
| `ibm-granite/granite-guardian-3.2-3b-a800m` | YES(2 分片) | 3B MoE/800M 活跃 | Apache-2.0 | 同上 | 未核验 |

### ⚠ VitaminC 污染,**不得作为评测臂**

`tals/albert-xlarge-vitaminc-mnli`(**我们现用的 albert 臂**)、`sileod/deberta-v3-base-tasksource-nli`、
`sileod/deberta-v3-large-tasksource-nli`、`tasksource/deberta-base-long-nli`、`tasksource/deberta-small-long-nli`
—— 全部带 `dataset:tals/vitaminc` 标签(HF API 实测)。

> **⇒ 对 R012e 的直接影响:** tasksource 族的 HF 元数据**直接标注了 `tals/vitaminc`**,
> 比 tracker R012e 追查的 `tasksource/bigbench` 子集路径更直接。**但这不替代 R012e** ——
> 元数据标签不等于训练清单核验,且 R012e 的 V2(去污染裁决)是另一个问题。
> **此发现应写入 R012e 的证据栏,并注意:`long-nli` 两个模型的污染只出现在元数据,model card 正文不提,
> 是易踩的坑。**

### 死路(逐个核验)

`lytang/MiniCheck-Flan-T5-Large` / `-DeBERTa-v3-Large` / `-RoBERTa-Large`(**均只有 `.bin`**,MIT)、
`tals/albert-base-vitaminc-fever`(`.bin`)、`google/t5_xxl_true_nli_mixture`(`.bin` + 11B + VitaminC 污染)、
`yzha/AlignScore`(只有 `.ckpt`)、`PatronusAI/...Lynx-8B`(safetensors 有但 CC-BY-NC + 8B)。

> **绕路存在且我们已用过:** 本地 `torch.load` → `safetensors.torch.save_file` 转换
> (transformers 的守卫只在其加载路径上,`torch.load` 本身在 torch<2.6 下仍可用)。
> **我们在 G 模块对 TRUE 做过这件事**(见 hpc-run-log G5,含值级验证)。
> ⚠ **但 A3 §11.1 明确论证过训练基座与验证器的不对称**(基座 provenance 流进交付物,
> 验证器两端由 hash 钉死)。**对 MiniCheck 做转换若用于评测臂,属验证器一侧,与该论证一致;
> 若用作训练基座,则触及 A3 的核心依据,须另开修订。**

### ⚠ 结构性警告

**FactCG / MiniCheck / HHEM / Granite Guardian / Lynx 全是二分类**(supported / not),
**无法区分 REFUTES 与 UNKNOWN**。图若需要有符号的边,二分类只是半个答案。
只有 NLI cross-encoder 是原生三类。**这条与 A2(§10)的三类范式直接相关。**

**明确排除:** `vectara/hallucination_evaluation_model`(HHEM-2.1-Open)——
AggreFact BAcc 76.55 但 **recall 68.48**,RAGTruth-Summ recall **31.86**,
**结构性偏精度,不可能到 .85 recall**。

---

## 5. 若继续微调:文献指向的杠杆(与我们的实测吻合)

**最重要的综合判断:风险 (a)(UNKNOWN 质量堆积)与形式敏感性不是两个问题,是同一个。**

Joshi & He(arXiv **2107.00753** [已核验] ACL 2022)给出机制:在最小编辑对上训练 ⇒ 模型学到
"表面微小差异 ⇒ 不支持";而 QA 式假设("the answer to Q is A")与证据的表面差异是**良性改述**,
于是该敏感性误触发、概率质量落到 NEI、`gold_supports_recall` 下降。
**他们开的药方是扰动多样性 —— 正是我们实测出的最大杠杆(形式,20–34pp)。**
另一独立负面结果:Huang, Liu & Bowman arXiv **2010.04762** [已核验](反事实增强不比等量原始数据泛化更好,
甚至可能有害)。

### 排序后的三条(附预注册句式)

**① 多形式假设增强** —— 每条训练 claim 用全部 K 种渲染各出一份,标签不变。
依据 UnifiedQA arXiv **2005.00700** [已核验];**纯数据侧,零代码、无阈值、不碰 dev**。
预期最差形式上 +10~25pp,孪生准确率大致中性(标签未变,反事实信号未被稀释)。

**② 训练期 logit adjustment** —— 在交叉熵内对 logit 加 `τ·log π_y`,**推理仍是纯 argmax**。
Menon et al. ICLR 2021,arXiv **2007.07314** [已核验]。
⚠ **关键的协议区分:事后 logit adjustment 是推理期决策边界移动,严格读就是阈值的别名,应被我们的无阈值条款排除;
同一修正放进训练损失则推理不变,协议干净。** 且 VitaminC 三类近平衡 ⇒ 按训练先验修正近乎无效,
必须用**预先声明的目标先验**。预期 +3~10pp recall,**但会以孪生准确率换**,故应在 ① 之后再用。

**③ Mukobara et al. 的 SR 损失** —— EACL 2024,arXiv **2403.08174** [已核验]。
仅对 SUP/REF logit 加互补概率罚项,使"真 SUP 被判 NEI"的惩罚更轻。
**其论文动机逐字就是我们的风险 (a)。**约 15 行。
⚠ 原文的 λ 在 dev 上调过 —— **我们必须先验固定(如 λ=1)并预注册**,否则违反 dev 保留条款。

**明确不要试:** label smoothing(arXiv 1906.02629 [已核验],对称平滑在 argmax 下决策不变,
是可预测的 null,浪费一个预注册名额)、focal loss(arXiv 1708.02002 [已核验],按难度不按类别加权,
无方向性 recall 效应)。

**备选(若上三条仍不过 .85):** 混入 QA 派生的 NLI 样本 —— Chen, Choi & Durrett
Findings of EMNLP 2021,arXiv **2104.08731** [已核验],其发现直击我们的风险 (b):
"只在 NLI 数据集上训练的模型无法单独作为 QA 的有效验证器"。生成机制可用 QACG
(arXiv **2105.14682** [已核验],**含 NEI 生成**)。

---

## 6. 必须写进 Limitations 的两条

**PURPOSE** — arXiv **2608.04756** [已核验],**2026-08-05,四天前**。
**专为击穿"检索后冲突消解类防御"设计的黑盒攻击**:不与共识正面矛盾,而是把注入写成
**最小化冲突的"更新"**,先抽取近似消解器参考的事实,再在其上锚定一个 pivot 事件。
**45 个设置中 35 个取得最高 ASR,平均 +9.7 ASP 点。**
⇒ 我们的方法正是它的目标类。**主动引用读作清醒,被指出读作漏洞。**
反面看:有人专门造攻击来绕过这类防御,本身即该类防御有效的证据。

**CRCP** — arXiv **2606.11265** [已核验]:多数既有投毒攻击**在 cross-encoder 重排后显著衰减**,
因为重排器偏好局部连贯段落。⇒ 该文认为重排器是对**优化型**投毒的意外防御。
**而我们的反事实恰是这个意外防御失效的情形(它们本就是局部连贯、含答案的文本)。**
这条**强化**我们"重排器按构造失败"的主张,应引为直接对话对象。

---

## 7. 数据集

| 名称 | 标识 | 为何相关 |
|---|---|---|
| **Faithfulness-QA** | arXiv **2604.25313** [已核验] | **99,094 样本,反事实实体替换**构建于 SQuAD/TriviaQA,76,953 实体库,8 类命名实体。**与我们的构造同法,规模现成。**框架是 context-vs-parametric 忠实性,非投毒威胁模型 |
| **Symmetric FEVER** | github.com/TalSchuster/FeverSymmetric [已核验] 53★ MIT | VitaminC 同作者,人工反事实 claim/evidence 对。小,适合鲁棒性探针 |
| **Counterfactually-Augmented SNLI** | github.com/acmi-lab/counterfactually-augmented-data [已核验] 172★ Apache-2.0 | 9,064 人工最小编辑反事实。许可最干净 |
| **MAGIC** | arXiv **2507.21544** [已核验] EMNLP 2025 Findings | KG 生成的细微跨上下文冲突。⚠ **它明确批评既有基准"过度依赖实体替换技术"——预期审稿人会引这条质疑我们的探针,须准备回应** |
| **CONFLICTS / DRAGged into Conflicts** | arXiv **2506.08500** [已核验] | RAG 知识冲突分类法 + **每类的期望模型行为** + 专家标注基准(Google) |
| **RAGuard(基准)** | arXiv **2502.16101** [已核验] | Reddit 真实误导信息;**头条发现:所有受测 RAG 系统在误导性检索下都不如其零样本基线。**强动机引文 |
| **RAMDocs** | github.com/HanNight/RAMDocs [已核验] 24★ MIT,COLM 2025 | 歧义+误导+噪声;MADAM-RAG(每文档一 agent 多轮辩论)是**逐文档投票的直接替代方案,可作对照臂** |
| **LLM-AggreFact** | `lytang/LLM-AggreFact` [已核验] | ⚠ **cc-by-nd-4.0,卡片明确禁止用于预训练/微调**,仅可评测。不含 VitaminC,但**含 WiCE** |

---

## 8. 我们可以主张的空白(四路检索均未找到)

1. **按独立来源(parent page)去重计票**作为防御信号 —— 无人做。所有方法把文档当可交换的票。
2. **以最小编辑/单实体替换为威胁模型、且在检索阶段防御并据此评测**的论文 —— 无人做。
   实体替换广泛用于**基准构造**,但不作为检索污染威胁模型。
3. **现代验证器(MiniCheck 世代)在最小编辑负例上的读数** —— 无人测。
4. **"embedding 聚类防御对 1-token 编辑构造性失明"这条批评** —— 无人写下。
5. **三类 argmax 无阈值决策规则下的 grounding 验证** —— 无先例。所有既有系统都是分数+阈值
   (MiniCheck 二分类 t=0.5;AlignScore 保留三类头但读连续概率)。
   ⇒ **我们 MiniCheck 臂的 5.2pp 形式鲁棒性与其决策规则混淆,这一点必须在报告中说明。**
   ⇒ 同时意味着**我们的 argmax 协议是自成一格的,应写成贡献而非缺陷**。
6. **SUPPORTS recall 与最小编辑孪生准确率之间的交换率** —— 无人同时报告两者。我们必须自己测。

---

## 9. `[未核验]` 清单 —— 引用前必须自行 fetch

HalluGuard(arXiv 2510.00880 / ACL 2026 Findings,**无 HF id**)、Paladin-mini(arXiv 2506.20384 存在,
但"79.31 均分"仅来自搜索片段)、Galileo Luna(arXiv 2406.00975,开放权重未确认)、
"When Context Bites"(SIGIR 2026,DOI 10.1145/3805712.3809904,**ACM DL 返回 403,无 arXiv 镜像**)、
RAGForensics(WWW 2025,arXiv 2504.21668)、Benchmarking Poisoning Attacks(2505.18543)、
DenialRAG 2608.02678、RefineRAG 2604.07403、2602.04711、2605.05632、2606.12469、2510.00586、
POISONCRAFT 2505.06579、FlippedRAG 2501.02968、One Shot Dominance 2505.11548、
WikiContradiction(arXiv 2111.08543,**未找到任何公开代码或数据**)。

---

## 10. 由本次检索产生的行动项

| # | 行动 | 状态 | 备注 |
|---|---|---|---|
| 1 | **复现 RADAR 的 NLI-骨干消融**(在我们的 0B-2 探针上换骨干) | TODO | **性价比最高**;成本低(已有三臂 dump);两个方向都出结论 |
| 2 | 定位改写:**从"跨源答案一致性投票"改为"独立来源计票"** | TODO | 报告与任何 paper 草稿;附 `collision_rate 0.931` 为证 |
| 3 | 引用并区分 RADAR / ReliabilityRAG / RA-RAG / Astute / RobustRAG / ArgRAG | TODO | 不做则审稿人必点 |
| 4 | PURPOSE(2608.04756)写进 Limitations | TODO | 主动引用 |
| 5 | tasksource 族 HF 元数据 `tals/vitaminc` 标签写入 **R012e** 证据栏 | TODO | **不替代 R012e 本身**;注意 `long-nli` 的污染只在元数据 |
| 6 | 若微调失败,按 §5 顺序试 ①→②→③,**每条先写预注册句** | 待触发 | ① 是纯数据侧,不需新阈值面 |
| 7 | FactCG-DeBERTa-v3-Large 作为廉价对照臂 | 待触发 | safetensors + MIT + VitC/FEVER 双清白 + 同架构;**但二分类** |
| 8 | HallDetect 的**非对称聚合**作为孪生门决策规则的候选 | 待触发 | 与模型选择正交 |
