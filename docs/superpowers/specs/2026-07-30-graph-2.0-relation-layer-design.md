# Graph 2.0 — NLI 关系层 · 设计

> **历史文档（已退役）:** Graph 2.0/MIS 路线未通过正式实验。当前方案见
> [Beam Selector 实验计划](../../selector/BEAM_SELECTOR_EXPERIMENT_PLAN.md)，失败证据与保留边界见
> [旧方法记录](../../selector/LEGACY_METHODS.md)。

状态:APPROVED(brainstorm 2026-07-30,Weikai;四处决定 + 四处协议修改经 AskUserQuestion 与逐段确认拍板)。

原先承接已删除的 `docs/selector/selector.md` S1–S6 与 [M0_PROTOCOL_FREEZE.md](../../selector/M0_PROTOCOL_FREEZE.md)(DRAFT)。
本文件曾是 M0 冻结前的设计依据；已删除的 `TRAINING_PLAN.md` 等旧资料见
[旧方法记录](../../selector/LEGACY_METHODS.md)。

---

## 0. 靶子

Graph 2.0 不是"把图做大",是**换掉一条边的生成方式**:把 [clusters.py:31](../../../src/evidence_rag/selector/clusters.py#L31)
"canonical 字符串不等 ⇒ 矛盾"这个确定性代理,换成关系预测 + UNKNOWN。两个已量化的缺口:

| 缺口 | 现值 | 已排除的解释 |
|---|---|---|
| `false_conflict`(同意但措辞不同被判分歧 → gold 碎裂 → 门误踢) | ≈.49 池内 | lenient 等价只回收 25%(S3);确定性规则原理上吃不到同义/缩写 |
| `missed_conflict`(孪生崩塌,毒与真判成同一簇) | .31–.43 池内 | 截断、检索、prompt、模型容量、抽取结构五种(S2/S4/S5/S6) |

S6 给了可证伪的必达目标:**压低 missed_conflict 而不推高 false_conflict**。

---

## 1. 已拍板的决定

- **D1 = A。** Graph 2.0 只升级"边与票的构造";[gated.py:113](../../../src/evidence_rag/selector/gated.py#L113)
  的四条件与整数票判据**一行不改**。NLI 只进"边怎么建",不进"踢留的刻度"(承诺 1)。走承诺 5 信号毕业制:过 Gate 0B 才准进门。
- **架构 = 方案 1(passage × claim 二部图),方案 2(claim–claim 互蕴含,票仍来自抽取)拆成独立消融臂。**
- **Relation Builder 先零训练探底**,现成 checkpoint 直接跑 Gate 0B;未过才训。
- **ContractNLI 踢出关系监督集**(理由见 §3.0)。
- **D2 消解。** D1=A 下选择器无参数;"训练集"只对关系模型有意义,而零训练路线下连关系模型都不训 → NIAH train 仅用于
  §3.2 的 0B-2 探针构造。2000-vs-500 的原问题不再适用,M0 写明消解理由而非留空。
- **交付分层。** 8/20 报告:M0 冻结 + Gate 0B + dev 同池配对(M3 消融)。9/4 最终:sealed 600 一次性确认(M5)。
  两条线并行(§6)。dev 结果在报告中必须标注"dev、非确认性"。

---

## 2. 架构

### 2.1 新模块(与 selector 平级)

```
src/evidence_rag/relations/
  models.py       RelationLabel(SUPPORTS|REFUTES|UNKNOWN)、ClaimNode、RelationEdge(冻结 dataclass)
  claims.py       抽取答案 → claim 节点 + hypothesis 文本(冻结模板)
  predictor.py    RelationPredictor Protocol + NLIRelationPredictor
  clustering.py   互蕴含 claim 聚类
  graph.py        QueryLocalGraph:建图 + independent_support
  cache.py        边缓存,键 = (model_version, premise_hash, hypothesis_hash)
```

放在 selector 外的理由:Relation Builder 要能独立跑 Gate 0B(ContractNLI/VitaminC 上没有 selector 的概念),
也要能被 `evaluation/` 直接调用做组件评估。selector 只消费其输出,依赖方向单向。

### 2.2 让"门不动"成为可执行断言

把簇的**构造**从 `_gate` 抽成 Protocol:

```python
class SupportProvider(Protocol):
    def clusters(self, window, answers) -> tuple[AnswerCluster, ...]: ...
```

- Graph 1.0 → `ExactSupportProvider` / `LenientSupportProvider`(包住现有两个函数,行为逐字不变)
- Graph 2.0 → `GraphSupportProvider`(NLI 二部图)

`_decide` 与四条件不改,两条臂共用同一份决策代码。单测断言"给定相同 clusters,两臂 GateDecision 完全相同"
—— 这是 D1=A 的机器可验证证据,也保证 E2 对照的唯一变量确实只有边的构造。

### 2.3 四条件的语义重映射(冻结)

| 门条件 | Graph 1.0 | Graph 2.0 | 打的缺口 |
|---|---|---|---|
| 1. c 抽出了有效答案 | `is_valid_answer(raw)` | c 至少有一条 SUPPORTS 边 | 弃权获一等地位:全 UNKNOWN 的干扰段永不可踢 |
| 2. 存在竞争簇 | 任一 canonical 答案不同的簇 | 见 `conflict_mode` | 同义/缩写已在聚类层合并,不再制造假冲突 |
| 3. `support(K′) − support(K) ≥ margin` | 不变 | 不变(整数票) | — |
| 4. `support(K) ≤ support_cap` | 不变 | 不变(整数票) | 孪生收益从这里体现 |

`independent_support(K)` = 与 K 中任一 claim 有 SUPPORTS 边的**不同 source_parent_id** 数
(§3.4 生效后;之前为 document_id,与 Graph 1.0 逐字相同)。

两条 Graph 1.0 里不存在、必须新冻结的语义:

1. **c 支持多个互不蕴含的簇** ⇒ c 视为"未表态",条件 1 判假,**不可踢**。保守方向,符合承诺 3(失效方向=闭嘴)。
2. **c 的"自己的簇"** = c 支持的簇中 `independent_support` 最大者。仅当 c 只支持一个簇时用得上,规则 1 已排除歧义情形。

**`conflict_mode` 参数(冻结两值):**

- `distinct_cluster`(**主口径**):竞争簇 = 任一其他 claim 簇。因聚类已按互蕴含合并,不同簇**按构造**即互不蕴含 ——
  实现上不需要额外的蕴含检查,与 Graph 1.0 结构逐条平行,唯一变量确实只有边的构造。
- `refutes_edge`(**消融臂**):竞争簇还须有成员向 c 的 claim 发出 REFUTES 边。更严,预期 harm 改善更小但 recall 更安全。
  这一臂使 TRAINING_PLAN Block 3 的"去掉 CLAIM_REFUTES"消融变得有实质含义 —— 主口径下 REFUTES 边只被记录不被门消费。

### 2.4 数据流与成本

```
CandidateSet(bm25/q2d 池)
  ↓ 窗口 top_n=20(窗外原序尾随,不抽取不建边)
  ↓ 一次答案抽取(extraction.py 不动,3B,21 次生成)
  ↓ claims.py:答案 → claim 节点 + hypothesis(冻结模板)
  ↓ clustering.py:claim × claim 互蕴含 → claim 簇        [~|C|² 次 encoder 前向]
  ↓ predictor.py:passage × claim → SUPPORTS/REFUTES/UNKNOWN [~20×|C| 次前向]
  ↓ graph.py:SUPPORTS 边 → 按 source_parent_id 去重 → independent_support
  ├─ 重排段:blended = α·relevance + (1−α)·corroboration    ← 不动
  ├─ 门段:四条件(同一份代码)                              ← 不动
  ↓ 截断 + 窗外尾随
SelectionResult
```

hypothesis 模板(冻结主口径):`The answer to the question "{Q}" is {A}.`
QA2D 式转换器(Chen et al. 2021,§10 [5])作**预注册消融**在 Gate 0B 上比。模板确定性可审计,符合 provenance 纪律;
QA2D 更自然但引入 seq2seq 依赖与非确定性。

三条冻结的构造规则(不写清楚实现必错):

1. **Graph 2.0 = Graph 1.0-lenient + NLI。** lenient 等价保留为 NLI 之前的**确定性预合并**,不被替换。
   理由:主对照就是 Graph 1.0-lenient,保留它使增量变量**纯粹是 NLI**;同时压低 `|C|` 降成本;
   且精确/包含匹配那部分 lenient 的精度高于 NLI。
2. **claim–claim 聚类比较的是 hypothesis 句子,不是裸答案串。** NLI 模型吃 `"Paul"` vs `"Apostle Paul"` 是退化输入;
   用两句完整 hypothesis 比较,共享的 question 让句子良构。
3. **parametric 自答不成 claim 节点。** 承诺 5 规定模型先验留在重排、禁入门内;claim 节点是门的输入,故它进不去。
   `AnswerExtractionEngine` 仍产出 parametric 供重排使用,关系层忽略之。

`|C|` 经 lenient 预合并后约 5–10,总计约 125–200 次 59M encoder 前向,相对现有 21 次 3B 生成,增量 <20% 计算。
`|C|` 分布、前向次数与 per-query 墙钟时间必须落盘(TRAINING_PLAN §11 要求主表报成本与延迟)。

**承诺改写(必须写进报告,不得含糊):** "门零新增 LLM 调用" → "门零新增**生成式**调用"。

### 2.5 argmax 主口径,τ 作预注册补救

RELATED_WORK 把与 ArbGraph 的差异化轴定在"绝对阈值 vs 池内相对"。引入"confidence > τ 才建边"会把这条轴弄浑。

**主口径 = argmax,不设阈值。** UNKNOWN 是模型预测的一个类别,不是阈值产物 → 踢人路径上零绝对刻度。

τ-gating 降级为补救措施:仅当 argmax 下 Gate 0B REFUTES precision < .85 时启用;τ 只在 VitaminC official **dev**
上按 coverage/precision 目标冻结一次,之后不许按 NIAH 结果调;启用则报告必须写明"补救措施被触发"。

### 2.6 错误处理

- NLI 模型加载失败 / OOM → **fail fast**,禁止静默退回 Graph 1.0(静默退回污染配对对照且无指标能发现)。
- premise 截断:`passage_chars=600` 后再按 tokenizer 截到模型上限;截断事件计数落盘。
- 退化 claim(`NONE`、空串、纯标点)不成节点。
- 边缓存命中率、UNKNOWN 率、截断率进 per-query dump(G-PQ)。

---

## 3. Relation Builder 与 Gate 0B

### 3.0 为什么踢掉 ContractNLI

[stanfordnlp/contract-nli](https://github.com/stanfordnlp/contract-nli) 是 **17 条固定假设 × 607 份 NDA** 的文档级三分类
(§10 [1]),假设形如 "Some obligations of Agreement may survive termination.",与"段落是否支持 'Q 的答案是 X'"
无结构相似性。v1 已量到 ContractNLI 迁移 −0.134。而 Gate 0B 原文要求"ContractNLI 与 VitaminC 必须分别通过"
⇒ 保留它等于要求在一个与任务无关的法律域上打到 REFUTES precision ≥ .85,是自设阻塞。

VitaminC 反向成立:45 万 claim-evidence 对取自 10 万+ Wikipedia 修订,**一对近乎逐字相同的证据只有一处事实被改动,
一条支持一条不支持**(§10 [2])—— 与 injector 造的孪生(`cf::<qid>::needle` = gold passage 改一个 span)同构。

### 3.1 候选模型(R012)

| 臂 | 模型 | 输出形态 | 角色 |
|---|---|---|---|
| 主 | [tals/albert-xlarge-vitaminc-mnli](https://huggingface.co/tals/albert-xlarge-vitaminc-mnli) | 原生三类,~59M | NEI 直接即 UNKNOWN,零映射损耗;域对口;开销可忽略 |
| 对照 | [MiniCheck-FT5](https://github.com/Liyan06/MiniCheck) 770M | 二分类 + prob | LLM-AggreFact <1B SOTA(§10 [3]);需 claim 否定双向探测拆 REFUTES/UNKNOWN |
| 上界 | [DeBERTa-v3-large-mnli-fever-anli-ling-wanli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) | 三类,MIT | 通用 NLI 上界,非域内 |
| 备选 | [AlignScore](https://github.com/yuh-zha/AlignScore) 355M | 对齐分 | §10 [4];MiniCheck 在 10 个数据集里 6 个更优,故排后 |

主口径选 albert-xlarge:三类原生对齐。MiniCheck 的二分类要靠否定 claim 反推 REFUTES,等于在关系层塞进一个未验证构造。
若 albert 明显输给 MiniCheck,结论是"需要更强核查器"而非"需要训练"——这个区分决定 §3.6 是否启动。

**LLM teacher 边界:** MiniCheck 由 GPT-4 合成数据训练。它作为**系统组件**(边预测器)使用不违反零新增人工标注约束;
禁止的是把它的输出当 primary label。Gate 0B 的验收标签仍只来自 official test 与 deterministic provenance。

### 3.2 Gate 0B 拆两层

原 Gate 0B 只在 ContractNLI/VitaminC 上验收。洞:一个在 VitaminC official test 上过关的模型,完全可能在
"NQ 段落 + 模板 hypothesis"上无用。踢掉 ContractNLI 后此洞更明显。

**0B-1 外部效度**(official labels,一次性):VitaminC official test。沿用原阈值不放宽 ——
REFUTES precision ≥ .85、support/refute macro-F1 ≥ .80、non-UNKNOWN coverage ≥ .80、SUPPORT/REFUTES 各 ≥ .70。

**0B-2 任务效度**(deterministic provenance,零新增标注):从 NIAH **train** split 的 mutation log 生成四类对 ——

| premise | hypothesis | 标签 | 确定性依据 |
|---|---|---|---|
| needle | gold claim | SUPPORTS | injector 已验证 gold alias 在 needle 中恰好出现一次 |
| `cf::needle` | replacement claim | SUPPORTS | mutation 定义 |
| `cf::needle` | gold claim | **REFUTES** | 单答案假设 + 同机械类别异值替换 |
| needle | replacement claim | **REFUTES** | 同上 |

第三行正是 S4/S5/S6 三轮认定"抽取层修不动"的孪生判别,现在变成纯 CPU 可测的分类题。

UNKNOWN 的 primary gate 只能用 0B-1 的 official NEI 类。跨 query 配对的"推定 UNKNOWN"只报 abstention rate 作
secondary proxy —— 干扰段是否真与 claim 无关没有确定性依据,按零标注约束不能当 primary label。

### 3.3 0B-2 阈值从选择器需求反推

| 失效 | 后果链 | 对应指标 | 预注册阈值 |
|---|---|---|---|
| `cf → gold claim` 误判 SUPPORTS | 毒进 gold 簇 ⇒ 条件 2 失效 ⇒ 不可踢 ⇒ harm 不降 | twin REFUTES accuracy | ≥ .70 |
| 含 gold 的段漏判 SUPPORTS | gold 票低估 ⇒ needle 孤立(support=1≤cap) ⇒ 被踢 ⇒ recall 掉 | gold-passage SUPPORTS recall | ≥ .85 |

两个指标的精确口径(避免歧义):

- **twin REFUTES accuracy** = 在 §3.2 表格**后两行**(`cf::needle` × gold claim、needle × replacement claim)
  构成的对集合上,预测标签 == REFUTES 的比例。UNKNOWN 计为**失败**。
- **gold-passage SUPPORTS recall** = 在**第一行**(needle × gold claim)上预测 == SUPPORTS 的比例。UNKNOWN 计为失败。

`.70` 的依据:当前池内 `1 − missed_conflict` = .57–.69,低于 .70 连"不比现状差"都保证不了。

**S6 纪律(写死):** 0B-2 是隔离对探针,**不许外推到池**。池级判定只认 E1 `cluster_eval`,G-FC 护栏在那一层生效。
S5→S6 翻车的机制就是隔离探针高估,这次预先堵上。

### 3.4 SAME_SOURCE:先量后修

[base_loader.py:56](../../../src/evidence_rag/materializer/base_loader.py#L56) 把 dpr-w100 的 title 拼进
`document.text` 首段;dpr-w100 是 100 词切分语料,**一个 Wikipedia 条目对应多个 doc_id**。
故现行 `len({candidate.document_id})`([clusters.py:36](../../../src/evidence_rag/selector/clusters.py#L36))
会把同一条目的多个 passage 算成多张独立票 —— 它连同条目转抄都拦不住,而检索恰恰倾向于把同条目相邻 passage 一起召回。
selector.md §3 那句"`document_id` 去重只能拦共享出处的转抄"因此**高估了现有实现**。

修法确定性、CPU 级、无契约变更(`EvidenceCandidate` 无 title 字段,[models.py:64](../../../src/evidence_rag/contracts/models.py#L64)):
从已物化 `documents.jsonl` 首段解析 title,建 `document_id → source_parent_id` sidecar(与现有 `provenance.jsonl` 同模式)。
规则单测 100% 满足 Gate 0B 的 SAME_SOURCE 条款。

**决定(2026-07-30,修正本节初稿的"先量后修"):无条件采用,不设阈值分支。**
若 dpr-w100 的 doc_id 确实是每 passage 一个,则数 `document_id` **违反了 `independent_support` 自身的定义**
—— 这是正确性缺陷,不是可调旋钮。用碰撞率阈值决定改不改,等于让正确性取决于缺陷有多严重。故:

- 冻结参数 `support_unit ∈ {"document", "parent"}`,**主口径 = `parent`**;`document` 保留仅为复现 S1–S6 旧臂。
- top-20 同 parent 碰撞率照测照报,但它是"这件事影响多大"的**背景量**,不是决策门。
- Graph 1.0-lenient 需补跑一条 `support_unit=parent` 基线臂 —— 主对照必须两侧同 `support_unit`,
  否则 SAME_SOURCE 修复的收益会混进 NLI 的账上。

**依赖链(M0 需高亮):** title 解析不只修计票,它还是 §4.2 五轴审计第二轴(父页面零重叠)的**硬性前置**。
即使 `support_unit` 保持 `document`,解析本身也不可省 —— 没有它 sealed 600 建不出来。

### 3.5 metamorphic 测试

同 parent 重复不得增加 `independent_support`;打乱边后图收益必须消失;
孪生对称性 `NLI(p,gold)=REFUTES` 与 `NLI(p,replacement)=SUPPORTS` 的一致率作诊断,不设门。

### 3.6 训练路径(仅当 0B-1 或 0B-2 未过才启动)

预注册,不看结果不改:

- 基座 `microsoft/deberta-v3-base`;脚手架用 sentence-transformers CrossEncoder 三类训练范式
  ([examples/cross_encoder/training/nli/](https://github.com/UKPLab/sentence-transformers/tree/master/examples/cross_encoder/training),
  CrossEntropyLoss over contradiction/entailment/neutral;§10 [6])
- 数据:VitaminC 主训(revision-family 去污染,official test 一动不动,删除清单存档)+ NIAH train 的 mutation-log 对做域适配
- 全链路按 parent page + synthetic family 做 5-fold OOF,不是只对关系分做 OOF
- 三 seed 13/42/73,dev 冻配置,official test 只跑一次
- 硬约束:NIAH 域适配对的 parent page 必须与 sealed-600 零重叠

**副作用:** 零训练路线下模型从未见过 NIAH 语料 ⇒ passage-hash 泄漏轴天然为空,§4.2 少一整轴的实质风险。
这是选零训练的第三个理由。

---

## 4. sealed 600

### 4.1 规模反推

dev 的 2000 采样落 1479 注入,skip rate .261。落 600 注入题需起始 `600/(1−.261) ≈ 812` 合格 query,取 **900** 留余量。
写进 manifest;不得跑到一半发现不够再补(补样本 = 看结果后改数据)。

### 4.2 五轴审计的可执行判据

| 轴 | 判据 | 来源 |
|---|---|---|
| query | query_id 与规范化 query 文本双重零重叠 | queries.jsonl |
| 父页面 | 规范化 title 零重叠 | documents.jsonl 首段(§3.4 sidecar) |
| answer entity | 规范化 gold value 零重叠 | qrels answers |
| passage hash | gold passage 文本 sha256 零重叠 | documents.jsonl |
| synthetic family | `(gold_value, replacement_value, 机械类别)` 三元组零重叠 | mutation log |

### 4.2b 检索器冻结(决定 2026-07-30)

sealed-600 的候选池用**与 E2 相同的 bm25 检索器**,不换成队友的 StrongBM25/hybrid。
理由:Graph 2.0 的主张是"给定固定候选池,选择器更可靠",而固定候选池正是冻结的模块接口;
换检索器会把池的组成变成混淆变量,且 E2 exact 臂的 dump 只有在池相同时才可复用。
StrongBM25/hybrid 作为**独立泛化臂单列一张表**,不与主表合并、不进 C1 判定。

### 4.3 语料重建(不共享现有 100k 子采样)

现有语料的 gold doc 是按 dev 那 2000 query 的 qrels 选进去的,sealed-600 新 query 的 gold 大概率不在 ⇒ 共享方案不成立。
方案:按 sealed-600 的 qrels 保留全部 gold doc + 重新蓄水池采样 distractor,seed 固定并记 hash。
父页面轴因此天然更易干净。代价是 index 重建,但队友 d5f7908 改过 IndexManifest schema、旧 cache 已全失效,反正要重建。

### 4.4 使用边界(写进 manifest 冻结)

- 一次性:M4 冻结后跑一次,同一次产出全部冻结系统 —— q2d 截断、fixed_0.6、Graph 1.0-exact、Graph 1.0-lenient、
  Graph 2.0、Graph 2.0 消融臂(方案 2 / 无 SAME_SOURCE / `conflict_mode=refutes_edge` / shuffled-graph)、oracle@20。
- 结果出来后不许改 δ、不许改阈值、不许补跑。
- per-query 全量落盘(G-PQ)。
- manifest hash 在任何 dev 实验之前冻结。

### 4.5 sealed 600 不承担什么

不承担 Relation Builder 验收(那是 0B-1),不承担机制诊断(那是 dev 上的 E1),只承担 C1 三联判定 + G-FC。
在 sealed-600 上做的任何诊断性观察只能进 discussion,不得进结论。

---

## 5. C1 重写与守卫

### 5.1 主对照与三联判定

D1=A 下主对照从"Graph 2.0 vs 学习型 v1"改为**同池配对的 Graph 2.0 (gate-on) vs Graph 1.0-lenient (gate-on)**,
唯一变量 = 边的构造;外加 gate-off 作绝对参照。

- ΔHarmful@10 ≤ −0.02 且双侧 95% CI 上界 < 0
- Required Recall@10 单侧非劣下界 ≥ −0.01(**两个基准,见 5.2**)
- NDCG@10 单侧非劣下界 ≥ −0.01
- G-FC:固定分母口径 false_conflict 不得劣化超 δ。**δ 的规则现在预注册,数值待 R001/R003:**
  `δ = max(0.05, R003 在固定分母口径下算出的 MDE)`。守卫不能定得比能检出的还紧;`0.05` 的下限依据是 S6 那次 +38pp
  —— δ=.05 拦下它绰绰有余。若 R003 算出 MDE > .05,必须**在冻结前**加样本或放宽 δ,绝不事后调。

三项必须同时满足,不得用次级指标改善替代失败的主指标。

### 5.2 recall 非劣必须对两个基准同时报(冻结)

- 口径 (a) vs Graph 1.0-lenient:证明"换边没让事情变糟"。
- 口径 (b) vs gate-off:这才是真正要闭合的守卫。Graph 1.0-lenient 在此为 −3.6pp,**FAIL**。

**头条判据是 (b)**,不是 (a)。§2.3 的机制预测说的正是 (b) 应当收窄:needle 不再需要自己抽出 gold 字符串,
只需支持池内已被提出的 gold claim ⇒ `support(K)` 升过 `support_cap` ⇒ 结构性不可踢;
而 cf 的 replacement claim 仍只有自己支持 ⇒ margin 变大 ⇒ 对毒更易 fire。
预测:**harm↓ 与 recall↑ 同向发生**。若实测为"harm↓ 但 recall 也↓",方案 1 的机制假设被证伪 —— 如实写。

只报 (a) 通过就宣布成功,等于把未闭合的守卫藏起来。禁止。

### 5.3 沿用 M0 已有三条新增守卫

G-FC(false_conflict 护栏,固定分母主口径 + 条件性次口径,与 `needle_gold_recovery`、abstention rate 三者联合判读)、
G-AB(弃权必须被度量)、G-PQ(per-query 必须落盘)。定义见 [M0_PROTOCOL_FREEZE.md](../../selector/M0_PROTOCOL_FREEZE.md) §3。

### 5.4 废除三-seed ensemble(条款原文)

**greedy decoding 已核实(2026-07-30):** [composition.py:336](../../../src/evidence_rag/composition.py#L336)
用 `GraniteLLMClient()` 无参构造 → [granite.py:39](../../../src/evidence_rag/generator/granite.py#L39)
`temperature = 0.0` → [granite.py:121](../../../src/evidence_rag/generator/granite.py#L121) `do_sample = False`,
且 `num_beams` 未设(HF 默认 1)。纯 greedy argmax,无采样无 beam。

> 由于 D1 决议采用零训练 NLI 边构造(zero-shot NLI Relation Builder)与确定性的四条件门控,选择器与关系层均无学习参数;
> 抽取层经核实采用 greedy argmax 解码,不含采样。全链路**无随机源**,seed 在本系统中没有作用点,
> 因此废除原计划的 3-seed ensemble 机制。

**不得写成"100% deterministic / 逐位可复现"。** `dtype="auto"` + `device_map="auto"` 下模型以 bf16/fp16 运行,
GPU kernel 选择与浮点规约顺序可在近似平局处翻转 argmax ⇒ 跨硬件重跑可能出现少量不同答案。
准确表述是"无随机源",不是"可复现到位"。

**替代物(取代 seed ensemble,而非简单删除):** rerun-stability 检查 —— 同硬件重跑 100 题抽取,
报答案级一致率与翻转样例。这比 seed ensemble 更贴近真实风险源,成本近零。列为 M4 冻结前的必做项。

若 §3.6 训练路径被启动,三 seed 条款**恢复生效**(仅对关系模型)。

---

## 6. 里程碑与并行排期

两条线并行,交叉点在"Graph 2.0 接进门"。

**线 A(CPU,技术债与地基)**

| # | 内容 | 依赖 | 产物 |
|---|---|---|---|
| A1 | `cluster_eval_cli --dump` per-case 落盘 | — | G-PQ 合规;S6 可重打分 |
| A2 | R001 G-FC 基线实测(固定分母口径重算 Graph 1.0+lenient) | A1 | δ 与基线值,回填 M0 §3 |
| A3 | `source_parent_id` sidecar + `support_unit` 参数(主口径 `parent`) | — | SAME_SOURCE 真实现;五轴审计第二轴解锁 |
| A4 | dev 窗口 top-20 同 parent 碰撞率实测 + Graph 1.0-lenient 补跑 `support_unit=parent` 基线臂 | A3 | 背景量 + 与 Graph 2.0 同 `support_unit` 的对照 |
| A5 | sealed-600 构建(900 起始 → 600 注入)+ 语料重建 + 五轴审计 | A4 | manifest + hash,冻结 |

**线 B(GPU,关系模型)**

| # | 内容 | 依赖 | 产物 |
|---|---|---|---|
| B1 | VitaminC / official test adapter + 去污染审计 | — | 0B-1 数据 |
| B2 | 0B-2 探针数据构造(mutation log → 四类对) | — | 纯 CPU,可先于 B1 |
| B3 | R012 sweep:albert-xlarge / MiniCheck-FT5 / DeBERTa-large **三臂同时**跑 0B-1 + 0B-2 | B1,B2 | Gate 0B 数字;**决定是否进 §3.6** |
| B5 | `relations/` 模块实现 + `GraphSupportProvider` | B3 过 | 可接门 |

B3 合并了原 B4:三臂全是推理、互不依赖,串行只是浪费并行槽;而"需要更强核查器"vs"需要训练"这个区分
**必须三臂同时在手**才能下 —— 只跑 albert 一臂时,未过 Gate 只能得出"albert 不够",无法排除"任何零训练模型都不够"。

**交叉与判定点**

- M3(dev,8/20 报告的可交付):Graph 2.0 vs Graph 1.0-lenient vs gate-off 同池配对 + E1 组件三指标 + 消融臂。依赖 A2、A4、B5。
- M4:冻结代码 / 配置 / hash。
- M5(9/4):sealed 600 一次性运行全部冻结系统。依赖 A5、M4。

**风险(如实记录):** 若 B3 在 albert 与 MiniCheck 上均未过 Gate 0B,§3.6 训练路径启动,M5 大概率落在 9/4 之后。
届时报告的头条只能是 M3 的 dev 结果 + Gate 0B 的负结果,不得把 dev 结果表述为确认性证据。

**实现计划的切分(本 spec 覆盖 M0–M5,不适合塞进单个 plan):**

- **Plan 1 —— 地基与探底(A1–A4 + B1–B3 + M0 冻结)。** 全部是确定性 / CPU 或轻 GPU 的工作,彼此无阻塞,
  终点是 Gate 0B 的数字。B3 的结果是一个**真实的分叉点**:过则进 Plan 2,不过则进 §3.6 训练路径
  —— 在拿到 B3 之前写后续 plan 是空写。
- **Plan 2 —— 关系层接门 + dev 判定(B5、A5、M3)。** B3 之后再写。
- **Plan 3 —— 冻结与 sealed 运行(M4、M5)。** M3 之后再写。

M0 协议冻结(§11 清单)属于 Plan 1,且必须在 B3 出数**之前**完成 —— 预注册的意义就在于此。

---

## 7. 测试策略(TDD)

- `relations/models`:RelationLabel 枚举完整性;边的 confidence/model_version/hash 字段必填。
- `relations/claims`:模板确定性(同输入同输出);退化 claim 不成节点;hypothesis 文本 hash 稳定。
- `relations/clustering`:互蕴含合并("paul" ≡ "Apostle Paul");异值不合并(gold vs replacement);无传递链;顺序确定性。
- `relations/graph`:同 parent 重复不增 support(metamorphic);SUPPORTS 边缺失 ⇒ 该 claim 无票;全 UNKNOWN 段不进任何簇。
- `selector/gated`:`SupportProvider` 注入后,exact provider 下**现有全部门测试保持绿**(行为逐字不变);
  相同 clusters 输入下两臂 GateDecision 相同;c 支持多簇 ⇒ 不可踢;c 无 SUPPORTS 边 ⇒ 不可踢。
- `relations/cache`:键含 model_version;命中不改变结果;model_version 变更导致 miss。
- fail-fast:模型加载失败必须抛错,不得退回 Graph 1.0(显式反例测试)。
- `source_parent`:title 解析规则单测 100%(Gate 0B 条款);无 title 首段的文档走确定性 fallback 并计数。
- CI:`ruff check src tests` + `mypy src tests/typecheck.py` 全树,Python 3.11。

---

## 8. 非目标

- 不改 `extraction.py` / `EXTRACT_PROMPT` / injector / `canonicalize_answer` / 三模块契约。
- 不引入学习型选择器、LightGBM、GNN、Neo4j、全语料知识图谱。
- 不上 `TIME_MISMATCH` / `CONDITION_MISMATCH`(无官方标签)。
- 不接 8B + decoupled 抽取(S6 后的决定,不是待办)。
- 不做 FinanceBench。
- 不把 sealed-600 上的诊断观察写进结论。

---

## 9. 立即停止条件(在 TRAINING_PLAN §9 七条之外新增)

1. Graph 2.0 相对 Graph 1.0-lenient 的 harm 改善,可被 shuffled-graph 负控复现 ⇒ 收益不可归因于关系层。
2. 方案 2 消融臂(只换等价、票仍来自抽取)复现全部收益 ⇒ 收益来自更好的等价判断,不是二部图 ⇒ 主张收窄为"学习型等价"。
3. G-FC 被推高超 δ,即使 missed_conflict 改善 ⇒ 重演 S6,不采用。

---

## 10. 参考文献

1. Koreeda, Y. & Manning, C. D. (2021). ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts.
   *Findings of EMNLP 2021*. https://aclanthology.org/2021.findings-emnlp.164/ · 代码 https://github.com/stanfordnlp/contract-nli
2. Schuster, T., Fisch, A. & Barzilay, R. (2021). Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence.
   *NAACL 2021*. https://aclanthology.org/2021.naacl-main.52/ · 代码 https://github.com/TalSchuster/VitaminC ·
   数据 https://huggingface.co/datasets/tals/vitaminc · 模型 https://huggingface.co/tals/albert-xlarge-vitaminc-mnli
3. Tang, L., Laban, P. & Durrett, G. (2024). MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents.
   *EMNLP 2024*. https://aclanthology.org/2024.emnlp-main.499.pdf · arXiv:2404.10774 ·
   代码 https://github.com/Liyan06/MiniCheck (Apache-2.0)
4. Zha, Y. et al. (2023). AlignScore: Evaluating Factual Consistency with a Unified Alignment Function.
   *ACL 2023*. https://aclanthology.org/2023.acl-long.634/ · arXiv:2305.16739 · 代码 https://github.com/yuh-zha/AlignScore
5. Chen, J., Choi, E. & Durrett, G. (2021). Can NLI Models Verify QA Systems' Predictions?
   *Findings of EMNLP 2021*. https://aclanthology.org/2021.findings-emnlp.324/ · arXiv:2104.08731 ·
   代码 https://github.com/jifan-chen/QA-Verification-Via-NLI (QA→NLI 转换器 + decontextualization)
6. Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
   代码 https://github.com/UKPLab/sentence-transformers ·
   CrossEncoder 三类 NLI 训练范式 `examples/cross_encoder/training/nli/`
7. MoritzLaurer. DeBERTa-v3-large-mnli-fever-anli-ling-wanli (MIT). 88.5 万 NLI 对。
   https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
8. ArbGraph(arXiv 2604.18362)—— 最近邻；旧版 related-work 对比已退役，见
   [旧方法记录](../../selector/LEGACY_METHODS.md)。
   差异化轴:绝对 credibility 阈值 vs 池内相对整数票;本设计 §2.5 的 argmax 决定即为保住此轴。

---

## 11. M0 需要落的改动清单

- [ ] D1 填 A;D2 记消解理由(§1)
- [ ] §2.3 两条新语义写进冻结的关系 schema
- [ ] §2.5 argmax 主口径 + τ 补救条款
- [ ] Gate 0B 拆 0B-1 / 0B-2,ContractNLI 移出并记理由(§3.0)
- [ ] 0B-2 阈值 .70 / .85 与反推依据(§3.3)
- [ ] SAME_SOURCE 无条件修复 + `support_unit` 参数 + 高亮其为五轴审计第二轴的硬性前置(§3.4)
- [ ] `conflict_mode` 两值与主口径(§2.3);三条构造规则 lenient 预合并 / hypothesis 句比较 / parametric 不入图(§2.4)
- [ ] sealed-600 检索器冻结为 bm25,hybrid 单列泛化臂(§4.2b)
- [ ] C1 重写 + recall 非劣双基准(§5.1、§5.2)
- [ ] 删除 3-seed ensemble,改写为"无随机源"+ rerun-stability 检查(§5.4;greedy 已核实)
- [ ] G-FC 的 δ 规则 `max(0.05, MDE)` 入协议;数值依赖 A2/R003
- [ ] 协议版本号 g2-proto-1 + hash 记入 EXPERIMENT_TRACKER R000
