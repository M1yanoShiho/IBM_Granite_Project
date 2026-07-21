# Complementarity-Aware Coverage Selection — 设计规格(Selector A2 / V3 覆盖层)

**日期:** 2026-07-21

**状态:** 设计草稿,承接已实现的门 spec;待用户书面确认后进 writing-plans → TDD。

**模块:** Selector(模块二);不改 Retriever/Generator,不改冻结契约。

**承接:** [gated-corroboration-selector-design](2026-07-20-gated-corroboration-selector-design.md)。
本设计在门(contrastive drop gate)之后新增一层**集合级覆盖选择**,把"幸存者按 blended
截断"换成"幸存者按覆盖打包"。门负责踢有害证据,覆盖层负责奖励互补证据、压制冗余。

## 1. 目标与范围

让选择器识别并保留**支撑答案但不含答案**的互补证据(slide 3 的 "Calculation detail →
supports → Claim"),同时压制近重复拷贝——而不引入 NLI 判断或绝对阈值。

适用范围沿用门 spec §1:单答案、单段可答的事实题。多跳回答与语义蕴含级 support
明确不在本层(见 §12,属 Graph 2.0)。

## 2. 核心洞察(决定所有设计)

**互补性是"选中集合"的属性,不是单条 passage 的属性。** 一条互补证据值不值钱,取决于
集合里是否已有它那块信息。当前选择器逐条独立打分再截断(pointwise),给每条固定分,
**结构上表达不了互补性**。所以本层把选择目标从"逐条打分 + 截断"换成"集合级覆盖",
兑现 slide 3 的 "a document's value depends on its role in the set"。

## 3. 设计承诺(继承门 spec 的两条不可退让性质)

1. **踢人路径不变、不新增误杀。** 覆盖层只在门**保留下来的**幸存者里做重排打包,
   自己**从不 drop**;因此不可能引入新的误杀。
2. **确定性 + 相对。** 覆盖特征建在词/实体/规范化数值上(复用 `answer_norm`),
   边际覆盖增益是集合内相对量,**无 NLI、无绝对阈值**。

## 4. 架构与数据流

```
CandidateSet → [门 spec §5 全流程:抽取/聚簇/独立票/重排/drop]
  ↓ 幸存者集合 S(含 blended 分)+ 已识别的答案簇
  ↓ 计算覆盖特征宇宙(query + 胜出簇相关的实体/数值)
  ├─ 钉住(pin):每个**幸存答案簇**取 blended 最高的一条代表,先入选
  │             (保护 required recall;1v1 暴露冲突时两边代表都 pin)
  ├─ 贪心覆盖填充:剩余预算按"边际覆盖增益"贪心加入,冗余特征递减折价
  ↓ 截断至 max_selected(不足不硬凑)
SelectionResult
```

覆盖层在门之后、截断之前替换原来的 `(surviving_window + tail)[:max_selected]`。

## 5. 覆盖特征(§6.2 复用,确定性)

`features(passage)` = { passage 内经 `answer_norm.canonicalize_answer` 规范化的数值 }
∪ { passage 内的实体 token 跨度 },**再过滤为同时出现在 query 文本或任一胜出簇 passage
里的特征**。

- 数值规范化复用现有 `answer_norm`("1,200 million" 与 "$1.2B" 记同一特征);
- 实体抽取用简单确定性启发式(大写跨度 + 停用词过滤;精确规则为实现细节,须单测);
- **过滤到 query+胜出簇相关**是关键——否则覆盖会奖励"题外多样性";过滤后覆盖才是
  "支撑答案主题"的代理。

## 6. support 边(V3 图的新边,确定性代理)【默认已定】

候选 c 对胜出答案 W 构成 **complementary-support 边** 当且仅当:

1. c **无有效自身答案**(`is_valid_answer` 为假)——即它不是竞争主张,只是支撑材料;
2. `features(c)` 与 W 的 asserting passages 的特征(或 W 的答案值本身)**至少有一个交集**。

这是纯确定性代理:比语义蕴含弱,但不碰 judge-kappa≈0 的坑。更严的定义(要求覆盖的是
"答案计算涉及的量"而非泛同主题实体)列为可选升级旋钮,不作 v1 默认。

## 7. 贪心覆盖选择(冻结规则)

设幸存者 S、胜出答案簇集合 C(每个是一个 valid-answer 簇)、覆盖宇宙 U(§5)。

1. **Pin:** 对每个 `cluster ∈ C`,取其 blended 最高成员入选 `output`;更新已覆盖特征集
   `covered`。(保证每个被门保留的答案簇——包括 1v1 暴露冲突的两边——都有代表进上下文。)
2. **贪心填充:** 当 `len(output) < max_selected` 且仍有未选幸存者:
   - 对每个未选 c 算 `gain(c) = |features(c) \ covered|`;
   - 按 **(gain 降序, blended 降序, retrieval_rank 升序, evidence_id 升序)** 取第一个入选,
     更新 `covered`;
   - `gain` 全为 0 时,该 key 自动退化为按 blended 填充(不浪费预算,也不硬凑超出 S)。
3. 截断至 `max_selected`;`len(S) < max_selected` 时输出短于预算(不硬凑)。

冗余近重复的 `gain` 为 0(没带来新特征)→ 排在最后 → 预算紧时自然被挤出。这就是
"duplicate = 一票不是三票"在选择层的兑现。

### 7.1 安全性质(验收标准)

- **required recall 结构性非劣:** 答案簇代表被 pin,答案所在证据永远在输出里,覆盖打包
  不会为多样性把它挤掉;
- **不新增误杀:** 覆盖层不 drop,只在幸存者内排序打包;
- **暴露冲突保持:** 多个幸存答案簇各 pin 一条代表,1v1 冲突两边都留;
- **coverage-off 对照干净:** E3 的 off 臂 = 直接用门 `GatedCorroborationSelector`(不走本层);
  本层在单一幸存答案簇且 U 无区分度时退化为 "pin 代表 + blended 顺序",与门只差 pin 的位置。

### 7.2 测试映射(逐条单测)

答案簇代表必被 pin;冗余近重复(同特征)gain=0 被挤出;互补证据(覆盖胜出簇实体、
无自身答案)被 pin 之后优先于纯相关无覆盖者入选;数值别名("1,200 million"/"$1.2B")
记同一特征;U 为空时退化为 blended 顺序(与门逐位相等);幸存者不足不硬凑;
1v1 两答案簇各留一代表;确定性(重复调用同结果)。

## 8. 信号毕业制(继承)

覆盖/support 属新增信号,先经 §9 的 E1 support-边准确率 + §9 的 E3 下游评测验证,
才写入主张。未验证前,覆盖层作为可切换选项(默认由实验决定),门单独可用。

## 9. 评估(新增两项)

- **E1-support(边检测准确率,承接门 spec §12 + 导师要求的第三种边):**
  在已知"真支撑证据"的数据上(如带 official supporting-fact 标注的题),测实体/数值重叠
  代理的 support-边 precision/recall;零人工标注(用 official supporting-fact 标签)。
  与 conflict 边(false/missed-conflict)、duplicate 边(去重单测)并列成 ArbGraph Table 4
  式的三边准确率表。
- **E3(覆盖层下游因果贡献):** 固定门,coverage-on vs coverage-off 配对,主指标
  grounded-answer F1 / faithfulness / cover-EM(承接 finding 17 的 k=10 F1);guardrail:
  required recall 非劣(§7.1 结构上已保证,仍需实测确认);辅助:选中集近重复对数下降。
- **数据耦合(诚实):** 互补性只在"答案受益于多条支撑"的题上显现,单段可答的反事实集
  (门用的 NIAH/NQ)发挥空间小;E3 需要互补压力数据(多支撑题),此为跨数据集决策项。

## 10. 诚实边界

- 覆盖是互补性的**代理,不是蕴含**;区分不了"支撑答案的计算"与"同主题无用题外话",
  真 support 判定属 Graph 2.0;
- 实体抽取启发式有噪声,E1-support 量化其代价;
- 被大量独立转抄的互补错误信息仍是 corroboration 一族原理边界(继承门 spec §9)。

## 11. 组件与文件

| 文件 | 内容 | 测试 |
|---|---|---|
| `src/evidence_rag/selector/coverage.py` | `features()` 抽取 + 贪心覆盖选择(§5-§7) | `tests/selector/test_coverage.py`(§7.2 全表) |
| `src/evidence_rag/selector/gated.py` | 新增 `GatedCoverageSelector`(复用门的 drop,替换截断为覆盖打包);或在 `GatedCorroborationSelector` 加 `coverage: bool` 开关 | `tests/selector/test_gated_coverage.py` |
| `src/evidence_rag/selector/answer_norm.py` | 复用数值规范化(不改) | 现有绿 |
| `src/evidence_rag/composition.py` | 注册 `gated-coverage-corroboration` + 参数(继承 alpha/margin/support_cap/top_n + coverage 开关) | swap-matrix 补行 |

实现顺序(TDD):`coverage.features` → 贪心选择 → `GatedCoverageSelector` → 注册。

## 12. 非目标(v1 明确不做)

- 不做 NLI/语义蕴含级 support(= Graph 2.0 升级路径);
- 不做多跳回答(答案跨 passage 拼,属另一套 pipeline);
- 不用嵌入相似度当覆盖判据(保持确定性、可单测);
- 不改三模块契约;覆盖层不 drop(drop 只归门)。

## 13. 待定项

1. support 边是否升级到"答案计算涉及量"的更严定义(§6 旋钮);
2. E3 的互补压力数据集(与 §14 数据决策耦合;多支撑题多在被保留为 final test 的多跳集);
3. 覆盖开关默认值(由 E3 结果决定,先默认 off、门单用)。
