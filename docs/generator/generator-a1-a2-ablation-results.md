# Generator A1/A2 受控消融实验结果

> 实验编号：G-A12
>
> 预注册：2026-08-15
>
> 完成与解盲：2026-08-17
>
> 结果性质：50-case calibration sample 上的单标注者描述性结果；未进行显著性检验

## 1. 研究问题

本实验单独评估 Generator A 部分的两项改动：

1. **A1 verification-oriented prompt**：要求初稿尽量采用一句一个可独立验证、自包含的事实，
   明确实体、时间、数量和条件，并在句末给出 evidence index。
2. **A2 robust claim splitter**：加强分句、真实 answer span 定位、rewrite faithfulness 检查和
   claim 去重，减少缩写、小数、initialism、同句多 claim 等场景下的错误。

它回答三个预注册问题：新版 A1 是否更适合逐句核验（H1）；在完全相同的 A1 answer 上，新版
A2 是否提高 claim 拆分质量（H2）；两项组合是否优于旧完整组合（H3）。实验不覆盖 NLI verifier、
实体一致性、A3–A5 repair、最终引用质量或拒答，因此不能用于评价 Generator B 或完整系统。

## 2. 实验设计

### 2.1 2×2 因子设计

| arm | A1 prompt | A2 splitter | 作用 |
|---|---|---|---|
| `old_old` | old | old | 优化前基线 |
| `old_new` | old | new | 在固定 old-A1 answer 上测 A2 |
| `new_old` | new | old | A1 改动及固定 new-A1 answer 下的 A2 基线 |
| `new_new` | new | new | 完整新版组合 |

A2 的合法因果比较只有 `old_old` vs `old_new` 和 `new_old` vs `new_new`，因为每一对共享逐字
相同的 A1 answer。`old_old` vs `new_new` 只能解释为组合效果，不能解释成 A2 单独效果。

### 2.2 冻结配置

- 数据：ALCE/ASQA calibration，50 个合格 case；top-5 GTR evidence；`seed=13`。
- 模型：`ibm-granite/granite-4.1-3b`；`max_new_tokens=256`；temperature 0；不采样。
- 历史边界：old=`11c03849b3bcb21bf83447b4726714d4bc762725`，
  new=`eda7065d8005e317ce951e2730f58996cd4757ec`。
- 运行：BluePebble Slurm job `18557307`；四臂在同一作业、同一模型实例中完成；运行代码
  commit=`9bcb862080dc365e746fe3328cb6190d94699d89`。
- 数据文件 SHA-256：
  `d72737ce7d46629d3504fc508f29ec0c270e0ddd5cff3fad69625d2c0e4be35b`。

完整 query IDs、selection order、文本 hash 和原始输出 hash 见实验目录的 `manifest.json`。

## 3. 人工盲评

四臂标签由 `annotation_key.json` 隐藏，所有 annotation rows 确定性打乱。共标注 639 行：

| unit | rows | 评测内容 |
|---|---:|---|
| A1 sentence | 163 | atomic、self-contained、independently verifiable、unresolved pronoun |
| A2 claim | 276 | atomic、self-contained、span aligned、rewrite faithful |
| A2 case | 200 | source-fact coverage（complete/partial/none/NA） |

标注完成后先检查唯一性、必填项和 coverage/error-type 一致性；639/639 行通过后才读取 arm key。
冻结工作簿 SHA-256 为
`11ba06f70ce4840d4b31f8ef2d37781f1d50d6511c9afc179a7c2a594018cb06`。
本轮只有一名标注者，因此不报告 Cohen's kappa；这不违反预注册，但属于结果局限。

所有比例排除 `NA`，并同时报告分子/分母。A1 atomic 中，无法合理应用原子性定义的句子按指南
记为 `NA`，所以其分母小于总句数。

## 4. 结果

### 4.1 H1：A1 prompt

| metric | old A1 | new A1 | Δ（new − old） |
|---|---:|---:|---:|
| atomic | 56/65 = 86.2%（NA=38） | 45/47 = 95.7%（NA=13） | +9.6 pp |
| self-contained | 50/103 = 48.5% | 47/60 = 78.3% | +29.8 pp |
| independently verifiable | 63/103 = 61.2% | 47/60 = 78.3% | +17.2 pp |
| unresolved pronoun ↓ | 14/103 = 13.6% | 0/59 = 0.0%（NA=1） | −13.6 pp |
| citation-format compliant | 77/103 = 74.8% | 54/60 = 90.0% | +15.2 pp |

新版在全部预注册质量方向上改善，且引用格式合规率没有下降。它同时把总词数从 1,490 降至
796，平均每题从 29.8 降至 15.9 words（−46.6%），总句数从 103 降至 60；两臂均为 50/50
非空回答。在固定 A2-old 时，complete source-fact coverage 从 73.3% 升至 85.4%（+12.1 pp）；
固定 A2-new 时从 80.0% 升至 87.5%（+7.5 pp），没有观察到因输出变短而导致的事实覆盖下降。
结果说明新版 prompt 生成了更短、更规整、更适合逐句验证的文本；但这里只检查了标注指南定义的
source facts，不能据此断言所有信息量都被保留。H1 获得**较强描述性支持，附带明显长度取舍**。

### 4.2 H2：A2 splitter

| arm | claim atomic | self-contained | span aligned | rewrite faithful | complete coverage |
|---|---:|---:|---:|---:|---:|
| `old_old` | 71/73 = 97.3%（NA=1） | 65/74 = 87.8% | 52/74 = 70.3% | 50/74 = 67.6% | 33/45 = 73.3%（NA=5） |
| `old_new` | 73/76 = 96.1%（NA=1） | 67/77 = 87.0% | 56/77 = 72.7% | 55/77 = 71.4% | 36/45 = 80.0%（NA=5） |
| `new_old` | 59/59 = 100.0%（NA=2） | 58/61 = 95.1% | 46/61 = 75.4% | 46/61 = 75.4% | 41/48 = 85.4%（NA=2） |
| `new_new` | 62/62 = 100.0%（NA=2） | 62/64 = 96.9% | 49/64 = 76.6% | 48/64 = 75.0% | 42/48 = 87.5%（NA=2） |

固定 old-A1 answer：

- span alignment +2.5 pp；rewrite faithfulness +3.9 pp；complete coverage +6.7 pp；
- atomicity −1.2 pp；self-containment −0.8 pp。

固定 new-A1 answer：

- span alignment +1.2 pp；complete coverage +2.1 pp；self-containment +1.8 pp；
- atomicity持平；rewrite faithfulness −0.4 pp。

两个合法配对的 coverage 和 span accuracy 都没有下降，符合预注册的最低成功条件；新版也在两种
答案上把 unlocatable claims 从 2/3 降为 0。不过，其余人工指标不是一致提高，且总体增量较小。
因此 H2 只能判为**部分描述性支持**，不应写成“A2 在全部维度全面提升”。

### 4.3 H3：完整组合

| metric | `old_old` | `new_new` | Δ |
|---|---:|---:|---:|
| claim atomic | 97.3% | 100.0% | +2.7 pp |
| claim self-contained | 87.8% | 96.9% | +9.0 pp |
| span aligned | 70.3% | 76.6% | +6.3 pp |
| rewrite faithful | 67.6% | 75.0% | +7.4 pp |
| complete fact coverage | 73.3% | 87.5% | +14.2 pp |

`new_new` 的所有预注册人工指标均不低于 `old_old`，且多项提高，H3 获得**描述性支持**。
但这是 A1 与 A2 的组合结果，不能把全部差值归因于 splitter。

## 5. 自动诊断

| arm | claims | split errors | unlocatable | deduplicated | meta filtered |
|---|---:|---:|---:|---:|---:|
| `old_old` | 74 | 3 | 2 | 1 | 3 |
| `old_new` | 77 | 3 | 0 | 0 | 3 |
| `new_old` | 61 | 0 | 3 | 0 | 0 |
| `new_new` | 64 | 0 | 0 | 0 | 0 |

unlocatable 的变化与 A2 的 span-anchoring 修复一致。split errors 只在接受 old-A1 answer 的两臂
出现，且 old/new A2 都为 3，因此不能把这项差异归功于 A2；它更可能反映 A1 输出形状差异。

## 6. 可用于论文的结论

在固定证据与模型的 50-case ASQA 受控消融中，verification-oriented A1 prompt 提高了句子的
自包含性、独立可验证性和引用格式合规率，并消除了已标注句中的未解析代词，但使回答平均长度减少
46.6%。A2 span-anchored splitter 在相同答案上稳定小幅提高 span alignment 与 source-fact
coverage，并消除了 unlocatable claims，但其他 claim-level 指标的变化不一致。完整新版组合在
全部预注册人工指标上不低于旧组合，其中 complete fact coverage 提高 14.2 个百分点。

该结论只适用于本次 calibration sample，属于描述性证据。它不证明统计显著性、跨数据集泛化、
最终答案正确性或下游 verification/repair 的改进。

## 7. 局限与后续工作

1. 只有 50 个 ASQA case，且属于 calibration data；不能当作独立 held-out test。
2. 只有一名标注者，无法估计 inter-annotator agreement。
3. A1 新版明显更短；事实覆盖没有恶化，但仍需在论文中明确质量—信息量取舍。
4. 当前只给出原始计数、比例与百分点差，没有预注册显著性检验；如补做 bootstrap，应明确标为
   exploratory analysis，不能改写预注册判断。
5. 后续若要证明系统收益，应固定 Retriever/Selector，单独评估最终 answer correctness、
   claim-level citation quality、coverage 与 abstention；这些不属于本次 A1/A2 结论。

## 8. 结果文件

正式结果位于 `results/generator-a1-a2-ablation/`：

- `annotation_blinded_final.xlsx`：冻结后的完整盲评工作簿；
- `annotation_unblinded.csv`：解盲后的逐项标注；
- `human_metrics.json`：完整机器可读指标与比较；
- `human_metrics_by_arm.csv`：论文制表用长表；
- `human_annotation_manifest.json`：标注冻结时间、计数与 SHA-256；
- `ablation_report.md`：由最终标注直接生成的简要统计报告；
- `manifest.json`、`cases.jsonl`、`outputs_by_arm.jsonl`、`raw_responses.jsonl`：原始实验证据。
