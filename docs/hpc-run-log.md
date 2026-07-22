# HPC run ledger (BluePebble)

规则(running-hpc-experiments):每个 HPC 实验 = **代码 + slurm 脚本 + 本台账条目** 三件套,
提交前齐备。提交前填 BEFORE(目的/假设/预期指标+方向/精确命令/commit hash);
拉回后填 AFTER(job id、raw 文件、headline 数字、写给 results-summary 的发现草稿)。
只存在于 `.out` 日志或散装 scp 文件里的结果不算已记录;raw 结果在 bp1 上
`git add -f results/...` 后经 git 拉回。

---

## E2 — gate-on/off 配对 selector 对照(spec §12 主对照)

**状态:** BLOCKED——等待 spec §14 数据集决定(团队会议,selector README 六条)
产出两个 TOML;另外 config 路径 retriever 目前只注册了 bm25,dense 检索器注册属模块一。

**BEFORE(提交时补 commit/日期):**

- 目的/假设:同一重排配置下,门(margin=2, support_cap=1)显著降低
  harmful-in-context,且 required recall 满足非劣下界 −0.01(V1 协议双 Gate)。
  失效方向预期:投票碎裂域中门趋于沉默(退化为纯重排),而非误杀。
- 预期指标 + 方向:harmful-in-context ↓(标签=deterministic counterfactual
  mutation log);Required Evidence Recall@selected 非劣(≥ −0.01);
  次级:选择质量排序指标不显著劣化。
- 精确命令:
  `mkdir -p logs runs && sbatch scripts/run_selector_gate.slurm configs/experiments/<gate_on>.toml configs/experiments/<gate_off>.toml`
  (两 TOML 仅 [selector] 不同:`gated-corroboration` vs `corroboration`,
  alpha/top_n 相同;文件名在数据集冻结时定)
- Git commit:提交时填
- Seed:experiment config `[run] seed`,提交时填

**AFTER:** 未运行。

**E2 端到端管道现在全齐(2026-07-21,登录节点 CPU 步骤):**
1. 登录节点 `ir_datasets` 下载 dpr-w100 NQ(21M passage,大,一次性);
2. `evidence-rag-load-niah-base --split dev --output <base> --corpus-size 100000` → base 数据集;
3. `evidence-rag-materialize-niah --base-manifest <base>/manifest.json --output <inj>` → 注入反事实 + provenance;
4. 两个 config TOML(gate-on `gated-corroboration` / gate-off `corroboration`,指向 `<inj>`)→ slurm 两臂 dump selected;
5. `evidence-rag-harm-report --selected-on ... --selected-off ... --provenance <inj>/provenance.jsonl --candidates ...` → Harmful on/off + pool-hit + 配对 p/CI。
唯一未做 = 真实数据下载 + 跑(ops,不是代码)。

**harm 指标已就绪(2026-07-21):** `evidence-rag-harm-report`(离线,读两臂 selected_evidence_sets.jsonl
+ provenance.jsonl → Harmful Rate on/off + pool-hit + 配对随机化 p + bootstrap CI)。E2 跑法:
现有 slurm 两臂 dump selected → `evidence-rag-harm-report --selected-on ... --selected-off ... --provenance ...`。
仍 BLOCKED 于基座数据集(Materializer A)。

---

## E1 — 边/簇检测组件评估(spec §12 增补,导师要求)

**状态:** BLOCKED——依赖 (1) §14 数据集决定;(2) 组件评估 harness 尚未实现
(false-conflict / missed-conflict rate 计算器 + gold-alias 加载)。
**三件套约定:** harness 实现时,其 slurm 与本条目的 BEFORE 一并补齐后才可提交;
不为不存在的入口预写脚本(避免虚构 CLI 参数)。

**BEFORE(harness 落地时补全):**

- 目的/假设:answer-equivalence 聚簇的 false-conflict rate(同池两条均含
  gold alias 的候选被劈进不同簇)足够低,使门的误杀通道可控;
  与 ArbGraph Table 4(200 人工对,96%)形成零人工标注的方法学对照。
- 预期指标 + 方向:false-conflict rate(首要,越低越好;阈值在数据集冻结时预注册)、
  missed-conflict rate;标签=official gold aliases + counterfactual mutation log。
- 精确命令:harness 落地时填。
- Git commit / Seed:提交时填。

**AFTER:** 未运行。

---

## E3 — coverage-on/off 配对(A2 互补覆盖层因果贡献)

**状态:** BLOCKED——依赖 (1) §14 数据决定,且需**互补压力数据**(答案受益于多条支撑的题;
单段可答反事实集发挥空间小,见 A2 spec §9);(2) 复用 run_selector_gate.slurm 三件套,
加第三个 config 臂(`gated-coverage-corroboration`)。

**BEFORE(数据落地时补):** 目的=固定门,coverage-on(`gated-coverage-corroboration`)vs
coverage-off(`gated-corroboration`)配对,主指标 grounded-answer F1/faithfulness/cover-EM
(承接 finding 17 k=10);guardrail required recall 非劣(pin 结构上保证,需实测);
辅助:选中集近重复对数下降。**AFTER:** 未运行。

## E1-support — support 边检测准确率(A2 spec §9,导师三边评估的第三边)

**状态:** BLOCKED——需带 official supporting-fact 标注的数据。
**BEFORE(补):** 实体/数值重叠 support-边 precision/recall(零人工标注,用 official
supporting-fact 标签);与 conflict 边(false/missed-conflict)、duplicate 边(去重单测)
并列成 ArbGraph Table 4 式三边准确率表。**AFTER:** 未运行。

## 本地(非 HPC)验证记录

- 2026-07-20:selector 门实现全套单测 LOCAL 通过(tests/selector 32 + registration 9),
  mypy strict 干净(49 files);全套 pytest 与基线逐项对比:失败集恒为 29 条
  已知 Windows TOML fixture 问题(与 selector 无关,已开修复任务),零回归。
- 2026-07-21:A2 互补覆盖层实现 LOCAL 通过(coverage 引擎 11 + 覆盖选择器 2 + 注册 1 = +14);
  gated.py 抽出 `_gate` 复用、门 parity 保持;全套 **332 passed / 1 xfailed**(基线 318/1 + 14,零回归);
  mypy strict 干净(51 files),ruff 干净。Windows fixture 问题已被队友修掉,不再计入。
