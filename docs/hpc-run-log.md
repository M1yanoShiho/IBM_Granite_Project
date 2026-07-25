# HPC run ledger (BluePebble)

规则(running-hpc-experiments):每个 HPC 实验 = **代码 + slurm 脚本 + 本台账条目** 三件套,
提交前齐备。提交前填 BEFORE(目的/假设/预期指标+方向/精确命令/commit hash);
拉回后填 AFTER(job id、raw 文件、headline 数字、写给 results-summary 的发现草稿)。
只存在于 `.out` 日志或散装 scp 文件里的结果不算已记录;raw 结果在 bp1 上
`git add -f results/...` 后经 git 拉回。

---

## E2 — gate-on/off 配对 selector 对照(spec §12 主对照)

**状态:** READY——三件套齐(configs + slurm + 本条目,commit 3c37d4c)。只差登录节点下 dpr-w100 + granite 后跑。

**BEFORE(预注册):**

- 目的/假设:同一 bm25 池、同 α,门(margin=2, support_cap=1)显著降低 harmful-in-context,
  且 Required Recall 非劣下界 −0.01(双 Gate)。**诚实预期:NQ 多为单 gold passage → 真答案 1 票 vs 孪生 1 票 = 1v1 → 门按设计双留不踢 → harm 可能不降(null),不是 bug**;pool-hit / gate-fire 诊断区分"门没用"与"池里没冲突可判"。
- 预期指标 + 方向:`selector.core.harmful_in_context` ↓(标签=注入 mutation log);guardrail `selector.core.conditional_document_recall` 非劣 ≥ −0.01;诊断:pool-hit rate、gate-fire。
- 精确命令(登录节点先建数据,再 sbatch):
  ```
  export PYTHONPATH=src HF_HOME=/user/work/$USER/hf_cache IR_DATASETS_HOME=/user/work/$USER/ir_datasets
  hf download ibm-granite/granite-4.1-3b
  python -m evidence_rag.materializer.base_cli --split dev --output runs/niah-base --corpus-size 100000 --query-limit 2000 --seed 42
  python -m evidence_rag.materializer.cli --base-manifest runs/niah-base/manifest.json --output runs/niah-injected --seed 42
  mkdir -p logs && sbatch scripts/run_selector_gate.slurm configs/experiments/niah_e2_gate_on.toml configs/experiments/niah_e2_gate_off.toml runs/niah-injected/provenance.jsonl
  ```
- Git commit:3c37d4c(configs+slurm);Seed:`[run] seed = 13`,harm-report seed 13。

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

## E2-lenient — lenient 聚类修复的因果度量(承接 E1 诊断)

**状态:** READY——config `niah_e2_gate_on_lenient.toml`(gate-on,equivalence=lenient,commit d4acd87)+
复用 `run_selector_gate.slurm`+ 本条目。设计:`specs/2026-07-23-lenient-clustering-selector-design.md`。
依赖 runs/niah-injected + E2 已记录的 exact 臂数(harm .569 / recall .820)。

**BEFORE(预注册):**

- 目的/假设:门内 lenient 聚类(rep-anchored 贪心)修自一致性计票 → **required recall 回升**(gold 碎片不再被误踢),
  **harm 不劣化**(gold/cf canonically 可分,lenient 不合并异值,植入冲突保留)。唯一变量=聚类等价(数据/检索/margin 全同)。
- 预期指标 + 方向:`conditional_document_recall` ↑(对比 exact 臂 .820,期望更接近 gate-off .868;至少不低于 .820);
  `selector.core.harmful_in_context` ≤ exact 臂 .569(不劣)。判定:recall 涨且 harm 不劣 = 修复成功。
- 精确命令(登录节点,gres 覆盖 gpu:1;lenient 作 ON、exact-gate-off 作 OFF,另与已记录 exact-on 数对比):
  ```
  mkdir -p logs && sbatch --gres=gpu:1 scripts/run_selector_gate.slurm \
    configs/experiments/niah_e2_gate_on_lenient.toml configs/experiments/niah_e2_gate_off.toml \
    runs/niah-injected/provenance.jsonl
  ```
- Git commit:d4acd87;Seed:`[run] seed = 13`,harm-report seed 13。

**AFTER(2026-07-23,job needle-gate lenient 臂;exact 臂复用原 E2 dump,corpus_snapshot hash 相同→同池):**
- 跑法:lenient 臂经 slurm 重跑(需先 `rm -rf runs/e2-gate-off/index runs/e2-gate-on-lenient/index` 清 pre-d5f7908 旧 schema index;
  gate-off 臂又撞 run_manifest hash 守卫,故改**离线**比:`harm_cli --selected-on <lenient> --selected-off <exact-gate-on>`
  + `paired_metric_cli --metric conditional_document_recall`(commit cffa2e4)。
- **Required recall +1.2pp**(exact .820 → lenient **.832**),paired **p≈0,CI[.006,.018]**,n=1848 —— **显著**。
- **Harmful-in-context +0.3pp**(exact .569 → lenient .572),paired p=0.55,CI[−.005,+.012] —— **不显著(无 harm 代价)**。
- 定论:门内 lenient 聚类**显著回收召回、零 harm 代价**,回收 −4.8pp 门代价的 ~25%;残差(~75%,含 .39 孪生 missed-conflict)= Graph 2.0 语义等价目标,已量化。发现落 `docs/results-summary.md` S3。
- 运维注:teammate retriever 大改(d5f7908)改了 IndexManifest schema → 旧 index cache 全失效,重跑须清 `runs/*/index`;run_manifest hash 守卫会拒重建的 index,离线比 selected dump 最省。

---

## E1 — 边/簇检测组件评估(spec §12,pool 级 B2 harness)

**状态:** READY——三件套齐:harness(cluster_eval + cluster_eval_cli + tests,commit
126c95f + d2e6631)、slurm(a19c90a)、本条目。§14 数据决定已由 niah-injected 落定(E2 已跑),
gold-alias 加载由 provenance sidecar(gold_value/gold_alias_used/replacement_value)解决,不再是 blocker。
依赖 E2 已产出的 `runs/e2-gate-on/candidate_sets.jsonl`(job 18130403)。
设计:`docs/superpowers/specs/2026-07-23-e1-cluster-eval-design.md`。

**BEFORE(预注册):**

- 目的/假设:量化 E2 池上 exact-string 答案簇的错误模式,把 E2 的 harm 收益 / recall 代价与簇错摘开。
  假设:missed-conflict 低(注入按设计使 gold 与 counterfactual canonically 可分)、needle-gold-recovery 高;
  **若 missed-conflict 偏高 → 门对部分注入题失明、E2 harm 收益被高估;若 recovery 偏低 → 簇指标被抽取失败主导,须如实标注。**
- 预期指标 + 方向:`selector.cluster.missed_conflict` ↓(主)、`selector.cluster.false_conflict` ↓、
  `selector.cluster.needle_gold_recovery` ↑,各带 Wilson 95% CI;**框架句 `injection selection-bias`
  (multi-canonical-key skip rate)必并列**——注入集已把别名假冲突过滤掉(injector `len(normalized)!=1` 跳过),
  低 conflict 值须据此解读,否则会被误读成"簇已验证正确"。标签=provenance + official gold_cases,零新增人工标注
  (与 ArbGraph Table 4 的 200 人工对形成方法学对照,呼应 judge-kappa≈0)。
- 精确命令(登录节点确认 E2 dump 在位后 sbatch):
  ```
  ls runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json runs/niah-injected/provenance.jsonl
  mkdir -p logs results && sbatch scripts/run_cluster_eval.slurm \
    runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json \
    runs/niah-injected/provenance.jsonl results/e1-cluster-eval/cluster_eval_report.json
  ```
- Git commit:harness 126c95f + d2e6631,slurm a19c90a;Seed:无(确定性抽取 + 计数,无随机化)。

**AFTER(2026-07-23,job cluster-eval,gres 覆盖为 gpu:1):**
- raw:`results/e1-cluster-eval/cluster_eval_report.json`。missed_conflict **.386**(n=941,CI[.355,.417])、
  false_conflict .594(n=836,文本代理 noisy)、needle_gold_recovery **.509**(n=1264,CI[.481,.536]);
  selection_bias multi_key_rate .106 / skip_rate .261。
- 定论:v1 exact-string 簇在机制层**没通过**——needle 在池里时抽取器只有 51% 能抽回 gold;missed-conflict .39 =
  门对孪生冲突两成半以上失明。E2 的 −11.2pp harm 有相当部分非来自有原则的冲突检测,−4.8pp recall 大部分是簇碎裂误杀。
- **诊断链(root cause):** needle-visibility 审计(`needle_visibility_cli`,CPU)显示 visible@600 **.949**、
  truncated .051、absent **0** → **不是截断/分块,是抽取能力**:答案 95% 可见却只 51% 抽出(conditional .536)。

**E1-diag — needle 抽取失败模式探针(systematic-debugging Phase 3):**
- 目的:visible-but-missed 里,3B 是输出 NONE(太保守 → prompt 免费修,8B 大概无用,呼应 [[progress-2026-07-06]] 8B 没帮助)
  还是 wrong-entity(真 QA 错 → 8B 才有戏)。只抽 needle 一条(~1264 call,~15-20min)。
- 指标:visible 子集内 recovered/wrong/none 分布。命令:
  ```
  mkdir -p logs results && sbatch --gres=gpu:1 scripts/run_needle_probe.slurm \
    runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json \
    runs/niah-injected/provenance.jsonl results/e1-needle-probe.json results/e1-needle-probe.jsonl
  ```
- Git commit:提交时填;Seed:无。**AFTER:** 未运行。

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
