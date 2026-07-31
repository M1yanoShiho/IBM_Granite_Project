# HPC run ledger (BluePebble)

规则(running-hpc-experiments):每个 HPC 实验 = **代码 + slurm 脚本 + 本台账条目** 三件套,
提交前齐备。提交前填 BEFORE(目的/假设/预期指标+方向/精确命令/commit hash);
拉回后填 AFTER(job id、raw 文件、headline 数字、写给 results-summary 的发现草稿)。
只存在于 `.out` 日志或散装 scp 文件里的结果不算已记录;raw 结果在 bp1 上
`git add -f results/...` 后经 git 拉回。

---

## OCR-smoke — PDF 内嵌图表 caption + 图内 OCR 功能冒烟

**状态:** READY——三件套齐(config `configs/ingestion/ocr_smoke.toml` + slurm
`scripts/run_ingest_ocr_smoke.slurm` + 造样脚本 `scripts/make_ocr_smoke_pdf.py` + 本条目)。
这是**功能冒烟**(能否端到端跑通新 OCR 路径),不是度量实验,无指标对照。

**BEFORE(预注册):**

- 目的/假设:验证新增的“PDF 内嵌图表 caption + 图内 OCR”在真实 Docling OCR + Granite
  Vision 下端到端跑通。触发链:`caption_pdf_pictures=true` → `generate_picture_images=True`
  建 converter → `extract_pictures` 抽内嵌图 → phase2 Vision caption → phase3
  `extract_ocr_text_from_image` OCR 同图 → 文档 text 追加 `Text in image:`。
- 判据(非指标,PASS/FAIL):`runs/ocr-smoke/out/documents.jsonl` 存在 image 源文档,其
  `text` 含 `Text in image:` 且 OCR 段含图内 sentinel 数字 `42`(sentinel
  `REVENUE 2024 42 PERCENT` 只画在图里、不在 PDF 正文,故命中即证明 OCR 生效)。作业内嵌
  断言,PASS 退出 0。
- 精确命令(登录节点预取后):
  ```
  # 登录节点(有网,一次性):
  source /user/work/$USER/venv/bin/activate
  export HF_HOME=/user/work/$USER/hf_cache
  pip install -e '.[ingestion]' && pip install fpdf2
  hf download ibm-granite/granite-vision-3.3-2b
  docling-tools models download -o /user/work/$USER/docling_artifacts layout tableformer easyocr
  PYTHONPATH=src python scripts/make_ocr_smoke_pdf.py
  # 提交:
  mkdir -p logs runs/ocr-smoke && sbatch scripts/run_ingest_ocr_smoke.slurm
  ```
- Git commit:待 OCR 特性(feat(ingestion))提交后填其 hash 并在 bp1 `git pull`。

**AFTER:** 未运行。<!-- 填:job id、OCR SMOKE PASS/FAIL、raw = runs/ocr-smoke/out/documents.jsonl -->

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

## E3-prompt — 孪生 missed-conflict 的 prompt 探针(systematic-debugging Phase 3)

**状态:** READY——harness(missed_conflict_probe + cli + tests,commit 444d2d8)+ slurm + 本条目。
承接 E1 顽疾(missed_conflict .386,lenient 修不了孪生崩塌)。全建 Graph 2.0/NLI 前的低成本判据。

**BEFORE(预注册):**

- 目的/假设:孪生崩塌 because 抽取器抓**共享的未替换实体**而非被查询属性的值;针对性
  "逐字 copy 回答问题的那个值"prompt 能把孪生分开(needle→gold、cf→replacement)→ missed_conflict 降。
- 预期指标 + 方向:每 prompt 的 `missed_conflict_rate` ↓、`needle_gold_rate`/`cf_replacement_rate` ↑。
  baseline(现行 EXTRACT_PROMPT)vs verbatim / attribute 两个 targeted。**判定:某 targeted 显著降 missed →
  prompt fix,拿 Graph 2.0 冲突检测主收益而不建 NLI;都不降 → Graph 2.0 有据。** baseline missed 应 ≈ E1 的 .386(sanity)。
  标签=provenance(gold_value/replacement_value);从 needle/cf 源文档直抽(隔离 prompt,去掉检索变量)。
- 精确命令(gres 默认已是 gpu:1):
  ```
  mkdir -p logs results && sbatch scripts/run_missed_conflict_probe.slurm \
    runs/niah-injected/manifest.json runs/niah-injected/provenance.jsonl results/e3-missed-conflict-probe.json
  ```
- Git commit:444d2d8;Seed:无(确定性分类)。

**AFTER(2026-07-25,job missed-conflict-probe,gpu:1,n=1479;walltime 40m→6h 后跑通):**
- raw:`results/e3-missed-conflict-probe.json`。每 prompt(missed_conflict / needle_gold / cf_replacement rate):
  baseline .281 / .486 / .211;**verbatim .238 / .363 / .183**;**attribute .254 / .479 / .269**。
- 定论:**prompt 修不了孪生崩塌。** verbatim 的低 missed 是**假胜**——靠抽取变噪(needle_gold −12.3pp)而非正确分离,弃。
  attribute 是**真但小**的收益:missed −2.7pp、**cf_replacement +5.8pp**(cf 更常抽出注入 replacement)、gold 持平——正确方向但幅度小。
- 最优正确 prompt 仍留 ~25% missed → 孪生崩塌是**抽取能力**受限(swapped value 只 ~48% needle / ~27% cf 抽得出),非 prompt 可修。
- **Graph 2.0 判定:** 按预注册 justified,**但带诚实 caveat**——missed 是抽取问题,NLI 坐在抽取之上,只在 claim 级抽取比单答案更 twin-robust 时才帮;是经验赌注非保证。发现落 results-summary S4。attribute 可作廉价小 win(但改共享 EXTRACT_PROMPT 会波及全抽取+metric,需重跑确认)。

---

## E3-2x2 — 8B × decoupled 抽取(Phase 3 follow-up,Graph 2.0 前的最后判据)

**状态:** READY——harness(decoupled 策略 + --limit + model-arg slurm,commit d56bb5a)+ 本条目。
承接 S4(missed 抽取受限):从**容量**(8B)和**结构**(解耦抽取)两轴攻 missed_conflict。用户已定:此探针后**上马 Graph 2.0**。

**BEFORE(预注册):**

- 目的/假设:missed_conflict 受抽取限。**解耦抽取**(Stage A 一次识别被查询目标 → Stage B 逐段抽该目标值)让抽取锁定 swapped value 而非共享实体 → missed 降;8B 提供容量。诚实预期:结构(decoupled)动得比容量(8B)多;8B 单独弱(twin 轴未测但 recovery/matching 轴三次否掉)。
- 预期指标 + 方向:2×2 = {3B, 8B} × {attribute 单段, decoupled 两段}(+baseline 参照),每格 `missed_conflict_rate` ↓、`needle_gold`/`cf_replacement` ↑。**判定:某格显著低于 baseline .281 且 recovery 不塌 → 廉价/结构性修复(可能 S5);都不动 → Graph 2.0 的抽取瓶颈坐实,NLI 有据。** 子采样 500 足够定方向(SE ~.02)。从源文档直抽,隔离检索。
- 精确命令(登录节点先 `hf download ibm-granite/granite-4.1-8b`;args=MANIFEST PROV OUTPUT [MODEL] [LIMIT]):
  ```
  mkdir -p logs results
  sbatch scripts/run_missed_conflict_probe.slurm runs/niah-injected/manifest.json runs/niah-injected/provenance.jsonl results/e3-2x2-3b.json ibm-granite/granite-4.1-3b 500
  sbatch scripts/run_missed_conflict_probe.slurm runs/niah-injected/manifest.json runs/niah-injected/provenance.jsonl results/e3-2x2-8b.json ibm-granite/granite-4.1-8b 500
  ```
- Git commit:d56bb5a;Seed:无(确定性分类;子采样取前 N 确定)。

**AFTER(2026-07-25,两轮;raw `results/e3-2x2-{3b,8b}-v2.json`):**
- **第一轮作废(探针两个缺陷,commit 1589a70 修):** (1) decoupled 的 Stage B **没传 question**(只给目标类型 → 无法判别 passage 里哪个实体,8B gold 塌到 .024);(2) `classify_pair` 只用 exact 等价 → **系统性惩罚"更啰嗦但正确"的输出**(8B exact→lenient gold 差 **+56~63pp**)。现报告同时含 exact/lenient,并加 `--dump` 存原始回答。
- **v2 结果(lenient 可信列,n=500/格):** 3B baseline/attribute/decoupled missed=.318/.286/.190、gold=.576/.562/.442;8B=.444/.232/**.224**、gold=.664/.650/**.690**。
- **定论(交互效应):** 容量单独无效(8B baseline missed 最差 .444);结构单独无效(3B decoupled gold 塌 .576→.442 = 假胜);**容量×结构真胜——8B+decoupled missed .444→.224(−22pp)且 gold 持平微升、cf 抽取升**。SE≈.02,22pp 远超噪声。
- **推翻 S4 的"prompt 修不了孪生崩塌"**(只在 3B 成立且受 exact 计分混淆);**也推翻"8B 无用"**——前三次否决只在 recovery/matching 轴,twin 轴 + 正确计分下 8B 有用。发现落 results-summary **S5**,S4 已加更正标注。
- **对 Graph 2.0:** 应建在更强抽取之上,其增量须对照"8B+decoupled"新基线。**限制:** 直抽源文档非全管道、未做配对显著性、未验证级联到 in-pool 指标。

---

## E1-cascade — S5 收益能否级联到 in-pool 指标(Graph 2.0 M0 前的最后验证)

**状态:** READY——harness(cluster_eval 加 `--extraction decoupled` + exact/lenient 双计分 + `--limit`,
model 走 slurm 参数;commit 82ee0e7)+ 复用 run_cluster_eval.slurm + 本条目。
依赖 `runs/e2-gate-on/candidate_sets.jsonl`(E2 池)+ runs/niah-injected。

**BEFORE(预注册):**

- 目的/假设:S5 在**源文档直抽**下把孪生 missed 从 .444 砍到 .224(8B+decoupled)。本实验验证该收益能否**级联到真实检索池**
  (top-20 窗口内、含干扰段)。假设:in-pool `missed_conflict` 与 `needle_gold_recovery` 同向改善;但 pool 内多了 18 条干扰段,
  **诚实预期收益打折**(decoupled 的 Stage A 目标只有一个,窗口内其他段可能被强行抽出该类型的值 → false_conflict 可能升)。
- 预期指标 + 方向:对照 **3B+single**(E1 原臂,lenient 重算)vs **8B+decoupled**:`missed_conflict` ↓(主)、
  `needle_gold_recovery` ↑;监控 `false_conflict`(可能升=代价)。**均以 lenient 列为准**(S5 教训:exact 计分对啰嗦输出有 +56pp 偏见);
  exact 并列报告。判定:missed 显著降且 recovery 不塌 → S5 级联成立,Graph 2.0 须以此为新基线。
- 精确命令(8B 必须 a100,见 [[hpc-cluster-gres]];limit 300 控 8h walltime,~6.3k calls):
  ```
  mkdir -p logs results
  # 基线臂(3B+single,lenient 重算):
  sbatch --gres=gpu:1 scripts/run_cluster_eval.slurm runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json runs/niah-injected/provenance.jsonl results/e1-cascade-3b-single.json ibm-granite/granite-4.1-3b single 300
  # S5 臂(8B+decoupled):
  sbatch --gres=gpu:a100:1 scripts/run_cluster_eval.slurm runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json runs/niah-injected/provenance.jsonl results/e1-cascade-8b-decoupled.json ibm-granite/granite-4.1-8b decoupled 300
  ```
- Git commit:82ee0e7;Seed:无(确定性;子采样取前 300 确定)。

**AFTER(2026-07-26,两臂 n=300;raw `results/e1-cascade-{3b-single,8b-decoupled}.json`):**
- **lenient 列:** missed_conflict .427[.358,.499] → **.308**[.253,.369](−11.9pp);needle_gold .609[.547,.668] → **.673**[.613,.729](+6.4pp);
  **false_conflict .494[.421,.568] → .877[.825,.916](+38.3pp)**。
- **定论:S5 收益不级联 —— 这是一个干净的负结果。** 三项里唯一**统计确凿**的是**坏的那项**:missed/recovery 的 CI 互相重叠(仅提示性),
  而 false_conflict 的 CI **完全不重叠**,退化无疑。
- **机制(预注册的担忧成立且远超预期):** decoupled 的 Stage A 每题只命名**一个**目标类型,再套到全部 20 段。孤立探针里只有 needle+cf 两段所以无害;
  真实池里其余 18 条干扰段被**逼着**吐出该类型的某个值 → 含 gold 的段与 needle 抽出不同答案 → **gold 碎裂率 88%**。
  而 false_conflict 正是 E2 −4.8pp recall 的机制通道 → 把 8B+decoupled 接进门**大概率使 recall 更差**。
- **S5 为真但不外推到两段以上**;孤立探针高估了它,因为**池结构**才是破点。
- 另:8B 的 exact→lenient recovery 差再次巨大(.093→.673),双计分必要性再确认。`selection_bias.skip_rate` 在 `--limit` 下曾误报(300/2000=.85),已修为按全量算(真值 .261);`multi_key_rate` .106 不受影响。
- **限制:** 两臂 denominator 不同(185 vs 237,"both clustered"条件随抽取策略变)→ 未做配对检验,只用各臂 Wilson CI 比较。发现落 results-summary **S6**。

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

## G1 — Generator B 验证器分诊,GPU 臂(Half 1.6 arms D-true / E-granite 3b+8b)

**状态:** READY——三件套齐:代码(`scripts/verifier_triage.py`)、slurm
(`scripts/run_verifier_triage.slurm`)、本条目。CPU 臂(base / large / minicheck)已在
本机跑完,结果 `docs/generator/verifier-triage-results.md`;本条目覆盖装不进 16GB CPU
的 GPU 臂:`true` + Granite-as-judge 两档 `granite3b` / **`granite8b`(本轮新增,容量对照)**。
三臂**顺序加载、臂间释放**(`_free_gpu`),单张 **a100** 一次只驻留一个大模型,故一作业跑全。
**commit hash:`c470b68`**(code + slurm;本条目在其后随台账提交)。

**BEFORE(预注册):**

- 目的/假设:SciFact 那轮(Half 1.5)测出通用 NLI 在 entailment 召回上只有 0.34 上限,
  怀疑**是模型族选错而不只是 checkpoint 选小**——真正的任务是 document-grounded
  verification,不是 sentence-pair NLI。本组上 `google/t5_xxl_true_nli_mixture`
  (ALCE 自己的引用评测所用的 TRUE 判定器,11B / fp32 42.5GB,bf16 约 21GB)与
  Granite-as-judge,和已跑完的 CPU 三臂**共用同一 pair set**做对照。**新增 `granite8b`
  是把"LLM-as-a-judge 路子不对"与"3B 太小"这两种失败摘开**——正是 Selector 团队
  (E3-2x2)遇到并用"两档都测"解决过的容量×方法混淆;3B 判官若输给 MiniCheck,单看
  一档无法区分是方法错还是容量小。Granite 相对 MiniCheck 的结构优势:三分类,能填
  `ClaimVerification.contradicted`,且不引入 self-host 之外的新模型依赖。
  **诚实预期:TRUE 是二分类(1/0),没有 contradiction 类**,无法给 `contradicted`
  信号;若 TRUE 召回也上不去,则问题不在模型族而在 pair set 或任务本身,须如实标注。
  **对 8B 的诚实预期:8B 可能在召回上胜过 MiniCheck 但每对成本高得多——那样结论就从
  "明确胜出"退成"成本/质量权衡";若 8B 相对 3B 没有实质提升,则指向判官方法本身而非容量。**
- 预期指标 + 方向:ASQA `entailment recall` ↑(主,对照 CPU 最好臂)、
  `hard-neutral -> entail` ↓(引用精度的分母)、2Wiki `atomic recall (any single chunk)` ↑、
  `residual multi-hop` ↓;两档 Granite 的 `hard-neutral -> contra` FP 率须清算(通用 NLI
  在此 28% / 44.7% 被判不可用,判官要过这条线才有价值);诊断:union diagnostic
  (把"分解没降维"与"验证器看不见已在的支撑"摘开)、**两档 Granite 的 ms/pair**
  (成本一侧要可见)。
- 精确命令(登录节点先下模型 + 建数据,再 sbatch;slurm 已请求 `--gres=gpu:a100:1`,
  `--arms true,granite3b,granite8b`):
  ```
  export HF_HOME=/user/work/$USER/hf_cache MODEL_CACHE_DIR=/user/work/$USER/hf_cache
  export TRIAGE_DATA_DIR=/user/work/$USER/triage_data
  hf download google/t5_xxl_true_nli_mixture
  hf download ibm-granite/granite-4.1-3b
  hf download ibm-granite/granite-4.1-8b
  PYTHONPATH=src python -c "import sys; sys.path.insert(0,'scripts'); import verifier_triage as v; v.ensure_asqa(); v.ensure_2wiki()"
  mkdir -p logs results && sbatch scripts/run_verifier_triage.slurm \
    docs/generator/verifier-triage-results-hpc.md results/verifier-triage/scores.jsonl
  ```
- Seed:`--seed 13 --n 150 --multihop 60`,**必须与 CPU 那轮完全一致**,否则 pair set
  不同、两轮不可比(pair set 由这三个参数确定性构造)。
- 数据合规:只用 ALCE/ASQA + 2WikiMultihopQA + 合成实体替换;**HotpotQA / RGB /
  MuSiQue-Full 是封闭最终测试集,脚本里从不加载**。

**AFTER(2026-07-28,job `18200000`,`gpu:a100:1`,bp1-gpu035,1189 对/臂,walltime ~3.4h):**

- Raw:结果 `docs/generator/verifier-triage-results-hpc.md`;逐对分数
  `results/verifier-triage/scores.jsonl`(HPC `/user/work/ri25947/IBM_Granite_Project/`,488KB;
  本地副本 `results/verifier-triage/scores-hpc.jsonl`,未入库大数据留 HPC)。
- 两次前置失败(**均为环境/依赖 bug,不碰任何测量口径**,seed/n/multihop/prompt/阈值/pair
  构造全原样):(1) job `18194795` — `MODEL_CACHE_DIR` 作为 transformers `cache_dir=` 传入时
  须指到 **hub 层**(`$HF_HOME/hub`),BEFORE 命令里少一层 `hub` → offline 找不到缓存;slurm
  已修。(2) job `18197370` — TRUE 权重是 `pytorch_model-*.bin`(pickle),新版 transformers
  因 CVE-2025-32434 拒绝对 `.bin` 用 `torch.load`,须 **torch ≥ 2.6**;venv 由 2.5.1 升到
  `2.6.0+cu124`(granite 两臂 safetensors 不受影响)。
- **Headline(六臂,ASQA 召回 / hard-neutral→entail FP / 推导引用精度,1-in-5 假设):**

  | 臂 | 召回 | hard-neu FP | 引用精度 | ms/对 | 备注 |
  |---|---|---|---|---|---|
  | base | 0.467 | 0.007 | 0.946 | 173 | CPU |
  | large | 0.540 | 0.073 | 0.648 | 503 | CPU |
  | minicheck | 0.620 | 0.020 | 0.886 | 539 | CPU,实用最优 |
  | **true** | **0.747** | **0.007** | **0.966** | 264 | **精度天花板** |
  | granite3b | 0.767 | 0.067 | 0.742 | **114** | LLM 判官 |
  | granite8b | 0.900 | 0.240 | 0.484 | 256 | 召回最高、精度崩 |

- **发现草稿(给 results-summary):**
  1. **模型族假设在顶端坐实。** TRUE 召回 0.747(> minicheck 0.620)且 hard-neutral FP 仍
     0.007(与 base 并列最干净)→ 推导引用精度 **0.966,六臂最高**。Half 1.5 的 0.34 主要是
     *模型族*(通用 NLI)而非领域的问题:换 grounding/citation-专用判定器把召回和精度同时抬起来。
  2. **容量对照给出决定性结论:是"LLM-as-a-judge 路子不对",不是"3B 太小"。** 3B→8B **没有**
     修好判官的 FP,反而大幅恶化 hard-neutral→entail 0.067→**0.240**;8B 用"更宽松"换召回
     0.900,把引用精度砸到 **0.484**。容量×方法混淆解开,判在"方法"一侧。
  3. **Granite `contradicted` 有值但边际。** hard-neutral→contra:true 0.000(二分类,无此类)/
     granite3b 0.020 / granite8b 0.087。Granite 确实能产出 contradiction(MiniCheck 二分类不能),
     且 0.087 远低于通用 NLI 的 **28%(base)/44.7%(large)** 不可用线——即 Granite 过了那条线;
     但这点唯一独有信号,是和一个"随规模恶化、砸引用精度"的 entail-FP 捆绑来的。
  4. **Granite 无法做阈值救援。** 两档 Granite 的 P(entail) 阈值扫描全程**持平**(一词输出近似
     0/1 退化),TRUE 扫描有效(召回 0.593–0.880 可换 FP 0.000–0.053)。又一条 LLM-as-judge 的减分。
  5. **反事实(实体替换)同向。** verifier-alone:true 0.963(最好档)> granite3b 0.945 >
     granite8b 0.872(又是宽松 8B 漏更多);entity_check 叠加后三臂皆 1.000。
- **结论:** 主判官保持 NLI/grounding-专用一路。TRUE 是精度天花板(0.966)但 11B/贵;MiniCheck
  仍是实用 CPU 选择(0.886)。**Granite-as-judge 在两个规模点上均被否决为主判官**——其唯一独有的
  `contradicted` 信号边际且随规模恶化,并捆绑一个摧毁引用精度的 entail-FP。

---

## G2 — 采用 TRUE 为主判官,跑 Half 2(B4 completeness + B5 全链)

**状态:** READY——三件套齐:代码(`src/evidence_rag/generator/nli.py` 提升 TRUE/MiniCheck、
`verified.py` 默认切 TRUE、runner `scripts/verified_generator_calibration.py`)、slurm
(`scripts/run_verified_generator.slurm`)、本条目。**commit hash:`ccafdc0`**(代码+slurm+测试+
`docs/generator/verifier-backends.md`;本条目在其后随台账提交)。G1 的 commit-hash 行当时留了占位、
事后才补;本条目按规矩**提交前就填实**,不重复那个做法。

**BEFORE(预注册):**

- 目的:G1 定案 TRUE 为主判官(六臂:召回 0.747 / 引用精度 0.966,MiniCheck 0.620 / 0.886)。
  本组把 TRUE + MiniCheck 从 `verifier_triage.py` 的臂**提升进生产 `nli.py`**(打分逐字照搬,
  避免与 G1 测量漂移),TRUE 设默认、MiniCheck 留 CPU 回退、DeBERTa 保留;阈值取 G1 TRUE 扫描的
  **0.50**(召回 0.747 / hard-neutral FP 0.007)。然后首次用**真 Granite** 跑 Half 2 的 B4/B5。
- 诚实预期:
  - **B4 completeness 首次真跑,最可能错的不是覆盖分类而是 gap 问题质量**——问题若空泛
    ("需要更多信息")则 `evidence_recheck` 无从搜起。故报覆盖准确率**之外**还要报 gap 问题的
    具体性、约束(年份/地域)是否活着进入问题,并附逐字样例。
  - **B5 共驻是最可能失败处**:链内 Granite(draft/completeness/recheck)与 TRUE(attribution)
    **逐查询交替**,completeness 与 recheck 在 TRUE 之后再次调用 Granite,故无法拆成"先生成后验证"
    两段而不重写编排。决定:**单张 a100 同时驻留两模型**(granite-3b ~6GB + TRUE ~21GB ≈ 28GB),
    并用 `torch.cuda.max_memory_allocated` **证明**而非假设其放得下(不用 3g.40gb MIG 切片,正是为
    避开 selector-gate 那次 40GB 双载 OOM)。
  - **`contradicted` 在 TRUE 下不计算**:TRUE 二分类无 contradiction 类,B5 中 `contradicted` 计数
    预期为 **0**——这是接口后果,不是 bug(见 `docs/generator/verifier-backends.md`)。
  - **repair 触发率若在真数据上几乎不触发,本身就是关于整套方法的发现**,须如实报,早知比晚知好。
- 预期指标 + 方向:B4——own-fact 覆盖率(gold 陈述,越高越好)、foreign-fact 未覆盖率(越高越好)、
  gap 问题空泛率(越低越好)、约束存活率;B5——draft/faithful 声明数、supported 比例、unsupported
  被 repair 丢弃、entity 不一致捕获、completeness gap 数、gap 被 recheck 补上比例、repair 触发率、
  诚实弃答率、`GenerationResult` 合同保持率、ms/case。硬约束须**断言而非假设**:无新检索
  (recheck 只引已选证据 id)、单轮、非空答案≥1 引用/空答案 0 引用。
- 精确命令(模型与 ASQA 数据 G1 已缓存,无需 prefetch;单 a100 同驻两模型):
  ```
  mkdir -p logs results && sbatch scripts/run_verified_generator.slurm \
    docs/generator/verified-generator-results-hpc.md results/verified-generator/cases.jsonl
  ```
- Seed:`--seed 13 --limit 60 --top-k 5`。
- 数据合规:只用 ALCE/ASQA;**HotpotQA / RGB / MuSiQue-Full 从不加载**。

**AFTER(2026-07-29,job `18206324`,MIG `3g.40gb`,60 用例,seed 13,walltime ~2.2h):**

- Raw:结果 `docs/generator/verified-generator-results-hpc.md`;逐例 `local/raw-results/
  verified-generator-cases-hpc.jsonl`(HPC `results/verified-generator/cases.jsonl`);
  日志 `local/raw-results/verified-generator-18206324.out`。
- 三次投递,前两次失败**均非测量口径问题**:(1) `MODEL_CACHE_DIR` hub 层(G1 已修,沿用);
  实际首崩是 (2) job 18203195 — **真 Granite 输出 `合法 JSON + 尾部解释`**,三处严格 `json.loads`
  抛 `Extra data`,B5 首查询即断。修:`generator/json_parsing.parse_json_object` 容忍尾部散文/
  代码围栏(commit `872ed61`),runner 加固(B4 先落盘、每例 try/except 记 chain-error)。
- **Task 2 共驻已证明:** `[mem] peak allocated 27.8GB / reserved 27.9GB of 39.2GB`(granite-3b +
  TRUE 同驻)。**注:调度实际给的是 3g.40gb MIG 切片(39.2GB),不是整卡 a100**——两模型峰值
  27.8GB 仍留 ~11GB 余量,**连 40GB 切片都放得下**,比预注册预期更宽松;slurm 里"须整卡"的
  注释偏保守,已在结果 md 如实标注。链内 Granite/TRUE 逐查询交替、无法拆段的判断成立。
- **B4 completeness(仅 Granite):** own-fact 覆盖 **0.786**(gold 陈述仍有 ~21% 被误判未覆盖=假
  gap);foreign-fact 未覆盖 **1.000**(从不假称覆盖缺失事实);**gap 问题空泛率仅 0.067**——指南
  最担心的一项反而是强项,问题具体可答(逐字样例见结果 md);约束(年份)存活 **0.423**,但这项被
  我合成方式拖累(把本例年份挂到**无关的** foreign fact,如给 Rock Hall 事实挂"as of 1880"本就荒谬),
  是下限而非真实力。
- **B5 全链(Granite + TRUE):** 60 例 **49 跑通 / 11 报错(18%)**。faithful 声明 69:supported
  **0.333**、unsupported 被 repair 丢弃 **0.667**(TRUE 很严,2/3 声明证据撑不住);`contradicted`
  **0**(TRUE 二分类,符合预期);entity 不一致捕获 **47**(实体层很活跃);completeness gap **88** 找到、
  recheck 补上 **0.477**;**repair 触发 0.673**(指南担心"几乎不触发"——真数据上恰相反,2/3 会改答案);
  **诚实弃答 0.571**(过半空答案:TRUE 严 + 必填事实缺失触发弃答);`GenerationResult` 合同 **1.000**;
  ~9.6s/例。
- **chain-error 是第二波 fake-vs-real 分歧(11 例):** 7× `claim source_text 不在 answer_text`
  (Granite 的 source_text 是**改写**而非答案里的逐字子串,claim_splitter 要求精确子串)、2× recheck
  缺 `found` 布尔、1× recheck `found=false 却带答案片段`、1× 连容忍解析都找不到 JSON 对象。
- **发现草稿(给 results-summary):**
  1. **B4 gap 问题质量过关**——空泛率 6.7%、foreign 精度 1.000、own 召回 0.786,证明 completeness 能给
     `evidence_recheck` 喂具体子问题;弱点是约束注入(合成 artifact,非组件缺陷)。
  2. **B5 端到端能跑,行为由 TRUE 的严格度主导**:仅 1/3 声明被撑住 → repair 高触发(0.673)、弃答高
     (0.571)。这是"宁可不答也不乱引"的设计在真数据上的表现,不是 bug;但也说明 top-5 选证据对多数
     ASQA 声明支撑不足,值得回看 Selector/证据粒度。
  3. **TRUE 生产操作点 0.50 实测偏严**:配合必填事实弃答逻辑,过半问题弃答。是否放宽阈值需与"引用精度
     0.966"权衡,留 held-out 前决定。
  4. **claim_splitter 逐字子串约束是下一个鲁棒性靶子**——真 Granite source_text 常改写,占 chain-error
     的 7/11;修好可把完成率从 82% 拉高。recheck JSON 契约(found 布尔 / found=false 不带内容)次之。
- 硬约束保持:recheck 只引已选证据 id(代码校验未触发违规)、单轮、合同 1.000;`--seed 13` 未变。

---

## Graph 2.0 M0 — 预注册条目(BEFORE 已填,AFTER 待回填)

协议 `g2-proto-1`([docs/selector/M0_PROTOCOL_FREEZE.md](selector/M0_PROTOCOL_FREEZE.md))。
台账规则:提交前填目的/假设/**预期指标与方向**/精确命令/commit hash,拉回填 AFTER。
**只存在于 `.out` 或散装 scp 文件里的结果不算已记录** —— S6 就是这么丢的,`results/` 被 gitignore,
回传后须 `git add -f`。

### R001 — G-FC 固定分母基线(关键路径)

- **目的:** 测出 Graph 1.0-lenient 在**固定分母**口径下的 false_conflict 基线,据此填 M0 §3.5 的 δ。
  这是 M0 从 DRAFT 转 FROZEN 的唯一剩余阻塞。
- **假设:** 固定分母口径的 rate 与现行条件性口径不同(分母更大,因为不再要求其他段抽取成功),
  但**两个口径都必须报**,以便与 S1–S6 的历史数字对照。
- **预期方向:** 无方向性预期 —— 这是基线测量,不是对照。**任何"预期"在这里都是污染。**
- **命令:**
  ```
  mkdir -p logs results && sbatch scripts/run_cluster_eval.slurm     runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json     runs/niah-injected/provenance.jsonl results/r001-e1-baseline.json     ibm-granite/granite-4.1-3b single 0 results/r001-e1-baseline-dump.jsonl
  ```
  拉回后纯 CPU 重打分:
  ```
  python -m evidence_rag.evaluation.cluster_rescore_cli     --dump results/r001-e1-baseline-dump.jsonl     --output results/r001-gfc-baseline.json     --per-query results/r001-gfc-per-query.json
  ```
- **commit:** 10289bc(rescore 工具)/ 70fca84(`--dump`)/ 7c15460(固定分母判定)
- **AFTER:** _待填 —— `lenient.fixed_false_conflict` 的 rate、n_scored、n_fixed_eligible,以及推出的 δ_

### R001b — parent 碰撞率 + `support_unit=parent` 基线臂

- **目的:** (a) 量出 top-20 里同 Wikipedia 条目的碰撞率;(b) 跑出与 Graph 2.0 同 `support_unit` 的
  Graph 1.0-lenient 对照臂 —— 主对照两侧必须同单位,否则 SAME_SOURCE 修复的收益会混进 NLI 的账上。
- **假设:** dpr-w100 是 100 词切分,检索倾向把同条目相邻 passage 一起召回,故碰撞非零。
- **(a) 已测(2026-07-31,dev,纯 CPU):** `collision_rate = 0.931`(1862/2000)、
  `mean_documents_per_window = 20.0` → `mean_parents_per_window = 16.4735`、`n_unresolved = 0`
  (101479 文档全部解析出 parent,91492 个不同条目)。**假设成立,且远超预期:93% 的题受影响,
  平均每窗口 20 段只对应 16.5 个真实来源。** `independent_support` 的虚增是普遍现象,不是边角情况。
- **预期方向(预注册,2026-07-31 修订 —— 在 (b) 运行之前):** ~~原写"票差缩小 ⇒ 门 fire 更少 ⇒ harm↑ recall↑"~~
  **该推导有误,只考虑了条件 3。** parent 单位让 support 单调变小,但它同时喂给两个方向相反的门条件:
  - **条件 3**(`margin = winner − own ≥ margin`):两边都缩,差值**非单调**。竞争簇通常更大、缩得更多,
    故票差多半缩小 ⇒ 门 fire **更少**。
  - **条件 4**(`own ≤ support_cap`):own 缩小 ⇒ **更容易满足** ⇒ 更多候选变得**可踢**。
    一个靠**同条目两段**撑到 support=2 的 gold 簇原本结构性不可踢,parent 单位下掉到 1 就可踢了。

  **净方向不可先验推出。两个指标都报,不作方向性主张。**
- **判读口径(事先固定):** 若 recall 经条件 4 那条通道下降,**那不是回归,是撤掉了假保护** ——
  被撤掉的保护本来就是同一个来源被数了两次。这类 recall 下降必须**作为"修正"报告**,
  并与"门真的误踢"造成的 recall 下降**分开计**。
- **命令(前两条纯 CPU,登录节点即可):**
  ```
  python -m evidence_rag.cli.build_source_parent     --documents runs/niah-injected/documents.jsonl     --output runs/niah-injected/source_parent.jsonl

  python -m evidence_rag.cli.parent_collision     --candidates runs/e2-gate-on/candidate_sets.jsonl     --parent-index runs/niah-injected/source_parent.jsonl --top-n 20

  sbatch scripts/run_selector_gate.slurm     configs/experiments/niah_e2_gate_on_lenient_parent.toml     configs/experiments/niah_e2_gate_off.toml     runs/niah-injected/provenance.jsonl     runs/niah-injected/source_parent.jsonl
  ```
- **commit:** 5740d1b(sidecar)/ d6607b3(`support_unit`)/ b4ca71c(碰撞率 CLI)
- **AFTER:** _待填 —— collision_rate、mean_parents_per_window、harm、conditional_document_recall_

### R012 — Gate 0B 零训练 sweep(真实分叉点)

- **目的:** 判定现成 checkpoint 是否够用。**过则接门(Plan 2),不过则启动 M0 §3.8 的训练路径。**
- **假设:** `tals/albert-xlarge-vitaminc-mnli` 域对口(VitaminC 的对比结构 = injector 孪生的同构),
  有机会直接过;通用 NLI 上界臂用于区分"域不匹配"与"能力不足"。
- **预期指标与方向(预注册):** 三臂中至少一臂同时满足
  `twin_refutes_accuracy ≥ .70` **且** `gold_supports_recall ≥ .85`(0B-2),
  以及 0B-1 的五项阈值。**UNKNOWN 一律计失败** —— 全弃权的模型必须两项都挂。
- **为什么三臂同场:** 只跑一臂时,未过 Gate 只能得出"albert 不够",**无法排除"任何零训练模型都不够"**。
- **命令:** 见 `scripts/run_gate0b.slurm` 头部(含登录节点的模型预下载与两个 pair 文件的生成)。
  ```
  mkdir -p logs results/gate0b && sbatch scripts/run_gate0b.slurm
  ```
- **前置:** `runs/niah-train/` 的 artifact 必须已物化(0B-2 探针必须建在 **train** split 上)。
- **commit:** be9dd2c(Gate 0B 指标 + runner)/ ddb5342(探针)/ b5dbb5d(VitaminC adapter)
- **AFTER:** _待填 —— 每臂的 0B-1 五项与 0B-2 两项,以及分叉判定_

---

## 本地(非 HPC)验证记录

- 2026-07-20:selector 门实现全套单测 LOCAL 通过(tests/selector 32 + registration 9),
  mypy strict 干净(49 files);全套 pytest 与基线逐项对比:失败集恒为 29 条
  已知 Windows TOML fixture 问题(与 selector 无关,已开修复任务),零回归。
- 2026-07-21:A2 互补覆盖层实现 LOCAL 通过(coverage 引擎 11 + 覆盖选择器 2 + 注册 1 = +14);
  gated.py 抽出 `_gate` 复用、门 parity 保持;全套 **332 passed / 1 xfailed**(基线 318/1 + 14,零回归);
  mypy strict 干净(51 files),ruff 干净。Windows fixture 问题已被队友修掉,不再计入。
