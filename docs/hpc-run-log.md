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
- Git commit:2a63edd(feat(ingestion): OCR embedded PDF figures)。

**AFTER(2026-08-04,job 18258588,gpu:rtx_3090:1,bp1-gpu030;首次尝试 job 18258455 因
`sentence-transformers/all-MiniLM-L6-v2`(Docling HybridChunker 默认 tokenizer)未在
登录节点预取,离线模式下 `LocalEntryNotFoundError` 失败;补 `hf download` 后重跑通过):**
- raw:`runs/ocr-smoke/out/documents.jsonl`(未 push,本地 bp1 上)。`document_count`=2
  (1 image + 1 pdf)。image 源文档 caption 正确复述 sentinel("...REVENUE 2024 42 PERCENT
  GROWTH...");`Text in image:` 段落存在,OCR 命中 sentinel 数字 `42`。
- **OCR SMOKE: PASS。** 判据(§BEFORE)达成——caption 与图内 OCR 均生效,新增的
  PDF 内嵌图表处理链路(Docling converter → extract_pictures → Vision caption →
  OCR 追加)在真实模型下端到端跑通,非 test fake。
- 已知environment 坑,供下次复用此脚本时参考:(1) 项目要求 Python 3.11(<3.12,>=3.11),
  登录节点默认加载的 3.12.3 装不上 `evidence-rag`,需手动 `module load languages/python/3.11.15`
  重建 venv;(2) Docling HybridChunker 的默认 tokenizer(`all-MiniLM-L6-v2`)未被脚本注释
  里列出的预取步骤覆盖,离线模式下会因缺模型报错,需额外 `hf download sentence-transformers/all-MiniLM-L6-v2`。
  两坑均与本次功能验证结论无关,已通过重试规避,不影响 PASS 结论。

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
- **AFTER(2026-07-31,job 18225682,COMPLETED,elapsed 02:55:40):**
  - **G-FC 基线 = `lenient.fixed_false_conflict` 0.4355**(429/985,Wilson CI [0.4049, 0.4667])。
    条件性口径对照 0.5132(429/836)——**分子相同,差异全在分母**。
  - `exact.fixed_false_conflict` 0.5046(497/985);条件性 0.5945(497/836)。同样是分子相同。
  - 其余指标(exact):missed .38576、false .59450、recovery .50870;
    (lenient):missed .42189、false .51316、recovery .63133。
  - **rerun-stability(白捡的证据,M0 §5.4 要求):** exact 三项与历史 E1 复现到**小数点后 4 位**
    (.386/.594/.509)。同池、greedy、同硬件 → 无随机源的说法被实测支持。
  - δ 回填见 M0 §3.5:暂定 .05,非配对 MDE 近似 .063(power .80),**待 R003 的配对 MDE 确认后才算冻结**。
  - 产物:`results/r001-e1-baseline.json`、`r001-gfc-baseline.json`、`r001-gfc-per-query.json`
    (dump 5MB 不入库,由记录命令确定性重建)。

### R001b — parent 碰撞率 + `support_unit=parent` 基线臂

- **目的:** (a) 量出 top-20 里同 Wikipedia 条目的碰撞率;(b) 跑出与 Graph 2.0 同 `support_unit` 的
  Graph 1.0-lenient 对照臂 —— 主对照两侧必须同单位,否则 SAME_SOURCE 修复的收益会混进 NLI 的账上。
- **假设:** dpr-w100 是 100 词切分,检索倾向把同条目相邻 passage 一起召回,故碰撞非零。
- **(a) 已测(2026-07-31,dev,纯 CPU):** `collision_rate = 0.931`(1862/2000)、
  `mean_documents_per_window = 20.0` → `mean_parents_per_window = 16.4735`、`n_unresolved = 0`
  (101479 文档全部解析出 parent,91492 个不同条目)。**假设成立,且远超预期:93% 的题受影响,
  平均每窗口 20 段只对应 16.5 个真实来源。** `independent_support` 的虚增是普遍现象,不是边角情况。
- **(a′) needle-parent 定向诊断(同次运行,commit a09f787):**
  `needle_parent_inflation_rate = 0.328`(415 / 1264)、`cf_shares_needle_parent = 1261 / 1264`、
  `n_injected_scored = 1264`。
  - **1264 / 1479 = 0.854** 的 needle 在窗口内,与已知 `pool_hit ≈ 0.842` 一致 ——
    独立口径的一致性核对通过,说明工具读的是同一批题。
  - **0.328 是上界,不是实测虚增。** 该指标只说"同条目另一段**在窗口里**",不说它**抽出了等价答案
    并落进 gold 簇**。真正落进 gold 簇需要抽取结果,那要 GPU,属 (b)。
  - **3 例 `cf` 与 needle 不同 parent**(1264−1261)。成因:gold alias 恰好只出现在**标题段**,
    mutation 因而改到了 title,孪生的 parent 随之变了。占比 0.24%,且**机制上惰性**
    (cf 簇的 support 两种单位下都是 1),不修,但如实记录而非隐去。
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
- **(a′) 之后收窄的预期(仍不作方向性主张,只把机制写细):**
  - **harm 侧:** 门要踢掉毒需 `support(gold) − support(cf) ≥ 2` 且 `support(cf) ≤ 1`;
    因 cf 簇恒为 1,等价于 **`support(gold) ≥ 3`**。原本靠 3 段以上撑到 ≥3 的题,若其中有同 parent 的,
    parent 单位下会掉到 <3 ⇒ **门停火 ⇒ harm 上升**。(a′) 无法预判有多少题落在这里,因为它没测 gold 簇。
  - **recall 侧:** gold 簇若靠**同条目两段**撑到 support=2,document 单位下 `2 > cap=1` 使 needle
    **结构性不可踢**;parent 单位下掉到 1 ⇒ 变可踢 ⇒ **recall 下降**。0.328 说明这条通道的
    **机会**在 1/3 的题上存在。
  - **必须一起读的推论:** 若 (b) 观测到 harm 明显回弹向 gate-off 的 0.680,则 **S1 的 −11.2pp
    有一部分是"同一篇条目被数多次"换来的记账收益,而非机制收益** —— 这必须写进报告,
    不得只报"parent 单位下门更保守"了事。
- **命令(前两条纯 CPU,登录节点即可):**
  ```
  python -m evidence_rag.cli.build_source_parent     --documents runs/niah-injected/documents.jsonl     --output runs/niah-injected/source_parent.jsonl

  python -m evidence_rag.cli.parent_collision     --candidates runs/e2-gate-on/candidate_sets.jsonl     --parent-index runs/niah-injected/source_parent.jsonl --top-n 20

  sbatch scripts/run_selector_gate.slurm     configs/experiments/niah_e2_gate_on_lenient_parent.toml     configs/experiments/niah_e2_gate_off.toml     runs/niah-injected/provenance.jsonl     runs/niah-injected/source_parent.jsonl
  ```
- **commit:** 5740d1b(sidecar)/ d6607b3(`support_unit`)/ b4ca71c(碰撞率 CLI)
- **AFTER:** _待填 —— collision_rate、mean_parents_per_window、harm、conditional_document_recall_

### R012 — Gate 0B 零训练 sweep(真实分叉点) [DONE 2026-08-02]

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
- **AFTER(实测):**

**Job:** `18235972` COMPLETED / 00:37:34 / exit 0:0。提交行(`sacct -j 18235972 -o SubmitLine%200`):
`sbatch scripts/run_gate0b.slurm data/gate0b/task_pairs.jsonl data/gate0b/vitaminc_test.jsonl results/gate0b/sweep-full.json`。
同批另有三个作业不产出结果,一并记录以免日后被误读为"结果丢失":`18231280` FAILED 02:15
(tokenizer/backend/GPU,对应 `f09440d`、`4af4986`、`af293d6` 三个 fix)、`18235953` FAILED 00:31
(踩 slurm 的 `none` 哨兵未接线 bug,见下)、`18235954` 提交后即撤。
**Raw:** `results/gate0b/sweep-full.json`(`git add -f` 已提交)。

**两臂七项(n_external=55197,n_task=5888):**

| 指标(阈值) | albert-xlarge-vitaminc | DeBERTa-v3-large-mnli |
|---|---:|---:|
| 0B-1 refutes_precision (≥.85) | .903 | .913 |
| 0B-1 macro_f1 (≥.80) | **.922** | **.758** ✗ |
| 0B-1 non_unknown_coverage (≥.80) | .872 | **.675** ✗ |
| 0B-1 support_coverage (≥.70) | .951 | .798 |
| 0B-1 refutes_coverage (≥.70) | .894 | **.543** ✗ |
| 0B-2 twin_refutes_accuracy (≥.70) | **.674** ✗ | **.638** ✗ |
| 0B-2 gold_supports_recall (≥.85) | **.192** ✗ | **.794** ✗ |
| (参考)unknown_rate | .482 | .176 |

**分叉判定:FAIL。** 无一臂七项全过。按预注册,接门(Plan 2)不成立,**M0 §3.8 训练路径进入待启动状态**——
但**启动前有两道未清的障碍**:(1) hypothesis 形式可能是测量伪影(R012b,见下);(2) 预注册三臂缺了
MiniCheck-FT5(见下)。在被证明有测量缺陷的探针上、且未跑完预注册臂就开训,会把缺陷训进模型。

**预注册与实跑的不一致(记录,不回改 BEFORE):** BEFORE 与 `M0_PROTOCOL_FREEZE.md` §3.1 写的是**三臂同场**,
实跑只有两臂。缺的是**对照臂 MiniCheck-FT5(770M,LLM-AggreFact <1B SOTA)**。两个原因叠加:
`scripts/run_gate0b.slurm` 的第 4 位置参数 `MODELS` 是死变量(model id 硬编码在 python 调用里);
更根本的是 §3.1 自己注明该臂"**二分类,需否定 claim 双向探测,单独一步**",而 `gate0b.py::load_score_fn`
只有三类 argmax 路径、`LABEL_ORDER` 对未登记 id 直接抛错 —— **即使 `MODELS` 接了线,MiniCheck 也上不了场**,
它需要一条尚未实现的双向打分通路。

**这个缺口对结论的影响必须写明:** 现有两臂确实构成 BEFORE 所要的"域对口 vs 通用 NLI 上界"对照,
故"albert 不够"以外的推理成立。但 MiniCheck-FT5 恰恰是三臂中**唯一为 document-grounded verification
专门训练**的一臂,也是 §3.1 判定最可能过关的一臂。因此本轮**不足以支撑"任何零训练模型都不够"**这一族级断言,
只能支撑"两类通用/域内 NLI checkpoint 都不够"。族级断言须待 MiniCheck 臂补齐后才能下。

**第二处记录缺陷:探针的构造命令从未被正确记录(2026-08-03 发现)。** 本条目 BEFORE 的"前置"写
`runs/niah-train/`,`scripts/run_gate0b.slurm` 的 header 与 Plan 1 第 3036 行同样如此。**该路径跑不出任何一对**:
探针需要 `counterfactual_document_id` 指向的文档与 mutation log,两者都只在**注入之后**存在。
实际可用的是 `runs/niah-train-injected/`——2026-08-03 用它导出 rung 2 得
`{n_pairs: 5888, n_records: 1472, n_skipped_records: 0}`,与本轮 sweep 的 `n_task_pairs` 5888 逐字吻合,
证明 18235972 消费的探针也来自同一来源。**即当初建探针用的是一条与文档不符、且未入台账的命令**——
R011b 是 M1 里唯一没有台账条目的一步,这个缺陷因此无人发现。slurm header 已改正(BEFORE 不回改)。
**待办:** R011b 补台账;确认 `runs/niah-train/` 究竟是否存在,若存在须说明它与 `-injected` 的关系,
因为"探针必须建在 train split 上"这条冻结要求目前无法仅凭文档核验。

**两类失败性质不同,必须分开读:**

1. ~~**`twin_refutes_accuracy` 是跨臂一致的失败**(.674 / .638,同量级同方向)。这正是 BEFORE 所要的信号:
   不是"albert 这个 checkpoint 不够",而是零训练 NLI 这一族在孪生判别上都不够。~~

   **[2026-08-03 撤回 —— 本条已被 R012b rung 2 证伪。]** 换用 `question_answer` 形式后,albert 的
   `twin_refutes_accuracy` 升至 **.760,越过 .70 阈值**。跨臂一致并不蕴含"能力上限":两臂在 rung 1 上
   一起低,是因为它们**共用同一个有缺陷的 hypothesis 形式**,而不是因为任务超出零训练模型的能力。
   正确的表述是:**孪生判别在冻结模板下对两臂都不可达,但对 albert 在去掉元指称后可达**。
   与 `task_probe.py` docstring 记的"三轮抽取层工作没能修好"仍同指一处——只是现在有一个 59M 的
   现成 checkpoint 做到了 .76,这本身比 Gate 的成败更值得写进报告。
2. **`gold_supports_recall` 的 .192 vs .794 不自洽,不可作能力读数。** albert 在 VitaminC official test 上
   macro_f1 .922、非弃权覆盖 .872,换到探针最容易的一类(needle 文档确实支持 gold claim)掉到 .192,
   且被一个 0B-1 挂三项的臂以 4.1 倍碾过。能力不足是两臂同向退,不是这个形状。

**排除的三个机制(均有实测,非推测):**

- **截断**:两臂 `model_max_length` 均 512,premise token 中位数 150/140、p95 188/170、最长 275/272,
  `frac_over_cap` 均为 **0.000**。无任何 premise 被截断。
- **标签序错配**:albert 正是用 `gate0b.py::LABEL_ORDER` 现有映射拿到 0B-1 macro_f1 .922。
  排列错了不可能有 .92。
- **`gold_value` 被 `canonicalize_answer` 改坏**(曾疑为主因):claim 中的 answer 串
  **97.6% 逐字出现在 premise 中**(`cf_replacement` 为 100%)。hypothesis→premise 内容词覆盖
  探针 .524 vs VitaminC .597,仅差 7pp。别名/释义错配不足以解释 4 倍差距。

**大小写:对 albert 为空,对 deberta 是实打实的减分(实测 `do_lower_case`)。**
`tals/albert-xlarge-vitaminc-mnli` → `True`,`MoritzLaurer/DeBERTa-v3-large-...` → `False`。
`canonicalize_answer` 会把 answer 串小写,而 premise 保留原始大小写,于是:

- albert 是 **uncased**,premise 与 hypothesis 一起被小写 ⇒ 该伪影**对塌陷的那一臂完全不可见**。
  故"改用 `gold_alias_used` 原始串"这一变量**无需作为消融臂**,它对 .192 不可能有解释力。
- deberta 是 **cased**,却在**每一对**上收到 premise/hypothesis 大小写不匹配的输入。

推论必须写清:**.192 vs .794 的 4.1 倍是差距的下界而非上界** —— 免疫伪影的一臂拿 .192,
被伪影拖累的一臂拿 .794。这加强而非削弱"塌陷来自句式不来自证据"的读法。同时 deberta 的 .794 是
**被压低的地板**,距 `gold_supports_recall` 阈值仅 5.6pp,大小写修正后有可能过该项(twin .638 仍不过,
Gate 判定不变)。

**前向风险(记录,当前不构成偏差):** `build_hypothesis` 目前全仓库只有 `task_probe.py` 一个非测试调用者,
`relations/` 下尚无 `clustering.py`/`graph.py`,故这不是"探针偏离生产路径"——生产路径还没建。
但 `selector/clusters.py:100` 把簇代表答案设为 `canonicalize_answer(...)`;若将来 claim 由簇代表构造,
小写化会传进 hypothesis,届时每个 cased 模型都吃同一份减分。接关系层时须一并决定 claim 文本
取原始抽取串还是簇规范键。

**收窄(2026-08-03,实测):** 本段原把"QA2D 的 T5/BART"也列为受害者,**过度外推了**。
实测 `MarkS/bart-base-qa2d` 只把句首字母大写、专有名词保持输入的小写(`Howard jones` /
`donny hathaway`),即它**不会重新引入 premise/hypothesis 的大小写错配**,rung 3 不吃这份减分。
前向风险对**未来由簇代表构造 claim** 的那条路径仍然成立,与 rung 3 无关。

**指向 hypothesis 形式的证据(由本文件 raw 数据直接算出,非假设):** 四类各 1472 条。albert 侧,
UNKNOWN 总数 `.48234×5888 = 2840`;twin 判对 `.67357×2944 = 1983` ⇒ twin 上 UNKNOWN ≤ 961;
needle_gold 判对 `.19158×1472 = 282` ⇒ 其上 UNKNOWN ≤ 1190。故

> **cf_replacement 上的 UNKNOWN ≥ 2840 − 961 − 1190 = 689 = 46.8%**

`cf_replacement` 是干净对照组:answer 串 100% 逐字在 premise 中,无别名问题、无截断、标签无误。
albert 在它上面至少弃权 46.8%,说明 SUPPORTS 侧的塌陷**不是 needle_gold 特有的**,而是覆盖整个
SUPPORTS 类。同一算式在 deberta 侧给出负下界(即空),故该下界只对 albert 成立——这一点要如实写。

设计文档 §3.2 原文预见了这个洞:"一个在 VitaminC official test 上过关的模型,完全可能在
**NQ 段落 + 模板 hypothesis** 上无用"。0B-2 这一层就是为抓它而建,它抓到了。剩下的问题是那句话里的
"模板 hypothesis"是可改的设计选择还是任务固有属性 —— 这是 R012b。

**写给 `docs/results-summary.md` 的发现草稿:** 零训练关系判别的 Gate 0B 未过。域对口 checkpoint
(albert-xlarge-vitaminc)在 VitaminC official test 上 macro-F1 .922、五项阈值全过,但在任务形状的
探针上 gold-supports recall 仅 .192,而在同一探针上被一个域外通用 NLI 模型以 4.1 倍超过——且因前者
uncased、后者 cased 而后者独自承担了小写化减分,**该倍数是下界**。截断、
标签序、答案规范化三项机制均经实测排除;由弃权率可导出该模型在**词面完全匹配的干净对照对**上
仍弃权 ≥46.8%,说明塌陷来自 hypothesis 的**句式**而非证据。**孪生判别的失败(.674/.638)同样是形式造成的**:
换掉 hypothesis 句式后 albert 升至 .760 并越过阈值(R012b rung 2),故不可读作能力上限——
本草稿初版曾把它写成"跨两臂一致的真信号",已撤回。结论:0B-1 式的外部效度**不能**外推到任务效度,
两层验收缺一不可;且**一个探针的 hypothesis 句式本身就能造出跨臂一致的假失败**,跨臂一致不足以证明能力上限。
**须同时声明的范围限制:** 预注册三臂中的 MiniCheck-FT5(唯一为 document-grounded verification
专训的一臂)因缺少双向打分通路而未上场,故本轮不支撑"任何零训练模型都不够"的族级断言。

### R012b — hypothesis 形式消融(R012 的归因实验)[PRE-REGISTERED 2026-08-02]

> **编号说明:** 本条目原拟编 R013,与 `EXPERIMENT_TRACKER.md` 已占用的 R013(fine-tuned Relation
> Builder,seed 13,CONDITIONAL 于 R012 未过)冲突 —— 撞的恰是本实验要决定是否启动的那件事。
> 依 R011b 的既有后缀惯例改为 `R012b`。

- **状态:** BEFORE 已锁,代码在做。本条目在**看到任何新数字之前**写定。
- **目的:** 判定 R012 中 SUPPORTS 侧塌陷是 hypothesis 形式造成的测量伪影,还是零训练 NLI 的能力上限。
  这直接决定 M0 §3.8 训练路径是否启动——在一个被证明有测量缺陷的探针上训练,会把缺陷训进模型。
- **预注册依据:** 设计文档 §2.4 已把 QA2D 式转换器(Chen, Choi & Durrett, Findings of EMNLP 2021)
  列为 Gate 0B 上的**预注册消融**。本条目执行它,并补一级确定性中间形式使归因可分解。
- **三级阶梯(每级只移除一个变量):**

  | form | hypothesis | 相对上一级移除了 |
  |---|---|---|
  | `template`(冻结主口径) | `The answer to the question "{q}" is {a}.` | — |
  | `question_answer` | `{q}? {a}.` | 元指称框架 |
  | `qa2d` | 融合后的陈述句 | question 的表层回声 |

- **预期指标与方向(预注册):** 主判据是 albert 的 `gold_supports_recall`。
  - `question_answer` 即显著回升 ⇒ 元指称框架是主因;
  - 须到 `qa2d` 才回升 ⇒ question 表层回声是主因;
  - **三级都不动 ⇒ 形式不是主因,R012 的 FAIL 按能力不足读,立即启动 §3.8。**
- **事先固定的判读纪律(这条比结果重要):**
  1. **R012 的 FAIL 是预注册主结果,不因本轮改写。** 报告中两轮并列,标明先后。
  2. albert 的 `twin_refutes_accuracy` 距阈值仅 2.6pp,本轮**有可能把它顶过 .70**。若发生,
     **不得宣布 Gate 0B 通过**;须写作"在修正后的 hypothesis 形式下重测通过",并标注这是第二次测量。
  3. `qa2d` 引入 seq2seq 非确定性,故 claim 一律在**登录节点 greedy 预生成到缓存**,
     缓存文件随结果一同提交;compute 节点只读缓存,不加载生成模型。
  4. **缓存缺键硬失败,禁止回落 template。** 回落会把 template 行混进 qa2d 臂,悄悄污染对照。
  5. **不设大小写臂。** 实测 albert `do_lower_case=True`,小写化对主判据那一臂可证无影响(见 R012 AFTER);
     加这一臂只会稀释阶梯。**但 rung 3 的输出大小写不受控**:输入是小写的 question 与 answer,
     而 seq2seq 是否恢复大小写取决于 checkpoint,无法先验断定。故**缓存生成后必须先量再读**——
     统计 `qa2d_cache.jsonl` 中输出串相对输入的大小写变化率,写进本条目的 AFTER。
     若 rung 3 输出确为 cased 而 rung 1/2 为小写,则 rung 3 相对前两级**多变了大小写这一个变量**,
     其增量不得整份记到 QA2D 头上;此时需补一个"QA2D 输出强制小写"的对照才能拆开两者。
- **新增诊断:** `gate0b --dump` 逐 pair 预测。R012 只能导出 albert 在 `cf_replacement` 上
  ≥46.8% 的**下界**(deberta 侧该下界为空),dump 之后四类各自的混淆矩阵是实测值,
  可直接验证"塌陷覆盖整个 SUPPORTS 类"这一断言。
- **同批修复的两个 slurm 缺陷**(已由 R012 暴露):`EXTERNAL_ARG` 算了不用,导致 task-only 快通道
  (5888 对,本层承载 go/no-go 阈值)一提交就崩;`MODELS` 为死变量,导致预注册的第三臂无法上场。
- **命令(rung 1/2 可跑;rung 3 待 QA2D checkpoint 选定):**
  ```
  # 登录节点,纯 CPU。注意是 niah-train-INJECTED,见 R012 AFTER 的路径更正
  PYTHONPATH=src python -m evidence_rag.cli.export_task_probe \
    --manifest runs/niah-train-injected/manifest.json \
    --provenance runs/niah-train-injected/provenance.jsonl \
    --output data/gate0b/task_pairs_qa.jsonl --hypothesis-form question_answer

  ARMS="tals/albert-xlarge-vitaminc-mnli MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
  sbatch scripts/run_gate0b.slurm data/gate0b/task_pairs.jsonl    none \
    results/gate0b/sweep-template.json "$ARMS" results/gate0b/dump-template.jsonl
  sbatch scripts/run_gate0b.slurm data/gate0b/task_pairs_qa.jsonl none \
    results/gate0b/sweep-qa.json       "$ARMS" results/gate0b/dump-qa.jsonl
  ```
- **rung 2 pair 文件已物化(2026-08-03,登录节点):**
  `{"hypothesis_form": "question_answer", "n_pairs": 5888, "n_records": 1472, "n_skipped_records": 0}`
  —— 与 rung 1 的分母逐字相同,两级之间**只差 hypothesis 形式**这一个变量。
- **rung 3 的 QA2D checkpoint 已选定(2026-08-03 核实):`MarkS/bart-base-qa2d`。**
  上面"命令"标题里的"rung 3 待 QA2D checkpoint 选定"按**不回改 BEFORE** 的惯例保留原文,以本条为准。

  | 候选 | 规模 | licence | 已发表评测 | 本环境可加载 |
  |---|---|---|---|---|
  | **`MarkS/bart-base-qa2d`(选定)** | BART-base(~140M) | **AFL-3.0** | model card 报 QA2D testset **BLEU 78.878**,对照 2019 年 2-Encoder Pointer-Gen 基线 **74.05** | 是 |
  | `domenicrosati/question_converter-3b`(弃) | T5-3B | **未声明** | **无** | **否** |

  **送进模型的输入格式逐字为 `question: {question} answer: {answer}`**(取自 model card 的用法示例)。
  弃用臂的格式是 `{question} </s> {answer}`,训练数据为 Demszky et al. 2018 的 QA2D 三元组(71.1k),
  两种格式不通用 —— 换 checkpoint 必须同时换格式。

  **弃用 3B 的三个理由各自独立成立,三条都记:**(1) model card **未声明 licence**,而本工作要进受评报告;
  (2) **没有任何已发表评测数字**;(3) **在本项目钉住的 `torch < 2.6` 下根本加载不了**,抛
  `ValueError: Due to a serious vulnerability issue in torch.load ... we now require users to upgrade
  torch to at least v2.6`,而升 torch 在本仓库被禁(会打断别处的 safetensors 加载)。第三条暴露之前
  已下载 **22.8 GB**(bin + safetensors),成本记在此处以免日后重复踩。

  **选定依据是 30 对真实数据的过目检查,不是测量。** 取本项目自己的探针
  `data/gate0b/task_pairs.jsonl` 中 `kind == needle_gold` 的 30 对(seed 0),用 rung 3 缓存生成将要用的
  同一套 greedy 解码(`do_sample=False, num_beams=1`)跑过一遍 —— **看到的就是将来会被缓存的那些串**。
  代表性输出:

  ```
  when did they stop making the half dollar / 2002
    -> "They stopped making the half dollar in 2002 ."
  who sings no one ever is to blame / howard jones
    -> "Howard jones sings no one ever is to blame ."
  when does star trek discovery season 2 air / 2019
    -> "Star trek discovery season 2 airs in 2019 ."
  which layer of the atmosphere is the ozone layer located / stratosphere
    -> "The ozone layer is located in the stratosphere ."
  who is the original singer of this christmas / american soul musician donny hathaway
    -> "The original singer of this christmas is american soul musician donny hathaway ."
  ```

  动词形态是真的在变(`stop -> stopped`、`air -> airs`、`is located in`),**故这是 QA2D 变换,不是模板拼接**。

  **判读标准在看到输出之前就定死了**,三条全部满足:(1) 输出是把 question 内容融进去的陈述句 —— 满足;
  (2) 只复读 answer 或回声输入 —— 未出现;(3) 流畅但丢掉 question 的关键实体 —— 未实质出现,
  30 例中最差的两例是 `young and the restless` → `young and restless`(掉一个冠词),以及
  `how many episodes are in season 1 the good doctor / 18` → `In season 1 the good doctor is 18 episodes .`
  (别扭,内容完整)。**n=30 只支持"变换确实在发生、格式正确"这一条**;
  **不支持任何"该模型准确"的定量断言,本条目没有测量生成质量。**

  **大小写:这兑现了判读纪律 5 的预测,不兑现它的测量要求。** BART 只把句首词大写,专有名词沿用输入里的
  小写(`Howard jones`、`donny hathaway`);rung 1 的冻结模板同样以大写 `The` 起头。故预期 rung 3 相对
  rung 1 **基本只差句式这一个变量**。**但这是 30 例过目得到的预期,不是那项测量** —— 判读纪律 5 要求的
  "缓存生成后统计输出串相对输入的大小写变化率、写进本条目 AFTER"仍未做,
  **rung 3 的增量在该比率报出之前不得读**。

  **两处残留表层伪影,各自的处置(在此声明以便审计):**
  1. **句号前多一个空格**(`"... in 2002 ."`),来自 BART 的 detokenise;rung 1/2 没有。
     **处置:在 `export_qa2d` 里规范化掉**——只折叠句末标点紧邻其前的空白,不动别处。
     否则 rung 3 会额外带一个与被测变量无关的表层差异。
  2. **ISO 日期漏进句子**(`"was held in 1967-08-08 ."`),来自 `canonicalize_answer` 对日期的转换。
     **处置:不动。** 它在三级阶梯上完全相同(三级共用同一份 answer 串),不构成混淆;
     规范化它反而会改掉三级共享的输入。
- **commit:** _待填_
- **AFTER(rung 1/2 实测,2026-08-03;rung 3 未跑):**

**Jobs:** `18246568`、`18246569`,均 COMPLETED,00:07:56 / 00:07:25,exit 0:0。
(两者与 template/qa 的对应关系未记录,可由 `grep -l sweep-template logs/gate0b-1824656*.out` 补。)
**Raw:** `results/gate0b/sweep-{template,qa}.json`、`results/gate0b/dump-{template,qa}.jsonl`。
**复现检查:** rung 1 的 task 层与全量 sweep(18235972)**逐位一致**(.1916/.6736/.4823 与 .7942/.6376/.1758),
确认该通路确定性无问题,且 `EXTERNAL_ARG` 修复后 task-only 快通道可用(8 分钟 vs 全量 37 分钟)。

**0B-2 两项:**

| 臂 | rung | twin_refutes_acc (≥.70) | gold_supports_recall (≥.85) | unknown_rate |
|---|---|---:|---:|---:|
| albert | 1 `template` | .6736 ✗ | .1916 ✗ | .4823 |
| albert | 2 `question_answer` | **.7602 ✓** | .3635 ✗ | .2858 |
| DeBERTa | 1 | .6376 ✗ | .7942 ✗ | .1758 |
| DeBERTa | 2 | .5547 ✗ | .6651 ✗ | .2884 |

**四类混淆矩阵(predicted 分布,S/R/U):**

| 臂 / rung | needle_gold | cf_replacement | cf_gold | needle_replacement |
|---|---|---|---|---|
| albert / 1 | .192/.168/.641 | .187/.173/.640 | .004/.629/.367 | .000/.718/.282 |
| albert / 2 | .363/.276/.361 | .357/.315/.329 | .024/.735/.240 | .001/.785/.213 |
| DeBERTa / 1 | .794/.088/.118 | .681/.196/.123 | .234/.515/.251 | .029/.760/.211 |
| DeBERTa / 2 | .665/.099/.236 | .532/.243/.226 | .154/.471/.375 | .044/.639/.317 |

**判读:**

1. **形式是真因,幅度很大,方向因臂而异。** albert 从 rung 1 到 rung 2:gold_supports **+17.2pp**、
   twin **+8.7pp** 并越过阈值、unknown **−19.7pp**。DeBERTa 反向:两项各退 12.9pp / 8.3pp。
2. **阶梯设计有缺陷,须如实记录。** BEFORE 称"每级只移除一样东西",**不成立**:rung 1 是合语法的
   元指称陈述句,rung 2 去了元指称**但也不再是陈述句**,rung 3 才是"−元指称 +合语法"。
   rung 2 一次动了两个变量,两臂反向正由此解释(albert 吃"去元指称"红利,DeBERTa 吃"失去句法"的亏)。
   **这使 rung 3 更重要,不是更不重要**——它是唯一未被占据的那一格。
3. **albert 在两类 SUPPORTS 上不作区分。** needle_gold 与 cf_replacement 的预测分布逐位对齐
   (rung 1 差 ≤.005,rung 2 差 ≤.006),而 DeBERTa 差 11.3pp。R012 AFTER 推的 ≥46.8% 下界
   实测为 **64.0%**——方向对,但严重低估。
4. **两臂失败模式互补,不是同一个。** albert 的 SUPPORTS 预测精度极高(孪生对上误判 SUPPORTS
   仅 .004/.000,rung 2 .024/.001),问题是纯粹的召回不足;DeBERTa 召回好但**栽在孪生上**
   (cf_gold 误判 SUPPORTS 达 .234 / .154)。真正的"孪生判别失败"只发生在 DeBERTa 身上。
5. **rung 2 的红利有代价:** albert 在真支持证据上判 REFUTES 由 .168 升至 **.276**,
   即部分召回是从弃权挪成了误判反证。
6. **cf_gold 系统性难于 needle_replacement,四种组合无一例外**(.629/.718、.735/.785、
   .515/.760、.471/.639)。两者都是"值被换掉"的反证对,差别在于 cf_gold 的文档是被篡改的那一份。
   该不对称尚未解释,单独列为待查。
7. **对门的后果:** `gold_supports_recall .363` 下,图会稀疏到凑不出 `support(gold) ≥ 3`,
   **该 checkpoint 即使 twin 过关也撑不起 `independent_support`**。
8. **premise 侧无缺陷(实测空结果,2026-08-03)。** 按 CSV 转义残留(`""`)与 premise 长度分桶,
   albert / needle_gold / rung 2 的 SUPPORTS 率为:带转义残留 .369(n=899)vs 干净 .354(n=573);
   长于中位数(634 字符).362(n=729)vs 短 .365(n=743)。**两个对比均远小于 1 个标准误**
   (差异 SE ≈ .026),且带残留的一桶反而略高。**premise 文本质量与长度都不解释 `.363`。**

   合并 1–8 与 R012 AFTER 的排除项:截断、标签序、答案规范化、大小写、premise 质量、premise 长度
   **全部实测排除**;hypothesis 形式**已证实有效**。**测量缺陷侧只剩 rung 3 一个未测变量** ——
   rung 3 之后,该量级即应作为能力读数。

   附带刻画:结合判读 3(needle_gold 与 cf_replacement 逐位对齐),albert 对 SUPPORTS 的判定
   **对所有已试输入特征均不敏感**。它并非随机(对真 REFUTES 仅 .024 判 SUPPORTS,说明确在读关系),
   但在真 SUPPORTS 类内部,无任何已试变量能预测它承诺哪一部分。

**二分类事后重分析(2026-08-03):**
twin 改判"预测 != SUPPORTED"后,albert .9980 / .9871,DeBERTa .8689 / .9008,**四种组合全过**;
`gold_supports_recall` 不变,**四种组合仍全部未过**。

**协议依据:** 修订案 **A1 已于 2026-08-03 批准**,协议升版 `g2-proto-1` → `g2-proto-2`
(见 `M0_PROTOCOL_FREEZE.md` §9)。但批准**不改变本读数的事后性质**:

- **两种口径并列,永久并列**(§9.6 第 2 条)。三类口径下的 FAIL 是**预注册主结果**,不因升版改写或撤回;
  二分类读数一律标注为 2026-08-03 事后重分析,任何场合不得单独作为"Gate 0B 结果"呈现。
- **A1 是在看到 FAIL 之后提出的**,时间线见 §9.0,利益冲突声明见 §9.9。批准不消除披露。
- **未产生任何通过者:** 二分类下 twin 四种组合全过,但 `gold_supports_recall` 四种组合**仍全部未过**
  (最高 .7942 < .85)。修订把 FAIL 的原因从两项收敛到一项(§9.5)。
- **阈值仍不得引入**(§9.5a):2026-08-03 的全量阈值扫描显示 DeBERTa/rung 1 在 θ≤.10 时两项同时过,
  该曲线**已污染,不得用于选定 θ**。
- **代码尚未按 §9.1 改动**,`relations/` 仍以三类运行,故本条目的三类数字保持可复现。

**QA2D 缓存已生成 + 判读纪律 5 的测量已执行(2026-08-03):**
`{"model": "MarkS/bart-base-qa2d", "n_pairs": 2944, "n_unique": 2944, "non_initial_uppercase_rate": 0.301}`。
rung 3 的 pair 文件同日物化(`n_pairs 5888 / n_records 1472 / n_skipped_records 0`,与 rung 1/2 分母逐字相同),
job **18257977** 已提交。

**判读:应急条款不触发,rung 3 与 rung 1/2 在大小写上可比。** 把 0.301 按输入是否已带大小写拆开:

| | n | 输出出现非句首大写 |
|---|---:|---:|
| 输入已带大小写 | 998 | **.867** |
| 输入全小写 | 1946 | **.011** |

**转换器不重新加大小写** —— 输入小写时输出 98.9% 保持小写。那 0.301 几乎全部来自输入自带,
而输入是三个 rung 共享的。故纪律 5 的"若 rung 3 输出确为 cased…需补强制小写对照"**不适用**。

**指标设计缺陷(记录,由本轮暴露):** `non_initial_uppercase_rate` 把"转换器引入的大小写"与
"输入自带的大小写"混进一个数,**它并不回答纪律 5 要问的问题**——0.301 曾被(本文件作者)读成
"转换器重新加了大小写",是错的,拆桶后才看清。该指标须改为按输入桶分别报告;在改之前,
**单看这个总数会误导**。

**由此暴露的一个更大的、写在冻结指标里的不对称(2026-08-03):**

| kind | 答案串含大写 |
|---|---:|
| needle_gold | **.000** |
| cf_gold | **.000** |
| cf_replacement | **.678** |
| needle_replacement | **.678** |

`gold_value` 经 `canonicalize_answer` 变小写,而 `replacement_value` 是 `answer_bank` 从语料选的
**原始串**(规范化只用于过滤比较,不做替换)。于是 **`gold_supports_recall` 的分母 100% 是小写答案**,
而 **`twin_refutes_accuracy` 的两半不同质**(cf_gold 全小写,needle_replacement 67.8% 带大写)。
DeBERTa 是 cased 模型,故两个门指标**建在系统性不同的输入分布上**。

**范围要写准:** 该不对称**三个 rung 完全一样**(答案串同源),故**不污染阶梯比较**,rung 3 的读数有效;
它污染的是**同一 rung 内的跨指标与跨臂比较**。也不夸大其解释力:cf_replacement 大小写匹配更好
(replacement 逐字 substituted 进 cf 文档)却反而更低(DeBERTa rung 1 .681 vs needle_gold .794),
说明大小写不是主导因素。但它是一个**此前从未被声明、且坐在门指标内部**的系统性差异 ⇒ 立 R012d。

**rung 3 实测(job 18257977,COMPLETED):三级阶梯完整。**
Raw:`results/gate0b/sweep-qa2d.json`、`dump-qa2d.jsonl`。

| rung | 臂 | twin(≥.70) | gold(≥.85) | unknown | twin(二分类) |
|---|---|---:|---:|---:|---:|
| 1 `template` | albert | .6736 | .1916 | .4823 | .9980 |
| 2 `question_answer` | albert | **.7602 ✓** | .3635 | .2858 | .9871 |
| 3 `qa2d` | albert | .5312 | **.5360** | .3680 | .9586 |
| 1 `template` | DeBERTa | .6376 | **.7942** | .1758 | .8689 |
| 2 `question_answer` | DeBERTa | .5547 | .6651 | .2884 | .9008 |
| 3 `qa2d` | DeBERTa | .5187 | .5870 | .3511 | .9222 |

**对预注册问题的回答:形式是主因,且远强于 rung 2 所显示。** albert 的
`gold_supports_recall` 沿阶梯**单调上升 .1916 → .3635 → .5360**,**2.80 倍**,唯一变量是
hypothesis 那句话。**`.1916` 就此确定不是能力读数。** 但 `.5360` 距 `.85` 仍差 **31.4pp**,
**形式也不解释全部。**

**BEFORE 的三种预设情形都没命中 —— 实际结果是两个指标反向:**

- albert 的 twin **在 rung 3 崩了**:.6736 → **.7602(过阈值)** → **.5312**。
- DeBERTa **两项单调下降**:gold .7942 → .6651 → .5870;twin .6376 → .5547 → .5187。

⇒ **不存在任何一种 hypothesis 形式使两个门指标同时变好。** 使 SUPPORTS 召回最高的形式,
恰好使 REFUTES 承诺最低。这比"形式是/不是主因"信息量更大,应作为本条目的主发现之一。

**机制(albert rung 2 → rung 3,分 kind 的预测迁移):**

| kind | ΔSUPPORTS | ΔREFUTES | ΔUNKNOWN |
|---|---:|---:|---:|
| needle_gold | **+17.3** | −12.5 | −4.8 |
| cf_replacement | **+17.4** | −15.0 | −2.5 |
| cf_gold | +3.9 | **−25.4** | **+21.6** |
| needle_replacement | +1.9 | **−20.3** | **+18.6** |

**REFUTES 预测在四类上全线下跌。** QA2D 产出的是良构陈述句,而元指称模板与 `q? a.` 片段都不是;
喂进一句像样的断言,模型更不愿判"矛盾"、更愿判"蕴含或中立" ⇒ **决策边界整体朝 SUPPORTS/中立移动**。
在 SUPPORTS 对上是红利,在孪生对上就是 REFUTES→UNKNOWN 的流失。

**要写准的一点:albert 并非被孪生骗了。** rung 3 的 cf_gold 假 SUPPORTS 仅 **.063**、
needle_replacement 仅 **.020**,它是**弃权**。故 `twin_bin` 仍有 .9586 而三类 twin 掉到 .5312。

**Gate 判定:六格无一通过,判定不变。** 最好的 twin 是 albert rung 2 的 .7602(其 gold 仅 .3635);
最好的 gold 是 DeBERTa rung 1 的 .7942(其 twin 仅 .6376)。**没有任何一格两项同时过。**
按已生效的 **g2-proto-2 二分类口径**读:twin 六格全过(.8689–.9980),**唯一约束仍是
`gold_supports_recall`,最好的一格仍是 DeBERTa rung 1 的 .7942,差 5.6pp**。
**即:对最接近通过的那一臂,形式消融在每一级都使其更糟。**

**三级一致的刻画:** albert 的 needle_gold 与 cf_replacement 在 rung 3 依然逐位对齐
(.536/.151/.313 vs .531/.165/.304)。**三级全程,albert 从未在 SUPPORTS 类内部作出区分。**

**尚未完成:** ~~rung 3(QA2D)未跑~~ —— **已于 2026-08-03 完成(job 18257977),见上。**
剩余:R012d(答案串大小写对照)未跑,是 §3.8 之前最后一个杠杆,且正对着现在唯一的约束
(DeBERTa 的 .7942 建在 100% 小写答案的分母上,而它是 cased 模型)。以下关于 checkpoint 的记录保留备查。

**旧文(2026-08-03 之前):** rung 3(QA2D)未跑。**checkpoint 不再是阻塞项** —— 2026-08-03 已选定
`MarkS/bart-base-qa2d`(依据、弃用臂、两处表层伪影的处置见上)。现存阻塞项换成三件:
QA2D 缓存尚未生成、rung 3 pair 文件尚未物化、这两条命令尚未入台账(上面的"命令"块只覆盖 rung 1/2)。
其 AFTER 须先报缓存的大小写变化率(见判读纪律 5)才能读 rung 3 的增量。
**更正(2026-08-03):** 本行原写"premise 侧诊断未跑"。**不准确** —— CSV 转义残留与 premise 长度
两个分桶已于同日跑完,均为实测空结果,见本条 AFTER 判读 8。真正未测的只有**"半句起始"**这一项,
且须先说明:dpr-w100 按 100 词切分,**绝大多数 passage 本就从半句开始**,该分桶很可能没有对照组,
届时应报告为"无足够变异,不可判读",而不是当作又一个空结果。

### §9.10a 等价性验证 — 四个 dump 全部复现 [2026-08-04]

M0 §9.10a 裁定 R012 / R012b 的二分类读数**从 dump 重算,不重跑**,并声称该重算与"模型原生二分类"
等价。该主张此前只是断言。`cli/recompute_binary --against` 以 `gold_supports_recall` 为不变量
(A1 §9.1 把它钉为不变)逐个核对,**四个 dump 全部通过,无一抛错**:

| 探针变体 / 臂 | `gold_supports_recall` | `twin_not_supported` |
|---|---:|---:|
| template / DeBERTa | .794158 | .868886 |
| template / albert | .191576 | .997962 |
| question_answer / DeBERTa | .665082 | .900815 |
| question_answer / albert | .363451 | .987092 |
| qa2d / DeBERTa | .586957 | .922215 |
| qa2d / albert | .536005 | .958560 |
| surface / DeBERTa | .779212 | **.863451** |
| surface / albert | .198370 | **.998302** |

(后两行的 twin 二分类值为本次首次计算。)全部与各自 sweep 已发表的三类读数一致到浮点精度,
**故折叠方式(取 max、即 argmax 重标签)得到实证确认** —— 若按求和实现,`gold_supports_recall`
会因 `.5` 阈值而移动,八条中任何一条都会抛错。

**由此得到的整体形状:八格的 `failures` 全部且仅为 `["gold_supports_recall"]`。**
在 g2-proto-2 下 Gate 0B 的成败**塌缩为单一指标**,其最优值为 DeBERTa/template 的 **.7942**(差 5.6pp);
twin 项在八格中介于 .8635–.9983,全部通过。该表由 `cli/recompute_binary` 机器产出,
第三方可用同一条命令复算,不依赖本文件的手工誊录。

### R012d — 答案串 canonical vs surface 对照 [PRE-REGISTERED 2026-08-03,DONE 2026-08-04]

> **标题已更正(2026-08-04):** 原题为"大小写对照",不准确 —— 见 AFTER 缺陷记录 1。
> **BEFORE 正文一字未改**,更正只以本行与 AFTER 呈现。

> **编号:** R012c 已在 MiniCheck-FT5 臂的任务简报中被指定占用,故本条目编 R012d。

- **状态:** BEFORE 已锁。**本条目写定于 job 18257977(rung 3)结果被读取之前**,时间戳即为证据;
  该 job 的任何数字都未参与本条目的设计。
- **动机(实测,见 R012b AFTER):** `gold_value` 经 `canonicalize_answer` 小写化,而 `replacement_value`
  是原始串。⇒ `gold_supports_recall` 的分母 **100% 小写答案**,`twin_refutes_accuracy` 的两半不同质。
  DeBERTa 是 cased 模型,且它的 `.7942` 是四种组合里**距 `.85` 阈值最近的一个(差 5.6pp)**。
- **目的:** 判定 DeBERTa 的 `gold_supports_recall` 是否被"claim 用小写答案、premise 是正常大小写"
  这一系统性错配压低。这是唯一还可能把它推过阈值、且**不涉及改指标或调阈值**的机制。
- **操作(只动一个变量):** 重建探针,gold claim 改用 `record.gold_alias_used`
  ——**即 needle 文档中真实出现的原始串**——替代规范化的 `gold_value`。replacement claim 不动
  (它本来就是原始串,且已逐字 substituted 进 cf 文档)。**hypothesis 形式不动**,仍用冻结模板(rung 1)。
- **范围:** rung 1 为主臂;rung 2 可顺带跑(确定性、零成本)。**rung 3 不跑** ——
  QA2D 缓存以 (question, answer) 为键,改答案串即需重新生成缓存,那会同时动两个变量。
- **预期指标与方向(预注册):**
  1. **DeBERTa 的 `gold_supports_recall` 上升。** 若大小写错配是真实减分,移除它应当抬高该项。
  2. **albert 的 `gold_supports_recall` 不动(容忍 ±1pp)。** —— **这是内建的阴性对照。**
     albert 实测 `do_lower_case=True`,它把 premise 与 hypothesis 一起小写,**大小写对它可证不可见**。
     **若 albert 明显移动,说明本次操作改变了大小写以外的东西,实验无效,须先查因再读 DeBERTa。**
  3. `twin_refutes_accuracy` 两臂均可能变动(cf_gold 那一半的答案串被改),**不作方向性主张**。
- **事先固定的判读纪律:**
  1. **本轮无论结果如何,Gate 0B 的判定不变。** 即使 DeBERTa 越过 `.85`,那也是**第三个探针变体上的
     第三次测量**,须与 R012 的预注册主结果、R012b 的形式阶梯**三者并列**呈现,
     并标明各自的探针版本。**不得称"Gate 0B 通过"**(与 M0 §9.6 第 2 条、§9.5a 同一纪律)。
  2. **不得借此引入阈值**(M0 §9.5a 仍然生效)。
  3. 本条目**修正**了我先前的一个判断:2026-08-03 早些时候曾以"albert uncased ⇒ 大小写可证无效"
     为由主张不设大小写臂。**该推理只对 albert 成立**,而 DeBERTa 是 cased 且是最接近阈值的一臂 ——
     当时的结论**外推过头了**,本条目即为更正。
- **前置:** 无新代码依赖;`gold_alias_used` 已在 `MutationRecord` 上(`provenance.py:21`),
  `task_probe.py` 只需读另一个字段。须加测试钉住"用的是 alias 不是 canonical"。
- **命令(代码已就位,2026-08-04):**
  ```
  # 登录节点,纯 CPU。--gold-answer surface 是本轮唯一变量,hypothesis 形式仍用冻结模板
  PYTHONPATH=src python -m evidence_rag.cli.export_task_probe     --manifest runs/niah-train-injected/manifest.json     --provenance runs/niah-train-injected/provenance.jsonl     --output data/gate0b/task_pairs_surface.jsonl --gold-answer surface

  ARMS="tals/albert-xlarge-vitaminc-mnli MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
  sbatch scripts/run_gate0b.slurm data/gate0b/task_pairs_surface.jsonl none     results/gate0b/sweep-surface.json "$ARMS" results/gate0b/dump-surface.jsonl
  ```
  `--gold-answer` 默认 `canonical`,有测试钉住"省略即与预注册主臂逐字节相同";
  报告行回显 `gold_answer_source`,作为该 pair 文件属于哪个变体的协议记录。
- **commit:** _待填(代码随本轮提交)_
- **AFTER(实测,2026-08-04):**

**Job:** `18259086` COMPLETED。**Raw:** `results/gate0b/sweep-surface.json`、`dump-surface.jsonl`。
pair 文件 `{"gold_answer_source": "surface", "hypothesis_form": "template", "n_pairs": 5888,
"n_records": 1472, "n_skipped_records": 0}` —— 分母与 rung 1 逐字相同。

**按预注册顺序读:**

**① 内建精确对照(超出预注册,更强):** `cf_replacement` 的 1472 对在两个 pair 文件中逐字节相同
(其 premise 与 hypothesis 都不受 `--gold-answer` 影响),故预测**必须**一致。实测
**2944 条比对,预测不同 0 条** ⇒ 操作精确地只动了 gold claim,贪心解码的确定性一并验证。

**② albert 阴性对照:按预注册标准通过。** `gold_supports_recall` .1916 → .1984,
**Δ +0.68pp,在 ±1pp 容忍内**。

**③ DeBERTa:预注册方向被证伪。**

| 臂 | 指标 | canonical | surface | Δpp |
|---|---|---:|---:|---:|
| albert | gold_supports_recall | .1916 | .1984 | **+0.68** |
| albert | twin_refutes_accuracy | .6736 | .6736 | +0.00 |
| DeBERTa | gold_supports_recall | **.7942** | **.7792** | **−1.49** |
| DeBERTa | twin_refutes_accuracy | .6376 | .6291 | −0.85 |

预注册写的是"DeBERTa 的 `gold_supports_recall` 上升",**实测下降**。

**配对检验(事后,未预注册 —— 如实标注):** McNemar on `needle_gold`,n=1472。
albert 仅canonical对 2 / 仅surface对 12,χ²=5.79,**p≈.0162**;
DeBERTa 仅canonical对 30 / 仅surface对 8,χ²=11.61,**p≈.0007**。**两者均显著,方向相反。**

**量级必须与显著性一起写:** DeBERTa 38 对不一致中净负 22 条 = −1.49pp,**而它需要 +5.6pp**。
**显著,但方向反了、且量级差一个数量级。** 该杠杆为空,无含糊余地。

**三条如实记录的缺陷,均属本条目设计者(即本文件作者):**

1. **本条目命名不准。** 标题写"大小写对照",实际是 **canonical vs surface 对照**:
   `canonicalize_answer` 除小写外还去冠词、去首尾标点、日期转 ISO、数字展开。
   **DeBERTa 的结果不可单独归因于大小写。**
2. **阴性对照按预注册通过,但它其实在报警。** albert 对大小写不可感知,却显著改善(p≈.0162)
   ⇒ 操作确实触达了非大小写成分,albert 的 +0.68pp 即该成分的效应量。
   **但 McNemar 是事后跑的**,用事后检验推翻预注册的通过判定本身即事后操作,
   故**正式判定仍为"按预注册标准通过"**,该观察并列记录、不改判定。
3. **反向结果只作提示,不作确立。** surface 使无大小写感知的臂变好、使 cased 臂变差;
   两臂在此唯一的相关差异是大小写敏感性,故大小写*看起来*是 DeBERTa 那份代价的来源。
   但这是跨模型比较,预设了"非大小写成分对两模型效应相同",**站不住,不予确立**。

**结论:测量缺陷侧到此穷尽。** 九项机制全部实测:截断、标签序、答案规范化、大小写、
premise 质量、premise 长度、hypothesis 形式、argmax 记账、canonical-vs-surface。
**DeBERTa 的 `.7942` 自此可作能力读数**,§3.8 训练路径在测量这一侧的启动理由已经干净。

**本条目不建立的东西(须与结论同时声明):** MiniCheck-FT5 仍未上场。它是 M0 §3.1 预注册三臂之一、
本项目实测最优的验证器、且是 M0 §9.8 指定的 A1 出样检验之一。**故"任何零训练模型都不够"的
族级断言依然不成立**,§3.8 可启动但不得携带该断言。

### R011 — VitaminC adapter 与 revision-family 去污染 [DONE 2026-07-31]

- **命令:** `python -m evidence_rag.cli.export_vitaminc --out-test data/gate0b/vitaminc_test.jsonl
  --removed-log data/gate0b/vitaminc_decontamination.json`(登录节点,纯 CPU,约 5 秒)
- **AFTER(实测):**

| split | official | 去污染后 | 移除 |
|---|---:|---:|---:|
| train | 370653 | **369843** | 810 行 / **38 个 page** |
| validation | 63054 | **62984** | 70 行 / **2 个 page** |
| test | 55197 | **55197** | **0 —— official test 一行未动** ✅ |

- **移除的 dev page:** `John Frusciante`、`Linkin Park`(两者均出现在 official test 中)。
- **移除的 train page(38):** 含 `World War II`、`China`、`Aristotle`、`French Revolution`、`YouTube` 等。
  `Linkin Park` 同时出现在两份移除清单里 —— 与实现一致(它在 test 中,故 dev 和 train 都要清)。
- **判读:** VitaminC 的 official split **确实存在跨 split 的 revision-family 重叠**(38 个 page),
  虽然量小(train 0.22% / dev 0.11%)。这条去污染步骤不是形式主义,它真的删掉了东西;
  同时 official test 逐行未动,满足"test-preserving"的冻结要求。删除清单已存档
  (`data/gate0b/vitaminc_decontamination.json`,注意 `/data/` 被 gitignore,须 `git add -f`)。
- **未导出 train/dev 的去污染副本:** Gate 0B-1 只需 official test;去污染后的 train/dev 仅在
  §3.8 训练路径被启动时才需要,届时加 `--out-train` / `--out-dev` 重跑即可(确定性,可复现)。

---

## G3 — 基线对照:verified vs 生成时引用(方法头条主张)

**状态:** READY——三件套齐:代码(entity_check 修复 `c75e991` + runner
`scripts/g3_baseline_comparison.py`)、slurm(`scripts/run_g3_baseline.slurm`)、本条目。
**commit hash:`68b51ae`**。**先决修复已做并验证**:entity_check 假否决(审计实测 60%)—— 15 个
已标注 stratum-B 项中 8/9 假否决在修复后通过 entity、真 swap(item-10 Sobers≠Gooch)仍否;
ClaimSplitter source_text 逐字约束已放宽(`872ed61`)、over-split meta 句已在 SPLIT_PROMPT 抑制。

**BEFORE(预注册):**

- 目的:检验方法的**头条主张** ——「后验验证产生的引用,比模型生成时自报的引用更忠实」。三臂同 query、
  同 Granite、同已选证据,只变后处理:baseline(`GraniteGenerator` 生成时引用+兜底)/ verify-only
  (draft→verify→repair,关掉 completeness/recheck)/ verified-full(全链)。第三臂是**消融**:把
  「忠实(丢无支撑)」与「完整(找补缺口)」两项贡献分开,否则改进无法归因。
- 判官:**MiniCheck(非 TRUE、非 TRUE 衍生、非 ALCE 的 TRUE 指标)**—— verified 臂的引用是 TRUE 选的,
  用 TRUE 评分是循环。MiniCheck 比 TRUE 弱(G1 召回 0.620 vs 0.747),是对**两臂同样**施加的保守判官,
  绝对值是下限、对照公平。另有 ~40 项人工盲审(跨 baseline/verified),与自动数并列;若显著不一致以人工为准。
- **诚实预期 + 失败判据:** 预期 verified 臂引用精度 > baseline(baseline 兜底会强引 evidence[0],精度低)。
  **失败判据(预注册):若在「两臂都作答」子集上 verified 的引用精度不显著高于 baseline,则头条主张不成立**——
  无论其他数字如何。同时诚实预期 verified **coverage 更低**(会弃答);故三轴必须**同时**报,单报引用精度是误导。
  verified 若靠「删掉大部分答案」换引用精度,会在 answer correctness(ASQA STR-EM)上暴露。
- 预期指标 + 方向:①coverage(非空答案率;baseline 也有 `_is_unknown_answer` 弃答路径,一并报);
  ②MiniCheck 引用 precision/recall(答作答题);③answer correctness(ASQA STR-EM vs gold short answers)。
  **核心对照:两臂都作答子集上的引用精度**(去掉弃答混淆),配对 p 值 + CI。verify-only vs verified-full 的
  delta = completeness 环节单独贡献。统计**复用 `evidence_rag.evaluation.paired_metric_cli`**(按 query_id 配对、
  均值差、随机化 p、bootstrap CI;None 值自动只留两臂都作答的题),不另写显著性实现。
- 精确命令(登录节点先取 MiniCheck + 查 ≥300 产量,再 sbatch;Granite-3b/TRUE 已缓存):
  ```
  export HF_HOME=/user/work/$USER/hf_cache
  hf download lytang/MiniCheck-Flan-T5-Large
  mkdir -p logs results/g3 && sbatch scripts/run_g3_baseline.slurm \
    docs/generator/g3-baseline-comparison-hpc.md results/g3
  ```
- Seed:`--seed 13 --limit 400 --top-k 5`(目标 ≥300 题作头条)。
- 数据合规:只用 ALCE/ASQA;**HotpotQA / RGB / MuSiQue-Full 从不加载**(留最终 held-out,冻结后此实验复跑作终值)。

**AFTER(2026-08-01,job `18238790`,a100,400 题,判官 MiniCheck):**

- Raw:结果 `docs/generator/g3-baseline-comparison-hpc.md`;三臂报告 + 人工盲审 dump 在
  `results/g3/`(本地 `local/raw-results/{baseline,verify-only,verified-full}-report.json`、
  `g3-human-subsample.jsonl`、`g3-baseline-18238790.out`)。
- **三轴表(coverage / STR-EM correctness / 引用 precision·recall(作答题) / 作答数):**
  baseline **0.932 / 0.273 / 0.602·0.646 / 373**;verify-only 0.552 / 0.189 / **0.762·0.862** / 218;
  verified-full 0.331 / 0.128 / 0.575·0.736 / 125。baseline 自身弃答 6.8%(非恒 1.0)。
- **配对(两臂都作答子集,MiniCheck,随机化 p + bootstrap CI):**
  - **verify-only vs baseline 引用精度 +0.090(0.764 vs 0.674),p=0.010,CI[+0.021,+0.158],n=215**;
    recall +0.130,p=0.0002。
  - **verified-full vs baseline 引用精度 −0.107(0.578 vs 0.685),p=0.034,n=123**。
  - verified-full vs verify-only(completeness 贡献):精度 −0.228(p=0)、recall −0.149(p=0.0006)、
    coverage −0.212(p=0)、correctness −0.057(p=0)—— **每一轴皆负**。
- **头条判定(对照预注册失败判据):**
  - **头条主张对「忠实半」(verify-only)成立**:后验验证的引用显著比生成时更忠实(+0.090 精度,p=0.010)。
  - **对「完整系统」(verified-full)不成立** —— 触发预注册失败判据(引用精度不高反低于 baseline,p=0.034)。
  - **completeness/recheck 环节是净负**(消融证明):recheck 追加的片段引用 MiniCheck 判不支持,拉低精度;
    missing-required-fact 弃答把 coverage 砍半。**结论:关闭/重设 completeness/recheck,价值在 verify-only。**
  - 精度增益有代价:verify-only 用 coverage(0.552 vs 0.932)与 correctness(0.189 vs 0.273)换来 +0.090 精度——
    三轴必须同报,真实定位是「在更小的自选作答集上给更忠实的引用」。
- **报告项 6(entity_check 修复对 stratum-B):** 本轮用修复后的 entity_check;对 15 个已标注 stratum-B 项验证
  **8/9 假否决修复后通过 entity**、真 swap 仍否、1 边界翻转(Victoria)。未修则 verify 臂会多丢支持型 claim、
  低估方法。
- **待办:** 人工盲审 ~40 项(`g3-human-subsample.jsonl` 已产出,含隐藏 arm + MiniCheck 判)尚未评审 ——
  与 MiniCheck 自动数并列后可坐实/修正上述精度差(若显著不一致以人工为准)。

**后续分析(follow-up,非新实验;job `18240137`,2026-08-02):**

指南原计划纯用 G3 dump 再分析、不跑新作业,但核实后 dump 只有 `query_id` + 4 个聚合指标与最终
答案/引用 —— **gaps、引用来源、弃答成因、TRUE 概率四样均未留存**,故经批准用插桩版重跑 verified-full
(`scripts/g3_diagnosis.py`,`src/` 零改动,全脚本侧 wrapper)。贪心解码确定性,**完美复现 G3**:
378 完成 / 253 弃答 / **125 作答**,与 G3 的 125/378 一致 —— 故结论可直接套用到上表数字。
详见 `docs/generator/g3-followup-diagnosis.md`。

- **false-gap 假设被推翻:** 真实数据 **43/654 = 0.066**(MiniCheck 召回校正上界 0.106),而 G2 的
  0.786 own-fact coverage 隐含 ~0.214。**gap 检测器基本是对的**(~93% 正确),重调它只能触及 6.6%。
- **精度暴跌不是 recheck 引用质量:** draft 源 0.539 vs recheck 源 0.435,delta 0.103,**置换检验
  p=0.184 不显著**。真正机制是**答案长度稀释**——同一批 draft 引用按 **claim 级判为 0.869**,按
  **整答案级判仅 0.539**;recheck 给 105/125 作答例追加片段,答案中位长 60→97 字符。baseline 答案短
  (中位 54)、引用少(1.24/题),answer-level 指标系统性偏袒它。**G3 的 −0.228/−0.107 应读作部分是
  度量伪影**,待人工裁定。
- **弃答洪水不是假 gap 造成:** 251/253 走 completeness 路径,但其中**全部触发 gap 皆假的仅 9/251
  = 0.036**(至少一个假 8.4%)。真因是 **recheck 在真实 gap 上只有 38.3% 找得到**(234/611),叠加
  **any-fact-missing 即弃答**的全有全无策略 + 平均 1.80 必填事实/题 → 63% 弃答是算术必然。
- **决策表:证据分裂到两行,不强行归一** —— coverage 伤害属第 2 行(须重设,但要改的是**弃答策略**
  而非 gap 检测器);precision 伤害属第 3 行(度量/判官局限,待 Task 4 人工定夺);第 1 行(假 gap)
  **被证否**。
- **Task 5 扫描:** 阈值 0.05→0.90 上 coverage 0.579→0.474、claim 级精度 0.815→0.921,**平滑单调无断崖**,
  生产点 0.50 在曲线中不特殊(非事后挑选的有利点)。**但该曲线是 claim 级**,baseline 无法放上同一坐标
  (其答案从不切分为 claim,claim 级精度无定义);若要合图需对每个阈值的 repaired 答案再做一遍
  answer-level 判定(短作业,建议与后续 GPU 工作打包)。

---

## G4 — checklist 重建(Route B:LLM 分析器)与其验证门

**状态:** 分析器已实现(`GraniteQueryAnalyzer`,commit `2c81708`,含 Route A 底线修复)。
**本条目的 BEFORE 在看到任何验证数字之前写入并提交** —— 这是第三版 checklist 构造,
前两版(gold 派生的背景注记、规则式关键词袋)都失败且**自动指标全都看不出来**,只有人工
审阅和逐条看输出才发现。所以本轮的结构是**先建 → 按分析器自身标准验证 → 通过门 → 才跑实验**,
绝不从下游结果反推分析器质量。

**BEFORE(预注册):**

- 目的:规则式分析器无法知道一个问题有多种读法(预判歧义需要世界知识),所以 completeness
  机制从未得到公平检验。Route B 用 LLM 仅从**问题文本**构造 checklist,输出**覆盖义务**
  (「答案必须指明男子国际纪录保持者」)而非断言(「Ali Daei 保持纪录」)。
- **输入硬约束:** 分析器只见问题文本。**不得**接触 gold 长答案、`qa_pairs`、标注者字段、
  检索证据 —— `QueryChecklist` 是 Generator 的**输入**,任何 gold 派生都是 oracle 泄漏
  (即上一轮的缺陷 #2)。`qa_pairs` 仅用于**评估侧**验证。
- **验证门(在看数字前定死):**
  > **checklist 精度 ≥ 0.60**(生成的 requirement 中,能对应到真实 gold 去歧义读法的比例),
  > **且**人工抽检(~20 项,盲)中被判「合理」的比例 **≥ 0.60**。
  > 两项**任一不达标 → 分析器不适合驱动实验**:回退到 Route A 结果,并把
  > 「规则式 checklist 构造在开放域受限」作为结论如实报告。
- 门为何设在精度而非召回:**虚假 requirement → 虚假缺口 → 虚假弃答**,正是此前损害
  `verified-full` 的机制;漏掉一个读法只是少测,不会制造伤害。召回照报但不设门。
- **诚实预期:** 语义匹配用 MiniCheck,其召回仅 0.620,会**漏判**真实对应,因此自动精度是
  **下限**,可能低估分析器。故并列人工抽检;若两者显著分歧,以人工为准。另:过于笼统的
  requirement 会匹配上多个 gold 读法从而虚高精度,单列「匹配 >1 个 gold」的诊断项。
- **诚实预期(结果方向):** 规则式分析器在开放域产出空 checklist 是**领域限制而非缺陷**;
  即便 Route B 通过门,`verified-full` 相对 `verify-only` **无显著贡献**仍是完全可能且
  **合法可报告**的结果 —— completeness 机制的价值取决于 checklist 质量,不是必须被工程掉的失败。
- 下游判据(沿用上一版指南,不变):
  > completeness 机制有效,当且仅当 `verified-full` 在 **qa_pairs STR-EM 覆盖**上显著优于
  > `verify-only`,且引用精度无显著下降。
- 纪律:分析器**通过门即冻结**,不得为了下游数字好看而迭代提示词。可陈述的缺陷(如
  「checklist 漏掉了被比较的一方」)才是修改理由;「改措辞能抬高 completeness 差值」是调参,越界。
- 数据合规:只用 ALCE/ASQA;**HotpotQA / RGB / MuSiQue-Full 从不加载**。

**AFTER:** 未运行。

---

## G5 — verify-and-annotate 重设计(三臂标定)

**状态:** 代码就绪(`generator/verify_annotate.py`,commit `bcfc842`)。
**这是设计变更而非缺陷修复,故 BEFORE 在运行前写入并提交。**

**BEFORE(预注册):**

- 背景:现方法是**过滤器**(生成→验证→删除失败者),而过滤器按构造就是拿召回换精度 ——
  实测引用精度 0.762(baseline 0.602),但 coverage 0.552(0.932)、correctness 0.189(0.273)。
  TRUE 标定召回 0.747,即**约四分之一真正有支撑的声明被漏判**;在删除策略下这些变成
  **被摧毁的正确内容**,正是 correctness 缺口的主因。
- 变更:**标注而非删除**。三分路由 —— 蕴含且实体一致→保留并附**已验证**引用;蕴含但实体冲突→
  丢弃(二分类后端下实体冲突是唯一具体的矛盾信号);两者皆非→**保留并标注 unverified、不附引用**。
  验证走**引用路由 + 强制回退全扫描**(模型常内容对而索引错,无回退会因错标引用误删真陈述)。
- completeness 作为运行时机制**退休**(三次 checklist 构造均失败:gold 派生测错对象、规则式
  0.2% 词表命中率、单遍 LLM 44.4% 无对应且违反自身提示词断言假事实)。综合性下沉为 draft 提示词的
  **软目标**,评估侧仍由 qa_pairs STR-EM 度量。**这是有量化归因的、有范围的负面结果,须在写作中明说,
  不得悄悄移除。**
- 三臂(≥400 标定题):`baseline`(两种引用惯例)/ `verify-only`(删除,即现published方法,消融项)/
  `verify-annotate`(本重设计)。**保留 verify-only 才能把 coverage 的恢复归因到 delete→annotate 这一改动。**
- **预期方向 + 失败判据(预注册):**
  > 预期:coverage 与 correctness 向 baseline 回升;**已引用声明**上的引用精度维持在 verify-only 附近;
  > 引用召回下降(标注句无引用,这是本设计**有意付出的代价,不得隐藏**)。
  > **失败判据:若已引用声明的引用精度退回 baseline 水平**(说明标注策略让未验证内容以"已引用"身份混入),
  > **或 coverage 相对 verify-only 没有改善**,则重设计失败。
- 度量规则:引用精度**只在已引用声明上计算**,标注声明既不进分子也不进分母;引用召回按**全部答案句**计,
  故标注句会拉低它。ALCE 句级 + MiniCheck 为独立判官;**TRUE 是生产验证器,永不担任判官**。
- 纪律:**预注册后冻结**,不得为移动数字而调提示词或阈值。不重写未蕴含的声明(与 RARR 的
  retrieve-and-revise 重叠,且重写内容是新生成的、需再验证,破坏单轮纪律)—— 记为 future work。
- 数据合规:只用 ALCE/ASQA;**HotpotQA / RGB / MuSiQue-Full 从不加载**。

**AFTER:** 未运行。

---

## 本地(非 HPC)验证记录

- 2026-07-20:selector 门实现全套单测 LOCAL 通过(tests/selector 32 + registration 9),
  mypy strict 干净(49 files);全套 pytest 与基线逐项对比:失败集恒为 29 条
  已知 Windows TOML fixture 问题(与 selector 无关,已开修复任务),零回归。
- 2026-07-21:A2 互补覆盖层实现 LOCAL 通过(coverage 引擎 11 + 覆盖选择器 2 + 注册 1 = +14);
  gated.py 抽出 `_gate` 复用、门 parity 保持;全套 **332 passed / 1 xfailed**(基线 318/1 + 14,零回归);
  mypy strict 干净(51 files),ruff 干净。Windows fixture 问题已被队友修掉,不再计入。
