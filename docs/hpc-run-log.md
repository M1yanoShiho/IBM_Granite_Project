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
- raw:`results/ocr-smoke-documents.jsonl`(已按台账规则 `git add -f` 拉回)。`document_count`=2
  (1 image + 1 pdf)。image 源文档 caption 正确复述 sentinel("...REVENUE 2024 42 PERCENT
  GROWTH...");`Text in image:` 段落存在,OCR 命中 sentinel 数字 `42`。
- **OCR SMOKE: PASS。** 判据(§BEFORE)达成——caption 与图内 OCR 均生效,新增的
  PDF 内嵌图表处理链路(Docling converter → extract_pictures → Vision caption →
  OCR 追加)在真实模型下端到端跑通,非 test fake。
- **⚠️ 读 raw 时发现一个被断言漏掉的真实缺陷(PASS 仍成立,但断言太弱):**
  OCR 把 `2023 TO 2024` 识别成 **`2023 T0 2024`**(字母 O → 数字 0),而 **Vision caption
  识别正确**(`GROWTH 2023 TO 2024`)。三点后果:(a) 这是在**干净高对比度合成图**上发生的
  字符级错误 → 真实扫描件只会更差,把"Scanned PDFs depend entirely on OCR quality,可能
  静默降级"这条限制从猜测变成**实证**;(b) 冒烟断言只查 sentinel 数字 `42`,而 `42` 恰好
  识别正确,故**这类字符错误当前测不出来**——若要守住,断言需覆盖整条 sentinel 字符串;
  (c) OCR 文本会进入**可检索的文档正文**,故查询 `2023 to 2024` 可能匹配不上 `2023 T0 2024`,
  且同一份文档内 caption 与 OCR 互相矛盾(检索/生成阶段无从判断该信哪个)。
  **此发现只有读 raw 才能看到**(`.out` 只截前 400 字符),正是台账"raw 必须拉回"规则的价值。
- **已据此加强断言(2026-08-04,未重跑):** 门从"含数字 `42`"改为**要求整条主 sentinel
  逐字命中**(`REVENUE 2024 42 PERCENT`,已用本条 raw 离线验证仍 PASS);第二行
  (`GROWTH 2023 TO 2024 18 PERCENT`)**只报警不拦门**并打印 OCR 原文。分开的理由:第二行
  的 `T0` 是 EasyOCR 的**精度**问题,不是本 ingestion 路径的回归(本测试的职责是后者),
  拿它当硬门会让作业永久红,而永久红的测试会被无视。sentinel 常量改为从
  `scripts/make_ocr_smoke_pdf.py` import,避免门与被检图两处写死后走样。
  **本条 AFTER 的 PASS 记录仍以当时的弱断言为准。**
- 已知environment 坑,供下次复用此脚本时参考:(1) 项目要求 Python 3.11(<3.12,>=3.11),
  登录节点默认加载的 3.12.3 装不上 `evidence-rag`,需手动 `module load languages/python/3.11.15`
  重建 venv;(2) Docling HybridChunker 的默认 tokenizer(`all-MiniLM-L6-v2`)未被脚本注释
  里列出的预取步骤覆盖,离线模式下会因缺模型报错,需额外 `hf download sentence-transformers/all-MiniLM-L6-v2`。
  两坑均与本次功能验证结论无关,已通过重试规避,不影响 PASS 结论。

---

## R-2wiki-decompose — decompose 在多跳上崩塌的复现 + per-case 定位

**状态:** RUNNING(job 18259143 已提交,**本条目在读任何数字之前写**;三件套:现有
`scripts/run_retriever_eval.slurm` + `configs/experiments/retr_2wiki_{decompose,strong-bm25}.toml`
+ 本条目)。诚实说明:预注册**晚于提交**(提交时未先写),但早于看结果,故假设未被数据污染。

**BEFORE(预注册):**

- 背景:MengW7 的 3 数据集矩阵(`docs/retriever/eval-results.md`,commit 886cc8f)测到
  2Wiki 上 decompose MRR **0.5702** vs strong-bm25 **0.9580**(Δ **−0.3878**,p<0.0001),
  是整个矩阵里唯一的灾难级退化(SciFact 只 −0.052、NQ −0.047)。但 `runs/` 被 gitignore,
  per-case raw 只在 MengW7 自己的 `/user/work` 下,无法查看 → 本次在 jp25459 下**重跑两臂**
  取 per-case,而非新方法实验。
- 目的/假设:2Wiki 是多跳。decompose 把 query 拆成**互相独立**的子查询、各自检索再 RRF 合并;
  独立检索丢掉**跨跳依赖**(第二跳依赖第一跳答出的实体)→ 子查询各自召回"局部像、全局错"的段落,
  RRF 再把这些排到真正的多跳 gold 之上。
- 预期指标 + 方向:(a) **复现**:decompose MRR ≈ 0.57、strong-bm25 ≈ 0.958(±噪声);
  (b) **per-case 形状**:失败**集中**在 strong-bm25 命中(gold rank 1)而 decompose 把 gold
  排低/排出的 case 上。**诚实的替代假设(须排除)**:若退化是**均匀**的(decompose 到处略差、
  不集中在多跳难例),则根因在合并/拆分 prompt 本身,而非"多跳依赖"这个解释——两种形状指向
  不同修法,不能只看聚合 MRR 区分。
- 判定:失败集中于 strong-bm25 成功 case → decompose 主动有害,对多跳型语料应 gate off;
  退化均匀 → 查 RRF 合并与子查询 prompt。
- 精确命令:
  ```
  # 登录节点(一次性):
  pip install pandas pyarrow && hf download ibm-granite/granite-4.1-3b
  python -m evidence_rag.materializer.twowiki_cli --output runs/twowiki
  #   → documents 11585 / queries 2000 / gold_cases 2000(与 MengW7 同规模)
  # 提交:
  mkdir -p logs runs && sbatch scripts/run_retriever_eval.slurm \
    configs/experiments/retr_2wiki_decompose.toml \
    configs/experiments/retr_2wiki_strong-bm25.toml
  ```
- Git commit:9191acf;Seed:7(config `[run] seed`);top_k=50。

**AFTER(2026-08-04,job 18259143,gpu:rtx_3090:1,bp1-gpu030,两臂各 2000/2000):**

- raw(已按台账规则 `git add -f` 拉回,不再只存在于 bp1):
  `results/r1-2wiki-decompose-per-case.json`、`results/r1-2wiki-strong-bm25-per-case.json`
  (各 2000 条 per_case;`.gitattributes` 已豁免行尾转换,保证字节级可复现)。
- **复现:逐位精确。** 两臂 5 个指标与 MengW7(886cc8f)**小数点后 4 位全部一致**:
  decompose MRR .5702 / R@5 .4716 / R@10 .5491 / R@20 .6506 / Recall .7610;
  strong-bm25 .9580 / .6766 / .7222 / .7468 / .7678。Δ MRR = **−0.3878**,与预注册一致。
  本轮**自跑配对随机化检验**(`bash scripts/retriever_significance.sh 2wiki`,非引用 MengW7 的):
  MRR Δ −0.3878 **p=0.0000**、R@10 Δ −0.1731 **p=0.0000**,n=2000。
  附带收获:decompose 依赖 LLM 拆分,两次独立运行(不同时间/节点)仍逐位相同 → 解码是
  greedy/确定性的,后续 per-case 分析不受随机噪声干扰。
- **预注册判定:第二分支命中,第一分支(多跳依赖丢失)被数据推翻。** per-case 配对(n=2000):
  worse 1138 / tied 827 / better 35;损失集中度 worst 5%=12.5%、10%=24.8%、20%=48.5%、
  50%=95.4% 净损失——只受影响的 1138 条上若均匀则 worst 20% 应 ≈35%,实测 48.5%,
  **仅轻微集中,不存在承载崩塌的少数灾难 case** → 根因在融合/拆分环节,非"多跳难例"。
- **决定性证据:这是排序失败,不是检索失败。** 伤害随深度单调收缩:
  R@5 −20.5pp、R@10 −17.3pp、R@20 −9.6pp、Recall(top-50)**−0.7pp**。strong-bm25 完美命中
  (MRR=1.0)的 1854/2000 条里,decompose **仅 8 条(0.4%)彻底丢 gold**,**1053 条(56.8%)
  gold 仍在池中只是被排低**,793 条(42.8%)保持 rank 1。降级子集平均 MRR ≈ .27 →
  **gold 典型地从 rank 1 滑到 rank 3–4**。
- **机制(读码 + 数据共同支持):** RRF 把各子查询列表的 `1/(k+rank)` 加总。多跳 gold 只回答
  **一跳**,在一个子查询列表里排高、在其余列表缺席;而"各跳都沾一点、都不精准"的文档在
  **所有**列表拿中等分,累加后反超。**RRF 结构性奖励广谱平庸、惩罚单点精准,而多跳需要的
  正是单点精准。** 加重此效应的实现细节:`DecomposingRetriever._subqueries`
  (`src/evidence_rag/retriever/granite.py:278`)只返回子查询,**原始 query 仅在 LLM 空输出时
  作 fallback**——即 BM25 在 92.7% case 上把 gold 排第 1 的最强信号,从未进入融合。
- **写给 results-summary 的草稿:** decompose 在多跳上的崩塌不是"拆分破坏了多跳检索",
  而是 **RRF 融合把单跳专家文档系统性降级**;候选池几乎无损(top-50 recall −0.7pp),
  失的全是排序。因此**不必**对多跳语料整体 gate off decompose,而应先改融合。
- **下一步(廉价、直接针对上述机制):** 把原始 query 作为融合的一个臂加入(现在完全没有),
  或对其加权;预期 MRR 大幅回升而 recall 基本不动。此为独立预注册条目,不在本条覆盖。

---

## R-2wiki-decompose-orig — 把原始 query 加回融合臂(R1 诊断出的机制的直接修法)

**状态:** READY——三件套齐(代码 `include_original` 开关 + config
`configs/experiments/retr_2wiki_decompose-orig.toml` + 本条目);**本条目在跑之前写**。
承接 R1(results-summary):崩塌是 RRF 排序失败,不是检索失败。

**BEFORE(预注册):**

- 目的/假设:R1 定位到 `DecomposingRetriever` 只融合子查询,**原始 query 从不进融合**
  (仅 LLM 空输出时作 fallback);而 strong-bm25 用完整 query 在 **92.7%** 的 case 上把 gold
  排第 1。把原始 query 作为**一个额外融合臂**加回 → 该 ranking 重新参与 RRF → 被降级的
  gold 应回到高位。
- 预期指标 + 方向:**MRR 大幅回升**(baseline decompose .5702;strong-bm25 .9580 是上界参照,
  预期落在两者之间、显著高于 .5702);**recall 基本不动**(top-50 本就只差 −0.7pp,没有可回收的
  空间);R@5/R@10 应回升最多(降级伤害在浅层最重)。判定=对 decompose 基线臂配对显著性
  (`scripts/retriever_significance.sh 2wiki`,新增 pair `decompose-orig vs decompose`)。
- **诚实的替代结果(必须接受并如实报告):** (a) 若 MRR 只小幅回升,说明原始 query 那一臂被
  N 个子查询臂的 RRF 质量稀释(1 票 vs N 票),则修法方向对但**需要加权**而非等权加入;
  (b) 若 recall **下降**,说明挤占了子查询召回的多样性,是真实权衡而非免费收益;
  (c) 若几乎不动,则 R1 的机制推断错,需回头重看融合。三种都不是 bug,是不同结论。
- 兼容性:`include_original` **默认 False**,且**默认时不写入 index 参数**,故 MengW7 已记录的
  全部结果与已有 index cache 均不受影响(有回归测试守卫)。
- 精确命令(数据集已物化于 `runs/twowiki`,LLM 已预取):
  ```
  mkdir -p logs runs && sbatch scripts/run_retriever_eval.slurm \
    configs/experiments/retr_2wiki_decompose-orig.toml
  # 回来后(登录节点,CPU,秒级):
  scripts/retriever_significance.sh 2wiki
  ```
- Git commit:0bb9262(feat(retriever): optional original-query fusion arm for decompose);
  Seed:7;top_k=50;base=strong-bm25、k=60(与基线臂完全一致,唯一变量=`include_original`)。

**AFTER(2026-08-04,job 18265982,gpu:rtx_3090:1,bp1-gpu030,2000/2000,walltime ~1h):**

- decompose-orig:MRR **.7155** / R@5 .5727 / R@10 .6639 / R@20 .7371 / Recall **.7675**。
  配对检验 vs decompose 基线臂:MRR **+0.1453 p=0.0000**、R@10 **+0.1148 p=0.0000**,n=2000。
- **主预期命中,且是显著的:** MRR 大幅回升(.5702→.7155)、recall 基本不动(+0.0065)——
  与预注册一致。修法方向由 R1 的机制推断而来,数据支持该推断。
- **但同时命中预注册的替代结果 (a):幅度不足,需加权。** 只回收了 **37%** 的 MRR 差距
  (原 .3878,回收 .1453)。各深度回收比例:MRR 37% < R@5 49% < R@10 66% < **R@20 90%**——
  **越深回收越彻底、越靠榜首回收越少**。这正是"1 票 vs N 票"稀释的指纹:原始 query 那一臂
  能可靠把 gold 拉回前 20,却抢不回 rank 1。故等权加入方向对但不够,下一步应**给原始臂加权**。
- **反向加强 R1 的诊断:** 修法后 Recall .7675 与 strong-bm25 的 .7678 仅差 .0003——
  候选池质量已经等同,差的**纯粹是排序**,与 R1"排序失败非检索失败"完全一致。
- **⚠️ 实用结论(必须如实报告):修完仍明显不如直接用 strong-bm25**(.7155 vs .9580)。
  即在 2Wiki 这类多跳语料上,decompose **即使修好融合也不划算**——本修法补回的是**自伤**,
  没有让 decompose 变得有竞争力。R1 说"不必对多跳整体 gate off"是就"池子没坏"而言;
  就"该不该用"而言,**当前证据支持在多跳上仍优先用 strong-bm25**。
- raw:`results/r2-2wiki-decompose-orig-per-case.json`(已 `git add -f` 拉回)。

---

## R3 — 修法的普适性(SciFact/NQ)+ 换融合数学(best-rank),两问并行

**状态:** READY——三件套齐(代码 `fusion="best-rank"` + 4 个 config + 本条目);**跑之前写**。
承接 R2:原始臂等权加入只回收 37% MRR,且回收比例随深度递增(37%→90%)。

**BEFORE(预注册):**

- **本条同时问两个独立问题,分开判定,不许互相解释:**

  **Q1(普适性):`include_original` 是普遍有效,还是只在给 2Wiki 擦屁股?**
  2Wiki 上 strong-bm25 MRR **.9580**(BM25 在 92.7% case 直接命中 rank 1)——该数据集 query
  词汇特征极鲜明,**天花板天生就高、留给分解的空间本就极小**。故"2Wiki 上修法有效"不足以
  说明修法好。SciFact(baseline decompose .5584 / strong-bm25 .6105)与 NQ(.7682 / .8153)
  差距小得多、有真实提升空间,是更公允的检验场。
  - 预期 + 方向:两数据集上 decompose-orig MRR **↑ 且显著**;recall 基本不动。
  - **诚实的替代:** 若 SciFact/NQ 上**不显著或反而下降**,则修法本质是"2Wiki 特有的
    自伤修复",不是通用改进——那是更弱但更真实的结论,必须如实写。

  **Q2(机制的直接解法):把 RRF 的 sum 换成 max,能否比加原始臂更对症?**
  R1/R2 的机制是"sum 奖励广谱平庸、惩罚单点精准,而多跳 gold 正是单点专家"。若该机制成立,
  **直接改融合数学**应比"再加一臂去对抗稀释"更有效。实现为 `fusion="best-rank"`
  (max 为主、sum 仅作平局裁决;见 `fusion.best_rank_fusion` 的两条 caveat)。
  - **判定场是 SciFact,不是 2Wiki(重要,本条初稿曾把 Q2 只放在 2Wiki,是 scoping 错误):**
    2Wiki 上 BM25 在 92.7% case 直接命中 rank 1 → 在该数据集上**任何稀释完整 query 排名的
    融合都会伤、任何恢复它的改动都会有效**,那检验的是 2Wiki 的词汇特性,而非融合规则的优劣。
    SciFact baseline decompose 仅 .5584、有真实提升空间,才是"融合规则谁更好"的公允检验场。
    2Wiki 两臂仍跑,但只作**机制一致性的旁证**,不作 Q2 的判据。
  - 预期 + 方向:两数据集上 decompose-bestrank MRR **显著高于**同数据集的 decompose;
    与 decompose-orig 比较**方向不预设**——这正是要测的。第四臂(orig + best-rank)测叠加性。
  - **诚实的替代:** (a) best-rank 可能因平局过多而不升甚至下降(纯 max 的已知弱点,已用
    sum 做二级键缓解,但未必够);(b) 若 best-rank 与 orig 收益**不叠加**,说明两者在修同一
    个损伤,不是两个独立问题;(c) 若 best-rank 只在 2Wiki 有效、SciFact 上无效,则"改融合
    数学"这条路被否掉,机制解释仅对 2Wiki 这种高词汇区分度语料成立。

- **⚠️ 判定的标尺是 .9580,不是 .5702(承接 R2):** 上述任何一臂"比 decompose 高"都只是
  在**补回自伤**。**真问题是有没有任何配置能超过"什么都不做、直接 strong-bm25"**——超过了
  才说明分解在多跳上贡献了额外信息。若全部低于 .9580,诚实结论是**分解在此数据集上无用**,
  这是正当结论而非失败,不许用"相对 decompose 提升了 X%"来包装。
- 兼容性:`fusion` 默认 `"rrf"` 且**默认时不写入 index 参数**(与 `include_original` 同处理),
  故 MengW7 已记录结果与既有 index cache 全不受影响;有回归测试守卫两者。
- 精确命令(SciFact 需先 materialize;NQ 用既有 `runs/niah-base`):
  ```
  # 登录节点(SciFact 首次):
  evidence-rag-materialize-benchmark scifact --split test --output data/benchmarks/scifact/test
  # Q1(普适性)+ Q2 的判定场(SciFact):
  sbatch scripts/run_retriever_eval.slurm \
    configs/experiments/retr_scifact_decompose-orig.toml \
    configs/experiments/retr_scifact_decompose-bestrank.toml \
    configs/experiments/retr_scifact_decompose-orig-bestrank.toml \
    configs/experiments/retr_nq_decompose-orig.toml
  # Q2 的旁证(2Wiki,数据集已在 runs/twowiki):
  sbatch scripts/run_retriever_eval.slurm \
    configs/experiments/retr_2wiki_decompose-bestrank.toml \
    configs/experiments/retr_2wiki_decompose-orig-bestrank.toml
  # 回来后:
  bash scripts/retriever_significance.sh 2wiki   # 已含 bestrank 两对
  bash scripts/retriever_significance.sh scifact
  bash scripts/retriever_significance.sh nq
  ```
- Git commit:待本次改动提交后填;Seed:7;top_k=50;base=strong-bm25、k=60 全臂一致
  (Q1 唯一变量=`include_original`;Q2 唯一变量=`fusion`)。

**AFTER(2026-08-05,jobs 18269693 / 18269694 / 18271354,gpu:rtx_3090:1,bp1-gpu030):**

- **NQ 臂未跑成:** 18269693 的前三个 SciFact 臂全部完成,第四臂 NQ 因
  `runs/niah-base/manifest.json` 不存在而 `FAILED`(物化 10 万文档语料的成本另计)。
  **故 Q1 只在 SciFact 上得到回答,不是预注册承诺的两数据集。** 如实记录,不含混。

**四臂 + 标尺(MRR / R@10 / R@20 / Recall):**

| 臂 | SciFact(n=300) | 2Wiki(n=2000) |
|---|---|---|
| decompose(基线) | .5584 / .7094 / .7823 / .8432 | .5702 / .5491 / .6506 / .7610 |
| decompose-orig | .5824 / .7304 / .8063 / .8666 | .7155 / .6639 / .7371 / .7675 |
| decompose-bestrank | .5745 / .7286 / .7977 / .8432 | .8059 / .7069 / .7368 / .7642 |
| **decompose-orig-bestrank** | .5938 / .7580 / .8195 / .8683 | **.9100** / .7105 / .7385 / .7659 |
| strong-bm25(标尺) | .6105 / .7604 / .8134 / .8624 | .9580 / .7222 / .7468 / .7678 |

- **Q1:普适性成立(在 SciFact 上)。** `include_original` vs decompose 在 SciFact 上
  **四个指标全部显著**:MRR +.0240 **p=.0000**、R@10 +.0209 p=.0382、R@20 +.0240 p=.0422、
  Recall +.0233 p=.0387;2Wiki 同样四项全显著。**且它是 SciFact 上唯一显著的修法**
  → 不是"2Wiki 特有的自伤修复",是通用改进。预注册的替代(修法只对 2Wiki 有效)被排除。
- **Q2:命中预注册的替代 (c) —— best-rank 是数据集依赖的,不是通用的融合改进。**
  - 2Wiki:best-rank **单独就比 orig 强**(MRR **+.2357** p=0 vs orig 的 +.1453),且可叠加
    (orig 之上再加 best-rank:MRR **+.1945** p=0 → .9100)。
  - SciFact:best-rank **单独无显著效果**(MRR +.0161 **p=.2455**,四项全不显著);
    叠在 orig 之上**也不显著**(MRR +.0114 **p=.3671**)。
  - **与机制一致的解释:** best-rank 保护的是"某一臂里的压倒性名次"。2Wiki 存在这种名次
    (BM25 在 92.7% case 把 gold 排第 1),故 max 有东西可保;SciFact 没有
    (strong-bm25 自己 MRR 仅 .6105),**没有压倒性名次可保,max 就无从发力**。
    ⇒ "RRF 的 sum 惩罚单点专家"这个批评**为真但有适用范围**:仅在完整 query 排名本身很强的
    语料上成立,不是对 RRF 的普遍改进。
- **⚠️ 标尺判定(预注册的核心问题):没有任何一臂超过 strong-bm25。**
  - 2Wiki:最好的臂 .9100 仍**显著低于** .9580(MRR −.0480 **p=.0000**;R@10 −.0117 p=0;
    R@20 −.0082 p=.0002),仅 Recall 不可区分(−.0019 p=.2426)。
  - SciFact:最好的臂 .5938 vs .6105,**四个指标全部不可区分**(p=.1523 / .8805 / .6952 / .5811)。
  - 即:分解**最好也只是打平**(SciFact),从未赢过;而它每 query 要多付 N 次 LLM 调用 + N 次检索。
- **⚠️ 撤回一个中途的猜测:** 读到 SciFact 点估计(Recall .8683 > .8624、R@20 .8195 > .8134)时
  我曾提出"分解牺牲榜首精度换取池子覆盖率"。**配对检验否掉了它**:Recall +.0059 **p=.5811**、
  R@20 +.0061 p=.6952,均不显著;2Wiki 上也未复现(Recall −.0019 p=.2426)。
  **该猜测作废,不得进入报告。** 当时已标为"待检验假设",这次按检验结果撤回。
- **修法确实几乎修完了自伤,但仍不够:** 2Wiki 上 MRR 差距回收 **87.6%**
  ((.9100−.5702)/(.9580−.5702)),SciFact 回收 **67.9%**——即"崩塌基本是自伤、且已被基本修复"
  这一点被证实;**而即便如此分解仍不划算**,这是比"分解坏"更强的结论:不是没修好,是修好了也不值。
- raw:`results/r3-{scifact,2wiki}-*-per-case.json`(已 `git add -f` 拉回)。
- **写给 results-summary 的草稿:** 见 R3 节。**实用建议:两个数据集上都不要用 decompose**;
  若要用,`include_original=true` 是唯一普遍有值的开关,`fusion="best-rank"` 只在完整 query
  排名强的语料上有用。

---

## R4 — 给原始融合臂加权:测的是有没有内部最优,不是"MRR 会不会涨"

**状态:** READY——三件套齐(代码 `original_weight`,commit `32f9c7c` + 6 个 config + 本条目);
**本条目写于跑之前,也写于 R3 结果读取之前。** 承接 R2 的替代结果 (a):等权加入只回收 37%
MRR,且回收比例随深度递增(MRR 37% < R@5 49% < R@10 66% < R@20 90%)= "1 票 vs N 票"稀释。

**BEFORE(预注册):**

- **⚠️ 先声明一个会让天真读法失效的结构事实(本条最重要的一句):**
  RRF 分数 = `w/(k+rank_原始) + Σ_i 1/(k+rank_子查询_i)`。当 `w → ∞`,原始臂压倒其余,
  排序**收敛到 base retriever 本身**,即 **`decompose-orig(w→∞) ≡ strong-bm25`**。
  故"MRR 随 w 单调上升"在 2Wiki 上(strong-bm25 .9580 远高于 decompose-orig(w=1) .7155)
  **几乎是结构性必然,不构成发现**——那只是在两个已知端点之间插值。
  **本条初稿只准备了 2Wiki 的 sweep,是与 R3 初稿同型的 scoping 错误,已在提交前改正:
  补齐 SciFact 三点,并把判据改写如下。**

- **真正的判据(承接 R3 那把尺:标尺是 strong-bm25,不是 decompose):**
  **有没有任何有限 w 使 MRR 超过同数据集的 strong-bm25?**
  - **超过** → 子查询臂在完整 query 排名之上确实补充了信息,分解有独立价值,存在内部最优。
  - **单调逼近但从不超过** → 分解贡献为零,最优策略就是"把权重开到无穷"= 直接用
    strong-bm25。这是**正当且信息量充足的负结论**,与 R2 的实用结论一致,必须如实写,
    **不许用"相对 w=1 提升了 X%"包装成成功。**
- 预期指标 + 方向(分数据集,不许互相解释):
  - **SciFact(判定场,有真实空间:decompose .5584 / strong-bm25 .6105):** 预期存在
    **内部最优**——中等 w 处 MRR 高于 w=1,且与 .6105 可比。这是本条的主问题。
  - **2Wiki(旁证,不作判据):** 预期 MRR 随 w 单调升、recall 基本不动(R2 已证池子完好,
    .7675 vs .7678)。若出现**非单调**(中间峰后回落),反而是有信息的——说明子查询臂
    并非纯噪声,存在真实的混合收益。
- **诚实的替代结果(必须接受并如实报告):**
  (a) **SciFact 上 w 越大越好、一路单调** → 无内部最优,分解无贡献,结论同"直接用
      strong-bm25";这会**同时削弱 R2 的实用价值主张**,要写进 results-summary。
  (b) **任何 w 下 recall 下降** → 原始臂挤占了子查询的多样性,是真实权衡而非免费收益。
  (c) **SciFact 与 2Wiki 方向相反** → 加权是数据集特异的,不是通用改进(与 R3 的 Q1
      同型风险,两条要合起来读)。
  (d) **w=2/3/5 三点全部与 w=1 无显著差异** → 稀释解释(R2 的机制推断)被推翻,损失在别处,
      需回头重看融合而非继续调权重。
- **与 R3 的关系(必须先读 R3):** R3 的 Q1 在测 `include_original` 是否普适。**若 R3 判定
  它只是 2Wiki 特有的自伤修复,则本条在 SciFact 上大概率也无效**——那不是本条失败,而是
  R3 的结论在本条上的自然推论。两条不可互相解释,但 R4 的解读必须以 R3 的 Q1 为前提。
- 兼容性:`original_weight` 默认 `1.0` 且**默认时不写入 index 参数**(与 `include_original`、
  `fusion` 同处理),故已记录结果与既有 index cache 全不受影响;有回归测试守卫。
  每个 w 生成**不同的 index signature**(实测 w2/w3/w5 各异),故 sweep 各点不会误共用索引。
- 精确命令(两数据集均已 materialize;**建议等 R3 回来、确认 Q1 后再提交 SciFact 臂**):
  ```
  # 判定场(SciFact):
  sbatch scripts/run_retriever_eval.slurm \
    configs/experiments/retr_scifact_decompose-orig-w2.toml \
    configs/experiments/retr_scifact_decompose-orig-w3.toml \
    configs/experiments/retr_scifact_decompose-orig-w5.toml
  # 旁证(2Wiki):
  sbatch scripts/run_retriever_eval.slurm \
    configs/experiments/retr_2wiki_decompose-orig-w2.toml \
    configs/experiments/retr_2wiki_decompose-orig-w3.toml \
    configs/experiments/retr_2wiki_decompose-orig-w5.toml
  # 回来后(登录节点,CPU,秒级):
  bash scripts/retriever_significance.sh scifact
  bash scripts/retriever_significance.sh 2wiki
  ```
- Git commit:`32f9c7c`(feat(retriever): weight the original-query fusion arm on decompose);
  Seed:7;top_k=50;base=strong-bm25、k=60、`include_original=true` 全臂一致,
  **唯一变量=`original_weight`**;对照臂=已有的 `decompose-orig`(w=1)。

**AFTER:** 未运行。<!-- 填:job id、两数据集各 w 的五指标、vs w=1 与 vs strong-bm25 的配对 p、
是否存在内部最优、是否有任何 w 超过 strong-bm25、SciFact 与 2Wiki 是否同向 -->

---

## R5 — 语料规模下的检索开销:先测基线曲线,再谈优化(Bharat 反馈第 4 项)

**状态:** READY——三件套齐(`scripts/retriever_scaling.py` + `scripts/run_retriever_scaling.slurm`
+ 本条目);**跑之前写**。这是 Bharat 2026-07-26 四条反馈里**唯一零数据**的一项。

**BEFORE(预注册):**

- 目的:测 sparse 检索的**索引构建时间**与**每 query 延迟**如何随语料规模增长,
  给"enterprise scale"一个有数字的答案,而不是"应该没问题"。两者分开计时:前者每语料付一次,
  后者每 query 付一次,合成一个数会掩盖是哪一头疼。延迟报 mean/p50/**p95**,不让长尾被均值吃掉。
- **读码得出的预测(在测之前写下,可被数据否掉):** `BM25Retriever.retrieve`
  (`src/evidence_rag/retriever/bm25.py`)**没有倒排索引**,每 query **线性扫全部 chunk**——
  哪怕 query 词只出现在 3 个文档里也要访问 N 个 chunk。更关键的是循环体内每个 chunk 都执行
  `counts = Counter(tokens)`,而该结果**与 query 无关**,即每次查询白算一遍"全语料词频"。
  故预测:
  1. **每 query 延迟随 chunk 数近似线性增长**(不是次线性);
  2. **`ms/1k_chunks` 近似常数**(若该比值随规模明显下降,则预测 1 错,需重看);
  3. 索引构建时间也随规模线性(tokenise + df 各一遍),但常数远小于 Q 次查询的累计开销。
- **诚实的替代:** 若延迟明显次线性,说明有我没读到的提前退出/稀疏性效应,预测 1 作废;
  若 p95 与 mean 差距很大,则瓶颈不是规模而是**查询长度分布**(长 query 触碰更多 term),
  那是另一条修法路线,不能混谈。
- **判定与后续:** 本条**只测不改**。若预测 1、2 成立,则下一条独立实验做两个可分离的修法并
  各自量化:(a) 把 `Counter(tokens)` 预计算到构造期——**纯缓存,检索输出应逐位不变**,
  可用 report 的 aggregate 与既有 R1–R3 raw 对比来证明零质量变化;(b) 建倒排索引(改动大得多)。
  **先测后改,才能给出"加速 N 倍"这种有基线的说法。**
- **范围限制(必须写进结论):** 这测的是**本实现**(手写 Python BM25),不是 BM25 这个算法。
  结论只归属于实现。且 SciFact 上限 5183 文档,只覆盖约 **10×** 规模跨度;
  再往上一个数量级需先物化 `runs/niah-base --corpus-size 100000`(与 R3 里 NQ 臂缺的是同一件事)。
- 精确命令(CPU 分区,无需 GPU/LLM/模型下载):
  ```
  mkdir -p logs results
  sbatch scripts/run_retriever_scaling.slurm
  # 或指定更大语料:
  # sbatch scripts/run_retriever_scaling.slurm runs/niah-base/manifest.json results/retriever-scaling-niah.json
  ```
- Git commit:待本次改动提交后填;固定项:top_k=50、chunk_size=180/overlap=30、queries=50、
  retriever=strong-bm25(k1=0.9/b=0.4);sizes=500/1000/2000/3000/4000/5183。

**AFTER(2026-08-05,job 18277023,partition=compute,bp1-compute136,CPU only,walltime <5min):**

| docs | chunks | build_s | mean_ms | p95_ms | ms/1k_chunks | peak_rss_MB |
|---|---|---|---|---|---|---|
| 500 | 850 | 0.107 | 14.67 | 19.29 | 17.25 | 71.0 |
| 1000 | 1704 | 0.211 | 28.80 | 39.52 | 16.90 | 73.2 |
| 2000 | 3338 | 0.408 | 54.98 | 73.18 | 16.47 | 91.1 |
| 3000 | 5091 | 0.630 | 87.69 | 117.73 | 17.23 | 112.1 |
| 4000 | 6767 | 0.877 | 120.83 | 160.40 | 17.86 | 123.7 |
| 5183 | 8778 | 1.094 | 153.51 | 201.55 | 17.49 | 161.5 |

- **预测 1 成立(线性):** chunk 数 850→8778(**10.33×**),mean 延迟 14.67→153.51ms(**10.47×**)。
- **预测 2 成立(比值恒定):** `ms/1k_chunks` 全程 16.47–17.86,围绕 **~17.2 波动 ±4%**,无次线性迹象。
- **预测 3 成立:** build 0.107→1.094s(**10.22×**,线性),但**仅 50 个 query 就耗 7.7s = 构建的 7 倍**
  → 任何真实查询量下索引构建开销可忽略,**痛点在每 query,不在建索引**。
- **预注册的替代假设被排除:** p95/mean 全程稳定在 **~1.33**(1.315/1.372/1.331/1.343/1.327/1.313),
  长尾**等比例**放大而非发散 → 瓶颈是规模本身,不是"查询长度分布",那条修法路线不成立。
- **外推(按 17.2 ms/千 chunk、SciFact 的 1.69 chunk/doc;假设线性延续):**
  10 万 chunk ≈ **1.7 s/query**;100 万 chunk ≈ **17 s/query**;
  **100 万文档(≈170 万 chunk)≈ 29 s/query**。内存亦近线性(71→161MB)。
  ⚠️ 外推**只在线性假设下成立**;内存压力与缓存失效只会使其更差,不会更好。**上界仍是实测的 5183
  文档(约 10× 跨度)**,再高一个数量级需先物化 `runs/niah-base --corpus-size 100000`。
- **定论:当前实现不具备 enterprise scale,且瓶颈已定位到具体两处**(见 §BEFORE 的读码预测):
  (a) 每 query 对每个 chunk 重建 `Counter(tokens)`——与 query 无关的纯重复计算,
  chunk_size=180 而 query 有效词 ~6,故这部分占了绝大多数常数;
  (b) 无倒排索引 → 每 query 必扫全部 chunk,是**线性本身**的来源。
  (a) 是纯缓存、可证明输出逐位不变;(b) 才改渐近复杂度。**两者独立立项,不在本条内做。**
- raw:`results/retriever-scaling-scifact.json`(已按台账规则 `git add -f` 拉回,commit `8e33a35`;
  `.gitattributes` 的 `results/** -text` 保证字节级不被行尾转换改写)。**本条 AFTER 至此完成。**
  上表六行已从该 raw 重新聚合复核,逐项一致;三条推导亦复核:chunk **10.33×** / 延迟 **10.47×** /
  build **10.22×**,p95/mean 全程 **1.313–1.372**。运行环境:Python 3.11.15、strong-bm25、top_k=50、
  chunk_size=180/overlap=30。
- **已写入 results-summary:** Retriever 节 **R5**(编号与本台账对齐;R4 未跑故缺位,该处有说明)。
  **注意措辞范围:这是本实现的性质,不是 BM25 算法的性质。**

---

## R6 — 常数项外提在**真语料**上的 before/after(把本地合成数升级为正式结果)

**状态:** READY——代码已合入(`159069c`)+ 复用 `scripts/run_retriever_scaling.slurm` + 本条目;
**跑之前写。** R5 把开销拆成 (a) 常数项与 (b) 无倒排索引两处,(a) 已修但**只在本地合成语料上量过**
(6.0–8.4×),按本台账规则该数不得当作正式结果。本条把它做实。

**BEFORE(预注册):**

- 目的:同一脚本、同一语料、同一参数,**唯一变量=代码版本**(R5 跑在 `bec0f72`,本轮跑在
  `159069c` 之后)。R5 的 raw 已在仓库(`8e33a35`),故 before 臂无须重跑,直接配对比较。
- **预期 1(加速幅度):** 8778 chunk 处 mean 从 **153.51 ms** 降到 **~20–26 ms**;
  `ms/1k_chunks` 从 **~17.2** 降到 **~2.2–2.9**。区间取自本地实测的 6–8×。
- **预期 2(线性必须保持——最能证伪的一条):** (a) 是常数项优化、**不动渐近复杂度**,
  故 `ms/1k_chunks` 应**仍近似恒定**,只是整体下移。若它转为随规模上升或下降,
  说明改动的影响不止常数项,须回头查。
- **预期 3(内存上升):** 缓存 ~2.9 KB/chunk ⇒ 8778 chunk 处 peak RSS 从 **161.5 MB** 升到
  **~185 MB**。**这条是代价不是收益,必须同报。**
- **诚实的替代结果(必须接受并如实报告):**
  (a) **真语料加速显著低于 6×** → 合成词表未能代表真实文本,本地数被高估,以真语料为准
      并撤回本地那组数字;
  (b) **内存涨幅显著超过 ~24 MB** → 2.9 KB/chunk 估计偏低,真实 chunk 词汇量更大,
      则 (a) 在大语料上的内存代价比已记录的更严重;
  (c) **`ms/1k_chunks` 不再恒定** → 见预期 2,改动有未预期的影响。
- **本条不测什么:** 不测检索质量。(a) 的输出逐位不变已由
  `tests/retriever/test_bm25_scoring_equivalence.py` 在代码层证明,**不需要也不应该**再用指标
  复跑去"验证"——那只会把一个确定性事实变成一次噪声测量。
- 精确命令(**必须给第二个参数,否则会覆盖 R5 的 before raw**):
  ```
  cd /user/work/$USER/IBM_Granite_Project && git pull
  mkdir -p logs results && sbatch scripts/run_retriever_scaling.slurm \
    data/benchmarks/scifact/test/manifest.json \
    results/retriever-scaling-scifact-hoisted.json
  # 回来后按 R5 同规则把 raw 拉回:
  git add -f results/retriever-scaling-scifact-hoisted.json
  ```
- Git commit:`159069c`(perf: hoist the per-query constants out of the BM25 scan);
  固定项与 R5 逐项一致:top_k=50、chunk_size=180/overlap=30、queries=50、
  retriever=strong-bm25、sizes=500/1000/2000/3000/4000/5183、partition=compute。

**AFTER(2026-08-06,job 18277238,partition=compute,COMPLETED,elapsed 00:00:30):**

> **⚠️ 本条的三处读数已被 R6b(同 job 同节点配对)推翻,原文全部保留存底,不删改。**
> (1) 加速比 **8.29–10.17× → 实为 5.27–5.40×**(被节点差异抬高 57–88%);
> (2) "加速比随语料增大而下降" → 同节点下**是平的**(±1.2%),趋势不存在;
> (3) "after 的 ms/1k 升 +26.9%、残留超线性" → 同节点下仅 **+5.0%**,该读数**证据不足,撤回**。
> **本条下文中凡涉及这三项的结论,一律以 R6b 为准。** 唯一未被推翻的是内存那条(见预期 3),
> 它在 R6b 中第二次归零。本条自身的价值在于**它的缺陷被自己的 build 数据暴露出来**,
> 从而催生了 R6b —— 保留它是为了记录这个链条。

| chunks | before(R5) | after | 加速 | ms/1k before | ms/1k after | p95/mean b→a | RSS b→a |
|---|---|---|---|---|---|---|---|
| 850 | 14.66 | **1.45** | **10.12×** | 17.25 | 1.705 | 1.315 → 1.609 | 71.0 → 71.1 |
| 1704 | 28.80 | **2.83** | **10.17×** | 16.90 | 1.662 | 1.372 → 1.663 | 73.2 → 71.1 |
| 3338 | 54.98 | **5.81** | **9.47×** | 16.47 | 1.740 | 1.331 → 1.618 | 91.1 → 87.1 |
| 5091 | 87.69 | **9.35** | **9.37×** | 17.22 | 1.837 | 1.343 → 1.721 | 112.1 → 99.5 |
| 6767 | 120.83 | **13.38** | **9.03×** | 17.86 | 1.977 | 1.327 → 1.755 | 123.7 → 132.7 |
| 8778 | 153.51 | **18.51** | **8.29×** | 17.49 | 2.108 | 1.313 → 1.750 | 161.5 → 154.5 |

- **预期 1(6–8×,8778 处 20–26 ms):成立,但实测落在预注册区间之外、在好的一侧。**
  实测 **8.29–10.17×**;8778 处 **18.51 ms**,低于区间下界 20。诚实说法:**方向对,区间估窄了**。
  归因:真实文本每 chunk 的不同词数高于本地合成语料的 124,故被省掉的重复计算更多。
  **本地合成数不是高估而是低估**,与预注册的替代结果 (a) 相反。
- **预期 2(ms/1k 应保持恒定):按字面被证伪,但证据指向的结论与字面相反。**
  after 的 ms/1k 从 1.662 升到 2.108(**+26.9%**),不再像 before 那样平(+8.4%)。
  **但把两臂逐点相减,被移除的那部分 ms/1k 是平的:15.55 / 15.24 / 14.73 / 15.39 / 15.88 / 15.38
  (±4%)** ⇒ **被外提的工作确实是纯线性的,改动确实只是常数项移除。**
  故残留的轻微超线性**不是本次改动引入的**,而是原先被大常数盖住、现在基线缩小 ~9× 后显形。
  绝对增幅 after 为 +0.45 ms/1k(10× 语料跨度),before 为 +1.02 —— **绝对值反而更小。**
  **实际后果:线性外推的依据变弱了。** R5 敢外推是因为 ms/1k 平;现在它在升,故
  "1.9 ms/1k × 100 万 chunk ≈ 1.9 s/query" 比 R5 的 17 s 更不可靠,**会偏乐观**,不得当作承诺。
- **预期 3(peak RSS 应升到 ~185 MB):被证伪。** 实测 **154.5 MB,反而低于 before 的 161.5**。
  逐规模 +0.1 / −2.1 / −4.0 / −12.6 / +9.0 / −7.0,纯噪声无趋势。
  ⇒ **"缓存 ~2.9 KB/chunk、100 万 chunk 即 ~2.9 GB"这一本地估算得不到真语料支持。**
  最可能的解释是**量错了工具**:peak RSS 是全进程峰值,被建索引阶段的临时分配主导,
  看不到稳态增量。**该声明已从 results-summary 撤回**(见该文件 R5 节),
  R6b 的替代结果 (b) 会给它第二次检验。
- **附带发现(预注册未预期):尾部相对变重。** p95/mean 从 **1.313–1.372** 升到 **1.609–1.755**。
  与预期 2 同源:去掉每 query 的大常数后,可变部分在总量中的占比上升。绝对 p95 全面大幅下降
  (201.55 → 32.39 ms),故这是**比例效应而非退化**。
- **⚠️ 本条存在一个未受控的混淆,必须与结论同时声明:**
  **index build 比 R5 快了 10–16%**(1.094→0.964 s 等),而 hoist 让 build **多做事**
  (要额外物化每 chunk 的 `Counter`)—— **build 只可能变慢,不可能变快。**
  故这 10–16% 只能来自机器:R5 跑在 `bp1-compute136`,本轮为另一次分配。
  ⇒ **上表的加速比里裹着一份节点方差。** 效应(8–10×)远大于混淆(10–16%),**主结论不倒**,
  但**精确数字不得当作定值引用**。已据此立 **R6b**(同 job 同节点配对 A/B,含噪声底判定)。
- raw:`results/retriever-scaling-scifact-hoisted.json`(已 `git add -f` 拉回)。
  上表已从该 raw 重新聚合复核,与 job `.out` 逐位一致;比较由
  `scripts/compare_scaling_runs.py` 产出,该脚本会自动标出上述 build 反常。

---

## R6b — 同 job 同节点的配对 A/B(修掉 R6 设计里的节点混淆)

**状态:** READY——三件套齐(`scripts/run_retriever_scaling_ab.slurm` +
`scripts/compare_scaling_runs.py` + 本条目);**跑之前写。**

**为什么要有这一条(R6 的设计缺陷,由 R6 自己的数据暴露):**

R6 把新代码的一次运行,与 R5 记录在案的旧数字相比 —— **两次运行在不同 job、不同节点、不同时间**。
混淆的证据是一个**逻辑上不可能的结果**:R6 的 index build 比 R5 **快了 10–16%**,而 hoist 让
build **多做事**(要额外物化每 chunk 的 `Counter`)。build 只可能变慢。所以那 10–16% 只能来自
机器,不可能来自改动 ⇒ **R6 测出的 8.29–10.17× 里裹着一份节点方差。**
效应远大于混淆,故 R6 的主结论不倒;但精确数字带约 10–15% 的不确定性,不该当作定值引用。

**BEFORE(预注册):**

- 设计:**两臂在同一个 slurm 分配里背靠背跑**,同节点、同时段。before 臂用
  `git worktree` 检出 **`159069c^`(= `62f04a3`)的真实历史代码**,不是转录副本,故无需维护同步。
- **顺序 before → after → before,第三臂是噪声底。** 两次 before 的差距即本次测量的
  run-to-run 漂移;若它与 before/after 的差距同量级,则本次**分辨不出**改动,判定为未测。
  `compare_scaling_runs.py` 直接输出该判定,不留给读者自己看出来。
- **预期 1:** 加速仍在 **6–10×** 量级,方向不变。R6 的 8.29–10.17× 若基本复现,则该区间坐实。
- **预期 2(本条真正要测的):** **build 必须变慢或持平,不得变快。** hoist 把工作移入 build,
  所以 `build before/after < 1.0`。若同节点下 build **仍然变快**,则我对成本归属的理解是错的,
  须回头查——这是本条最强的证伪点。
- **预期 3:** 两次 before 的漂移 **< 5%**,远小于效应。若漂移与效应同量级,本条自身作废,
  须改用更多重复或更长采样。
- **诚实的替代结果:**
  (a) 同节点加速**明显低于 R6 的 8–10×** → R6 的数被节点差异抬高,以本条为准并修正 R6 的 AFTER;
  (b) **peak RSS 仍不上升** → 与 R6 一致,则"缓存内存 ~2.9 KB/chunk"这一**本地估算被两次真语料
      实测否定**,须从 results-summary 撤回,并说明 peak RSS 可能根本测不到该增量
      (被建索引阶段的临时分配主导),即**量错了工具**;
  (c) 漂移过大 → 见预期 3,本条作废而非强行解读。
- **本条不测什么:** 不测检索质量(输出逐位不变已由 `test_bm25_scoring_equivalence.py` 在代码层
  证明);不改 R5 的结论(R5 是绝对量级与线性性,与本条的相对比较无关)。
- 精确命令(**必须走登录节点的 wrapper,不要直接 sbatch**):
  ```
  cd /user/work/$USER/IBM_Granite_Project && git pull
  bash scripts/submit_scaling_ab.sh
  # 回来后三份 raw 一并拉回:
  git add -f results/retriever-scaling-ab-{before-1,after,before-2}.json
  ```
  **环境坑(实测,job 18280216):计算节点 PATH 里没有 `git`**,故 before 臂的 worktree
  必须在**登录节点**建好,作业只跑 python。首次直接 sbatch 的版本在 `git worktree add`
  处被 `set -e` 中止,**未产生任何结果**,不影响本条的预注册。wrapper
  `scripts/submit_scaling_ab.sh` 负责建 worktree、记录两臂 commit 到
  `logs/scaling-ab-refs.txt`(作业自己解析不了,不记就无从知道比的是哪两个 commit)、再提交。
- Git commit:待本次改动提交后填;固定项与 R5/R6 逐项一致(top_k=50、chunk_size=180/overlap=30、
  queries=50、strong-bm25、sizes=500/1000/2000/3000/4000/5183、partition=compute)。
  **唯一变量=代码版本**,且**节点、时段、进程环境全部受控**。

**AFTER(2026-08-06,job 18280337,partition=compute,节点 `bp1-compute129`,三臂同 job 同节点;
首次提交 job 18280216 因计算节点无 `git` 在 worktree 创建处被 `set -e` 中止,未产生任何结果):**

| chunks | before-1 | after | 加速 | before-2 | build b→a | RSS b→a |
|---|---|---|---|---|---|---|
| 850 | 12.94 | 2.45 | **5.27×** | 12.69 | 0.094 → 0.098 | 71.0 → 72.5 |
| 1704 | 25.89 | 4.91 | **5.27×** | 25.85 | 0.186 → 0.199 | 73.6 → 72.5 |
| 3338 | 51.17 | 9.51 | **5.38×** | 50.15 | 0.369 → 0.413 | 91.3 → 86.6 |
| 5091 | 78.67 | 14.57 | **5.40×** | 77.57 | 0.567 → 0.615 | 112.1 → 99.5 |
| 6767 | 105.17 | 19.73 | **5.33×** | 104.69 | 0.754 → 0.842 | 123.8 → 132.7 |
| 8778 | 138.36 | 26.26 | **5.27×** | 135.90 | 0.983 → 1.067 | 161.9 → 152.4 |

- **预期 2(build 必须变慢)成立 —— 混淆确认被消除。** `build before/after = 0.89–0.96×`,
  **build 一致变慢**,与 hoist 把工作移入建索引相符。R6 里那个"**变快 10–16%**"的不可能结果
  **在同节点下完全消失** ⇒ 它确实是节点,不是改动。本条最强的证伪点未触发。
- **预期 3(漂移 < 5%)成立。** 两次 before 相差 **≤2.0%**,比最小加速比小 **214 倍**。
  `VERDICT: the effect is resolved well clear of the noise floor.` 本次测量有效。
- **预期 1(加速仍在 6–10×)不成立,预注册的替代结果 (a) 命中。**
  同节点实测 **5.27–5.40×**,**低于**预注册区间,也**远低于 R6 报告的 8.29–10.17×**。
  ⇒ **R6 的加速比被节点差异抬高了 57–88%,以本条为准,R6 的 AFTER 已据此修正。**
- **⚠️ 本条最重要的发现(预注册未预期):混淆远大于它的代理指标。**
  build 时间只差 10–16%,据此我把节点效应估作同量级 —— **错了**。同一份代码换节点:
  **before 臂只差 6.9–13.0%,而 after 臂差 41.9–73.5%。**
  机制:外提之后热点从"重建 `Counter`"(CPU 受限)变成"随机访问 8778 个已缓存的 `Counter`"
  (**内存延迟受限**),后者对节点的缓存/内存子系统敏感得多。
  ⇒ **优化后的代码比优化前对机器敏感一个数量级,所以任何用未优化部分(如 build)去估计
  节点效应的做法都会系统性低估。** 教训:**跨运行比较必须同节点配对,不能靠代理指标校正。**
- **⚠️ R6 的另两处读数同样是跨节点假象,一并撤回:**
  (1) **"加速比随语料增大而下降"(10.17×→8.29×)** —— 同节点下加速比**是平的**:
      5.27 / 5.27 / 5.38 / 5.40 / 5.33 / 5.27(**±1.2%**)。原趋势不存在。
  (2) **"after 的 ms/1k 升 +26.9%,残留超线性"** —— 同节点下 after 仅 **+5.0%**
      (2.849→2.992),before **+3.7%**(15.195→15.763),**两臂都平**,after 只多涨 1.2 个百分点。
      R6 那个 +26.9% 主要是跨运行噪声,不是曲线形状。**"残留超线性原先就有"这一说法证据不足,
      撤回**;真实的残留超线性(若有)在 10× 跨度内**不超过 ~1 个百分点**,须更大跨度才测得出。
- **预注册的替代结果 (b) 命中:peak RSS 仍不上升**(−12.6 至 +8.9 MB,无趋势)。
  ⇒ **"缓存内存 ~2.9 KB/chunk、100 万 chunk 即 ~2.9 GB"已被真语料否定两次(R6、R6b),
  从"暂缓引用"升级为确定作废。** 该本地 `sys.getsizeof` 估算与 peak RSS 的分歧仍未解释,
  最可能是 peak RSS 被建索引阶段的临时分配主导、测不到稳态增量 —— **若要给出内存代价,
  须换工具(如 `tracemalloc` 在稳态取样),不能继续用 peak RSS。此项列为未测。**
- raw:`results/retriever-scaling-ab-{before-1,after,before-2}.json`(已 `git add -f` 拉回)。
  上表已从三份 raw 用 `scripts/compare_scaling_runs.py` 重新聚合复核,与 job `.out` 逐位一致。
- **写给 results-summary 的草稿:** 已并入 R5 节(该节的加速比、趋势与内存三处均按本条修正)。

---

## R7 — 检索的提升能否**送达下游**?(第一次把 retriever 接进完整 pipeline 测量)

**状态:** READY——三件套齐(四个 `configs/experiments/pipe_2wiki_*.toml` + 现有 pipeline runner
+ 本条目);**跑之前写。**

**为什么要有这一条:**

至今所有 retriever 工作(MengW7 的 3×8 矩阵、R1–R3)测的都是 **MRR/Recall**,所有 Generator 工作
(G1–G5)测的是**给定证据后**的生成质量。**中间那条链——检索指标的提升能否传导到下游——从未被测过。**
共享基础设施的 pipeline runner 与 `system.core.*` 指标正是为此而建,建成后几乎没被使用。

风险有本项目自己的先例:**S6 证明了"孤立探针上很漂亮的收益,接到真实池里不但不级联,反而制造了
38pp 的 false_conflict"**。同型风险原样适用于检索,不能假定传导成立。

**⚠️ 必须先讲清楚本条测的到底是什么(否则结果一定被误读):**

`build_generator` 目前**只支持 `extractive`**(`composition.py:457`),而
`ExtractiveGenerator` **不生成答案**,它把选中的证据原文拼接后加引用(`extractive.py:24`)。
`answer_match` 是**规范化后的子串包含**(`scoring.py:127`,仅小写化+取词)。两者相乘的含义是:

> **`system.core.answer_match` = 标准答案串是否逐字出现在 top-`max_selected`(=5)个选中 chunk 中。**

所以本条测的是「**证据传递**」——检索到了正确文档之后,**含答案的那个 chunk 有没有真的送到下游**
——**而不是答案质量**。二者的区别是实质性的:`retriever.core.document_recall` 问"gold 文档在池里吗",
本指标问"gold **答案文本**在送给 generator 的那 5 个 chunk 里吗"。**两者可以大幅背离**
(文档命中但含答案的 chunk 没进 top-5),而这个背离正是本条要量的东西。

**本条不测生成质量,不得如此引用。** 真实 generator(`granite.py`/`verified.py`)**尚未接入
`composition.py`**,这本身是一个待办(见下"附带发现")。

**BEFORE(预注册):**

- 设计:**唯一变量 = retriever**。selector(`top-k`)、generator(`extractive`)、
  `top_k=50`、`max_selected=5`、`seed=7`、数据集(`runs/twowiki`)四臂逐项相同。
  selector 为纯直通,故链路是 retriever → top-5 chunk → 答案串包含,因果干净。
- 四臂横跨已记录的极宽 MRR 区间(取自 `docs/retriever/eval-results.md`,n=2000):

  | 臂 | 已记录 MRR | 已记录 Recall |
  |---|---|---|
  | decompose | **.5702** | .7610 |
  | bm25 | .9434 | .7621 |
  | strong-bm25 | .9580 | .7678 |
  | hybrid-rrf | **.9828** | .7995 |

  **MRR 跨度 .41,而 Recall 跨度仅 .039** —— 这个不对称是本条的核心工具:
  它能把「**排序**改善的传导」与「**池覆盖**改善的传导」分开。
- **内建的 harness 自检:** 四臂的 `retriever.core.document_mrr` **必须复现上表**
  (top_k/seed/数据集与 `retr_2wiki_*` 完全一致)。**若不复现,先查 harness,本条的下游数字一律不读。**
- **预期指标 + 方向:** 主指标 `system.core.answer_match`;同时记录
  `retriever.core.{document_mrr,document_recall}`、`selector.core.{conditional_document_recall,
  document_precision}`、`system.core.{final_document_recall,cited_document_precision}`。
  预期 answer_match 随 MRR **单调上升**;真正要量的是**传导比**——MRR 涨 .41 换来 answer_match 涨多少。
- **诚实的替代结果(四种,全部有价值,不许事后挑一个说):**
  (a) **answer_match 基本持平**(跨 .41 的 MRR 跨度)⇒ **文档级检索指标是下游所需之物的劣质代理**,
      本组一年的优化方向需要重估。**这是最有价值也最难堪的结果,若出现必须照写。**
  (b) **answer_match 紧跟 MRR** ⇒ 传导成立,Hybrid RRF 的推荐从"检索更好"升级为"系统更好"。
  (c) **answer_match 跟 `document_recall`(跨度 .039)而非 MRR(跨度 .41)走** ⇒ 起作用的是
      **池覆盖不是排序**,则 `top_k`/`max_selected` 比换 retriever 更值得调。
  (d) **⚠️ 天花板效应:** 5 个 chunk × 180 词 ≈ 900 词证据,而 2Wiki 答案多为短实体
      ("Paris"、"1923")。**若四臂 answer_match 全 >0.9,该指标已饱和、无分辨力**,
      本条判定为"未能分辨",**须以 `max_selected=1` 复跑**恢复分辨率,而不是把饱和读成"传导良好"。
- **⚠️ 已知的计分伪影,与 S2/S4/S5 同族:** `answer_match` 是 exact-string 包含,
  释义/别名/单位差异一律记为失败(S5 实测同族偏差可达 56pp)。故本条数字是
  **证据传递率的下界**,四臂之间的**相对比较**可信,**绝对值不可当作真实传递率**。
- **本条不测什么:** 不测生成质量;不测 selector 优劣(selector 固定为直通);
  不改任何已记录的 retriever 结论(那些是检索指标,本条是下游指标,两层不互相覆盖)。
- 精确命令(数据集已物化于 `runs/twowiki`;`prepare` 与 `pipeline` 两步,不可省 `prepare`):
  ```
  mkdir -p logs runs && sbatch scripts/run_pipeline_eval.slurm \
    configs/experiments/pipe_2wiki_bm25.toml \
    configs/experiments/pipe_2wiki_strong-bm25.toml \
    configs/experiments/pipe_2wiki_hybrid-rrf.toml \
    configs/experiments/pipe_2wiki_decompose.toml
  ```
- Git commit:待本次改动提交后填;Seed:7;n=2000/臂。
  hybrid-rrf 与 decompose 臂需 GPU(dense 编码 / LLM 拆分),bm25 与 strong-bm25 臂纯 CPU。

**附带发现(不属本条实验,但应在组会提出):`composition.py` 的 `build_generator` 只注册了
`extractive` 一个**,而 `src/evidence_rag/generator/` 下已有 `granite.py`、`verified.py`
(G1–G5 的成果)。selector 有 4 个可选实现,generator 只有 1 个玩具级实现。
⇒ **"三模块可通过配置实时连成完整 pipeline"这条验收,目前只在玩具 generator 上成立过。**
把真实 generator 接进 `composition.py` 属 Generator 组范围,不在本条内做,
但**它是 R7 之后能否测"真实答案质量"的前置条件。**

**AFTER:** 未运行。<!-- 填:job id、四臂七项指标表、MRR 复现是否通过、传导比、
四种替代结果命中哪个、是否触发天花板效应需 max_selected=1 复跑 -->

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

### R012c — MiniCheck-FT5 臂(§3.1 预注册第三臂 + §9.8 的 A1 出样检验)[PRE-REGISTERED 2026-08-04]

> **编号:** R013/R014/R015 已被 fine-tuned Relation Builder 的三个 seed 占用(CONDITIONAL,见
> `EXPERIMENT_TRACKER.md`),R012b 为 hypothesis 形式消融,R012d 为答案串 canonical/surface 对照。
> R012c 自 2026-08-03 起已在 R012d 条目中记为"已由 MiniCheck-FT5 臂占用",本条目认领该编号。

- **状态:** BEFORE 已锁,代码已就位(TDD,测试先行)。本条目写定于本臂产出任何数字之前。
- **双重身份(与其它 R012x 的关键差别):**
  1. **§3.1 预注册三臂中从未上场的一臂。** R012 实跑只有两臂,故"任何零训练模型都不够"这一
     **族级断言至今不成立** —— 而 §3.1 写明,该区分正是 §3.8 训练路径是否启动的唯一依据。
  2. **§9.8 指定的两项 A1 出样检验之一。** A1 由 albert / DeBERTa 两臂的数据促成,故那两臂
     **不能**验证它;本臂的数字在 A1 起草时并不存在。§9.8 原文:
     **"若出样结果与本修订预期相悖,以出样结果为准。"** 即本臂被明文授予推翻 A1 的权力,
     **故本臂不得朝"与 A1 一致"的方向调整任何东西**。
- **此前跑不了的真实原因(与任务简报所述不同,记录以免重犯):** 简报称阻塞在"二分类需否定 claim
  双向探测"。A1 落地(`221c34a`)后输出空间即为二分类,该阻塞**已消失,无双向探测可写**。
  真正的阻塞是**架构**:本 checkpoint 是 `T5ForConditionalGeneration`,`load_score_fn` 的
  `AutoModelForSequenceClassification` 根本打不开它,且它**没有 id2label**,`LABEL_ORDER` 对它无意义。
  设计文档 §3.1(第 168 行)"MiniCheck 的二分类要靠否定 claim 反推 REFUTES,等于在关系层塞进一个
  未验证构造"**是 A1 之前的推理,已被 A1 作废** —— 二分类即是修订后的输出空间,无 REFUTES 需要反推,
  故本臂不含任何否定 claim 构造。该句应视为 pre-A1 记录,不再是对本臂的约束。
- **预期指标与方向(预注册,可证伪):** 主判据仍是 §9.1 的联合门 —— `gold_supports_recall ≥ .85`
  与 `twin_not_supported_accuracy ≥ .70` **同时**满足,阈值一个不动。
  1. **本臂的 `gold_supports_recall` 应显著高于同 rung 的 albert / DeBERTa。** 依据:三臂中只有它
     是**专为 document-grounded 支持判断训练**的,而 0B-2 正是该任务形状。
  2. **rung 排序预期 `qa2d` > `question_answer` > `template`。** 依据:MiniCheck 的训练 claim 是
     陈述句,rung 1 的元指称模板离其训练分布最远。**该方向与 albert 相反**(albert twin 峰值在
     rung 2、rung 3 崩到 .5312),故这是一个可被数据打脸的预测,不是事后叙述。
  3. `twin_not_supported_accuracy` 预期高但**不作选型判据** —— §9.5 已承认它在二分类下近饱和,
     退化为下限守卫。
- **事先固定的判读纪律:**
  1. **不得引入阈值(§9.5a 仍然生效)。** 本臂原生二分类,argmax 即其参考实现的 `raw_prob > .5`,
     是 checkpoint 自带的决策规则而非对着 Gate 结果调出的自由参数;任何偏离 .5 的 θ 须另开 A2。
  2. **若本臂也卡在 `gold_supports_recall`** ⇒ 族级断言成立,且 A1 的出样检验通过
     (即"A1 未制造任何通过者"得到独立证据),§3.8 可启动。
  3. **若本臂过门** ⇒ **§3.8 不得启动**,且 R012 的"零训练不够"读数被本臂推翻,
     族级结论须以本臂为准重写。这是本臂唯一可能的高价值负结果,不得因它不方便而淡化。
  3b. **"更强核查器"与"需要训练"是两个结论,设计文档已预注册该区分**(设计文档 §3.1 第 169 行:
     "若 albert 明显输给 MiniCheck,结论是'需要更强核查器'而非'需要训练'——这个区分决定 §3.6 是否启动")。
     故**即便本臂未过门**,只要它明显优于 albert / DeBERTa,§3.8 的立论就从"零训练模型不够"退为
     "这两个零训练模型不够",训练路径的依据相应减弱,须在报告中如实分开写。
  4. **若二分类在本臂上暴露出三类才能捕捉的东西**(例如弃权与反对的混同在本臂上产生了
     albert/DeBERTa 看不到的后果),须依 §9.8 如实提出 A1 可能有误,而不是绕开。
  5. **独立臂报告,不并入 R012b 三级阶梯**(§9.10a 例外条款):并入会同时变动模型与标签空间两个变量。
  6. **0B-1 不跑**(`EXTERNAL_PAIRS=none`):该层已于 2026-08-04 挂起待 A2(§9.11,tracker R020 BLOCKED)。
- **checkpoint 核实(2026-08-04,三处独立来源,均记于 `relations/minicheck.py` 模块 docstring):**
  - `https://huggingface.co/api/models/lytang/MiniCheck-Flan-T5-Large` → `architectures:
    ["T5ForConditionalGeneration"]`、`model_type: "t5"`、**无 `id2label`**。§3.1 写作 "MiniCheck-FT5
    (770M)";Flan-T5-Large 为 780M,card 自述 "best fact-checking model with size < 1B",与 §3.1 的
    "LLM-AggreFact <1B SOTA" 对应。Hub id 为 `MiniCheck-Flan-T5-Large`,"FT5" 是协议的简称。
  - 打分协议取自该 repo 的 `minicheck_web/inference.py` 与 `github.com/Liyan06/MiniCheck`:
    `predict: {doc}</s>{claim}` 单序列、`decoder_input_ids=zeros` 单步、读 token id `3`/`209`
    (源码注释 "# 3 for no support and 209 for support")、仅对该两列 softmax、index 1 = support、
    `max_model_len=2048`。论文:Tang, Laban & Durrett, EMNLP 2024(arXiv:2404.10774)。
  - **label id 实测(本地 transformers 5.10.2):`convert_tokens_to_ids(["0","1"])` = `[632, 536]`,
    两者都不是 label id**;而 `encode("0")` = `[3, 632, 1]`、`encode("1")` = `[209, 1]`。
    即 3/209 是两个 label **字符串的首 token**(T5 词表有 `▁1` 而无 `▁0`,故 "0" 拆成裸 `▁` 加数字)。
    **按 token 文本查表会静默打分到两个无关词表项且不报错** —— 故代码按 `encode(text)[0]` 推导,
    并与记录值对拍,不一致即硬失败。这是 `LABEL_ORDER` 纪律在本臂上的对应物。
- **前置(2026-08-04 更正):** ~~pair 文件须是 A1 之后重建的三份…必须显式重建。~~
  **本条说过头了,且照它执行会造成不可逆损失。** 实测:`task_report` **根本不读 gold 列**
  (A1 后两个指标都对 SUPPORTS 定义,gold 仅作 arity 守卫),故 pre-A1 与 post-A1 的 pair 文件
  **产出逐字相同的报告** —— 重建对正确性无任何收益。而原命令块写的是**原地重建**,
  那会**覆盖 R012 与 R012b rung 1 实际消费过的历史输入**,而 `/data/` 被 gitignore,覆盖即永久丢失。

  **改为:直接复用现有三份 pair 文件。** 除"零收益且有损失"外还有一条更强的理由 ——
  **用与 albert / DeBERTa 完全相同的输入文件,是三臂可比性的最强保证**;而 `export_task_probe`
  自那三轮之后已有改动(新增 `--gold-answer`、fixture 语义更正),重建就得额外证明 premise 与
  hypothesis 未变。复用则无需证明。若仍要 post-A1 版本的 artifact,**必须写到新路径**,不得覆盖。
- **方向冒烟(2026-08-04,本地真实权重,4 对手工样本):** `4/4` 方向正确,
  支持对 `P(SUPPORTS)` = .9748 / .9733,孪生对 = .0122 / .0057,且用的是 **rung 1 冻结模板**。
  `model_version` = `lytang/MiniCheck-Flan-T5-Large@f4f447f5877fc162`(权重指纹;
  **bp1 上应复现同一指纹,不同即说明下到了不同的 checkpoint**);tokenizer 走 fast 路径。
  **纪律声明:这不是结果。** 4 对样本由我手工构造、不属于 0B-2 探针,样本量与构造方式都不足以
  对任何门指标发言;它只排除"label id 取反"这一类**结构性**错误(该错误会让一切照常运行且数字合理)。
  本条目的**预注册方向写定于该冒烟产出之前**,顺序保留于此以备核。真实读数一律以 sbatch 的三个
  rung 为准。
- **命令:**
  ```
  # 登录节点(有网,一次性)
  export HF_HOME=/user/work/$USER/hf_cache
  hf download lytang/MiniCheck-Flan-T5-Large          # ~3.1GB

  # pair 文件不重建 —— 复用 R012b 三个 rung 用过的那三份(见更正后的「前置」),
  # 三份均已在 bp1 上:task_pairs.jsonl / task_pairs_qa.jsonl / task_pairs_qa2d.jsonl

  # 提交前先验权重指纹,免得下错 checkpoint 要用三个 job 的代价才发现
  export PYTHONPATH=src
  python -c "
from transformers import AutoModelForSeq2SeqLM
from evidence_rag.cli.gate0b import _weight_buffers
from evidence_rag.relations.predictor import fingerprinted_version, weight_fingerprint
mid='lytang/MiniCheck-Flan-T5-Large'
got=fingerprinted_version(mid, weight_fingerprint(_weight_buffers(AutoModelForSeq2SeqLM.from_pretrained(mid))))
print(got); assert got.endswith('@f4f447f5877fc162'), 'checkpoint 与核实时不同,先查再跑'"

  # 三个 rung,各自独立 job(0B-1 挂起,故 EXTERNAL_PAIRS=none)
  ARM="lytang/MiniCheck-Flan-T5-Large"
  for RUNG in template:task_pairs qa:task_pairs_qa qa2d:task_pairs_qa2d; do
    sbatch scripts/run_gate0b.slurm "data/gate0b/${RUNG#*:}.jsonl" none \
      "results/gate0b/sweep-minicheck-${RUNG%%:*}.json" "$ARM" \
      "results/gate0b/dump-minicheck-${RUNG%%:*}.jsonl"
  done
  ```
  报告 headline 时必须同时写明该 pair 文件的 `hypothesis_form`(exporter 回显该字段)。
- **环境前置(2026-08-04 实测,四步缺一不可;干净机器上重跑 R012c 必须照做):**
  本 checkpoint 的**主 revision 只有 `pytorch_model.bin`,没有 safetensors**;safetensors 仅存在于
  一个**转换 PR 的 ref** 下,而定位该 ref 需要访问 Hub。计算节点 HF-offline ⇒ 直接失败。
  1. `hf download lytang/MiniCheck-Flan-T5-Large`(登录节点,~3.1GB)。此时
     `snapshots/<main-sha>/` 只有 `.bin`。
  2. **在有网的登录节点**用 `use_safetensors=True` 加载一次。transformers 会经 `auto_conversion`
     取回转换 PR 的 safetensors,落到**另一个 snapshot 目录**(本次为 `c3f6482d…`)。
  3. **把该 safetensors 链进主 snapshot**(`refs/main` 指向的那个,本次为 `96eafd01…`):
     `ln -sfn ../../blobs/<safetensors-blob> snapshots/<main-sha>/model.safetensors`
  4. **删掉负缓存标记** `.no_exist/<main-sha>/model.safetensors`。第 2 步失败时 huggingface_hub
     记下了"主 revision 无此文件"的 0 字节标记,该标记会**短路查询、根本不看目录**,故第 3 步单独无效。
     **只删这一个**;`custom_generate/generate.py` 与 `model.safetensors.index.json` 两个标记要留着 ——
     删了它们,离线模式下反而会去请求 Hub 而触发新的失败。
  5. **验收(必须做,且必须带离线开关):**
     `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "from evidence_rag.cli.gate0b import
     load_score_fn; print(load_score_fn('lytang/MiniCheck-Flan-T5-Large')[2])"` ——
     调作业真正用的那个函数、在作业真正的环境条件下。输出须为 `…@f4f447f5877fc162`。
  **三步各自的失败长得完全不同**(CVE 报错 / `OfflineModeIsEnabled` on `auto_conversion` /
  同样的 `OfflineModeIsEnabled` 但文件明明在),故必须整套记录而不是只记结论。
  **诊断过程中我两次判错根因**(先判"两种格式选错了",后判"主分支无 safetensors 故须升 torch"),
  两次都是靠实测推翻的;若当时按第二次判断升了 torch,会白改环境且引入一个跨臂变量。
- **地雷(2026-08-04,提交后才发现,已修):`torch < 2.6` 下必须显式要求 safetensors。**
  bp1 上 `hf download lytang/MiniCheck-Flan-T5-Large` 抓下 17 个文件含 `pytorch_model.bin`,
  随后 `from_pretrained` **解析到了 `.bin`**,在 `check_torch_load_is_safe()` 上抛
  `CVE-2025-32434`(torch 需 ≥ 2.6)。**即使 `model.safetensors` 随后也落了盘,重跑仍失败** ——
  说明这不是环境问题,是代码没有指定格式。
  - **实据(非推断):** 计算节点上的 job `18264967` / `18264968` 各跑 **00:01:30 后 FAILED**,
    `logs/gate0b-18264967.out` 的 traceback 终止于
    `model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id)` ——
    **即该行的修复前版本,不带 `use_safetensors=True`** —— 再经 `check_torch_load_is_safe()`
    抛同一个 CVE。登录节点与计算节点在同一行、以同一原因失败,修复正打在该行上。
    (`18264969` 为同批第三个,提交后即撤。)
  - **修法:** `cli/gate0b.py` 两处 `from_pretrained` 均加 `use_safetensors=True`(含测试)。
    这使安全路径成为唯一路径,且把"这个 repo 没有 safetensors"变成错误信息真正说的那句话,
    而不是一条关于 CVE 的、看不出真因的消息。**失败时不得靠放宽这个参数来"修"**。
  - **本约束今日第二次咬人**(第一次:QA2D 的 `question_converter-3b`,同样死在 torch<2.6,
    见 R012b 的 checkpoint 选型)。凡 `hf download` 抓到 `pytorch_model.bin` 的 repo 都要过这一关。
  - ~~**跨模块的假设冲突,须知晓:** …哪一个描述的是真实部署环境,须与 G 模块对齐后写死一处。~~
    **[同日更正 —— 本条为事实性错误。]** 两个模块**跑在不同账号的不同 venv 里**,不存在冲突:
    本轮 verifier triage 的产物记于 `/user/work/**ri25947**/IBM_Granite_Project/`(见本文件
    2026-07-28 那条 AFTER),而 Gate 0B 跑在 `/user/work/**uz25020**/venv`。
    `generator/nli.py:230` 那句"the deployment already has torch >= 2.6"指的是前者,它是对的。
  - **"升 torch 会打断 safetensors 加载"这一说法亦无实证支持,反被同一条记录否证:**
    ri25947 的 venv 由 2.5.1 升至 `2.6.0+cu124` 时,记录明写"**granite 两臂 safetensors 不受影响**"。
    `pyproject.toml` 钉的是 `torch>=2,<3`,**并不禁止 2.6**。故"跟 G 对齐(升 torch)"技术上可行。
  - **但本轮不升,理由是实验控制而非技术:** R012 / R012b / R012d **全部在当前 torch 下跑完**。
    此刻升级会使 MiniCheck 臂运行在与它要对比的三臂**不同的 torch 上**,给一组极力控制变量的
    对比引入一个无谓的新变量。而 `use_safetensors=True` 已经解决了真实问题且零成本。
    **仅当日后确需加载只有 `.bin` 的 checkpoint 时才升,届时须与 G 模块同步版本并重跑受影响的臂。**
- **预检失效的教训:** 本条目的指纹预检**本应在花 job 之前拦住它**,实际是预检失败后三个 job
  仍被提交(18264967/68/69)。预检只有在"失败即停"被执行时才有价值。
- **commit:** _待填(代码随本轮提交:`relations/minicheck.py` + `cli/gate0b.py` 双注册表分发 + 测试)_
- **AFTER(实测,2026-08-05):**

**Jobs:** `18269630` / `18269631` / `18269632`,均 COMPLETED(00:03:31 / 00:02:25 / 00:01:50)。
按提交顺序对应 template / qa / qa2d(精确对应可从各自 `.out` 的 `--task-pairs` 读回)。
**Raw:** `results/gate0b/sweep-minicheck-{template,qa,qa2d}.json` + 同名 dump。
读数经 `cli/recompute_binary` 统一到二分类口径后与另两臂并列(旧 sweep 为 pre-A1 字段名)。

**九格(gold_supports ≥ .85 ∧ twin_binary ≥ .70):**

| rung | albert | DeBERTa | **MiniCheck** |
|---|---|---|---|
| `template` | .1916 / .9980 | **.7942** / .8689 | .5727 / .9463 |
| `question_answer` | .3635 / .9871 | .6651 / .9008 | .5883 / .9446 |
| `qa2d` | .5360 / .9586 | .5870 / .9222 | .5360 / .9375 |

**九格无一通过联合门。** 唯一约束仍是 `gold_supports_recall`;twin 九格介于 .8689–.9980,全过。

**两条预注册预测均被证伪(这正是它们写下来的目的):**

1. ~~"MiniCheck 的 `gold_supports_recall` 应显著高于同 rung 另两臂"~~ —— **三个 rung 全部不成立**,
   它反而**低于 DeBERTa**(.5727 vs .7942、.5883 vs .6651、.5360 vs .5870)。
2. ~~"rung 排序应为 `qa2d` > `question_answer` > `template`"~~ —— 实测 **qa > template > qa2d**。

**触发的判读规则:第三条。** 九格无 PASS(排除规则 1);MiniCheck **不**优于另两臂(排除规则 2 与 3b)。
⇒ **族级断言成立**:"任何零训练模型都不够"现由三臂支撑,其中包含**唯一为 document-grounded
verification 专训、且本项目自己实测最优**的那一臂。⇒ **§9.8 的 A1 出样检验通过**:本臂由 A1 未参与设计
的数据产生,其形状与 A1 预期一致(twin 近饱和、约束落在 gold 上),**A1 未制造任何通过者**得到独立证据。
⇒ **§3.8 训练路径解锁。**

**本轮最有价值的发现(非预注册,故只作观察不作主张):MiniCheck 对 hypothesis 形式近乎免疫。**

| 臂 | 三 rung 的 gold_supports | 极差 |
|---|---|---:|
| albert | .1916 / .3635 / .5360 | **34.4pp** |
| DeBERTa | .7942 / .6651 / .5870 | **20.7pp** |
| **MiniCheck** | .5727 / .5883 / .5360 | **5.2pp** |

**唯一为该任务形状专训的臂,恰是最不受 claim 措辞扰动的那个。** 这从一个独立角度印证 R012b 的结论:
另两臂的大幅摆动是**形式伪影**,而形式鲁棒的模型不出现这种摆动。

**跨任务倒挂,须与 S7 一并报告:** 本项目 G 模块的 verifier triage 中 **MiniCheck 胜 deberta-large**
(2Wiki atomic .917 vs .817,见本文件 2026-07-28 条目);**在 0B-2 上倒过来了**。
这是"一个任务上测出的排序不能外推到另一任务形状"的第二个实例
(第一个:VitaminC 上 macro-F1 .922 的臂在探针上 .19)。

**已排除的巧合:** albert 与 MiniCheck 在 `qa2d` 上同为 `.5360`。实测两者均命中 **789/1472**,
但**判对的 query 集合不同**(交集 626),故为巧合而非同一批预测泄漏。

**本条目不建立的东西:** 三臂均为零训练现成 checkpoint,**不涉及 §3.8 训练后的能力**;
0B-1 仍挂起(§9.11),故 Gate 0B 的外部效度层在本轮无证据。
- **commit:** `69a0c04`(代码)/ `53a31cb`(环境前置)/ 本条 AFTER 随后提交

---

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

**AFTER(第一轮,job `18260560` 生成 + `18260561` 评分):**

- 三轴表:baseline 0.932 / 0.273 / 0.613 / 0.647;verify-only 0.641 / 0.214 / 0.760 / 0.852;
  verify-annotate 0.641 / 0.224 / **0.830** / 0.788。配对:精度 vs verify-only **+0.070(p=0.0006)**、
  vs baseline +0.142(p<0.0001);召回 −0.065(有意代价);correctness +0.010(p=0.0019);coverage 0.000。
- 声明引用存活 **247/429 = 0.576**;回退扫描救回 **47/429 = 0.110**;标注率 33/380 = 0.087。
- 预注册判据:精度判据**通过且超预期**;coverage 判据**触发**,但该判据写的是 0.552 的旧 verify-only,
  部分作答早已把它抬到 0.641,两臂现在同条件弃答,标注在构造上无法移动 coverage。两种读法均记录。
- **此轮数字有两处已知限定**,详见 `docs/generator/g5-verify-annotate.md`。

**重跑溯源(第二轮)—— 数字变动的原因是审计驱动的缺陷修复,不是调参:**

这一点必须可追溯,故单列。第二轮相对第一轮的差异**全部**来自以下三项,每一项都能**脱离任何指标**陈述缺陷,
且均由**人工盲审**确立(20/72 抽样:假否决 0.700、另有 0.100 本该标注),而非因为"改了数字会好看":

1. **专有名词检测改用 `SpacyEntityExtractor`**(`verification` extra 里早已存在、从未启用)。
   缺陷陈述:大写启发式把句首普通名词当作专有名词并据此否决 —— 实测触发词包括
   `name:some`、`name:season`、`name:small`、`name:substitutions`、`name:unemployment`。
2. **只有"真冲突"才丢弃**(`EntityMismatch.evidence_values` 非空,即证据确实携带同角色的竞争取值)。
   缺陷陈述:证据从未提及的实体是**缺失**而非矛盾;annotate-not-delete 下缺失不得摧毁内容。
3. **verify-only 补记逐句引用映射**(脚本侧 `RecordingRepairer`,不改 src)。
   缺陷陈述:两臂此前在**不同引用口径**下计分 —— verify-only 退回扁平口径(1.92 引用/句)而
   verify-annotate 用精确映射(1.13),ALCE 的冗余消融因此系统性压低前者。**预期 +0.070 与 +0.142 都会缩小**,
   修正后的数字才是应报的头条。

**未改动、且经核实本就正确的一项**:成员资格语义(`_is_consistent` 用claim取值在证据同角色**集合**中查成员),
"1954 ∈ {1954, 1974}" 本就通过。审计暴露的残留是**形态/别名变体**("west germany" vs "west german")与
**精度失配**(证据陈述了别的日期但没有该日),二者按你的要求在重跑后单独报告其占比。

**AFTER(第二轮,job `18267966` 生成 + `18268709` 评分):**

- 两臂同口径后的三轴表:baseline 0.932 / 0.273 / 0.613 / 0.647;verify-only 0.633 / 0.197 / **0.871** / 0.871;
  verify-annotate 0.633 / 0.196 / **0.890** / 0.868。
- **verify-annotate 与 verify-only 在每一根轴上统计无差异**(精度 +0.024 p=0.27、召回 +0.001 p=0.98、
  correctness −0.001 p=0.89、coverage 相同)。第一轮的 +0.070 **基本是评分伪影**:扁平口径通过 ALCE
  冗余消融替 verify-only 承担了它没附到该句的引用,拉平后它从 0.760 升到 0.871。
- 两个验证臂相对 baseline:精度 **+0.173 / +0.188**、召回 **+0.13**(均 p<0.0001),
  代价 coverage −0.30、correctness −0.077。**这个权衡才是结果,delete-vs-annotate 不是。**
- **机制**:82 条声明被标注,**只有 15 条进入答案** —— 零已验证即整例弃答,标注随之丢弃,
  作用面封顶在 4.8% 的保留句。
- 实体丢弃**未变罕见**:66/439 = 15.0%(修复前 14.7%),但构成全变为真实 NER 类型;
  78 个值冲突中别名残留 9(11.5%)、词面不相交 69(88.5%)。**不相交是词面代理而非判决**,
  0.700 假否决率来自旧抽取器样本群,需重新盲审才能声称已下降。
- 过程中修掉两个**我自己的** bug,均记录在案:`declared_indices` 越界读到下一句引用(污染声明引用存活率
  且会改变附上的引用);评分器 routing↔句子匹配方向反了(227/269 已引用句被判无引用,导致 0.105 的假塌陷)。

---

## G6 — 解除弃答封顶(契约变更,团队已批准)

**状态:** 代码就绪。**这是经批准的契约变更,故 BEFORE 在运行前写入并提交。**
**本轮数字变动的原因是契约变更,不是调参** —— 溯源需可追。

**BEFORE(预注册):**

- 背景:G5 第二轮显示 annotate 相对 delete 全轴为空,但**机制是它从来没有作用空间** ——
  82 条标注只有 15 条进入答案,因为零已验证就整例弃答并丢掉标注。该封顶**只**为满足
  `GenerationResult`「非空答案必须≥1 引用」而存在。
- **契约互换(非删除)**:旧不变式保证「每个答案都有据」,新不变式保证
  **「每个无据的句子都被标注」**,并以 validator 强制(未标注的无引用句必须被拒,与从前拒绝无引用答案一样)。
  空答案路径保留 —— **「弃答」与「作答但无一验证」是不同结果,不得合并**。
- 四臂:baseline / verify-only(删除+弃答)/ **verify-annotate-capped**(旧封顶,消融项)/
  **verify-annotate-open**(解封,本轮主体)。保留 capped 臂才能把效果**归因**到契约解除本身。
  四臂**同一作业内**重跑 —— 跨运行比较在本项目已造成两次混杂,一次作业的成本远低于该风险。
- **预期方向:** coverage 大幅上升趋近 baseline 的 0.932(自 0.633);correctness 上升(保留的未验证内容
  有时含 gold 答案);**已引用句**的引用精度维持在 0.87–0.89(标注句无引用,不进精度计算);
  **引用召回大幅下降**(无引用句留在召回分母)—— 这是**有意且已认领的代价**。
- **失败判据(预注册):** 若**已引用句**的引用精度向 baseline 退化(说明未验证内容漏进了已引用池,
  或标注未被正确排除),**或** coverage 未显著高于 0.633,则本次解除失败。
- 纪律:**一轮只改一件事** —— 实体冲突路由本轮**不动**;不调提示词/阈值;预注册后冻结。
- 报告须含**每臂答案句构成**(已验证并引用 / 标注未验证 / 丢弃)。理由:单看引用召回会**惩罚本方法主张的行为** ——
  一个给每句都编造引用的系统召回反而更高,而诚实标注不可验证内容的系统更低。
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
- 2026-08-05:**R5 瓶颈 (a) 的修法 —— BM25 每 query 常数项外提**。R5 读码定位的四处
  与 chunk/query 无关却在内层重算的量,全部提到 `_set_chunks` 或查询循环之外:每 chunk 的
  `Counter(tokens)`(现为 `term_frequencies`,建索引时算一次)、query 的 analyzer 调用
  (原来**每个 chunk 重算一次**)、每个 query term 的 IDF、每个 chunk 的长度归一化。
  **纯常数项优化,不动渐近复杂度**(倒排索引=瓶颈 (b),仍未做)。
  - **输出逐位不变,有测试守卫:** `tests/retriever/test_bm25_scoring_equivalence.py` 把改写前的
    打分循环逐字转录为参考实现,用 `==` 而非 `approx` 断言(算术分组刻意保持原样,任何漂移
    都说明分组变了)。另钉住三处易被静默改坏的行为:query 重复词仍按出现次数各计一次
    (去重成 set 不会被任何聚合指标发现)、缓存的 Counter 不因未命中词增长、StrongBM25 继承同一引擎。
    retriever + architecture + evaluation 三套 **322 passed**,零回归。
  - **本地合成语料实测(chunk_size=180/overlap=30,20k 词 Zipf 词表 → 124 distinct/chunk,
    与真实散文相当;20 query 取均值):**

    | chunks | old ms/query | new ms/query | 加速 | `term_frequencies` 内存 |
    |---|---|---|---|---|
    | 2258 | 25.50 | 3.04 | **8.4×** | 6.5 MB |
    | 6725 | 71.89 | 10.31 | **7.0×** | 19.6 MB |
    | 11618 | 129.09 | 21.50 | **6.0×** | 33.8 MB |

  - **代价必须同时报:缓存内存随语料线性增长**,约 **2.9 KB/chunk**。R5 实测 8778 chunk 时
    peak RSS 161.5 MB,按此比例约 **+25 MB(+16%)**——当前规模划算;但外推到 100 万 chunk
    即 **~2.9 GB 仅这一项**,故 (a) 缓解不了 enterprise scale,只是把常数压下来。**真正的
    渐近问题仍须靠 (b) 倒排索引。**
  - **⚠️ 加速比随语料增大而下降(8.4× → 6.0×)**:Counter 重建的占比被"扫全部 chunk"的固有
    开销稀释——这恰是 (b) 才能治的部分,与 R5 的诊断一致。
  - **限制:** 本地合成语料,非 SciFact;**权威 before/after 须在 bp1 上用同一个
    `scripts/retriever_scaling.py` 对真语料重跑**,与 R5 曲线直接对比。在此之前上表只作量级参考,
    不得写入 results-summary。
  - **⚠️ 事后订正(2026-08-06,两轮,后一轮推翻前一轮;三段原文全部保留存底)。**
    **第一轮(据 R6,跨节点)曾在此写:"加速比被本条低估,真语料 8.29–10.17×,高于本地的
    6.0–8.4×"。该订正本身已被 R6b 推翻,作废。**
    **第二轮(据 R6b,同 job 同节点配对,以此为准):**
    (1) **加速比是本条略微高估,不是低估:** 同节点实测 **5.27–5.40×**,**低于**本地的 6.0–8.4×。
    R6 的 8–10× 是跨节点假象。本地合成语料给出的量级大体正确(同一数量级、偏高约 15–55%),
    **作为量级参考是称职的,作为定值则不行** —— 这正是当初把它限制在本节、不写进
    results-summary 的理由,该纪律事后被证明是对的。
    (2) **"加速比随语料增大而下降(8.4× → 6.0×)"同样不成立:** 同节点下加速比**是平的**
    (5.27–5.40,±1.2%)。本地看到的下降与 R6 看到的下降,**是同一个跨运行噪声的两次显形**。
    (3) **内存代价一条确定作废:** 上面"~2.9 KB/chunk、100 万 chunk 即 ~2.9 GB"来自本地
    `sys.getsizeof` 估算,**真语料 peak RSS 两次(R6、R6b)均未见增长**。最可能是**量错了工具**
    ——peak RSS 被建索引阶段的临时分配主导,看不到稳态增量。**内存代价现列为未测**,
    要给须换工具(稳态 `tracemalloc`)。详见 R6b 的 AFTER。
