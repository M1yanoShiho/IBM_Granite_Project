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

**AFTER(读数 2026-08-10;回填 2026-08-13 —— 运行早已完成而本条一直停在"未运行",
见下方"回填延迟"):**

**命中预注册的替代结果 (a):单调,无内部最优,且没有任何有限权重越过 strong-bm25。**

MRR,配对随机化,基线分别为 strong-bm25 与 `decompose-orig`(w=1):

| 数据集 | w=1 | w=2 | w=3 | w=5 | strong-bm25 |
|---|---|---|---|---|---|
| SciFact(n=300) | 0.5824 | 0.5873 | 0.5949 | 0.5986 | **0.6105** |
| 2Wiki(n=2000) | 0.7155 | 0.7912 | 0.8186 | 0.8538 | **0.9580** |

**vs strong-bm25(预注册的判定标尺):**

| 数据集 | w=2 | w=3 | w=5 |
|---|---|---|---|
| SciFact | −0.0232 p=0.1301 | −0.0156 p=0.2905 | −0.0120 p=0.3742 |
| 2Wiki | −0.1668 **p=0.0000** | −0.1394 **p=0.0000** | −0.1042 **p=0.0000** |

**⇒ 全部低于 .6105 / .9580,一个都没越过。** 按 R3 定下的标尺,诚实结论是**加权不能让分解
在这两个数据集上变得有用**;2Wiki 上更是每个权重都显著更差。

**vs w=1(问"加权到底做了事没有"):** SciFact MRR w3 +0.0125 **p=0.0027**、w5 +0.0161 **p=0.0239**
(w2 +0.0049 p=0.1477 不显著);R@10 w3 +0.0283 **p=0.0111**、w5 +0.0390 **p=0.0008**。
2Wiki MRR w2/w3/w5 = +0.0757 / +0.1031 / +0.1383,**均 p=0.0000**。
⇒ **另一个预注册结局(三点全与 w=1 不可区分 ⇒ 稀释解释错误)未发生:加权确实有效。**

**两条同时成立才是本条的完整结论:稀释解释站得住(加权显著有效),而它买到的东西全部可以由
"更靠近基础检索器"解释干净** —— 因为 RRF 下 `w → ∞` 就是 strong-bm25,单调上升近乎结构性。
SciFact 的 MRR 缺口回收:w1 46% → w2 55% → w3 70% → w5 77%,总召回全程持平
(w5 −0.0042 p=0.6781)⇒ 动的是排序不是候选池,与 R3 Step 1 的诊断一致。

**⚠️ 回填延迟(记录以免重演):** 本条的 sweep 在 2026-08-10 就已跑完并被写进
`docs/retriever/report/retriever-progress.md` 的 R4,但**台账的 AFTER 一直停在"未运行"**,
直到 2026-08-13 的对齐检查才发现。台账是预注册的权威记录,报告是对外读物;**结果只回填其中一处,
等于让预注册与结果脱钩** —— 而预注册的全部价值就在于事后能被对照。此后:出结果先回填台账,
再写报告。

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

**⚠️ 提交前的设计变更(2026-08-09,写于**跑之前、读任何数字之前**,与 R1 同类披露):**

- **变更内容:** GPU 请求从 `--gres=gpu:rtx_3090:1` 放宽为 `--gres=gpu:1`(任意型号),
  四臂仍在**同一个 job、同一个节点**内跑。作业 18324520(限定 3090)已 `scancel`,
  改提 18325274。
- **原因(实测,非猜测):** `sinfo -p gpu -o "%n %G %t"` 显示**全集群只有一台 rtx_3090 节点**
  (`bp1-gpu030`,8 卡),其余为 rtx_2080 / V100 / 单台 a100。限定 3090 等于排单节点资源,
  `squeue --start` 预计 6+ 小时,且该瓶颈**每次都会重现**。`sinfo -R` 确认无节点排空。
- **代价:自检强度下降,须如实降级。** 原条目写"四臂 MRR 必须复现已记录矩阵,否则先查 harness"。
  已记录矩阵跑在 rtx_3090 上,故:
  - **bm25 / strong-bm25 两臂仍必须精确复现 `.9434` / `.9580`** —— 它们是纯 CPU、与显卡无关,
    **harness 的正确性由这两臂单独担保**,自检目的不受影响。
  - **hybrid-rrf / decompose 两臂若偏离 `.9828` / `.5702`,先归因于 GPU 架构,不得直接判为 harness 故障**;
    偏离幅度须如实记录。风险不对称:hybrid 的 dense 编码只在接近的排序上翻转,量级极小;
    **decompose 风险大得多——贪心解码对浮点差异是混沌的**,一个 token 翻转就换掉整个子查询。
    若 decompose 偏离明显,**不得解释为方法差异**,只能记为"换硬件后不可逐位复现"。
- **为什么这个取舍是对的:** R7 问的是**四臂之间的传导关系**,要的是**臂间可比**——
  四臂同 job 同节点满足了这一点。与三天前矩阵的逐位一致是**加分项而非前提**,
  而它在两个 CPU 臂上仍然保留。

**附带发现 —— 已于 2026-08-09 被 Generator 组解决,原文保留以记录时序:**

~~本条预注册时(`b82a4a2`),`composition.py` 的 `build_generator` **只注册了 `extractive` 一个**,
而 `src/evidence_rag/generator/` 下已有 `granite.py`、`verified.py`。selector 有 4 个可选实现,
generator 只有 1 个玩具级实现 ⇒ "三模块可通过配置实时连成完整 pipeline"这条验收,
当时只在玩具 generator 上成立过。~~

**现状(`229388f` 起):`build_generator` 已支持 `extractive` / `granite` / `verify-annotate`**
(commit 由 Generator 组提交,含 `NLIModel`、`VerifyAnnotateGenerator` 注入)。**前置条件已清除。**

对本条的影响:**本轮仍按预注册跑 `extractive`,配置一字未改,结论范围不变**(测的是证据传递)。
但"只能测证据传递、测不了答案质量"从**约束**变成了**选择** ⇒ **R8 现在可跑**:同样四个检索臂,
仅把 `generator` 换成 `granite`,即可回答"检索提升能否变成更好的答案"这一完整问题。
R7 与 R8 的差别只有 generator 一项,故两者可直接配对比较,**证据传递与答案质量的差值本身就是结果**。

**AFTER(2026-08-09,job 18325687,COMPLETED,elapsed 01:36:00,`bp1-gpu002`(rtx_2080),
四臂各 2000/2000;前两臂实由 job 18325274 产出,见下"三次失败"):**

| 臂 | MRR | Recall | selRecall | selPrec | sysRecall | **answer** | citePrec |
|---|---|---|---|---|---|---|---|
| decompose | 0.5733 | 0.7598 | 0.6626 | 0.2351 | 0.4756 | **0.3540** | 0.2351 |
| bm25 | 0.9434 | 0.7621 | 0.8947 | 0.3319 | 0.6653 | **0.5080** | 0.3319 |
| strong-bm25 | 0.9580 | 0.7678 | 0.9015 | 0.3485 | 0.6766 | **0.5185** | 0.3485 |
| hybrid-rrf | 0.9826 | 0.7995 | 0.9036 | 0.3610 | 0.7073 | **0.5610** | 0.3610 |

- **harness 自检通过,且两个 GPU 臂的偏差与修订后预注册的预测逐条吻合:**
  bm25 **Δ0.0000**、strong-bm25 **Δ0.0000**(纯 CPU,精确复现);
  hybrid-rrf **Δ−0.0002**(dense 编码,"量级极小");decompose **Δ+0.0031**(贪心解码,大 15 倍,
  "风险大得多")。**跑之前写下的相对风险判断被数据印证**,故两处偏差归因于 GPU 架构成立,
  不是 harness 故障。四臂 `dataset_signature` 同为 `69dce7a2d909808d`,与 R1/R2 同一份 2Wiki。
- **预注册结果 (a) 被否 —— 这是本条最重要的一句。** answer_match 从 **0.3540 到 0.5610**
  (跨度 **0.207**),**远非持平**。"文档级检索指标是下游所需之物的劣质代理、本组一年的优化
  方向需要重估"这个最难堪的可能性**没有发生**。检索的改善确实到得了下游。
- **预注册结果 (c) 被否,(b) 成立,而且是被一个池子匹配的天然对照证实的:**
  **decompose vs bm25** —— Recall 差 **−0.0024,配对 p=0.3797(不显著,池子统计上不可区分)**;
  MRR 差 **−0.3701 p=0.0000**;answer 差 **−0.1540 p=0.0000**。
  ⇒ **在候选池不变的前提下,单靠排序崩塌就能让下游证据传递掉 15.4 个百分点。
  传导走的是排序通道,不是覆盖通道。**
- **机制被定位到具体一步:top-50 → top-5 的收窄是按排名门控的。**

  | 臂 | Recall(50) | sysRecall(5) | **存活率** | answer/sysRecall |
  |---|---|---|---|---|
  | decompose | 0.7598 | 0.4756 | **0.626** | 0.744 |
  | bm25 | 0.7621 | 0.6653 | **0.873** | 0.764 |
  | strong-bm25 | 0.7678 | 0.6766 | **0.881** | 0.766 |
  | hybrid-rrf | 0.7995 | 0.7073 | **0.885** | 0.793 |

  四臂拉开差距的是**存活率**(0.626 vs 0.873–0.885),而最后一列相对稳定(0.744–0.793)。
- **⚠️ 一个反面结果,同样重要:小幅排序改善测不出下游收益。**
  strong-bm25 vs bm25 上游 MRR **+0.0146 p=0.0000(显著)**,下游 answer **+0.0105 p=0.0513
  ——未达 .05**。与 MengW7 矩阵里"StrongBM25 vs BM25 不是可靠的胜利"在下游一致。
  ⇒ **传导为真,但需要足够大的上游改动才看得见;不能拿小幅 MRR 提升去许诺系统收益。**
- **可用于决策的结论:hybrid-rrf 是唯一在系统层面显著优于 strong-bm25 的臂**
  (answer **+0.0425 p=0.0000**)。此前"Hybrid (RRF) 最强"只是**检索指标**上的结论,
  现在**在下游也成立**,推荐的分量因此不同。
- **预注册的天花板情形 (d) 未发生。** answer_match 落在 0.354–0.561,远低于 0.9 阈值,
  指标有充分分辨力,无须以 `max_selected=1` 复跑。

**限制(须与结论同时声明):**

1. **hybrid 那一段是混淆的,不可用于归因。** hybrid vs strong-bm25 的 Recall(+0.0318 p=0)
   与 MRR(+0.0246 p=0)**同时**显著改善,故其 +0.0425 的下游收益**无法拆分到单一通道**。
   **本条唯一干净的因果对照是 decompose vs bm25**(Recall 不显著),其余只作趋势。
2. **⚠️ 汇总脚本打印的 `transfer: 0.506` 有误导性,是本工具的缺陷。** 它用首尾两点算跨度比,
   把强非线性抹平了。分段实测:decompose→bm25 **0.416**、bm25→strong **0.719**、
   strong→hybrid **1.728** —— **相差逾 4 倍,根本不是常数**。该行不得单独引用。
3. **`answer/sysRecall` 是聚合比值,不是条件概率。** 它提示"gold 文档进了最终 5 条之后,
   答案串仍约 21–26% 不在其中",但要坐实须走 per-case 条件统计。若成立,
   **这部分损失与检索器基本无关,属 chunk 粒度问题**,正是 chunking sweep 的靶子。
4. 本条测的是**证据传递**,不是答案质量(generator 为 `extractive`,见上文对该指标的界定)。
   R8 用 `granite` 重跑同四臂即可给出答案质量,两者差值即为落差。
5. `answer_match` 是 exact-string 包含,与 S2/S4/S5 同族伪影,**绝对值是下界,臂间相对比较可信**。
   **⚠️ 该“下界”表述已被 R16 作废(2026-08-14):实测为双向偏差 —— 答案为 `yes` 时是下界,为 `no` 时是**上界**(2Wiki 上 52/54 判为答对)。臂间相对比较仍可信,故本条结论不变。**

**三次失败与环境坑(如实记录,均未产生任何结果数据):**

- job **18324520**:限定 `gpu:rtx_3090:1`,`squeue --start` 预计 6+ 小时。`sinfo` 查明
  **全集群仅一台 3090 节点**,遂 `scancel` 并放宽为 `gpu:1`(设计变更已记于上文 BEFORE)。
- job **18325274**:跑完两个 CPU 臂后在 hybrid 臂崩于
  `No module named 'sentence_transformers'`。**我最初误判为 `module load` 的 Python 版本不符
  并据此改了脚本(`af9db60`),那不是根因** —— venv 实为 `python3.11`,加载哪个 module
  都不影响。改脚本本身无害(与 `run_retriever_eval.slurm` 对齐),但其 commit message
  的归因是错的,**在此更正**。
- job **18325479**:换 3.12.3 后**报同一个错**,由此证伪了上述误判。
- **真正的根因:jp25459 的 venv 从未安装 `sentence-transformers`。** 因为本账号此前
  只跑过 sparse + LLM 的臂(R1–R3),而带 `granite-dense`/`hybrid` 的矩阵是
  **MengW7 在他自己账号下**跑的。**教训:"队友跑过"不等于"我的环境能跑",每个账号的 venv 独立。**
  修法(登录节点):`pip install "sentence-transformers>=3,<6"` —— **刻意不用
  `pip install -e ".[granite]"`**,因为该 extra 带 `transformers<5` 而环境里是 5.14.1
  (队友的训练代码需要),装 extra 会把它降级。`--dry-run` 已确认只新增 5 个包、不降级任何包。
  另需在登录节点 `hf download ibm-granite/granite-embedding-english-r2`(计算节点离线)。
- **`pyproject.toml` 声明 `transformers>=4.45,<5` 而实际环境是 5.14.1,该约束已过时**,
  值得在组会提出(不属本条范围)。

- raw:`results/r7-2wiki-{bm25,strong-bm25,hybrid-rrf,decompose}-per-case.json`
  (已 `git add -f` 拉回)。上表已从 raw 重新聚合复核,逐位一致。
  **⚠️ 已剥离每条的 `trace` 字段后入库:原始四份合计 342 MB(trace 占 97.6%,含全部候选与
  证据原文),剥离后 8.0 MB。** trace 对复核与配对检验无用,而 342 MB 会永久拖慢全组 clone
  (GitHub 单文件 >50 MB 即告警)。**完整 trace 仍留在 bp1 的 `runs/pipe-2wiki-*/`,未删除。**

---

## R8 — 换上真实 generator:排序的传导还在吗,以及模型有多少答案不是从证据来的

**状态:** READY——三件套齐(四个 `configs/experiments/pipe8_2wiki_*.toml` + 现有 pipeline runner
+ 本条目);**跑之前写。** 四个 config 与 R7 的**逐字相同**,唯一差别是
`[generator] name` 由 `extractive` 改为 `granite`,以及输出目录。

**BEFORE(预注册):**

- **⚠️ 首先声明一件必须先讲清楚的事:R7 与 R8 的 `answer_match` 是同一个指标名,
  测的却是两件不同的事,绝不可直接相减。**
  - R7 的 `answer` 字段是 `ExtractiveGenerator` 拼接的**约 900 词证据原文**
    ⇒ 命中 = **答案串被送到了生成器面前**(证据传递)。
  - R8 的 `answer` 是 `GraniteGenerator` 生成的**一句话**
    ⇒ 命中 = **模型真的把答案说出来了**(答案正确性)。
  分母性质变了,**R8 的绝对值必然远低于 R7,这不是退步**。把两者之差当作"生成损失"是错的。
- **主问题:R7 测到的排序传导,在链条末端换成真实生成器之后还成立吗?**
  R7 的干净对照(decompose vs bm25,池子 p=0.38 不可区分、排序崩塌、下游 −15.4pp p=0)
  在 R8 上应当**同号且显著**。若消失,说明生成器自身的方差淹没了检索差异。
- **⚠️ 范围界定:本条是 Retriever 组的实验,目的只有一个 —— 守住 R7 得出的检索建议。**
  R7 的头条("hybrid-rrf 在系统层面也显著更好")目前建立在**玩具 generator** 之上;
  换真实 generator 看它还成不成立,是在验证**自己的**结论,不是评估生成质量。
  生成器在本条中是**固定的下游部件**,与 selector 同等地位,不是被研究对象。
- **预期指标 + 方向:**
  - `system.core.answer_match` 四臂**绝对值远低于 R7**(见上,分母性质不同),
    但**臂间序关系保持**(decompose 最低,hybrid-rrf 最高)。
  - decompose vs bm25 的下游差**仍显著为负**(配对随机化,n=2000)。
  - **判据只有一条:R7 的检索建议在真实 generator 下是否依然成立。**
- **诚实的替代结果(全部有价值,不许事后挑):**
  (a) **排序传导在 R8 上消失/不显著** ⇒ 生成器方差压过检索差异,**R7 的结论只对"证据传递"成立,
      不能外推到答案质量**。这会直接下调检索优化对最终系统的价值主张,必须照写,
      并把 R7 与周报里的措辞收紧到"证据传递"为止。
  (b) **臂间序关系与 R7 相反** ⇒ 存在与检索质量反向相关的生成行为,本条无法解释,
      须交由 Generator 组查,**Retriever 侧只报现象不作解释**。
  (c) **⚠️ 地板效应(R7 天花板风险的镜像):** 若四臂 answer_match 全部逼近 0,
      指标在下端饱和、失去分辨力,本条判为**未能分辨**,须改用更宽松的答案匹配或换带
      长答案标注的数据集,**不许把"全低"读成"检索无用"**。
- **⚠️ 与 S2/S4/S5 同族的计分伪影,在本条上更严重且方向已知:** `answer_match` 是 exact-string
  包含。R7 的"答案"有 900 词,包含容易;**R8 只有一句话,模型换个说法就判负**。
  故 **R8 的绝对值是答案正确率的下界,且低估幅度大于 R7**。**臂间相对比较仍可信,
  而本条的判据本来就只用相对比较。**
- **自检:** 四臂 retriever 指标必须与 R7 一致(检索链完全相同)。CPU 两臂应精确复现
  `.9434`/`.9580`;GPU 两臂按 R7 的实测偏差(hybrid −0.0002、decompose +0.0031)量级。
  不一致则先查 harness,下游数字不读。
- **已知风险(须在提交前知情):**
  1. **显存。** hybrid 臂需要**同时**持有 Granite embedder 与生成用 LLM,而 R7 中这两个模型
     **从未同时载入**(分属不同臂)。rtx_2080 只有 8GB,granite-4.1-3b fp16 权重即约 6GB。
     若报 `CUDA out of memory`,**改钉 `--gres=gpu:rtx_3090:1` 或 a100 并接受排队**,
     不要为此改小 `max_selected` —— 那会改变与 R7 的可比性。
  2. **运行时长。** R7 是 1h36m,其中只有 decompose 臂调 LLM。R8 每臂每 query 都要生成一次
     ⇒ 约 8000 次额外 LLM 调用。故申请 `--time=08:00:00`,不要沿用 4 小时。
- **本条不测什么(范围红线):** 不评估生成质量本身;不测引用正确性
  (`citation_validity` 会被记录但**不作判据**);不改 selector(仍为 `top-k` 直通,
  与 R7 保持唯一变量);不引入 `verify-annotate` generator(那是又一个变量,应另立条目)。
  **凡属 Generator 模块的度量,本条只在它们恰好被记录时保留数据,不解释、不下结论、不据以提建议。**
- **顺带可得、但不属本条的数据:** 两轮 query_id 与检索链完全相同,故 R7×R8 逐条配对
  在技术上可给出一个 2×2(证据里有无答案 × 模型答对与否)。**该分析不在本条范围内,
  也不作为判据**;若 Generator 组需要,数据在两轮的 per-case raw 里,自取即可。
- 精确命令:
  ```
  mkdir -p logs runs && sbatch --gres=gpu:1 --time=08:00:00 scripts/run_pipeline_eval.slurm \
    configs/experiments/pipe8_2wiki_bm25.toml \
    configs/experiments/pipe8_2wiki_strong-bm25.toml \
    configs/experiments/pipe8_2wiki_hybrid-rrf.toml \
    configs/experiments/pipe8_2wiki_decompose.toml
  ```
  拉 raw 时**同样须剥离 `trace`**(R7 实测 342 MB → 8.0 MB),理由见 R7 的 AFTER。
- Git commit:待本次改动提交后填;Seed:7;n=2000/臂;
  与 R7 唯一变量 = `[generator] name`。

**AFTER:** 未运行。<!-- 填:job id、四臂表、retriever 自检、decompose vs bm25 的配对 p、
臂间序关系是否与 R7 一致、三种替代结果命中哪个、是否触发地板效应、
R7 的检索建议是否依然成立 -->

---

## R9 — chunk 粒度扫参:第一次让"切多大"这件事有证据

**状态:** READY——三件套齐(配置化已合入 `0792d22` + 七个
`configs/experiments/chunk_2wiki_*.toml` + 本条目);**跑之前写。**
纯 CPU(retriever=strong-bm25,generator=extractive),不需要 GPU、不调 LLM。

**为什么现在做:**

`chunk_size`/`overlap` 自项目开始就固定在 `WordChunker` 的默认值,且**实验配置根本够不着**
(`experiment.py` 一直是空参数的 `CorpusBuilder()`),所以从未被扫过 —— 而 chunk 是
**三个模块共用的证据单元**,共享计划 §4.3 也明写它是公共能力、后续要作可替换实现测试。
R7 给出了具体动机:它量出一段**与检索器无关**的损失 —— gold 文档进了最终 5 条,
答案串仍约 21–26% 不在其中(`answer/sysRecall` 0.744–0.793,四臂相近)。那是粒度问题。

**⚠️ 一个会制造假结论的混淆,必须先摁住(本条最重要的设计):**

固定 `max_selected=5` 去扫 chunk_size,送到下游的**文本总量**随之改变:

| chunk_size | 60 | 120 | 240 | 480 |
|---|---|---|---|---|
| top-5 总词数 | 300 | 600 | 1200 | **2400** |

而 `answer_match` 是**子串包含**。文本量涨 8 倍,命中率**必然机械上升**,与检索质量无关。
若只跑这一组,几乎注定得到"chunk 越大越好"——**那是体积效应,不是粒度效应**。
故本条分两组:

- **Set A(粒度,固定选中条数):** chunk_size ∈ {60,120,240,480},overlap = chunk_size/6,
  `max_selected=5`。**文档级指标(recall / MRR / sysRecall)在此组有效**——它们问的是
  "找没找到那篇文档",与文本量无关。**`answer_match` 在此组跨点不可比,只作记录。**
- **Set B(粒度,固定证据预算):** chunk_size × max_selected ≡ **600 词**
  ——(60,10)、(120,5)、(200,3)、(300,2),overlap 同为 chunk_size/6。
  **体积被摁住,此组的 `answer_match` 才可跨点比较。**

两组共用 `a-c120o20`(= 当前生产值 120/20/5),它同时是两组的锚点与基线。

**BEFORE(预注册):**

- **主问题:存不存在一个内部最优的 chunk 粒度?** 机制上应有拮抗:
  chunk 变大 → 答案串更可能落在某个被检回的 chunk 内(利)、但 BM25 词频被稀释、
  定位变粗(弊);chunk 变小 → 定位精准(利)、但答案可能被切断在 chunk 边界外(弊)。
  ⇒ **预期 Set B 的 `answer_match` 呈单峰,峰不在两端。**
- **预期指标 + 方向:**
  - Set A:`retriever.core.document_recall` 随 chunk_size **上升**(整篇文档更容易被覆盖),
    `document_mrr` **下降**(词频稀释,定位变粗)。两者反向是本条机制成立的标志。
  - Set B:`answer_match` **单峰**;若最优点不是 120,则**当前生产值就是选错的**。
  - `chunk_count` 随 chunk_size 单调下降 ⇒ 按 R5 的 `ms/1k_chunks` 常数,
    **大 chunk 同时更快**,这是与质量正交的一项收益,须一并记录。
- **诚实的替代结果(全部有价值,不许事后挑):**
  (a) **Set B 的 answer_match 基本持平** ⇒ 在固定证据预算下粒度无关紧要,
      **R7 那 21–26% 的损失不是粒度造成的**,需另找解释(如 answer_match 的 exact-string 伪影本身)。
      这会直接否掉本条的动机,必须照写。
  (b) **Set B 单调上升到端点** ⇒ 无内部最优,应继续加大 chunk 直到某个约束(上下文长度)生效;
      **不许把"最大的最好"读成"不该切分"** —— 480 那点已接近整篇文档,其行为需单独说明。
  (c) **Set A 的 recall 与 MRR 未出现反向** ⇒ 上述拮抗机制不成立,本条的解释框架需重写。
  (d) **Set A 的 answer_match 单调上升而 Set B 持平** ⇒ 正是体积效应的指纹,
      **证实本条的分组设计是必要的**,且任何只跑 Set A 的研究都会得出错误结论。
- **自检:** `a-c120o20` 必须精确复现 R7 中 strong-bm25 臂的 retriever 指标
  (`MRR .9580` / `Recall .7678`)—— 它与 R7 唯一的差别是 chunker 现在走配置而非默认值,
  而 `0792d22` 的测试已证默认路径 corpus_signature 逐位不变。**若不复现,说明配置化改坏了东西,
  本条全部作废。**
- **本条不测什么:** 不换 retriever(固定 strong-bm25;粒度 × 检索器的交互**未测**,
  须另立条目);不换 generator(固定 extractive,故本条测的仍是**证据传递**);
  不测 token-aware / 结构感知切分(那需要新的 Chunker 实现,是另一件事)。
- **⚠️ 已知会被质疑的一点:** `overlap = chunk_size/6` 是**约定而非受测变量**。
  本条把 overlap 与 chunk_size 绑定,故**无法分离二者**。若结果显示粒度重要,
  下一步才值得单独扫 overlap;现在就做二维扫参是浪费。
- 精确命令(纯 CPU,`compute` 分区,数据集已物化于 `runs/twowiki`):
  ```
  mkdir -p logs runs && sbatch --partition=compute --gres=none --time=04:00:00 \
    scripts/run_pipeline_eval.slurm \
    configs/experiments/chunk_2wiki_a-c60o10.toml \
    configs/experiments/chunk_2wiki_a-c120o20.toml \
    configs/experiments/chunk_2wiki_a-c240o40.toml \
    configs/experiments/chunk_2wiki_a-c480o80.toml \
    configs/experiments/chunk_2wiki_b-c60o10.toml \
    configs/experiments/chunk_2wiki_b-c200o33.toml \
    configs/experiments/chunk_2wiki_b-c300o50.toml
  ```
  **注意:每点都要重建语料与索引**(corpus_signature 依 chunk 设置而变,`0792d22` 有测试守卫),
  故 `prepare` 不可跳过;这也是各点不会误共用索引的机制。
- Git commit:待本次改动提交后填;Seed:7;top_k=50;n=2000/臂;
  retriever=strong-bm25(k1=0.9/b=0.4)全程不变。

**AFTER(2026-08-10,job 18329959,partition=compute,`bp1-compute215`,七点各 2000/2000;
另有 job 18329957 为重复提交,`scancel` 于 1:52,未污染——锚点逐位复现可证):**

| 组 | chunk × max_sel | MRR | Recall | selRecall | selPrec | sysRecall | **answer** |
|---|---|---|---|---|---|---|---|
| A | 60 × 5 | 0.9643 | 0.7645 | 0.8965 | 0.3617 | 0.6684 | 0.4775 |
| A/B | **120 × 5**(生产值) | 0.9580 | 0.7678 | 0.9015 | 0.3485 | 0.6766 | 0.5185 |
| A | 240 × 5 | 0.9571 | 0.7688 | 0.9070 | 0.3313 | 0.6825 | 0.5420 |
| A | 480 × 5 | 0.9563 | 0.7709 | 0.9065 | 0.3211 | 0.6839 | 0.5505 |
| B | **60 × 10** | 0.9643 | 0.7645 | **0.9461** | 0.1956 | **0.7135** | **0.5405** |
| B | 200 × 3 | 0.9573 | 0.7694 | 0.8486 | 0.5074 | 0.6338 | 0.4700 |
| B | 300 × 2 | 0.9570 | 0.7699 | 0.7904 | **0.6730** | 0.5854 | **0.4145** |

- **自检通过。** `a-c120o20` 精确复现 R7 中 strong-bm25 臂的 `MRR .9580` / `Recall .7678`
  ⇒ chunking 配置化(`0792d22`)与队友的索引 `write_bytes` 改动(`5d59b22`)**均未影响检索**。
  七点 `dataset_signature` 同为 `69dce7a2d909808d`。

- **⚠️ 本条最重要的结果,而且它是关于方法而非关于 chunk 的:混淆把效应的符号翻了。**

  | 同一变量 chunk_size 变大时 | `answer_match` |
  |---|---|
  | Set A(体积随粒度从 300 涨到 2400 词) | **+0.0730 上升** |
  | Set B(体积固定 600 词) | **−0.1260 下降** |

  **只跑 Set A 会得出"chunk 越大越好";固定证据预算后真相是"越小越好"。**
  预注册时我把这个混淆写成"会机械抬高 answer_match",**低估了它** —— 它不是抬高幅度,
  是**反转方向**。任何不控制证据体积的 chunking 研究,结论可能整个是反的。
  这命中并强化了预注册的替代结果 (d)。

- **预注册的"单峰"预测被证伪。** Set B 不是单峰,是**单调**:0.5405 → 0.5185 → 0.4700 → 0.4145,
  三个对照全部显著(vs 120×5 基线,配对随机化 n=2000):
  **60×10 `+0.0220 p=0.0008`;200×3 `−0.0485 p=0.0000`;300×2 `−0.1040 p=0.0000`**。
  命中的是替代结果 (b),但方向与我描述的相反 —— 我写的是"单调升到大端",实测是**单调偏向小端**。
- **⇒ 当前生产值 120 不是最优。** 在完全相同的 600 词预算下,**60×10 高 2.2pp(p=0.0008)**。
- **⚠️ 而 60 是本次扫描的最小值,即最优点落在扫描边界上 ⇒ 真正的最优可能还在更小处,未测。**
  这是本条**未能回答**的部分,不得声称"60 就是最优"。

- **机制:chunking 通过"最终文档召回"起作用,几乎是一比一传导。**
  Set B 中 `Δanswer` 与 `ΔsysRecall` 同号同量级(比值 0.60 / 1.13 / 1.14,三者 p 均 =0.0000)。
  同预算下摊成更多小块 → `selRecall` 0.9461 vs 0.7904、`sysRecall` 0.7135 vs 0.5854
  ⇒ **覆盖到更多不同的 gold 文档**;而 `selPrec` 反向(0.1956 vs 0.6730)。
  **answer 跟覆盖走,不跟精度走** —— 与 R7"下游要的是含答案的文本到没到眼前"一致。

- **⚠️ 更正我从点估计得出的说法:Set A 的拮抗在方向上成立,但多数相邻差不显著。**
  昨日据点估计称"四个点都成立",配对检验后须收紧:
  60 vs 120 显著(MRR `+0.0063 p=0.0033`、Recall `−0.0032 p=0.0232`);
  **240 vs 120 两项均不显著**(p=0.5806 / 0.4040);
  480 vs 120 仅 Recall 显著(`+0.0031 p=0.0154`),MRR 不显著(p=0.3143)。
  ⇒ **拮抗真实但很小,只在小端被清晰分辨**;预注册替代结果 (c)(拮抗不成立)未命中,
  但也不能说四点全部确立。

**限制(须与结论同时声明):**

1. **Set B 固定总词数就必然同时改变选中条数(10/5/3/2),"小 chunk 更好"与"多条目更好"
   在本设计下无法分离。** 这是固定预算设计的固有代价。诚实的表述是:
   **在固定证据预算下,把预算摊到更多更小的片段上,比集中在少数大片段上更常把答案送到位。**
2. **`overlap = chunk_size/6` 全程绑定,无法分离 overlap 的独立作用。** 既然粒度确已重要,
   单独扫 overlap 现在才值得做。
3. 仅在 2Wiki、仅 strong-bm25、仅 extractive generator 上测得。
   **粒度 × 检索器的交互未测**;换真实 generator 后是否同向亦未测。
4. `answer_match` 是 exact-string 包含,与 S2/S4/S5 同族,绝对值为下界;
   **⚠️ 该“下界”表述已被 R16 作废(2026-08-14):实测为双向偏差 —— 答案为 `yes` 时是下界,为 `no` 时是**上界**(2Wiki 上 52/54 判为答对)。臂间相对比较仍可信,故本条结论不变。**
   **本条全部判据均为组内相对比较,不受影响。**
5. 最优点在扫描边界上(见上),范围未覆盖 60 以下。

- raw:`results/r9-2wiki-{a-c60o10,a-c120o20,a-c240o40,a-c480o80,b-c60o10,b-c200o33,b-c300o50}-per-case.json`
  (已 `git add -f` 拉回)。上表已从 raw 重新聚合复核,逐位一致。
  **同样剥离了每条的 `trace`:七份合计 660 MB → 14 MB**,理由见 R7 的 AFTER;完整 trace 留在 bp1。
- **写给 results-summary 的草稿:** 见下条 R9。**标题应是方法结论而非参数结论** ——
  "证据体积不控制会让 chunking 结论反号",比"chunk 该设 60"更重要也更可迁移。

---

## R10 — 单独扫 overlap:R9 无法分离的那个变量

**状态:** READY——三件套齐(四个 `configs/experiments/ovl_2wiki_c120o*.toml` +
现有 pipeline runner + 本条目);**跑之前写。** 纯 CPU,不需要 GPU、不调 LLM。
第五个点(overlap=20)即 R9 的锚点 `chunk_2wiki_a-c120o20`,**已跑,不重复**。

**为什么是 overlap,而不是"继续往更小的 chunk 扫":**

R9 发现固定预算下 answer 单调偏向小 chunk,最优点落在扫描边界(60)上。直觉的后续是往 60 以下扫,
**但那条路会走进退化区**:固定 600 词预算时 `max_selected = 600 / chunk_size`,

| chunk | 60 | 40 | 30 | 20 |
|---|---|---|---|---|
| max_selected | 10 | 15 | 20 | **30** |
| 占 `top_k=50` | 20% | 30% | 40% | **60%** |

chunk 越小,选中条数越逼近整个候选池,**selector 退化为"几乎全要"**。那时 sysRecall 上升是
因为几乎没在筛,不是因为粒度合适 —— **与 R9 刚揭穿的体积混淆是同一件事换了副面孔**。

overlap 则是唯一能在**选中条数不变、名义预算不变**的前提下调节"每块携带多少上下文"的旋钮,
也正是 R9 限制 2 点名的、被绑定为 `chunk_size/6` 而无法分离的那个变量。

**⚠️ overlap 自己的混淆,以及由此确定的推断规则(本条最重要的设计):**

同一文档的相邻块共享 `overlap` 个词。overlap 越大 → 相邻块越相似 → 越容易同时进 top-5
⇒ **实际送到下游的唯一词数越少**(名义仍是 600)。该混淆**与"overlap 有帮助"的假设方向相反**,
故推断是不对称的:

| 若实测 | 可否解读 |
|---|---|
| **overlap 越大越好** | ✅ **可以** —— 混淆在反方向拖后腿,真实效应只会**更大**,不会更小 |
| **overlap 越大越差** | ❌ **不可以** —— 分不清是覆盖损失(机制)还是重复稀释(混淆) |

**这条规则在跑之前写死:出现"越大越差"时不得声称已解释,只能报告并说明二者未分离。**

**BEFORE(预注册):**

- 设计:**唯一变量 = overlap ∈ {0, 20, 40, 60, 80}**。`chunk_size=120`、`max_selected=5`、
  `top_k=50`、retriever=strong-bm25、generator=extractive、seed=7、数据集 `runs/twowiki` 全部固定。
- **主假设(又一个拮抗):** overlap 越大 → 答案跨块边界被切断的概率越低(利);
  但相同 50 个候选位覆盖的**不同语料范围**越窄(弊)⇒ **预期 answer 呈单峰,峰不在两端。**
  ⚠️ 注意:R9 也预期单峰而实测单调,**本条不因"上次错了"就改口,仍按机制预期写单峰。**
- **预期指标 + 方向:**
  - `chunk_count` 随 overlap **上升**(step 从 120 降到 40 ⇒ 约 3 倍),这是**成本项**:
    按 R5 的 `ms/1k_chunks` 近似常数,overlap=80 的每 query 检索开销约为 overlap=0 的 **3 倍**。
    **质量若无明显提升,该成本足以否掉高 overlap。**
  - `retriever.core.document_recall`(top-50)在高 overlap 处**下降** —— 50 个候选位覆盖的
    不同语料变少。这是"弊"侧的直接读数,也是与重复稀释区分开的一个旁证。
  - `system.core.answer_match` 为主指标,判据见上表的不对称规则。
- **诚实的替代结果:**
  (a) **五点基本持平** ⇒ 在此范围内 overlap 无关紧要,**R9 观察到的粒度效应与"边界切断"无关**,
      应转而解释为覆盖/条数效应。这会缩小 R9 结论的机制解释范围,必须照写。
  (b) **单调升到 80** ⇒ 边界切断确实重要;因混淆反向,真实效应**至少这么大**。
      但须同时权衡 3 倍的检索开销,**不得只报质量不报成本**。
  (c) **单调降** ⇒ 见上表,**不可解读**;只报现象,并说明须测"唯一词数"才能分离。
  (d) **内部最优** ⇒ 预期形状成立,给出推荐值。
- **自检:** `overlap=20` 一点直接复用 R9 的 `chunk_2wiki_a-c120o20`,其
  `MRR .9580` / `Recall .7678` 已两次复现(R7、R9)。新四点的 retriever 指标应落在其邻域,
  **若某点 MRR 偏离超过 R9 中 60↔480 的整个跨度(0.008),先查 harness。**
- **⚠️ 跨代码版本复用旧臂的前提,已实测确认(2026-08-10,提交作业后补记):**
  锚点跑在 R9 时的代码上,新四点跑在此后的代码上,中间队友改了
  `TopKSelector`(`5210b03`):排序键由 `(-retrieval_score, evidence_id)` 改为
  `(retrieval_rank, evidence_id)`。**若二者顺序不同,"overlap 是唯一变量"即不成立。**
  实测(300 query × 50 候选,含 9012 个并列分数)**两种键给出的顺序完全一致** ——
  因为检索器本就按 `(-score, evidence_id)` 排序后依序赋 rank,rank 与该键同序。
  故该改动是表述澄清而非行为变更,**复用锚点成立**。
  **教训:复用跨版本的旧臂前应先验证中间改动是否触及该臂的代码路径,不能默认没变。**
  本次是补验(作业已提交后才想到),下次应前置。
- **本条不测什么:** 不动 chunk_size(固定 120 = 生产值,便于与 R9 直接对照);
  不换 retriever/generator;**不测"唯一词数"这一混淆量本身** —— 它需要保留 trace 才能算,
  而 trace 因体积原因不入库(见 R7/R9)。若结果落到 (c),再单独立项测它。
- 精确命令(纯 CPU,`compute` 分区;overlap=20 那点不重跑):
  ```
  mkdir -p logs runs && sbatch --partition=compute --gres=none --time=04:00:00 \
    scripts/run_pipeline_eval.slurm \
    configs/experiments/ovl_2wiki_c120o0.toml \
    configs/experiments/ovl_2wiki_c120o40.toml \
    configs/experiments/ovl_2wiki_c120o60.toml \
    configs/experiments/ovl_2wiki_c120o80.toml
  # 汇总时把 R9 的锚点一并列入(它就是 overlap=20 那一点):
  #   configs/experiments/chunk_2wiki_a-c120o20.toml
  ```
- Git commit:待本次改动提交后填;Seed:7;n=2000/臂。

**AFTER(2026-08-10,job 18357155,partition=compute,`bp1-compute173`,COMPLETED,
elapsed 00:06:21;overlap=20 一点复用 R9 锚点,未重跑):**

| overlap | step | MRR | Recall | selRecall | selPrec | sysRecall | **answer** |
|---|---|---|---|---|---|---|---|
| 0 | 120 | 0.9588 | 0.7681 | 0.9024 | 0.3414 | 0.6777 | **0.5200** |
| **20**(生产值) | 100 | 0.9580 | 0.7678 | 0.9015 | 0.3485 | 0.6766 | 0.5185 |
| 40 | 80 | 0.9568 | 0.7690 | 0.8975 | 0.3555 | 0.6745 | 0.5180 |
| 60 | 60 | 0.9582 | 0.7672 | 0.8957 | 0.3651 | 0.6716 | 0.5150 |
| 80 | 40 | 0.9554 | 0.7659 | 0.8890 | 0.3828 | 0.6649 | **0.5060** |

- **命中替代结果 (a),不是我在点估计上以为的 (c)。** 配对随机化(n=2000,基线=生产值 20):
  o0 `+0.0015 p=0.6916`、o40 `−0.0005 p=1.0000`、o60 `−0.0035 p=0.2979` —— **三点均与生产值
  统计上不可区分**;只有 **o80 `−0.0125 p=0.0018` 显著更差**。
  端点全跨度 0 vs 80:answer `+0.0140 p=0.0007`、sysRecall `+0.0129 p=0.0000`。
  ⇒ **overlap 在 0–60 区间内无关紧要,只有推到 2/3 冗余(80)才真的伤到。**
- **⚠️ 更正我从点估计得出的读数。** 看到表格时我称"answer 单调下降";点估计确实单调,
  但**检验后只有极端点被分辨出来**,中间三点全不显著。**"单调下降"这个说法过强,收回。**
- **⇒ 一个有内容的负结果:边界切断在本设置下不是有意义的因素。**
  overlap 的作用机制就是减少"答案跨块被切断",而**把 overlap 从 0 加到 60 完全没有带来收益**
  (三点全不显著,点估计还略降)。⇒ **R9 观察到的粒度效应不能用"边界切断"解释**,
  只能归于覆盖/条数。这缩小了 R9 结论的机制解释空间 —— 正是预注册 (a) 写明的后果。
- **`answer_match` 对 overlap 的敏感度比对 chunk 粒度低约一个数量级:**
  R9 固定预算下跨度 **0.126**,本条全程仅 **0.014**(且其中大部分来自 80 那一个点)。
  **粒度是主旋钮,overlap 不是。**
- **成本项(预注册要求与质量同报):** step 从 120 降到 40 ⇒ chunk 数约 **3 倍**,
  按 R5 的 `ms/1k_chunks` 近似常数,每 query 检索开销亦约 3 倍。
  **overlap=80 质量显著更差且成本 3 倍,被双重支配,明确排除。**
- **实用结论:** 0–60 任取皆可,**现行生产值 20 无需更改**(与 R9 不同 —— 那里 chunk_size=120
  被证明不是最优)。若要在等价的质量中挑最省的,**overlap=0 的 chunk 数最少**,
  但其质量优势 `+0.0015 p=0.6916` **不成立**,故这是成本理由,不是质量理由。
- **⚠️ top-50 层面的覆盖损失未被确立。** 0 vs 80 的 `Recall`(top-50)仅 `+0.0022 p=0.0634`
  ——**未达显著**;而 `sysRecall`(top-5)`+0.0129 p=0.0000` 显著。
  ⇒ 损失几乎全部发生在**选择阶段**(近似重复的块挤占 top-5 名额),而不是候选池阶段。

**限制(须与结论同时声明):**

1. **o80 那一点的显著下降,按预注册的不对称规则仍不可归因** —— 覆盖损失与重复稀释
   在本设计下分不开。**只报现象。** 上面"挤占 top-5 名额"的说法是对**位置**的描述
   (损失发生在哪一步),不是对**成因**的判定。
2. **⚠️ 一个事后才想到的重新解读,明确标为事后:** 预注册时我把"唯一词数变少"当作**混淆**,
   但它其实是**系统真实的行为**——高 overlap 确实让下游拿到更少的不同证据,
   这是效应而非测量伪影。若如此,(c) 未必真的不可解读。**但该推理产生于看到结果之后**,
   故它是下一条的假设,不是本条的结论;本条仍按预注册处理。
3. 仅在 2Wiki(答案多为短实体)、chunk_size=120、strong-bm25、extractive generator 上测得。
   **"边界切断不重要"这一结论对长答案数据集不可外推** —— 短实体本就不易被切断。
4. 未测 overlap 与 chunk_size 的交互(本条 chunk_size 固定 120)。

- raw:`results/r10-2wiki-c120o{0,40,60,80}-per-case.json`(已 `git add -f` 拉回,
  每份 ~2 MB,已剥 `trace`,原始四份合计 367 MB);overlap=20 一点见
  `results/r9-2wiki-a-c120o20-per-case.json`。上表已从 raw 重新聚合复核,逐位一致。

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
- **Harmful-in-context +0.3pp**(exact .569 → lenient .572),paired p=0.55,CI[−.005,+.012] ——
  ~~**不显著(无 harm 代价)**~~ **不显著;代价上界 +1.2pp(CI 上端)**。
  **措辞 2026-08-09 修订,与 R001b 统一口径:不显著只把代价夹在 CI 里,不等于代价为零。**
  数字一个未动,改的只是能从它主张什么。
- 定论:门内 lenient 聚类**显著回收召回,harm 代价上界 +1.2pp**,回收 −4.8pp 门代价的 ~25%;残差(~75%,含 .39 孪生 missed-conflict)= Graph 2.0 语义等价目标,已量化。发现落 `docs/results-summary.md` S3。
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
- **AFTER(2026-08-09 补录,job `18235971`,提交 08-01T00:09,FAILED 1:0,elapsed 05:19:20,bp1-gpu025):**

  **(a) 的两个数此前已填在本条目上方**(2026-07-31),那行"待填"从那时起就是过时的,已改。
  本次为读 (b) 重跑了一次 `parent_collision`,**四个数逐字复现**(`.931` / `16.4735` / `20.0` /
  `n_unresolved 0`)—— 白捡一次 rerun-stability,同 R001 的先例。

  **作业本身 FAILED,但 (b) 的数据全在:** 第一臂 `niah_e2_gate_on_lenient_parent`
  prepare→retriever→selector 三步跑完(产物 00:09 / 01:12 / 05:28),死的是第二臂 gate-off 的
  **prepare**,`ValueError: upstream artifact hashes mismatch for runs/e2-gate-off/run_manifest.json`。
  **五小时买到了完整的 ON 臂,零 GPU 秒买到了 OFF 臂的拒绝。**

  **根因 = `d5f7908` 的 schema 重塑,不是索引变了。** 该 commit(07-24 10:54)把 BM25 的 `k1`/`b`
  从 `IndexManifest` 的平铺字段折进嵌套 `parameters`;`_index_signature` hash 的正是这个 payload,
  于是同语料同参数算出不同 `index_signature`(`fa117ff6…` → `fba3a1ae…`)。gate-off 建于 07-23(重构前),
  其 `index/` 于 07-24 22:22 被清后按新 schema 重建(E2-lenient 条目记的 `rm -rf runs/*/index`),而
  `run_manifest.json.metadata.json` 记的是**被删那个文件**的 sha256,删不掉 ⇒ 守卫永久拒绝。
  **`prepare()` 写索引(146)→ 校验(153)→ 写 run_manifest(155)** 的顺序加上两边 mtime 证明:
  07-24 22:22 那次执行也停在 153,**这堵墙 08-01 是第二次撞,第一次没留下条目**。

  **等同性已证,不是论证:** `scripts/verify_legacy_index_identity.py` 按重构前 schema 重建那个已删文件的
  字节并 hash,得 `947844363ceee…f5b0`,**与 sidecar 记录逐字相同**;`index/corpus_snapshot.json` 两侧
  sha256 本就相同。⇒ implementation / version / corpus_signature / k1 / b **全部相同,唯一差异是 k1/b 的编码位置**。
  判别性已验:k1 由 1.5 改 1.2 即 FAIL。**⇒ 两臂可比,(b) 不必重跑;但该目录经 `read_manifest` 的复用仍永久不可行**
  (它 hash 的文件已不存在),离线比 dump 是唯一路径。

  **(b) 读数(离线配对,seed 13,10000 次随机化):**

  | 对照 | 轴 | ON | OFF | delta | p | 95% CI | n |
  |---|---|---:|---:|---:|---:|---|---:|
  | vs gate-off | harm | **.5747** | .6802 | **−10.55pp** | 0.0 | [−.1237, −.0879] | 1479 |
  | vs gate-off | recall | **.8306** | .8680 | **−3.74pp** | 0.0 | [−.0452, −.0302] | 1848 |
  | **vs lenient/document(唯一变量 = `support_unit`)** | harm | .5747 | .5720 | **+0.27pp** | **.7522** | [−.0095, +.0156] | 1479 |
  | **vs lenient/document(唯一变量 = `support_unit`)** | recall | .8306 | .8324 | **−0.18pp** | **.5377** | [−.0075, +.0040] | 1848 |

  两个独立口径核对通过:`pool_hit_rate` **.8425** 与 (a′) 记的 `≈0.842` 一致;`harm_off` 对 gate-off
  臂读出 `.6801893` 与 S1 记录的 `.680` 同源同分母。

  - **预注册条件推论(本条目"必须一起读的推论")—— 前件不成立。** harm **没有**明显回弹向 `.680`:
    回弹 0.57pp,占门总效应 11.2pp 的 5%。⇒ **S1 的 −11.2pp 是机制收益,不是"同一篇条目被数多次"的记账收益。**
    该推论的后件因此不触发,但前件与后件都在此登记。
  - **事先固定的判读口径(条件 4 通道)—— 触发但不可分辨于零。** recall 经该通道降 0.18pp,**p=.5377,CI 跨零**。
    "撤掉假保护 vs 门真的误踢"的分开计因此无实质可分,登记为已测、量级在噪声内。
  - **两轴皆 null,但 null ≠ 零代价。** 可主张的是**代价上界**:harm **+1.56pp**、recall **−0.75pp**
    (CI 上端),均远低于它们要保护的效应量(11.2pp / 4.8pp)。**不得写成"零代价"。**
  - **⇒ `support_unit=parent` 可以无条件采用**,Graph 2.0 主对照两侧同单位这个前提以低于噪声的代价买到。

  **本轮最值钱的一句 —— `independent_support` 的虚增在统计上普遍,在因果上惰性:**
  机会侧 `collision_rate` **.931**、`needle_parent_inflation_rate` **.328**(415/1264);兑现侧 harm
  **恰好 4 题**(846/1479 → 850/1479,二值可整除)、recall 约 **2.6 题份**(每题均降 .0014 × 1848;
  recall 是每题分数非 0/1,只能算等效题数)。**(a′) 把 .328 定性为"上界不是实测"是对的,而这个上界松了两个数量级。**
  93% 的窗口含重复来源,门的决策几乎不靠那些重复票撑着。发现待落 `docs/results-summary.md`。

  **纪律教训(比读数更该记):E2-lenient 条目末尾那条运维注在 07-25 就写下了** —— "d5f7908 改了 IndexManifest
  schema → 旧 index cache 全失效 → run_manifest hash 守卫会拒重建的 index → 离线比 selected dump 最省"。
  R001b 的提交命令走的正是它说会被拒的那条路,**六天后原样再撞,代价 5h19m**。
  这是 R013-smoke 那节「可复用教训二」的教科书实例:**纯文字的义务没有代码执行,迟早被跳过。**
  ⇒ **新纪律,已落地而非写下:`scripts/preflight_arms.py` + `run_selector_gate.slurm` 在第一个 `prepare`
  之前调用它**,把所有臂的 stored manifest 与盘上 index 逐一对拍,**一次报完所有臂**(预检的意义是一分钟
  看清全部问题,不是一次提交查出一个)。四种判读:`OK` / `FRESH`(无 run_manifest,prepare 会写)/
  `INDEX ABSENT`(**故意判失败** —— prepare 会重建索引再跟一个已不存在的文件的哈希比,能否通过取决于当前
  代码是否与当初逐字同序,从这里不可知;真想干净重建就把 `run_manifest.json` 与其 sidecar 一并删掉)/
  `MISMATCH`。四格与两个退出码均已用 fixture 实测。`set -euo pipefail` 使其非零退出在任何 `prepare` 之前终止作业。
  它查的东西第 0 秒就完全可判定,成本零 GPU 秒;R013–R015 是十五次全量 fine-tune,这条会立刻回本。
  与 R013-smoke 那节"五次全部死在训练开始之前,说明守卫的位置是对的"恰成反例——同一原则,位置放反。

  **未做,理由在此:** 不把 `implementation_version` 补 bump 到 `bm25-v2`。盘上每个索引 manifest 都写着
  `bm25-v1`,bump 会让 `load_index` 对**所有**既有 run 目录失配(E2 三臂 / lenient / lenient-parent /
  niah-injected 全部要重 prepare),距 9/4 四周不划算;且 bump 是一次性动作、不产生纪律。
  **`bm25-v1` 现同时指两套 payload schema,这是真实的溯源缺陷,在此登记。**
  **替代防守已落地**(`tests/retriever/test_indexing.py`):对固定 fixture 语料钉住**两个**常量 ——
  `index_signature`(两次运行据以判同一的身份)与 **`index_manifest.json` 文件本身的 sha256**
  (`_validate_stored_manifest` 实际比对的那个量,且它对键序、分隔符、尾换行这些 signature 抓不到的
  改动也敏感)。payload 形状一改,CI 秒级红。配一条判别性测试:k1 由 1.5 改 1.2 两个常量都必须变,
  否则这个钉子是假的。零运行时代价、对既有产物零影响、且是**测试非行为改动**,加进他人模块摩擦最小。
  测试的 docstring 写明红了以后怎么办:要么撤回,要么在同一个 commit 里 bump `implementation_version`
  并重录两个常量 —— 并写明 bump 的代价。真要 bump,应挑一次自然的全量重建索引作边界。

  **它落地当天就抓到一个缺陷,而且不是它被设计来抓的那个。** 常量首次是在 Windows 上录的,而
  `write_index` 当时用 `write_text` 且未给 `newline=`,文本模式把结尾的 `\n` 译成 `\r\n`
  ⇒ **同一个索引在 Windows 与 Linux 上持久化出的字节不同**,而 `upstream_artifact_hashes` 记的正是
  这些字节 —— **平台相关的字节造出一个平台相关的守卫**,与本条目查了一整天的那个失败同族。
  本地绿、CI 红(`9daeaeff…` vs `51204c1e…`),混合平台的团队里它只会表现为又一个无从解释的哈希不匹配。
  已改 `write_bytes` 并加一条"持久化字节不得含 `\r`"的断言:**Linux 输出一字未变,集群上已建的索引
  全部保持原哈希**,变的只有 Windows。模块 docstring 自称 "deterministic",此前那句话是假的。

  **命令(登录节点,秒级):**
  ```
  python scripts/verify_legacy_index_identity.py runs/e2-gate-off
  PYTHONPATH=src python -m evidence_rag.evaluation.harm_cli --provenance runs/niah-injected/provenance.jsonl --selected-on runs/e2-gate-on-lenient-parent/selected_evidence_sets.jsonl --selected-off runs/e2-gate-on-lenient/selected_evidence_sets.jsonl --dataset-signature eb7760674bf3aace707acd932c67c86694d560510f949e3c73dea1ca7353db87
  PYTHONPATH=src python -m evidence_rag.evaluation.paired_metric_cli --on-report runs/e2-gate-on-lenient-parent/selector_report.json --off-report runs/e2-gate-on-lenient/selector_report.json --metric selector.core.conditional_document_recall
  ```
  `--metric` 必须是完整键名;本条目上方 (b) 的旧记法是简写,`_read_metric` 是精确查表,简写会静默返回全 None。

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

**同批另有三个 FAILED 从未进台账(2026-08-09 由 `sacct` 对账查出,补录):**
`18265521` / `18265522` / `18265523`,08-04 15:59–16:10 十分钟内三连,各 00:00:28 / 00:00:17 / 00:00:32,
exit 1:0,全在 `bp1-gpu035`;另有 `18264967` / `18264968`(00:01:30,已记)与 `18264969`(提交后即撤,已记)。
**三份日志(`logs/gate0b-1826552{1,2,3}.out`,各 4254 字节、大小逐字节相同 ⇒ 大概率同因)至今未读,
故此处只登记存在性,不写原因** —— 按本节相邻的 R013-smoke 纪律「FAIL 后先读日志再重交」,
写"原因未记录"正是 `18300333` 那次让根因推迟一整个提交周期的做法。读法:
`tail -40 logs/gate0b-18265521.out`。**GPU 计算量合计 77 秒,不影响 R012c 的任何读数**
(那三个 rung 的产物全部来自 18269630-32)。
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

### R011 — VitaminC adapter 与 revision-family 去污染 [DONE 2026-07-31;读数 2026-08-08 SUPERSEDED]

- **命令:** `python -m evidence_rag.cli.export_vitaminc --out-test data/gate0b/vitaminc_test.jsonl
  --removed-log data/gate0b/vitaminc_decontamination.json`(登录节点,纯 CPU,约 5 秒)
- **AFTER(实测)—— ⚠ 本表已于 2026-08-08 被 SUPERSEDED,保留不删,修正表在其下:**

| split | official | 去污染后 | 移除 |
|---|---:|---:|---:|
| train | 370653 | **369843** | 810 行 / **38 个 page** |
| validation | 63054 | **62984** | 70 行 / **2 个 page** |
| test | 55197 | **55197** | **0 —— official test 一行未动** ✅ |

- **SUPERSEDING 读数(2026-08-08,commit `c901310` 之后重导出):** 上表的量是在**原始字符串**
  这个等价关系下数的,而下游每一个泄漏检查用的是 `normalize_parent`(casefold + 折叠空白)。
  缺陷本身与三条轴的全量实测记在下一条目 R013-smoke。

| split | official | 去污染后 | 移除 |
|---|---:|---:|---:|
| train | 370653 | **369819** | **834 行 / 39 个 page** |
| validation | 63054 | **62984** | 70 行 / 2 个 page(**未变**) |
| test | 55197 | **55197** | **0 —— 仍逐行未动** ✅ |

- **差额 24 行 / 1 个 page:** 新增移除 train 侧的 `XXx-COLON- Return of Xander Cage`,
  因为 dev 保有同一页的另一种拼法 `XXX-COLON- Return of Xander Cage`。
  **dev 侧不变,而且这是可推导的、不是巧合:** 该页在 dev 的拼法不出现在 test 里,
  故它留在 `kept_dev`,让路的必然是 train 那侧。
- **本条的定性结论不变** —— official split 确有跨 split 的 revision-family 重叠,
  去污染不是形式主义,test-preserving 满足。**变的是量,以及量是在哪个等价关系下数的。**
- **三项预测在重导出之前写死,回来后逐条命中:** `n_dev` 不变 ✅ /
  `n_train` 低于 369843 ✅(369819)/ `removed_train_groups` 增至 39 条且新增项恰为该页 ✅。

- **移除的 dev page:** `John Frusciante`、`Linkin Park`(两者均出现在 official test 中)。
- **移除的 train page(38):** 含 `World War II`、`China`、`Aristotle`、`French Revolution`、`YouTube` 等。
  `Linkin Park` 同时出现在两份移除清单里 —— 与实现一致(它在 test 中,故 dev 和 train 都要清)。
- **判读:** VitaminC 的 official split **确实存在跨 split 的 revision-family 重叠**(38 个 page),
  虽然量小(train 0.22% / dev 0.11%)。这条去污染步骤不是形式主义,它真的删掉了东西;
  同时 official test 逐行未动,满足"test-preserving"的冻结要求。删除清单已存档
  (`data/gate0b/vitaminc_decontamination.json`,注意 `/data/` 被 gitignore,须 `git add -f`)。
- **未导出 train/dev 的去污染副本:** Gate 0B-1 只需 official test;去污染后的 train/dev 仅在
  §3.8 训练路径被启动时才需要,届时加 `--out-train` / `--out-dev` 重跑即可(确定性,可复现)。

### R013-smoke — §3.8 训练路径的首次执行尝试 [五次 FAIL、四种挂法;数据缺陷在第三、四次两度被拦,已修 2026-08-08]

**本条不是 R013。** R013 是三 seed 的正式 fine-tune;这里记的是冒烟(`--max-examples 2000`),
它只验代码能否在真实数据上走完一遍,**不产生任何可引用的读数**。

**五次 FAIL,五次都在任何训练步之前失败,GPU 计算为零:**

| job | 臂 | Elapsed | ExitCode | 原因 |
|---|---|---:|---|---|
| `18290519` | VitaminC 半 | 00:00:37 | 1:0 | `export_vitaminc` 尚未跑 ⇒ `data/gate0b/vitaminc_train.jsonl` 不存在 |
| `18290571` | 含域适配半 | 00:00:04 | 1:0 | 同上;`FileNotFoundError` 原文即该路径 |
| `18300333` | VitaminC 半 | 00:00:12 | 1:0 | ~~原因未记录~~ **2026-08-09 从日志补录:与 `18318996` 同因** —— `assert_decontaminated` 报 train∩dev 共享同一个页(`xxx-colon- return of xander cage`)。**即数据缺陷在第三次就被守卫拦下,当时日志未读,第四次原样再撞,根因定位晚了一整个提交周期。**"不得假定同因"按方法论是对的,如今由实测闭合。附带证据:它能走到该守卫 ⇒ 彼时 `export_vitaminc` 的产物已存在(pre-fix 版) |
| `18318996` | VitaminC 半 | 00:00:25 | 1:0 | `assert_decontaminated`:train 与 dev 共享 1 个页。**这一条是真实数据缺陷,详见「第四次拦出的是数据缺陷」一节** |
| `18319801` | VitaminC 半 | 00:01:37 | 1:0 | **基座不在 work 缓存**(HF_HOME 坑复发)。`LocalEntryNotFoundError` → `OSError`,栈顶是 `_construct_scaffold` 的 `CrossEncoder(...)`。**死因由物证确认而非推断**:blob 落盘 08-08 21:14:49–21:14:59,晚于本作业死亡(17:14:37 + 97s)约四小时 ⇒ 作业运行时该缓存确为空。**排除** `.no_exist` 负缓存解释——该目录是 HF 缓存的标准组成,其存在本身无异常,且离线 scaffold 检查随后通过 |

- **失败位置本身就是结果:** 五次全部死在训练开始之前,说明守卫的位置是对的 ——
  缺前提的代价是秒,不是一个 GPU 小时。**第五次走得最远**(97s):它过了 VitaminC 加载、
  `assert_decontaminated`(37 万行上首次通过,`c901310` 的修复在作业环境下随之得证)与 fold 规划,
  死在模型构造的第一行。**数据路径自此全绿。**
- **前三条此前一条都没进台账。** 按本文件规则,没有条目的运行不算已记录,故补录;
  `EXPERIMENT_TRACKER.md` 的 R013 行当时写的是"smoke 验证中",与实际不符,已同步改正。
  **第五条同样漏记过一轮:** 2026-08-08 的 `HANDOVER.md` 在正文里数了"五次",而本表与台账当时都只有四行,
  三份文档彼此不一致 —— 恰是 `HANDOVER.md` §0 自己禁止的"把事实读数复制进交接文档"所导致。已一并补齐。
- **第一、二次的前提已补齐(2026-08-08):** `export_vitaminc` 已跑,train 369843 / dev 62984 /
  test 55197 —— 该组数字随后被下节的修法 supersede(见 R011)。

#### 第五次的根因与两条由它带出的纪律 [2026-08-08]

**根因不是"忘了下载",是"预检与作业活在两个缓存里"。** 2026-08-06 的 §11.9 预检在**有网、未 export
`HF_HOME`** 的登录 shell 里跑,`from_pretrained` 于是**顺手把基座下到了 `~/.cache/huggingface`**,
六项全 PASS;而作业体把 `HF_HOME` 指向 `/user/work/$USER/hf_cache`,那里始终是空的。
**预检不是没做,是它自己制造了它所验证的前提** —— 又一例"产出合理数字而非崩溃",
且这次产出的不是数字,是一份绿色的 PASS。

⇒ **纪律一:预检必须在作业同款环境变量下执行。** 否则它验证的是另一个世界。
落地形式是先 `export HF_HOME=/user/work/$USER/hf_cache`,再跑 `scripts/a3_preflight.py`;
随后另跑一次带 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 的 `--scaffold-check-only`,
**它执行的正是作业里崩掉的那一行代码、在作业的离线约束下** —— 这是可证伪的读数,不是"命令没报错"。

**2026-08-08 补齐后的实测(登录节点,均 PASS):** §11.9(A) 六项全过;
基座权重指纹 `cross-encoder/nli-deberta-v3-base@c95d83f857fd4fcd`,**与 08-06 记录逐字相同**
⇒ work 缓存中这份与当初选型所测为同一份权重(指纹只证权重同一,不证当时位于哪个缓存,故不能替代上面的时间证据);
`id2label = {0: contradiction, 1: entailment, 2: neutral}`,待注册条目
`"cross-encoder/nli-deberta-v3-base": ('REFUTES', 'SUPPORTS', 'UNKNOWN')` —— **第三种顺序**
(albert 为 SUPPORTS/REFUTES/UNKNOWN,DeBERTa-large-mnli 为 SUPPORTS/UNKNOWN/REFUTES),按位置猜不会报错,
只会把每条边重新贴标签而下游每个数字看着都正常。离线 `--scaffold-check-only` 输出的 `label_order` 与之一致。

⇒ **纪律二(§11.9(B) 之外的新增前置):`--gres` 必须钉死支持 bf16 的卡。** 详见下节。

#### bf16 与硬件:`--gres=gpu:1` 在本集群上不可能跑通 [2026-08-08 实测,提交前发现]

**这一条在任何作业失败之前就成立,靠的是查 `sinfo` 而不是再挂一次。**

`sinfo -p gpu -N -o "%N %G"` 实测的 gpu 分区构成:`rtx_2080` 十三个节点(Turing,compute 7.5)、
`V100`/`v100` 三个(Volta,7.0)—— **两类均无 bf16**;支持 bf16 的只有 `bp1-gpu030` 的 `rtx_3090`
(Ampere 8.6)与 `bp1-gpu035` 的 `a100`(Ampere 8.0)。而 `--gres=gpu:1` 不挑卡:
**`18319801` 落的 `bp1-gpu002` 就是 rtx_2080**。

§3.8(b) 把 **bf16 冻进配方**且明写无合法调参面 ⇒ **不能改配方去迁就硬件,只能让硬件满足配方**。
在 Turing 上 `torch.cuda.is_bf16_supported()` 的行为取决于是否计入模拟路径:
**报错与"以模拟方式跑完"都是可能结局,而后者更坏** —— 它会产出一个 per-class F1 看着完全正常的模型,
而执行的数值路径与记录在案的配方不是一回事。**故不去赌它是哪一种,直接不落到那类卡上。**

**改动(本次一并落地):** `scripts/run_r013_train_relations.slurm` 的 `--gres=gpu:1` → **`--gres=gpu:a100:1`**,
与仓库其余九个重活脚本一致(`run_g5_*`、`run_g3_*`、`run_verified_generator` 等,G8 昨日即跑在 `bp1-gpu035`)。
**这是调度改动,不是配方改动** —— bf16 本身一字未动,§3.8(b) 不受影响,不需要开修订。

**代价必须同时记:** `HANDOVER.md` 原记"R013 用通用卡、不抢 a100、两条线可并行"**自此作废** ——
R013/R014/R015 与 G 模块的评分作业**将争同一个节点**。5 折 × 3 seed = 15 次全量 fine-tune,
排队压力须计入 9 月 4 日的日程。备选是 `--gres=gpu:rtx_3090:1`(同为 Ampere,仓库另有三个脚本在用),
**但该节点历史上长期 drain,选它须先看 `sinfo` 当时状态**。

#### 第三、四次拦出的是数据缺陷,不是配置失误 [根因已闭合;18300333 于 2026-08-09 补录归入]

**补录(2026-08-09):`18300333` 的日志与本节完全同因** —— 同一守卫、同一个页。缺陷首次被拦是第三次
提交,当时日志未读、台账记为"原因未记录";第四次(`18318996`)再撞后才定位。
⇒ **新纪律:FAIL 后先读日志再重交。** 守卫的收益 = 拦截 × 日志是否被读;
拦下而无人读,等价于把发现推迟一个提交周期 —— 本例的实价是一次排队加一次提交。

守卫报 train 与 dev 共享 1 个页(`page:xxx-colon- return of xander cage`)。
**它给的补救办法是错的** —— 它说"用 `export_vitaminc` 导出,别直接加载 `tals/vitaminc`",
而导出器已经用过:`_check_against_decontamination_log` 在守卫之前通过,
逐项对上了日志的 `n_train` / `n_dev`。再跑一次导出器不会有任何变化。

**根因:两层对"同一个页"用了不同的定义。**

| 位置 | 比什么 |
|---|---|
| `relations/vitaminc.py` `decontaminate` | **原始字符串** `pair.group` |
| `relations/training.py` `assert_decontaminated` | `page_key` = `page:` + `normalize_parent` |

`normalize_parent` = `" ".join(title.split()).casefold()`。同一个维基页面的两种拼法
(只差大小写或空白)在导出器眼里是**两个页、一个都不删**,在守卫眼里是**一个页、交集非空**。
**`page_key` 的 docstring 恰好写着这条** —— "comparing them raw would report no shared article
for two spellings of one — an audit that passes because it cannot see" ——
而 `decontaminate` 就是那个 raw comparison。**导出器犯了守卫的辅助函数为之而写的那个 bug。**

**三条轴全量实测(bp1 登录节点,秒级):**

| 轴 | 归一化后共享的页 |
|---|---|
| train ∩ dev | **1** —— train `XXx-COLON- Return of Xander Cage` / dev `XXX-COLON- Return of Xander Cage` |
| train ∩ **test** | **0** |
| dev ∩ **test** | **0** |

**评测面从未被污染。** R012 / R012b / R012c / R012d 一个数字都不动 —— 它们跑在 official test 上,
而 test 逐行原样导出、从不过滤。**但这条只在"test 从不被过滤"成立时才成立**,
且 `assert_decontaminated` 只查 train-vs-dev,**test 两条轴上没有守卫** ——
这次崩溃是撞上了有守卫的那条轴,不是把 bug 抓全了。零碰撞是实测,不是推论。

**修法(`c901310`):** `decontaminate` 改按 `normalize_parent` 分组,与 `page_key` 对齐。
**守卫不动** —— 放松成 raw 才是反方向。`removed_*_groups` 仍报原始拼写,
报归一化 key 会把造成删除的那个差异本身藏掉。
先写两条红测试(train/dev 轴与 test 轴各一)再改;逐字 CI:`ruff check src tests` 通过、
`mypy src tests/typecheck.py` 122 文件通过、`pytest` **1216 passed / 1 xfailed**。

**一次被证伪的部署预测,以及它为什么留在这里。** 修完后第一次重导出,
`n_train` 仍是 369843、`removed_train_groups` 仍是 38 条 —— **预注册的预测失败**。
原因不是根因链断了:**bp1 是另一个 checkout,改动还在本地未提交,跑的是旧代码**。
提交推送并在 bp1 拉取后重跑,三项预测全部命中(见 R011)。
**若当时没有把预测写死,"369843,和之前一样"会被自然读成"没影响,继续"**,
下一次提交会在同一个守卫上第五次挂掉。
⇒ **纪律:部署也要有可证伪的读数,不能只看命令有没有报错。**
- **过了再跑带域适配的那一半**(`--niah-manifest` / `--niah-provenance` / `--niah-parents` /
  `--sealed-dir runs/niah-sealed600` / `--niah-twin-label REFUTES`)。
  **那一次才是 Line A / Line B 接口的首次真实数据执行** —— sealed-600 零重叠检查、
  两侧同一个 `normalize_parent` 归一,在此之前全部只有 fixture 覆盖。

**跑完须人工确认的清单,三项已缩为一项(2026-08-08 更新):**

1. ~~fit-API 走哪条路~~ —— **提前有答案,日志里只做确认。** 实测 venv 装的是
   **sentence-transformers 5.5.1**,且 18319801 的 traceback 已显示 5.x 的模块结构
   (`base/model.py` / `_load_default_modules`)。`_train` 按**模块存在性**分派
   (`_optional_module("sentence_transformers.cross_encoder.trainer")`),5.5.1 上该模块存在
   ⇒ 走 `CrossEncoderTrainer`。日志判据:`[train_relations] fold 0: CrossEncoderTrainer API`。
2. ~~bf16 报不报错~~ —— **该岔路已由实测消解。** 原两难(装 ≥4 是环境变更 vs 改 §3.8(b) 是协议修订)
   的前提是"装的是 3.x",而实测是 5.5.1,早已 ≥4:`bf16=hyperparameters.bf16` 原样进
   `CrossEncoderTrainingArguments`,3.x 那段 `RuntimeError` 在本环境是死代码。
   **bf16 存活的风险换了位置——硬件**(gpu 分区多数卡无 bf16),已由 `--gres=gpu:a100:1` 钉死,
   见上方「bf16 与硬件」一节。
3. ~~分类头重初始化警告~~ —— **无须人工看。** `_construct_scaffold` 在
   `CrossEncoder(base, num_labels=3)` 之后回读 `id2label` 并送进 `derive_label_order`,
   名字一旦丢成 `LABEL_0/1/2` 即硬失败,且明确拒绝就地补回名字(补回去的顺序是断言,不是读数)。
   §11.6 的前提因此是机制而非约定。

**重交命令(gres 已钉 a100 之后):**
`sbatch scripts/run_r013_train_relations.slurm 13 runs/r013/smoke "--max-examples 2000"`。
交前按「第五次的根因」一节的纪律一走一遍作业同款 env 的预检;
**与 g5-score 争 a100 的代价已在「bf16 与硬件」一节记账。**

#### 第六次提交与 a100 队列的实测代价 [2026-08-08 深夜]

**六次提交前的状态:已知死法全部排除** —— 上表五行(2026-08-09 补录后实为**四种**挂法:
导出缺失 ×2、decontaminate 缺陷 ×2、缓存 ×1),另有一种(`CrossEncoderTrainer` 路径的
`datasets` / `accelerate` 未装)于提交前在登录节点实测排除:**datasets 4.8.5 / accelerate 1.13.0**。
bp1 已 fast-forward 到 `02b08a4` 并**逐字核对 `131:#SBATCH --gres=gpu:a100:1`**
—— 该核对不是形式:`c901310` 那次"bp1 是另一个 checkout、跑的是旧代码"就发生在同一个位置。

**a100 排队实测(这组数字要进止损材料):**

| 观测 | 值 |
|---|---|
| bp1-gpu035 的 a100 | **12 张全满**(`AllocTRES gres/gpu=12`),`State=MIXED+PLANNED` |
| 占用者的 walltime 上限 | wf23947 ×3 已跑 4 天 / 上限 **14 天**;ea23166 ×3 / 上限 7.5 天;qh23464 ×6 / 上限 3.5 天 |
| 调度器预计开跑 | g5-score **2026-08-12T00:29**(约 3 天后);r013 **N/A**(排在其后,估不出) |

⇒ **§3.8 的 15 次全量 fine-tune(5 折 × 3 seed)只能在这一个节点上排。** 单是本次冒烟就见到三天的
等待,而 R013–R015 是十五次。**这不是资源不足的抱怨,是 9 月 4 日交付日程的一个硬输入** ——
`HANDOVER.md` §7 第 3 条(止损线)的裁决须把它计入:主轮 + 预注册应急臂(同族 large 再跑三 seed)
在这个队列上排不下。

**两条缓解手段,均为纯调度、不动任何语义:**

1. **把 pending 作业的 `TimeLimit` 降到真实量级** —— `scontrol update jobid=<id> TimeLimit=02:00:00`,
   不需重交、不丢队龄(只许降不许升)。冒烟申请的 24h 挤不进任何 backfill 空档,2h 到处都能塞;
   g5-score 由 4h 降到 1h(历史耗时 16–20 分钟,见 G7/G8)。
2. **rtx_3090 对冲** —— gpu030 同为 Ampere(8.6),bf16 齐全,**故走哪张卡都不违反 §3.8(b)**。
   本次实测它是 `alloc` 而非历史上的 drain,且占用者中有 2h 上限的数组任务
   ⇒ **周转是小时级,而 a100 是天级**。已交 `18321139`(`--gres=gpu:rtx_3090:1 --time=02:00:00`,
   输出目录另设 `runs/r013/smoke-3090` 以免两份写同一处)。

**纪律(第 4 条的同构情形):任一份进入 RUNNING 即 `scancel` 另一份。**
两份都自称冒烟、跑在不同型号的卡上,同时跑完则台账里的作业号不再唯一对应一次执行。
冒烟虽不产生可引用读数,该对应关系仍须成立。**a100 那份(`18321128`)在被取消前
同时充当正式 R013 的排队实测,其等待时间本身就是上表那个数字的来源。**

---

### R012f — 跨骨干一致性(RADAR 消融的复现)[2026-08-09,BEFORE 已写,登录节点级]

**为什么做:** [related-work.md](selector/related-work.md) §1 记载 RADAR(arXiv 2605.22041)换三个 NLI 骨干
只见 "minor changes",据此可推断关系模型不是承重墙 —— 若在我们数据上也成立,§3.8 这两周问错了问题。
**本轮不需要 GPU:R012/R012b/R012c 的 dump 已存在,这是对已有数据的重新提问。**

**第一部分:已可从仓库内 `results/gate0b/sweep-full.json` 读出(R012,job 18235972,template rung,
同一作业内比较,不跨作业):**

| 骨干 | VitaminC official test macro-F1 | 探针 `gold_supports_recall` | `twin_refutes_accuracy` | `unknown_rate` |
|---|---:|---:|---:|---:|
| `tals/albert-xlarge-vitaminc-mnli` | **0.9215** | **0.1916** | 0.6736 | 0.4823 |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 0.7582 | **0.7942** | 0.6376 | 0.1758 |

- **换骨干使 `gold_supports_recall` 移动 60.3pp** —— 与 "minor changes" 不相容。
- **排名反转:** albert 在官方 test 上领先 **16.3** macro-F1 点,在最小编辑探针上落后 **60.3** recall 点。
  **同作业、同代码、同批对**,故该反转不受跨作业漂移影响(第 4 条纪律)。
  这与 related-work §3 的 SummEdits 证据同向:**榜单名次不向最小编辑 regime 迁移。**
- **两个指标的敏感度差一个数量级:** twin 只差 **3.6pp**,gold_supports 差 **60.3pp**。
  换言之"骨干无关紧要"这个判断在一个指标上近似成立、在另一个上灾难性失败。
- **诚实的限定,必须与上述数字同时陈述:** (1) RADAR 换的是三个**通用 MNLI 家族**模型
  (DeBERTa-v3 / BART / ModernBERT),我们换的是**三个不同族**(通用 NLI / VitaminC 专训 / 任务专训),
  模型多样性更宽;(2) RADAR 测的是**端到端 Acc 与 ASR**,我们测的是**部件级指标**,聚合层可以吸收部件方差。
  **⇒ 本轮证伪的是"部件级低敏感",不是 RADAR 的流水线级主张。** 后者需要带门的 dev 运行(R031/R032 量级)。
- 加入 MiniCheck 臂(template rung .5727,R012c)会**跨作业**(18269630-32),故三臂极差只作参考,
  两臂结论以同作业为准。

**第二部分(待跑):逐对一致性 —— 聚合率答不了的那个问题。**
两个分类器可以有相同的率而在每一对上都不一致。dump 带 `premise_hash` / `hypothesis_hash`,可精确 join。

**预注册的两个互斥假设(跑之前写死):**

- **H-NEST:** 两臂的 SUPPORTS 集**嵌套**(albert ⊂ DeBERTa),`gold_only_albert ≈ 0`。
  ⇒ 骨干只是一个保守度旋钮,RADAR 的读法在部件级也成立,60.3pp 只是同一条曲线上的两个点。
- **H-CROSS:** 两臂**交叉**(各自捞到对方漏掉的 gold 对),`gold_only_albert > 0` 且 `gold_only_deberta > 0`。
  ⇒ 它们是真正不同的分类器,骨干是承重的;**并且 union-of-SUPPORTS 值得测**——
  那是绕过 Gate 0B 最廉价的可能路径:零训练、零新 checkpoint。

**预测(署名写下):** 倾向 **H-CROSS**,依据是 albert 的 `unknown_rate` .48 对 DeBERTa 的 .18 ——
若纯粹是保守度差异,albert 的 UNKNOWN 应当均匀覆盖 DeBERTa 的 SUPPORTS,而 albert 在官方 test 上
supports_recall 高达 .9508,说明它并非一律保守,而是**在这个任务形式上**塌陷。若 H-CROSS 成立,
预测 union 的 `gold_supports_recall` 落在 **.80–.88**、`twin_not_supported_accuracy` 落在 **.45–.60**
—— 即**很可能买到 recall 却买不起 twin**,这正是 related-work §8 记载文献从未测过的交换率。

**⚠ union 若同时过两个阈值也不构成 Gate 0B 通过** —— 它是未预注册的臂,要用须另开修订。本轮是诊断。

**工具(先红后绿,11 个测试):** `relations/backbone_agreement.py`(纯函数:按 hash join、
拒绝重复键、拒绝无交集、raw agreement + Cohen's kappa、nested/crossing 判定、union 两指标)
+ `scripts/r012f_backbone_agreement.py`(登录节点 CLI)。
**守卫两条,都是本项目的老病:** 按 hash 而非按位置 join(位置 zip 会从一个无意义的连接产出一个可信的数字);
两臂无共同对时**拒绝**而不是在交集上算(那多半意味着两次 sweep 之间探针被重建过)。
kappa 与 raw agreement 并列报告,因为 albert 的 UNKNOWN 占 .48,单看 raw agreement 会被多数类抬高。

**运行(bp1 登录节点,秒级):**
`export PYTHONPATH=src && python scripts/r012f_backbone_agreement.py results/gate0b/dump-template.jsonl`

**⚠ 前置问题:R012b/R012c/R012d 的结果包与 dump 从未提交** —— 仓库里只有 `results/gate0b/sweep-full.json`
一个文件。按本文件"只以 `.out` 或散落文件存在的结果不算已记录"的规则,**那几轮的结果包欠着**,
且本分析所需的 dump 目前只在 bp1 上。**跑本轮时一并 `git add -f` 补交。**

**AFTER(bp1 登录节点,`dump-template.jsonl`,2026-08-09):**

```
per arm         gold_supports_recall   twin_not_supported
DeBERTa                    0.7942 (n=1472)        0.8689 (n=2944)
albert                     0.1916 (n=1472)        0.9980 (n=2944)
pairwise        n_common 5888 (only-left 0, only-right 0)
                raw_agreement 0.4969   cohen_kappa 0.2906
                gold recovered only by albert 6, only by DeBERTa 893  => CROSSING
union           gold_supports_recall 0.7982   twin_not_supported 0.8689
```

**读数一(锚点通过):** 两臂 `gold_supports_recall` 逐字复现 sweep 自己发表的 .7942 / .1916,
且两臂覆盖同一批 5888 对、无单边缺失 ⇒ join 正确,后续统计有意义。

**读数二(RADAR 的问题,部件级答案):** **两臂在半数对上就不一致** ——
raw agreement **.4969**、Cohen's kappa **.2906**。合并 60.3pp 的率差,
**"换骨干只带来 minor changes"在部件级被明确证伪**(限定仍如 BEFORE 所述:族更宽、且我们测部件不测端到端)。

**读数三(集成逃生口 —— 关闭,这是本轮最有用的负结果):**
判定确为 **CROSSING**,但**极不对称:DeBERTa 独捞 893 对,albert 独捞 6 对,比例 149:1** ——
形式上交叉,实质上近乎嵌套。**union 只把 recall 从 .7942 抬到 .7982(+0.4pp),离 .85 还差 5.2pp,
且 twin 与 DeBERTa 一字不差(.8689)。**
⇒ **"用已跑过的零训练模型做并集绕过 Gate 0B"这条路当场关闭,代价是零 GPU。**
**精确的刻画:两臂分歧极大(κ=.29)但分歧是单向的**(DeBERTa 判 SUPPORTS 处 albert 判 UNKNOWN),
即它们不是互补的能力,而是**同一条保守度轴上的两点**。这两句话必须并列,否则任一句单独看都误导。

**⚠ 预注册预测被部分证伪,如实记录(BEFORE 由本轮执行者署名写下):**

| 预测 | 实测 | 判定 |
|---|---|---|
| H-CROSS 成立 | crossing = True | **命中**(但 149:1 的不对称未被预见,使"命中"在实质上误导) |
| union recall 落在 .80–.88 | **.7982** | **落在区间外**(低 0.2pp),按字面判 **MISS** |
| union twin 落在 .45–.60 | **.8689** | **大错,偏离 27pp** |

**错因已定位:** 预测假定 albert 会把相当多 twin 判成 SUPPORTS,实测它 **99.8%** 的 twin 都不判 SUPPORTS
—— albert 不是"另一种判法",是**几乎不说 SUPPORTS**(全局 unknown_rate .48)。
**这正是本项目反复记录的那类错误的又一例:我用一个聚合率(.6736 的旧 twin 数)去推断逐对行为,
而聚合率恰恰是本轮设计出来要绕开的东西。** 教训:**在做逐对分析之前,不要用聚合率做逐对预测。**

#### ⚠ 由本轮牵出的一个指标口径不一致 —— 影响已发表判读的表述,须核实

**我算出的 twin 与台账所记不是同一个指标。** `relations/gate0b.py::task_report` 的门指标是
`p is not SUPPORTS`(**REFUTES ∪ UNKNOWN 皆算 not-supported**);而 `sweep-full.json` 里的字段叫
`twin_refutes_accuracy`(.6376 / .6736),是**要求 predicted == REFUTES 的严格版**,
该字段名在现行代码中**已不存在** —— 它是 A1 之前的产物。

⇒ **在现行(A1/A2)判读下,template rung 上两臂的 twin 都远超 .70 阈值**(.8689 / .9980),
而台账 R012 行记的是 .638 / .674。**§9.10a 明文要求 R012/R012b 从 dump 重算而非重跑,
工具亦已存在(`cli/recompute_binary.py`),但重算结果似未全部回写台账。**

**若核实成立,需要改写的表述(结论不变,FAIL 仍是 FAIL —— 门是合取,`gold_supports_recall` 处处不过):**

- R012b 的「**两个指标反向**」与「**六格无一两项同时过**」可能是读**严格 REFUTES 版**造成的伪影;
- 更强也更简洁的正确表述可能是:**九格全 FAIL 是单一原因 —— 只有 `gold_supports_recall` 在挂;
  这些模型拒斥孪生没有问题,问题在识别真实支持。** 这个表述对报告更有利,但**必须先核实再用**。

**核实命令(bp1,秒级,用 §9.10a 的官方工具,自带 `--against` 交叉校验):**
`python -m evidence_rag.cli.recompute_binary --dump results/gate0b/dump-template.jsonl --against results/gate0b/sweep-full.json`

**在核实之前,不得据此改写任何已发表判读。** rung 2 / rung 3 与 MiniCheck 臂需各自的 dump 重算,
本轮只看了 template。

#### 核实结果:成立。§9.10a 官方工具的输出 [2026-08-09,bp1 登录节点]

`python -m evidence_rag.cli.recompute_binary --dump results/gate0b/dump-template.jsonl --against results/gate0b/sweep-full.json`

| 臂 | `gold_supports_recall` | `twin_not_supported_accuracy` | `unknown_rate` | `failures`(工具自判) |
|---|---:|---:|---:|---|
| DeBERTa-v3-large-mnli-… | 0.7942 | **0.8689** | 0.1758 | **`["gold_supports_recall"]`** |
| albert-xlarge-vitaminc-mnli | 0.1916 | **0.9980** | 0.4823 | **`["gold_supports_recall"]`** |

`--against` 未报错 ⇒ `gold_supports_recall` 逐字复现该 dump 自己 sweep 发表的值,§9.10a 的不变式满足;
`unknown_rate` 亦与 `sweep-full.json` 一致。**`failures` 是工具自己算的,不是解读:两臂都只挂一项。**

**⇒ 确立(限 template rung、R012 两臂):在现行(A1/A2)门判读下,twin 从不是失败原因。**
台账 R012 行所记 `.638 / .674` 是 **A1 之前要求 `== REFUTES` 的严格版**,现予以**并列标注而非覆盖**
(照 R011 的先例:被取代的读数留在原处并注明)。

**Gate 0B 的 FAIL 判定不变** —— 门是合取,`gold_supports_recall` 处处不过。
**族级断言不变。** 改变的是**刻画**,而且新刻画更简洁也更有力:
**这些模型拒斥孪生没有问题;它们失败在识别真实支持。**

**一个对 §3.8 直接可用的推论 —— 交换率有余量,这一点此前被旧口径掩盖了:**
DeBERTa 需要 **+5.6pp** recall(.7942 → .85),而它在 twin 轴上距阈值有 **18.9pp 余量**(.8689 → .70)。
按 [related-work.md](selector/related-work.md) §5 对 logit adjustment 的量级估计(recall:twin 约 1:0.5–1:1),
换到 +5.6pp recall 的代价约 2.8–5.6pp twin,落点 **.813–.841,仍远在 .70 之上**。
**⇒ 以 twin 余量换 recall 是负担得起的**,而按旧口径(.638 对阈值 .70)看,twin 本身就在失败、毫无余量可换。
**这条改变了训练杠杆的取舍算式,故必须写下。**(仍是量级推演,不是实测;真实交换率须由 §3.8 的训练读数给出。)

**仍未做,且必须做完才能改写"六格/九格"的表述:** rung 2 / rung 3 与 MiniCheck 三臂各自的 dump 重算。
R012b 的「**两个指标反向**」「**六格无一两项同时过**」与 R012c 的「**九格无一过门**」目前都建立在旧口径上;
**FAIL 结论不会变,但"因为什么而 FAIL"很可能要改写。** 命令同上,换 `--dump` 与 `--against` 即可。

#### 全格重算完成:九格 + surface 两格,失败轴处处单一 [2026-08-09,bp1 登录节点]

七份 dump 全部过 `recompute_binary`(现行门口径,`not SUPPORTS` = REFUTES∪UNKNOWN):

| rung | albert(gold/twin) | DeBERTa(gold/twin) | MiniCheck(gold/twin) |
|---|---|---|---|
| template | .1916 / .9980 | .7942 / .8689 | .5727 / .9463 |
| qa | .3635 / .9871 | .6651 / .9008 | .5883 / .9446 |
| qa2d | .5360 / .9586 | .5870 / .9222 | .5360 / .9375 |
| surface(R012d 对照) | .1984 / .9983 | .7792 / .8635 | — |

**工具自判的 `failures` 在全部 11 格里都是且仅是 `["gold_supports_recall"]`。
twin 全域最低 .8689,距 .70 阈值 18.9pp;全域最高 .9980。**

**三条表述的改写自此有据(原读数按 R011 先例并列保留):**

1. R012 的「twin .674/.638」→ 严格 REFUTES 口径的历史读数;门口径下 twin 两臂均大幅通过。
2. R012b 的「**两个指标反向**」「六格无一两项同时过」→ **旧口径伪影确认**。现行口径下
   albert twin 沿阶梯 .9980→.9871→.9586、DeBERTa .8689→.9008→.9222 —— 方向仍相反但量级仅 4~5pp
   且全程远离阈值;旧口径下 albert rung 3 的"崩到 .5312"在门口径下不存在(实为 .9586)。
   **阶梯是一个 recall 现象,不是 twin 现象。**
3. R012c 的「九格无一过门」→ 保持成立,但**原因单一化:九格全因 `gold_supports_recall` 失败**。
   MiniCheck 的 twin(.9375–.9463)本就按二分类口径计,不受本次订正影响。

**报告级的正确刻画(取代"两项都难"的旧叙事):**
**零训练族在拒斥孪生上没有问题 —— 十一格里最差也有 .8689;它们全部失败在识别真实支持。**
Gate 0B 实际上是单轴问题:把 `gold_supports_recall` 抬 5.6pp,而 twin 轴处处有 ≥18.9pp 的可交换余量。

**顺带的量级观察(与 R012b 阶梯同等的跨作业限定):** 形式阶梯对两臂的交换率悬殊 ——
albert template→qa2d 用 3.9pp twin 换了 34.4pp recall(约 9:1),DeBERTa 反向用 20.7pp recall
换了 5.3pp twin(约 4:1)。**qa2d 之下两个通用 NLI 臂收敛到几乎同一点**(.536/.587 recall,
unknown_rate .35/.37)—— 形式移除后剩下的差距才是族的真实能力差。

**下一个十秒问题(直接决定 §3.8 的杠杆选型):** gold 漏掉的那部分去了哪 —— UNKNOWN 还是 REFUTES?
若主要落 UNKNOWN ⇒ related-work §5 的训练期 logit adjustment(压 NEI 先验)正中要害;
若主要落 REFUTES ⇒ 模型在**积极地不相信**真实支持,那是另一个病,logit adjustment 药不对症。
dump 里有答案,一条循环即可:按 `kind=="needle_gold"` 取行,数 `predicted` 的分布。

#### gold-miss 去向分解 + NEI 压制的可满足性曲线 [2026-08-09,本地,dump 已入库故可复算]

**分解(gold 行 n=1472,miss 的去向):**

| rung | DeBERTa: miss(UNKNOWN/REFUTES) | albert: miss(UNKNOWN/REFUTES) |
|---|---|---|
| template | 303(**173**/130),57% 落 UNKNOWN | 1190(**943**/247),79% 落 UNKNOWN |
| qa | 493(**347**/146),70% | 937(**531**/406),57% |
| qa2d | 608(**458**/150),75% | 683(**461**/222),68% |
| surface | 325(**193**/132),59% | 1180(**944**/236),80% |

(MiniCheck 为二分类臂,miss 全落 NOT_SUPPORTED,无此分解。)

**两个立即可读的事实:**

1. **UNKNOWN 是主要去向,且池子够大:** DeBERTa template 到 .85 需再翻 **83** 个 gold 行,
   而 UNKNOWN-miss 池有 **173** 个 —— 需求的 2.1 倍。§10.11 预registered 的"UNKNOWN 质量堆积"
   风险方向正确,且现在有了逐行的量。
2. **DeBERTa 的 REFUTES-miss 对形式近似不变**(130/146/150/132),而 UNKNOWN-miss 随形式翻倍再翻倍
   (173/347/458)⇒ **形式敏感性几乎完全是一个 UNKNOWN 现象** —— R012b 的阶梯移动的是弃权质量,
   不是"积极不信"。(albert 侧该分离较弱,如实记。)

**可满足性模拟(DeBERTa template,对 dump 里的既有概率做"UNKNOWN logit 减 s"的重判):**

| s(nat) | recall | twin | 联合门(.85 ∧ .70) |
|---:|---:|---:|---|
| 0.00 | .7942 | .8689 | |
| 1.00 | .8132 | .8590 | |
| 2.00 | .8322 | .8461 | |
| 2.50 | .8397 | .8390 | |
| **3.00** | **.8505** | **.8322** | **PASS**(恰 1252/1472,压线一对) |
| ∞(极限) | .8723 | .8033 | PASS |

**判读,按重要性排:**

1. **联合门在零训练 DeBERTa 的概率排序内部是可满足的。** 过门所需的判别信息**已经存在**,
   只是决策几何偏向 NEI —— **Gate 0B 对这个臂是校准难,不是判别难**。这重述了 §3.8 要完成的任务:
   微调至少只需重摆类先验,而非教会模型新的区分。
2. **实测交换率 ≈ 1 : 0.65**(s 0→3:recall +5.6pp,twin −3.7pp),落在 related-work §5 文献猜测
   (1:0.5–1:1)之内 —— 那条猜测现在有了本任务上的实测锚点。twin 全程离 .70 阈值 ≥10pp,
   **约束自始至终只有 recall 一条**。
3. **不可触及的残余 = 188 行(12.8%)**:130 个直接 REFUTES-miss + 58 个 REFUTES 为次选的 UNKNOWN-miss
   —— 任何 NEI 压制都救不了它们,极限 .8723 正由此而来(1 − 188/1472)。这部分才需要真正的训练/形式工作。
   余量意义:极限比阈值高 2.2pp,**紧但非零**。
4. albert 不适用此杠杆:即使全部 UNKNOWN-miss 翻转也远不及 .85(其塌陷不是先验问题)。

**协议边界,一次说死(比结果更重要):**

- **本模拟的干预本身是被禁止的**:推理期 logit 平移 = 换名字的阈值(§2.4 / §9.5a / related-work §5 的
  同一结论)。**上表不构成任何 Gate 0B 读数,PASS 一词只描述可满足性。**
- 它的合法用途只有一个:作为**训练期** logit adjustment(推理仍是纯 argmax,协议干净)的可行性诊断。
- **s=3 对应的声明先验是激进的**(π_NEI 被压约 e³≈20 倍),且训练会移动表征,模拟与训练后读数
  不存在 1:1 映射 —— 这条曲线证明的是"决策几何有余地",不是"τ 该取 3"。
- **披露义务(A1 同款):** 本诊断消费了探针读数。若未来某修订(A5?)以此曲线为动机把 logit adjustment
  写进配方,必须披露动机来源于探针侧诊断,并依赖训练后 OOF/门读数作为出样检验。

---

### NIAH split 的可核验性 —— **冻结要求无法核验,须裁决** [2026-08-09,R011b 核验中发现]

**⚠ 本条不是事故报告,是一个在造成损害之前抓到的缺陷。已发表读数无一受影响;
它影响的是尚未跑过一步的 R013 域适配半。域适配那一半在裁决前不得提交。**

**起因:** 关 R011b 那笔账,须核验冻结要求「0B-2 探针必须建在 NIAH **train** split 上,不得用 dev」。
先确认了 `runs/niah-train/` **确实存在**(此前连存在性都未知,故该要求"无法仅凭文档核验")
—— 它的存在同时解释了文档为何写错路径:错误路径不是虚构的,是一个真实存在的邻居目录。
随后做 ID 级核验,得到的结果不是通过或不通过,而是**这个核验做不了**。

**实测(bp1 登录节点,秒级,读三份 manifest 的 `queries_file` 取 `query_id` 集合):**

| 运行目录 | manifest `split` | `dataset_id` | n_queries |
|---|---|---|---:|
| `runs/niah-train` | **`'dev'`** | `niah/dpr-w100-nq` | 2000 |
| `runs/niah-train-injected` | **`'dev'`** | `niah/dpr-w100-nq` | 2000 |
| `runs/niah-injected`(dev 侧评测用) | `'dev'` | `niah/dpr-w100-nq` | 2000 |

- `niah-train-injected` ⊂ `niah-train`:**True**,且双向差集均为 **0** ⇒ 两者 query 集合**完全相同**。
- `niah-train-injected` ∩ `niah-injected` = **202**;`niah-train` ∩ `niah-injected` = **202**。

**判读(两个错误读法都要避开):**

1. **不能读作"通过"** —— 三个不同用途的运行 `split` 字段同为 `'dev'`,该字段显然是构建脚本的默认值、
   从未被正确填过。**它不承载真实语义,故「探针建在 train 上」这条冻结要求依旧无法核验**,
   只是这次知道了原因:核验所依赖的字段本身是空壳。
2. **也不能读作"dev 被污染"** —— 三边各 2000、`train` 与 `train-injected` 逐个相同,
   说明它们是从同一个 2000 池子里抽的样,202 是抽样重叠。**问题不是划分被违反,是划分从来没做。**

**为何仍然严重:** §3.8 的整条推理建立在训练面与校准面不相交上,而 0B-2 探针的题与 dev 侧评测的题
有 202 个重合。**且 NIAH 这条轴上没有守卫:** `relations/niah_adaptation.py` 检查的是
「域适配对的 parent page 与 sealed-600 零重叠」(防的是**评测集**泄漏),
**没有任何代码检查「探针 query 与 dev 评测 query 不相交」** —— 与 VitaminC 轴上的
`assert_decontaminated` 不对称。若未查,R013 会正常训完、产出模型、per-class F1 完全正常,
而它训过 202 道随后要用于评测的题。**这是本文档第 4 节那句话的第七个实例,且是代价最高的一类:
不崩溃、不报错、每个数字都在合理区间。**

**影响面(已逐项核对,不外推):**

- **不受影响:** R012 / R012b / R012c / R012d 的 **0B-1** 读数跑在 VitaminC official test 上,与 NIAH 无关;
  其 **0B-2** 读数是**零训练**模型的诊断 —— 没有训练,就不存在"训练面见过校准面"。
  R001 / R001b 跑在 `runs/niah-injected` 上,是评测面自身。
- **受影响:** **R013–R015 的域适配那一半**(尚未跑过一步)。**VitaminC 主训那一半不受影响**,
  冒烟与不带 NIAH flag 的正式训练可照常进行。

**处置:本条属协议层,不由操作员就地决定。** 三种可能的裁决方向(不在此预判):
(a) 真正建立 NIAH train/dev 划分并重建探针;(b) 论证 202 例重合在本任务下不构成泄漏并写入协议;
(c) 域适配半退出本轮,R013 只训 VitaminC。**在裁决落地之前:域适配半不提交,
`smoke-niah` 一并暂停。** 并建议无论裁决为何,都补一条与 `assert_decontaminated` 对称的 NIAH 轴守卫
—— 按本项目第 1 条工作方式,写成纪律而无代码强制者迟早被跳过。

**R011b 的状态因此不变(仍 TODO)**:存在性已确认、rung 2 的三个数(`n_pairs 5888 /
n_records 1472 / n_skipped_records 0`)已于 2026-08-03 重导对上,但**冻结要求本身尚未获得可核验的依据**。

#### 裁决记录 [2026-08-09,签署于任何 R013 训练数字存在之前]

**四条待裁,三条已裁、一条明确暂缓。时间戳是本记录的要点:全部条款先于第一个训练读数落定。**
正式修订文本(A4)待按 §9/§10/§11 的模板起草入 M0;本记录先行钉住内容与时间。

1. **bf16 岔路 —— 注销,非裁决。** 前提("集群是 sentence-transformers 3.x")被实测证伪:
   venv 为 **5.5.1**,`CrossEncoderTrainer` 路径存在、bf16 原生。关闭证据 = 冒烟日志中的
   `[train_relations] fold N: CrossEncoderTrainer API` 行,出现后回填于此。
   **✅ 已回填(2026-08-09):job 18321128 五折全部打印该行,a100 上 bf16 无异常,本条正式关闭。**
2. **official test 一次性额度 —— 按「门 × 模型族」计。** R012(job 18235972)消耗的是
   **零训练族在 Gate 0B-1** 的那一次;**训练所得模型在 R020(M2 关系门)保有其一次**。
   依据:两者都是 M0 预注册的独立门(run table 同时排有 R012 与 R020);§3.6 所防的
   "failed 后调参再重测"模式在此不存在 —— 配方无调参面(flag 已删)、基座选型测量面是 dev
   (§11.2a)、五项阈值冻结于 R012 之前且 A2 逐字未动,test → R013 模型构造的信息通路被机制堵死。
   **三个生效条件:(i)** 训练模型的那一次跑完即终局,任何结果都不得重跑;**(ii)** 报告明文披露
   official test 共被访问两次(哪两个门、何日);**(iii)** 本条在 R020 上场之前进入正式修订文本。
3. **止损线 —— 2026-08-09 决定暂缓,此为有意决定而非遗忘。** 队列实测(a100 三天/空位)已入材料;
   延迟的代价随日历递增,后续每次触及此条须在台账留痕。
4. **NIAH split —— 采纳「剔除加守卫」。** 要点:**(a)** 域适配训练对只取与 dev 评测运行
   (`runs/niah-injected`)零重合的 query family,重合 family 从训练面剔除
   (**计数订正 2026-08-09:query 级重合 202/2000,其中有记录、实际构成 family 的为 151/1472**,
   保留 1321 ≈ 89.7% 适配信号;原记"202 个 family"把 query 数误作 family 数);
   **(b)** 探针不重造 —— 探针是评测件,池内循环性由 5-fold 按 parent page + synthetic family
   分组的 OOF 处理;**(c)** 新增与 `assert_decontaminated` 对称的 NIAH 轴守卫,
   **守卫比对 query ID 集合、不读 `split` 标签**(本次事实已证标签不承载语义:标签会说谎,集合不会);
   **(d)** 本条构成 §3.8 配方修改 ⇒ **正式修订(A4)与守卫代码落地之前,`smoke-niah` 与
   域适配半保持暂停;VitaminC 主训半不受影响**。已排队的两份冒烟(18321128 / 18321139)
   均为 VitaminC 半,不因本裁决取消。

**执行进度(同日):** 守卫代码 + 红测试已落地 —— `relations/niah_adaptation.py` 新增
`DevEvaluationQueries` / `load_dev_queries` / `partition_dev_overlap`,`build_niah_examples`
对未剔除的相交记录拒绝而非静默过滤;`cli/train_relations.py` 新增必填 `--niah-dev-manifest`
(域适配 flag 集合五→六,仍全有或全无),manifest 记录 dev 集身份与 `n_families_excluded_dev_overlap`;
slurm 头部同步。逐字 CI 全树:ruff 干净、mypy 122 文件干净、pytest **1245 passed / 1 xfailed**
(含并发合入的 G9 代码)。**A4 正式文本已入 M0 §12;同日批准,协议升 `g2-proto-5`,域适配半暂停解除。**

#### 第七次提交:COMPLETED —— 训练路径首次在真实数据上走通 [job `18321128`,2026-08-09]

| 项 | 值 |
|---|---|
| State / Exit | **COMPLETED / 0:0**(sacct 按 id 直查) |
| Elapsed | **00:03:31**(2000 例 × 5 折 + 5 次模型加载 + OOF,量级符合预期) |
| 节点 | bp1-gpu035(a100 ⇒ bf16 硬件无碍) |
| 启动方式 | **backfill 兑现**:g5-score(18318915)04:31:39 FAILED 释放卡,本作业 **04:31:40** 接上 —— 间隔 1 秒。TimeLimit 降到 1h/2h 那次操作的直接收益,比调度器原估的 08-12 提前约三天 |

六种已知死法全部在提交前排除,第七次一次通过。**对冲副本 18321139 按第 4 条纪律的同构规则撤销**
(赛跑结束,保留者 = 18321128;smoke-niah 18322821 是独立作业,不在此规则内,继续排队)。

**验证读数(2026-08-09 回填,两项全过):**
1. **五折全部打印 `[train_relations] fold N: CrossEncoderTrainer API`**(fold 0–4),
   前置 `CUDA: True | torch 2.5.1+cu121` —— fit-API 走 v4/v5 路径实锤,
   **裁决 1(bf16 岔路注销)的关闭证据就此到位**;
2. manifest:`is_smoke_run True | n_examples 2000 | protocol g2-proto-5 | niah not run` ——
   四字段逐字命中预期;`g2-proto-5` 说明作业跑在 A4 批准后的 checkout 上。
3. 对冲副本 **18321139 已 scancel**(在其起跑前),台账里"冒烟"唯一对应 18321128 一次执行。

**冒烟不是 R013**:不产生任何可引用读数;`runs/r013/smoke/oof_predictions.jsonl` 是训练数据上的诊断。
下一步:smoke-niah(18322821,在队)→ 两个冒烟都过 ⇒ 提交正式 R013/R014/R015(六 flag 全量,无 `--max-examples`)。

顺带:`18318915`(g5-score)31 秒 FAILED —— G 线作业,前提类失败形态,已提醒其负责人自查日志。

#### smoke-niah BEFORE(job `18322821`,提交于 2026-08-09,rtx_3090 / 2h 上限)

**六 flag 全量命令**(含 `--niah-dev-manifest runs/niah-injected/manifest.json`),
`--max-examples 2000`,输出 `runs/r013/smoke-niah`。**三个"第一次":** 剔除逻辑首碰真实数据、
sealed-600 零重叠检查首次真实执行、`g2-proto-5` 的第一份 manifest。

**提交行(六 flag 逐条到位。⚠ 引号是语义的一部分,见下):**

```
sbatch --gres=gpu:rtx_3090:1 --time=02:00:00 scripts/run_r013_train_relations.slurm \
  13 runs/r013/smoke-niah \
  "--max-examples 2000 \
   --niah-manifest runs/niah-train-injected/manifest.json \
   --niah-provenance runs/niah-train-injected/provenance.jsonl \
   --niah-parents runs/niah-train-injected/source_parent.jsonl \
   --niah-dev-manifest runs/niah-injected/manifest.json \
   --sealed-dir runs/niah-sealed600 --niah-twin-label REFUTES"
```

**本条曾以无引号形式记录,那份记录是错的,不可执行。** 脚本的 `EXTRA="${3:-}"` 只取**第三个**位置
参数,六个 flag 必须整体作为一个 `$3`;裸着排在后面时 `$4` 起全部被静默丢弃。
错误来源:该记录是从 `sacct -o SubmitLine` 的输出重建的,**而 SubmitLine 不保留引号** ——
它把解析后的参数列表用空格拼回一行,看起来是可直接执行的命令行,实际上不能往返。
**与 `scontrol` 的 `Command=` 不显示位置参数是同一族陷阱,且两次都咬在同一个作业上。**
⇒ **纪律:要逐字复现一次提交,取台账里记录的命令,不取调度器的显示输出。**

适配池(`niah-train-injected`)与 dev(`niah-injected`)是两个不同目录,与守卫按 query-id **集合**比
而不读 `split` 标签的设计一致(脚本头部注明三个 NIAH 目录的标签都写着 `dev`,标签无意义)。
`--niah-twin-label REFUTES` 按 §3.8(a) 显式传,该 flag 无默认值。
注:`scontrol show job` 的 `Command=` **不显示位置参数**,一度据此误以为本作业未传参
(那将使它变成 seed 13 → `runs/r013/seed-13`、不带 `--max-examples` 的全量跑,顶着冒烟的作业号)。
**这条 `5afd69d`(同日 03:18)已写进 `HANDOVER.md`**,原话称其为"又一个因看不见而通过的审计" ——
**今日第三次撞上"知识已成文却未被用上"**(另两次:E2-lenient 的 index 运维注、本节的多臂预检)。
三次的共同点仍是同一条:**文字义务没有代码执行,就等于不存在。**

**执行的树 ≠ 提交时的树(排队作业的一般性质,此处已实际发生):**
提交时刻(03:32:30)分支头为 **`3637934`**;bp1 工作副本在起跑前被 `git pull` 前移到 **`9dcefc0`**
(2026-08-09 16:5x 实测 `rev-parse`)。排队作业执行的是**起跑那一刻**的工作副本,不是提交那一刻的。

**漂移已证无害,而且是逐字证的:** `git diff 3637934..9dcefc0 -- src/` 的全部内容是
**新增一个文件** `relations/backbone_agreement.py`(+160,R012f 用),`src/` 其余部分零改动。
该文件对训练路径**不可达** —— `cli/train_relations.py`、`relations/training.py`、
`relations/niah_adaptation.py` 均无引用,`relations/__init__.py` 只有 docstring 不做导入。

**该纪律当天即失效,故已改为代码执行。** 上面那条"在队作业期间不在 bp1 上 pull"写下数小时后
bp1 再次被 pull,`9dcefc0` → **`1558ee4`**(2026-08-09 17:5x,快进,无合并提交,工作树仅三个未跟踪文件)。
**这是同日第四次"成文的纪律没能阻止它所禁止的事"**(另三次:E2 索引运维注、多臂预检、`SubmitLine`)。
⇒ 不再试图冻住树,改为**让作业自己记**:`run_r013_train_relations.slurm` 与 `run_selector_gate.slurm`
在 `cd` 之后打印 `[tree] <rev-parse HEAD>[ +uncommitted]`,`.out` 日志从此自带执行树,
台账引它而不是引某人恰好记得去取的一次 `rev-parse`。**漂移由此变成被记录的,而不是被禁止的。**

**本作业的最终认定:执行树 `1558ee4`(非提交时的 `3637934`)。** 两次漂移合计对 `src/` 的改动是
**两个文件**:新增 `relations/backbone_agreement.py`(训练路径不可达,已证)与
`retriever/indexing.py` 的 `write_bytes` 修复(`5d59b22`)。后者只在 `ExperimentWorkflow.prepare`
路径上被调用,而 `train_relations` 不走 `ExperimentWorkflow` ⇒ **训练路径零改动,漂移可证惰性,无需重跑。**

**第一次提交:FAILED [job `18322821`,2026-08-09]**

| 项 | 值 |
|---|---|
| State / Exit | **FAILED / 1:0** |
| Elapsed | **00:00:28**(死于任何训练步之前,GPU 计算为零) |
| 起跑 / 节点 | 2026-08-09T20:24:08 / bp1-gpu030(rtx_3090) |
| 死因 | `FileNotFoundError: runs/niah-train-injected/source_parent.jsonl` —— `_load_niah_examples` 第一行 `read_parent_index` |

**根因:train split 的 parent sidecar 从未构建过。** dev 那份(`runs/niah-injected/source_parent.jsonl`)
早就存在并在同日 R001b 里用过,**train 那份没有** —— 两个 split 的产物不对称,而没有任何东西检查这件事。
已补建(登录节点,纯 CPU):`n_documents 101472 / n_parents 91595 / n_resolved 101472 / **n_unresolved 0**`
(对照 dev:101479 / 91492 / 0)。**`n_unresolved 0` 是必须核的一项** —— 若有未解析文档,
`ParentIndex` 的自 parent 回退会让后续零重叠比较**必然通过**,又是一个"因看不见而通过的审计"。

**这是冒烟第三次死于"命令行里写的路径在盘上没有"**(前两次 `18290519` / `18290571`,缺
`data/gate0b/vitaminc_train.jsonl`)。三次的 GPU 计算量都是零 —— 守卫的位置是对的;
**但本次的真实代价是 17 小时排队**,而该判定在提交前一秒即可完成。
与本日反复出现的同一形状一致:**能在零成本处判定的事被放到了昂贵的位置上。**
⇒ 待办(未做,见下轮):给 `cli/train_relations.py` 加 `--check-paths-only`
(已有 `--scaffold-check-only` 先例),**一次报出全部缺失路径而非死在第一个**,并写进 slurm 头部作提交前必跑步骤。
八条路径的临时 stat 检查已用过一次,但那是文字纪律,按本日四次实证迟早被跳过。

**第二次提交:FAILED [job `18329580`,2026-08-09 23:26,bp1-gpu030]**

`FAILED / **2:0** / 00:00:23` —— **退出码 2 而非 1**,是 argparse 拒绝参数,比前几次更早一层:

```
train_relations.py: error: argument --max-examples: expected one argument
```

**根因是引号丢失,不是缺文件。** 重交时六个 flag 裸着排在位置参数里,于是
`$3 = "--max-examples"`(无值)、`$4` 起**六个 flag 全部被静默丢弃**。命令是从
`sacct SubmitLine` 重建的,而那个输出不保留引号(见上)。

**这次失败是运气好。** 若引号恰好只裹住 `"--max-examples 2000"`、其余仍裸着,EXTRA 就合法,
作业会**正常起跑**,跑成一个只训 VitaminC 的运行,manifest 写 `niah_domain_adaptation: not run`,
**却顶着 smoke-niah 的作业号和输出目录** —— 而一个 COMPLETED 的作业没人会回头读 manifest。
**argparse 那声报错挡住的是这个。** GPU 计算量 0,23 秒。

⇒ 已加守卫(`b384101`):`scripts/run_r013_train_relations.slurm` 检查 `$#`,超过 3 个位置参数
即 **exit 2 并打印带引号的正确写法**,不再静默丢弃。两个方向都实测过。
**该守卫不在 `18330465` 的执行树内**(bp1 停在 `1558ee4`),从正式 R013–R015 起生效。

**第三次提交:COMPLETED —— 域适配半首次在真实数据上走通 [job `18330465`,2026-08-10]**

| 项 | 值 |
|---|---|
| State / Exit | **COMPLETED / 0:0** |
| Elapsed | **00:02:55** |
| 节点 | **bp1-gpu030(rtx_3090)** |
| 五折 API | 全部 `CrossEncoderTrainer API`(fold 0–4) |

**预注册读数十一项全过,`151` 与 `5284` 逐字命中。** 两者都是**提交前**在登录节点算好写进上面那张表的,
不是事后对上的 —— A4 的 dev 重叠剔除逻辑首次接触真实数据即复现预注册值:

```
niah_domain_adaptation.status                    = enforced
n_families_excluded_dev_overlap                  = 151      (预注册 151)
chain.n_niah                                     = 5284     (预注册 5284)
dev_eval_n_queries                               = 2000
chain.n_vitaminc                                 = 369819   (与 c901310 重导出后一致)
sealed_n_parent_pages                            = 2148
protocol_version / twin_label / seed / is_smoke_run / max_examples / hyperparameters.bf16 —— 全中
```

**sealed-600 零重叠检查通过**(该守卫是拒绝而非静默过滤,作业未崩即其 PASS)。

**附带结论:bf16 在 rtx_3090 上端到端可用,已实测而非由计算能力推断。**
`hyperparameters.bf16 = True` + `NodeList = bp1-gpu030` + 五折跑完,三者合起来即为证据。
⇒ **§3.8 的 15 次全量微调可分布在两个 bf16 节点(gpu035 a100 / gpu030 rtx_3090)**,
不必在争用的 a100 上串行。台账此前记的「rtx_3090 周转是小时级」曾被 18322821 的 17 小时排队证伪,
但本轮 18330465 于 00:17:59 起跑(提交后约一小时,backfill),**该结论恢复成立,且 2h TimeLimit 是主因**。

**两个冒烟至此全过**(`18321128` VitaminC 半 / `18330465` 域适配半)⇒ **正式 R013/R014/R015 的前提清空。**
执行前须在 bp1 `git pull`(冒烟已结束,此时拉取无害),以取得三样本轮新增的守卫:
`--check-paths-only`、位置参数 `$#` 守卫、`[tree]` 自记录与 bf16/卡名打印。

**待办:** `runs/niah-train-injected/source_parent.jsonl` 是本轮新建产物且 `runs/` 被 gitignore,
须 `git add -f` 入库,否则它只存在于 bp1。

**预注册读数(提交前在登录节点算好,跑完必须逐字相等):**

| 读数 | 预期 | 依据 |
|---|---:|---|
| `n_families_excluded_dev_overlap` | **151** | 有记录 query 1472 ∩ dev 2000 = 151(实测) |
| `chain.n_niah` | **5284** | (1472 − 151) × 4 pairs = 1321 × 4 |
| `dev_eval_n_queries` | **2000** | dev 评测运行的 query 数 |
| sealed-600 零重叠检查 | PASS | Gate 0A 审计五轴零重叠(§8),此为其一的真实数据复核 |
| `is_smoke_run` | true | `--max-examples` 存在 |

**判读注记(提前写死,防事后误读):** `--max-examples 2000` 截断的是 chain,而 chain 的顺序是
VitaminC 在前 ⇒ **本次实际参与训练的 2000 行全部是 VitaminC**;NIAH 行在构建、剔除、守卫、
sealed 检查层被完整执行但不进训练。这正是本冒烟要验的东西(接口而非训练效果);
**真正把 NIAH 行训进模型的是不带 `--max-examples` 的正式 R013**。manifest 的 `chain.n_vitaminc` /
`chain.n_niah` 记的是截断前的两半规模,与 `n_examples=2000` 不相加相等,属已知展示口径,不是缺陷。

---

### R013 — §3.8 训练路径,seed 13(**只交一个 seed,见范围披露**)[BEFORE,2026-08-10]

**披露(先写,免得被当成事后补的):本条撰写于 `18330608` 提交之后、任何结果存在之前。**
提交与撰写相隔约十分钟,作业当时仍 `PENDING`,**未观察到任何读数**,故仍是预注册而非事后叙述。
时间线如实记于此,与 A1 的处理同例:披露不因合规而取消。

**范围披露 —— 这是对 §5.4 三 seed 条款的一次有意偏离。** 本轮**只提交 seed 13**,42 / 73 暂不交。
理由不是技术性的:selector 线目前有两条并行路径(本路径与 Beam 三分类),而 15 次全量微调
(5 折 × 3 seed)与另一条路争同一批 bf16 节点。**在路线未定之前先交一个 seed,把承诺限制在 1/3。**
⇒ **本轮不构成 R013 的完整执行,不得作为三 seed 结果引用。** 若最终只有此一 seed,
结论必须写成"单 seed,未做稳定性检验"。

**目的:** Gate 0B 留下的唯一缺口是 **`gold_supports_recall` 最优 .7942(DeBERTa/template)、
距门限 .85 差 5.6pp**,而九格无一过门 ⇒ 族级断言"任何零训练模型都不够"成立。
**本轮问的是那个断言的另一半:训练能不能补上这 5.6pp。**

**假设:** 在去污染 VitaminC(369819 行)加 NIAH 域适配对(5284 行,已剔除 151 个 dev 相交 family)
上按 §3.8(b) 的冻结配方微调,`gold_supports_recall` 可越过 .85。

**预期指标 + 方向:** OOF 预测的 per-class F1(越高越好),重点看 SUPPORTS 类的召回。

**本作业不产生任何 Gate 0B 读数 —— 这一条必须先写死。** OOF 预测跑在**训练数据**上;
0B-1 是 VitaminC official test、§3.8 规定只跑一次,由 `scripts/run_gate0b.slurm` 执行,
且须把 manifest 里的 `gate0b_registry_line` 贴进 `cli/gate0b.py::LABEL_ORDER`(§11.10 第 7 项要求
走注册表而非调用点硬编码)。**把本轮的 OOF 数字当成过门证据,是这条路径上最容易犯的错。**

**失败判据(预注册):**
1. 训练完成但 OOF 的 SUPPORTS 召回**未显著高于**零训练基线 .7942 ⇒ 训练路径对该缺口无效,
   §3.8 的预注册应急臂(同族 large 再跑三 seed)才有依据启动;
2. 五折中任一折未产出 OOF 预测 ⇒ 本轮作废,不得以四折报告;
3. manifest 的 `niah_domain_adaptation.status` 非 `enforced` ⇒ 域适配半未真正执行,读数无效。

**精确命令(引号是语义的一部分,六 flag 必须整体作为一个 `$3`):**

```
PYTHONPATH=src python -m evidence_rag.cli.train_relations --check-paths-only \
  --vitaminc-train data/gate0b/vitaminc_train.jsonl \
  --vitaminc-dev data/gate0b/vitaminc_dev.jsonl \
  --decontamination-log data/gate0b/vitaminc_decontamination.json \
  --seed 13 --output-dir runs/r013/seed-13 \
  --niah-manifest runs/niah-train-injected/manifest.json \
  --niah-provenance runs/niah-train-injected/provenance.jsonl \
  --niah-parents runs/niah-train-injected/source_parent.jsonl \
  --niah-dev-manifest runs/niah-injected/manifest.json \
  --sealed-dir runs/niah-sealed600 --niah-twin-label REFUTES

sbatch --gres=gpu:rtx_3090:1 --time=24:00:00 scripts/run_r013_train_relations.slurm \
  13 runs/r013/seed-13 \
  "--niah-manifest runs/niah-train-injected/manifest.json \
   --niah-provenance runs/niah-train-injected/provenance.jsonl \
   --niah-parents runs/niah-train-injected/source_parent.jsonl \
   --niah-dev-manifest runs/niah-injected/manifest.json \
   --sealed-dir runs/niah-sealed600 --niah-twin-label REFUTES"
```

`--check-paths-only` 已跑,八条输入全在(首次实用)。**未跑 `--sanity-gate-only`** —— 它在提交后才落地,
本轮属于"该有而未用",如实记录。

| 项 | 值 |
|---|---|
| Job | **`18330608`**,提交 2026-08-10,`(Resources)` 排队 |
| 卡 / 上限 | **rtx_3090(gpu030)** / 24h。bf16 于 `18330466`(smoke-niah)在同一节点实测可用 |
| bp1 提交时 HEAD | **`3ee61a5`**(其后 `6f54d1d` 提交了 parent sidecar;**执行树以起跑时为准,由日志的 `[tree]` 行自记**) |
| 预期日志新增 | `[tree] <sha>` 与 `bf16: True | NVIDIA GeForce RTX 3090` —— 本作业是**第一个**带这两行的 |

**资源披露:** 本作业独占 gpu030 至多 24 小时,而并行的 Beam 路线同样需要 bf16 节点
(gpu030 / gpu035 是仅有的两个)。**已知会另一条线的负责人,不作为既成事实。**

**CONFOUNDERS 八条(逐条,见 `docs/selector/CONFOUNDERS.md`):**
1. 候选池 —— 不适用,本轮不做检索;2. 基线 —— 对照是 Gate 0B 的零训练 .7942,**非 TopK**;
3. n —— OOF 覆盖全部 375103 行,单一 n;4. 形式 vs 能力 —— hypothesis 模板与标签空间沿用
`g2-proto-5`,与 Gate 0B 逐字相同,**这正是本轮可与 .7942 比较的前提**;
5. 隔离探针外推 —— **本轮不适用,但 0B-2 的读数适用,评分时须复述 S6 纪律**;
6. 同源重复 —— 分折按 parent page + synthetic family 分组,已在代码强制;
7. 同次运行 —— 单作业单 seed,无跨运行比较;8. split —— A4 守卫按 query-id 集合比,
`n_families_excluded_dev_overlap` 已在 smoke 上复现预注册值 151。

**AFTER(job `18330608`,2026-08-10):§3.8 训练路径首次完整执行**

| 项 | 值 |
|---|---|
| State / Exit | **COMPLETED / 0:0** |
| Elapsed | **03:35:37** |
| 节点 | bp1-gpu030(rtx_3090) |
| 日志自记 | `bf16: True | NVIDIA GeForce RTX 3090`;五折全部 `CrossEncoderTrainer API` |
| 执行树 | **`6f54d1d`**(由登录节点 `rev-parse` 确认,原因见下) |

**三条预注册失败判据逐条核对,全部不触发:**
1. 见下"判据一写松了"—— 原文的比较不成立,已更正,新表述下不触发;
2. **五折 OOF 齐全**:75021 / 75021 / 75021 / 75020 / 75020 = **375103 = `chain.n_examples`**;
3. `niah_domain_adaptation.status = **enforced**`,`is_smoke_run = **False**`。

**读数(OOF per-class,按 source 拆):**

| | macro-F1 | SUPPORTS | REFUTES | UNKNOWN |
|---|---:|---|---|---|
| 全部(n=375103) | **.8786** | P .9397 / R .9552 / F1 .9474 | P .9076 / R .8925 / F1 .9000 | P .7950 / R .7821 / F1 **.7885** |
| `source=niah`(n=5284) | **.9695** | F1 .9700 | F1 .9690 | — |
| `source=vitaminc`(n=369819) | .8781 | F1 .9471 | F1 .8986 | F1 .7885 |

- **NIAH 半只有两类且完全平衡**(REFUTES / SUPPORTS 各 2642)—— 构造使然:每个 family 出一条 needle 与一条 twin,
  且 §3.8(a) 把 twin 标为 REFUTES,**UNKNOWN 只由 VitaminC 的 NEI 提供**。
- **UNKNOWN 是最弱的一类(.7885)**,与 §10.11 记录的"UNKNOWN 概率质量堆积"风险方向一致,应带入 0B 打分时的判读。

**⚠ 这不是 Gate 0B 读数,重复一遍。** OOF 跑在**训练数据**上;NIAH 那 .9695 尤须克制 ——
那些行出自同一注入器、同一构造,分折虽按 parent page + synthetic family 分组,仍属同分布。
**它证明"训练学到了东西",不证明"能过门"。**

**判据一写松了(BEFORE 的自我更正):** 原文写「OOF 的 SUPPORTS 召回未显著高于 .7942 ⇒ 训练路径无效」。
**该比较不成立** —— `.7942` 是 0B-2 探针(mutation-log 任务对,n=5888)上的 `gold_supports_recall`,
而 OOF 是训练数据上的,两个不同总体。**正确表述:OOF 只能判"训练是否收敛到可用的判别力",
过门与否必须由 `cli/gate0b.py` 单独打分。** 这正是本条目开头警告过的那个错,而预注册自己踩了半只脚,如实记录。

**打分钥匙(逐字,勿重排):**

```
    "runs/r013/seed-13": ('REFUTES', 'SUPPORTS', 'UNKNOWN'),
```

第三种标签序(albert 为 SUPPORTS/REFUTES/UNKNOWN,DeBERTa-large-mnli 为 SUPPORTS/UNKNOWN/REFUTES),
按位置猜不会报错、只会把每条边重新贴标签而下游数字看着都正常。**必须走注册表,不得在调用点硬编码**(§11.10 第 7 项)。

**成本读数,推翻了本条目 BEFORE 里的资源论证:** 单 seed 五折 **3h36m**,而非 slurm 申请的 24h。
三 seed 合计约 **11 GPU 小时**,且跑在**不争用的 rtx_3090** 上。
⇒ **「15 次全量微调与 Beam 路线争 bf16 节点」这条理由基本不成立**,BEFORE 中据此限制到单 seed 的
范围披露,其**资源依据已被实测削弱**(路线不确定这条依据仍然成立)。

**⚠ `[tree]` 自记录在计算节点上失效 —— 守卫本身有缺陷,不是树脏。**
日志打印 `[tree] unknown +uncommitted`,两个回退**同时**触发,这是 `git` 不可用的签名。
登录节点上实测 `git rev-parse HEAD` 正常返回、`git diff --quiet` 干净
⇒ **代码正确,计算节点没有 git。** 与"本地绿、目标环境红"是同一族(参见索引持久化那次的换行符)。
执行树因此由登录节点手工确认为 `6f54d1d`。**修法:改为直接解析 `.git/HEAD` 与 `refs/`(纯文本,不需 git 二进制);
dirty 状态无 git 不可查,应诚实地不报。** 待办。

**下一步不是补交 seed 42/73。** 真正的问题是"训练有没有补上 Gate 0B 的 5.6pp",
而回答它需要**给这个 checkpoint 打分**,不是再训两个 seed:过了,三 seed 才值得补(§5.4 稳定性);
没过,补两个 seed 也不改变结论。

#### NIAH 半的 .9695 不是表层伪影 —— 打分前的判据 [2026-08-10,`scripts/niah_surface_baseline.py`]

**为什么必须先问:** 孪生是**替换 needle 里一个实体**造出来的,所以 needle 与 twin 近重复、共用同一个 claim。
模型只要判断"claim 里的答案串在不在段落里"就能分开它们 —— **纯表层,不含任何矛盾判断,
而这两种故事在本数据上都预测 .97。** 它们在真实检索池上预测的却完全不同(那里的误导证据
不是任何段落的实体替换版),而这正是 S6 量到的那种"隔离构造不外推"。

**读数一 —— 表层基线(纯 CPU,无模型,且刻意做强:阈值事后扫最优):**

| | macro-F1 |
|---|---:|
| always-SUPPORTS(退化下界) | .3333 |
| **表层词重叠**(阈值 .4545) | **.7386**(REFUTES .7275 / SUPPORTS .7496) |
| 训练模型 | **.9695** |
| **gap** | **+.2309** |

表层能到 .7386 —— **线索确实很强**,所以"模型会用它"不是空担心。但按剩余错误看:表层差 26.1pp、
模型差 3.05pp,**模型消掉了表层剩余错误的约 88%。**

**读数二 —— 消融:表层规则判错的地方,模型对不对(决定性的那一个):**

| 子集 | 行数 | 占比 | 模型准确率 |
|---|---:|---:|---:|
| 表层已经判对 | 3905 | 73.9% | .9749 |
| **表层判错** | **1379** | **26.1%** | **.9543** |

⇒ **模型在修正表层规则,不是复述它**,且两个子集只差 2 个点。**"`.9695` 是字符串匹配换身衣服"这个解释被排除。**
1379 行不是小样本,结论不受"从一小撮读大结论"的限制。

**排除了什么、没排除什么(须同时声明):**
- **已排除:** claim–passage 词重叠这一种表层故事 —— 它是最可能的一种(孪生就是实体替换),现已否定。
- **未排除:** 注入器可能留下的**其它**浅痕迹(替换实体在上下文中的不合理、位置、长度分布…),本轮未测。
- **未触及:** 本消融仍是「一段 + 一句」的成对形式。**S6 崩掉的机制是池结构**(一个 claim 面对 20 段),
  这一维完全没有被检验。⇒ **不得据此声称可外推到真实检索池。**

**方法注:** 首版按 synthetic family 配对,假设每 family 两行,**被真实数据当场拒绝** ——
每个 family 实为**四行**(needle/twin 两个段落 × 两个 claim),即预注册的 `5284 = 1321 × 4`。
守卫抛错而非静默分错组,故代价是五分钟而不是一个"给错行算出来的合理数字"。改版改按
"表层规则判对/判错"逐行切分,无需分组结构。OOF 与重建链按位置合并,**每行 gold 必须逐一对上,不对即拒**。

**单边痕迹整类由构造排除 [2026-08-10,登录节点秒级]。** 上面只否定了"词重叠"一种表层故事,
而枚举其余的(替换实体的位置、长度、来自答案库、上下文不合理…)没有尽头。改用标准的**单边基线**
(NLI 文献的 hypothesis-only baseline):**只看段落、或只看 claim,能不能预测标签。**
一个测试覆盖全部单边痕迹,不必枚举。

| 单边 | 去重 | 同时带两种标签 | 单边准确率上限 |
|---|---:|---:|---:|
| premise(段落) | 2616 | **2616(100.0%)** | **.5000** |
| hypothesis(claim) | 2642 | **2642(100.0%)** | **.5000** |

⇒ **2×2 交叉是完整的**({needle, twin} 段落 × {gold, twin} claim),每个段落与每个 claim 都各带一次
SUPPORTS 与一次 REFUTES。**任何只看一侧的规则上限即掷硬币,模型必须使用这一对。**
这是**结构性**结论,不依赖挑了哪个特征 —— 单边痕迹这一类到此关闭。
(注:段落去重 2616 < claim 去重 2642,26 条段落文本跨 family 重复,约 1%,不影响上限,如实记录。)

**⇒ 剩下唯一未检验的维度是池结构,而它排不掉、只能测。** 训练数据里根本没有"一个 claim 对 20 段"
这种形状,任何对训练数据的重新切分都回答不了它。**注意:打 Gate 0B 的分也回答不了** ——
0B-2 本身就是隔离对探针(§本文件 S7 限制第 2 条),与本节的消融同形。
待测:用 `runs/niah-injected` 的真实 top-20,对每题以同一 claim 跑过全部 20 段,
**读那 18 段非 needle/twin 的误判率** —— 本该 UNKNOWN 却被判成 SUPPORTS/REFUTES 的比例,
即假冲突,即 S6 中让 recall 崩 38.3pp 的机制。
**先验不乐观:OOF 里 UNKNOWN 是最弱的一类(F1 .7885),而池内绝大多数段本应 UNKNOWN。**

⇒ **判据满足,可以去打 Gate 0B 的分**(但须带上以上限制:过门不等于池内可用)。

#### 池级探针:成对上出色的模型,在真实 top-20 里几乎不可用 [job `18366462`,2026-08-10]

`COMPLETED / 0:0 / 00:03:11 / bp1-gpu030`。一个 claim(同一个 `build_hypothesis`)对**全部 20 段**,
1479 题(29580 / 20,**与 harm 指标的分母逐字相同**,独立口径再次对上)。

| 角色 | n | SUPPORTS | REFUTES | UNKNOWN |
|---|---:|---:|---:|---:|
| needle | 8601 | **97.80%** | 2.20% | — |
| twin | 1246 | 7.38% | **92.62%** | — |
| **其余(应为 UNKNOWN)** | **19733** | 3.49% | **96.50%** | **0.01%** |

**假冲突率 .9999,平均每窗口 13.34 个。**

**UNKNOWN 不是弱,是塌了:19733 段无关证据里只有 2 段被判 UNKNOWN。** 模型在池内的默认输出是
**REFUTES** —— **每一段与问题无关的证据都被当成毒**,对选择器是最坏的默认值。

**机制清楚,且与预注册的担心一致但更极端:** UNKNOWN 只由 VitaminC 的 NEI 提供(训练链的 14%),
而 NIAH 半**根本不含"这段与问题无关"这种样本** —— 那一半的每一行,答案要么在(SUPPORTS)、
要么被替换(REFUTES)。**池形负例在训练集中完全不存在**,所以模型对池形输入没有 UNKNOWN 先验。
部署分布约 90% 应为 UNKNOWN,训练分布 14% 且形状不同。

**这一条比 S6 更硬,因为伪影已被先行排除。** S6 可以被"也许那个方法本来就没学到东西"消解;
这里不能 —— 词重叠基线只到 .7386,而模型在它判错的 1379 行上仍有 .9543,单边基线上限恰好 .5000。
⇒ **模型确实学到了真实的成对判别力,而那个判别力仍然不迁移到池结构。**
**「在隔离对上训练关系模型」这条路径,由此有了一个带机制的否定结论,而不是一句"没做出来"。**

**对另一条路线的直接含义(应转告):** Beam 三分类的 `IRRELEVANT` 类**直接从真实 Top-20 采样困难负例**,
正是本轮缺失的那一味。**本结果是支持那个数据设计的证据**,不是与之竞争的读数。

**因此不再打 Gate 0B 的分。** 0B-2 本身是隔离对探针,与本轮已证不足以外推的那个形状相同;
在池级已判不可用之后,再花那次一次性机会去测同一形状,不会增加任何信息。
**§3.8 训练路径至此收口:代码可用、训练收敛、伪影排除、池级不可用,四项俱全。**

**未做且不建议在剩余时间内做:** 用池形负例重建训练集并重训(治本的唯一改法)。
成本是重做数据面加十一 GPU 小时以上,而另一条路线的数据设计已经包含它。

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

**AFTER(job `18269703` 生成 + `18269704` 评分)—— G6 的读数出自这一次:**

- **预注册判据:通过。** coverage 0.635 → **0.784**(+0.147,p=0.0,CI [0.112, 0.183]);
  **已引用句**的引用精度 **0.883 → 0.883,配对 delta 恰为 0.000,p=1.0** —— 未验证内容零泄漏。
- **契约解除是纯增益**:capped 的 251 条答案在 open 中**逐字节相同**(251/251),
  引用句数两臂同为 **313**、有引用的样本同为 **251**。open 只是**新增**了 58 例全标注答案,
  没有改动任何已有答案。correctness +0.021(p=0.0)。
- **标注触达**:77 条标注声明中 **76 条进入答案**(封顶下仅 15 条)。标注率从保留句的
  4.3% 升到 **19.2%**。
- **相对 baseline**:精度 **+0.192**(p=0.0)、召回 **+0.061**(p=0.032)—— 召回**反超** baseline
  且显著;代价 coverage −0.145、correctness −0.043。
- **句子构成**:baseline 427 引用 / 0 标注;verify-only 275 / 0;capped 313 / 14 / 1 无据未标注;
  open 313 / 75 / 3 无据未标注。
- **arm 级 0.717 精度不是退化,是口径**:ALCE 给"完全没有引用"的样本记精度 0,而 open 首次
  产生了这类样本(58 例)。0.883 × 251 / 309 = 0.717,逐位吻合。ALCE 这一约定与本研究**预注册的**
  口径("标注声明既不进分子也不进分母")相冲突,冲突只在能产出全标注答案的臂上才暴露。
  故两个口径**并列报告**,不静默替换。

**本轮修掉的评分器缺陷(评分侧,非 src;均只曾压低 annotate 臂):**

1. **标注标记落在错误的句子上。** 生成器写 `"<句子>. [unverified]"`,标记在终止符**之后**,
   切句器因此把它带到**下一句**开头(或在答案末尾孤立成段)。后果:真正未验证的句子被当作未标注,
   其后继被当作标注;孤立标记又作为一个无引用句进入 ALCE **召回分母**。
   两个后果都只发生在标注臂上。分子不受影响(两段都不带引用),故修正后召回可由
   `entail = round(rec × 旧句数)` 精确还原 —— 仅 8 例(标记溢出到下一句)的分子为近似,
   且只会**低估**。修正后"无据未标注"句从 69 降到 **3**。
2. **`summary["verify-annotate"]` 键名过期**,四臂改名后未同步 —— 报告生成段崩溃(四臂数值已全部算出)。
   评分作业因此以 exit 1 结束;数值在崩溃前已全部落盘,故未重跑 GPU。

**尚待裁决后落定的**:实体冲突盲审(见 `docs/generator/g5-entity-conflict-audit.md`)—— 已于 G7 完成。

**AFTER(重跑尝试,job `18281379`,COMPLETED 01:16:04,exit 0:0)—— 作废,三个验证臂全崩:**

> **两条记录的关系(合并时厘清,并更正一处根因):**
> `18269703` 在 08-05 04:41 由 `ri25947` 完成并产出了上面的读数;`18281379` 是其后由
> **`uz25020`** 在 `/user/work/uz25020/IBM_Granite_Project` 提交的重跑,三个验证臂全崩。
> 两条都属实、都保留:**前者是 G6 的读数,后者是环境事故与其修复的记录。**
> 原记述称 `18281379` 为"第一次尝试",按时间线更正为"重跑尝试"。
>
> **根因更正(已核实,非推断):** 原记述归因于"环境被刷过一次,缓存随之清空"。
> 实测:`ri25947` 的缓存里 `google/t5_xxl_true_nli_mixture` **一直在**,且**至今仍只有 `.bin`**
> (`pytorch_model-0000{1..5}-of-00005.bin`,05-28 落盘,从未清空)。
> 真正区分两个账户的是 **torch 版本**:`ri25947` 的 venv 是 **torch 2.6.0+cu124**,
> 高于 CVE-2025-32434 的门槛,故 `.bin` 照常加载 —— 这正是 `18269703` 能跑通的原因;
> `uz25020` 的环境是 torch 2.5.1,低于门槛,transformers 拒绝加载同一批 `.bin`。
> **即"两层串行根因"里的第一层(缓存缺失)在本账户上不成立,第二层(torch<2.6)才是全部原因。**
> 这不减损那次转换的价值:转成 safetensors 后**与 torch 版本无关**,是更稳的落法;
> 且该事故点出的三个工具缺陷完全独立成立,已在本轮修掉。
> 预检脚本现在**显式检查这条版本边界**,而不是假定它。

| 臂 | 结果 |
|---|---|
| baseline | 370/400 answered,**0 errors** |
| verify-only | **0/400,365 errors** |
| verify-annotate-capped | **0/400,365 errors** |
| verify-annotate-open | **0/400,365 errors** |

`routing-stats.json` 全零,`claims_routed: 0` —— **一条 claim 都没被路由过**。
`results/g5/*.jsonl` 里三个验证臂各只有 **35 行**(400 − 365),且零答题。
**没有任何指标产生,本轮不构成 G6 的读数。**

**根因是两层,且串行 —— 修掉第一层才看见第二层:**

1. **`google/t5_xxl_true_nli_mixture` 根本不在离线缓存里。** 目录整个不存在
   (非负缓存,`.no_exist` 标记也没有)。slurm 设 `HF_HUB_OFFLINE=1`,于是每次调用抛 `OSError`。
   最可能的时间线:**环境被刷过一次,缓存随之清空** —— G5 第二轮(job `18267966`)
   用的是同一个验证器,当时是跑通的。
2. **补下之后仍然加载不了:该 repo 只发 `pytorch_model-*.bin`,不发 safetensors**,
   而集群是 **transformers 4.57.6 + torch 2.5.1**,后者 < 2.6 ⇒ transformers 按
   CVE-2025-32434 **拒绝 `torch.load` 任何 `.bin`**。直接实测确认而非推断:
   `check_torch_load_is_safe()` 在本集群抛异常。

**四个观测量与该解释逐一对得上,这是采信它的依据:** 三臂错误数完全相同(它们共用 `nli`,
baseline 不用);baseline 全好(Granite 本身无恙);35 条活下来的是**根本没抽出 claim** 的例
——不调验证器就不报错,这也解释了 `claims_routed: 0`;`[gen]` 时间全程平在 **10.3s/case**,
那只是 Granite 的开销,**NLI 调用是快速失败,没有任何资源耗尽的斜率**。

**处置:本地转 safetensors(`scripts/convert_bin_to_safetensors.py`)。**
以 `weights_only=True`(CVE 自身指定的缓解手段)逐分片反序列化并重写,5 分片 512 权重,
每个输入输出文件的 sha256 记入 `results/true_nli_safetensors_conversion.json`。
**T5 的三名绑定嵌入按克隆而非丢弃处理** —— 丢一个键会被当作缺失并随机初始化,
**模型照样跑、照样给出看着合理的分数**。

**转换的值级验证(job `18288235`,COMPLETED 00:04:45,exit 0:0):**
转换脚本只能做头部校验(keys/shapes/dtypes),查不到数值,故另跑三对已知答案的
(`scripts/verify_true_nli.slurm`):蕴含 **0.9989** / 矛盾 **0.0005** / 无关 **0.0006**。
三个数量级的分离 ⇒ 权重正确。**判据写成"蕴含必须跨过 0.5 高于另两个",
因为一个被转坏成'什么都打 0.5'的模型加载正常、跑得动、日志好看。**

**该验证日志里两条 warning,均已判读,不阻塞:**

- `device_map keys do not match any submodules: [decoder.embed_tokens, encoder.embed_tokens]`
  —— 克隆决定的直接后果:checkpoint 有三个嵌入键,而 `tie_word_embeddings=True` 下模型里
  只有 `shared` 是独立子模块。**它无害,但不是碰巧无害 —— 是因为三份克隆逐字节相同**,
  走哪一份值都一样。
- `Some parameters are on the meta device because they were offloaded to the cpu`
  —— TRUE 是 fp32 约 45GB,单张 40GB A100 装不下。**重跑时须看第一条 `[gen]` 的 s/case:
  上一轮的 10.3s/case 是 NLI 从未运行时测的,不能当基线。**

**本次事故暴露的三个工具缺陷(均在 G 模块,未修,留给该模块负责人):**

1. **`run_g5_verify_annotate.slurm` 缺登录节点预下载。** `run_gate0b.slurm` 有(其头部明写
   "含登录节点的模型预下载")。**这是根因得以拖到 GPU 上才爆的直接原因。**
2. **`g5_verify_annotate.py` 的 `except` 只打 `type(exc).__name__`**,不打 message、不打
   traceback,且每臂只打前 3 次。**若它打的是 `exc` 本身,本次五秒即可定位。**
3. **吞掉 365 次异常后退出码仍为 0,并照常写出结果文件。**
   **这次是靠人工看 `[arm]` 四行才发现的** —— 评分作业本会从 35 条空记录产出一份格式完好、
   数字齐全的报告。建议改为错误率超阈值即失败。

**与 A3 的牵动(已回写 M0 §11.1 限定 2):** 本次采纳的"本地转 safetensors"正是 A3 当时
列为"未采纳但未否决"的那条绕路,**四天后在另一个模型上被需要并采纳**。于是
"凭什么验证器可以转、训练基座不可以"成为必须回答的问题。不对称的论证
(训练基座的 provenance 流进交付物;验证器是两边都由 hash 钉死的仪器)已写入 §11.1,
**并明确标注它是需要被同意的论证而非既定事实** —— 若该区分不成立,
**A3 的核心依据相应减弱**。

**另需一并修正的记述:** `run_g5_score.slurm` 的配对显著性循环曾使用 G6 拆分前的旧臂名
`verify-annotate`,且每次调用带 `|| true` ⇒ **作业会退出 0 而一个检验都没跑**,
且**归因所需的 open vs capped 那一对连旧名下都不在循环里**。已于 `c2b137e` 修正:
四个真实臂名、open-vs-capped 排第一、去掉 `|| true`、循环前检查四份 report 是否存在。

**下一步:** 重交生成作业。**先看 `[arm]` 四行确认四臂均有合理答题数,再交评分** ——
本次的教训正在于此。

---

## G7 — 移除实体门(依据审计发现,非调参)

**状态:** 代码就绪。**BEFORE 在运行前写入并提交。**
**本轮改动源自两轮盲审的一致结论,不是为了移动数字。**

**BEFORE(预注册):**

- 依据:两轮盲审,样本群互不相交、触发构成完全不同,假否决率**同为 14/20 = 0.700**;
  错误摧毁 0.850,CI [0.640, 0.948]。spaCy 切换清除了全部 `name:some` 类触发、`absent` 降到 1/78,
  **错误率一动不动** —— 失败从来不在"抽取了哪些 span"。三条判对的丢弃(claim 不完整 / 量词辖域 /
  approved-vs-implemented)**没有一条是实体冲突**,故 0.150 的正确率是巧合。
- **改动:仅凭蕴含决定引用,不存在丢弃路径。** entailed → 保留并附已验证引用;
  未 entailed → 保留并标注未验证。**系统不再摧毁任何内容。**
- **实体检查保持运行,observe-only**:每条 claim 都记录它**本会**给出的判决,但不影响路由。
  这把"约 51 条被错误摧毁"从外推变成**直接测量** —— 门本会摧毁什么、那些 claim 实际变成了什么。
- **不**把冲突路由到"证据与此声明冲突"标注:约 70% 的该标签是错的,对一条证据实际支持的声明
  断言"证据与之冲突",是在**对证据作出虚假断言** —— 比删除更糟(删除至少不断言任何东西)。
  如此不可靠的信号不得驱动任何用户可见标签。
- 五臂**同一作业**:baseline / verify-only / verify-annotate-capped /
  **verify-annotate-open(控制:门开启)** / **verify-annotate-nogate(本轮主体)**。
- **预期方向:** coverage 升过 0.784;已引用句数升过 313;引用召回上升;correctness 上升;
  **引用精度小幅下降**(真冲突现在会被引用,审计估计约占门控 60 条的 15%,约一个百分点)。
- **失败判据(预注册):** 若**已引用句**的引用精度跌破 verify-only 的 0.862,
  说明实体门阻止的坏引用远多于两轮盲审所示,则移除失败。
- **主报 ALCE 标准口径**的引用精度,cited-sample 口径并列 —— 结论在不利口径下也成立,
  用有利口径打头会招致"挑口径"的指控。
- 实体层按**威胁模型相关**报告,而非失败组件:G1 对抗性实体替换切片 0.963(仅验证器)→ **1.000**(接入该层),
  良性数据上错误摧毁 0.850。**两个数字都要进报告。**
- 纪律:一轮只改一件事(仅移除门);提示词/阈值不动;预注册后冻结。
- 数据合规:只用 ALCE/ASQA;**HotpotQA / RGB / MuSiQue-Full 从不加载**。

**G6 配对统计的复核(本轮 Task 1):**

- G6 汇报的配对数字**不是**来自 slurm 显著性循环 —— 该循环**根本没执行**:
  `set -euo pipefail` 下 `g5_score.py` 在报告拼接处崩溃,作业在到达循环前就终止了。
  汇报的数字来自我在本地对落盘 report 的直接运行,**已在切句修复之后**。
- 但该循环的隐患**属实且更严重**:它用的是过期臂名 `verify-annotate`(四臂改名后已不存在),
  每次调用都会失败,而**每行都挂了 `|| true`**,作业仍会以 0 退出。
  且它**从未包含 open-vs-capped** —— 唯一能把契约解除与验证过滤分开的那一对。
- 复核结果:重跑全部配对,**open-vs-capped 四轴与已汇报数字逐位一致**(delta 0.147 / 0.021 / 0.000 / 0.000)。
  该对的精度/召回对口径**免疫** —— 配对天然只取两臂都作答的 251 例,恰为有引用的样本。

**接手队友在 G6 事故记录里点名的三个工具缺陷(全在 G 模块,本轮修掉):**

1. **`run_g5_verify_annotate.slurm` 缺登录节点预下载** —— 这是"根因拖到 GPU 上才爆"的直接原因。
   已加:提交前在登录节点验证 Granite 与 TRUE 均可加载,加载不了就**不提交**。
2. **`except` 只打 `type(exc).__name__`** —— 已改为打完整 message,并对**每臂第一次**异常打完整
   traceback。队友的判断成立:"若它打的是 `exc` 本身,本次五秒即可定位。"
3. **吞掉 365 次异常后退出码仍为 0** —— 已加**错误率阈值**(默认 0.10):任一臂超阈值即**非零退出**,
   且**不写结果文件**。理由与队友一致:评分作业本会从 35 条空记录产出一份格式完好、数字齐全的报告。

**AFTER(job `18295681` 生成 03:23:50 + `18295682` 评分 00:20:29,均 exit 0):**

- **预注册失败判据被触发,先说这个。** nogate 的 cited-sample 精度 **0.8476 < verify-only 的 0.8617**,
  差 1.4 个百分点。按字面判据,本次移除**失败**。
- **但该判据本身写错了**:它拿一个臂的均值去比另一个臂的均值,而两臂**作答的 query 集合不同**。
  分解:两臂共同作答的 253 例上 nogate **0.8841 > verify-only 0.8617**(配对 +0.022,p=0.238);
  arm 级的缺口来自 nogate 多答的 **51 例难题**(其精度 0.6667)。
  **这是我写判据时的规格错误,不是结果** —— 记在这里,不用它来消解那次触发。
- **但门也不是在乱开火**:含"本会被门摧毁"声明的 58 例精度 **0.6782**,其余 246 例 **0.8875**,
  差 21 个百分点。人工审计的 0.850 错误摧毁率与这里的 0.678 自动精度**不矛盾** ——
  审计问"这段证据是否支持该声明",判官问"被引证据是否蕴含成句的该句"。
  门对第一个问题是差工具,对第二个问题**优于随机**。
  即便如此,0.667 仍**高于 baseline 的 0.597 平均值**:系统新增的**最差**内容,也比 baseline 的典型内容引得更准。
- **门的代价(nogate vs open,配对):** coverage **+0.127**(p=0.0)、correctness **+0.044**(p=0.0);
  引用精度 −0.005(p=0.698)、引用召回 +0.004(p=0.722)。
  **门花掉 12.7 点 coverage 和 4.4 点 correctness,在两根引用轴上什么也没买到。**
  预注册预期精度代价"约一个百分点",配对实测 0.5 点且不显著 —— 方向与量级都对。
- **对 baseline(本轮的头条):** coverage −0.018(**p=0.297**)、correctness +0.001(**p=0.888**)
  —— **两轴均与 baseline 统计无差异**;而引用精度 **+0.191**(p=0.0)、引用召回 **+0.071**(p=0.009)。
  此前每一轮都要拿大幅 coverage/correctness 赤字换引用质量,**这个代价现在没有了。**
- **observe-only 直接测量:** 门**本会摧毁 60 条**,其中 **60 条(100%)改为获得引用**、0 条落到标注;
  另有 14 条两边都引用但门会选别的证据。
  **自检精确通过**:控制臂上 `gate_would_drop` = 60 = `dropped_entity_conflict`,
  故这 60 是**测量值而非外推**(G6 的外推是"约 51/60")。
  另有 2 条原本被标注的声明现在通过验证(缺失而非冲突),declared 引用存活率
  223/399(0.559)→ **283/399(0.709)**。
- **句子构成:** baseline 427/0/0;verify-only 275/0/0;capped 313/14/1;open 313/75/3;
  **nogate 379/73/3**(已引用 / 标注 / 无据未标注)。较控制臂 **+66 已引用句,且无丢弃路径**。
- **控制臂 394/394 逐字节复现 G6**,baseline 与 verify-only 连错误计数都逐位复现。
  据此可断言:本轮路由重构(一趟同时算两个判决)**对控制臂零影响** —— 这个推论成立。
  > **更正(G8 后回填):** 原文接着写了"交叉运行一致性已验证而非假定"。**该推论错误,予以撤回。**
  > G8 重跑同样五臂,发现**不含任何改动的 baseline 臂**仅以 **356/400(0.890)** 复现 G7。
  > 两个候选解释已对着作业日志核查:**"G7 复用了缓存生成"——排除**
  > (G7 日志有完整生成过程,**6.06 s/arm/case**,G6 为 **6.03**;runner 也没有任何缓存路径);
  > **"执行条件恰好相同"——成立**(G6 与 G7 每 arm-case 相差 0.5% 以内,
  > 而 G8 在**同一节点** `bp1-gpu035` 上以 **2.70 s/arm/case** 跑完,快 2.2 倍,11% 的 query 结果不同)。
  > 解码是贪心,算术确定但 GPU 归约顺序不确定,接近平手处即翻转。
  > **正确的说法是:两次执行条件恰好相同的运行之间观察到了逐字节一致;它取决于执行条件,不是保证,不可依赖。**
  > 由此确立并已在全库执行的规则:**任何被报告的比较都不得跨作业。**
  > 具体地,两轮之间未作任何改动的 `verify-annotate-capped`,引用精度在 G7 读作 0.890、G8 读作 0.911 ——
  > **这 0.021 是纯粹的跨作业漂移,不是结果。**

**本轮暴露的一个契约缺陷(未修,已定位):** open 与 nogate 各丢一例,
`GenerationResult` 抛 *uncited answer must mark every sentence with '[unverified]'*。
`count_sentences` 按 `[.!?]` 切句,故声明内含缩写句点(`won on Jan. 11, 1970`、
`Acme Inc. in Ohio`)会被数成两句而只有一个标记,**validator 于是误拒一条本已全标注的答案**。
已对该函数直接实测确认,非从报错推断。**是真缺陷(误拒摧毁整条答案),但本轮不修**:
400 例中占 1 例,且对 open/nogate 影响相同,**不会污染 nogate-vs-open 的归因**;
修了就要重跑才能让代码与已报数字一致,不值。**列为下一轮第一项。**

**实体层按威胁模型记录(两个数字都在):** G1 对抗性实体替换切片 0.963 → **1.000**;
良性数据错误摧毁 **0.850**(人工,两轮盲审),其针对的样本群引用精度 **0.678 vs 其余 0.888**(自动,本轮)。
门以 `entity_gate=True` 保留在代码里、默认 observe-only 运行 —— **该开关就是声明威胁模型的地方**,
而不是一件需要重建的东西。

---

## G8 — 最终标定轮(此后冻结)

**状态:** 代码就绪,BEFORE 在运行前写入并提交。
**本轮数字变动的原因是 `count_sentences` 修复,不是调参。** 审阅标记为纯呈现层,不得移动任何指标。

**BEFORE(预注册):**

- **Task 1 — 修 `count_sentences`。** G7 的实测定位:每个 annotate 臂各丢一条答案,
  因为 `[.!?]` 切句把缩写和首字母缩写当句末。**实测 7 例全部属于此类**
  (`Mount St. Helens`、`Patrick S. Castagne`、`Brown v. Board`、`1913 U.S. Open`、
  `St. Petersburg`、`John L. O'Sullivan`、`Eduardo J. Padrón`),
  **零例**属于"句点后接小写"。后者(每臂 30–36 处,如 `World Cup. in 2010.`)
  是**声明切分器的真实产物**:它们本就是答案里的独立句、各自带标记,**合并才是错的**,故不动。
  规则只做一件事:**缩写与首字母后的句点不算句末**。缩写表刻意保持极小 ——
  错加一条会**合并两个真句**,那是更糟的错误。
- **同一条规则由契约与评分器共用**(`contracts.split_sentences` / `ends_with_abbreviation`),
  于是"validator 要求打标记的那个句子"与"指标计分的那个句子"是同一个。
  预期影响:每臂救回约 1 条答案;句数减少 baseline 11 / verify-only 5 / capped 6 / open 6 / nogate 7。
- **Task 2 — 审阅标记。** 实体层此前 observe-only、对输出零贡献。现给它一个**非破坏性**角色:
  **被它标记的声明仍然引用,并附低置信标签。**
  依据是测量而非直觉:G7 中被标样本 cited-precision **0.678**,其余 **0.888**
  —— 错误率 0.322 对 0.112,约**三倍富集**。筛查信号不需要高精度,需要的是相对基率的提升。
  - **措辞不断言冲突**(逐字):**`[may warrant review]`**。
    "证据与此声明冲突"是对证据的事实断言,两轮盲审显示它约 70% 是错的 ——
    断言它比它所取代的删除**更糟**(删除至少不断言任何东西)。该标签只断言**它对自己没把握**。
  - **与 `[unverified]` 严格区分**:后者是"未找到支持证据、本句无引用",
    前者是"已找到并已引用,但二次检查有异议"。**被标句一定带引用。**
  - **不影响路由与指标**:标签在送判官与算 STR-EM 前被剥除,被标句照常进入精度与召回。
    门开启的臂里 would-drop 会被丢弃,**故消融臂永远不会出现该标记**,ablation 保持原样。
- **Task 3 — 五臂同一作业**。**中性检验**:相对 G7 的唯一预期差异是 `count_sentences` 修复。
  任一臂的移动超出该修复所能解释的范围,即说明审阅标记漏进了指标,报告前必须修正。
- **Task 4 — 留出集协议预注册**,见 `docs/generator/heldout-preregistration.md`,
  **在任何留出数据加载之前提交**。判据按你的要求写成**共同作答 query 上的配对比较**,
  且以**固定数值**(−0.018)为参照,而非另一个同时在动的臂均值 ——
  连续两轮判据被触发并证明规格有误(G6 参照臂在其下方变动;G7 跨不同作答集比 arm 均值,
  结构性惩罚任何 coverage 增益),**第三次就会被读成"结果不合意就重新解释规则"**。
- 数据合规:本轮只用 ALCE/ASQA;**HotpotQA / RGB / MuSiQue-Full 从不加载**。

**中性检验的量化门槛(把新切句器作用于 G7 自己的答案算出,运行前写下):**

| 臂 | G7 句数 | 新规则下句数 | delta | 受影响答案 |
|---|---|---|---|---|
| baseline | 427 | 415 | **−12** | 10 |
| verify-only | 275 | 268 | **−7** | 6 |
| verify-annotate-capped | 328 | 320 | **−8** | 7 |
| verify-annotate-open | 391 | 383 | **−8** | 7 |
| verify-annotate-nogate | 455 | 446 | **−9** | 8 |

外加:open 与 nogate 各**救回 1 条**被 validator 误拒的答案(coverage +1/394)——
G7 的记录里看不到,因为被拒的答案根本没写盘。

**判读规则:** 答案是重新生成的,不预期逐字复现;上表是**量级门槛**。
**baseline 与 verify-only 不含任何标记,若这两臂的移动超出上表,那是生成随机性;
而 nogate 若相对 open 出现超出该量级的、方向一致的偏移,即须怀疑审阅标记漏进了指标。**

**AFTER(job `18307720` 生成 01:31:24 + `18307721` 评分 00:16:24,均 exit 0):**

- **头条复现。** nogate vs baseline:coverage **−0.015(p=0.352)**、correctness **−0.006(p=0.572)**
  —— 两轴仍与 baseline 无差异;引用精度 **+0.203(p=0.0)**、召回 **+0.079(p=0.002)**。
  对照 G7 的 −0.018 / +0.001 / +0.191 / +0.071,**在一批独立生成的答案上全部复现**。
  nogate vs open 同样复现:coverage +0.128、correctness +0.048(均 p=0.0),
  精度 −0.014(p=0.145)、召回 −0.002(p=0.924)。
- **Task 1 已在它摧毁的那条 query 上验证。** G7 丢失的 `-5608871660568079389` 在 G8 存活:
  `'Matt Kuchar won the 2018 U.S. Open golf championship. [unverified]'`,句数 1 = 标记 1。
  G8 日志中 **`ValidationError` 一次都没有出现**,每臂错误由 5–6 降到 **2**,且两条都是
  `LLM output must be valid JSON`,与此无关。
- **Task 2 措辞逐字:`[may warrant review]`。** nogate 标记 66 条,**其余四臂均为 0**
  —— 门开启时 would-drop 会被丢弃,故消融臂**在构造上**不可能产生该标记。
  **本轮富集:** 被标样本精度 **0.763**(n=59)、未标 **0.897**(n=247),
  错误率 0.237 对 0.103,**lift 2.3×**(G7 为 2.9×)。以本轮数字为准,不沿用 G7。
- **Task 3 —— 预设的跨运行中性检验不可用,已换成更强的证明。**
  **generation 跨作业不复现:baseline 臂不含本轮任何改动,却有 11% 的 query 结果不同。**
  各臂逐条相同率:baseline 356/400、verify-only 345/395、capped 347/395、open 337/394、
  nogate 331/394(忽略标记)。解码是贪心(`temperature=0.0`),seed、代码、模型、
  **甚至节点(`bp1-gpu035`)都与 G7 相同**,但本次快 2.3 倍(13.5 对 31.0 s/case)。
  贪心在算术上确定,在 GPU kernel 选择与归约顺序上不逐位确定,logit 接近平手处即翻转。
  **故 baseline 自身的 G7→G8 位移就是噪声底:** coverage +0.008、correctness +0.007、
  精度 +0.016、召回 +0.012。**nogate 的位移(+0.006 / −0.001 / +0.013 / +0.009)全部落在其内**,
  两个方向都不可归因;预注册的 −7~−12 句差在 11% 的答案级混杂面前根本测不出来。
  **这印证并扩展了"五臂同一作业"的纪律:同作业内比较可靠,跨作业不可靠 ——
  即便固定 seed、代码与硬件。本文档中每一个结论都是同作业内的。**
- **中性性改用精确证明(无需运行):** 用 G8 自己的记录重建 nogate 的评分输入,
  一次照原样、一次剥除全部审阅标记 —— **送进判官的 `ScoredExample` 序列逐字节相同**,
  kept 451 对 451、annotated 80 对 80、coverage 与 STR-EM 完全相同。
  **标记不可能移动任何指标,因为算指标时它已经不在了。** 这比任何跨运行比较都强。
- 附带:G7 那条失败判据(cited-sample 精度低于 verify-only)**本轮不再触发**:
  0.8715 对 0.8710。**该差距是 0.0005,应读作"这是同一个数",不是通过。**
  判据本就规格有误,留出集的判据因此换了写法。
- **句子构成:** baseline 384/0;verify-only 257/0;capped 290/15;open 290/82;
  **nogate 358 已引用 / 80 标注 / 其中 66 条为已引用且被标记 / 共 438**。
  路由:439 声明,359 验证,80 标注,**0 丢弃**;门本会摧毁 **66**,**全部 66 条改为被引用**,
  控制臂自检精确相等(66 = 66)。

- **作业号已核实(2026-08-08),本条与 G7 条目的号都是对的。** 按 id 直查:

  | JobID | JobName | State | Elapsed | Exit | Submit | Start |
  |---|---|---|---:|---|---|---|
  | `18295681` | g5-annotate | COMPLETED | 03:23:50 | 0:0 | 08-07T11:51:35 | 08-07T20:52:46 |
  | `18295682` | g5-score | COMPLETED | 00:20:29 | 0:0 | 08-07T11:51:35 | 08-08T00:33:47 |
  | `18307720` | g5-annotate | COMPLETED | 01:31:24 | 0:0 | 08-08T01:32:10 | 08-08T02:34:04 |
  | `18307721` | g5-score | COMPLETED | 00:16:24 | 0:0 | 08-08T01:32:10 | 08-08T04:05:28 |

  Elapsed 与 G7/G8 两条目记录的逐项吻合,**追溯链完整**。
- **⚠ 但一个查询侧的异常仍未解释,且它比作业号本身更值得记:**
  `sacct -X -S 2026-08-07` 的输出里**这四条一条都没有**,而同窗口的其他十条都在
  (`18288533`、`18290519`、`18290571`、`18299845`、`18299952`、`18300249`、`18300333`、
  `18308579`、`18318915`、`18318996`)。**原先写在此处的"`-S` 窗口效应"解释已被证伪** ——
  四条的 Start 全部落在 2026-08-07 之后,都在窗口内。原因不明。
  **这是又一个"产出合理数字而非崩溃":** 那次查询返回十行、格式完整、无报错、
  没有任何迹象表明少了四条;若不是按 id 直查,唯一自然的读法就是"台账的号写错了" ——
  **一个错误的结论,由一份看不出残缺的输出支撑。**
  ⇒ **纪律:核对作业时按 id 直查,不要用 `-S` 时间窗口做存在性判断。**
  时间窗口能证明"在",不能证明"不在"。

**标定到此冻结,不再有设计改动。** 留出集协议见 `docs/generator/heldout-preregistration.md`。

---

## 收尾轮 — 记录更正、留出集管路

**Task 1 — 交叉运行可复现性的说法已更正(两处文档)。**
G7 曾写"交叉运行一致性已验证而非假定",G8 却测得**不含任何改动的 baseline 臂**仅
356/400 复现 G7。两者不能并存,已对着作业日志判定:

| 配对 | 每 arm 速度 | baseline 逐条相同 |
|---|---|---|
| G6 `18269703` → G7 `18295681` | 6.03 → 6.06 s/arm/case | **394/394** |
| G7 `18295681` → G8 `18307720` | 6.06 → **2.70** s/arm/case | **356/400** |

**"G7 复用缓存"已排除** —— G7 日志有完整生成过程、速度与 G6 一致、每臂 answered 与
error 计数均为新产出,且 runner 无任何缓存路径。**成立的解释是执行条件恰好相同。**
贪心解码在算术上确定、在 GPU 归约顺序与 kernel 选择上不确定。
**观察成立,推论撤回;由此确立规则:任何被报告的比较都不得跨作业。**
未被改动的 `capped` 臂引用精度 G7 读 0.890、G8 读 0.911,**该 0.021 已被点名为漂移而非结果**。

**Task 2 — 残余解析失败已刻画。** G8 每臂 2 条,**与 G7 是同两条**(G6 4 条、G7 6 条、G8 2 条,
G8 ⊂ G7)。两条均为 claim splitter 的 JSON 发射失败,**在任何可测属性上都不聚集**
(证据词数 500 = 语料中位数、问题长度正常、baseline 均能正常作答)。
**它们发生在共享的上游阶段**,四个验证臂丢的是同两条,配对检验按 query_id 匹配,
故**不会污染任何臂间比较**;但在 arm 级**使验证臂占了 +0.003 coverage 的便宜**(252/398 对 252/400),
已写入局限。按冻结要求未修。

**Task 3 — 富集数字已加区间。** flagged 错误率 0.237,95% Wilson **[0.147, 0.360]**,n=59;
unflagged 0.103,**[0.070, 0.145]**,n=247。**两区间确实分离,但只差 0.002**;
按区间端点,lift 可能落在 **1.01×–5.18×**。故正确说法是"被标群的引用错误率在 95% 水平上更高,
但其**倍数**在 n=59 下并未被确定"。跨轮估计本身从 2.9× 滑到 2.3×,说的是同一件事。

**Task 4 — 留出集协议已补"作业中途失败"条款:**
**结果产生前**的基础设施故障(OOM / 超时 / 节点故障 / 模型加载失败)**可重跑**,未观察到任何东西即无污染;
**结果一旦产生即定稿,无论其内容如何,不得重跑**。**边界是评分器,不是"谁看没看"** ——
以人是否记得看过为判据的规则不成其为规则。生成完成但评分失败者**只重评分不重生成**(同输入同判官下评分确定)。
每次重跑及其理由当场记入本台账。**唯一明确不允许的:因为数字不好看而重跑。**

**Task 5 — 留出集管路(schema 层已通过,GPU 层排队中 job `18318986`)。**
三个集此前从未有 loader。已建 `scripts/heldout_data.py` + `heldout_dryrun.py`,
**只报 schema 与记录数,不算任何指标、不导入评分器、不打印任何答案文本**。
干跑当场抓到两个源错误:**HotpotQA 的 CMU 源在集群上超时**(表现为卡住而非拒绝)—— 已换 HF 镜像;
**MuSiQue 原指向的仓库只有 `musique_ans`(可答半边)** —— 那会把留出集悄悄换成更容易的任务,
已改为显式读取 `musique_full_v1.0_dev.jsonl`。
最终:hotpotqa **7405**、musique-full **4834**、rgb **300**,schema 问题 **0**。

**并由此触发一条预注册补充(在任何留出结果存在之前):**
实测而非假设地确认 MuSiQue-Full dev 是 **2417 个 id 各出现两次**(可答/不可答),
且**两个变体都带非空 gold answer**。不可答项的支撑段落已被移除,
**字符串匹配"正确性"会因此奖励系统硬答、惩罚弃答或标注未验证 —— 恰好是本项目主张的反面。**
故:两变体 id 加后缀区分;**两个子集分开报告**;**预注册判据在可答子集上评估**;
不可答子集上的 STR-EM 照报但**不称作 correctness**;并报联合数字以免有所隐藏。

**Task 7 — 冻结产物已出:`docs/generator/frozen-results.md`**,五节均标注作业号与判官,
且**没有任何一张表混用作业**。

---

## G9 — 一次性修复轮 + 标定重跑(此后硬冻结)

**改动本身在运行前提交(`27e8280`);六项改动中五项行为中性,各有测试直接比对两种配置而非论证其等价。**

**AFTER(job `18322642` 生成 03:13:39 + `18322643` 评分 00:20:33,均 exit 0):**

| arm | coverage | STR-EM | 引用精度(ALCE) | 引用精度(cited) | 引用召回 | answered |
|---|---|---|---|---|---|---|
| baseline | 0.925 | 0.266 | 0.597 | 0.597 (370) | 0.635 | 370/400 |
| verify-only | 0.641 | 0.206 | 0.862 | 0.862 (253) | 0.862 | 253/395 |
| capped | 0.635 | 0.202 | 0.890 | 0.890 (251) | 0.870 | 251/395 |
| open | 0.785 | 0.223 | 0.720 | 0.890 (251) | 0.705 | 310/395 |
| **nogate** | **0.911** | **0.267** | 0.716 | 0.848 (304) | 0.701 | 360/395 |

- **nogate vs baseline:** coverage **−0.0152(p=0.386)**、correctness **+0.0015(p=0.892)**
  —— 两轴仍与 baseline 无差异;引用精度 **+0.1913(p=0.0)**、召回 **+0.0711(p=0.0094)**。
- **nogate vs open:** coverage +0.1266、correctness +0.0441(均 p=0.0);
  精度 −0.0046(p=0.698)、召回 +0.0043(p=0.720)。
- **observe-only 实测:** 门本会摧毁 **60** 条,**60 条全部改为被引用**、0 条落到标注;
  控制臂自检精确相等(60 = 60)。declared 引用存活 283/399(0.709)。
- **审阅标记富集:** 被标错误率 0.322,95% Wilson **[0.221, 0.456]**,n=58;
  未标 0.112,**[0.080, 0.160]**,n=246。**lift 2.86×**,端点 1.38×–5.70×,**区间干净分离**。

**这一轮最有价值的结果不是上表,而是可归因性:**

> **G9 与 G7 在每一个共享 query 上逐字节相同** —— baseline 400/400,其余四臂各 395/395。
> 唯一差异是 open 与 nogate 各**多出一条** query:`-5608871660568079389`
> (`"Matt Kuchar won the 2018 U.S. Open golf championship."`),即 G7 被 validator 误毁、
> 现已存活的那一条。

指南预期"G9 减 G8 的任何差异都会与 11% 的生成 churn 混杂,无法归因"。**在本轮不成立** ——
G9 落回 5.78 s/arm/case(G7 为 6.06,G8 为 2.70),与 G7 执行条件相同,**churn 为零**,
因此差异**完全可归因**。这是运气(执行条件),不是我控制的东西,但它同时为跨作业不确定性
提供了一个非设计得来的确证:

| 配对 | 每 arm 速度 | baseline 逐条相同 |
|---|---|---|
| G6 → G7 | 6.03 → 6.06 | **394/394** |
| G7 → G8 | 6.06 → **2.70** | **356/400** |
| G8 → G9 | 2.70 → **5.78** | **356/400** |
| **G7 → G9** | 6.06 → 5.78 | **400/400** |

**执行条件相同则一致,不同则不一致,与先后顺序无关。**

**切句器统一的隔离测量(在固定输入上,非 G9 减 G8):**
1808 条已记录答案上,**69 条(3.816%)边界改变**,句数 2336 → 2262(−74),全部是把错切的句子合回去
(`Patrick S. | Castagne`、`Brown v. | Board`、`"The Parent Trap. | "`)。
**该测量当场抓到一个本会发出去的回归:** 第一版统一规则要求终止符后紧跟空白,于是 `."` 永不切分,
`'"Manifest Destiny." It means …'` 被并成一句 —— **合并两个真句是更糟的方向**,
因为契约靠数句子判断"每个无引用句是否都打了标记",少数一句就会放过未标注的句子。运行前已修并钉死测试。
**对标定答案的净影响:1 条 query。** 即该统一移除的是**潜在**不一致,而非正在发作的。

**留出集管路干跑(job `18318986`,COMPLETED 00:03:06):** 三个集 schema 问题 **0**,
hotpotqa 7405 / musique-full 4834 / rgb 300,每集 3 条样本在 baseline 与 nogate 两臂上
**均产出合法结果对象**。**未计算任何指标、未检视任何输出。**

**Task 8(在任何留出数据加载前完成):** 预注册参照值 −0.018 → **−0.0152**,
**只动参照点,判据形式/比较方式/阈值/跨数据集规则一律不变**,旧值保留在案;
`frozen-results.md` 已在 G9 上重建,§4 不与 G8 混用任何数字。

---

## 模块级留出集 — QAMPARI(系统冻结于 G9,`src/` 零改动)

**背景:** HotpotQA / RGB / MuSiQue-Full 是**团队的系统级**留出集。用它们做 Generator 的模块级评估
会把系统测试花掉,而等整合流水线会把本模块的结果押在别人的时间线上。故 Generator 需要自己的留出集。
三个集的准备工作**未浪费**,已打包移交(`docs/heldout-three-set-handover.md`)。

**QAMPARI 从未被加载过 —— 已核实而非假定:** 任何被跟踪文件中无引用,
`git log --all -S` 在任何分支的任何提交中均无。ALCE 压缩包自 07-28 起就在盘上,
但**只抽取过 ASQA 那个成员**,QAMPARI 成员从未被打开。

**预注册修订已在任何加载之前提交:** commit `c559d5c`,**2026-08-09 15:57:20 +0100**。
**只改数据集与 correctness 指标**;判据形式、共同作答配对、−0.0152 参照、ALCE 主口径、
只加载一次、如实报告、事后不改系统、部分复现规则、失败/重跑条款**全部不变**。旧文保留在页面上。

- **证据设置与标定完全一致:** `qampari_eval_gtr_top100.json` 是标定所用
  `asqa_eval_gtr_top100.json` 的同族文件 —— 同检索器(GTR)、同 top-100 池、同 **top-5** 截断。
  实测两者证据规模一致(5 段 / top-5 共 500 词 / 每段 100 词),故比较测的是方法而非证据质量。
- **correctness 换成"按包含计的答案召回"**(对每个 gold 别名集,答案含任一别名即计中),
  **不加帽与 @5 各报一次**。三点提前写明:①**不是** ALCE 官方 QAMPARI F1
  (那个把输出解析成逗号分隔实体表,会误读本系统所有臂的散文输出);
  ②与 ASQA 的 STR-EM **是同一个函数**,故 correctness 轴内部一致,但**跨数据集数值不可比**;
  ③**只有召回没有精度**(无表解析则"预测实体"无定义),故无法惩罚冗长作答 ——
  对各臂一致施加,不偏袒比较,但确实使 QAMPARI 的 correctness 弱于 ASQA。
  **引用指标不变,且是主张所依赖的那根轴。**

**干跑(schema 层已过,GPU 层排队中 `18325888`):**

| | |
|---|---|
| 记录数 | 1000 |
| 每条段落数 | 5(min=max=5) |
| top-5 词数 | 500(min=max=500) |
| 每段词数 | 100 |
| **每条 gold 答案数** | min 4 / **中位 8** / p90 28 / **max 200** |
| **>5 个 gold 答案的记录** | **770(77.0%)** |
| schema 问题 | **0** |

77% 的记录 gold 答案多于 5 个 —— **这就是预注册里必须同时报 rec@5 的原因**,
而它是在看到这个数字之前定下的。

**声明切分探针(CPU)按其构造无法回答问题,如实记录:** stub LLM 每句返回一条声明,
而逗号分隔的实体表就是一句,故它必然返回 1 条 —— 测的是 stub 不是 Granite。
**真正的问题(多答案问题会切成"每实体一条声明"还是"一条声明扛五个实体")由 GPU 干跑回答**,
已改为在真实草稿上报告每条答案的路由声明数与结局分布。

**作业链:** `18325888`(干跑)→ `18326078`(五臂 400 题 seed 13)→ `18326079`(评分),
以 `afterok` 串联 —— **管路不过则正式跑根本不会启动。**

**AFTER(`18325888` 干跑 + `18326078` 生成 01:44:26 + `18326079` 评分 00:27:26,均 exit 0):**

**预注册判据,先报这个:**

| 合取项 | 结果 | 判定 |
|---|---|---|
| 引用精度 > baseline,p<0.05 | **+0.2685**(0.8499 vs 0.5813),**p=0.0**,CI [0.198, 0.336],n=233 | **满足,且幅度更大** |
| coverage 赤字不劣于 −0.0152 | **−0.0157**,p=0.469,CI [−0.050, +0.018],n=382 | 按实测满足 —— **但见下** |

**引用精度这一半复现且无疑义:标定的 +0.19 在一个从未加载过、任务形态不同的数据集上变成 +0.27。**

**切分器条款触发了(预注册在先,非事后):**
失败 **18/400 = 4.50%**,对比标定 0.5%(G8)/ 1.25%(G9),**九倍**。全部是
`LLM output must be valid JSON`。故 **QAMPARI 上的 coverage 比较按预注册报告为不可靠。**

条款触发的理由是**实测的、不是预防性的**:baseline 在这 18 条上答出 **17/18(0.944)**,
而在其余 382 条上只有 **0.801**。切分器在所有验证臂的上游,故其失败把同一条 query 从所有验证臂移除,
却把它留给 baseline 计分 —— 而**那些恰好是 baseline 表现更好的题**。

| 处理方式 | nogate | baseline | 赤字 |
|---|---|---|---|
| 配对、剔除失败(所报) | 300/382 = 0.785 | 306/382 = 0.801 | **−0.0157** |
| 失败计为未作答(上界) | 300/400 = 0.750 | 323/400 = 0.808 | **−0.0575** |

**−0.0157 过判据,−0.0575 不过。真值在两者之间,本次运行无法确定在哪。**
诚实的读法是:**coverage 这一半在本集上未被检验,而不是通过了。**
失败**并非集中在多答案题**(失败题 gold 中位 7.5,其余 8.0),故不是干跑提出的"实体表"假说,
而是新分布上 JSON 畸形率本身更高。

**五臂(correctness = 预注册的按包含计答案召回,与 ASQA 的 STR-EM 不可比,且只有召回):**

| arm | coverage | recall | rec@5 | 精度(ALCE) | 精度(cited) | 引用召回 | answered |
|---|---|---|---|---|---|---|---|
| baseline | 0.808 | 0.075 | 0.121 | 0.535 | 0.535 (323) | 0.578 | 323/400 |
| verify-only | 0.597 | 0.055 | 0.086 | 0.802 | 0.802 (228) | 0.802 | 228/382 |
| capped | 0.579 | 0.052 | 0.084 | 0.844 | 0.848 (220) | 0.828 | 221/382 |
| open | 0.720 | 0.058 | 0.096 | 0.679 | 0.848 (220) | 0.665 | 275/382 |
| **nogate** | **0.785** | **0.065** | **0.107** | 0.696 | **0.848 (246)** | 0.684 | 300/382 |

- **引用召回同样复现:+0.0929(p=0.010)**,标定为 +0.071。
- **correctness 未复现:** ASQA 上与 baseline 无差异(+0.0015,p=0.892),此处**落后 −0.0105(p=0.025)**;
  rec@5 −0.0141(p=0.101,不显著)。如实报告。两点相关但不消解该结果:指标只有召回,
  奖励罗列而从不惩罚冗长;且各臂绝对值都极低(0.05–0.12),因为 5 段证据装不下中位 8 个 gold 实体。
- **门依旧只花钱不办事:** nogate 相对 open,coverage +0.065、correctness +0.007,
  而两根引用轴无差异(p=0.35 / 0.87)—— 与标定同形。
- **路由:** 441 声明 → 364 验证 / 77 标注 / **0 摧毁**;门本会摧毁 **48**,**48 条全部改为被引用**;
  控制臂自检精确相等(48 = 48);declared 引用存活 310/400 = 0.775(标定 0.709)。
- **审阅标记未复现:** 被标错误率 0.184(n=44)对未标 0.145(n=202),**lift 1.28×**,标定为 **2.86×**。
  **该筛查主张是标定专有的,不得作为实体层的一般性质陈述。**

**答案长度:** 干跑实测 baseline 最长 37 token、nogate 34 token,**远低于 256 上限**,无截断风险。

**结论:项目所依赖的头条(经独立验证的引用,精度与召回均显著优于生成时引用)在未接触过、
任务形态不同的数据集上复现,且幅度更大。两项未复现的(correctness、审阅标记富集)按未复现报告。**

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

---

## R11 — NQ 上那个"分解赢了"的结果,是不是语料规模的函数 [PRE-REGISTERED 2026-08-12]

### 为什么要跑

NQ 上 `decompose-orig` 的总召回**显著高于** strong-bm25(+0.0122,p=0.0000,CI [+0.0061,+0.0187],
n=2000),而 MRR(−0.0076,p=0.1157)与 R@10(−0.0067,p=0.1185)都不显著。这是**任何分解臂
第一次在任何数据集的任何指标上超过基础检索器**,直接限定了 R4 "修好了仍然不值得"的结论。

但它只在 NQ 上出现,SciFact 与 2Wiki 毫无迹象,**机制未定**。最省事的一个候选解释是语料规模:
NQ 语料十万段落,SciFact 只有 5183 篇。若子查询的作用是"扩大候选池的多样性",那么可错过的东西
越多、它越该有用。这个解释可以被直接证伪,而且不需要新代码。

### 假设与可证伪结局(读数前写定)

**假设 H:** Δ(总召回, decompose-orig − strong-bm25) **随语料规模单调上升**。

- **证伪 A —— 曲线在 25k→200k 之间是平的(各点 CI 重叠)**:效应与语料规模无关,NQ 的特殊之处
  在别处(最可能是问题形态:NQ 是单跳事实问答,SciFact 是科学声明核查,2Wiki 是多跳)。
  这是有价值的否定,它把解释空间从"规模"挪到"问题形态",并直接指向下一个实验。
- **证伪 B —— 曲线下降**:与该解释正相反,H 直接作废。
- **证伪 C —— 只有 100k 显著、两侧都不显著**:该点是侥幸的嫌疑大增。本 sweep 的设计目的
  之一就是让这种情况暴露出来,而不是让一个孤立显著值留在报告里。

**注意 H 为真也不等于"分解值得用"**:每条 query 仍要多付 N 次 LLM 调用与 N 次检索。H 只决定
这份收益是否会随规模增长,即它是否值得在更大语料上重新评估。

### 设计

- **语料规模:** 25k / 50k / 100k(已有,复用)/ 200k。materializer **保留全部 gold 文档**,
  只用带种子的蓄水池采样填充干扰文档到指定规模(gold 缺失会直接报错),所以 query、gold、
  切片与检索器参数全部固定,**唯一变动的是干扰项数量**。
- **臂:** `strong-bm25` 与 `decompose-orig`,两者配对求差。四个指标全报,但**预注册的量是
  总召回的 Δ**;其余三个作为形状参考,不作为结论依据。
- **LLM 方差不是混淆项:** 生成为 greedy(`temperature=0.0` ⇒ `do_sample=False`),同一条 query
  在各 sweep 点得到**相同的子查询**,故点与点之间的差异不可能来自分解结果的抖动。
- **命令:**

  ```bash
  for s in 25000:25k 50000:50k 200000:200k; do
    n=${s%%:*}; tag=${s##*:}
    python -m evidence_rag.materializer.base_cli --split dev \
      --output runs/niah-base-$tag --corpus-size $n --query-limit 2000 --seed 42
  done
  sbatch scripts/run_retriever_eval.slurm \
    configs/experiments/retr_nq{25k,50k,200k}_{strong-bm25,decompose-orig}.toml
  # 读数(login node):
  for tag in 25k 50k 200k; do bash scripts/retriever_significance.sh nq$tag; done
  bash scripts/retriever_significance.sh nq   # 已有的 100k 点
  ```

### 已知限制(写在读数之前,免得事后当成解释)

- **各规模的语料不是嵌套子集。** 蓄水池采样在不同 `corpus_size` 下给出不同的抽样轨迹,所以
  50k 的干扰集**不是** 100k 的子集。规模与成分同时在变;n=2000 且抽样随机,预期不引入系统偏差,
  但这不是配对设计,曲线上的点之间有额外方差。
- **绝对召回会随语料增大而下降**(干扰更多),两个臂都是如此。**结论只看配对差值**,不看绝对值。
- 只有一个数据集。即使 H 成立,它解释的也只是 NQ 内部的规模依赖,**不能推广成"分解在大语料上
  普遍有用"** —— 那需要在 SciFact 或 2Wiki 上放大语料复现同样的斜率,而两者都没有那么多文档。

### R11 AFTER — H 被证伪(结局 A),但效应本身 4/4 复现 [2026-08-13,jobs 18431334 / 18431335,均 COMPLETED 0:0]

**总召回 Δ(decompose-orig − strong-bm25),n=2000/点:**

| 语料 | Δ recall | p |
|---|---|---|
| 25k | +0.0105 | 0.0001 |
| 50k | +0.0145 | 0.0001 |
| 100k | +0.0122 | 0.0000 |
| 200k | +0.0122 | 0.0013 |

**H 作废,走的是预注册的结局 A —— 曲线是平的。** 语料放大 8 倍,效应量在 +0.0105 ~ +0.0145
之间无趋势,四个点全部落在彼此的 CI 内(100k 点的 CI 为 [+0.0061, +0.0187])。"子查询扩大候选池
的多样性,所以语料越大越有用"这个解释**不被数据支持**。

**但结局 C 被彻底排除,这比 H 成立更有价值:效应在四个规模上 4/4 全部显著为正。** 单点显著时
最该担心的就是侥幸;现在它在独立采样的四个语料上复现了同一量级。设计的对照也如预期成立:
strong-bm25 的绝对召回随语料单调下降(0.9307 → 0.9202 → 0.9098 → 0.8965),说明干扰项确实在起
作用,而配对差值不受其影响。

**解释空间因此移动:** 既然不是规模,NQ 的特殊之处更可能在**问题形态** —— NQ 是单跳事实问答,
SciFact 是科学声明核查,2Wiki 是多跳。这是预注册里就写明的、结局 A 应当指向的方向。下一个实验
应当在这条轴上设计,而不是继续扫规模。

**一并修正 R4 Step 5 的一句话(基于 100k 单点写下的,四点看来不成立):**

| 语料 | Δ MRR | p |
|---|---|---|
| 25k | −0.0170 | **0.0005** |
| 50k | −0.0175 | **0.0001** |
| 100k | −0.0076 | 0.1157 |
| 200k | −0.0096 | 0.0509 |

R4 Step 5 写的是"在顶端没有可测量的代价(MRR 与 R@10 均不显著)"。那在 100k 上为真,**在 25k 和
50k 上为假** —— MRR 的损失在那两点是显著的。100k 恰好是四点中最不显著的那个,而当时只有它。

**所以真实形状是一个干净的交换:损失 top-rank 精度,换来池覆盖率。** 这正是本报告在 SciFact 上
提出、又因 p=0.5811 而撤回的那个读法;当时撤回对 SciFact 是对的,而在 NQ 上它现在有四个语料规模
的支撑。这也意味着"值不值"取决于下游消费哪个指标:`top_k=50` 喂给保留 5 条的 selector 时,池
深度是这个架构消费的量;若下游只看 rank 1,这笔交换是亏的。
- raw:`results/r11-nq{25k,50k,100k,200k}-{strong-bm25,decompose-orig}-per-case.json`
  (八份,已 `git add -f` 拉回)。**检索阶段报告本就不含 `trace`,故未剥离,是原件逐字副本。**
  **拉回后已用仓库自带的 `paired_metric_cli` 复核:四个 Δrecall 与上表逐位一致
  (+0.0105 / +0.0145 / +0.0122 / +0.0122),`mean_off` 亦复现 .9307 / .9202 / .9098 / .8965。**

### R12 — 问题形态,而不是语料规模:同一对在三个数据集上的读数 [PRE-REGISTERED 2026-08-13]

### 为什么现在就能做,以及为什么必须先写下预测

R11 排除了语料规模,并把解释指向**问题形态**。这条轴不需要新的机时:三个数据集正好是三种形态
——NQ 单跳事实问答、SciFact 科学声明核查、2Wiki 多跳——而 `decompose-orig` 与 `strong-bm25`
的产物在三者上**都已存在**。`decompose-orig vs strong-bm25` 这一对是 2026-08-12 才接进
`scripts/retriever_significance.sh` 的(此前从没有迹象显示它可能为正),所以它只在 NQ 上读过。
SciFact 与 2Wiki 只差一次脚本调用。

正因为几乎零成本、且部分均值已经印在报告里,**更要先写死预测再读 p 值**:事后从三张表里挑一个
说法出来是最容易发生的自欺。

### 假设与可证伪结局(读数前写定)

**假设 H12:** 总召回上的增益是**单跳事实问答特有的**。NQ 的子查询把一个事实问题拆成若干个仍然
自足的事实问题,每个都能独立命中一篇文档,于是候选池被真正拓宽;而声明核查的子句、多跳问题的
单跳,都不是自足的检索目标。

**预测:** Δ(总召回) 在 NQ 上显著为正(已知 +0.0122),在 SciFact 与 2Wiki 上**不显著**。

- **证伪 A —— SciFact 或 2Wiki 也显著为正**:那么增益不是 NQ 特有,H12 作废,而且"分解不值得"这个
  结论要在那个数据集上一并重估。这是三种结局里对现有结论冲击最大的一个。
- **证伪 B —— 某个数据集显著为负**:说明分解在那里主动损害池覆盖率,机制与 NQ 相反,需要单独解释。
- **确证但弱:** 两者都不显著。注意这**只是与 H12 一致**,不等于证明了"因为单跳所以有效"——三个
  数据集在语料、领域、query 长度上同时不同,形态只是其中一条。要真正归因,需要在**同一语料**上
  变换问题形态,那才是下一个真正的实验。这一条必须写在结论里,不能让"一致"被读成"证实"。

### 命令(login node,纯读取,秒级)

```bash
for ds in scifact 2wiki nq; do
  echo "== $ds"
  bash scripts/retriever_significance.sh $ds | grep "decompose-orig vs strong-bm25"
done
```

### 已知限制

- **三个数据集不是受控对照。** 形态、语料规模、领域、gold 密度同时不同。R11 已经排除了规模这一条
  (在 NQ 内部),但没有排除领域或 gold 密度。所以本条最多能说"增益只在 NQ 出现,与形态解释一致",
  不能说"形态是原因"。
- **部分均值此前已出现在报告中**(SciFact decompose-orig 总召回 0.8666 vs strong-bm25 0.8624;
  2Wiki 0.7675 vs 0.7678),所以方向不是全新信息;**未知且被预注册的是显著性**。这一点如实记录,
  避免把"早已可见的方向"包装成新发现。

### R12 AFTER — 预测成立,但 SciFact 的"不显著"不能当证据用 [2026-08-13]

**总召回 Δ(decompose-orig − strong-bm25):**

| 数据集 | 问题形态 | Δ recall | p | n |
|---|---|---|---|---|
| NQ | 单跳事实问答 | **+0.0122** | **0.0001** | 2000 |
| SciFact | 声明核查 | +0.0042 | 0.6781 | **300** |
| 2Wiki | 多跳 | −0.0003 | 0.9155 | 2000 |

两个证伪结局都没有发生,结果与 H12 一致:增益只在 NQ 上显著。

**但读数时发现一条预注册没有想到的限制,必须写在结论前面:SciFact 的样本量不足以否定一个
NQ 量级的效应。** n=300,是 NQ 的 1/6.7。NQ 的 CI 半宽为 ±0.0063;若每例方差同量级,按
√(2000/300)≈2.58 折算,SciFact 的半宽约 **±0.016** —— **即使 SciFact 上真存在 +0.0122 的效应,
这个样本量也看不出来**。所以 SciFact 的 p=0.6781 是"没有证据",不是"证据表明没有",两者在
本条结论里不能互换。(方差按数据集不同,上述折算只作量级参考。)

2Wiki 则是干净的空结果:n=2000,Δ=−0.0003,不存在功效问题。

**因此本条能支持的最强表述是:增益在多跳上确证不存在、在声明核查上未知、只在单跳事实问答上
确证存在。** 比预注册的预测弱一档,弱的那一档正是样本量造成的。

**这个缺口靠多跑补不上:** SciFact 的 test split 一共就 300 条 query。要在声明核查这一形态上得到
可用的功效,需要换一个更大的声明核查数据集,或者做本条限制里已经写明的那个真正的实验——
**在同一语料上变换问题形态**。后者同时解决功效与混淆两个问题,是唯一干净的路。

**一个与 R4 Step 1 呼应的细节:** 2Wiki 上 MRR / R@10 / R@20 三个指标全部显著为负,而总召回恰好
为零(−0.0003, p=0.9155)。这正是 R4 Step 1 诊断的形状——**排序失败,候选池完好**——在一个独立
的臂上再次显形。

### 结构审计 × R9 的交叉读数 —— 一句当日提出、当日撤回的建议 [2026-08-13]

`scripts/chunker_structure_audit.py` 在 `docs/` 的 121 份真实 Markdown 上测得:默认 120/20 时
`word` 切开 **46.9%** 的表格,而 `section` 为 0%。据此我在 R7 里写了一句"更便宜的替代方案从没被
比较过 —— 单把 `chunk_size` 提到 240 就能把切开率降到 13.9%",并要求 `section` 先赢过它。

**该建议当日撤回:方向与 R9 的实测相反,而我提出时没有查本台账。**

R9(job 18329959)在**固定证据预算 600 词**下测得 `answer_match` 随 chunk 变小**单调上升**:
60×10 = 0.5405 / 120×5 = 0.5185 / 200×3 = 0.4700 / 300×2 = 0.4145,三个对照对生产值全部显著
(60×10 **+0.0220 p=0.0008**),且最优落在扫描下边界。**加大 chunk_size 是已被测量为更差的方向。**
R9 同时记录了:不控制证据体积会把这个效应的符号整个翻过来——而我那句建议正是在不控制体积的
直觉下提出的。

**两条合起来反而加强了 `section` 的理由:**

| chunk_size | `word` 切开表格 | R9 答案质量(固定预算) |
|---|---|---|
| 40 | 96.5% | 更好(外推) |
| 60 | 82.1% | **实测最优** |
| 120 | 46.9% | 生产基线 |
| 240 | 13.9% | 显著更差 |

**对答案有利的方向正是摧毁表格的方向。** 在实测最优的 60 词处,`word` 切开 82.1% 的表格。
`section` 的边界来自结构而非词数预算,超大表格保持完整,因此它是这里唯一能同时取到"小块"与
"表格完整"的选项。

**两条限制:** R9 跑在 2Wiki 上,是无表格的散文,故它约束的是粒度方向,**不直接说明结构化语料**;
`section` +42% 的块数代价依然成立且依然未定价。

**方法教训(与本条结论同等重要):** 我在提出"更便宜的替代方案"时,只看了自己刚测的那张表,
没有查同一台账里三天前的结果。**新测量必须先与既有结果对齐再下建议** —— 否则最容易发生的
不是测错,而是把一个已被证伪的方向重新提上来。

### 方法纪律 — 手调的合成语料在一天里连错三次,且三次都朝有利方向 [2026-08-13]

**事实。** 2026-08-11/12 我为倒排索引(报告 R9)做了三个测量,全部用手写的合成语料。2026-08-13
在真实文本(仓库自带的 121 份 Markdown)上复测,三个全部被推翻:

| 主张 | 合成语料 | 真实文本 | 偏差 |
|---|---|---|---|
| 倒排相对正排省内存 | 6.3× | **1.6×** | 4× |
| 百万 chunk 内存外推 | 1.06 GB | **7.8 GB** | 7× |
| StrongBM25 的延迟优势 | ~900× | **1.8–3.1×** | ~300× |

三次的**机制方向都是对的**,三次的**量级都被放大**,而且三次都放大成"我刚做的改动更有价值"。

**偏差的来源不是算错,是调参。** 手调一个分布时,人会一直调到效应清晰可见为止——而让效应清晰
的那个参数,恰恰就是让语料不具代表性的那个参数:

- 内存那两条:我用 `paretovariate(1.2)` 得到每 chunk **13.6** 个不同词,真实散文是 **95.3**。
  正排的 `tokens` 随**总**词数走,倒排的 postings 随**不同**词数走——把不同词数压低,就是专门
  给倒排放水。
- 延迟那条:我让 13 个停用词占 45% 词频,于是每个停用词的倒排链覆盖近 100% 的 chunk,而内容词
  被 Pareto 压到极稀疏。这是人造的极端,不是散文。

**代价对比:真实语料一直就在仓库里**(`docs/**.md`),读取成本接近零。我用合成语料不是因为
没有真实数据,是因为合成的"更好控制"——而所谓更好控制,这里就等于更好地控制出我想看到的结论。

**纪律(此后按此执行):**

1. **任何要写进报告或推荐表的量级,必须来自真实文本。** 合成语料只用于演示机制(如"成本随
   倒排链覆盖率线性上升"这种形状),**且必须在原地标注为合成,不得被引用为量级**。
2. **语料与查询分开变。** 延迟那条第一次复测时我同时换了语料和查询风格,分不清是谁的功劳;
   拆开后才知道停用词机制成立(1.8× → 3.1×)而量级不成立。
3. **合成语料必须报告它的分布统计**(此处是每 chunk 不同词数),并与真实语料对照。差三倍以上
   即视为不可用。
4. **自查触发条件:** 当一个测量的结论恰好支持测量者本人刚做的改动时,在真实数据上复测之前
   不得写入报告。今天三条全部命中这个条件。

**仍未清理的合成数字(报告 R9 内,已在原地标注):** 成本随 df 覆盖率的曲线(df=1 时 0.008ms
对 69.5% 覆盖时 4.445ms)、以及"生僻词成本对语料规模持平"。前者是机制演示,按纪律 1 可以保留;
**后者是一个渐近主张,必须在真实语料上按规模复测,当前仅有 1345 chunk 的单点,不成立也不证伪。**

### R6b 遗留项结清 — 缓存内存不是"被否定",是量它的工具看不见 [2026-08-13]

R6b 的 AFTER 把这一项写成:"缓存内存 ~2.9 KB/chunk、100 万 chunk 即 ~2.9 GB 已被真语料否定两次
(R6、R6b)……须换工具(如 `tracemalloc` 在稳态取样),不能继续用 peak RSS。此项列为未测。"

**换工具测了,结果是:那个估计基本正确,"否定"这个词当时用错了。**

`scripts/retriever_memory.py` 以 `tracemalloc` 稳态取样(build 前后各取快照,第二次前 `gc.collect()`,
故只计**保留**不计瞬时),在**真实散文**(仓库 121 份 Markdown,每 chunk 95.3 个不同词,
`chunk_size=180/overlap=30` 与 R5 同设置)上分项测得:

| 组成 | B/chunk |
|---|---|
| `tokens` | ~9,300 |
| **`term_frequencies`(R5 估的就是这一项)** | **~3,200** |
| `document_frequency` | ~4 |
| 正排合计 | 12,539 |

R5 的估计是 **~2,900 B/chunk**,实测 **~3,200** —— 同一量级,偏差约 10%。**估计是对的。**

**为什么 peak RSS 两次都看不见它:** 8778 chunk 时该项约 **28 MB**,而当时 peak RSS 在 **150–160 MB**
量级且由建索引阶段的瞬时分配主导。一个 28MB 的稳态增量藏在这种量级的瞬时峰值之下毫不意外。
R6b 自己的诊断("peak RSS 是错的工具")完全正确,**但由此写下的"被否定两次"是过强的表述** ——
零证据不是反证。这一处措辞据此更正。

**顺带修正一个我自己在报告里写下的错误说法。** 我曾在报告 R9 写"R5 的估计低估了,实际 6.7 KB/chunk"。
那是**拿整个正排索引去比 R5 对单一部件的估计**,而且用的还是不真实的合成语料(每 chunk 仅 13.6
个不同词)。两处都错。已撤回,见报告 R9 与同日的方法纪律条目。

**外推更正:** 按真实散文,正排 12.5 KB/chunk ⇒ 100 万 chunk 约 **12.5 GB**;倒排 7.8 KB/chunk
⇒ 约 **7.8 GB**。R5 当年"~2.9 GB 仅这一项"说的是 `term_frequencies` 单项,按实测应为 **~3.2 GB**,
同样是对的。

**此项自此为已测。** 剩余未测的是它在 SciFact/NQ 上的值 —— 合成到真实的一步就改变了 7 倍,
故数据集之间的差异在实测前应假定为重要。

### R13 — 倒排索引的渐近主张,在真实语料上按规模复测 [PRE-REGISTERED 2026-08-13]

**为什么必须跑。** 报告 R9 做倒排索引的**核心理由**是"成本正比于查询自己的倒排链,而不是语料
规模",由此宣称生僻词查询的耗时对语料规模持平。该主张**只在合成语料上验过**;真实文本上目前只有
1345 chunk 的单点,既不成立也不证伪。同日的方法纪律条目已把它单独点名为未清理项 —— 这一条就是
清理它。R5 当年测出的线性(chunk 数 ×10.33 → 延迟 ×10.47)是**改动前**的曲线,本条测改动后的。

**设计。** 用 R11 已经物化的 `runs/niah-base-200k`,`--sizes` 取**同一语料的前缀**
(12500/25000/50000/100000/200000,跨 **16×**)。前缀是严格嵌套的,故**规避了 R11 里"各规模语料
不是嵌套子集"那条限制**:成分不随规模漂移。查询用数据集自带的真实 NQ 问句(自然语言,含停用词),
不是我构造的词 —— 今天三个合成数字全错的教训就在于此。延迟与内存在**同一作业同一节点**上测
(R6b 的教训)。

**假设 H13 与可证伪结局(读数前写定):**

- **H13:** `ms_per_query` 随语料规模**显著次线性**(远低于 R5 改动前的 ~1.0 斜率)。
- **证伪 A —— 仍然近似线性(比值 ×16 语料 → ×10 以上延迟):** 对**真实 NQ 查询**而言倒排索引
  没有改变渐近性质。这不会推翻"成本正比于倒排链覆盖率"这个机制(它已被 df 扫描证实),但会说明
  **真实自然语言查询的倒排链覆盖率本身就随语料线性增长** ⇒ 报告 R9 关于渐近的说法必须收回,
  只保留常数项收益。**这是最可能发生的结局,因为真实问句含停用词。**
- **证伪 B —— 完全持平:** 强于预期,需检查是否有测量伪影(如 top_k 提前退出)。
- 内存:预期 B/chunk 在各规模上**平坦**;若随规模上升,则"每 chunk 常数"的外推方式作废。

**注意本条不能回答什么:** 它测的是 NQ 的真实问句。生僻词查询的持平性(合成语料上的那条)
**仍然只对生僻词成立**,而本条不构造生僻词查询 —— 因为需要知道的是**真实负载**的渐近行为,
不是最有利情形的。

**命令(compute 节点,纯 CPU):**

```bash
sbatch scripts/run_retriever_scaling.slurm \
  runs/niah-base-200k/manifest.json \
  results/retriever-scaling-nq-inverted.json \
  "12500 25000 50000 100000 200000"
```

**⚠️ 内存上限:** 20 万文档下倒排索引按 docs/ 实测的 7.8 KB/chunk 约需 1.6 GB,`tracemalloc`
本身还会翻倍。作业申请 32G,应当够;若 OOM,砍掉 200000 那一点再报,**不要偷偷降 sizes 后
当作完整曲线报告**。

### R14 — "同预算下 chunk 越小越好"在 NQ 上复现吗 [PRE-REGISTERED 2026-08-13]

**赌注比以往几条都大:这一条若复现,生产设置就该改。** 台账 R9 在 2Wiki 上、**固定证据预算
600 词**的条件下测得 `answer_match` 随 chunk 变小单调上升:60×10 = 0.5405 / 120×5 = 0.5185 /
200×3 = 0.4700 / 300×2 = 0.4145,三个对照对生产值全部显著(60×10 **+0.0220 p=0.0008**)。
⇒ **当前生产值 `chunk_size=120` 已被证明不是最优。** 但它只有一个数据集,所以至今没人敢动;
改 `chunk_size` 会改 `corpus_signature`,进而作废所有已持久化索引与已记录数字,确认必须够硬。

**为什么是 NQ:** 它是三个数据集里除 2Wiki 外唯一带标准答案文本的(`base_loader` 从 dpr-w100
的 `answers` 字段取,写进 `GoldCase.reference_answers`),而本设计的主指标 `answer_match` 没有
答案就无法评分——SciFact 因此**不可用**,这不是选择而是约束。

**设计:** 完全镜像 R9 的 set B,只换数据集。四点 (60,10)/(120,5)/(200,3)/(300,2),
overlap = chunk_size/6,`chunk_size × max_selected ≡ 600 词`,retriever 全程 strong-bm25
(CPU only,不引入 LLM 成本与方差),`top_k=50`。除 `[chunker]` 与 `max_selected` 外逐字相同。

**假设 H14:** `answer_match` 随 chunk 变小**单调上升**,且 60×10 显著优于 120×5。

- **证伪 A —— NQ 上不显著或反向:** 粒度效应是 **2Wiki 特有的**(多跳:答案分散在多篇文档,
  更多小块直接提高覆盖到不同 gold 文档的机会;NQ 单跳则不然)。**这会把 R9 的结论从"粒度是主旋钮"
  收缩为"多跳语料上粒度是主旋钮",并解除改动生产值的理由。** 这是最有价值的一种否定。
- **证伪 B —— 单调但方向相反(大块更好):** 与 R9 直接冲突,两条不可同时为真;须先排除
  证据体积是否真的被控住(核对四点的 `selRecall`/`sysRecall` 与选中文本词数)。
- **确证:** 两个数据集同向 ⇒ 建议把生产值下调,并**在同一提案里**给出迁移成本
  (重建索引、既有数字作废的范围)。**不得只报质量收益而不报迁移代价。**

**必做的自检(照抄 R9 的纪律):** `chunk_nq_b-c120o20` 是生产值锚点,其 retriever 指标必须与
既有 `runs/retr-nq-strong-bm25` **逐位一致**(MRR .8153 / Recall .9098)。不一致即说明配置或
语料有别,**先查清再读结果**。

**⚠️ 本条不测什么:** 不测 set A(不控制体积的那组)。R9 已经证明不控制体积会把符号翻过来,
重跑一遍只会重复一个已知的错误答案,浪费机时。

**命令(⚠️ 修正于 2026-08-13,首次提交用错了脚本):**

`run_retriever_eval.slurm` 只跑 `prepare → retriever`,**不产生 `generator_report.json`**,
而本条的主指标 `answer_match` 是系统级指标,必须走完整 pipeline。首次提交(job 18445256)
用了它:四个臂的检索部分正常完成,锚点自检也**逐位通过**(MRR .8153 / Recall .9098),
但读数时才发现根本没有 generator 报告。**台账 R9 用的是 `run_pipeline_eval.slurm`,
并以 `--partition=compute --gres=none` 覆盖到 CPU 分区** —— 本条照抄即可。

重交前须先归档已有产物:`prepare` 对已存在的输出目录做严格校验(含 `source_tree_signature`),
源码树自那次运行后变过,直接重交会失败(与 job 18380601 同一种失败)。

```bash
mkdir -p runs/_archive-$(date +%Y%m%d)
mv runs/chunk-nq-b-c{60o10,120o20,200o33,300o50} runs/_archive-$(date +%Y%m%d)/
sbatch --partition=compute --gres=none --time=04:00:00 \
  scripts/run_pipeline_eval.slurm \
  configs/experiments/chunk_nq_b-c{60o10,120o20,200o33,300o50}.toml
# 读数(login node):
python -m evidence_rag.evaluation.paired_metric_cli \
  --on-report runs/chunk-nq-b-c60o10/generator_report.json \
  --off-report runs/chunk-nq-b-c120o20/generator_report.json \
  --metric system.core.answer_match
```

### R13 AFTER — H13 被证伪,且比预注册押的更糟:不是线性,是超线性 [2026-08-13,job 18442787,COMPLETED 00:06:46]

**延迟(真实 NQ 问句,50 query/点,`chunk_size=180/overlap=30`,同作业同节点):**

| chunks | mean ms | p50 ms | p95 ms | ms/1k chunks |
|---|---|---|---|---|
| 12,500 | 3.375 | 2.218 | 12.8 | 0.27 |
| 25,000 | 9.823 | 4.552 | 30.6 | 0.39 |
| 50,000 | 17.929 | 8.304 | 61.9 | 0.36 |
| 100,000 | 45.021 | 17.37 | 278.8 | 0.45 |
| 200,000 | 110.376 | 54.232 | 385.6 | **0.55** |

**命中预注册的证伪 A,而且更强。** 语料 **16×** → mean 延迟 **32.7×**、p50 **24.4×**,拟合指数
约 **1.15–1.3**。`ms/1k_chunks` 从 0.27 单调涨到 0.55(**×2.0**)——**倒排索引对真实自然语言查询
不但没有改善渐近性质,实测还比线性更差。** 预注册里我押的是"仍然近似线性",实际比那还糟一档。

**⇒ 报告 R9 关于渐近的说法必须收回。** 原文称"成本正比于查询自己的倒排链,而不是语料规模",
并据此说生僻词耗时对规模持平。**机制没错**(df 扫描已证实成本随倒排链覆盖率线性上升),
**但对真实负载的推论错了**:真实问句含高覆盖率的常见词,其倒排链本身就随语料线性增长,再叠加
累加器字典变大后的缓存失效,合起来就是超线性。**幸存的是常数项收益,不是渐近收益。**

**p95 比 mean 恶化得更快**(12.8 → 385.6 ms,**30×**;而 p50 只有 24×)⇒ 长尾随规模变差,
按均值报告会掩盖它。20 万 chunk 时 p95 已近 **0.4 秒/查询**。

**内存(同作业同节点):** `index_bytes_per_chunk` 5245 → 5218 → 5034 → 4991 → **4949**,
跨 16× **基本平坦(−5.6%)**⇒ **按 chunk 外推成立**,这一点与延迟相反。20 万 chunk 索引 990 MB;
外推 100 万 chunk 约 **5.0 GB**。对照仓库自身 Markdown 的 7.8 KB/chunk,**数据集间差约 1.6×** ——
与"数据集变异应假定为重要直到实测"一致,现在有两个真实语料的数了。
登录节点上手测的 5552/5101 B/chunk 与本次作业一致 ⇒ 内存测量确实对节点争用不敏感。

**⚠️ 不可做的比较:** 不得把这里的 `ms/1k_chunks`(0.27–0.55)与 R5 改动前的 16.47–17.86 直接相除
说"快了 30 倍"。R5 跑的是 SciFact、另一个节点、另一份语料。**本条唯一干净的结论是自身的斜率**,
绝对量级的 before/after 需要在同一语料同一节点上重跑,与 R6b 的教训同理。

**本条未回答:** 生僻词查询是否仍持平。本条只测真实负载,这是刻意的 —— 需要知道的是实际
系统会遇到什么,不是最有利情形。合成语料上"生僻词持平"的结果仍然只对生僻词成立。

### R14 AFTER — H14 被证伪,NQ 上峰落在生产值上;且 R9 的机制解释在此不成立 [2026-08-14,job 18487392,COMPLETED 00:12:39]

**自检通过。** `chunk_nq_b-c120o20` 的 `MRR .8153` / `Recall .9098` 与既有
`runs/retr-nq-strong-bm25` **逐位一致** ⇒ 配置与语料无误,结果可读。

**`answer_match`(配对随机化,n=2000,基线一律为生产值 120×5):**

| 臂 | answer_match | Δ vs 120×5 | p | 95% CI |
|---|---|---|---|---|
| 60×10 | 0.8350 | **−0.0465** | 0.0001 | [−0.0595, −0.0335] |
| **120×5(生产值)** | **0.8815** | — | — | — |
| 200×3 | 0.8515 | −0.0300 | 0.0001 | [−0.0375, −0.0225] |
| 300×2 | 0.8315 | −0.0500 | 0.0001 | [−0.0595, −0.0410] |

- **H14 被证伪,命中预注册的证伪 A,且强于预注册所押。** 预注册写的是"NQ 上不显著或反向";
  实测是**三个对照全部显著低于生产值**,形状为**单峰,峰在 120**。R9 在 2Wiki 上同一点
  (60×10)是 **+0.0220 p=0.0008**,NQ 上是 **−0.0465 p=0.0001** —— **同一设计,符号相反,
  两侧都显著。**
- **⇒ 粒度效应是 2Wiki 特有的,不是通用规律。** 按预注册,R9 的结论须从"粒度是主旋钮"收缩为
  **"多跳语料上粒度是主旋钮"**,并**解除改动生产值 `chunk_size=120` 的理由**。
  这不是"证据不足所以不动",是**数据主动支持保留 120**。
  **⚠️ 本行的"多跳/单跳"归因已被 R15 取代,保留原文以存记录。** R15 实测表明分界不是跳数,
  而是 `chunk_size` 与语料原生 passage 长度的相对关系;"解除改动生产值的理由"这一结论不变。
- **本条在一个点上强于 R9:R9 的最优点落在扫描边界(60,最小值),自己标注了"真正最优可能还在
  更小处,未测";NQ 的峰落在扫描区间内部(120,四点中的第二点),不受边界限制。**
  **⚠️ 经 R15 收窄:"不受限制"应为"不受扫描边界限制"。该峰仍受语料预切长度限制 ——
  dpr-w100 是固定 100 词 passage,120 是"恰好不切开"的最小扫描点。换预切长度不同的语料,峰会移动。**

**⚠️ 本条最重要的结果,而且它和 H14 无关 —— R9 的机制解释在 NQ 上不成立:**

| 臂 | Δ`answer_match` | Δ`system.core.final_document_recall` | 比值 |
|---|---|---|---|
| **60×10** | **−0.0465** | **+0.0879** | **−0.53(反号)** |
| 200×3 | −0.0300 | −0.1158 | 0.26 |
| 300×2 | −0.0500 | −0.2088 | 0.24 |

(全部 p=0.0001,即 10000 次随机化的下限。)

R9 的机制结论是"**answer 跟覆盖走**",Δanswer 与 ΔsysRecall **同号同量级**(比值 0.60 / 1.13 / 1.14)。
NQ 上:

- **60×10 两者反号。** 送到 generator 眼前的 gold 文档**显著变多**(+0.0879),答对率却**显著下降**
  (−0.0465),两侧 p 均 =0.0001。**"几乎一比一传导"这句话必须收回。**
- 另外两臂虽同号,比值仅 **0.24–0.26**,远低于 R9 的 1.13–1.14。**覆盖动得多,答案动得少。**

**⇒ 报告中凡引用 R9"chunking 通过最终文档召回近似一比一传导"之处,须加数据集限定。
幸存的是"覆盖是必要条件",不是"覆盖决定答案"。**

**一个可检验的解释(假设,未测,不得写成结论):** `answer_match` 是规范化后的**子串包含**。
固定 600 词预算下切成 60 词的块,覆盖到更多不同文档,但**每块连续文本更短,答案跨块被切断**的
概率上升 ⇒ "文档覆盖到了,答案串却拼不完整"。2Wiki 的答案多为短实体,不易被切;NQ 答案跨度
更长,则会。**直接判据:统计各臂中 `final_document_recall > 0` 但 `answer_match = 0` 的 case
占比,若 60×10 显著更高即坐实。** 这是 R15 的候选。

**⚠️ 本段的两个论断均已被证伪,保留原文以存记录(2026-08-14):**
(a)"NQ 答案跨度更长"被 R15 的前提核对推翻(NQ 中位 2.0 / 均值 2.21,2Wiki 中位 2.0 / 均值 2.35);
(b)"答案跨块被切断"被算术推翻 —— `overlap = chunk_size/6` 使四臂的完整包住阈值为 10/20/33/50 词,
均大于最长答案 5 词,**任何答案都从未被切断**(推导见 R15 AFTER 的更正段)。
**"直接判据"本身是对的**,R15 照此测得 CMR 阶跃;错的只是对该阶跃成因的猜测。

**限制(须与结论同时声明):**

1. 与 R9 同源的固有代价:固定总词数必然同时改变选中条数(10/5/3/2),
   **"小 chunk 更好/更差"与"多条目更好/更差"在本设计下无法分离。**
2. `overlap = chunk_size/6` 全程绑定,overlap 的独立作用未分离(与 R9 同)。
3. 仅在 NQ、仅 strong-bm25、仅 **extractive generator** 上测得。**换真实 generator 后是否同向未测。**
4. `answer_match` 是 exact-string 包含,绝对值为下界;**本条全部判据均为组内相对比较,不受影响。**
   **⚠️ 该“下界”表述已被 R16 作废(2026-08-14):实测为双向偏差 —— 答案为 `yes` 时是下界,为 `no` 时是**上界**(2Wiki 上 52/54 判为答对)。臂间相对比较仍可信,故本条结论不变。**
   但注意:第 2 节那个"答案跨块被切断"的假设**正是针对该指标的伪影本身**,若坐实,
   则受影响的是 R9 与 R14 两条的机制解释,而非它们的相对排序。
5. `c120o20` / `c200o33` / `c300o50` 三臂的 `MRR`/`Recall` **四位小数逐位相同**(.8153/.9098),
   仅 `c60o10` 不同(.7948/.8925)。**此项已由 R15 实测解除,不再是待验证项:**
   `corpus_snapshot.json` 的 chunk 计数为 100,000 / 100,000 / 100,000(与 documents **恰好 1:1**)
   对 200,191(**2.002 倍**)⇒ dpr-w100 的 100 词 passage 在 chunk_size ≥ 120 时从不被切开,
   chunk 与 document 严格一一对应,三臂 top-50 覆盖同一批 gold passage,**逐位相同是必然而非巧合**。

**⚠️ 工具问题,已在本条暴露并须修正台账:** 预注册里给的读数命令用
`evidence_rag.evaluation.paired_metric_cli`,**该 CLI 只接受 `StageEvaluationReport`
(per_case 下需有 `metrics` 键),无法读 pipeline 阶段写出的 `evaluation_report.json`
(`DatasetEvaluationReport`,per_case 按 system/retriever/selector/generator 分组)**。
无论填 `generator_report.json` 还是 `evaluation_report.json` 都跑不通 —— 前者 pipeline
根本不产出,后者 schema 不匹配。本条改用 `scripts/paired_dataset_metric.py`
(直接调 `paired_metric.compare_paired`),**并已用 R9 的已发表结果验证:
60×10 mean 0.5405 / 120×5 mean 0.5185 / delta +0.0220、200×3 delta −0.0485,
与台账 R9 逐位一致**(p 值 0.0006 vs 0.0008 为随机化种子差异)。

- raw:`results/r14-nq-b-{c60o10,c120o20,c200o33,c300o50}-per-case.json` 待 `git add -f` 拉回,
  按 R7/R9 惯例剥离每条 `trace`,完整 trace 留在 bp1。
- **写给 results-summary 的草稿:** 标题应是方法结论 ——
  **"同一 chunking 设计在多跳与单跳语料上给出相反的最优点"**,
  比"NQ 上 120 最好"更重要也更可迁移。
  **⚠️ 该标题已被 R15 取代,勿照此写入 results-summary。** R15 查明分界不是跳数而是
  passage 长度,正确的标题见 R15 AFTER 末尾。

### R15 — R14 的反号是"答案跨块被切断"造成的吗 [PRE-REGISTERED 2026-08-14]

**与既有结论的关系:**(2026-08-14 规则通过时补记,如实记录本条当时的遗漏)
镜像 R9 的 set B 设计,填补 R14 打开的机制缺口。
**⚠️ 本条应当引用 R10 而没有引用 —— 这正是该规则存在的原因。**
R10 于 08-10 固定 `chunk_size=120` 单独扫 overlap,测得 0→60 全不显著,
并在标题里写明"**否掉了边界切断作为 R9 的机制解释**"。而本条把"答案跨块被切断"
当作待检验的解释写了进去,R15 AFTER 更把它写成机制,直到同日的算术更正才移除。
**台账四天前已有答案,本条没去查。**

**⚠️ 关于本条预注册效力的如实声明,写在最前面:本条的假设与判据确实在读数前写定,
但它与自己的 AFTER 是在读数之后同一批提交的 —— commit 时间戳无法证明先后顺序。
按本台账"预注册须先落盘"的规矩,本条的预注册效力弱于 R9–R14,读者应据此打折。
后续条目须在提交预注册之后再提交作业,不得再出现这种顺序。**

**动机:R14 留下的是一个机制缺口,不是一个参数问题。** R14 在 NQ 上测得 60×10 臂
`final_document_recall` **+0.0879**(p=0.0001)而 `answer_match` **−0.0465**(p=0.0001)——
**送到眼前的 gold 文档变多了,答对率反而掉了。** R9 在 2Wiki 上的机制结论("answer 跟覆盖走",
比值 0.60/1.13/1.14)在此不成立。**在解释清楚之前,R9 与 R14 两条的机制叙述都不可用于报告。**

**⚠️ 本条的数据已全部存在(R9 与 R14 的 `evaluation_report.json`),不产生新作业。
正因如此,假设与判据必须在读数前封死 —— 否则本条退化为对已知数字的事后叙事。**

**待检假设 H15(R14 AFTER 中记为"可检验的解释",当时明确标注为假设、未测):**
`answer_match` 是规范化后的**子串包含**。固定 600 词预算下切成更小的块,覆盖到更多不同文档,
但**每块连续文本更短**,答案跨度被块边界切断的概率上升 ⇒ 出现"**文档覆盖到了、答案串却拼不完整**"。
若为真,则该效应应随 chunk 变小而变强,且在答案跨度更长的 NQ 上强于 2Wiki。

**主指标 —— 条件漏答率 `CMR`(conditional miss rate):**

    CMR = P(answer_match == 0 | final_document_recall > 0)

即**在 gold 文档确已进入最终证据的 case 中,答案仍未命中的比例。** 选它而非原始 `answer_match`,
是因为它把"没检索到"与"检索到了但没拼出答案"分开 —— 后者才是本条要测的量。

**设计(不跑新作业,全部为已有 per-case 数据的重聚合):**
- **NQ 四臂:** `runs/chunk-nq-b-{c60o10,c120o20,c200o33,c300o50}`(R14,job 18487392)。
- **2Wiki 对照四点:** `results/r9-2wiki-{b-c60o10,a-c120o20,b-c200o33,b-c300o50}-per-case.json`
  (R9,已拉回本地)。两组的设计、预算、检索器、generator 全部相同,仅数据集不同。
- 配对随机化检验,基线一律为生产值 `c120o20`,n=2000,与 R14 同一协议
  (`scripts/paired_dataset_metric.py`)。

**假设 H15 的三条可证伪推论(全部须成立才算确证):**
1. **NQ 上 `CMR` 随 chunk 变小单调上升**,且 60×10 显著高于 120×5。
2. **NQ 的 `CMR` 差距(60×10 vs 120×5)显著大于 2Wiki 的同一差距** —— 即效应随答案跨度放大。
3. **`CMR` 的上升足以解释 R14 的 `answer_match` 缺口**:在覆盖率已升 +0.0879 的前提下,
   由 `CMR` 反推的 `answer_match` 预测值应落在 R14 实测值 −0.0465 的 95% CI 内。

**证伪路径:**
- **证伪 A —— `CMR` 在 NQ 上不随 chunk 变小上升(或反向):** 切断假设**错误**。
  则 60×10 的答对率下降另有来源,最可能的替补是**选中条数**本身(10 条 vs 5 条:
  更多短片段稀释了 extractive generator 的抽取目标)。**这会把问题从"块边界"移到"条目数",
  而这两者在固定预算设计下不可分离(R9 限制 1、R14 限制 1)——
  即本设计族已到极限,需要新设计(固定条数、只变块长)才能继续。这是最有价值的一种否定。**
- **证伪 B —— NQ 与 2Wiki 的 `CMR` 差距无显著区别:** 切断效应存在但**与数据集无关**,
  则它解释不了 R14 与 R9 的符号相反,**机制缺口仍然敞开**,不得声称已解释。
- **证伪 C —— 推论 1、2 成立但 3 不成立:** 切断是**真实但次要**的成因。
  须报告其解释份额,并明确剩余部分未解释。

**⚠️ 本条不测什么:**
- **不改 `chunk_size` 生产值。** R14 已解除改动理由,本条只解释机制,**不产出参数建议**。
- **不引入真实 generator。** 全部四臂仍为 `extractive`;换 generator 是独立变量,
  混进来会使本条不可解释(与 R9 限制 3、R14 限制 3 同)。
- **不测 SciFact。** 无标准答案文本,`answer_match` 无法评分(R14 已记录,是约束而非选择)。

**辅助记录(不作判据,仅供解释):** 各数据集 gold 答案的**词数分布**(中位数/p90)。
H15 的前提是"NQ 答案跨度长于 2Wiki";**若该前提本身不成立,推论 2 即失去依据,须在读数前就地记录。**

**命令(login node,无需 sbatch):**

```bash
# 1) 前提核对:两个数据集的 gold 答案词数分布
python scripts/answer_span_stats.py \
  runs/chunk-nq-b-c120o20/gold_cases.jsonl \
  <2wiki gold_cases 路径>

# 2) 主指标 CMR,八个点(NQ 四臂 + 2Wiki 四点)
python scripts/conditional_miss_rate.py \
  --reports runs/chunk-nq-b-c{60o10,120o20,200o33,300o50}/evaluation_report.json \
  --reports results/r9-2wiki-{b-c60o10,a-c120o20,b-c200o33,b-c300o50}-per-case.json

# 3) 配对检验,基线 c120o20(每组各三个对照)
for ARM in c60o10 c200o33 c300o50; do
  PYTHONPATH=src python scripts/paired_dataset_metric.py \
    --on-report runs/chunk-nq-b-$ARM/evaluation_report.json \
    --off-report runs/chunk-nq-b-c120o20/evaluation_report.json \
    --metric system.core.answer_match
done
```

**⚠️ 前置:`scripts/conditional_miss_rate.py` 与 `scripts/answer_span_stats.py` 尚不存在,须先写。
`paired_dataset_metric.py` 已在 R14 中用 R9 已发表结果验证过(逐位一致),沿用。**

### R15 AFTER — H15 确证,但它陈述的原因是错的:不是答案跨度,是源 passage 长度 [2026-08-14,login node,无新作业]

**结论先行:R14 留下的机制缺口已完全闭合,且闭合它的规律比 R14 自己的结论更可迁移 ——
`chunk_size` 低于语料原生 passage 长度时,文档覆盖与答案命中会脱钩。**

**前提核对(预注册要求在读主指标前记录):**

| gold 答案词数 | median | mean | p90 | max | >1 词占比 |
|---|---|---|---|---|---|
| NQ | 2.0 | 2.21 | 4 | 5 | 0.690 |
| 2Wiki | 2.0 | **2.35** | 4 | 10 | 0.638 |

**⇒ 前提被证伪。** 预注册押的是"NQ 答案跨度长于 2Wiki",实测 **2Wiki 略长**。按预注册
"若该前提本身不成立,推论 2 即失去依据" —— **推论 2 陈述的理由作废,须替换(见下)。**

**主指标 `CMR` = 在 gold 文档已进入 generator 输入的前提下,答案仍未命中的比例。**

**⚠️ 条件事件的定义,以及为何两种定义在此可互换(已验证,勿凭直觉):**
`system.core.final_document_recall` 是 `recall(cited, relevant)` ——
**generator 引用了的**文档,不是"送到 generator 眼前的"。而 `evaluator.py` 在
`generator_ineligibility` 触发(即无 gold 文档被 selector 选中)时把
`generator.core.conditional_answer_match` 记为未计分,**其非空本身就是"gold 文档进入了
generator 输入"这一事件**。两者是不同的条件,本条按 `--condition {selected,cited}` 两种口径
**各跑一遍,NQ 与 2Wiki 上八个点的 CMR、delta、p、CI 全部逐位相同**。
**原因是 extractive generator 引用的恰好就是它抽取所依据的那些证据,故"被选中"与"被引用"同事件;
换成会挑选性引用的真实 generator,两者将分离,届时必须显式声明用的是哪一个。**(见限制 3)

| NQ 臂 | 覆盖 case 数 | CMR | vs 120×5(配对,仅两臂皆覆盖的 case) |
|---|---|---|---|
| 60×10 | 1782 | **0.0668** | **+0.0596 p=0.0001 CI [+0.0486,+0.0712] n=1727** |
| **120×5(生产值)** | 1758 | **0.0000** | — |
| 200×3 | 1700 | **0.0000** | +0.0000 p=1.0000 n=1700 |
| 300×2 | 1661 | **0.0000** | +0.0000 p=1.0000 n=1661 |

| 2Wiki 点 | 覆盖 case 数 | CMR | vs 120×5 |
|---|---|---|---|
| 60×10 | 1998 | 0.4590 | **−0.0221 p=0.0005 CI [−0.0341,−0.0100] n=1992** |
| 120×5 | 1992 | 0.4799 | — |
| 200×3 | 1975 | 0.5261 | +0.0486 p=0.0001 n=1974 |
| 300×2 | 1952 | 0.5789 | +0.1030 p=0.0001 n=1951 |

- **推论 1 ✓ 成立,但形态与预注册所押不同。** NQ 上 60×10 的 CMR 显著高于生产值
  (+0.0596,p=0.0001)。**但不是"随 chunk 变小单调上升",是阶跃:200/300/120 三点恰好并列 0.0000。**
- **推论 3 ✓ 成立 —— CMR 的上升完整解释了 R14 的缺口,无剩余。**
  c60:覆盖 1782 × (1−0.0668) ≈ **1663** 条答对;c120:1758 × (1−0.0000) = **1758**。
  差 −95/2000 = **−0.0475**,落在 R14 实测 `answer_match` delta **−0.0465** 的
  95% CI [−0.0595,−0.0335] 内。**R14 的机制缺口闭合。**
- **推论 2 的方向成立(NQ +0.0596 vs 2Wiki −0.0221),但其理由被前提核对证伪,必须替换。**

**⚠️ 机制:阈值卡在源 passage 长度上,与答案长度无关。**

`base_loader.py:5` "No slicing — dpr-w100 is pre-chunked";`source_parent.py:3`
"dpr-w100 splits each Wikipedia article into **100-word passages**"。
**NQ 的每篇"文档"就是一段固定 100 词的 passage。**

- `chunk_size` ∈ {120,200,300} **≥ 100** ⇒ passage 永不被切开 ⇒ 覆盖到它即拿到全文
  ⇒ 答案串必定在 ⇒ **CMR 恰好 0.0000**(不是"很小",是恒等于零)。
- `chunk_size` = 60 **< 100** ⇒ 每个 passage 必被切成两块(60 + overlap 10)
  ⇒ 只选中其中一半时,文档级覆盖成立而答案在另一半 ⇒ **CMR 0.0668**。

**⚠️ 更正(2026-08-14,同日,算术,无新数据):上面这两行里"答案串被切断/拼不完整"的措辞是错的,
本条与 R14 AFTER 中所有同义表述一并作废。** 切分从未物理破坏过任何答案:

`corpus.py:62-72` 是滑窗,`step = chunk_size − overlap`,第 k 块覆盖
`[step·k, step·k + chunk_size)`。长度 L 的跨度 `[p, p+L)` 取 `k = ⌊p/step⌋`,则
`step·k ≤ p < step·k + step`,故 `p+L < step·k + step + L`;要被该块完整包住只需
`step + L ≤ chunk_size`,即 **L ≤ overlap**。本设计 `overlap = chunk_size/6`,四臂阈值为
**10 / 20 / 33 / 50 词**;NQ 答案**最长 5 词**(见前提核对表)。
**⇒ 四个臂中任何答案都必定完整存在于至少一个块内,一次都没有被切断过。**

**幸存的机制是另一个:切开 passage 把「答案」与「命中查询词的文本」分到了两半,
而 strong-bm25 按查询词排序,于是选走了不含答案的那一半。** R15 的全部证据
(≥100 词时 CMR 恒为 0、<100 词时 0.0668、chunk 计数 1.000 对 2.002)对该机制同样成立 ——
passage 不被切开时,答案与查询词必然同处一块,无从分离。
**⇒ 本条确证的是"阈值卡在 passage 长度上"这一事实,不是它的成因。成因由 R16 分辨。**

**该机制已由 `corpus_snapshot.json` 的 chunk 计数直接实测坐实,不再是推断:**

| 臂 | chunks | documents | chunks/doc |
|---|---|---|---|
| **60×10** | **200,191** | 100,000 | **2.002** |
| 120×5 | **100,000** | 100,000 | **1.000** |
| 200×3 | **100,000** | 100,000 | **1.000** |
| 300×2 | **100,000** | 100,000 | **1.000** |

**三个 ≥120 的臂是恰好 1:1,不是近似** —— chunker 在这三个设置下一个 passage 都没切开、
也没合并。60 词臂为 2.002 倍:10 万 passage 各切两块,另有 **191 个** passage 切成三块
(步长 50,即长度超过约 110 词的那些)。**"chunk 与 document 一一对应"由此从解释变为事实。**

**⇒ 替换后的规律(比 R14 的"多跳 vs 单跳"更可迁移):
`chunk_size` 不得低于语料的原生 passage 长度;低于它,文档覆盖与答案命中脱钩。**
2Wiki 是变长维基段落,没有这个整齐阈值,故小块只带来覆盖收益 ——
其 CMR 反而**下降** 0.0221(p=0.0005),与 R9 的"answer 跟覆盖走"一致。

**⚠️ 已排除的平凡解释:** 若 `answer_match` 是在 gold 文档原文(而非选中的 chunk 文本)上评分,
则 CMR 恒为 0 将毫无信息量。**c60o10 臂的 0.0668 恰好排除了这一点** ——
它证明评分确实发生在选中的 chunk 文本上。

**⇒ 必须回改 R14 AFTER 的两处:**
1. **"粒度效应是 2Wiki 特有的(多跳/单跳)"这一框架被本条取代。** 真正的分界不是跳数,
   是 `chunk_size` 与源 passage 长度的相对关系。
2. **R14 "峰在生产值 120"须加限定:该峰有一部分是 dpr-w100 预切成 100 词的产物,
   不是粒度本身的普适性质。换一个 passage 长度不同的语料,峰会移动。**
   R14 原文称"NQ 的峰落在区间内部,不受边界限制"——**这句仍然成立,但"不受限制"要收窄为
   "不受扫描边界限制",它仍受语料预切长度限制。**

**限制(须与结论同时声明):**

1. **本条不产生新证据,是对 R9/R14 已有 per-case 数据的重聚合。** 它解释机制,不扩展结论域。
2. **阈值 100 词只在 dpr-w100 上被观测到一次。** "chunk_size 不得低于原生 passage 长度"
   目前是**一个数据集上的一个阈值点**支持的规律,**未在第二个预切语料上复现**。
   最便宜的复现是取任一 passage 长度已知且不同的语料重跑四点。
3. 仍仅 strong-bm25、仍仅 `extractive` generator(与 R9 限制 3、R14 限制 3 同)。
   **真实 generator 有可能从被切开的半块中重建答案,那样 CMR 效应会被削弱甚至消失 —— 未测。**
   **且换 generator 后,本条"被选中 ≡ 被引用"的等价关系会破裂**(见主指标下的说明),
   CMR 的条件事件届时必须显式选定并声明,不能沿用本条的"两者相同故不必区分"。
4. `answer_match` 的 exact-string 伪影在本条不是干扰而是**被测对象本身**;
   本条的判据全部为组内相对比较,不受绝对值偏低影响。
5. **R14 限制 5(三臂 MRR/Recall 逐位相同)已由本条解除,不再是待验证项。**
   `corpus_snapshot.json` 的 chunk 计数实测为 100,000 / 100,000 / 100,000(恰好 1:1)
   与 200,191(2.002 倍),⇒ 120/200/300 三臂的 chunk 与 document 严格一一对应,
   top-50 因此覆盖同一批 gold passage,MRR/Recall 逐位相同是必然而非巧合;
   60 词块使一个 passage 占两个 chunk,top-50 跨越的 passage 数下降(.7948/.8925)。

- 工具:`scripts/conditional_miss_rate.py`、`scripts/answer_span_stats.py`(本条新增),
  `scripts/paired_dataset_metric.py`(R14 新增,已用 R9 已发表结果逐位验证)。
  **三者均已入库并过 ruff;`conditional_miss_rate.py` 在本地用 R9 的 2Wiki per-case 副本
  复现了 bp1 上的同一组数字(含 CI 逐位一致),该复现同时证明本地剥离 trace 的副本与
  bp1 的 `evaluation_report.json` 是同一份数据。**
- **⚠️ 一个必须记下的工具约束:`paired_metric.compare_paired` 会拒绝两臂可计分集合不同的输入
  ("on/off scoring masks differ")。条件指标的可计分集合随 chunker 变动,因此
  `generator.core.conditional_answer_match` 无法直接跨臂配对 —— 这正是
  `conditional_miss_rate.py` 先取两臂条件集交集再配对的原因,不是重复造轮子。**
- **写给 results-summary 的草稿:标题应是
  "chunk_size 低于语料原生 passage 长度时,文档覆盖与答案命中脱钩" ——
  它同时解释了 R9 与 R14 的符号相反,比任一条单独的参数结论都更可迁移。**

### R16 — 含答案的块去哪了:没检索到,还是检索到了没被选中 [PRE-REGISTERED 2026-08-14]

**与既有结论的关系:**
承接 R15 及其同日的机制更正 —— 缺口是那次更正打开的,不是遗留的。
与 R10 同向:R10 用 overlap 扫描、本条用算术,**两条独立路径都否掉了边界切断**,
本条设计时已核对 R10 并确认不冲突。对照 R14 的 `answer_match` 缺口与 R9 的覆盖机制。

**动机:R15 确证了阈值的位置,同日的更正又推翻了它对成因的说法,缺口是被本条重新打开的,不是遗留的。**
R15 证明 `chunk_size` 低于源 passage 长度时 CMR 从 0.0000 跳到 0.0668,并用 chunk 计数
(1.000 对 2.002)把"passage 被切成两半"坐实为事实。但同日的算术更正表明**答案从未被物理切断**
(完整包住阈值 = `overlap` = 10/20/33/50 词,NQ 答案最长 5 词)。
**⇒ 含答案的块一直存在,它只是没有出现在最终的十条证据里。本条问的就是它去哪了。**

**本条提交时机声明:按 R15 记下的规矩,本预注册在作业/读数之前单独提交。**

**评分链的三个事实(读设计前须先认,均来自代码而非推测):**
1. `ExtractiveGenerator.generate` 把**每个选中 chunk 的全文原样拼接**,块间以 `\n- ` 和
   `[evidence_id]` 分隔 ⇒ `answer_match == 1` **等价于**"gold 答案串完整落在**某一个**选中的
   chunk 内",跨块拼接会被分隔符打断。
2. 同函数把 `cited_evidence_ids` 设为**全部** selected ⇒ **`cited` 文档集恒等于 `selected` 文档集**。
   R15 观测到的"两种条件口径逐位相同"由此有了代码依据,不再只是巧合。
3. `[selector] name = "top-k"` ⇒ 选择就是按检索序截断到 `max_selected`(本臂 = 10),
   `top_k = 50`。**"没被选中"因此精确地等于"检索名次 > 10"。**

**总体(population):** `runs/chunk-nq-b-c60o10` 中同时满足 gold 文档已被选中
(`generator.core.conditional_answer_match` 非空)与 `system.core.answer_match == 0` 的 case,
即 R15 测得的那 CMR 群体,**n ≈ 1782 × 0.0668 ≈ 119**。这是 NQ 四臂中唯一 CMR > 0 的臂;
另外三臂 CMR 恰好 0.0000,**没有 case 可分类,这本身就是本条的退化对照**。

**分类(对每个 case,在其 top-50 候选表内扫描"含答案的块" —— 即归一化文本包含归一化 gold 答案的
候选;归一化必须复用 `evidence_rag.evaluation.scoring` 的 `_normalise`,不得另写):**

- **B2a 同源丢失** —— 存在含答案的候选,且它与某个已选中 chunk **共享 `document_id`**。
  即:passage 被切成两半,选中的是含查询词的那半,答案在另一半。
- **B2b 他源丢失** —— 存在含答案的候选,名次落在 11–50,但与任何已选中 chunk **不同源**。
- **B1 检索失败** —— top-50 内**不存在**任何含答案的候选。
- **C 不适用** —— 全语料内不存在含答案的块。该 case 与 chunking 无关,须单列并**排除出 B 的分母**。

**假设 H16:B2a 占主导,预注册阈值为 > 60%。**
理由:`top_k = 50` 对 20 万块的语料相当宽松,gold passage 的两半通常都能进候选表;
真正的损失发生在 50 → 10 的截断处,而截断按查询词打分,系统性地偏向不含答案的那半。

**证伪路径(读数前写定):**

- **证伪 A —— B1 占主导:** 含答案的半块**连候选表都进不去**。
  ⇒ 病灶在检索侧而非选择侧,`top_k = 50` 是绑定约束;修法是提高 `top_k` 或让打分看得到 parent,
  **而不是改 selector**。这会把后续工作整个换一个方向,是最有价值的一种否定。
- **证伪 B —— C 占比 > 5%:** 说明 R15 更正里那段算术有漏洞,或 `answer_match` 的归一化
  与本条假定不符。**此时本条的其余结论一律不得采信,须先重新推导。**
- **证伪 C —— B2b 占主导:** 损失与"被切开的 passage"无关,答案在另一篇文档里且名次 11–50。
  ⇒ 现象不是同源半块问题,R15 的阈值叙事须重新审视。

**次要记录(仅记录,不作判据):** 各 case 中**名次最靠前的含答案候选**的名次分布。
若 B2a 成立,预期其密集堆在 11 附近;若大量散布在 40–50,则即使分类命中,
"只差一点点"的叙事也不成立,提高 `max_selected` 的收益会远小于直觉。

**跨数据集对照(仅记录,不作 H16 判据):** 对 `runs/chunk-2wiki-b-c60o10` 跑同一分类
(该臂 CMR = 0.4590,n ≈ 917)。2Wiki 是多跳,一个 case 需要两篇 gold 文档,
**失败可以有与切分无关的原因**,故不能用来判 H16;记录它是为了看该机制是否跨数据集成立。

**⚠️ 本条不测什么:**
- **不测 overlap。** 提高 overlap 是这个病最直觉的解药,但把解药和诊断混在一起,
  就无法分辨"症状缓解"与"病因查清"。**overlap 是独立的一条,不进本条。**
- **不换语料。** "chunk_size 不得低于原生 passage 长度"在第二个预切语料上的复现留给 R17。
  **R16 填的是刚被打开的因果空洞,R17 拓宽的是已有硬证据的事实 —— 故 R16 在前。**
- **不产出生产建议。** R14 已解除改动 `chunk_size=120` 的理由,本条只解释,不建议。

**成本:零 GPU,零新作业。** 所需字段全在 `pipeline_runs.jsonl` 的候选记录内
(`text` / `document_id` / `retrieval_rank`),配 `gold_cases.jsonl` 的 `reference_answers` 即可。
按行流式读取,不整份载入。

**命令(login node):**

```bash
PYTHONPATH=src python scripts/answer_chunk_forensics.py \
  --run runs/chunk-nq-b-c60o10 \
  --report runs/chunk-nq-b-c60o10/evaluation_report.json
# 跨数据集对照:
PYTHONPATH=src python scripts/answer_chunk_forensics.py \
  --run runs/chunk-2wiki-b-c60o10 \
  --report runs/chunk-2wiki-b-c60o10/evaluation_report.json
```

**⚠️ 前置:`scripts/answer_chunk_forensics.py` 尚不存在,须先写并提交。
它必须 import `evidence_rag.evaluation.scoring` 的归一化函数,不得自行实现 —— 本条的全部判据
都建立在"与 `answer_match` 使用同一归一化"之上,重写一份就等于换了指标。**

### R16 AFTER — H16 被证伪:答案半块不是输在选择,是根本没进候选池 [2026-08-14,login node,无新作业]

**分类结果(`runs/chunk-nq-b-c60o10`,conditional misses n=119,`top_k=50` / `max_selected=10`):**

| 类别 | n | 占比 | H16 预注册 |
|---|---|---|---|
| **sibling** 同源丢失 | 25 | **21.0%** | 押 **> 60%** ❌ |
| **other** 他源 | 11 | 9.2% | — |
| **retrieval** 未进候选池 | **83** | **69.7%** | ← **命中证伪 A** |
| **absent** 全语料无 | **0** | **0.0%** | ← **证伪 B 干净排除** |

- **H16 被证伪。** 押的是"损失发生在 50 → 10 的截断处",实测**近七成的损失发生得更早 ——
  含答案的块连 top-50 都没进**。

- **⚠️ 证伪 B 排除得很干净,这一条要单独说:`absent = 0/119`。**
  **R15 更正里那段"答案从未被物理切断"的算术,不再只是推导,而是 119 个 case 无一例外的实测。**
  答案始终完整存在于某个 gold 文档的块内。**"答案跨块被切断"这一说法至此彻底作废,
  R9 / R14 / R15 中所有同义表述一并失效。**

- **⚠️ 本条最需要说清楚的一点:被推翻的是位置,机制本身反而被强化了。**
  R15 更正提出的机制是"切开 passage 把答案与命中查询词的文本分到两半,而 strong-bm25
  按查询词排序,于是取走了不含答案的那半"。本条不但没有推翻它,还表明它**比预注册押的更极端**:
  在 20 万块的语料里,**同一段 100 词 passage 的两半,一半进了前 10,另一半掉出了前 50**。
  **⇒ 机制成立,H16 错在假定分离只够把答案挤出前 10;实际它把答案挤出了整个候选池。**

- **⇒ 提高 `max_selected` 基本无用,这一点有数字支撑,不是判断。**
  进了池子的仅 36 个(sibling + other),名次 **min 11 / median 23 / max 47**,
  **落在截断线后 5 名内的只有 11 个(30.6%)**。即便把 `max_selected` 从 10 提到 20,
  可捞回的也远不足以解释那 119 个损失。**"只差一点点"的直觉不成立。**

**⇒ 修法的方向由本条改变:不在 selector,也不在 `chunk_size`,而在"检索单元与 passage 对齐"。**
最直接的候选是**按 parent passage 打分或召回**,使被切开的两半绑定同进同出。
**本仓库已有该映射:`materializer/source_parent.py`** —— 它原本为 dpr-w100"一篇文章跨多个
passage"而写(防止 `independent_support` 把同源 passage 当成独立票数),
**同一套确定性映射正好可用于把被切开的两半重新绑回。** 这是 R17 之后的工程候选,非本条结论。

**⚠️ 本条无法回答、且它直接决定下一步可行性的问题:那 83 个的答案半块究竟排在第几名。**
候选表在 `top_k=50` 处截断,现有数据不含更深的名次。**若它们排在 60 名附近,把 `top_k` 提到 100
就能解决;若排在数千名,这条路是死的。** 二者对工程的含义完全相反,**不得在测得之前选边**。
这需要一个新作业(CPU,重跑检索并放大 `top_k`),定为 **R17 首选**。

**限制(须与结论同时声明):**

1. **仅 `c60o10` 一个臂。** NQ 另外三臂 CMR 恰好 0.0000,**没有 case 可分类** ——
   那是退化对照,不是独立复现。本条的全部占比都来自单一设置下的 119 个 case。
2. **⚠️ 仅 strong-bm25,而生产配置是 Hybrid RRF(strong-bm25 + Granite Dense)。**
   本条的机制**整个建立在"检索按查询词打分"之上**;稠密臂不需要查询词逐字出现,
   被切开的答案半块在语义空间里未必掉名次。**⇒ R9 / R14 / R15 / R16 这一整条链
   是否适用于生产的混合检索,完全未测,且有具体理由怀疑它不适用。**
   这是本链条目前最大的外部效度缺口,优先级应高于换语料复现(原 R17)。
3. **⚠️ selector 侧是占位实现,而本条的判据直接建立在它之上 —— 这是本条最容易被误读的一点。**
   `composition.build_selector` **只注册了 `top-k` 一个**,其余一律 `raise ValueError`;
   仓库里 `selector/` 下的 `dual_head` / `nli_dual_head` / `risk_controlled` / `guidance`
   **无法从 config 选到**,92 个实验 config 全部用 `top-k`。
   本条据此把"没被选中"定义为"检索名次 > `max_selected`",**该等价仅在纯截断下成立**。
   换成带门控的真实 selector,一个 case 可能因被门挡下而落选,与名次无关,
   **四类划分届时必须重新定义,不可沿用本条的占比。**
   **对照:generator 侧已在 G9 做过这件事** —— `verify-annotate`(项目的主方法)已注册进同一工厂,
   其注释明确记着"在 G9 之前,config 驱动的 CLI 跑不了本项目要做的方法"。
   **selector 侧尚未走完这一步,故其占位程度比 generator 侧更深:generator 至少选得到主方法。**
4. 仍仅 `extractive` generator。真实 generator 可能从半块中重建答案,亦可能不能;未测。
   且换 generator 后"被选中 ≡ 被引用"的等价会破裂(依据见 R15 AFTER 主指标段)。
5. **`sibling` / `other` 的划分依赖 `document_id` 相等。** 在 c60o10 下同一 passage 的两块
   共享 `document_id`,这一点由 R15 的 chunk 计数(2.002 chunks/doc)保证,**该前提已实测**。
6. **跨数据集对照(`runs/chunk-2wiki-b-c60o10`,CMR 0.4590,n≈917)已于同日运行,
   结果与由此查出的一个指标伪影见下一条"R16 AFTER 续"。** 预注册中它是
   "仅记录、不作 H16 判据",故不改变上述任何结论。

- 工具:`scripts/answer_chunk_forensics.py`(本条新增,commit 32dc6bb)。
  归一化 import 自 `evaluation.scoring`,与 `answer_match` 同源;
  **使用前已在一份四类各一的样例上验证分类正确**(`reference-baseline` 只有 2 个 case、
  0 个 conditional miss,跑不到分类逻辑)。
- **写给 results-summary 的草稿:标题应是
  "把 passage 切小,丢的不是答案文本,是答案的可检索性" ——
  近七成的损失发生在候选池之前,而非选择环节;因此修法在检索单元,不在 chunk 大小或选中条数。**

### R16 AFTER 续 — 跨数据集对照,以及它意外查出的一个指标伪影 [2026-08-14,login node,无新作业]

**本段取代 R16 AFTER 限制 6 中"尚未运行"的记载。**

**分类结果(`runs/chunk-2wiki-b-c60o10`,conditional misses n=917):**

| 类别 | NQ (n=119) | 2Wiki (n=917) |
|---|---|---|
| sibling | 21.0% | **4.0%** |
| other | 9.2% | 22.5% |
| **retrieval** | **69.7%** | **59.9%** |
| absent | **0.0%** | **13.6%** |

- **主结论跨数据集成立:两边 `retrieval` 都占大头 ⇒ "损失发生在候选池之前,而非选择环节"
  不是 NQ 特有的。** 名次分布也一致(2Wiki:n=243,min 11 / median 23 / max 50,
  截断线后 5 名内仅 26.3%;NQ 为 30.6%)。**提高 `max_selected` 在两个数据集上都不解决问题。**
- **`sibling` 从 21.0% 降到 4.0%,符合预期且是对机制的正面佐证。** 2Wiki 是变长维基段落,
  大量段落本就短于 60 词、根本没被切开,故"同源半块"这一子机制**确为预切语料特有**。

**⚠️ `absent` 13.6% 的成因已查明,并且它查出的东西比 `absent` 本身重要得多。**

125 个 `absent` case **全部**是是非题(125/125),抽样全为 `yes`。顺此查下去:

| gold 答案 | n | `answer_match` 均值 | ==1 | ==0 |
|---|---|---|---|---|
| `yes` | 155 | **0.0323** | 5 | 150 |
| `no` | 54 | **0.9630** | **52** | 2 |
| 其他 | 1791 | 0.5717 | — | — |

**⇒ 在这 209 个 case 上,`answer_match` 不携带任何关于正确性的信息。**
这不是推测,是由生成器的定义确定的:**`ExtractiveGenerator` 只拼接选中 chunk 的原文,
从不输出是非判断**。因此 `no` 的 52 个"答对"全部来自 "no" 作为普通词出现在证据文本里
("no longer"、"no. 5" 之类);`yes` 的 150 个"答错"则因为系统根本不可能产出 "yes"。
**分数完全由答案词是否为常见英语词决定,与检索、切分、选择全都无关。**

**拆解自检:** `(5 + 52 + 1791 × 0.5717) / 2000 = 0.5405`,与台账 R9 发表的
`b-c60o10` 的 `answer_match` **逐位相同** ⇒ 上表的拆分正确。

**⇒ 两处必须更正:**

1. **"`answer_match` 绝对值是下界"这一表述是错的,R7 / R9 / R14 / R15 中所有该表述一并更正。**
   实测是**双向偏差**:`yes` 上是下界(假阴性 150/155),`no` 上是**上界**(假阳性 **52/54**)。
   正确的表述是:**在 2Wiki 上,`answer_match` 对答案为常见英语词的 case 系统性高估,
   对不可能被抽取的 case 系统性低估;二者不能合并成一个方向。**
2. **R9 / R14 的 2Wiki delta 被稀释,但符号与显著性不受影响。** 209 个无信息 case 进了分母
   且跨臂近似恒定(其得分由答案词决定,与臂无关),⇒ **真实效应约比记录值大 11.7%**
   (2000 / 1791)。R9 的 `60×10 +0.0220` 在 1791 个有信息 case 上约为 **+0.0246**。
   **⚠️ 不得据此改写 R9 / R14 的已发表数字** —— 那是在全体 2000 个 case 上如实测得的;
   **应做的是在引用其绝对值时声明该稀释,并在后续 2Wiki 实验中预先排除是非题。**

**NQ 不受此伪影影响:** 其答案均为实体片段,无是非题,`absent = 0/119`。
**R14 与 R15 在 NQ 上的全部结论不动。**

**⚠️ 本段未做的事:** 未重算 R9 / R14 在 1791 个子集上的配对检验。上面的 11.7% 是由
均值分解推出的**估计**,不是实测的 delta。**若要引用"有信息子集上的效应量",必须实跑,
不得引用该推算值。**

- 工具:`scripts/answer_chunk_forensics.py --dump`(commit 41c689a)。
  该伪影正是靠 dump 把分类与答案 join 起来才浮现的;
  **仅看四类占比会把 13.6% 当成一个待解释的疑点,而看不到它背后的指标问题。**
- **写给 results-summary 的草稿:两条,分开写 ——**
  **(a)"把 passage 切小,丢的不是答案文本,是答案的可检索性"(主结论,跨两个数据集成立);**
  **(b)"2Wiki 上有一成的题,`answer_match` 只在测答案词是不是常见英语词" ——
  这是一条关于指标而非关于检索的发现,应写进指标一节,不是检索一节。**

### R17 — R9→R16 这条链跑的全是 strong-bm25,它对生产的混合检索还成立吗 [PRE-REGISTERED 2026-08-14]

**与既有结论的关系:**
本条检验的正是 R9 / R11 / R12 / R14 / R15 / R16 这整条链的外部效度 ——
它们全部只在 strong-bm25 上测得。R16 的机制("按查询词打分")是本条怀疑的直接来源。
R7 已证实检索改善经排序通道到达下游,故本条若否定,受影响的是整条链而不止一条。
与 R10 不冲突:R10 只扫 overlap、不换检索器,本条不重复它。

**赌注:这一条若否定,前面五条对生产系统的适用性一起失效。**
R9 / R11 / R12 / R14 / R15 / R16 全部只在 `strong-bm25` 上测得,而**生产选定的是 Hybrid RRF
(strong-bm25 + granite-dense)**。R16 查明的机制**整个建立在"检索按查询词打分"之上**:
切开 passage 把答案与命中查询词的文本分到两半,BM25 取走含查询词的那半,含答案的那半
**连 top-50 都进不去(69.7%)**。**稠密检索不需要查询词逐字出现** —— 一个 60 词的半块,
只要话题仍然贴近问题,嵌入相似度未必崩。**⇒ 有具体的、机制层面的理由怀疑这条链换到生产配置就不成立,
这不是例行的"再测一个设置"。**

**主指标:`CMR = P(answer_match == 0 | gold 文档已被选中)`,以及 R16 的四类划分。**
次要记录 `answer_match` 的 60×10 vs 120×5 配对 delta(用于对照 R14 的 −0.0465)。

**设计:完全镜像 R14 的 NQ set B,只换检索器。**
- **核心对(必须先读):** `granite-dense` × {60×10, 120×5}
- **扩展对(同批提交,后读):** `hybrid`(`fusion=rrf`, `k=60`, `[strong-bm25, granite-dense]`)× {60×10, 120×5}
- 四臂 `chunk_size × max_selected ≡ 600 词`(60→10 / 120→5),`overlap = chunk_size/6`,
  `top_k = 50`,`generator = extractive`,除 `[retriever]` / `[chunker]` / `max_selected` 外逐字相同。
- **只取两个 chunk 点,不重跑 200/300。** R14 已确立 ≥120 的三点在 bm25 下并列 CMR 0.0000,
  且 chunk 计数证明它们与 document 严格 1:1;**本条要分辨的是"切开 vs 不切开",两点足够,
  多跑两点只是多花 GPU。**

**假设 H17:稠密臂的 CMR 差距显著小于 bm25 的 `+0.0596`,预注册阈值为 `< +0.02`。**
理由:嵌入编码整段文本的语义,60 词半块仍与问题同主题,故不会像 BM25 那样因缺少查询词而跌出候选池。

**证伪路径(读数前写定):**

- **⚠️ 证伪 A —— 稠密臂的 CMR 差距与 bm25 相当(`≥ +0.04`):**
  **损失与"按查询词打分"无关**,切碎 passage 会以任何打分方式损害可检索性。
  **这会把 R16 从"BM25 的性质"升级为"检索的普遍性质",并使其成为生产系统的实际风险,
  而不是一个只在基线上出现的现象。这是最有价值、也是后果最重的一种否定。**
- **证伪 B —— 稠密臂差距更大:** 切碎对稠密检索伤害**更重**(半块上下文更薄,嵌入更不稳)。
  与 H17 方向相反,且会**反转工程建议** —— 那样"对齐 passage"对稠密比对稀疏更要紧。
- **⚠️ 证伪 D(结构性,优先于以上三条检查)—— 120×5 臂的 CMR 不为 0.0000:**
  在 `chunk_size=120` 下 passage 从不被切开(R15 chunk 计数 100,000 = documents,严格 1:1),
  故"gold 文档被选中"就等于"该 passage 全文进入证据",而 dpr-w100 的 gold passage
  **按构造包含答案**。**⇒ 任何检索器在该点的 CMR 都必须是 0.0000,这是结构决定的,与检索器无关。
  若不为 0,说明"gold passage 按构造含答案"这一前提不成立,R15 / R16 的读法须整体重审,
  本条其余结论一律不得采信。** 这一条先看。

**必做的自检(照抄 R14 的纪律,两个锚点均已存在):**
`configs/experiments/retr_nq_granite-dense.toml` 与 `retr_nq_hybrid-rrf.toml` 同样跑在
`runs/niah-base` 上,**无 `[chunker]` 段即默认 120/20,与 c120o20 臂逐字相同**,
`top_k=50` / `max_selected=5` 亦一致。**⇒ 新建的两个 120×5 臂,其 retriever 指标必须与既有
`runs/retr-nq-granite-dense` / `runs/retr-nq-hybrid-rrf` 逐位一致。**

**⚠️ 新臂的 MRR / Recall 从哪读 —— 写死在这里,因为同一个坑已经栽过两次:
`run_pipeline_eval.slurm` 只跑 `prepare → pipeline`,**不产生 `retriever_report.json`**
(那是 `retriever` 阶段的产物,R14 首次提交 job 18445256 即因此报废)。
新臂的检索指标在 `evaluation_report.json` 的 `aggregate` 里,键为 `retriever.core.*`,
亦见作业日志末尾 `summarize_pipeline_eval.py` 打印的 MRR / Recall 两列。
**锚点侧的 `runs/retr-nq-*` 则确有 `retriever_report.json`,因为它们由 `retriever` 阶段产出 ——
两侧读法不同,不可互抄。**
**锚点值(2026-08-14 于 bp1 读出,记入本条,读数时逐位核对;不一致即停,先查清再读结果):**

| 既有运行 | `document_mrr` | `document_recall` |
|---|---|---|
| `runs/retr-nq-strong-bm25`(R14 已用) | .8153 | .9098 |
| `runs/retr-nq-granite-dense` | **.9323** | **.9202** |
| `runs/retr-nq-hybrid-rrf` | **.8873** | **.9709** |

**⚠️ 顺带记下一个影响读法的事实,读数前必须先认:三种检索器在同一 chunking(120/20)下的
检索强度本就不同**(稠密 MRR 高出 bm25 **11.7 个百分点**,混合的 Recall 最高)。
**⇒ 各臂的"gold 文档被选中"这一条件事件的总体大小必然不同,故 CMR 的绝对值跨检索器不可比。
本条唯一可比的量是每个检索器内部的 60×10 vs 120×5 差距**,H17 的阈值(`< +0.02` 对
bm25 的 `+0.0596`)比较的正是这两个组内差距,而非两个 CMR 绝对值。**记数时须同时报告各臂的
条件总体 n,否则读者无从判断差距是否建立在可比的基数上。**

**⚠️ 本条不测什么:**
- **不测 200×3 / 300×2**(理由见设计)。
- **不换语料。** 阈值在第二个预切语料上的复现仍然待做,但**本条问的是"对生产是否成立",
  优先级高于"规律有多普遍"**。
- **不测那 83 个的真实名次。** 放大 `top_k` 是另一条(见 R16 AFTER),
  且**若本条命中证伪 A,那条的设计要跟着改** —— 故本条在前。
- **不产出生产建议。** 本条只回答适用性,任何"改检索单元"的提案须另立条目并附迁移成本。

**⚠️ 提交注意(前两条都在这里栽过,写下来免得再犯):**
`scripts/run_pipeline_eval.slurm` 内部 `export LLM_DEVICE=cuda` 且申请 `gpu:rtx_3090:1`。
**R14 因主指标只需 CPU,曾用 `--partition=compute --gres=none` 覆盖到 CPU 分区;本条不可照抄 ——
稠密臂要编码约 30 万 chunk(c60o10 的 200,191 + c120o20 的 100,000),混合臂再各编一次,
必须走 GPU 分区,按默认提交即可。**
另:`prepare` 对已存在的输出目录做严格校验(含 `source_tree_signature`),
**本条四个输出目录均为新名,不与既有冲突,无需归档。**

**命令:**

```bash
# 0) 先读锚点,记入本条,再提交作业:
python -c "import json;d=json.load(open('runs/retr-nq-granite-dense/retriever_report.json'));print(d['aggregate'])"
python -c "import json;d=json.load(open('runs/retr-nq-hybrid-rrf/retriever_report.json'));print(d['aggregate'])"

# 1) 核心对 + 扩展对,同批提交(GPU 分区,不加 --gres=none):
mkdir -p logs runs && sbatch scripts/run_pipeline_eval.slurm \
  configs/experiments/chunk_nq_dense-c60o10.toml \
  configs/experiments/chunk_nq_dense-c120o20.toml \
  configs/experiments/chunk_nq_hybrid-c60o10.toml \
  configs/experiments/chunk_nq_hybrid-c120o20.toml

# 2) 读数(login node):
PYTHONPATH=src python scripts/conditional_miss_rate.py \
  runs/chunk-nq-dense-c{60o10,120o20}/evaluation_report.json \
  --baseline runs/chunk-nq-dense-c120o20/evaluation_report.json
PYTHONPATH=src python scripts/answer_chunk_forensics.py \
  --run runs/chunk-nq-dense-c60o10 \
  --report runs/chunk-nq-dense-c60o10/evaluation_report.json
PYTHONPATH=src python scripts/paired_dataset_metric.py \
  --on-report runs/chunk-nq-dense-c60o10/evaluation_report.json \
  --off-report runs/chunk-nq-dense-c120o20/evaluation_report.json \
  --metric system.core.answer_match
```

**⚠️ 前置:四个 config 尚不存在,须先写并提交。** 它们必须逐字镜像
`chunk_nq_b-c60o10.toml` / `chunk_nq_b-c120o20.toml`,仅改 `[retriever]` 与 `[output]`;
**任何其他差异都会使本条与 R14 不可比,而与 R14 可比正是本条的全部意义。**

**成本:GPU,四臂。** 稠密编码约 30 万 chunk × 2(稠密臂与混合臂各一次)。
**若 GPU 预算被迫削减,保留核心对(granite-dense),它承载机制问题;
只跑混合对会把机制留在未决状态**,因为混合是两者的混合,分不清是哪一侧在起作用。
