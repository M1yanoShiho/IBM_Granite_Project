# Experiment 05 — 分目标跟踪表

**当前状态：** `GOAL 1–5 PASS / FINAL PASS`  
**最近更新：** 2026-08-24

## 当前唯一阶段

`无 active execution goal — Goal 5 已 FINAL PASS，实验按计划停止`

Goal 1–5 全部通过技术审计。三个数据集、十个实验臂共 `12,000` 条 generation outputs
与 `12,000` 条正式评分均完整，`scorer_errors=0`；Table 1/2、附录诊断、10,000 次配对
bootstrap、Claim A/B 标签与最终 hash 审计均已生成。科学结论为 Claim A 和 Claim B 均
`NOT SUPPORTED`；这是预注册判定规则下的负结果，不影响实验流程的 `FINAL PASS`。

### Goal 2 PASS 证据

- development 两阶段检索：jobs `18684758/18684759/18684760`，三个数据集各一张
  A100 并行；先前 22 秒即退出的 import-only jobs `18684677/18684678/18684680`
  已保留为可修复依赖故障记录，不含实验结果。
- development rerank/Provence/threshold-only Selector：成功 jobs
  `18685932/18684915/18685933`；早期 import-only failures `18684914/18684916`
  不含实验结果并已保留日志。
- 150-output 共同 smoke：成功 jobs
  `18685953/18685935/18685948/18685949/18685936/18685937`；三套数据、十臂各 5 条，
  总计 `150/150`，runtime errors `0`。
- Goal 2 总审计 job `18685954`：`COMPLETED / 0:0 / 00:01:14`，结果 `PASS`；
  审计 SHA-256 `6f493ad99d1572782013e03dbb6d676850ff1ff2420f660843ad7a97febbdfde`。
- 自动契约测试：`58 PASS`；目标文件：
  `artifacts/experiment05_goal2/test_reports/goal2_contracts.xml`。
- Selector 双门槛在 `360/360` 条 development query 上逐条一致；删除 `0`、fail-open `0`、
  all-delete `0`。这是诚实的 development 诊断，不据此调阈值。
- Goal 2 冻结时 formal generation/scoring 均为零。

### Goal 3 运行中证据

- formal 两阶段检索：jobs `18685969/18685970/18685971`（NQ/TQA/ASQA，各 400 题）。
- formal rerank/Provence/threshold-only Selector 准备：dependent jobs
  `18685972/18685973/18685974`。
- 七个主实验臂共 21 个并行 generation jobs：`18685975–18685995`；它们只在对应
  dataset 准备成功后启动。
- Goal 3 冻结审计 job：`18685996`，只在上述 21 个 generation jobs 全部成功后启动。
- 2026-08-23 恢复记录：NQ/ASQA formal retrieval、preparation 和七个主臂均已完成，
  即 Goal 3 已产生 `5,600/8,400` 个完整输出。TriviaQA retrieval job `18685970`
  在 `281/400` 时因单批证据长度峰值触发 CUDA OOM；checkpoint 保留。恢复 job
  `18687320` 使用等价的 `dense_batch_size=128` 从 281 条继续，且已把原 preparation job
  `18685973` 的依赖重新接到恢复 job。检索算法、模型、候选集与正式样本均未改变。
- 因学校恢复 job 预计等待至 2026-08-23 晚间，已启用教师服务器
  `fl25387@10.70.71.11` 的空闲 A4000 作为更快恢复路径：隔离目录
  `/scratch/fl25387/experiment05_recovery`，冻结 index/runtime 先做 hash 校验，随后以
  `dense_batch_size=64` 从同一 281-row checkpoint 续跑。学校 job 在教师服务器产生第
  282 条前保留为保险，确认实际进展后才取消；400/400 PASS 后将结果按 hash 传回学校并
  重接原 Goal 3 依赖链。
- 2026-08-23 14:05 更新：学校恢复 job `18687320` 提前获得 RTX 3090，并已从
  `281` 推进到 `282/400`，因此当前优先继续学校 checkpoint。教师服务器索引传输在
  `15G / 213 files` 处安全暂停并保留为可续传热备，避免两个后端重复计算及并发读取学校
  存储；若学校 job 停滞或再次失败，heartbeat 立即从该 transfer checkpoint 切换恢复。
- 2026-08-23 15:28 更新：学校恢复 job `18687320` 已完成 TriviaQA `400/400`，
  Slurm `COMPLETED (0:0)`、stderr 为空、retrieval manifest `status=PASS`。输出 SHA-256
  为 `f409dfc0b2867f29c41b299fe31d41e8104dab25fe25287356044cb14498f891`，runtime
  SHA-256 仍为 `03e12beccb382e40339105834eebf8ef933bbcf611c0b5c9aac291cc9ad53026`；
  manifest 明确记录原 281 条使用 batch 512、续跑使用 batch 128。原 preparation job
  `18685973` 已自动解除依赖并进入调度队列，主运行 jobs `18685982–18685988` 与后续
  freeze/Goal 4/Goal 5 门控保持不变。教师服务器不启动重复计算，热备保留到 Goal 3 PASS。
- 2026-08-23 15:46 更新：`18685973` 解除依赖后因 A100 priority 预计到午夜才启动；
  在不改变任何科学参数的前提下，已把该未启动 job 缩短为 1 小时并放宽为通用 GPU，
  同时启用教师服务器的两阶段准备热备。`screen exp05_tqa_recovery` 正在从学校迁移冻结的
  Granite reranker、Provence、Selector snapshots、selector checkpoint、已 PASS retrieval
  与当前源码；迁移完成后用同一 `experiment05_goal2_prepare.py` 和同一模型/阈值在 A4000
  生成 400 条 prepared artifact。学校 job 若先启动则停止热备；教师服务器若先完成则按
  retrieval/output hash 与 schema 校验传回学校，通过轻量 import audit gate 后才重接
  `18685982–18685988`，不得直接绕过准备审计。
- 2026-08-23 15:48 更新：资源放宽已生效，学校 job `18685973` 在 `bp1-gpu037`
  V100 上启动，并在约 2 分钟内推进到 `70/400`。教师服务器 screen、rsync 和重复准备已
  立即停止，仅保留约 1.8 GB partial cache/checkpoint 作为热备；当前继续以学校原 job
  作为唯一 preparation 运行，预计约 10–15 分钟完成后自动释放七个 TriviaQA 主臂。
- 2026-08-23 15:52 更新：`18685973` 已在 `00:05:08` 内 `COMPLETED (0:0)`，正式
  prepared artifact 为 `400/400`，SHA-256
  `196e4df36d7db3e0fe91c1a6d62bc92446f6620d4242eff40364421a22924630`；七个 TriviaQA
  主臂 `18685982–18685988` 已解除依赖。为减少 A100 backfill 排队且保持所有正式 generation
  继续使用相同 A100 硬件，根据已完成 NQ/ASQA jobs 的实测主机内存峰值约 12.2 GB、
  直接臂最长 11 分钟、grounded 臂最长 60 分钟，把七个未启动 job 的纯调度预留从
  `128G/16h` 调整为 `32G`，直接臂时限 `45m`、grounded 臂时限 `2h`；模型、dtype、
  输入、解码、输出与 checkpoint 逻辑均未改变。
- 2026-08-23 16:01 更新：调度预留优化已让第一个 TriviaQA 主臂 `18685982`
  (`bm25_rag`) 在 A100 上启动；约 4 分钟已推进到 `200/400`，无 stderr。其余六臂已获得
  顺序 backfill 预计时间；由于实际任务通常远早于安全时限结束，后续 job 会随前一 job
  提前完成而自动提前启动，无需按 Slurm 显示的最坏时限等待到深夜。
- 2026-08-23 16:12 更新：`18685982/18685983` 已分别以 `400/400 PASS` 完成；
  `18685984` 已推进到约 `262/400`，`18685985` 同时获得第二张 A100 并启动。调度已从
  单臂顺序运行转为两臂并行，stderr 仍为空；后续 grounded 三臂预计也会随空闲 A100
  提前于 Slurm 的保守 start-time 启动。
- 2026-08-23 16:17 更新：四个 direct 主臂 `18685982–18685985` 已全部 `400/400 PASS`；
  三个 grounded 种子臂 `18685986–18685988` 随三张 A100 同时释放而并行启动。当前只剩这
  1,200 条即可触发 Goal 3 freeze，且启动日志无报错；按 NQ/ASQA 历史耗时预计约 25–60
  分钟完成，显著早于先前串行保守估计。
- 2026-08-23 16:50 更新：grounded jobs `18685986/18685987` 已分别在 `32:21/31:47`
  完成 `400/400 PASS`；仅剩 `18685988`，当前 `207/400` 且持续增长、无 stderr。按其当前
  实测速度，约 30 分钟后应完成并自动触发 Goal 3 freeze `18685996`。
- 2026-08-23 17:28 更新：`18685988` 已 `400/400 PASS`，Goal 3 共 `21` 个输出文件、
  `8,400/8,400` 条。冻结审计 `18685996` 在 test backfill 分区用 `00:01:47` 完成并
  `PASS`：所有臂各 400 条、runtime errors `0`、formal scores `0`、scorer sidecar 仍为
  `0600_unread`，且 `outputs_frozen_before_scoring=true`。Goal 3 正式 PASS，Goal 4 自动成为
  唯一 active goal；九个消融 jobs `18686007–18686015` 的纯调度预留已按 Goal 3 实测值
  优化为 `32G`，direct 臂 `45m`、grounded 臂 `2h`，模型与科学参数不变。
- 2026-08-23 17:34 更新：连续两次检查时学校 A100 与 RTX 3090 均被长任务占满，九个
  Goal 4 jobs 无预计启动时间，因此启用教师服务器 `/scratch/fl25387/experiment05_goal4_recovery`
  的双 A4000 热备。当前只迁移冻结的 Granite 3B、TRUE、三 adapter、三套 runtime/prepared
  与源码，不启动重复计算；若学校任一对应 job 先启动即停止相关热备传输/运行。首个 transfer
  命令因变量展开错误把教师服务器根目录内容读入两个新建的隔离 cache 目录，约 `9.7G×2`；
  已立即停止所有相关 rsync，明确删除并重建仅这两个 task-owned cache 目录，正式仓库、学校
  数据和实验输出均未被修改。随后改用完全显式源路径重启，当前路径与进程已核验正确。
- 2026-08-23 17:37 更新：学校给出的九个最坏调度时间已变为 23:36 至次日 13:07
  串行启动，因此教师服务器热备成为当前预计最快路径。正确传输已真实增长到 Granite
  `1.1/6.4G`、TRUE `5.1/43G`；三 adapters `179M`、prepared `89M`、三 runtime 各
  `400` 和源码均已完整到位。两张 A4000 保持空闲，尚未生成任何 Goal 4 正式输出；
  Granite 完整且校验通过后可先并行运行三个 direct 消融，同时继续传输 TRUE。
- 2026-08-23 18:10 更新：Granite 两个分片与关键配置已完整传到教师服务器，并逐文件与
  学校冻结快照完成 SHA256 对照，六个核验文件全部一致。学校 direct jobs
  `18686009/18686012/18686015` 仍为 PENDING 且零输出，因此已按完全相同的脚本、模型、
  BF16、输入和解码参数在两张 A4000 启动 direct 消融：GPU0 依次运行 `kilt-nq`、
  `alce-asqa`，GPU1 运行 `kilt-tqa`。18:10 首检已产出 NQ `9/400`、TriviaQA `18/400`，
  两 GPU 各使用约 `7.2G/16G`、无错误；TRUE 仍在后台继续传输，尚未启动 grounded 消融。
- 2026-08-23 18:33 更新：教师服务器三个 direct 消融均已 `400/400 PASS` 并生成 manifest。
  TRUE 五个权重分片与五个关键配置也已完成双服务器 SHA256 对照，十个文件全部一致。
  学校 grounded job `18686007` 仍 PENDING 且零输出，因此已在 GPU1 启动同配置的
  `kilt-nq / ablation_bm25_retriever` 作为首个 grounded 实测；Granite 与 TRUE 均成功加载，
  TRUE 依冻结脚本 `device_map=auto` 使用 GPU+CPU offload，首条已写入且未 OOM。待达到首
  `10` 条后用真实吞吐决定双 GPU 队列；不得为提速改变 BF16、模型或其他科学参数。
- 2026-08-23 18:42 更新：首臂达到 `29/400`，第二张 A4000 上的
  `kilt-nq / ablation_no_selector` 也已达到 `17/400`；两臂的 generations 与 internal traces
  条数一致，GPU/RAM 正常、无 OOM。排除加载后实测约 `15–25` 秒/条，双 GPU 路径预计仍早于
  学校从 23:36 起的串行排队，因此确认继续双队列并行；每个当前臂 PASS 后再接下一数据集。
- 2026-08-23 18:52 更新：学校 A100 job `18686007` 意外提前启动并已快速产生约 `49` 条
  `kilt-nq / ablation_bm25_retriever`。按去重与最快路径规则，已立即停止教师服务器同一臂，
  保留其 generations/internal traces 各 `72` 条作为 superseded checkpoint，不得导入正式结果。
  释放的 A4000 已改为运行学校仍 PENDING 且零输出的
  `kilt-tqa / ablation_bm25_retriever`（对应 `18686010`），模型加载成功、无 OOM；另一张 A4000
  继续 `kilt-nq / ablation_no_selector`。学校 `18686007` 成为 NQ BM25 消融的唯一正式来源。
- 2026-08-23 19:03 更新：教师服务器三个 direct 臂已回传学校隔离 staging；整体 tree hash、
  每臂 `400` 条、schema、ordered IDs hash、正式 runtime ID 顺序、sealed evidence hash 与
  `runtime_errors=0` 全部通过。CPU import gate `18689443` 用 `00:00:54` 完成 `PASS` 并将
  `1,200` 条安全导入学校 runroot。Goal 4 freeze `18686016` 已先改接到六个 grounded jobs 与
  新 gate，再取消未启动的旧 direct jobs `18686009/18686012/18686015`；依赖链保持有效且
  不会重复生成 direct 结果。
- 2026-08-23 19:14 更新：学校 job `18686008` 也提前在 A100 启动 NQ no-Selector，因此按
  去重规则停止教师同臂，保留其 generation/trace 各 `183` 条为 superseded checkpoint，
  不得导入；释放的 A4000 已切到 `kilt-tqa / ablation_no_selector`（学校 `18686011` 仍
  PENDING 且零输出），模型加载成功。学校 `18686007` 随后在 `00:26:46` 完成
  `kilt-nq / ablation_bm25_retriever` 的 `400/400 PASS`，generation/trace 各 400、无错误；
  学校 `18686008` 成为 NQ no-Selector 的唯一正式来源。
- 2026-08-23 19:20 更新：学校 A100 job `18686010` 提前启动并产生 `kilt-tqa /
  ablation_bm25_retriever` 的真实输出；教师同臂随即停止，generation/trace 各 `88` 条标为
  superseded checkpoint，不得导入。释放的 A4000 已切到学校仍 PENDING 且零输出的
  `alce-asqa / ablation_bm25_retriever`（`18686013`），Granite+TRUE 加载成功、无 OOM。
  当前三条唯一有效路径为学校 NQ no-Selector、学校 TQA BM25、教师 TQA no-Selector；教师
  ASQA BM25 正在完成加载并将成为第四条并行路径。
- 2026-08-23 21:15 更新：Goal 4 的六个 grounded 消融 jobs `18686007/08/10/11/13/14`
  均已在学校完成 `400/400 PASS`；三个 direct 消融已由 import gate `18689443` 审计导入。
  freeze job `18686016` 随后以 `00:00:52 / 0:0` 完成，报告
  `{"phase":"goal4","status":"PASS","outputs":3600}`。Goal 4 正式 PASS，十臂共
  `12,000` 条冻结输出满足 sidecar 解锁条件，Goal 5 自动成为唯一 active goal。
- 2026-08-23 21:23 更新：首个评分 job `18686017` 在第 2 条遇到 Granite 将答案内人物别名
  双引号原样复制进 JSON、但未转义的运行时语法错误；已保留 1 条有序 checkpoint。修复仅在
  严格 JSON 解析失败时转义不可能作为字符串终止符的内部引号，不改变任何生成文字、claim
  内容、模型或指标；评分器测试 `15 PASS`。恢复 job `18689816` 已从同一 checkpoint 提交，
  final compile `18686047` 依赖已安全替换。其余 29 个评分 job 的调度预留由
  `128G/16h` 缩为 `32G/4h`，仍固定 A100 且科学参数不变；`18686018` 已启动并持续评分。
- 2026-08-23 21:34 更新：后续 `18686018/18686019` 分别保留 `48/10` 条有序评分
  checkpoint 后触发另外两种 scorer JSON 基础设施错误：Granite 对单引号使用 JSON 不支持的
  `\'` 转义，以及 256-token 输出上限截断长答案的 claim JSON。前者已按 JSON 语义移除多余
  反斜杠；后者已确认 384 仍截断，而 1024 在同一模型、prompt、greedy decoding 下自然结束并
  完整解析 8 条 claims，因此 formal scorer 上限改为 1024，仅防止 JSON 被截断，不改变已生成
  文本或评分定义。测试仍为 `16 PASS`。学校原 jobs 均保留 checkpoint；教师服务器已完成约
  210 MB 冻结评分输入迁移，正在传输 3.0 GB MiniCheck 快照，完成 hash 核验后用双 A4000
  恢复三个失败臂，并通过 score import gate 安全替换 final compile 依赖。
- 2026-08-23 21:46 更新：MiniCheck 3.13 GB 快照已完整传到教师服务器，15 个文件的
  SHA-256 与学校逐一一致。Granite 还会偶发把 claims 数组末尾的 `]}` 写成 `}}` 或漏写
  closing delimiters，因此增加了字符串外部的确定性括号配平；不重写任何字符串内容，相关
  scorer tests 增至 `18 PASS`。教师双 A4000 已从 checkpoint 恢复 NQ `bm25_rag` 与
  `hybrid_rag`，21:46 分别为 `45/400`、`90/400`，scores/traces 同步且无错误。学校新脚本
  job `18686021` 同时达到 `328/400`、无错误。四个 held import gates
  `18689905–18689908` 已预先接入 final `18686047`，替换旧 failed/pending jobs
  `18686017–20/18689816`；学校重复恢复 job `18689816` 已取消，依赖链当前有效。
- 2026-08-23 21:48 更新：学校新代码评分 job `18686021` 已以 `00:10:55 / 0:0`
  完成 NQ `ours_seed13` 的 `400/400 PASS`，aggregate 已生成且全程无错误。教师双臂继续
  增长至 BM25 `85/400`、Hybrid `130/400`，无 JSON/OOM 错误。根据学校真实单臂耗时，
  未启动 jobs 的纯调度时限进一步收紧为 NQ `1h`、TQA `1.5h`、ASQA `2h`，仍保持
  `32G`、固定 A100 与全部科学参数不变；当前 A100 节点 12 张卡中 11 张已占用、最后一张
  处于 planned reservation，因此教师双 A4000 仍是必要的并行恢复路径。
- 2026-08-23 22:02 更新：Hybrid 在第 224 条发现 Granite 偶发漏写已生成
  `source_text` 的 closing quote；新增修复只尝试在已知 JSON key 前补齐缺失引号，并且仅在
  解析后所有 `source_text` 仍逐字出现在原答案中时接受，不改写字符串，scorer tests 增至
  `20 PASS`。恢复后 Hybrid 已越过故障点并完成 `400/400 PASS`。教师 BM25 也完成
  `400/400 PASS`；两臂的 scores/traces/aggregate 已逐文件 SHA-256 一致回传学校，import
  gates `18689905/18689906` 均审计 `400` 条、ordered IDs、schema、六指标、零 scorer error
  与 aggregate/hash 后 `PASS`。两张 A4000 随即分别从保留的 `40/10` 条 checkpoint 接跑
  Provence/Granite Rerank，22:02 为 `135/400`、`12/400`，持续增长且无 OOM。学校
  `18686022` 已提前在 A100 启动 NQ `ours_seed42`，约 `164/400`，作为该臂唯一正式来源。
- 2026-08-23 22:17 更新：学校 `18686022/23` 已先后完成 NQ `ours_seed42/73` 的
  `400/400 PASS`（各约 11 分钟），下一个 `18686024` 等待 A100。教师 Provence 已
  `400/400 PASS`，逐文件 hash 一致回传后
  import gate `18689908` 审计 PASS。Granite Rerank 在 133 条 checkpoint 后发现固定 claim
  字段间漏写 JSON 逗号；通过同一输入确定性复现并先加入失败回归样例，最小修复只在
  `source_text/text` 固定 key 前补结构逗号，不改变字符串，完整 scorer tests 为
  `21 PASS`。同步两端后 Granite 已从第 134 条续跑并越过故障点至 `171/400`。为继续三路
  并行，学校仍 PENDING/零输出的 `18686046`（ASQA direct 消融评分）已由 held import gate
  `18690028` 安全替换 final 依赖后取消；教师 GPU0 已启动完全相同评分，首检 `3/400`，GPU1
  继续 Granite Rerank，两路均正常占用且 checkpoints 有序。
- 2026-08-23 23:02 更新：NQ 其余学校 jobs `18686024–26` 全部 `400/400 PASS`，至此 NQ
  十臂评分完整。教师 Granite Rerank 也已 `400/400 PASS` 并经 gate `18689907` 导入审计
  PASS。ASQA direct 在 54 条后出现“未转义引号 + 一条 `source_text` 少句点而无法逐字定位”
  的组合；既有合同本就规定 non-locatable claim 跳过，修复仅让无歧义语法修复继续进入该
  既有过滤路径，不改任何字符串，红绿回归后完整 tests `22 PASS`。该臂从 54 条恢复并完成
  `400/400`，gate `18690028` import PASS。学校首个 TQA job `18686027` 在 `390/400`
  触发同一旧代码路径，390 条 hash 一致迁到教师后补完，gate `18690167` import PASS；
  `18686028` Hybrid 已学校 `400/400 PASS`，`18686029` Granite Rerank 运行至 `307/400`。
  final 依赖已移除 failed/cancelled jobs 并恢复为可满足状态。教师双 GPU 现分别运行 ASQA
  `ablation_bm25_retriever`（gate `18690176`）和 `ablation_no_selector`（gate `18690142`），
  首检 `5/400`、`50/400`，对应学校 jobs `18686044/45` 均在零输出时取消，无重复来源。
- 2026-08-23 23:13 更新：教师 ASQA `ablation_bm25_retriever` 与
  `ablation_no_selector` 均已 `400/400 PASS`，逐文件 hash 回传后 gates
  `18690176/18690142` import audit PASS。学校 TQA 当前保持双 A100：Provence 约
  `342/400`、`ours_seed13` 约 `138/400`。为避免教师 GPU 空闲，学校仍 PENDING/零输出的
  ASQA `ours_seed42/73`（旧 jobs `18686042/43`）已分别由 held gates
  `18690212/18690213` 安全替换 final 依赖后取消；教师双 A4000 已同时启动，首检均
  `5/400`，scores/traces 同步、无重复来源。final 依赖仅包含可满足的 school jobs
  `18686030–41` 与上述两个 held gates。
- 2026-08-23 23:15 更新：学校 TQA Provence `18686030` 已 `400/400 PASS`；TQA
  `ours_seed13` job `18686031` 在 `179/400` 遇到新的 claim JSON 语法形态并保留有序
  checkpoint 后 FAILED。已先创建 held import gate `18690218`，把 final 依赖中的 failed
  job 安全替换，final 恢复为全部可满足状态；179 条 scores/traces 已逐文件 hash 一致迁到
  教师 recovery，待任一 A4000 当前 ASQA 臂完成后精确复现第 180 条并续跑。教师 ASQA
  `ours_seed42/73` 当前约 `54/55`，两路持续增长。学校下一 TQA job `18686032` 仍 PENDING，
  不受 final 依赖修复影响。
- 2026-08-23 23:20 更新：学校 TQA `ours_seed42` job `18686032` 在 `8/400` 也遇到
  claim JSON 语法异常并 FAILED；8 条有序 checkpoint 已 hash 一致迁到教师 recovery。
  held gate `18690231` 已替换 final 中的 failed job，final 再次恢复全部依赖可满足；下一臂
  `18686033`（TQA `ours_seed73`）已在学校 A100 运行。教师 ASQA `ours_seed42/73` 当前约
  `270/274`，持续增长；它们完成后两张 A4000 将优先分别复现并恢复 TQA `ours_seed13`
  的 `179/400` 与 `ours_seed42` 的 `8/400` checkpoints。
- 2026-08-23 23:36 更新：教师 ASQA `ours_seed42/73` 已各 `400/400`，回传文件逐一
  SHA256 一致，import gates `18690212/18690213` 均 `PASS`。学校 TQA `ours_seed73`
  `18686033` 与 `ablation_no_selector` `18686035` 均在 `179/400` 保留有序 checkpoint
  后 FAILED，`ablation_bm25_retriever` `18686034` 在首条失败；已分别创建 held gates
  `18690247/18690253/18690248` 并替换 final 中的 failed 依赖，三份可用 checkpoint 均
  hash 一致迁到教师 recovery。故障点由冻结 Granite 精确复现为：答案内成对双引号导致
  JSON string 边界误判、固定 `text` key 漏引号，以及仅返回可逐字定位 `source_text` 而
  漏 `text`。按 TDD 新增三项精确回归测试并先验证 RED；最小修复只在字符串外补固定 key、
  只把普通 prose comma 前的引号视为字符串内容，并仅在 source 为答案逐字子串时复用该
  source 作为漏失的 claim text。没有改写任何字符串或科学参数；教师与学校 scorer 全文件
  测试均 `25 PASS`。实际故障行复测通过后，教师两张 A4000 已分别从 `179/400` 与 `8/400`
  恢复 TQA `ours_seed13/42` 并越过原故障点；学校 `18686036`（TQA direct）继续作为唯一
  正式来源运行。
- 2026-08-23 23:40 更新：教师 TQA `ours_seed13/42` 已分别从 `179→225`、`8→56`，
  scores/traces 同步且双 A4000 正常。学校 TQA direct `18686036` 已到 `312/400`；学校
  ASQA BM25 `18686037` 在 `25/400` 保留有序 checkpoint 后遇到另一 JSON 语法异常并
  FAILED，后续 Hybrid `18686038` 已自动接棒。已创建 held gate `18690262` 替换 final
  中的 failed job，25 条 checkpoint 逐文件 SHA256 一致迁至教师 recovery；待当前 TQA
  恢复臂释放 GPU 后精确复现第 26 条并按同一 TDD/不改字符串约束处理。
- 2026-08-23 23:42 更新：学校 TQA `ablation_direct_generator` job `18686036` 已
  `400/400 PASS`（`00:13:05`，scores/traces/aggregate 完整）；ASQA Hybrid
  `18686038` 继续运行并已越过 `80/400`。教师 TQA `ours_seed13/42` 已分别推进至约
  `349/173`，scores/traces 同步、GPU 正常；final 当前只保留未完成 school jobs 与 held
  import gates，不含任何 failed dependency。
- 2026-08-23 23:45 更新：教师 TQA `ours_seed13` 已 `400/400 PASS`，三文件回传
  staging 后 SHA256 逐一一致，gate `18690218` import audit `PASS`；GPU0 已立即从
  `179/400` checkpoint 启动 `ours_seed73`，首检到 `190/400` 并越过原故障点。
  `ours_seed42` 同时推进至约 `369/400`。学校 ASQA Hybrid/Granite Rerank 两臂继续
  并行运行；final 已自动移除 fulfilled gate，只剩可满足的未完成依赖。
- 2026-08-24 00:28 最终更新：NQ、TQA、ASQA 各十臂均完成 `400/400` 正式评分，合计
  `30` 臂、`12,000/12,000` 条 query scores，`scorer_errors=0`。最终汇总 job
  `18686047` 为 `COMPLETED / 0:0 / 00:00:16`，`final_audit.json` 返回
  `status=FINAL PASS`。独立机械复核确认 30 个 generation 文件与 30 个评分目录均各
  `400` 条，generation/scores/traces 的 query ID 顺序一致；三个数据集的六项指标均完成
  `10,000` 次 bootstrap（seed `13`、全部 `ESTIMABLE`、invalid UCR replicates `0`）；
  Table 1 为 `15` 行、Table 2 为 `12` 行，九个发布工件的 SHA-256 全部与 final audit
  一致。预注册三值规则给出 Claim A=`NOT SUPPORTED`、Claim B=`NOT SUPPORTED`。
  最终工件已保存至 [`results/final/`](results/final/)。Goal 5 通过后不再创建额外 Goal。
- 运行期间 heartbeat `experiment-05` 每 5 分钟重连两台服务器并检查进度停滞、
  GPU/process、日志、磁盘、网络、Slurm 依赖和 freeze/final audit；它只自动修复不改变
  冻结科学协议的运行时故障。Goal 5 `FINAL PASS` 后该监控已停止并删除。
- 为消除阶段间排队空窗，后续 jobs 已按 `afterok` 预登记，但不会在前一 Goal PASS 前运行：
  Goal 4 三消融 jobs `18686007–18686015`、冻结审计 `18686016`；Goal 5 三数据集十臂
  评分 jobs `18686017–18686046`、最终汇总审计 `18686047`。
- 这些 dependency-gated jobs 当前不构成 active Goal；任一前置冻结审计失败都会阻断全部
  下游运行。formal scorer-only sidecar 保持 `0600`，最早只会在 Goal 4 freeze PASS 后由
  Goal 5 jobs 读取。

## 执行目标状态

| Goal | 独立目标 | 核心交付 | 当前状态 |
|---:|---|---|---|
| 0 | 计划审查与交接 | 修改写入、独立复审、冻结交接 | **PASS / HANDOFF READY** |
| 1 | 数据、语料、隔离与 scorer readiness | 3×400 manifest、index/provenance、隔离与评分验证 | **PASS** |
| 2 | 新 Selector、五系统接线与 dev smoke | threshold-only 实现、10 configs、统一 smoke | **PASS** |
| 3 | 五系统正式主运行 | 8,400 frozen outputs、presented-text artifacts、hash audit；不评分 | **PASS** |
| 4 | 三模块消融正式运行 | 3,600 frozen outputs；凑齐十臂后仍不评分 | **PASS** |
| 5 | 统一评分、统计、审计与最终报告 | 解锁 sidecar、Table 1/2、CSV/JSON、Claim A/B 结论 | **PASS / FINAL PASS** |

## 用户明确启动后的接力规则

1. 用户在交接后的新对话中明确要求开始执行后，Goal 1 才成为唯一 active goal。
2. 当前 Goal `PASS`：冻结交付物，立即把下一 Goal 设为唯一 active goal 并继续，不等待普通中间确认。
3. 当前 Goal `FAIL`：留在本 Goal 内修复并复测，不带问题进入下一 Goal。
4. 若失败要求更换数据集、指标或科学范围，停止并交回用户决定；不得自动扩大任务。
5. Goal 5 `FINAL PASS/FAIL` 后停止，不创建额外 Goal。

## v4 快速执行修订

用户于 2026-08-22 在 formal 输出与评分均为零时明确授权切换快速方案。全量 dense jobs
`18684111/18684112` 已取消并保留日志/partial artifacts；这些 partial artifacts 不进入正式系统。

当前检索冻结为：完整语料 BM25 Top-1000 -> Granite dense candidate scoring -> RRF/rerank。
正式样本、十个运行臂、12,000 outputs、六主指标和 Goal 接力均不缩减。

## 冻结设计摘要

### 三个数据集

- KILT–Natural Questions：400
- KILT–TriviaQA：400
- ALCE–ASQA：400

### 五个系统

1. BM25 RAG
2. Hybrid RAG
3. Granite Rerank RAG
4. Provence RAG
5. Ours

### 三个模块级消融

1. w/ BM25 Retriever
2. w/o Selector / Keep-all Top10
3. w/ Direct Generator

### 主指标

`RFC / VRFC / UCR / CP / CR / RR`

### 附录诊断信息（不参与主张或 PASS）

`ER@10 / SELR / CRR`

- ER@10：Retriever 找到多少支持事实的证据；
- SELR：Selector 误删多少已找到的支持证据；
- CRR：只描述上下文删除量，不定义越高或越低更好。

Table 1 与 Table 2 均只使用相同的六个主指标。Selector 的系统贡献由 Full 与 Keep-all 在六个主指标上的配对差异决定。

### Gold 解锁与实际呈现证据

- Goal 3 只冻结七个主实验臂，Goal 4 只冻结三个消融臂；三个数据集的十臂、12,000 条正式输出全部完成前，scorer-only sidecar 始终锁定。
- Goal 5 才统一解锁并评分，因此 Goal 3/4 不产生正式分数或 Table 1/2。
- 每条 presented evidence 必须冻结 prompt ordinal、canonical ID、实际裁剪后文本的只读 artifact URI、文本 hash 和 token 数；评分不得按 ID 回读原始完整 passage。
- Provence 保留 passage ID 但裁掉支持句时，SELR 必须记为支持证据丢失。

### Claim 标签唯一规则

Claim A/B 分别判定：至少两个数据集通过 superiority 且全部 harm gates 通过为 `SUPPORTED`；恰好一个通过且全部 harm gates 通过为 `MIXED`；零个通过或任一 harm gate 失败/不可估计为 `NOT SUPPORTED`。

### Selector

```text
delete iff harm_score >= 0.9212157130241394
       AND protect_score <= 0.0787842869758606
```

无 `max_delete`，无 `minimum_retained`；全删时 abstain；异常时 query-level keep-all。

## 正式运行量（已完成）

| 阶段 | New outputs |
|---|---:|
| Goal 3 | 8,400 |
| Goal 4 | 3,600 |
| **总计** | **12,000** |

## 当前审核入口

- 完整方案：[PLAN.md](PLAN.md)
- 自一致性审查：[PLAN_SELF_REVIEW.md](PLAN_SELF_REVIEW.md)
