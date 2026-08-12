# R005A/R005B 修订实验跟踪表

**对应计划：** [`EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v2.md`](EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v2.md)  
**状态：** `DRAFT / WAITING FOR USER APPROVAL`  
**生产默认：** TopK10  
**禁止：** 未批准不得实现/训练；未冻结不得读取 screen/confirm；失败后不得重用 confirm 调整方法

## 1. 队列

| ID | 目的 | 数据 | 决定性门 | 当前状态 | 下一步许可 |
|---|---|---|---|---|---|
| D005-1 | 当前 R005 runner/finalizer 复验 attestation | 正式 R005 bundle | 两个 verifier exit 0、hash 不变 | COMPLETE | 只支持 R005 FAIL 可复验 |
| D005-2 | train-fit-only 失败诊断 | 原 R005 640 分数 | 无 modelval；阈值不可行性/配对 margin/实现事实 | COMPLETE | 只允许写 amendment |
| D005-3 | 独立样本、统计、代码设计审计 | R002/R004/R005 train-fit pins + 代码 | 历史 v1 草案终审；后来发现四项遗漏 | SUPERSEDED AUDIT SNAPSHOT | 不再支持批准 v1 草案 |
| D005-4 | v2 协议纠错与独立复审 | terminal 顺序、ordered schemas、B authorization、旧回归边界 | 新稳定 blobs 上 P0=0、P1=0、P2=0 | COMPLETE | 只允许请求批准 |
| A000 | 用户批准 amendment v2 | v2 计划全文 | 用户明确同意实现并运行 R005A/R005B | WAITING APPROVAL | 未批准不得写 scorer 代码或训练 |
| A001 | 先实现并冻结预结果代码 | 仅原 R005 train-fit + 合成 fixture | 第 11 节测试、旧 R005 回归、clean commit | NOT RUN | PASS 后先提交 GitHub，再允许 A002 |
| A002 | sealed 样本复算与 fit 物化 | fresh component-disjoint train-fit | fit 可物化；held-out 只输出聚合 hash/PASS，不输出内容 | NOT RUN | PASS 后可训练 fit；held-out 仍密封 |
| R005A-V0-S13 | amendment control fit | A-fit 64+64 | execution + 两 endpoint 各≥61/64 | NOT RUN | 与 V1/V2 一起冻结 |
| R005A-V1-S13 | repaired NLI path fit | A-fit 64+64 | 同上 | NOT RUN | 与 V0/V2 一起冻结 |
| R005A-V2-S13 | V1 + pair objective fit | A-fit 64+64 | 同上 | NOT RUN | 与 V0/V1 一起冻结 |
| R005A-SCREEN | 选择唯一最简 variant | A-screen 96+96 | 首个 V0→V1→V2；两 endpoint 各≥92/96 且 CP lower≥.90 | NOT RUN | PASS 才可进入 R005B |
| R005B-{V*}-S42 | 独立确认训练 1 | B-fit 64+64 | execution + 两 endpoint 各≥61/64 | NOT RUN | 与 S73 一起冻结 |
| R005B-{V*}-S73 | 独立确认训练 2 | B-fit 64+64 | 同上 | NOT RUN | 与 S42 一起冻结 |
| R005B-CONFIRM | 一次性两种子确认 | B-confirm 128+128 | seed×dataset 四格各≥122/128 且 CP lower≥.90 | NOT RUN | 四格全过才可新建 R006A |
| R006A | amended full train seed13 | 全量 frozen train-fit | 另按后续协议 | CUT / LOCKED | 仅 R005B PASS 后状态更新 |

原 R005 保持 FAIL；原 R006–R015 保持 CUT。上表任何 `NOT RUN` 都不是效果失败。

## 2. 批准前检查

- [x] 原 R005 没有被改判 PASS。
- [x] train-modelval/dev/sealed/heldout 效果没有用于诊断或样本选择。
- [x] 只调阈值无法满足原门已被数值证明。
- [x] snapshot 的 pooler/classifier 被原实现丢弃是已核实事实；因果仍待实验。
- [x] 2Wiki 资格改为 TopK10 至少一条 official support，避免错误的“全 active chunk Top10”筛选偏差。
- [x] 主统计单位改为每 component 一个代表 query 的复合成功。
- [x] 三种 variant、两种子确认与 one-shot reveal 规则已写死。
- [x] exact CP 离散通过数已独立复算。
- [x] v1 草案未获批、未实现、未运行，已保留并从执行入口移除。
- [x] v2 已修正 FIT terminal 顺序、五个 ordered manifest 的 literal paths/exact schemas、B-fit authorization SHA 链和旧 R005 回归禁读边界。
- [x] v2 新稳定 blobs 的独立状态机与跨文档复审均为 P0=0、P1=0、P2=0。
- [ ] 用户明确批准。

## 3. A001 实现冻结 Gate

- [ ] 未物化或读取任何新 A-screen/B-confirm 的 query ID、文本、标签、token 或 model-dependent cache。
- [ ] 只用原 R005 train-fit 与合成 fixture 实现 V0/V1/V2、sealed materializer、pair batching、threshold/gate、durable reveal 与 closed-world verifier。
- [ ] V1 完整加载 encoder/pooler/classifier，无 missing/unexpected keys。
- [ ] exact one-vs-rest 初始化等于原 NLI entailment/contradiction softmax。
- [ ] 两个 3-logit head storage 不共享；shared dropout 只调用一次。
- [ ] V2 loss、λ、pair mean 与梯度方向测试通过。
- [ ] V0/V1 `pair_loss_weight=0`；V2=`0.5`。
- [ ] variant-aware checkpoint strict load；V0↔V1 cross-load 拒绝。
- [ ] threshold fit、tie-break、query composite 与 CP 边界测试通过。
- [ ] screen/confirm 顺序、two-seed one-shot、closed-world row/cell/prefix completeness 测试通过。
- [ ] canonical namespace/session/split-anchor/veto、full members + config/checkpoint/fit-gate/threshold 五个 literal-path manifest 的 exact envelope/fields/projection、bundle schema 和 verifier file SHA 在 clean commit 冻结且禁止 CLI override。
- [ ] durable 状态机故障注入通过：逐级 directory `fsync`、canonical OWNER_LOCK 全临界区互斥、STARTED owner、no-replace publish、COMPLETE/BURNED、invalid COMPLETE 永久 veto、orphan/并发/每个崩溃窗口均 STOP；拿不到锁者零 access/零 mutation 退出。
- [ ] semantic verifier 从 checkpoint+sealed input 重算 float32 raw score bit patterns，再从 raw scores/fit artifacts逐层重算 threshold、qualification、query composite、cell/CP、prefix与 gate；三个派生文件 canonical bytes 必须一致，不能只验 summary。
- [ ] 旧 `dual_head.py`、`selector_sanity.py`、`run_selector_sanity.py`、`finalize_selector_r005.py` Git-blob 不变；A001 只跑 synthetic/fixture 旧 finalizer零写回归与全仓测试，历史真实 R005 复验引用已有 attestation，不在 A001 再读取含 train-modelval 的输入；Ruff、mypy PASS。
- [ ] clean implementation commit 已提交并同步 GitHub；服务器 worktree clean 且同 commit。

## 4. A002 sealed 样本 Gate

- [ ] 所有固定输入 SHA 与 amendment 相同。
- [ ] 排除原 R005 任一 query 所在的整个 component。
- [ ] NIAH 合格池 `778 query / 719 component`。
- [ ] 2Wiki 合格池 `2677 query / 2078 component`。
- [ ] 四角色每数据源为 `64 / 96 / 64 / 128`。
- [ ] 四角色 query/component、复合 candidate identity、text-pair、active-supervised content overlap 均为 0。
- [ ] 不把共享语料 raw document/text overlap 误写为 0。
- [ ] Combined assignment hashes：
  - A-fit `342d90e4cf324c96a541cee6fce962be770b57ca52ab4c1b93f330c3c8af7491`
  - A-screen `a992f1b76abfac727981a79cdaf20d034aa7f8641d8e5d4a7713c9ecfa5bd15c`
  - B-fit `d571a85eb91c3a3a9128124c0c627f6de27d604b5eb4a49510edf0a6fb573111`
  - B-confirm `676cdb6cfb51626b98959a53ea7d81774b042cc64be0e4e6a09ba3046016832a`
- [ ] 四角色 assignment 总 SHA `8b5551e8265ed67c76fd43cd8cf8886892d5a22e143af86411dc98ed64a19fb5`。
- [ ] sealed materializer 正在 A001 clean commit 上运行；实现者未在 A001 后改代码。
- [ ] A-fit/B-fit canonical files 已正式物化并双重 verify；A-screen/B-confirm 只在隔离子进程中流式复算，外部输出仅 aggregate row/hash/isolation PASS。
- [ ] held-out query ID、文本、标签、token 与 raw rows 均未在 reveal 前写盘或输出；任何人工/agent/其他命令访问都定义为 invalidation/STOP。
- [ ] 生成与独立 verify-only 都 PASS；fit manifests 与 held-out aggregate pins 在任何模型训练前冻结。

### Formal fit 全局注册与预算

- [ ] write-once `FORMAL_FIT_REGISTRY.json` 在首个 formal fit 前冻结且 `fsync`；恰含计划第 9.0.1 节的 5 个 literal job/run/audit/claim/lock 路径、seeds、A variants、B selected-V* 与固定 `R005A_TO_R005B.json` 路径约束、A001 commit/A002 fit hashes、150 epochs 与 2.0 GPU-hour 上限。
- [ ] A job 只能在 exclusive job lock 下依次安装 anchor → ordinal claim → `FIT_STARTED`；B job 按 A-screen owner lock→B job lock 的固定顺序重验 A valid COMPLETE/无 veto/authorization，再安装直接绑定相同 authorization path/SHA、A attempt与V*的 anchor→claim→STARTED。STARTED 都在 forward/optimizer init 前；拿不到任一锁者零训练/零变更退出。
- [ ] claim/STARTED/terminal/alternate root/duplicate job/seed 或 registry 外 checkpoint 任一冲突都整体 STOP；STARTED 无 valid terminal 只写 FAIL 并停止，不 resume。
- [ ] FIT COMPLETE 只能在 final checkpoint frozen→strict reload/fingerprint→fit scores→threshold table/tie trace/thresholds→qualification/full-fit loss/technical gate→artifact closure 全部冻结并复验后安装；terminal 之后只准零变更 verify-only。COMPLETE/FAIL 对同一 literal `FIT_TERMINAL.json` 用 canonical temp→close/read-only/file-fsync→atomic no-replace hard link→parent-fsync 安装；terminal 不完整、无 matching STARTED、EEXIST/hash 冲突均永久整体 STOP，FAIL/invalid 不能覆盖或复活为 COMPLETE。
- [ ] 任一 claim/STARTED 后没有新 A001 commit、A002 重跑或预算重注册；ordered reveal member 绑定 registry/anchor/claim/STARTED/COMPLETE-terminal SHA，而非只选 checkpoint。
- [ ] closed-world fit verifier 证明 claim 总数≤5、每 job≤1、epochs≤150、GPU-hour≤2.0、没有可供挑选的第二 checkpoint/run root，并证明两个 B job 使用同一 valid A authorization SHA/A attempt/V*。

## 5. R005A Gate

在任何非 sealed-preflight 的 A-screen held-out access 前：

- [ ] V0/V1/V2 三个 checkpoint 均从 base 独立完成并冻结。
- [ ] 每个正式 fit 的 write-once/fsynced `FIT_STARTED` 都先于模型 forward/optimizer 初始化；此后任一异常均整体 STOP，正式 fit 不重跑。只有 marker 前且零 step/checkpoint/fit-score/gate 的结构错误可回到新 A001 commit/A002。
- [ ] 三个 checkpoint 各自 fit-only protect/harm threshold 已冻结。
- [ ] 三个 run 的 config/code/sample/batch/objective/checkpoint/threshold hash 完整。
- [ ] 三个 run 的 technical/execution gate 全过：数值有限、无 OOM/NaN/Inf、checkpoint 完整 strict reload、两 head 更新且不共享；任一失败整体 STOP。
- [ ] 初始化与 final strict-reload 均在 `eval()`/dropout-off、同一完整 fit rows/masks/class weights 下重算；NIAH protect、NIAH harm、2Wiki protect full-fit BCE 均有限且严格 `final < initial`。
- [ ] 2Wiki masked harm-head gradient exact zero。
- [ ] NIAH 与 2Wiki 均≥61/64 才取得 screen classification qualification；单个 variant 低于门只预先淘汰该 variant，按复杂度继续其他已冻结合格者；三个都不合格则不读 screen并 STOP。

A-screen：

- [ ] canonical roots 固定为计划第 9.0 节的 `/scratch/fl25387/IBM_Granite_Project_latest/{runs,audit,audit-journal}/selector-r005-amendment-v1`，无 CLI override/symlink；新目录及逐级 parents 已 `fsync` 到 namespace anchor。
- [ ] session ID 固定为 `selector-r005-amendment-v1--R005A-screen--a992f1b76abfac727981a79cdaf20d034aa7f8641d8e5d4a7713c9ecfa5bd15c`，唯一 split anchor/final bundle/veto 路径全部绑定；换 root/session/path 一律 invalidation/STOP。
- [ ] canonical OWNER_LOCK inode/`st_dev` 已由 anchor 绑定；owner 在揭示、发布、五件链检查和授权记录全程持有 exclusive flock；竞争者拿不到锁即零 access/零 mutation 退出，只有取得锁的 recovery 可 burn incomplete STARTED。
- [ ] A session `inputs/` 下恰有计划列出的五个 literal-path canonical JSON object：full members及 config/checkpoint/fit-gate/threshold projections；逐文件 `schema_version`/`manifest_type`/session/split、exact member fields、ordinal/ID、path/SHA/count和投影一致性全部复验，缺失/多余/交换/独立构造均拒绝。split anchor、STARTED与reveal-input manifest均绑定五者。
- [ ] 除冻结 sealed preflight 外，held-out assignment/query/text/label/token 的任何读取都须发生在 write-once STARTED marker + 唯一 matching fsynced STARTED event 之后。
- [ ] 一个冻结命令按 V0→V1→V2 reveal。
- [ ] 第一个两 endpoint 都≥92/96 且 lower≥.90 的 variant 立即冻结。
- [ ] 前一个未失败前不读取下一个；第一个 PASS 后不读取更复杂者。
- [ ] 没有按最高分挑选、没有修改阈值/代码/样本。
- [ ] 无中间 score/计数输出；V0/V1 的内部失败只留私有 staging，只在首个 PASS 或三者全 FAIL 后发布一次完整 session bundle。
- [ ] 任何 held-out access 后异常、STARTED owner/EEXIST 冲突、orphan、目标已存在或 verify 失败都先安装 split-level veto并 STOP；没有在同一 split 换 session 或重跑。
- [ ] 8-file bundle 的 schema/rows/cells/合法 eligible-prefix closed-world 完整；staging files/dirs fsync、read-only、`RENAME_NOREPLACE`、destination-parent fsync、final-path 全量 verify 后才写 matching fsynced COMPLETE_INTENT并竞争唯一 COMPLETE/BURNED terminal。
- [ ] verifier 已从 sealed input+strict checkpoint重算 raw float32 score bit patterns，并从 raw/fit source 重算 thresholds、qualification、composites、cell counts/rates/CP、合法 prefix与 gate；派生 canonical bytes 全相等。
- [ ] 仅“实际 STARTED marker + 唯一 matching STARTED event + verified bundle + 唯一 matching COMPLETE_INTENT + COMPLETE terminal”五者 ID/hash 一致、event 集精确且 split veto 不存在时接受；任何 invalid COMPLETE 永久安装 veto，后来补文件也不能复活。
- [ ] valid A COMPLETE 与无 veto在持有 A OWNER_LOCK 时重验；matching `R005A_TO_R005B.json` authorization 写入 session events 目录外并 `fsync` 后才释放锁，session events 仍恰好两项。
- [ ] 无 variant PASS 时正式 STOP，不运行 R005B。

## 6. R005B Gate

在任何非 sealed-preflight 的 B-confirm held-out access 前：

- [ ] 只使用唯一 V*。
- [ ] seed42/73 都从原 base 在相同 B-fit 独立训练。
- [ ] 两个 checkpoint、各自 fit-only threshold、所有 hashes 和 verifier 先全部冻结。
- [ ] 两 seed 的 NIAH 与 2Wiki fit composite 均≥61/64。
- [ ] 任一 fit gate FAIL 时 B-confirm 保持未读。

B-confirm：

- [ ] canonical roots/目录持久化规则与 A-screen 相同；session ID 固定为 `selector-r005-amendment-v1--R005B-confirm--676cdb6cfb51626b98959a53ea7d81774b042cc64be0e4e6a09ba3046016832a`，唯一 split anchor/final bundle/veto 路径全部绑定。
- [ ] B-confirm 的 canonical OWNER_LOCK 规则与 A-screen 相同；竞争/恢复/授权均没有 lock-veto TOCTOU。
- [ ] B session `inputs/` 下恰有计划列出的五个 literal-path canonical JSON object；full members按 seed42,seed73，四 projections的 exact fields、path/SHA/type/count 与 full 投影逐项一致，且两个 member 的 A authorization SHA/A attempt/V*一致；缺失/多余/交换均拒绝。split anchor、STARTED与reveal-input manifest均绑定五者。
- [ ] 除冻结 sealed preflight 外，held-out assignment/query/text/label/token 的任何读取都须发生在 write-once STARTED marker + 唯一 matching fsynced STARTED event 之后。
- [ ] 同一冻结命令第一次读取并评分两个 seed。
- [ ] S42 NIAH ≥122/128 且 lower≥.90。
- [ ] S42 2Wiki ≥122/128 且 lower≥.90。
- [ ] S73 NIAH ≥122/128 且 lower≥.90。
- [ ] S73 2Wiki ≥122/128 且 lower≥.90。
- [ ] 未合并成 n=256、未挑 seed、未平均抵消。
- [ ] 无中间 seed/dataset 结果输出；四格联合 gate/checksum 只发布一次完整 session bundle。
- [ ] 任何 held-out access 后异常、STARTED owner/EEXIST 冲突、orphan、目标已存在或 verify 失败都先安装 split-level veto并 STOP；没有在同一 split 换 session 或重跑。
- [ ] 8-file bundle 恰有 10,240 candidate、512 composite、4 cell rows；staging fsync/read-only、`RENAME_NOREPLACE`、destination-parent fsync、final-path 全量 verify 后才写 matching fsynced COMPLETE_INTENT并竞争唯一 terminal。
- [ ] verifier 已从两 strict checkpoints 与 sealed input 重算全部 raw score bit patterns，并逐层重算四格 counts/rates/exact CP 与 all-must-pass；派生 canonical bytes 全相等。
- [ ] 仅“实际 STARTED marker + 唯一 matching STARTED event + verified bundle + 唯一 matching COMPLETE_INTENT + COMPLETE terminal”五者 ID/hash 一致、event 集精确且 split veto 不存在时接受；任何 invalid COMPLETE 永久安装 veto，后来补文件也不能复活。
- [ ] valid B COMPLETE 与无 veto在持有 B OWNER_LOCK 时重验；matching `R005B_TO_R006A_PROTOCOL_PROPOSAL_ONLY.json` 写入 session events 目录外并 `fsync` 后才释放锁，且明确不授权 R006A GPU run。
- [ ] 任一格失败后未换 variant 或调参重用 B-confirm。
- [ ] 一份 R005A screen session bundle 与一份联合 R005B confirm session bundle 的 verify-only/checksum/独立审计均通过；不是按 seed 分别发布两个 confirm bundle。

## 7. 解释边界

R005A/B PASS 只支持“选中的训练配方在满足预冻结 eligibility 的 fresh train-fit components 上跨两个新种子复现 sanity 分类/配对能力”。它不支持：

- 所有 query 95% 安全；
- Selector 已产生非零删除；
- 优于 TopK10；
- harmful reduction/recall/chain/CRC/formal PASS；
- 现实世界 misinformation detection。

确认样本以后可进入 R006A 的全量 train-fit 训练，所以它们只保留为 development sanity 证据；真正独立效果证据仍依赖原冻结的后续角色。

## 8. 决策日志

| 日期 | 决策 | 理由 | 影响 |
|---|---|---|---|
| 2026-08-12 | 不以移动 0.5 作为修复 | 原 R005 两头都不存在满足原门的单一阈值 | 新实验必须改变/检验表示或目标 |
| 2026-08-12 | V0→V1→V2，第一通过者优先 | 每步只回答一个假设，并固定简洁性优先 | 不按最高 screen 分数挑模型 |
| 2026-08-12 | V1 使用 exact NLI one-vs-rest log-odds | 线性“目标行减平均”不能精确复现原 softmax | 初始化时严格保留原 NLI 概率 |
| 2026-08-12 | 三 variant 共用 pair-preserving batch | V2 不能多一次 forward/dropout/计算预算 | V0/V1 是 amendment control |
| 2026-08-12 | 采用 2Wiki eligibility B | eligibility A 引入检索偏差且不提高真实 chain complete | TopK10 至少一条 support；保护全部 TopK10 support |
| 2026-08-12 | query composite 为主 endpoint | 同题多行/多方向相关，不能冒充独立 n | 每 component 代表 query 只计 0/1 |
| 2026-08-12 | 两 B seed 在 confirm 首读前全部冻结 | 防止 seed42 结果影响 seed73 或方法 | confirm 一次性四格 all-must-pass |
| 2026-08-12 | R005B PASS 只解封 R006A | sanity 不是 Selector 效果 | TopK10 继续默认，R007–R015仍 CUT |
| 2026-08-12 | full-fit loss 只按 eval/dropout-off 同行同权重重算 | 避免用训练态或不同分母事后解释“loss下降” | 三个 active source/head BCE 必须严格 final<initial |
| 2026-08-12 | screen/confirm 只发布一次；canonical split anchor/veto 与 durable STARTED 后由 COMPLETE/BURNED 互斥终态收口 | 防止换 root/session、部分结果、崩溃恢复或并发进程形成重试通道 | valid COMPLETE 五件链、精确 event 集与无 veto 缺一即 STOP；同一 split 不得重跑 |
| 2026-08-12 | v1 未批准草案由 document revision v2 取代 | A001 代码映射复核发现 terminal 顺序、projection schema、B authorization与旧 verify-only 数据边界四处遗漏 | v1 永不执行；v2 重新审计通过后才可请求批准 |
