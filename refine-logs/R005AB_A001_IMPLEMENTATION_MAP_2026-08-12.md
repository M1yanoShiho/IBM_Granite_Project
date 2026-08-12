# R005A/R005B A001 代码实施与验收映射

**日期：** 2026-08-12

**依据：** [`EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v2.md`](EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v2.md)

**状态：** `PRE-APPROVAL MAP / NOT IMPLEMENTED / NOT RUN`

这份文件回答一个具体问题：用户批准后，A001 到底要改哪些代码、哪些旧代码只能复用一小部分、每一步用什么测试证明正确。它不是 A001 实现，也不授权 A002、训练或 held-out access。

## 1. 总体决定

A001 使用一套独立、带版本的 R005AB 路径，不原地改造旧 R005。旧 R005 的四个核心 Git blob 必须保持不变：

| 旧文件 | 当前 Git blob | A001 规则 |
|---|---|---|
| `src/evidence_rag/selector/dual_head.py` | `6eb95ad0c3ab69d94366cdaa1e8d2f38da1f8775` | 只复用 tokenizer、masked BCE 等窄函数；文件不改 |
| `src/evidence_rag/evaluation/selector_sanity.py` | `1ecd822a76309fb7da3c694f1f2d047deee1cb38` | 旧 R005 语义保留；文件不改 |
| `src/evidence_rag/cli/run_selector_sanity.py` | `0f623dbe5678099fac507d0455cd4be57695ac77` | 不作为 A001/A002/reveal 入口；文件不改 |
| `src/evidence_rag/cli/finalize_selector_r005.py` | `6486cafc564dd20edbf4c75735f616fdc96e7507` | 只做 synthetic fixture 旧回归；文件不改 |

原因很简单：旧 loader 只实现 V0 型 raw CLS；旧 runner 会在 `--verify-only` 分支前加载包含 train-modelval 的旧输入；旧 finalizer 也没有新状态机要求的全程锁、目录 `fsync` 与永久 veto。把新功能塞回旧路径会同时破坏历史可复验性和新数据防泄漏边界。

## 2. 可以复用的窄能力

| 现有能力 | 允许怎样复用 | 禁止误用 |
|---|---|---|
| `tokenize_question_candidates`、`audit_token_lengths` | 保留 question + candidate、只截断第二序列、max length 512 | 不让 query ID、rank、label、provenance 进入模型 |
| `masked_dual_head_bce(..., active_count)` | 作为三 variant 共同 BCE 核心 | 不复用旧随机-head builder 充当 V1/V2 |
| `project_text_pair`、`text_pair_sha256` | 唯一模型输入投影与哈希 | 不用旧 label writer 输出 held-out raw rows |
| `component_id_for_queries`、`build_allowed_key_components` | component identity 与 DSU 规则 | 不复用旧 fold→role 自动分配 |
| `FilePin`、`pin_file`、`verify_file_pin` | 公开 fit 输入的普通 byte/SHA pin | 不作为 held-out capability boundary 或 durable marker |

旧 helper 多使用 `ensure_ascii=true`，不能用于 amendment 的 canonical bytes。R005AB 必须有独立 encoder：`ensure_ascii=false`、`sort_keys=true`、separators `(',', ':')`、UTF-8 无 BOM、末尾恰一个 LF。

## 3. 新增模块边界

| 新文件 | 核心对象或函数 | 单一责任 |
|---|---|---|
| `src/evidence_rag/infrastructure/durable_fs.py` | `canonical_json_bytes`、`fsync_file`、`fsync_directory`、`mkdir_durable_chain`、`atomic_install_noreplace_file`、`publish_directory_noreplace`、`exclusive_owner_lock` | hard-link no-replace、`renameat2(RENAME_NOREPLACE)`、same-device、逐级目录持久化与 nonblocking flock |
| `src/evidence_rag/selector/r005ab.py` | `AmendmentVariant`、`load_r005ab_model`、`pairwise_logistic_loss`、versioned checkpoint/fingerprint save/load | V0/V1/V2 模型、exact NLI one-vs-rest、pair loss、跨架构 checkpoint 拒绝 |
| `src/evidence_rag/evaluation/selector_r005ab_protocol.py` | namespace、formal-fit、reveal、authorization、ordered-manifest 的 strict schemas | 所有 literal path、五 job、两 session 与 exact fields 的机器合同 |
| `src/evidence_rag/evaluation/selector_r005ab.py` | batch manifest、threshold enumeration、query composite、exact CP、fit/screen/confirm gates | 无 I/O 的确定性统计与训练合同纯函数 |
| `src/evidence_rag/evaluation/selector_r005ab_state.py` | registry freeze、fit begin/terminal/recovery、reveal start/complete/recovery、authorization | formal-fit 与 reveal 两套不可重试状态机 |
| `src/evidence_rag/evaluation/selector_r005ab_bundle.py` | 八文件 schema、row counts、checksums、A prefix/B four-cell closed-world verify | 发布前后的结构闭包 |
| `src/evidence_rag/evaluation/selector_r005ab_verifier.py` | fit closed-world audit、checkpoint raw-forward 重算、逐层派生重算 | 不信任自报 summary 的独立语义复验 |
| `src/evidence_rag/materializer/selector_r005ab.py` | assignment/candidate/pair schemas、fit materialization、aggregate-only sealed preflight、STARTED 后 private rebuild | A001/A002 数据能力边界 |
| `src/evidence_rag/cli/materialize_selector_r005ab.py` | 唯一 sealed A002 worker 入口 | 不提供任意 root、role 或 raw held-out export 参数 |
| `src/evidence_rag/cli/run_selector_r005ab.py` | formal fit、reveal 与 verify-only orchestration | 生产路径只来自 protocol constants |
| `configs/selector/adaptive_risk_r005_amendment_v1.toml` | 训练与协议固定值 | 绑定 amendment JSON SHA；不改旧 R005 config |

## 4. A001 必须冻结的数据 schema

### Assignment row

每个 role × dataset × component 一个代表 query，字段恰为：

```text
schema_version, protocol_version, sample_seed, eligibility_id,
role, dataset_kind, role_index, component_id, query_id,
representative_digest, component_order_digest, sample_digest
```

### Candidate row

每个 assignment 的 Top20 candidate 一行，字段恰为：

```text
schema_version, protocol_version, role, dataset_kind, component_id,
query_id, evidence_id, document_id, retrieval_rank,
candidate_text_sha256, text_pair_sha256,
protect_label, protect_mask, harm_label, harm_mask
```

公开 sidecar 不包含 question、candidate text、provenance、gold、source-parent 或 retrieval score。文本只能由获授权 worker 从 pinned source 重建。

### NIAH strict-pair row

```text
schema_version, protocol_version, role, dataset_kind="niah",
component_id, query_id, pair_id,
clean_evidence_id, counterfactual_evidence_id,
clean_text_pair_sha256, counterfactual_text_pair_sha256
```

缺失、重复、同一 evidence 双占位、跨 query/component、标签不匹配、非 TopK10 或歧义 pair 全部拒绝。不得只凭两个 label 组合临时猜 pair。

### Pair-preserving batch row

```text
schema_version, protocol_version, epoch, dataset_kind, source_round,
microbatch_index, member_index, query_id, evidence_id,
pair_id_or_null, batch_digest
```

三个 variant 必须使用完全相同的 batch manifest SHA。每个 strict pair 同一 microbatch；每 epoch 每个 candidate 与 pair 恰出现一次。

### Reveal candidate-score row

除计划要求的 candidate identity、rank、mask、label 与两只 JSON number 外，同时固定保存 `protect_float32_hex` 与 `harm_float32_hex`。verifier 以 bit field 为主，从 strict checkpoint 重做 forward 后要求 IEEE-754 float32 bits 完全一致，避免 Python float 序列化掩盖差异。

## 5. A001 实施顺序

1. **纯 schema 与统计层。** 先实现 canonical encoder、strict records、pair batch、threshold/tie-break、query composite、Clopper–Pearson 与三个 gate；只跑 synthetic fixture。
2. **版本化模型层。** 实现 V0/V1/V2、exact one-vs-rest、一次 shared dropout、独立 classifier storage、pair loss与 checkpoint cross-load 防护。
3. **sealed materializer 防火墙。** 以 fake provider 证明 screen/confirm preflight 只能返回 aggregate count/hash/isolation，不能写 raw row、text、label、token 或 score。
4. **durable 文件与状态机。** 实现 literal namespace、五 job registry、lock/no-replace/fsync、fit terminal顺序、五件 reveal 链、永久 veto与 B authorization SHA 链。
5. **closed-world bundle 与语义 verifier。** 冻结八文件和所有 row schema；从 strict checkpoint+sealed source重算 raw bits，再重算 threshold→qualification→composite→cell/CP→gate。
6. **CLI 与配置。** 只暴露协议允许的动作，不允许覆盖生产 root/session/split/final path；测试注入 root 只能存在于测试依赖层。
7. **全量验收。** 跑专项测试、旧文件 blob 检查、synthetic 旧 finalizer zero-write回归、全仓 pytest、Ruff、mypy和独立代码审计。
8. **冻结出口。** 形成 clean A001 commit并推送 GitHub；服务器安全快进并证明 branch/commit相同且 clean。完成前不允许 A002。

## 6. 最低完整测试矩阵

| 组 | 必须覆盖 |
|---|---|
| 模型 | label map/shape/key fail-closed；V1概率等价；head storage独立；dropout一次；V2 `log(2)`与梯度方向；2Wiki pair/harm exact zero；strict checkpoint roundtrip/cross-load拒绝 |
| 数据 | whole-component exclusion；64/96/64/128切片；四角色 query/component/candidate/text-pair/active-content隔离；NIAH strict pair；2Wiki至少一条TopK10 support；canonical hash mutation |
| batch/threshold/gate | pair不拆/不漏/不重；三variant相同SHA；`nextafter`、midpoint、tie-break；61/64、92/96、122/128与 exact CP 边界 |
| sealed firewall | pre-STARTED provider zero-call；aggregate-only无raw落盘/日志；无matching STARTED+owner authority时 private rebuild拒绝 |
| formal fit | 五 literal jobs；duplicate/alternate root/second checkpoint/超预算拒绝；STARTED在forward/optimizer前；crash后只FAIL；terminal前全部fit artifact已冻结 |
| B authorization | 缺失/篡改/换路径 auth；A veto；两B job auth SHA/A attempt/V*不同；锁序错误均拒绝 |
| reveal durability | 两进程竞争；loser zero access/zero mutation；每个fsync/crash窗口；orphan；STARTED无terminal；invalid COMPLETE补文件不能复活 |
| bundle/semantic | 八文件少/多/symlink；A非法prefix或PASS后多评分；B非10240/512/4；raw bit或任一派生层mutation全部拒绝 |
| 回归/质量 | 旧四文件blob不变；synthetic旧finalizer零写；全仓pytest、Ruff、mypy；独立xhigh代码审计无阻断项 |

## 7. A001 的明确数据边界

A001 允许读取：原 R005 的 `train-fit` 证据、计划/代码、synthetic fixture、公开模型 snapshot metadata。A001 不得调用会装载全旧 role universe 的 component/label/preflight/R005 runner 来处理新角色。

A001 禁止读取、打印或物化任何新 A-screen/B-confirm query ID、assignment、文本、标签、token或score；禁止运行 A002、任何 formal fit 或 GPU训练；禁止修改旧四个核心文件。

历史正式 R005 的真实 runner复验不在 A001 重跑，因为旧 runner在进入 `--verify-only` 前会读取含 train-modelval 的输入。A001只引用现有 [`R005_VERIFICATION_ATTESTATION_2026-08-12.md`](R005_VERIFICATION_ATTESTATION_2026-08-12.md)，并用 synthetic fixture 验证旧 finalizer仍为零写。

## 8. A001 PASS 的唯一证据组合

A001 只有在以下全部成立时才是 PASS：

- amendment v2 的 MD/JSON/tracker SHA 与实现 config pin 一致；
- 第11节17组要求及本文件测试矩阵全部有机器日志；
- 旧四个 Git blob保持上表值；
- 没有新角色 raw/held-out artifact，没有 canonical formal/reveal root被消费；
- 代码审计无 P0/P1；
- clean implementation commit已到 GitHub；
- 本地、GitHub、服务器三处指向同一commit，服务器worktree clean。

任何一项缺失都不能开始 A002。
