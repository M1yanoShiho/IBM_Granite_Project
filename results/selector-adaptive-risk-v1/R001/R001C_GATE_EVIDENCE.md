# R001C Hybrid-v2 Candidate Pool Freeze

**状态：** PASS

**实现提交：** `f05060a24b25423918f8b5f27465929d606e5047`

## 结论

六个历史 Hybrid RRF Top20 pool 已在原服务器位置生成独立、write-once 的 `SelectorCandidatePoolManifestV2`，随后使用 `--verify-only` 从磁盘重新读取并完整复验，六个角色全部通过。R001 的 recover/rebuild 决策因此从“可恢复”升级为“已经按 v2 协议重新冻结并验证”。

这一步没有训练 scorer，也没有读取 sealed600 或 2Wiki heldout 上的新 Selector 效果。旧 BM25-only `CandidateFreeze`、Gate0A 和生产 `top-k` 默认行为均未改变。

## 实际验证内容

每个 pool 的验证同时要求：

1. `RunManifest.top_k=20`，不是 Top50 事后截断；
2. retriever 固定为 `hybrid/hybrid-v1`，参数摘要固定为 `67333c6f…d78d`；
3. fusion 为 RRF `k=60`；
4. run manifest、candidate metadata、index manifest、root corpus/query artifact 的 signatures 与 bundle hash 互相一致；
5. dataset manifest、documents、queries、gold 的原始 hash 与 dataset signature 一致；
6. candidate query 内容、顺序和集合与 dataset queries 完全一致；
7. 每题恰好 20 条，tuple 顺序中的 rank 必须严格是 `1..20`；
8. 每条 candidate 的 evidence/document/chunk ID、文本、URI 和 metadata 必须逐字段匹配 signed corpus chunk；
9. 每个完整 `CandidateSet` 保存独立 canonical SHA-256；
10. `exact-recovery` 不是自报标签：每个 data role 必须命中 R001A/B 已审计的 candidate hash、dataset signature 和 query 数。

## 六池结果

| Data role | Queries | Candidate SHA-256 | v2 manifest SHA-256 | Freeze | Verify-only |
|---|---:|---|---|---|---|
| NIAH train | 2,000 | `09e8c9b4…908f` | `cdab40bc…a5b5` | PASS | PASS |
| NIAH dev | 2,000 | `89ede824…908f` | `80224272…6472` | PASS | PASS |
| NIAH sealed600 | 600 | `777391fa…590f` | `18e7a7da…fea3` | PASS | PASS |
| 2Wiki train | 3,000 | `0ab0fc92…8887` | `067cad9e…326d` | PASS | PASS |
| 2Wiki dev | 2,000 | `26442003…607f` | `fd63e3e6…9202` | PASS | PASS |
| 2Wiki heldout | 2,000 | `fcca691e…219` | `2e8d2f4f…c53e` | PASS | PASS |

完整 hashes、query/index signatures 和机器可读结论见 `R001C_VALIDATION_REPORT.json`；六份逐 query manifest 位于 `manifests/`。

## 测试证据

- 本地合并最新远端提交后：`1179 passed`；
- 服务器 Python 3.11.15 新增测试：`21 passed`；
- Ruff：PASS；
- 新模块 Mypy strict：PASS；
- 旧 BM25/Gate0A 定向回归：PASS。

## 边界

R001C 的 pool-freeze 部分已 PASS，但 R001 整体和 M0/Gate 0 还没有 PASS。R002 仍需把 component map、严格 paired cluster inference、四项 expected-risk CRC 的代表 query 与有效 component 数正式冻结；R003 还需冻结数量基线和 count-matched controls。在这些步骤完成前，不开始 scorer 训练。
