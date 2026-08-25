# Experiment 05 — General RAG reliability and traceability

本目录是 Experiment 04 之后的新一轮完整系统实验计划。

计划 v3 已完成独立复审。用户随后启动实验，并在 formal 输出与评分均为零时授权 v4
快速执行修订。Goal 1–5 现已全部 PASS，最终技术状态为 `FINAL PASS`；实验已按计划停止。
三个数据集、十个实验臂共完成 12,000 条 generation outputs 和 12,000 条正式评分，
`scorer_errors=0`。预注册规则下 Claim A 与 Claim B 均为 `NOT SUPPORTED`，属于完整且可追溯
的科学负结果，而不是实验失败。

快速修订保留完整语料检索，但采用实用的两阶段方式：先用完整 BM25 index 取 Top-1000，
再由 Granite dense 在候选内评分并与 BM25 rank 融合。它不再构建或声称使用全语料独立
dense index。

## 这轮只改变什么

1. 主数据改为三个普通 RAG 任务：KILT–Natural Questions、KILT–TriviaQA、ALCE–ASQA；
2. Selector 去掉 `max_delete` 和 `minimum_retained`，只按 harm/protect 双门槛删除；
3. 主评分改为事实覆盖、带有效引用的事实覆盖、无依据事实、引用 precision/recall 与回答率。

主文 Table 1 与 Table 2 使用相同的六个主指标 `RFC/VRFC/UCR/CP/CR/RR`。`ER@10/SELR/CRR` 只作为附录诊断信息；CRR 仅描述上下文删除量，不参与系统优越性或 PASS 判断。

正式阶段采用一次性 gold 解锁：Goal 3/4 只生成并 hash 冻结全部十臂、12,000 条输出，Goal 5 才解锁 scorer-only sidecar 并统一生成 Table 1/2。评分只读取 Generator 实际看到的裁剪后 sealed text artifact，不能因 passage ID 相同而回读原始全文。

## 保留什么

- BM25、两阶段 Hybrid、Granite Rerank、Provence、Ours 五系统矩阵；
- Retriever、Selector、Generator 三个模块级消融；
- 相同 Top10、token budget、Granite base、三个 Ours seeds、公平失败分母与独立 MiniCheck scorer；
- Experiment 04 的全部负结果和审计边界。

## 文件

- [PLAN.md](PLAN.md)：完整可执行方案与 Goal 接力规则
- [TRACKER.md](TRACKER.md)：完整执行、恢复与最终审计记录
- [PLAN_SELF_REVIEW.md](PLAN_SELF_REVIEW.md)：自一致性核对与独立复审记录
- [results/final/](results/final/)：最终报告、Table 1/2、bootstrap、Claim 标签与 hash 审计
