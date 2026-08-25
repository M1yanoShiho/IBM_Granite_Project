# G220-S draft LoRA smoke 报告

**日期：** 2026-08-16
**状态：** `COMPLETE / SMOKE PASS`
**正式服务器路径：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G220-v1`

## 1. Smoke 验证了什么

G220-S 不判断答案是否提升。它只验证正式训练所需的工程条件：

- 训练输入确实是 G200 的生产 `DRAFT_PROMPT`，不是 F006 key-fact notes；
- GC/GM 使用同一 query、同一 8 个 example slots、同一语义目标和相同更新步数；
- loss 只作用于 assistant target tokens；
- 最长上下文不会截断或 OOM；
- LoRA 权重能保存，并能在 fresh Granite base 上重新加载；
- 训练不读取 decision-dev、sealed600 或 system held-out。

## 2. 长度冻结

长度审计覆盖 577 个 G200 train/model-val 问题，共 4,616 个样本/arm。

| Arm | p50 | p95 | p99 | Max | 超过 2048 | 超过 2304 |
|---|---:|---:|---:|---:|---:|---:|
| GC | 544 | 1,396 | 1,536 | 1,605 | 0 | 0 |
| GM | 1,744 | 1,907 | 1,996 | 2,120 | 18 | 0 |

因此正式配置冻结 `max_length=2304`。选择 2048 会静默丢失 18 个 GM 样本，不满足等量、零截断设计。

## 3. 最长上下文 smoke

两臂都使用相同的 4 个最长 train query 和 4 个最长 model-val query。每臂 32 个训练样本、4 次 optimizer update，seed 为 13。

| 项目 | GC | GM |
|---|---:|---:|
| observed max train length | 1,480 | 2,120 |
| truncated examples | 0 | 0 |
| peak CUDA memory | 9.18 GB | 10.09 GB |
| mean smoke train loss | 0.9177 | 0.9716 |
| adapter fresh-base reload | PASS | PASS |

Smoke loss 只表明 forward/backward 和目标格式工作正常；样本只有 4 个问题，不能用于比较 GC 与 GM 的方法效果。

## 4. 正式训练配置

以下配置在查看正式训练或 dev 结果前冻结：

| 项目 | 冻结值 |
|---|---|
| Granite base | `ibm-granite/granite-4.1-3b@c0650403...` |
| Arms | GC support-only control；GM eight-context mixed |
| Seeds | 13, 42, 73 |
| Train queries/examples | 515 / 4,120 per arm |
| Epochs / optimizer steps | 1 / 515 per arm |
| Microbatch / gradient accumulation | 1 / 8 |
| Learning rate | `1e-4` |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Max length / truncation | 2,304 / 必须为 0 |
| Adapter scope | 只在 draft generation call 启用 |
| Claim splitter / TRUE | frozen；不参与训练 |

这不是超参数搜索。正式训练只能按上表运行三种子；如果发生 non-finite loss、OOM、截断或 reload 失败，该 run 失败并记录，不能在 dev 上挑配置补救。

## 5. 产物

- 长度审计：`artifacts/G220/smoke/length-audit.json`；
- GC smoke manifest/config：`artifacts/G220/smoke/gc-seed13/`；
- GM smoke manifest/config：`artifacts/G220/smoke/gm-seed13/`。

Smoke adapter 权重保存在服务器 runtime，Git 归档 manifest、配置和权重 SHA256，不提交模型权重。
