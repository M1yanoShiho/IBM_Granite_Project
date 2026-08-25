# G210 stop and authorization packet

**日期：** 2026-08-18  
**状态：** `STOPPED / G210 HARD DATA GATE FAILED / USER AUTHORIZATION REQUIRED`  
**G210 failure triage commit：** `17380c80dc82e98d2ea42546d09402ceb91b1ad4`

## 停止原因

G210 已按计划完成 structural audit、TRUE audit 和 pre-manual finalize。结果是 hard data gate failure：

```text
2Wiki model-val answerable groups = 76 < required 100
```

因为这是计划中的最低数据门，不是建议门或软门，所以当前路线不能进入：

- G300 training implementation；
- Generator LoRA training；
- utility label generation；
- S 阶段 Utility Selector；
- I 阶段完整系统；
- H 阶段 held-out。

## 已证明的边界

- G200 数据预物化完成，但只是 `PRE_AUDIT PASS`；
- G210 structural audit：3,108/3,108 case 通过；
- G210 TRUE audit：2,758 rows 中 2,140 entailed、618 not entailed；
- TRUE 过滤后：
  - NIAH train/model-val：515 / 215；
  - 2Wiki train/model-val：683 / 76；
  - unsupported update ratio：12.4855%；
  - train/model-val group overlap：0；
  - train/model-val component overlap：0。

## 当前禁止事项

在没有用户单独授权前，禁止：

1. 降低 2Wiki model-val 最低门；
2. 降低 TRUE threshold 或改 TRUE 模型；
3. 改 seed、重切 split 或把 dev/held-out 混入训练选择；
4. 用 HotpotQA、MuSiQue-Full、RGB 或 sealed600 调参；
5. 在 G210 failure 数据上启动训练；
6. 声称 Generator 修复成功、GQ 已冻结、S 阶段可启动或 SystemF 可开发。

## 若要恢复推进，必须另行批准的新计划

任何恢复方案都必须是新的受控数据修订计划，并且至少明确：

- 从哪个阶段重新开始：通常应回到 G200/G210，而不是从 G300 继续；
- 是否允许修改 2Wiki target construction 模板；
- 是否允许修改 2Wiki target audit protocol；
- 是否仍使用 TRUE 作为 hard support audit；
- 如何保持 train/model-val component isolation；
- 如何证明没有读取 sealed600 或 system held-out；
- 新的 manifest、ordered IDs、hash、server runtime 和 GitHub 同步要求。

## 可审议的修订方向

下面只是候选方向，不是授权：

| 方向 | 可能解决的问题 | 主要风险 |
|---|---|---|
| 改写 2Wiki triple 模板 | 减少 `country`、`publication date` 等模板与 evidence sentence 的 TRUE mismatch | 必须重新 G200/G210；不能比较为同一数据 |
| 对 2Wiki 使用 support sentence target 而非 triple target | 更接近 TRUE premise，可能提高 model-val 保留数 | 可能弱化多跳 reasoning supervision |
| 增大 2Wiki train split 内预审计候选池但保持 component isolation | 让 TRUE 后仍有 >=100 model-val | 必须证明不碰 dev/held-out，不重切泄漏 |
| 对 yes/no 题单独构造 explicit answer target | 让 answer alias 更直接 | 容易产生无单条 citation 支持的复合 claim |

## 归档引用

- G210 target audit：`G210_TARGET_AUDIT_REPORT.md`
- G210 failure triage：`G210_FAILURE_TRIAGE_REPORT.md`
- G210 artifacts：`artifacts/G210/`

## 阶段判定

当前路线按计划停止。下一步不是技术训练任务，而是用户是否批准新的数据修订计划。没有该授权时，本任务不能继续实现 G -> freeze -> S -> freeze -> I。
