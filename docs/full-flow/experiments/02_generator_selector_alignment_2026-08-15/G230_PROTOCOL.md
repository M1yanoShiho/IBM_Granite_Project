# G230 Generator 开发验证冻结协议

**状态：** `FROZEN BEFORE GENERATION`
**用途：** 在读取 G230 结果前固定输入、运行臂、adapter 作用域和通过门。

## 1. 研究问题

G230 只判断同一 Granite 4.1-3B 基座上的新 Generator 是否值得冻结为后续 Selector utility 教师。它不形成 held-out 最终结论，也不改变 Retriever 或 Selector。

## 2. 运行臂

| 配置 | 训练/adapter | adapter 启用位置 | seeds |
|---|---|---|---|
| G0 | 无 | 无；复用 A002/B100 的确定性 Base 输出 | 固定基线 |
| GN | 历史 F006 Mixed LoRA | 仅 key-fact extraction call | 历史 seed 13，一次 |
| GC | G220 clean LoRA | 仅 draft generation call | 13/42/73 |
| GM | G220 mixed-context LoRA | 仅 draft generation call | 13/42/73 |

所有配置共享 frozen Granite base claim splitter、frozen TRUE、相同 prompt 和 greedy decode。GC/GM 每个 seed 在同一进程共享 Granite/TRUE，并按 task 稳定轮换执行顺序。

## 3. 数据与上下文

主结果使用完整 NIAH decision-dev 739 题的原 TopK10：

- 这是隔离 Generator 主效应的完整分布比较；
- 与后续 I410 的 `TopK + G*` 对照一致；
- gold/reference 不进入生成命令。

B100 冻结的 218 题 matched diagnostic subset 只作机制分析：

- Legacy Selected；
- O support-only；
- O+B support + benign；
- O+H support + harmful；
- O-P support-last。

该子集不能替代 739 题主结果。sealed600 和 system held-out 均不读取。

## 4. 运行与评分边界

`run` 命令只接受 query、候选、Selector trace、B100 runtime contexts、模型和 adapter，不接受 gold。每个长运行先写 hash-bound `run_spec.json`，随后 append-only 写逐 task 产物；中断后只允许在相同 spec 下按严格前缀续跑。

生成全部完成后，`score` 才读取 dev gold。引用由独立 MiniCheck 逐句评分；生产路径 TRUE 不作自己的裁判。

## 5. 冻结开发门

GC 和 GM 分别先检查：

1. 三个 seed 在完整 739 题上的 answer-match 点估计都高于 G0；
2. 三个 seed 的 coverage 配对 95% CI 下界都不低于 `-1pp`；
3. 三个 seed 中，`draft empty / zero claims / final empty` 至少一项相对 G0 绝对下降至少 `1pp`；
4. 三个 seed 在 support-only 上同时满足 answer-match 高于 G0 且 final-empty 低于 G0；
5. 三个 seed 的 MiniCheck citation precision 和 recall 配对 95% CI 下界都不低于 `-2pp`。

GM 还必须在 O+B、O+H、O-P 三个 stress context 上，每个 seed 的 answer-match 点估计均高于同 seed GC。只有这样才能把收益归因于 mixed-context robustness。

候选选择规则固定为：

```text
GM 通过自身门且通过 mixed-context robustness 门 -> 选择 GM
否则 GC 通过自身门                         -> 选择 GC
否则                                       -> G230 无候选
```

不选择单个最好 seed，不在 G230 结果后修改阈值、训练配置、上下文集合或失败率定义。若 GC/GM 都不能改善 support-only answer 和 final-empty，则按 PLAN 停止进入 Selector utility 训练。

## 6. 必须产物

- GN 一份、GC/GM 三 seed 三份 runtime manifest 与逐 task generations；
- answer/coverage/failure-stage 配对报告与逐 task score；
- MiniCheck citation 配对报告与逐 task score；
- 三 seed 方向、W->R/R->W、错误数、缺失 trace、输入/adapter/prompt/model hash；
- 最终 `GM / GC / no candidate` 判定。
