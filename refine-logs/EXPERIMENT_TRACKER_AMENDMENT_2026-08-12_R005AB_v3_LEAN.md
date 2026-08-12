# Selector Lean v3 实验跟踪表

**对应计划：** `EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v3_LEAN.md`

**状态：** `L002 PASS / NLI-BASE POLICY FROZEN / READY FOR L003 / FINAL UNOPENED`

**状态词：** `NOT RUN / RUNNING / PASS / FAIL / BLOCKED / CUT`

## 当前一句话

旧 R005 证明 raw-CLS 双头没有形成可用的绝对删除边界；v3 保留 R001–R004 与现有保守动作层，只实现 NLI-aware scorer，用标准训练—开发—最终盲测检验是否能比 TopK10 略有提升。

## Run 队列

| Run | 目的 | 数据 | 关键输出 | 状态 | 下一步许可 |
|---|---|---|---|---|---|
| L000 | 冻结 Lean v3 计划、JSON 合同与 tracker | 不读取效果数据 | 一致文档；独立复核 P0=0/P1=0 | PASS | 进入 L001 |
| L001 | 实现 NLI scorer、pair loss、lean evaluator/CLI 与必要测试 | synthetic + 已暴露旧 R005/train smoke | clean commit；本地测试；服务器真实模型 smoke | PASS | 进入 L002 |
| L002 | 两 seed 训练并只在 development 选择 variant/quantile/cap | train-fit + crc-calibration | 两 checkpoint、唯一冻结策略、开发报告 | PASS | 允许一次 L003 final |
| L003 | 一次性 final：先证据门，再答案指标 | R002 `decision-dev`（NIAH 739q；2Wiki 1000q） | decisions、metrics、CI、最终报告 | NOT RUN | PASS 才能提出后续确认；FAIL 保持 TopK10 |

## L000 — 文档冻结

- [x] 用户批准保留必要隔离/盲测并删除过重工程防御层。
- [x] 方法、评估、代码三路只读盘点完成，未读取 final Selector 效果。
- [x] 主张收缩为 C1 与 C2。
- [x] V0 降为可选 appendix；主路线为 `NLI-base → 必要时 NLI-pair`。
- [x] 数据收缩为 train-fit / crc-calibration / decision-dev；sealed/heldout 本轮不读取。
- [x] 工程收缩为 3 个小模块、1 个配置与必要测试。
- [x] v3 MD/JSON/tracker 一致性复核通过，最终复核 `P0=0 / P1=0`。
- [x] 更新固定入口、manifest 与第08归档。

Git commit/push 只记在项目历史与本次交付说明中，不再为它增加单独状态机。

## L001 — 最小实现

- [x] 新增 `selector/nli_dual_head.py`，不改旧 R005 模型文件。
- [x] NLI 初始化的 protect/harm 概率与原三分类 softmax 对应。
- [x] 两 classifier 初值相同但存储不共享；shared dropout 每次 forward 只调用一次。
- [x] masked BCE、空 mask、NLI-pair `log(2)`/梯度方向和 2Wiki zero-gradient 测试通过。
- [x] 固定 source×head×class inverse-sqrt 权重、active-count BCE、完整 AdamW 参数与 1:1 source schedule。
- [x] pair/row digest、pair填充singleton、两源交替的每epoch batch顺序与合同完全一致。
- [x] NLI strict pair 同 microbatch/同 forward；其余 NIAH 与 2Wiki 只贡献合法 masked BCE。
- [x] 复用 `RiskControlledSelector`；0删除、cap1/2、TopK10 子集、冲突/缺分数 keep-all 测试通过。
- [x] Lean evaluator 的只读 evaluation projection 能从 R002 role/component 与 R001 pins 重建开发/最终指标字段；R004 labels 只用于 train-fit。
- [x] 阈值只由 train quantile + development 选择；final evaluator拒绝重新选择。
- [x] 本地全量相关测试、格式与静态检查通过。
- [x] 服务器 pinned model 的真实初始化、forward、短训练与 checkpoint reload smoke 通过。
- [x] clean commit 并 push GitHub。

## L002 — 训练与开发选择

- [x] `NLI-base` seed13、42 使用完全相同配方训练。
- [x] 每 seed 从自己的 train-fit safe-score 生成固定四分位点。
- [x] 每 seed 只使用一只全局阈值；固定池为全部 train-fit TopK10 分数按 NIAH→2Wiki 拼接，禁止 dataset-specific threshold。
- [x] nearest-rank=`ceil(pN)`、float32 score、`safe>=threshold` 与重复 threshold identity 均按合同实现。
- [x] development 只比较 `4 quantiles × cap{1,2} + P0`。
- [x] 先按两 seed NIAH harmful reduction算术平均找全局最大，`≤0.5pp`等价集内再按高分位点→小cap→少删除选择。
- [x] 两 seed 都有正 harmful reduction；seed13 recall/chain 各≤1pp；seed42 各≤3pp。
- [x] Selector 的 NIAH deletion precision/harmful reduction 优于等量 random 和 bottom-rank。
- [x] `NLI-base` 已通过，因此按计划不运行 `NLI-pair`，也不增加第三个 variant。
- [x] 冻结 `NLI-base`、两个 checkpoint、q=.99、cap=2、commit `4fb68fa`、最终输入、development projection 与 Granite 配置。
- [x] development PASS；允许且只允许一次 L003 final，TopK10 仍是当前生产默认。

## L003 — 最终盲测

- [ ] 开始前复核 frozen config/commit/checkpoint/data hashes，final 之前无效果访问。
- [ ] 两 seed 同批运行 R002 `decision-dev`，不看一个结果后修改另一个；sealed600/2Wiki heldout 保持未读。
- [ ] seed13 harmful reduction CI lower>0；seed42 harmful point>0。
- [ ] seed13 recall/chain 点损失各≤1pp、CI upper各≤3pp；seed42 点损失各≤3pp。
- [ ] seed13 的 count-matched random-100/bottom-rank 对照完成，harmful reduction 与 deletion precision 分别优于两者；seed42只做方向/保护复现。
- [ ] 分别报告 Selector 相对 random mean 与 bottom-rank 的 harmful-reduction paired component CI；两条下界都>0才称 C2 有统计支持。
- [ ] P0逐题等于 TopK10；全部输出为 TopK10 子集。
- [ ] 正式 CI只用seed13逐题差值做component bootstrap；seed42单独报告方向/保护点值，不与seed13拼样本。
- [ ] 证据门 PASS 后，用 seed13 与冻结 Generator 评估 `system.core.answer_match`。
- [ ] Generator 固定为 Granite 4.1 3B revision `c065040...`、固定 prompt、32 tokens、greedy；不能误用 extractive。
- [ ] 实际加载前核对本地 snapshot revision、weights/config/tokenizer/chat-template hashes，并把 resolved snapshot path 传入客户端；不回落到环境变量/最新版。
- [ ] 两数据分别按 component bootstrap 后 50:50 macro answer delta>0，且任一数据集 answer loss不超过1pp。
- [ ] 报告全 decision-dev、2Wiki parent-seen-in-train-fit 与 unseen-in-any-used-data 敏感性。
- [ ] 保存 PASS/FAIL 结果；final 后不改方法重新试。
- [ ] commit 并 push GitHub。

## 决策日志

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-12 | v2 标为 superseded/never run，改用 Lean v3 | v2 的锁、状态机、五任务注册与当前小幅提升目标不成比例 |
| 2026-08-12 | R001–R004 全部复用，旧 R005 保留 FAIL | 前四阶段提供有效资产；R005 精确定位到评分器绝对边界失败 |
| 2026-08-12 | 只保留 NLI-base 和条件触发的 NLI-pair | 直接针对丢失 NLI 表示与配对 margin 两个已知问题，避免多路线并跑 |
| 2026-08-12 | 每题允许删0条，cap只在1/2中选择 | 比固定最多1条灵活，同时至少保留8条，符合保守目标 |
| 2026-08-12 | L001 PASS，进入 L002 | 本地相关回归、格式、类型检查通过；服务器真实 NLI forward、单步训练、checkpoint reload 与 38 项测试通过 |
| 2026-08-12 | L002 PASS，冻结 NLI-base q=.99/cap2 | 两 seed harmful reduction 10.77%/10.47%，保护损失受控，优于等量删除对照；final 尚未打开 |
| 2026-08-12 | 不再声称 CRC guarantee | 只保留开发/最终隔离和paired CI；方法名称改为经验验证的保守 Selector |
| 2026-08-12 | 最终答案主指标固定为 macro `system.core.answer_match` | 直接回答是否比 TopK10 有哪怕小幅下游提升，避免结果后挑指标 |
| 2026-08-12 | train-modelval 不读取；decision-dev 成为本轮唯一 final | 保持真正的 train-fit→development→final 三角色，并直接复用 R002 已冻结 component map |
| 2026-08-12 | sealed600/official heldout 本轮保留不读 | sealed600 已有历史 aggregate exposure；decision-dev 足以完成当前最小验证，外部集留待后续确认 |
| 2026-08-12 | 固定真实 Granite Generator，不使用 extractive | delete-only 子集在 extractive generator 下不可能产生正 answer delta |
| 2026-08-12 | 删除“training loss 必须下降”硬门 | 只保留有限值、无 NaN/OOM、双 head 确实更新；避免为非必要趋势检查增加额外执行层 |
