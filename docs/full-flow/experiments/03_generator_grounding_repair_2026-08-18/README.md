# Generator 修复与 Selector 分阶段协同

**日期：** 2026-08-18
**状态：** `I090 FAST-PATH SYSTEM CANDIDATE SELECTED / S110 RUNNING BACKGROUND / HELD-OUT BLOCKED`
**详细计划：** [PLAN.md](PLAN.md)
**执行跟踪：** [TRACKER.md](TRACKER.md)
**原始方案快照：** [snapshots/PLAN_v1_generator_only_2026-08-18.md](snapshots/PLAN_v1_generator_only_2026-08-18.md)

## 当前处在哪里

三个模块都有可运行代码，但新的最终方法还没有确定：

- Retriever 保持当前 frozen Hybrid RRF/default；
- Legacy Lean Selector 使用 NIAH 和 2Wiki 训练，但 harmful 监督只来自 NIAH synthetic counterfactual；
- F005 的 74/82、90.24% deletion precision 只证明 NIAH 风险代理上的高精度删除；
- Legacy Selector 在 L003 的 1,000 个 2Wiki 问题上删除 0 条，不能据此声称跨数据 harmful filtering；
- G230 的 GC/GM 提高回答率并减少空答案，但 citation precision/recall 下降约 6–8pp；
- G110 独立审计通过，主要断点判为 TRUE routing/attachment；G130 已完成一次共享 deterministic runtime 修复；
- G200 已完成数据预物化：NIAH train 515、新 NIAH model-val 307、2Wiki train 1,075、2Wiki model-val 136、unsupported update ratio 10.17%，但 TRUE/minimal-support/sample/length 审查仍未完成；
- G210 structural audit 通过，但 TRUE 过滤后 2Wiki model-val 只剩 76 个 answerable groups，低于最低门 100；
- G215 已把这个失败解释为“当前数据不能冻结，但路线有积极诊断信号，可回到 G200R/G210R 做受控数据修订”，不是 Generator/Selector 路线失败；
- G200R 已把 2Wiki target construction 改为 support-sentence-aligned，并重新物化为 2,665 个 train cases、443 个 validation cases，pre-audit gate 全部通过；
- G210R-v1 structural audit 发现 25 个 2Wiki answerable case 没保留 answer alias，因此 TRUE 没有启动，必须回到 G200R2 过滤这些 case；
- G200R2 已过滤 answer alias 不保留的 2Wiki support-sentence targets，剩余 2Wiki train/model-val 为 1,053/133，pre-audit gate 全部通过；
- G210R2 已完成 structural、TRUE 和 pre-sample finalize：过滤后 NIAH train/model-val 为 515/215，2Wiki train/model-val 为 828/106，unsupported ratio 为 11.3068%，split overlap 为 0，自动数据门通过；但固定样本和 length/truncation 审查仍未完成，因此还不能训练；
- G212 长度审计发现 11,348 个 examples 中有 4 个超过 `max_length=2304`，全部来自同一个 2Wiki train group；fixed sample 已准备 100 条但未进入样本判定，因此 G212 失败，不能训练；
- G214 已成组排除这个唯一超长 2Wiki train group 及其 unsupported counterpart；修订后 2Wiki train/model-val 为 827/106，unsupported ratio 为 11.3033%，split overlap 仍为 0；
- G212R 对 G214 bundle 重跑长度审计后通过：11,342 个 examples 中 0 个超过 2,304；
- G212M 已完成固定 100 条 sample review/adjudication：92 条通过、8 条失败、0 条不确定；失败集中在 2Wiki 目标句自洽性和 NIAH 新 model-val 的少数 QA2D 错配，因此不能冻结数据或启动训练；
- G216 已执行受控修复：排除 8 条失败样本对应的 11 个 case；修复后 NIAH train/model-val 为 515/213，2Wiki train/model-val 为 824/103，unsupported ratio 为 11.2929%，split overlap 为 0，预冻结数据门仍通过；
- G212R2 对 G216 bundle 重跑长度审计后通过：11,295 个 examples 中 0 个超过 2,304，最大长度为 2,120；新的固定 100 条 sample 已生成但仍未判定；
- G212M2 对新固定 100 条 sample 完成判定：92 条通过、8 条失败、0 条不确定；这仍是积极信号，但失败模式再次集中在 2Wiki target self-containment 和少数 NIAH QA2D 标题截断，因此不能靠继续窄删解锁训练；
- G217 已按用户要求修订判定措辞：后续统一使用中性的 sample review/adjudication 或固定样本判定，不强调执行主体；积极信号导向受控修复，重复缺陷才阻止直接训练；
- G218 已完成系统性 target 修复：2Wiki target 改为 title/subject anchored support sentence，NIAH 只过滤 1 条可检测 QA2D 标题截断；structural 2,703/2,703 通过，TRUE 2,074/2,078 通过；finalize 后 NIAH train/model-val 为 515/212，2Wiki train/model-val 为 820/103，unsupported ratio 为 11.3173%，split overlap 为 0；
- G212R3 对 G218 bundle 重跑长度审计后通过：11,268 个 examples 中 0 个超过 2,304，最大长度为 2,120；新的固定 100 条 sample 已生成但仍未判定；
- G212M3 对 G212R3 固定 100 条 sample 完成判定：91 条通过、9 条失败、0 条不确定；unsupported 层 20/20 通过，失败集中在 2Wiki answerable target 自洽性和 NIAH QA2D 句子构造；
- G219 尝试系统修复这些失败，但 finalize 后 2Wiki model-val 只有 94，低于当前最低 100，因此 G219 失败且不能解锁训练；
- G220 改用保守隔离：不再改写 target，只隔离 G212M3 固定样本失败的 9 条 case 以及 1 条对应 unsupported counterpart；修订后 NIAH train/model-val 为 512/211，2Wiki train/model-val 为 819/99，unsupported ratio 为 11.3432%，split overlap 为 0，达到 pre-sample pass；
- G212R4 对 G220 bundle 重跑长度审计后通过：11,211 个 examples 中 0 个超过 2,304，最大长度为 2,120；新的固定 100 条 sample 已生成但仍未判定；
- G212M4 对 G212R4 固定 100 条 sample 完成判定：90 条通过、10 条失败、0 条不确定；unsupported 层 20/20 通过，失败集中在 2Wiki answerable 关系自洽和 NIAH QA2D 语义改写；
- G221 对 G212M4 失败样本做保守隔离：共隔离 12 个 case；修订后 NIAH train/model-val 为 511/207，2Wiki train/model-val 为 817/96，unsupported ratio 为 11.3461%，split overlap 为 0，达到 pre-sample pass；
- G212R5 对 G221 bundle 重跑长度审计后通过：11,148 个 examples 中 0 个超过 2,304，最大长度为 2,120；新的固定 100 条 sample 已生成但仍未判定；
- G212M5 对 G212R5 固定 100 条 sample 完成判定：97 条通过、3 条失败、0 条不确定；unsupported、NIAH train 和 NIAH model-val 都是 20/20，通过失败只剩 2Wiki answerable 关系自洽；
- G222 已把硬门解释修订为三档：100/100 是 clean freeze，97/100 且失败局部集中是 controlled continuation，只允许继续残余修复/隔离，不允许直接训练；
- G223 已隔离剩余 3 条失败及必要 counterpart，得到 controlled-continuation candidate：train/validation 为 2,370/302，2Wiki model-val 为 95，split overlap 为 0，unsupported ratio 为 11.3392%；下一步可进入 G300 limited draft entry，但不是 clean freeze；
- G300 已完成 draft LoRA 训练入口实现和 smoke：长度审计覆盖 train 2,370 groups / 9,207 examples、validation 302 groups / 1,924 examples，0 个超过 2,304；GR-F seed13 的 1-group smoke 成功保存并 fresh-base reload adapter，显存峰值约 9.12GB；这是可执行性通过，不是 Generator 效果结论；
- G310 已完成正式 seed13 screen：GR-F/GR-C formal adapters 均训练完成并 fresh-base reload 通过；2968 条正式生成和 2968 条评分行数一致；MiniCheck post-generation screen 选择 `GR-C`，GR-F 因 answer regression >2pp 被排除；这是配方筛选通过，不是最终 Generator qualification；
- G320 已冻结唯一 Generator 训练配方 `GR-C`，G330 只能用该配方训练 seeds 13/42/73；这仍不是教师 Generator 冻结，也不允许启动 utility labels 或 held-out；
- G330 已完成 `GR-C` seeds 13/42/73 三 seed fit，三份 manifest 均为 COMPLETE，fresh-base reload 均 PASS，未读 held-out/sealed，未生成 utility labels；
- G331 已按用户要求修订后续判定解释：未达某个单一数字门不再被写成“路线无意义”；强冻结/强结论条件与继续推进条件分开记录，积极信号可以进入受控下一步，但不能被包装成已经通过强结论；
- G400 implementation/smoke 已完成：新增 locked NIAH qualification runner，服务器和本地相邻测试均 11/11 通过，三 seed 各 1 题真实 smoke 均 COMPLETE、0 runtime error、0 missing trace，score smoke 能合并固定 G0、三 seed GR-C、NIAH dev gold 和 MiniCheck；这只是接线通过，不是正式 G400 效果结论；
- G400 formal locked NIAH qualification 已完成：三 seed 各 2873 条生成全部 COMPLETE，0 runtime error、0 missing trace，评分状态为 `G400_NIAH_RESPONSIBILITY_PASS`；NIAH 各 full/stress family 的 `correct+cited` 相对固定 G0 为 +5.35pp 到 +11.93pp，unsupported ungrounded assertion 从 30.08% 降到接近 0；
- G410 implementation/smoke 已完成：新增 2Wiki cross-data runner，生成阶段只读 no-answer task packet；prepare smoke 从 G223 生成 475 个 2Wiki tasks 和独立 references，三 seed 各 1 题真实 smoke 均 COMPLETE，score smoke 只用显式 `--allow-subset` 裁剪到 1 题并通过；
- G410 formal locked 2Wiki cross-data qualification 已完成：seeds 13/42/73 均生成 475 rows，generation runtime 未读取 gold/reference，sealed/held-out=false，utility labels 未启动；正式评分状态为 `G410_CROSS_DATA_RESPONSIBILITY_PASS`，all_2wiki family delta 为 correct+cited +39.65pp、answer +6.95pp、coverage +2.53pp、citation precision +31.37pp、citation recall +35.47pp；
- G420 combined Generator gate/statistics 已完成：只读 G400/G410 正式评分行，不重新生成、不训练、不读 held-out；状态为 `G420_NEW_GENERATOR_QUALIFIED_G430_READY`，teacher recommendation 为 `FREEZE_NEW_GRC_IN_G430`，strong claim 为 `ESTABLISHED`；
- G430 teacher Generator freeze 已完成：按 G420 建议冻结新的 `GR-C` 三 seed 教师家族为 GQ，三颗 adapter 的服务器实体 hash 已核对一致，utility labels 尚未启动；
- S100 implementation/smoke 已完成：新增 prepare/run/score runner；本地和服务器相关测试均为 8 passed；服务器 prepare smoke 生成 2 questions / 22 tasks，三 seed 各 2 条真实生成 smoke 均 COMPLETE 且 errors=0、trace_missing=0；score smoke 能写报告/清单，但因子集不完整不形成正式标签；
- S100 formal utility pilot 已完成：固定 50 NIAH train + 50 2Wiki train 问题，每题 10 条证据，三 seed 各跑 1,100 个 full/leave-one-out task；三 seed 全部 COMPLETE、errors=0、trace_missing=0，生成阶段未读取 reference/support provenance/held-out。评分后得到 1,000 条 evidence-level utility labels：MUST_KEEP 90、SAFE_DROP 815、NEUTRAL 32、UNCERTAIN 63，stable label rate=0.937，utility label rate=0.905，覆盖 NIAH 和 2Wiki；状态为 `S100_PILOT_COMPLETE`，建议 `S110_READY`。
- S090 early full-flow triage 已完成：在不新增训练、不读 held-out 的前提下复用 G400/G410/L003 结果。218 个 NIAH matched 问题上 `TopK+GQ` 相对 `TopK+G0` 的 correct+cited 为 +9.33pp，而 `Legacy SL+GQ` 相对 `TopK+GQ` 为 -0.46pp；G410 2Wiki `TopK+GQ` 仍有 +39.65pp correct+cited。结论是当前正信号主要来自 GQ，旧 Selector 没有同 GQ 额外端到端收益。
- 因此截止日期优先路线改为：S110 继续后台运行，但不再阻塞；先以 `TopK Selector + GQ` 作为 fast-path 系统候选推进完整开发评估和 baseline 对比。Utility Selector 若后续赶上并证明相对 `TopK+GQ` 有正作用，再作为增强路线加入。
- I090 fast-path system decision 已完成：当前主系统候选冻结为 `SystemF-fast-S0-GQ-2026-08-21`，即 frozen Retriever + TopK keep-all Selector + frozen GQ。它可以支持 main result/baseline packaging；但不能声称 Selector 学习成功，也不能运行 held-out。

## 修订后的核心方法

Selector 与 Generator 有联系，但不同时自由修改：

```text
1. 修复 Generator
2. 冻结一个教师 Generator GQ
3. 用 GQ 测量删除每条证据对答案和引用的影响
4. 训练 Utility 候选 SU，并冻结职责合格 Selector SQ
5. 冻结 Retriever + SQ + GQ 做完整系统开发验证
6. 只有 SystemF 冻结后，才一次性运行最终三个数据集
```

如果新 Generator 没有通过模块职责门，不会因此完全阻断 Selector：可靠基线 G0 可以作为冻结教师，但不能声称 Generator 修复成功。

主目标是 `SQ=SU`。只有 SU 失败、而旧 Legacy Selector 在同一 GQ 下确有正的端到端作用时，才允许 `SQ=SL` 作为系统 fallback；这时不能声称 Utility Selector 方法成功。

## 三种不同的“通过”

1. **模块职责通过：** 组件足够稳定，可以进入下一阶段；不要求每个 slice 的 CI 都显著。
2. **强统计结论通过：** 预注册主要指标的 CI 支持“优于基线”。
3. **完整系统通过：** Utility Selector 对同一个 GQ 有净作用，完整系统主要联合结果正向且没有实际不可接受的答案、引用或安全退化。

CI 跨 0 不再自动淘汰职责合格组件，但也不能写成统计显著提升。

## 修订后的继续原则

门槛的作用是保护结论边界，不是把研究路线一票否决。后续每个阶段同时记录四类判定：

1. **技术有效：** 运行完整、hash 可复算、没有读取禁止数据、没有 gold/reference runtime 泄漏。
2. **职责可用：** 主要职责指标相对固定基线没有实际不可接受退化，并且至少有可解释的正向信号。
3. **强结论成立：** 预注册统计门支持更强表述。
4. **受控继续：** 未达到强冻结或强结论，但正向信号明确、失败局部可解释、没有触发安全或泄漏 tripwire，因此可以继续做下一步受控验证。

例如早期 2Wiki model-val 过滤后只有 76 个 answerable groups，含义是“这个数据版本不能 clean freeze”，不是“Generator/Selector 路线失败”。同理，后续若出现 97/100、方向性提升但 CI 不够强、或某个 slice 没过强结论门，只能限制结论写法，不能自动中断有价值的受控推进。

## 数据角色

- NIAH train：抗 harmful/benign、位置、unsupported、utility；
- 2Wiki train：多证据链和冻结 GQ 下的 evidence utility；
- NIAH/2Wiki dev：锁定资格，不作最终结论；
- ASQA/QAMPARI：已揭示 citation catastrophe guard；
- sealed600：退休，只读历史；
- HotpotQA、MuSiQue-Full、RGB：SystemF 冻结后一次性最终测试，不参与方法选择。

## 当前仍不会执行

- 不把 G300 smoke adapter 当作正式候选效果；
- 不跳过 G310/G320 直接进入三 seed 正式训练；
- 不生成 utility labels；
- 不修改 Retriever、Legacy Selector 或历史结果；
- 不使用 sealed600；
- 不读取/评分 system held-out；
- 不在原 G210 failure 数据上启动任何 Generator 训练。

## 下一步

下一步分成前台和后台两条：

```text
前台主线：
fast-path full-system development
-> keep frozen Retriever and GQ unchanged
-> use S0 TopK as the deadline-safe Selector
-> run/aggregate baseline comparisons for TopK+G0 vs TopK+GQ
-> current system candidate is SystemF-fast-S0-GQ-2026-08-21
-> do not claim Selector contribution
-> do not run held-out

后台增强：
S110 utility materialization
-> continue only as optional Utility Selector evidence
-> S200/SU enters only if S110 completes and later same-GQ tests are positive
-> do not let it block the main deadline path
```

G218/G212R3/G212M3/G219/G220/G212R4/G212M4/G221/G212R5/G212M5/G222/G223/G300/G310/G320/G330/G400/G410/G420/G430/S100-smoke/S100-formal/S090/I090 说明当前路线仍有积极信号：长度、split、NIAH 和 unsupported ratio 都稳定，unsupported 固定样本层已经稳定通过，最近一轮固定样本达到 97/100，已知残余失败已隔离，训练入口、正式筛选、locked NIAH qualification、2Wiki cross-data qualification、combined Generator gate、GQ freeze、S100 工具链接线、formal utility pilot、early full-flow triage 和 fast-path system decision 均跑通；GR-C 在 2Wiki citation grounding 上有明显正信号，并大幅降低 unsupported safety 层乱答，三 seed adapter 已完成保存和 reload，G400 NIAH full/stress family 全部显示 `correct+cited` 正增益，G410 all_2wiki `correct+cited` 为 +39.65pp，G430 已冻结新 GR-C 为 GQ，S100 formal 产生 905 条可用 utility labels 且建议 `S110_READY`，I090 支持 fast-path `TopK+GQ` 先进入 main result/baseline packaging。但这仍不是 Utility Selector 或 held-out 结论；不得降低 TRUE 阈值、改最终测试集边界、读取 held-out，或按已看到的结果临时改规则凑通过。
