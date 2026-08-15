# Generator-first、Selector-aligned 执行跟踪表

**对应计划：** [PLAN.md](PLAN.md)  
**当前状态：** `PLAN ONLY / NO RUN AUTHORISED`  
**更新规则：** 未实际执行不得填写 PASS；每个 COMPLETE 项必须附 commit、input hash、run manifest 和结果路径。

| Run | Milestone | 目的 | 主要输入 | 必须产物 | 优先级 | 状态 |
|---|---|---|---|---|---|---|
| A000 | 协议冻结 | 冻结数据、Granite-centered 模型栈、主比较、统计门和 gold 边界 | 旧 query/provenance 清单、新数据来源、模型 snapshots | `A000_PROTOCOL.md`、data/model/power manifest | MUST | TODO |
| A001 | Trace schema | 记录 draft→claims→faithfulness→routing 全链路 | 当前 Generator 代码 | schema、单测、逐题 trace | MUST | TODO |
| A002 | 基线/方差 | 复现基线并测同输入生成波动 | 已揭示 dev | baseline report、repeat report | MUST | TODO |
| B100 | Context matrix | 测 support-only、noise、position、Selected | changed + matched unchanged dev | cases JSONL、diagnostic report | MUST | TODO |
| B110 | Bottleneck decision | 将问题路由到 Retriever/Generator/Selector | B100 产物 | decision record | MUST | TODO |
| G200 | Draft data | 构造可溯源的 evidence→draft targets | NIAH train、provenance groups | train/val cases、manifest | CONDITIONAL | BLOCKED BY B110 |
| G210 | Split fallback | 只在 splitter 归因为主时增加降级路径 | A001/B100 trace | tests、implementation report | CONDITIONAL | BLOCKED BY B110 |
| G220-S | Training smoke | 验证 draft adapter、loss mask、长度和显存 | 小样本 train cases | smoke manifest | CONDITIONAL | BLOCKED BY G200 |
| G220-GC | Clean draft LoRA | 普通 draft 微调控制组 | clean/support-only contexts | 3-seed checkpoints/manifests | CONDITIONAL | BLOCKED BY SMOKE |
| G220-GM | Mixed draft LoRA | 训练 TopK/Selected/noise/position 鲁棒性 | matched mixed contexts | 3-seed checkpoints/manifests | CONDITIONAL | BLOCKED BY SMOKE |
| G230 | Generator dev gate | 选择或拒绝 G* | G0/GN/GC/GM | paired report、citation report | MUST AFTER TRAIN | BLOCKED BY G220 |
| S300 | Freeze G* | 冻结 utility-label 教师 | G230 通过版本 | model/runtime manifest | CONDITIONAL | BLOCKED BY G230 |
| S310-P | Utility pilot | 验证 leave-one-out 标签产出与稳定性 | 100 train queries、G* | utility cases、yield report | CONDITIONAL | BLOCKED BY S300 |
| S310-F | Utility full | 构造 Selector utility 训练集 | NIAH train、G* | utility dataset、manifest | CONDITIONAL | BLOCKED BY PILOT |
| S320 | Utility Selector | 训练 legacy safety + utility 目标 | S310-F dataset | checkpoints、training report | CONDITIONAL | BLOCKED BY S310-F |
| S330 | Selector dev gate | 验证 SU 是否提高同一 G* 的答案 | TopK/SL/SU + G* | answer/evidence/citation report | MUST AFTER TRAIN | BLOCKED BY S320 |
| I400 | Retriever freeze | 用真实 Hybrid RRF 生成并冻结候选 | dev/new-held-out corpus/index | candidate pools、manifest | MUST | BLOCKED BY A000 |
| I410 | 五臂联合 dev | 估计 Generator、Selector 和交互作用 | A/B/C/D/H | paired report、trace、CI | MUST | BLOCKED BY G230/S330/I400 |
| I500 | Fresh held-out | 一次最终确认 | 冻结系统、新 held-out | final report、all manifests | MUST | BLOCKED BY I410 |

## 每个 Run 的完成清单

- [ ] 输入路径、角色和 SHA256 已记录
- [ ] runtime gold boundary 已审计
- [ ] 模型、adapter、prompt、decode 和软件环境已记录
- [ ] 主 Generator 仍为冻结的 Granite 4.1-3B 基座，非 Granite 模型只承担 Selector/Verifier/Evaluator 的既定角色
- [ ] 错误数和缺失输出已报告
- [ ] 完整分布为主结果，changed subset 仅作诊断
- [ ] 逐题产物和聚合报告同时保存
- [ ] 配对置信区间与 W→R/R→W 已报告
- [ ] 未读取或调试新 held-out 逐题结果
- [ ] 结果为负时按停止规则归档，没有继续在同一数据上选方法

## 当前禁止项

- sealed600 再利用
- Selector threshold/cap 扫描
- 新方法在 held-out 上调参
- runtime 多次 Generator 调用或读取 reference answer
- 同时改动 Generator、Selector 和 Retriever 后只做单一前后比较
- 未完成 A001/B100 就启动新 LoRA
