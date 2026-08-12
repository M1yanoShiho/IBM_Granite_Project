# Selector–Generator 跨阶段桥接跟踪表

**对应计划：** [PLAN.md](PLAN.md)
**当前状态：** `F000 PASS / F001 RUNNING`

| Run | 目的 | 比较/输出 | 优先级 | 状态 | 备注 |
|---|---|---|---|---|---|
| F000 | 冻结联合实验入口 | 数据、四臂、Granite/TRUE 配置、同作业协议 | MUST | PASS | 739 题；109 题改变；运行时不读 gold |
| F001 | 现有三模块联合 | A/B/C/D 四臂、总体与109题结果 | MUST | RUNNING | 1题 TRUE smoke 四臂零错误；完整739题已启动 |
| F002 | 联合结果归因 | 细节遗漏、空输出、无支持、多跳、真实缺证据分类 | MUST | TODO | 决定 F003/F004 的具体范围 |
| F003A | 传递但暂不使用跨阶段信号 | `SelectionGuidance` allowlist + 默认无信号回退 + gold 泄漏测试 | CONDITIONAL | TODO | 只承载 Selector 原生运行时信号 |
| F003B | 关键事实读取 | 问题需求、notes-only、guided-notes | CONDITIONAL | TODO | 细节遗漏/空答案占主导时才运行 |
| F003C | 运行时证据准备度 | `READY/PARTIAL/CONFLICTED/UNKNOWN` | CONDITIONAL | NOT AUTHORISED | 只由问题和 selected evidence 估计；多跳组合失败占主导时才运行 |
| F004 | 轻量 Generator 消融 | G0 当前、G1 notes-only、G2 guided notes | CONDITIONAL | TODO | 只有 G2>G1 才保留跨阶段主张 |
| F005 | 独立最终确认 | 冻结方法、未参与选择的数据、配对 CI | CONDITIONAL | TODO | `decision-dev` 不再是新方法盲测 |
| F006 | 小规模鲁棒训练 | base、clean-only、mixed-context LoRA | CONDITIONAL | NOT AUTHORISED | 仅当前两阶段证实需要训练时启动 |

## F002 结果路由

| 主要错误 | 后续 |
|---|---|
| 高级 Generator 已解决 | 跳过不必要改造，准备 F005 |
| 正确证据在但细节漏答 | F003A/F003B + G1/G2 关键事实笔记 |
| 高风险保留证据误导 | F003A/F003B + 风险感知核验/引用 |
| parser/splitter/JSON 失败 | 只做工程 fallback，再重跑 F001 |
| Selector 真正删掉必要事实 | 返回 Selector；不让 Generator 猜 |
| 多跳事实齐全但未组合 | 才授权 F003C：设计不看 gold 的 `EvidenceReadiness` |

## 更新规则

1. 未运行不得预填结果。
2. 每个完成阶段补充 commit、配置、输入和结果路径。
3. 第一阶段结果出来前，不把第二阶段候选写成最终方法。
4. gold chain、required/supporting IDs 和 reference answer 只能在生成后评测，不能进入 runtime payload。
5. 失败结果保留，不用新文件覆盖。
