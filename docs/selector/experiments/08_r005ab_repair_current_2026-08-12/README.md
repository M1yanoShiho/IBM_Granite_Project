# 08 — 当前 R005A/R005B recovery v2

**时间：** 2026-08-12 起

**状态：** `DRAFT / WAITING A000 EXPLICIT APPROVAL / NOT IMPLEMENTED / NOT RUN`

## 这次到底改了方法，还是只改了执行

必须指定比较对象：

1. **amendment v1 草案 → document revision v2：只改执行与审计协议。** 四项修正是 terminal 顺序、五份输入清单的精确路径/schema、B 对 A→B 授权 SHA 的绑定，以及 A001 旧回归的禁读边界。V0/V1/V2、pair loss、样本角色、门槛和预算都没有改变。
2. **正式失败的旧 R005 → 整个 R005A/R005B recovery：既有方法层变化，也有实验设计变化。** 方法层包括恢复预训练 NLI pooler/classifier 路径和 V2 pairwise loss；实验设计层包括 fresh component-disjoint 数据、fit/screen/confirm 隔离、首个最简通过者和两 seed one-shot confirm。

所以，“v2 修订只改执行层”是对的；“当前 recovery 相对旧 R005 也完全没改方法”是不对的。

## 当前许可边界

- amendment v1 从未获批、从未实现、从未运行；
- v2 已完成设计复审，但审计 PASS 不等于模型一定通过；
- A001、A002、formal fit、A-screen、B-confirm 全部 `NOT RUN`；
- 当前唯一合法下一步仍是用户明确批准 A000；
- TopK10 保持唯一生产默认。

## 文件入口

- [计划导航](PLAN.md)
- [状态记录](TRACKER.md)
- [证据索引](EVIDENCE_INDEX.md)
- [来源清单](SOURCE_MANIFEST.json)
