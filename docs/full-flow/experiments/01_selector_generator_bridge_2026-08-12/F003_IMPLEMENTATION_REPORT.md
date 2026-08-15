# F003A/F003B 实施报告：让 Generator 先读关键事实

**状态：** `COMPLETE / TESTED / F004 RUNNING`

## 1. 零基础说明

F002 告诉我们：Selector 删除一条干扰证据以后，正确证据通常还在，但 Generator 有时反而漏掉日期、地点、人名等具体答案，甚至输出空答案。

因此 F003 没有重新做一个模块，也没有重新训练 Retriever 或 Selector。它只在现有三模块之间增加一张“小便条”：

```text
问题 + Selector 保留的证据
          ↓
先找问题要求的关键事实（日期、地点、人名、数字、名称）
          ↓
再交给原有 Verify-and-annotate Generator 生成、拆句和核验
```

这张便条的目的不是告诉模型标准答案，而是减少“答案明明在证据中，模型却没有读出来”的情况。

## 2. F003A：Selector 可以传什么

Selector 现在可以额外输出一个可选 `SelectionGuidance`：

- 本题是否真的发生过删除；
- 删除了几条；
- 每条**保留证据**的编号、原始排序、protect/harm 运行时分数和保留动作。

明确禁止传递：

- 标准答案；
- official supporting/required evidence 身份；
- gold chain 或 `chain_loss`；
- 被删除证据的文本；
- 事后才知道的“这条是否真正正确”。

没有 `SelectionGuidance` 时，原有 Generator 仍按原路径运行，不改变已有接口行为。

## 3. F003B：Generator 新增什么

只对可以可靠从问题文字判断的五类需求生成关键事实笔记：

| 问题需求 | 例子 |
|---|---|
| `DATE_OR_YEAR` | 什么时候、哪一年、完整日期 |
| `LOCATION` | 在哪里、哪个城市或国家 |
| `PERSON` | 谁、哪位人物 |
| `NUMBER` | 数量、百分比、金额、比分 |
| `NAME_OR_TITLE` | 作品名、组织名、获胜者名称 |

开放式描述题不强行套用这些槽位；笔记解析失败或没有可用笔记时，自动回退到原有 draft prompt。

## 4. F004 中的三组

| 组 | 输入和方法 | 回答的问题 |
|---|---|---|
| G0 | F001 已冻结的 `Selector + Verify-and-annotate` 输出 | 现有系统做到什么程度 |
| G1 | 保留证据 + 普通关键事实笔记 | “多认真读一步”本身是否有效 |
| G2 | G1 + Selector 对保留证据的 protect/harm 信号 | 跨阶段信号是否在普通笔记之外还有额外价值 |

只有 `G2 > G1` 才能把改进称为 Selector→Generator 跨阶段机制；若 G1 更好或两者相同，就采用更简单的 G1，不夸大创新点。

## 5. 实施与验证记录

| 内容 | 记录 |
|---|---|
| F003A runtime-safe sidecar | commit `6d1b831` |
| F003B notes-first Generator | commit `9cb742f` |
| F004 runner 与 1 题服务器 smoke | commit `6d8f982` |
| F004 独立引用评分 | commit `36413ed` |
| F004 失败转移诊断 | commit `de57b0e` |
| F003A 相关测试 | 89 个通过 |
| F003B 相关测试 | 64 个通过 |
| F004 runner 回归测试 | 53 个通过 |
| F004 smoke | 1/1 完成；G1/G2 均产生笔记；三组零错误；运行时未加载 gold |

测试数量来自各阶段执行时的相关测试集合，集合之间可能有重叠，因此不相加宣称总测试数。

## 6. 未实施的功能

- `F003C EvidenceReadiness`：F002 中多跳组合不是主导错误，未授权、未实现；
- `F006 LoRA`：必须先看到提示级方法有正向但不稳定的迹象，目前未授权；
- 恢复被删证据：Generator 无此权限，也不会看到被删文本。

## 7. 当前边界

F003 已经证明功能可以运行且没有明显数据泄漏；它**尚未证明效果变好**。效果结论必须等待 F004 的 109 题比较以及独立 MiniCheck 引用评分。
