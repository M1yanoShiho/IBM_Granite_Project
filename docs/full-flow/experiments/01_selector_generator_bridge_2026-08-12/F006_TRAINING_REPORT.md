# F006 小规模鲁棒训练报告

**日期：** 2026-08-13
**当前状态：** `TRAINING COMPLETE / FORMAL DEVELOPMENT EVALUATION RUNNING`

## 零基础结论

这一步没有重新训练整个 Granite，也没有新建第四个模块。我们只给现有 Generator 的“关键事实笔记”步骤加了一个很小的可训练补丁（LoRA）：

```text
问题 + Selector 保留下来的证据
        ↓
LoRA 只练习提取关键事实笔记
        ↓
原来的 Granite 根据笔记起草答案
        ↓
原来的 TRUE 核验并保留有证据支持的内容
```

两组训练唯一的重要区别是：

- **Clean-LoRA：** 只看干净、直接相关的证据；
- **Mixed-LoRA：** 同时看干净证据，以及“正确证据仍在但周围混有无关内容”的证据。

这样才能公平回答：如果最终更好，究竟只是普通微调的效果，还是“学习在噪声中抓住正确事实”真的有效。

## 数据隔离

- 训练数据只来自 NIAH train；
- 发现 train 与当前 development 问题有 202 个 query ID 重叠后，已全部排除；
- development 只作为 ID 排除集合，不读取它的答案来训练；
- sealed / held-out 数据完全未读取；
- 最终得到 1,443 个合格训练池问题，冻结 1,000 个训练问题和 100 个训练内部验证问题；
- Clean 与 Mixed 使用相同的 1,000 个问题、相同的目标答案、相同的样本数和更新步数。

数据清单见 [`artifacts/F006/data/manifest.json`](artifacts/F006/data/manifest.json)。

## 训练规模

| 项目 | Clean-LoRA | Mixed-LoRA |
|---|---:|---:|
| 训练问题 | 1,000 | 1,000 |
| 训练样本 | 2,000 | 2,000 |
| 训练轮数 | 1 | 1 |
| 参数更新步数 | 250 | 250 |
| LoRA rank / alpha | 8 / 16 | 8 / 16 |
| 实际训练参数 | 15,564,800 | 15,564,800 |
| 占模型参数比例 | 0.4553% | 0.4553% |
| 耗时 | 770 秒 | 1,559 秒 |

因此，这确实是“小规模 LoRA”，不是全量 Granite 训练。

## 训练内部结果

损失越低，表示模型在训练规定的“关键事实提取”任务上出错越少。

| 验证输入 | Clean-LoRA | Mixed-LoRA | 当前观察 |
|---|---:|---:|---|
| 干净证据 | 0.0825 | **0.0775** | Mixed 没有牺牲干净输入 |
| 混合/带干扰证据 | 0.1527 | **0.1134** | Mixed 更能适应无关内容 |

这是一条正向的训练信号，但不是最终系统结论。原因是训练损失只检查“能不能提取关键事实”，还没有检查经过起草和 TRUE 核验后的最终答案是否真的更正确。

## 正式比较及通过条件

在 F004 已冻结的 109 个 Selector 改变问题上，同一次作业比较：

| 路线 | 含义 |
|---|---|
| TopK_frozen | 不经过 Selector 的原始 TopK 基准 |
| Base_notes_frozen | F004 的未训练关键事实笔记 |
| Clean_LoRA | 只用 clean 数据训练的关键事实提取器 |
| Mixed_LoRA | 用 clean + 干扰上下文训练的关键事实提取器 |

只有当 `Mixed_LoRA` 的最终答案点估计同时高于 Base、Clean 和 TopK，才允许进入 F005 独立最终确认。这个规则在查看正式结果前已经写入代码和测试，结果出来后不临时改变。

## 可复现记录

- Clean 权重 SHA-256：`cb41b40a4b7a7d3c46b94effc4cc77f883ab92491c890d783246666bf05babf4`
- Mixed 权重 SHA-256：`87646ed957c36e333cf2ab1997b077e2694e3dfdcfccecebcd9ae02dd52d8bcd`
- 权重保存在服务器实验目录，不纳入 Git；Git 只保存配置、训练清单、代码和哈希。
- Clean 训练清单：[`artifacts/F006/clean/training_manifest.json`](artifacts/F006/clean/training_manifest.json)
- Mixed 训练清单：[`artifacts/F006/mixed/training_manifest.json`](artifacts/F006/mixed/training_manifest.json)
