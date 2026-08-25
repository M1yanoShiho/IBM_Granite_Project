# F002：Selector 改动题的结果归因

**状态：** `COMPLETE`

范围：F001 中 Selector 真正改变证据的 109 题。
Gold 只在生成完成后用于体检，没有传给 Generator。

## 基础 Generator：B 对 A

### 回答变化

| 类型 | 数量 |
|---|---:|
| right_stayed_right | 59 |
| right_to_wrong | 6 |
| wrong_stayed_wrong | 41 |
| wrong_to_right | 3 |

### 首要体检信号

| 类型 | 数量 |
|---|---:|
| answer_extraction_failure_with_evidence_present | 15 |
| correct_preserved | 59 |
| empty_output | 11 |
| generator_context_sensitivity | 6 |
| multi_document_composition_candidate | 11 |
| partial_evidence_or_other | 4 |
| selector_helped_generation | 3 |

## 高级 Generator：D 对 C

### 回答变化

| 类型 | 数量 |
|---|---:|
| right_stayed_right | 57 |
| right_to_wrong | 7 |
| wrong_stayed_wrong | 41 |
| wrong_to_right | 4 |

### 首要体检信号

| 类型 | 数量 |
|---|---:|
| answer_extraction_failure_with_evidence_present | 16 |
| correct_preserved | 57 |
| empty_output | 15 |
| generator_context_sensitivity | 7 |
| multi_document_composition_candidate | 9 |
| partial_evidence_or_other | 1 |
| selector_helped_generation | 4 |

这些分类用于决定下一步查哪一类案例；其中 `candidate`/`failure_with_evidence_present`
表示应人工查看的候选原因，不等于已经证明了因果。
