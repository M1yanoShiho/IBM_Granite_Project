# F004 独立逐句引用评分

**裁判：** MiniCheck；系统内部 TRUE 不参与评分。

| 组别 | Answered | Citation precision | Precision（实际有引用） | Citation recall |
|---|---:|---:|---:|---:|
| G0_current | 92 | 80.43% | 93.67% | 80.43% |
| G1_notes | 92 | 80.43% | 92.50% | 80.43% |
| G2_guided_notes | 92 | 80.43% | 92.50% | 80.43% |

三组都使用运行时保存的精确句子—引用对应。
ALCE precision 将回答中没有引用的句子计入；有引用样本 precision 另行报告。
