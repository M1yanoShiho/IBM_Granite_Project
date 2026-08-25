# G230 Generator 开发门结果

**日期：** 2026-08-17
**状态：** `COMPLETE / NO CANDIDATE`
**冻结协议：** [G230_PROTOCOL.md](G230_PROTOCOL.md)
**服务器产物：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G230-v1`

## 1. 执行边界

G230 按冻结协议完成 G0、GN、GC 和 GM 比较：

- 主结果为完整 NIAH decision-dev 739 题的原 TopK10；
- 机制诊断为 B100 冻结的 218 题，并分别运行 Selected、support-only、support+benign、support+harmful 和 support-last；
- GN 运行一次；GC/GM 使用 seeds 13/42/73；
- 四个正式生成 run 各覆盖 1,829 个 task，运行错误和 trace 缺失均为 0；
- 生成命令未加载 gold；gold 只由生成后的 answer scorer 读取；
- Granite 4.1-3B 仍为主 Generator，claim splitter 使用 adapters disabled 的 frozen Granite base，TRUE 保持冻结；
- MiniCheck-Flan-T5-Large 仅在生成完成后独立评估引用，TRUE 不评价自身输出。

四份 `generations.jsonl` 的实际 SHA256 均与各自 runtime manifest 一致。

## 2. 完整 739 题主结果

| 配置 | Answer match | 相对 G0 | 95% CI | Coverage | Draft empty | Zero claims | Final empty |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 64.01% | - | - | 87.96% | 2.03% | 4.87% | 12.04% |
| GN | 63.73% | -0.27pp | - | 84.71% | 6.77% | 9.07% | 15.29% |
| GC13 | 66.04% | +2.03pp | [-1.35,+5.61] | 93.91% | 0.00% | 0.95% | 6.09% |
| GC42 | 65.90% | +1.89pp | [-1.68,+5.47] | 94.45% | 0.00% | 0.81% | 5.55% |
| GC73 | 64.41% | +0.41pp | [-3.07,+3.92] | 93.91% | 0.00% | 0.81% | 6.09% |
| GM13 | 69.42% | +5.41pp | [+1.83,+9.07] | 95.81% | 0.00% | 0.95% | 4.19% |
| GM42 | 70.37% | +6.36pp | [+2.68,+10.17] | 96.21% | 0.00% | 0.81% | 3.79% |
| GM73 | 67.52% | +3.52pp | [-0.27,+7.27] | 96.08% | 0.00% | 0.68% | 3.92% |

GC 和 GM 的三个 seed 均满足冻结的 answer 点估计方向、coverage 非劣、主要失败率下降和 support-only 门。GC 的 answer CI 均跨 0；GM13/GM42 的 answer CI 高于 0，GM73 跨 0。冻结门只要求三个 seed 的点估计方向一致，因此两类均通过 citation 之前的 family gate。

## 3. Support-only 与 mixed-context 诊断

Support-only 上，G0 为 68.81% answer / 9.63% final empty。六个新配置均同时提高 answer 并降低 final empty：

| 配置 | Answer match | Final empty |
|---|---:|---:|
| GC13 / GC42 / GC73 | 80.73% / 77.98% / 76.61% | 4.13% / 4.59% / 5.50% |
| GM13 / GM42 / GM73 | 76.15% / 73.39% / 74.77% | 3.67% / 5.05% / 3.67% |

GM 相对同 seed GC 的 stress answer 差值为：

| Seed | Support+benign | Support+harmful | Support-last |
|---|---:|---:|---:|
| 13 | -0.92pp | +18.81pp | 0.00pp |
| 42 | +5.05pp | +17.89pp | +8.26pp |
| 73 | -2.75pp | +12.39pp | +1.38pp |

GM 在 support+harmful 上三个 seed 方向一致，但 support+benign 的 seeds 13/73 未超过 GC，support-last 的 seed 13 也未严格超过 GC。因此预注册的 mixed-context robustness 门未通过，不能选择 GM；按冻结 fallback 规则，citation 前候选为 GC。

## 4. 独立引用门

MiniCheck 在完整 TopK 上的聚合结果为：

| 配置 | Answered | Citation precision | Citation recall |
|---|---:|---:|---:|
| G0 | 650 | 75.17% | 74.28% |
| GN | 626 | 73.28% | 72.38% |
| GC13 | 694 | 65.92% | 65.92% |
| GC42 | 698 | 66.33% | 66.33% |
| GC73 | 694 | 66.07% | 66.07% |
| GM13 | 708 | 67.73% | 67.68% |
| GM42 | 711 | 66.39% | 66.39% |
| GM73 | 710 | 65.70% | 65.70% |

引用门使用双方共同有答案题的配对 component-cluster bootstrap，而不是直接比较上述不同 answered 集合的聚合均值。GC 相对 G0 的门控结果为：

| Seed | Precision delta (95% CI) | Recall delta (95% CI) |
|---|---:|---:|
| 13 | -7.10pp [-10.57,-3.69] | -6.28pp [-9.73,-2.94] |
| 42 | -6.96pp [-10.45,-3.50] | -6.19pp [-9.70,-2.71] |
| 73 | -7.71pp [-11.35,-4.19] | -6.97pp [-10.63,-3.48] |

预注册非劣边界为 CI 下界不低于 -2pp。GC 三个 seed 的 precision 和 recall 均越过退化边界，因此 GC citation gate 失败。GM 的六项 citation 检查也全部失败。

## 5. 冻结判定

```text
GC family gate:                 PASS
GM family gate:                 PASS
GM mixed-context robustness:    FAIL
Candidate before citation:      GC
GC citation non-inferiority:    FAIL
Final selected Generator:       none
G230 final gate:                FAIL
```

因此 G230 的正式结论为 `NO CANDIDATE`。GC/GM 的 draft LoRA 明显减少空答案并提高 answer 点估计，但该收益伴随超过冻结边界的独立引用质量下降。不能冻结 GC 或 GM 作为 `G*`，S300 及其后的 Selector utility-label 路线未解锁。本结论只用于已揭示 dev 上的方法选择，不是 held-out 最终结论。

## 6. 归档

- `artifacts/G230/{gn,seed13,seed42,seed73}/`：run spec、runtime manifest 和 gzip 压缩的完整逐 task generations；
- `artifacts/G230/score/`：答案聚合报告与 1,829 条逐 task score；
- `artifacts/G230/citation/`：MiniCheck 聚合报告与 1,829 条逐 task citation score；
- `artifacts/G230/jobs/`：两条 GPU 队列日志和自动后处理日志；
- `artifacts/G230/ARCHIVE_MANIFEST.json`：压缩包、评分文件及原始 generations 的 SHA256。

服务器 runtime 保留未压缩原始文件。归档使用确定性 gzip；解压后 SHA256 必须与相应 `run_manifest.json` 的 `generations_sha256` 一致。
