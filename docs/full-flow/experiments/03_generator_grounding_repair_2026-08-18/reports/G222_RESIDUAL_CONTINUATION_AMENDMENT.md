# G222 residual continuation amendment

**日期：** 2026-08-18
**状态：** `CONTROLLED CONTINUATION ALLOWED / NO TRAINING STARTED`
**输入状态：** `G212M5 NOT FREEZE READY / 97 PASS / 3 FAIL`

## 为什么需要 G222

G212M5 的结果不是 100/100，因此不能叫 clean freeze，也不能直接进入 G300。

但它也不是路线失败：100 条固定样本里 97 条通过，0 条不确定，0 条结构性强制失败；unsupported、NIAH train、NIAH model-val 三个层都是 20/20 通过，剩余 3 条失败全部集中在 2Wiki answerable 的关系自洽问题。

所以 G222 把原先容易被误解的硬门拆成三种状态：

| 状态 | 含义 | 是否能训练 |
|---|---|---|
| clean freeze | 固定样本 100/100 通过 | 可以解锁 clean G300 |
| controlled continuation | 有局部失败，但失败少、集中、可定位，且没有 unsupported/NIAH/结构性失败 | 只能继续残余修复/隔离，不能直接训练 |
| stop/fallback | 失败多、扩散、不确定，或触及 unsupported/NIAH 安全层 | 不进入 G300，转 G0 fallback 或停止 Generator 修复 claim |

## G212M5 属于哪一类

G212M5 属于 `controlled continuation`，不属于 `clean freeze`。

依据：

- overall：97 PASS / 3 FAIL / 0 UNCERTAIN；
- forced structural failures：0；
- unsupported：20/20 PASS；
- NIAH train：20/20 PASS；
- NIAH model-val：20/20 PASS；
- failures：3 条，全部是 2Wiki answerable target relation self-containment。

这说明当前路线仍有积极信号，继续做受控残余处理有意义；但它不允许直接跳到训练。

## 剩余失败

| sample_index | case_id | failure class |
|---:|---|---|
| 14 | `2wiki::7e38489c0bda11eba7f7acde48001122` | evidence states France Gall collaborated with Michel Berger, not that he was her spouse |
| 17 | `2wiki::c083071908cf11ebbd95ac1f6bf848b6` | Jessica Birkel evidence gives nationality and birth date, not birthplace |
| 56 | `2wiki::23257d6e087c11ebbd69ac1f6bf848b6` | Rio Grande Band evidence does not directly state the band origin country |

## 新规则

### Clean Freeze

只有以下条件同时满足，才叫 clean freeze：

- 固定样本全部通过；
- 0 uncertain；
- 0 forced structural failures；
- 未启动训练、utility labels；
- 未读取 dev、sealed 或 held-out。

### Controlled Continuation

以下条件允许继续做下一步受控残余处理，但不能直接训练：

- 固定样本通过率至少 95%；
- 0 uncertain；
- 0 forced structural failures；
- unsupported fail = 0；
- NIAH train fail = 0；
- NIAH model-val fail = 0；
- 所有失败都有 case_id、sample_index 和原因；
- 未启动训练、utility labels；
- 未读取 dev、sealed 或 held-out。

G212M5 满足 controlled continuation。

## 风险和限制

G222 不改变以下边界：

- 不把 G212M5 改写为通过；
- 不启动 G300、utility labels 或 Selector 训练；
- 不读取 held-out、sealed、official dev；
- 不把 97/100 写成强统计结论；
- 不把任何后续更小的 2Wiki model-val screen 写成等价于原始 floor。

如果下一步残余隔离会让 2Wiki model-val screen 继续缩小，必须如实报告更小 screen size 和内部验证方差风险。

## 下一步

下一步是 G223：只围绕这 3 条失败做 residual sample-failure quarantine or repair candidate。

G223 可以做两类事情之一：

- 若证据能直接支持目标关系，则做窄 target 修复；
- 若证据不能直接支持目标关系，则隔离对应 case，并记录 model-val screen 影响。

G223 完成前仍不能训练。

## 产物

Machine manifest:

```text
artifacts/G222/residual_continuation_amendment_manifest.json
```

SHA256:

```text
0b4be58c8f1ab1aa5cd2592c96ef9c7548b2907d3df4cb67ec92058b6a28c42d
```
