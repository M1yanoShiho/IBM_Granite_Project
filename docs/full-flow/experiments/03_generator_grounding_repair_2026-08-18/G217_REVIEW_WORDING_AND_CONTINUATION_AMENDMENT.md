# G217 review wording and continuation amendment

**日期：** 2026-08-18  
**状态：** `COMPLETE / WORDING AMENDED / NO TRAINING STARTED`  
**输入状态：** `G212M2 FAIL / 92 PASS / 8 FAIL / 0 UNCERTAIN`

## 用户问题

用户指出：计划不需要强调固定样本判定到底由谁完成，也不应把未达到某个数字误写成路线没有意义。只要结果有积极信号，就应该继续推进；但如果重复问题说明继续盲目推进会浪费时间，就要转入受控修复。

本修订只处理解释和措辞，不改变任何实验结果、阈值、数据边界、模型、seed、训练状态或 held-out 边界。

## 修订后的解释

`76/100` 和 `92/100` 都不是路线失败的证明。

- `76/100` 的含义是：原 G210 数据不能冻结，因为 2Wiki model-val 太小，不适合做正式 Generator 配方选择。
- `92/100` 的含义是：当前数据有积极信号，大多数固定样本可用；但重复失败集中在 target construction，因此不能直接训练。

所以继续推进的规则是：

```text
积极信号 + 可解释重复问题 -> 受控修复并重跑冻结前检查
积极信号 + 检查全部通过 -> 允许进入下一阶段
无积极信号或大面积随机污染 -> 停止正式路线或降级结论
```

## 措辞修订

后续文档统一使用中性表达：

- `sample review/adjudication`
- `fixed sample`
- `固定样本判定`
- `独立样本审查`

不再围绕执行主体来表述该步骤。重要的是判定规则固定、样本固定、输入边界固定、产物可复查。

## 对当前路线的影响

G217 不解锁训练。

当前路线仍是：

```text
G218 systematic target repair
-> rerun structural/TRUE as required
-> rerun length/sample review
-> only if freeze readiness passes, enter G300
```

G300、utility labels、S/I/H 和 held-out 仍 blocked。

## 本阶段产物

| 产物 | 用途 |
|---|---|
| `G217_REVIEW_WORDING_AND_CONTINUATION_AMENDMENT.md` | readable amendment |
| `artifacts/G217/review_wording_manifest.json` | machine-readable amendment manifest |

Manifest SHA256:

```text
1f0302569c85a46a1252a5f78dffc2dfc77a77a48a3e565a575b77ef7d9ff65d  artifacts/G217/review_wording_manifest.json
```

## 阶段判定

G217 通过。下一步继续执行 G218 systematic target repair，不启动任何训练。
