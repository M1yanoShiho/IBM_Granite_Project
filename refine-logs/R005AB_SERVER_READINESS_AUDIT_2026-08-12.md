# R005A/R005B 服务器实施就绪性只读审计

**日期：** 2026-08-12

**范围：** 只读核验 Git 对齐、服务器基础能力、canonical namespace 是否已被消费，以及是否存在未经批准的新 amendment 运行迹象。

**结论：** `WARN（P0=0，P1=1，P2=0）`。设计协议可在用户明确批准后进入 A001；当前没有批准，也没有实施、训练或 held-out reveal。

## 1. 一句话解释

服务器具备实现安全状态机所需的基础接口，而且新实验的正式目录仍完全未创建；唯一需要处理的 P1 是服务器仓库仍停在旧 R005 提交。它不是科学方法失败，而是 A001 开始后必须完成的代码版本对齐工作。

## 2. 只读证据

| 检查项 | 观测 | 判定 |
|---|---|---|
| 本地分支 | `refactor/three-module-baseline` @ `454acdfc60118e0cbb4c6fe1647c93e1aa384f47` | PASS |
| GitHub 同名分支 | `454acdfc60118e0cbb4c6fe1647c93e1aa384f47` | PASS |
| 服务器工作树 | 同名分支、clean，但仍为 `33c95a84c4edeb6d9ec85a3fa74cbf9d62fc0e3b` | WARN / P1 |
| amendment 计划 blob | MD `a1073626ec6560b5dbea37477ff728e20ade68cd`；JSON `d52e76c4be5814b74bb77834915a40eb26e9b8e1` | 与终审稳定快照一致 |
| canonical run root | `/scratch/fl25387/IBM_Granite_Project_latest/runs/selector-r005-amendment-v1` 不存在 | 未消费 |
| canonical audit root | `/scratch/fl25387/IBM_Granite_Project_latest/audit/selector-r005-amendment-v1` 不存在 | 未消费 |
| canonical external audit root | `/scratch/fl25387/IBM_Granite_Project_latest/audit-journal/selector-r005-amendment-v1` 不存在 | 未消费 |
| 对应运行进程 | 排除查询命令自匹配后，没有匹配进程 | 未启动 |

服务器环境的只读能力证据：

- Linux `5.15.0-139-generic`，glibc `2.31`，Python `3.8.10`；
- `/scratch` 是本地 XFS，挂载为可写；
- `flock` 可用；Python 可见 `fcntl.flock`、`os.link`、`O_DIRECTORY`；libc 可见 `renameat2` 符号。

这些证据只说明“有实现基础”。本审计没有做写入，因此没有证明 `RENAME_NOREPLACE`、hard-link no-replace、directory `fsync`、并发互斥、崩溃窗口和恢复语义真的实现正确。那些必须在获批后的 A001 中用临时 synthetic fixture 和故障注入测试验证。

## 3. 唯一 P1

服务器仓库比本地/GitHub 落后。A001 的 clean-commit 出口要求三处最终指向同一个实现提交，因此 A001 获批后必须：

1. 先让服务器工作树安全快进到获批基线；
2. 只使用旧 R005 train-fit 与 synthetic fixture 完成实现和测试；
3. 形成新的 clean A001 commit 并推送 GitHub；
4. 再让服务器与该 A001 commit 对齐并复验 clean 状态。

在这条链完成前，不得运行 A002，不得物化新 held-out 样本，不得启动 formal fit 或读取 A-screen/B-confirm 内容。

## 4. 当前合法下一步

当前唯一合法下一步是 **A000：取得用户对 R005A/R005B amendment 的明确批准**。批准后才能开始 A001。此次审计不代表 A001 已通过，也不代表 R005A/R005B 的科学效果门一定会通过；TopK10 默认路径保持不变。
