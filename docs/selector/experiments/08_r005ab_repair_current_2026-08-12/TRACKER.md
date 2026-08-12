# 状态记录

- [v2 tracker 快照](snapshots/EXPERIMENT_TRACKER_AMENDMENT_R005AB_v2.md)
- [canonical tracker 固定入口](../../../../refine-logs/EXPERIMENT_TRACKER_AMENDMENT.md)

| ID | 当前状态 | 许可边界 |
|---|---|---|
| D005-1 | COMPLETE | 原 R005 FAIL 可复验 |
| D005-2 | COMPLETE | train-fit-only 诊断完成 |
| D005-3 | SUPERSEDED AUDIT SNAPSHOT | v1 审计有四项漏报，不能授权 |
| D005-4 | COMPLETE | v2 纠错与两路复审完成 |
| A000 | WAITING APPROVAL | 当前唯一合法下一步 |
| A001 | NOT RUN | 未获批准不得实现 |
| A002 | NOT RUN | 必须等 A001 clean commit |
| R005A/B | NOT RUN | 无训练、无 screen/confirm reveal |

归档文件的创建不会改变这些状态。
