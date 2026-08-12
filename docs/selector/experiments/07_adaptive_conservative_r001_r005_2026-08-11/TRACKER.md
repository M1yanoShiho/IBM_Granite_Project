# 状态记录

完整 contemporaneous tracker 见 [snapshots/EXPERIMENT_TRACKER_v2.md](snapshots/EXPERIMENT_TRACKER_v2.md)。最终摘要：

| Run | 状态 | 可保留的成果 |
|---|---|---|
| R001 | PASS | 六个 Hybrid RRF Top20 pool 与哈希冻结 |
| R002 | PASS | component/role 隔离、样本量与审计协议 |
| R003 | PASS | TopK10/9/8/7 数量基线和 count-matched 协议 |
| R004 | PASS | 80,460 标签、资源与 token preflight |
| R005 | COMPLETE / TRAINING-GATE FAIL | 失败可复验；train-fit-only 诊断可继续指导设计 |
| R006–R015 | CUT / NOT RUN | 没有因为失败而偷跑后续效果实验 |

原 R005 的失败状态永久保留，不因第 08 路线而改判。
