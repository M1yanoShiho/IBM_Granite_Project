# R005 当前独立复验凭据

**记录时间：** 2026-08-12T04:00:05Z  
**服务器：** `fl25387@10.70.71.11`  
**代码目录：** `/home/fl25387/projects/IBM_Granite_Project_latest`  
**代码提交：** `33c95a84c4edeb6d9ec85a3fa74cbf9d62fc0e3b`  
**工作树：** clean  
**正式 bundle：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/selector-adaptive-risk-v1/R005`

## 1. 这份文件证明什么

这份 attestation 保存的是 **2026-08-12 当前重新执行** 的两个只读 verifier，而不是试图证明此前没有留存原始输出的历史执行次数。两个命令均在正式生成 commit 的 clean worktree 上退出 `0`；它们没有改写正式 R005 文件。

## 2. Runner 语义复验

runner 会重新加载 checkpoint、重建固定样本与输入 pin、重新计算 640 条分数、loss、梯度探针和训练门结论，再逐字节比较已有 artifact。

```text
ssh fl25387@10.70.71.11 env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/fl25387/projects/IBM_Granite_Project_latest/src /scratch/fl25387/IBM_Granite_Project_latest/envs/selector_mis_py311/bin/python /home/fl25387/projects/IBM_Granite_Project_latest/src/evidence_rag/cli/run_selector_sanity.py --config /home/fl25387/projects/IBM_Granite_Project_latest/configs/selector/adaptive_risk_r005_sanity.toml --repo-root /home/fl25387/projects/IBM_Granite_Project_latest --model-snapshot /scratch/fl25387/IBM_Granite_Project_latest/hf-cache/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7 --r004-root /scratch/fl25387/IBM_Granite_Project_latest/runs/selector-adaptive-risk-v1/R004 --count-matched-protocol-dir /home/fl25387/projects/IBM_Granite_Project_latest/results/selector-adaptive-risk-v1/R003/count-matched-protocol --output-dir /scratch/fl25387/IBM_Granite_Project_latest/runs/selector-adaptive-risk-v1/R005 --device cuda:0 --verify-only
```

**退出码：** `0`

```json
{"action":"independently-verified","candidate_scores":640,"checkpoint_weights_sha256":"6ec87f8f07cb0275b28d7fbf126ea14d5992683988b28ddae52f300dcfe8d0bd","epochs_completed":30,"output_dir":"/scratch/fl25387/IBM_Granite_Project_latest/runs/selector-adaptive-risk-v1/R005","status":"FAIL","termination_reason":"completed","training_gate":"FAIL"}
```

## 3. Finalizer 结构复验

finalizer 会检查 exact file inventory、symlink、manifests、input pins 与 checksums。此次复验时间窗口为 `2026-08-12T03:59:54Z` 至 `2026-08-12T04:00:05Z`。

```text
ssh fl25387@10.70.71.11 env PYTHONPATH=/home/fl25387/projects/IBM_Granite_Project_latest/src /scratch/fl25387/IBM_Granite_Project_latest/envs/selector_mis_py311/bin/python /home/fl25387/projects/IBM_Granite_Project_latest/src/evidence_rag/cli/finalize_selector_r005.py --run-root /scratch/fl25387/IBM_Granite_Project_latest/runs/selector-adaptive-risk-v1/R005 --verify-only
```

**退出码：** `0`

```json
{"action":"verified","checksums":"/scratch/fl25387/IBM_Granite_Project_latest/runs/selector-adaptive-risk-v1/R005/CHECKSUMS.sha256","checksums_sha256":"384e54dac8e3871e41c58ef55638243df6454cb990ccabe5a0fddad0f8522567","kind":"selector-r005-finalization","manifest":"/scratch/fl25387/IBM_Granite_Project_latest/runs/selector-adaptive-risk-v1/R005/selector_experiment_manifest.json","manifest_sha256":"3d5707358cd4a02f96b094c71ae5910b2c769ca90311583127a88c13ac203a5a","status":"FAIL"}
```

## 4. 复验后的内容哈希

| 文件 | SHA-256 |
|---|---|
| `selector_experiment_manifest.json` | `3d5707358cd4a02f96b094c71ae5910b2c769ca90311583127a88c13ac203a5a` |
| `CHECKSUMS.sha256` | `384e54dac8e3871e41c58ef55638243df6454cb990ccabe5a0fddad0f8522567` |
| `sanity/sanity_report.json` | `7f60c4b490d36f16e119dbfbdc0d6d8601ea4ba1794c502bc36cc74d4b130ec9` |
| `sanity/candidate_scores.jsonl` | `11afdc70c63df49dac81a469b9ecbfb6ccb6d61c982cd10d2c65e6bb1d2b9d1c` |

## 5. 结论边界

本次当前复验证明 formal bundle 仍能独立重算为 `TRAINING-GATE FAIL`，并证明没有把失败后的 modelval/删除 artifact 伪装成已运行。它不把 R005 改为 PASS，不证明自然世界 misinformation 检测，也不授权运行 R006。
