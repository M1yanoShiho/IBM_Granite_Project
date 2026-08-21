# G300 training implementation report

**日期：** 2026-08-18
**状态：** `COMPLETE / SMOKE PASS / G310 NEXT`
**入口：** `G223 CONTROLLED_CONTINUATION_READY`

## 做了什么

G300 新增了 Generator draft LoRA 的受控训练入口：

- 只接受 G223 controlled-continuation candidate；
- 校验 G223 manifest、train/validation cases、ordered IDs 的 SHA256；
- 使用 IBM Granite 4.1-3B；
- LoRA 固定为 `r=8`、`alpha=16`、`dropout=0.05`；
- 支持 GR-F fresh LoRA 和后续 GR-C continuation adapter；
- loss 只作用于 assistant target；
- prompt 权重为 0，answer/punctuation 权重为 1，citation bracket/index 权重为 4；
- 同一个 case 的多个 context variants 先取平均，再进入梯度累积，避免把 8 个变体当作 8 个独立问题；
- adapter 记录为 draft generation call only，claim splitter 默认使用 adapters disabled 的 frozen Granite base；
- 不读取 dev、sealed、held-out，不生成 Selector utility labels。

这一步是训练实现和 smoke，不是 Generator 效果结论，也不是 clean freeze。

## 长度审计

正式 runtime：

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G300-v1
```

| split | groups | examples | max length | >2304 |
|---|---:|---:|---:|---:|
| train | 2,370 | 9,207 | 2,120 | 0 |
| validation | 302 | 1,924 | 2,030 | 0 |

长度审计状态为 `PASS`。G300 不需要额外 truncation 或 length repair。

## Smoke

正式 smoke 只取最长的 1 个 train group 和 1 个 validation group：

| 项 | 值 |
|---|---:|
| recipe | GR-F |
| seed | 13 |
| train groups | 1 |
| training examples | 8 |
| optimizer steps | 1 |
| train mean group weighted loss | 3.691103 |
| validation mean group weighted loss | 2.535270 |
| peak CUDA memory | 9,118,088,192 bytes |
| trainable parameters | 15,564,800 |
| trainable percent | 0.455324 |

选中的 train case 为 `niah-old::11690`；选中的 validation case 为 `niah-new-modelval::1148`。adapter 保存成功，并从 fresh base strict reload 通过。

## 判定

G300 通过：训练入口、长度审计、token weighting、query-group equalization、LoRA 保存和 reload 都可执行。

G300 不代表以下结论：

- 不是 clean 100/100 freeze；
- 不是 Generator repair 成功；
- 不是正式候选训练结果；
- 不是 Selector utility labels 的授权；
- 不是 held-out 授权。

下一步可以进入 G310 seed13 screen：用同一 G223 数据和同一 G300 训练入口，对 GR-F 与 GR-C 做受控配方筛选。

## 验证

服务器测试通过：

```text
6 passed in 0.02s
```

本地语法检查通过；本机 Python 环境没有 pytest，因此 pytest 在服务器环境执行。

## 产物

| 产物 | SHA256 |
|---|---|
| `scripts/full_flow_g300_draft_lora_train.py` | `8973ac773c4da18dd1648c88de8ab316378ffb99eb59d080e170ac9767014b77` |
| `tests/scripts/test_full_flow_g300_draft_lora_train.py` | `ce2f54672a3dd7c7587df93a1cdf9fe147e8a4e238fbf2ac675dbc0b6dc0357b` |
| `artifacts/G300/G300_LENGTH_AUDIT.json` | `1880d625f897453c0badcc1ffe76eeee970c601aa4f1d6a950af3b0649a35184` |
| `artifacts/G300/G300_SMOKE_TRAINING_MANIFEST.json` | `c21a8a6369d65ab12789f6959f682fb884a5c4f097f3843deb9fdb8921a34f85` |
| `artifacts/G300/logs/g300_length_audit.log` | `d5fbc97e007d4874dc077ba3667b43cf3ea828db34b9f83eb161268b8d1042a5` |
| `artifacts/G300/logs/g300_smoke_grf_seed13_g1.log` | `d9c00d8e4cb79761253e19886d9b9a72d72365315d46aa35716abc1de778fc6e` |
| `artifacts/G300/G300_EXECUTION_AUDIT.json` | `18fb92a4e728f397f5e7ce0914e962faec7bba630a33207c0bee12a5598da20e` |
| runtime smoke adapter weights | `ad6f968728d817cd8aabf79f516b2590dca09d506e5afc08ae2c9a979b8b13ec` |
| runtime smoke adapter config | `eb24bc1c24ba55081bc7e3168d02a6a59e717d607912be75db0b331bb9f90c45` |
