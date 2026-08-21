# G000 协议和统计范围冻结报告

**日期：** 2026-08-18
**状态：** `PASS`

## 完成内容

- 已把第 03 路线从等待确认推进到受控执行，并冻结 G000 的协议、输入清单、denylist 和服务器实体核验。
- 未启动训练、未生成 utility labels、未运行 held-out，也未修改 Retriever、Legacy Selector、TRUE 或历史 G230 结果。
- 正式 runtime 边界保持为 `Retriever -> Selector -> Generator -> one answer`；gold/reference 只允许离线构造目标、utility label 和生成后评分。

## 核验结果

- `route_authorized`: `PASS`
- `git_branch`: `PASS`
- `git_remote_sync`: `PASS`
- `g230_archive_hashes`: `PASS`
- `runtime_gold_boundary`: `PASS`
- `heldout_boundary`: `PASS`
- `server_entity_verification`: `PASS`
- `training_or_utility_generation_started`: `PASS`

## 冻结身份

- Retriever: `67333c6f3fcf6567756b382048961bc150da0ee26af4f9cecc59f58f30b5d78d`
- Legacy Selector checkpoint: `86622bd9ab6391c9eb560133b01b0cf3744c3706638ff8b0bd38925b84bf72bf`
- Generator base: `ibm-granite/granite-4.1-3b@c0650403e44e78ec0262dab1c90914c65b196c4e`
- G230 archive: `53ddd20376db7478d91cbf1a20bfca20118df314c5b20c8cf98a992ae471f8be`

## 下一步

G000 通过后，下一阶段只允许进入 G010/G100 的预注册功效范围与 citation 断点归因；不得跳到训练。
