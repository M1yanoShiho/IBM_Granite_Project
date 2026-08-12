# Independent reviewer output

独立只读复审结论：`P0=0，P1=0，P2=0`。v2 可作为“待用户明确批准”的稳定草案；仍是未实现、未运行，未对代码或实验结果作验证性背书。

- FIT terminal：满足。计划规定唯一顺序为 checkpoint 冻结 → strict reload/fingerprint → fit scores → threshold table/trace/thresholds → qualification/loss/technical gate → closure → `FIT_TERMINAL`，且 COMPLETE 绑定 registry、anchor、claim、STARTED、checkpoint、fit-score、threshold/gate hashes；终态后仅零变更 verify-only。
- 五份 ordered manifest：满足。A/B 均有 full/config/checkpoint/fit-gate/threshold 五条绝对 literal path；统一 canonical object envelope、固定 type、严格 ordinal/member fields，四个 projection 只能由 full members 原序投影，anchor/STARTED/reveal-input 都绑定 path/SHA/type/count。STARTED 前只准这五项；任何其他 orphan/临时/事件/bundle 均 veto/STOP。
- B 授权与锁：满足。两个 B jobs 预注册同一授权路径；各自 anchor/claim/`FIT_STARTED` 直接记录同一 authorization path/SHA、A split/session/attempt 与 V*，并经 anchor→claim→STARTED SHA 链复验；closed-world verifier 再确认两 B jobs 完全一致。锁顺序固定为 A-screen owner lock → B job lock，B STARTED fsync 后才放 A lock；竞争者零训练/零变更退出，未见 TOCTOU 或合规路径下的死锁环。
- 旧 R005 回归边界：已修正。A001 只做 synthetic/fixture 的旧 finalizer zero-write verify-only 与全仓测试，不再运行真实旧 runner；历史 real-data 复验只引用既有 attestation。旧 runner 确实在 `--verify-only` 分支前验证 R004 并装载 datasets。
- canonical/state recovery/五件链：无绕过。编码、fsync/no-replace、同锁全临界区、orphan/STARTED-without-terminal 的 BURNED/veto、五件链的每次下游授权前重验均明确 fail-closed。

未运行测试或服务器操作；未读取 held-out。
