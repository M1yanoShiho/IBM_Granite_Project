# 当前 Selector 实验完整性审计

**当前审计：** [`R005_EXPERIMENT_INTEGRITY_AUDIT_2026-08-12.md`](R005_EXPERIMENT_INTEGRITY_AUDIT_2026-08-12.md)  
**总判定：** `WARN`；R005 `COMPLETE / FAIL` 与 fail-closed 停止决定成立，P0=0  
**主要整改：** 旧的历史 verify/audit 次数没有保存独立 attestation；现已保存带命令、commit、时间、退出码与哈希的 [`R005_VERIFICATION_ATTESTATION_2026-08-12.md`](R005_VERIFICATION_ATTESTATION_2026-08-12.md)，并修正文档中过强的历史表述。历史事件无法事后补证，因此审计仍保留 WARN 边界。

完整的六项检查、artifact 对账、泄漏核验和问题分级见上面的带日期审计文件。该固定入口只指向最新审计，不覆盖带日期版本。
