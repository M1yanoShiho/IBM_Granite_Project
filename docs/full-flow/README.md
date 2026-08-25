# Full evidence flow（完整证据流）

这里保存跨越 Retriever、Selector 与 Generator 的联合研究计划和实验记录。

它与三个模块自己的目录分工如下：

- `docs/retriever/`：只记录 Retriever 的方法与模块实验；
- `docs/selector/`：只记录 Selector 的方法与历次实验；
- `docs/generator/`：只记录 Generator 的方法与模块实验；
- `docs/full-flow/`：记录三个阶段如何连接，以及局部改善能否传递到最终答案。

当前入口：[冻结三模块系统完整评估](experiments/04_frozen_three_module_system_evaluation_2026-08-21/README.md)。该路线使用五个防偏移执行目标并在 PASS 后自动接力；Goal 2 已 `PASS (= READY)`，Goal 3 当前 active。其 runner/评分实现与公开模型缓存已就绪，但冻结自定义 checkpoint 需从备份恢复；尚未生成或评分 held-out。

全部文件与状态见 [MANIFEST.md](MANIFEST.md)。
