# 实验结果图表与一句话结论

数据截点为 2026-10-13 05:41 UTC。Live profile 使用 GELAB-ZERO-4B；这是 pilot 结果，不是 MAI-UI 2B 最终结果。下列离线 proxy 不等于执行正确率或 task success。

## P0：source/action-slot/replay 机制 smoke

![P0 机制结果](figures/current_progress_20261013/p0_exact_match_validation.png)

一句话结论：3/3 个受控导航场景通过 exact source、action-slot、evidence retrieval、错误 source/action 拒绝及 replay successor 校验，且 VLM 调用数为 0。

![Markor 的真实 source、probe successor 与 replay successor 截图](figures/p0_actual_gui_example.png)

一句话结论：Markor To-Do tab 的真实截图对证明了同 Activity 内 UI successor 可被实际 probe 并被正式 replay 重现。

## 两任务 one-call 并行探索 live smoke

![请求完整性、恢复与 task outcome](figures/current_progress_20261013/onecall_live_smoke.png)

一句话结论：19/19 个请求保持单次推理且 7/8 个 probe 恢复，但两任务 outcome 与历史 baseline 相同，样本只验证运行链路。

## 完整 116-task acquisition profile

![任务进度与 outcome](figures/current_progress_20261013/progress_and_outcomes.png)

一句话结论：40-task pilot 与 76-task extension 均跑完，最终 outcome 为 51 success、54 failure、11 unknown，不能从中推断 exploration 的增益。

![样本 baseline 与复杂度构成](figures/current_progress_20261013/cohort_profile.png)

一句话结论：样本覆盖 53 个历史成功、62 个历史失败、1 个未知以及 low/middle/high 难度 62/31/23。

![Probe 严格恢复](figures/current_progress_20261013/probe_recovery.png)

一句话结论：245 次 profile probes 中 224 次严格恢复成功（91.4%），剩余 21 次是当前安全与可用性短板。

![统一定义下的 runtime UI structure change](figures/current_progress_20261013/runtime_state_deltas.png)

一句话结论：按 interaction-skeleton 签名统计，138/245 次 probe 改变 UI 结构（56.3%），且 245 次都没有改变 Activity。

![真实 UI 截图与 BrowserMaze trace 状态](figures/current_progress_20261013/ui_structure_real_examples.png)

一句话结论：BrowserMaze trace 显示相同 `FilesActivity` 内滚动改变 layout signature，但其 post-probe screenshot 没有保存，图中后态来自 hierarchy/signature。

![证据从检索到决策的 funnel](figures/current_progress_20261013/evidence_funnel.png)

一句话结论：24 条事实被选进 prompt，但 1,019 次 action-slot audit 仅 5 次精确 source/action/successor 匹配，证据可用率仍低。

## 24-state 离线 evidence-utility pilot

![No-evidence、fresh、fresh+affordance 与 wrong-source proxy](figures/current_progress_20261013/offline_evidence_proxy.png)

一句话结论：fresh 仅比 no-evidence 多 1/24 条参考路径一致，wrong-source 与 fresh 持平，因此没有 source-specific utility 证据。

## 设计图（不是实验结果）

![Runtime exploration、图和单次推理的设计闭环](figures/current_progress_20261013/exploration_graph_reasoning_loop.png)

一句话设计结论：并行 explorer 验证 action-slot 后果并追加多分支 runtime 图，未来匹配到该 source 的单次推理读取它，正式后态再验证它。

细节、定义与设计含义见[完整阶段报告](interim_progress_and_design_20261013_zh.md)；原始计数见[data snapshot](data_interim_progress_20261013.json)。
