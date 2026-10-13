# MobileExplorer 现有实验审计

日期：2026-10-13（香港时间）。范围：116-task seed34 baseline、5 条人工参考轨迹、4-task evidence pilot、graph-study R3/R5、现有探索与图代码。区分“代码存在”“在设备上执行过”和“结论已成立”。

| 组件 | 已有材料 | 当前验证 | 缺口 |
|---|---|---|---|
| Runtime variation data | graph-study 与 116-task 真实任务轨迹 | 有任务 outcome profiling | 116 baseline 多数每模板单 seed；未量化同一 source/action 在 runtime variants 下的后态变化 |
| Reference trajectories | 5 条 direct manual expert trajectory，均 evaluator=1；另有 VLM baseline traces | 5 条人工路径经过 evaluator 验证 | 不能把成功 baseline trace 当 gold；手工 action 标签只覆盖少量决策 |
| A11y instrumentation | expert harness 逐步保存 UI tree/activity/screenshots；baseline 保存 action 与截图 | 人工路径有完整 before/after 观测 | baseline 不含完整 A11y tree，末步稳定后态不足，未探索边不存在 |
| Model-free exploration | explorer_agent_gelab_light、live_probe、OnlineExplorationRuntime、deadline scheduler | 安全候选、live probe、rollback/shadow-isolation 均有代码与测试 | 需要在统一主模型/协议下验证实测的 transition evidence 是否改善 reasoning |
| Rollback / isolation | live_probe 同设备恢复；shadow_isolation 双 emulator 前缀与状态校验 | 实现与测试存在 | R3/R5 summary 的 rollback 字段为 0；需从 raw trace 核实字段语义。online gate 还没有完成的真实配对报告 |
| Memory Graph | ExecutableExplorationMemory、AgentEvidenceGraph、OnlineEvidenceIndex | 多个原型含 state/action/outcome、多个观测结果及 freshness 信息 | 并行表示多，未证明复杂层必要；先以最小 state→action→observed outcome 图为基线 |
| Cross-run matching | activity、A11y structure、image fingerprints、source version 等实现 | 单测存在 | false merge/split、variant 匹配 precision/recall 未测 |
| Evidence retrieval | graph prompt、post-fusion、route retrieval、freshness gate | graph-study 小规模运行过注入/融合条件 | 未显示稳定正增益；相关性与陈旧证据错误需单独量化 |
| Inference integration | GELAB agent 与 explorer GELAB agent；8095/8096 目前返回 GELAB-ZERO-4B | R3/R5 有 evaluator outcome | 本文目标模型为 MAI-UI 2B；当前结果必须标成 GELAB-ZERO-4B 工程筛选，不能冒充 2B 结果 |

## 可确认结果

- 116/116 个 AndroidWorld task template 有有效 seed34 baseline：54 成功、62 失败。这是 profiling，不是 exploration 增益，也没有证明失败由动态 UI 导致。
- 4-task offline evidence pilot 共 16 次请求。无证据 A 为 4/4 可接受；关系证据 C 为 4/4；当前 UI 事实 B 与无关关系 D 各 2/4。每臂精确坐标匹配均 0/4。B/C 改变了证据来源与表示，不能归因关系结构；没有执行动作，不能评价 rollback、transition verification 或 evaluator completion。
- graph-study R3 的 baseline 为 10/10 trials、5/10 成功；prompt/fusion arms 的点估计没有稳定胜过 baseline。R5 计划 30 tasks，但多数 arms 实际仅 3–4 个完成：baseline 4/4 成功、probe-only 1/3 且有 1 exception、graph-prompt 2/3、verified-skip 2/4、post-fusion 2/4。样本不足以推出正面或负面普遍结论。
- R3/R5 确实记录过探测观测，但没有可靠报告 transition prediction accuracy、runtime branch recall、false reuse。图增长不是证据有效性的替代指标。
- experiments/online_exploration_gate 目前含协议与实现约束，未找到已完成的 raw trial JSONL/report。

## 审计结论

现有原型已覆盖真实 UI probing、恢复、transition storage、retrieval 和有限实验，但尚未证明“探索的信息能改善主要模型 reasoning”，也没有验证图后态匹配的可靠性。新的 motivation 应聚焦：当前截图描述的是 state，却不保证 agent 知道自己选择的 action 在当前 runtime 会导致哪种后态；探索要把真实 action→observed outcome 记录成可复用证据，并在主 action 执行后检查实际转移。位置漂移和动态 UI 是待用 paired traces 量化的原因，不能由 62/116 的 baseline 失败直接推出。
