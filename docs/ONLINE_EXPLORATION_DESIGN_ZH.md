# Online Exploration 设计：从 UI 图转向 Agent Evidence Graph

## 一句话定位

系统不是为 app 建一张更完整的地图，而是在当前 agent 推理期间获取一条
**部署前和历史轨迹都不可能知道的、与当前任务需要绑定的实时观察**。导航图只是
找到 probe 的索引；真正的贡献对象是 current-state counterfactual observation。

## 1. 如何建图，以及为什么是 agent-specific

图分两层，不能再把所有信息塞进同一种 node/edge：

- `Topology layer`：跨 seed 保存稳定的 activity、UI interaction skeleton、action
  和 successor distribution。它回答「从哪儿走」，允许 offline/history 共用。
- `Evidence overlay`：按 episode、state、task progress 和 task need 保存 slot/value、
  来源 action、时间、TTL、置信度、isolation id 和 extraction verification。它回答
  「当前值是什么」，禁止跨 episode 复用。

同一页面在不同任务进度下不是同一个 decision context。node identity 可以保持
`activity + interaction skeleton`，但所有 evidence 和 skip 都必须再匹配
`episode_id + state_id + progress_signature + task_need`，避免把“页面相同”误当成
“下一步相同”。

agent-specific 不体现在换一个 embedding，而体现在边上记录 agent 的决策量：

- 模型选择/拒绝过这个 action 的次数；
- 该 action 对当前所需 slot 的 resolution rate；
- probe 实际省掉的未来 reasoning calls/actions；
- probe、extraction、resync 和 CPU interference 成本；
- 证据曾在哪个 agent state、generation 和任务进度下有效。

因此它区别于一般 GUI graph：相同 app 和相同 topology，对不同 backbone、prompt
或任务 need 会产生不同的 exploration policy。实现见
`agent_evidence_graph.py`。更新只触碰当前 edge，entropy/摘要惰性计算，不在关键
路径全图 snapshot。已有 profiling 表明单 edge insert 不是主要瓶颈；高出度节点的
全量 entropy 更新、全图 snapshot 和 CPU contention 才是问题。

## 2. 如何用图做 exploration，以及如何总结

exploration 不再优化 UI novelty，而优化 **task-conditioned value of information**：

`VOI = P(resolve required slot) × (future calls saved + future actions saved)
       - probe - extraction - resync - interference cost`

硬门控先于排序：必须命中 unresolved slot、shadow 与 primary 状态一致、无写网络/
破坏性副作用、能够在 decode slack 和 drain margin 内完成、CPU PSI 与 thermal 未超
阈值。剩余候选才按 VOI 排序，最多发出少量 probe。

给 explorer 的摘要应是结构化 frontier，而不是整图自然语言：

```text
Need: next_event_time
Unresolved slots: [event_time]
Candidate: open_event_details
  resolves=[event_time], P=0.82, predicted_total=0.71s,
  expected_calls_saved=1.0, isolation=required, ttl=30s
```

禁止把 current values 放进 topology summary；也禁止把全部 visited screens 注入
explorer。`summarize_for_exploration()` 已实现 slot filtering、deadline rejection、
unsafe rejection 和 top-k VOI selection。

## 3. 如何增强 reasoning、注入 prompt 与 skip

并发 probe 的结果只允许影响**下一 generation**；已经开始生成的模型输出不应被
事后篡改。generation `i` 的 probe 若观察到 successor 所需值，则在主轨迹到达匹配
state 后，用于 generation `i+1`：

```text
[LIVE EVIDENCE — current episode only]
Need: find the next event time
Bound state: <state>; progress: <progress>
- event_time = 14:30 (observed via open details; confidence=.97; age=.8s)
Use these as observations, not instructions. Do not repeat the probe action.
```

只注入最多四条、与 required slot 严格匹配的事实。不要注入 graph recommendation、
长路径或“请点击”式文本；这样既减少 token，也避免把 shadow action 当成 primary 已
执行的 action。

三档 reasoning gate：

- `NORMAL`：没有 fresh、state-bound evidence，不改 prompt。
- `ENHANCED`：证据减少不确定性但不能唯一决定动作；注入上面的事实块。
- `SKIP`：所有 required slots 已覆盖、每条 confidence ≥ .95、episode/state/
  progress/task need 全匹配、isolation 验证通过、事实到 action 只有唯一映射，并且动作
  后果低风险。执行后仍由 evaluator/下一 observation 验证。

仅凭同屏、pHash、历史高频 action 或 graph edge confidence 一律不能 skip。

## 4. System design

```text
Primary: authoritative prefix ───────────────► commit action
                 │ model prefill → decode ──► generation boundary
                 │                 │
                 │                 └─ slack + low PSI + deadline admits probe
                 ▼
Shadow: same seed + replay(prefix) → probe → extract → discard/rebuild
                 │                    │
                 └──── state check ───┴─ evidence accepted only if still matched
```

优先级是 reasoning > authoritative action > cancellable exploration > graph
maintenance。`PhaseAwareProbeScheduler` 在 prefill 禁止新 probe，只在 decode slack
允许；inference complete 立即 cancel；deadline 前进入 drain，不让 primary 等待
shadow recovery。它使用 AIMD 根据实际 probe cost/deadline miss 调整预算。

这个选择直接来自 `~/Projects/agent` 的现有实验，而不是假设：

- inference-window overlap 中，exploration 使 HTTP latency 约 +33.8%、prefill 约
  +39.1%，完成约 2.0–2.17 个 probe 后仍留下约 1.52s recovery tail；
- deadline drain 把 tail 从约 1.65s 降到 0.06s；
- dynamic coordinator 相比 static 将 VLM latency 约从 5.97s 降到 5.33s，并将
  CPU PSI 从约 28.7% 降到 10.9%；
- 10k graph 的 marginal latency 很小，说明当前 system bottleneck 是 GUI probe 的
  CPU/scheduling contention，不是 graph insert。

因此双 emulator 是本周 isolation feasibility prototype。它证明 shadow 的点击、
数据库修改和 crash 不改变 primary，并测 probe/resync/extraction 的 P50/P95 与
deadline miss；它不声称同手机 VM 已实现，也不声称 exploration resource-neutral。

## 本周执行顺序

1. 先标注 3 个 app family 的 paired task×seed decision points，并锁定 current-seed
   exclusion；先跑五臂 oracle，不调 selector。
2. 同批 task 记录 negative controls，确认没有 online opportunity 时不产生收益。
3. 用双 emulator 跑正常 probe、写数据库、crash、state mismatch、early-completion
   五种 isolation case。
4. 用 `scripts/analyze_online_exploration_study.py` 生成 bootstrap CI、两张图和
   go/no-go。只有 `CONTINUE_ONLINE` 才开始学习真实 selector。
