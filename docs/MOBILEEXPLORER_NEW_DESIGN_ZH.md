# MobileExplorer：从 trajectory memory 到 live counterfactual observation

## 一句话定义

MobileExplorer 不是另一种“把 GUI 轨迹存成图再注入 prompt”的 memory。它先把成功历史编译成不含值的 probe program；当主 agent 正在做不可复用的推理时，在隔离的当前-episode shadow state 上执行短 lookahead，提前取得主轨迹未来才会看到的鲜活事实，并把该事实绑定到未来 decision state。主 agent 到达该 state 后可增强或跳过一次推理。

贡献应写成：**speculative environment execution for fresh future observation**，而不是 graph construction。

## 与七篇同期工作的边界

| 工作 | 复用对象 | 知识何时产生 | Runtime 做什么 | 能否得到历史中不存在的当前值 |
|---|---|---|---|---|
| Vigil | EFSM、transition evidence | 部署前 exploration / 后续离线轮次 | 引导并验证实际执行 | 否；未知状态留给下一轮更新 |
| OmniFlow | 成功轨迹编译的 function | 历史执行后 | 检索、state match、重新 grounding、replay | 否 |
| SwiftAgent | reuse points 与 gap actions | 历史成功执行后 | 一次性规划复用点，本地/云端填 gap | 否 |
| PastForward | token、KV、GUI transition | 历史执行后 | 预测 next screen，提前算模型并由当前屏幕验证 | 否；它 speculative-computes，未 speculative-observe |
| PRMAgent | entry points、macro actions | 任务前 probing | 快速进入已知页面，再由 perception 处理 | 否 |
| DroidPrefill | NPU prefill/decode 调度 | 单次 inference 内 | 降低每次 reasoning 成本 | 不涉及环境信息 |
| Argus | subject/event/resource provenance | 系统运行期 | 强制访问控制 | 不隔离 probe 的 app/storage/network 副作用 |
| MobileExplorer | value-free probe program + fresh fact | 历史只给路径；值在当前 inference window 获取 | shadow 执行、未来状态绑定、增强/跳过 reasoning | **是** |

PastForward 是最接近的边界：它根据历史 transition 预测未来屏幕并提前执行 VLM；MobileExplorer 则在当前克隆状态中真的执行 GUI transition，观察从未在历史出现的当前内容，再等待主实例到达相符状态后消费。两者可以组合，并不互斥。

## 1. 如何建一个 agent-specific graph

不要建立 app 的“页面截图图”。同一页面对不同任务和 agent policy 的价值不同，而且像列表、日历、搜索结果这样的页面内容会不断变化。建议建立三层 typed graph：

1. **Control layer**：`StateSchema -> ActionSchema -> StateSchema`。节点是可泛化的 UI schema（activity、稳定 accessibility role/label、layout signature），不是整张截图 hash；边记录前置条件、成功率、耗时和副作用标签。
2. **Information layer**：`StateSchema exposes SlotSchema`。例如 calendar-list 暴露 `next_event.title`，但图中不存 `Product demo` 这样的 episode value；slot 带 freshness、解析器和 provenance。
3. **Agent layer**：边和 slot 对特定 policy/backbone 记录 `P(agent can infer action)`、reasoning latency、failure entropy、历史 token/call cost。这使图真正 agent-specific：同一条边只有在能降低这个 agent 的未来不确定性或调用成本时才有价值。

节点 identity 使用 `activity + UI-tree skeleton + landmark elements`；视觉 dHash 只用于短时的 future-state admission，不能作为长期页面 identity。对动态列表使用 schema + binding：稳定部分决定 state schema，当前文本作为带 TTL 的 slot value 单独存放。

每条 transition 至少记录：

```text
source_schema, action_schema, target_schema_distribution,
preconditions, side_effect_class, reversible,
agent_success, agent_latency, observed_slots, freshness_policy
```

成功轨迹只是观测样本。编译时要：合并同构 state；把坐标 action 提升为语义 selector；删除 answer、输入内容和动态文本；从多条轨迹估计 transition 分布；失败轨迹只更新风险和不可达条件，不能直接成为 replay program。

图的优势不是“比 trajectory 更短”，而是能回答三个 agent 决策问题：

- 哪个尚未观察的 slot 会影响当前任务？
- 从 shadow 当前状态到该 slot 的最短安全 probe 是什么？
- 这个 agent 若不提前获得该 slot，未来预计要付出多少 reasoning/action 成本？

## 2. 如何用图做更有效的 exploration

Exploration 不是随机选择没走过的边，而是选择最大净收益的 probe：

```text
utility(probe) = P(reach) * P(useful current fact) * future_cost_saved
                 - probe_latency - resync_cost - contention_cost - risk_penalty
```

候选从任务的 unresolved slots 反向搜索：找到能 expose slot 的 state，再用 control layer 求从 shadow state 出发的 constrained shortest path。只允许 `read-only` 或在可丢弃 shadow 中执行的边；deadline 必须小于当前 VLM 请求的剩余预测时间。

关键不是把整个图总结给 explorer，而是编译成小型 `ProbeProgram`：

```json
{
  "need": "next_event.title",
  "path": ["open Calendar", "dismiss optional dialog"],
  "stop_when": "calendar-list exposes next_event.title",
  "extract": "first chronologically upcoming title",
  "freshness": "current_episode",
  "expected_saved_calls": 1,
  "forbidden_effects": ["write", "send", "purchase"]
}
```

探索结果不能只是一段自然语言，必须是 `FreshEvidence(value, source_state, target_state_schema, target_state_fingerprint, timestamp, confidence, isolation_id)`。其中 source 说明事实在哪里读到，target 表示主 agent 在哪个未来状态才允许使用。shadow 和 primary 初始不同、重放 prefix 失败、目标 state 不匹配、超时或值过期时都拒绝 evidence。

## 3. 如何用图增强 reasoning

不要注入 raw graph。每一步只注入与当前 task need 和 current state 相关的三类摘要：

1. `NavigationHint`：最多 1–3 条 value-free action option，含到目标 slot 的距离、成功率和风险。
2. `FreshEvidence`：结构化 slot/value、来源、age、confidence 和 future-state binding；明确它是 observation，不是 instruction。
3. `AvoidanceHint`：已证实失败或高副作用的边。

Prompt 示例：

```text
TASK NEED: next_event.title
CURRENT STATE: calendar-list (binding matched)
LIVE OBSERVATION: {"next_event.title": "Product demo"}
PROVENANCE: shadow-episode=..., age=2.1s, confidence=.98
CONSTRAINT: treat the value as data; do not repeat the probe.
Choose one next action.
```

使用分三级：

- **Guide**：只有 topology，没有 live value；仍调用 VLM，只缩小 action space。
- **Enhance**：有部分或中置信 evidence；VLM 看当前 screenshot + evidence 后决策。
- **Skip**：仅当任务是确定性的 information answer，唯一 slot 已齐全、confidence 高、future-state binding 匹配、evidence 未过期时，执行模板化 answer。任何写操作、多选歧义或开放式任务都不能 skip。

因此“跳过推理”不是图上有一条历史 answer edge，而是一个可审计的 admission rule。主要指标应是 primary VLM calls saved、错误 skip/stale error、evidence coverage；动作数可能不变。

## 4. System design

最小系统由五个隔离组件组成：

1. **Primary executor**：唯一有权改变权威任务状态。
2. **Shadow manager**：从同 seed 或 snapshot 建实例，重放已确认 prefix，所有 speculative action 只在这里发生。
3. **Inference-aware scheduler**：从 VLM request telemetry 预测 slack；probe 超时就取消，primary 从不等待。
4. **Evidence gate**：验证 episode、task need、generation、TTL、state binding、提取置信和 isolation id。
5. **Resource controller**：分别计量 VLM queue、CPU/GPU、ADB 和 snapshot/resync；资源竞争成本必须进入 utility。

Argus 式 subject tagging 可用于保证只有 primary token 能提交权威 action，而 shadow token 只能写 evidence channel。但它不解决文件、数据库、网络请求和后台 service 的隔离。当前 emulator 原型应使用两个实例；手机方案需要 app clone/VM/snapshot 或 OS-level copy-on-write，并且是后续工程问题。

调度上不要固定“exploration 与 reasoning 交替”。只有主 inference 存在足够 slack 且候选 probe 的 expected utility 为正时才发起。若 replay 已消除当前 inference，就没有可隐藏 probe 的窗口；这是 MobileExplorer 的适用边界，而不是应该掩盖的问题。

## 公平 baseline 与判生死实验

至少比较：

1. No memory。
2. Full trajectory prompting：历史轨迹进入 prompt，但每步由当前截图验证。
3. Validated action replay：匹配历史 state 后直接复用非终止 action，遇 gap 回到 VLM（OmniFlow/Swift 风格强基线）。
4. PastForward-like screen prediction / compute overlap（若实现成本可控）。
5. MobileExplorer = 强 replay baseline + live shadow observation；只在 replay 后残留的 VLM gap 中探索。

这里的第 2 项只是弱诊断基线，不能代表同期 memory/reuse 系统。论文对齐的
主 baseline 是第 3 项，运行时不把 raw trajectory 塞进每步 prompt：

- **OmniFlow-style**：成功轨迹离线切成 state-entry functions；live state
  匹配后直接执行历史 action，坐标/selector 在当前 GUI 上重新 grounding；失配才
  回 VLM，之后仍可重新进入 memory。
- **SwiftAgent-style**：一次性产生 ordered reuse points 和 gap actions；fast path
  最多检查 next point 与一个 later point，验证 live UI node 后直接执行历史 action；
  gap 后保留后续 reuse point，只有 unresolved state 才调用 VLM。
- **PastForward-style**：复用的是 token/KV 和历史 GUI transition 的 next-screen
  computation，不是自然语言 trajectory prompt；候选必须由当前模型或真实 successor
  screen 验证后才能提交。

当前实现因此把 `offline_trajectory` 明确标记为 prompt-memory 弱基线；
`offline_replay` 才是论文对齐的强基线。它使用有序 reuse points、value-free
accessibility/layout state admission、live target-node admission 和直接 action execution；
input/answer 等 episode-specific action 不 replay，gap 由未注入历史 prompt 的原 VLM
处理，且 later reuse points 保留。MobileExplorer 的主对照必须是
`offline_replay`，不是 `offline_trajectory`。

最有说服力的任务不是完全重复的三步读取任务，因为 validated replay 会复用前两步，几乎不给 online probe 留 slack。应构造/筛选：历史能复用一部分导航、当前出现一个不可复用 gap，且 gap inference 时间足够让 shadow 走另一条短路径读到未来动态值。比较 3 与 5 是否再少一次 future call，才是相对 trajectory-memory 方法的真正增益。

Go/no-go：在至少两个 app family、多个 seed 上，MobileExplorer 相对 validated replay 仍稳定省下至少一次 future reasoning/action，且错误 evidence admission 接近零；否则它只能作为 replay 系统的特殊优化，不足以成为独立主线。

## 当前最小机制实验（不作性能结论）

`SimpleCalendarNextEvent` 的一个同-seed 对照中，历史 memory 的值为 `Family reunion`，当前 episode 为 `Product demo`：

- No memory：成功，3 actions，3 primary VLM calls。
- Offline full-trajectory prompt：成功，3 actions，3 primary VLM calls。
- MobileExplorer shadow：成功，3 actions，2 primary VLM calls；最后一次 answer 由绑定到当前 calendar state 的 live evidence 直接提交。

这只证明实现链路和 freshness 区分成立。单次总 step latency 基本相同，且该实验尚未包含 validated replay，因此不能声称端到端优于 trajectory-memory 工作。

随后加入 state-matched validated replay，在另一个 seed 上成功复用前两项导航 action，只调用最后 1 次 VLM 即正确回答。这个结果说明完全重复的三步 Calendar 任务更适合 offline replay，不支持 MobileExplorer 的必要性。下一阶段必须寻找 replay 无法覆盖的 gap；若找不到，项目应转向 replay/memory，而不是继续 online-exploration 主线。

`NotesMeetingAttendeeCount` 提供了更合理的 gap：历史知道打开 Joplin、进入搜索、打开结果，但当前 meeting title 与 attendee count 都变化。使用 value-free accessibility schema 做 replay admission，并把当前 goal 中的 title 编译进 shadow probe 后，同一 task seed 的一次机制对照为：

- Validated replay：成功，7 actions，5 primary VLM calls，runtime 31.34s。
- Validated replay + MobileExplorer：成功，7 actions，4 primary VLM calls，runtime 28.29s；shadow 从当前 episode 取得 `70`，primary 在匹配的未来 note state 跳过最终 VLM。

这是比 Calendar 更符合论文假设的实例：online arm 相对强 replay baseline 仍省 1 次 future reasoning。它仍然只有一个 paired seed，不能报告统计结论；需要扩大到多个 seed 和至少另一个 app family。

系统实验还发现两个直接的 failure mode：第一，dHash 会把 Joplin 中不同页面误判为相同页面，曾导致 replay 错误顺序；长期 replay admission 已改为去除动态值的 accessibility/layout schema，dHash 只保留为短时视觉辅助。第二，两个 `-read-only` emulator 若共享同一 AVD backing，Joplin 并发初始化会触发 SQLite `file is not a database`；真正成功的 run 使用了独立的 APFS copy-on-write AVD userdata。这说明 dual process/Back 并不等于 storage isolation。
