# AndroidWorld MobileExplorer：Executable GUI Memory 实现 Prompt

你现在在以下 AndroidWorld 代码库中工作：

`/Users/huangrunxi/Projects/android_world`

请直接实现和验证，不要只写设计文档，也不要只增加日志。目标是把现有 MobileExplorer 打通为真正可执行的 GUI memory：

> 在线建图 + 图检索 + exploration 使用图 + reasoning prompt 增强 + action 后融合 + 高置信 shortcut + 实时验证 + 持久化。

必须复用 AndroidWorld 现有 runner、LabServer、Ex5 agent、action parser 和 evaluator。不要另起一套无法接入评测的 runner。

## 1. 首先阅读现有链路

优先阅读以下文件：

- `android_world/agents/mobileexplorer.py`
- `android_world/agents/explorer_agent_gelab_light.py`
- `android_world/agents/explore_agent_text.py`
- `android_world/parallel_exploration/executable_memory.py`
- `android_world/parallel_exploration/live_probe.py`
- `android_world/parallel_exploration/agent_evidence_graph.py`
- `android_world/parallel_exploration/state.py`
- `android_world/parallel_exploration/rankers.py`
- `scripts/run_sensys30_online_task.py`
- `scripts/run_mobileexplorer_full_suite.py`
- `results/4B_2.txt`
- 最新 MobileExplorer v40 实验目录、汇总报告和失败报告

先画出当前真实调用链：

```text
页面观察
  -> state identity / dedup
  -> exploration candidate selection
  -> graph update
  -> reasoning prompt construction
  -> LabServer model call
  -> GELAB action parser
  -> graph retrieval
  -> action fusion / coordinate relocation
  -> action execution
  -> post-transition observation
  -> graph validation / persistence
```

不要假设某个函数已经完成了上述功能；必须通过代码和日志确认。

## 2. 当前最佳结果和核心问题

当前以 Claude 侧已有实验中表现最好的 **v40** 作为参考，不要引用更早的实验版本作为主要基线。

当前问题不是“完全没有图”，而是图没有形成闭环：

1. graph 可以生成节点和边，但 retrieval、prompt、fusion、shortcut 没有完整连通。
2. exploration 会受到大量安全过滤，无法稳定完成 5 个 probe。
3. graph candidate 被检索到，但很少真正影响 reasoning action。
4. 高置信 shortcut 基本没有执行，图没有转化为可执行路径。
5. dynamic 标记可能过于粗糙，系统时钟、status bar 或动态文字可能把整页判为不可复用。
6. parser 与 GELAB 输出格式仍存在不一致，需要和 `explorer_agent_gelab_light.py` 完全对齐。
7. task-level metrics 与 global metrics 存在累计快照混用问题。

最终目标不是让图拥有更多节点，而是让以下链路真实发生并且可审计：

```text
实时页面
  -> 找到当前 graph state
  -> 检索 task-relevant path
  -> exploration 选择未验证或低置信 frontier
  -> 记录 transition evidence
  -> 将压缩后的 path 注入 reasoning prompt
  -> 解析 reasoning action
  -> selector relocation / action fusion
  -> 条件满足时执行 verified shortcut
  -> 验证目标 state
  -> 更新 confidence、hit/miss 和 graph
```

## 3. 不可违反的约束

### 3.1 保留 Ex5 作为主策略

不能改变以下 baseline 行为：

- baseline reasoning prompt 的核心语义
- Ex5 action parser
- Ex5 action 执行逻辑
- Ex5 安全过滤逻辑
- 原有候选过滤和恢复策略
- AndroidWorld evaluator 和 runner

新机制只能作为 sidecar：

- exploration sidecar
- persistent graph memory sidecar
- prompt context sidecar
- post-fusion sidecar
- optional verified shortcut sidecar

任何 graph 异常、解析异常、恢复异常都必须 fallback 到原始 Ex5 reasoning/action。

### 3.2 禁止引入复杂模型

不要使用：

- GNN
- embedding model
- LLM 图总结
- ONNX Runtime
- 新的模型服务
- 新的 evaluator
- 另一个独立 runner

图信息必须是轻量、可序列化、可执行的结构化 memory。

### 3.3 reasoning 和 exploration 互相不可见

reasoning 和 exploration 必须从同一个初始页面开始。

- exploration 不能读取 reasoning 的 thought、候选 action 或最终 action。
- reasoning 不能读取 exploration 的中间结果。
- 只有 reasoning action 解析完成后，才允许进行 post-fusion。
- 每个 reasoning step 的 exploration 预算目标为 5 个 probe。
- 候选不足、安全恢复失败或没有安全候选时可以提前停止，但必须记录停止原因。

## 4. 统一的 GraphMemory API

不要让 `mobileexplorer.py`、`live_probe.py`、prompt builder 各自维护一份图逻辑。

请将现有 `executable_memory` 重构或扩展为统一服务，至少提供以下接口：

```python
memory.observe_page(page, task=None) -> StateNode
memory.retrieve_paths(page, goal, top_k=3) -> list[PathEvidence]
memory.plan_exploration(page, goal, candidates, budget=5) -> ProbePlan
memory.record_probe_transition(record) -> EdgeEvidence
memory.record_reasoning_transition(record) -> EdgeEvidence
memory.build_prompt_context(page, goal) -> str
memory.fuse_reasoning_action(action, page, goal) -> FusionDecision
memory.find_verified_route(page, goal) -> Route | None
memory.execute_verified_route(route, live_page) -> RouteResult
memory.record_route_result(route, hit, reason)
memory.begin_task(task_id)
memory.end_task(task_id, success)
memory.save()
memory.load()
```

所有 exploration、reasoning、fusion、shortcut 事件必须写入同一份 canonical graph。

## 5. Graph schema

### 5.1 StateNode

至少包括：

```json
{
  "node_id": "...",
  "schema_version": 3,
  "package": "...",
  "activity": "...",
  "structural_signature": "...",
  "semantic_signature": "...",
  "landmarks": [],
  "elements": [],
  "observation_count": 0,
  "visit_count": 0,
  "dynamic_elements": [],
  "dynamic_content": false,
  "parent_state_ids": [],
  "incoming_action_tokens": [],
  "depth": 0,
  "semantic_aliases": [],
  "reversible": false,
  "return_cost_s": 0.0
}
```

动态内容必须是 element-level 或 landmark-level 信息，不能因为一个系统时钟、通知、动态标题就把整页变成不可复用。

需要保存 `dynamic_reasons`，以便报告中解释某个 state 或 edge 为什么被判定为 dynamic。

稳定 `resource-id` 的控件，即使旁边有动态文字，也应继续允许 selector relocation。

### 5.2 EdgeEvidence

至少包括：

```json
{
  "edge_id": "...",
  "source_node": "...",
  "selector": {
    "resource_id": "...",
    "text": "...",
    "content_desc": "...",
    "class_name": "...",
    "relative_region": [0, 0, 1, 1],
    "bbox": [0, 0, 100, 100],
    "dynamic_text": false
  },
  "action_type": "click",
  "normalized_action_token": "...",
  "function": "...",
  "target_states": {
    "state_id": 3
  },
  "support_count": 0,
  "meaningful_count": 0,
  "no_op_count": 0,
  "external_count": 0,
  "trap_count": 0,
  "recovery_success_count": 0,
  "recovery_failure_count": 0,
  "task_success_count": 0,
  "alpha": 1.0,
  "beta": 1.0,
  "total_cost_s": 0.0,
  "provenance": {
    "exploration": 0,
    "inference": 0
  },
  "dynamic": false,
  "route_hit_count": 0,
  "route_miss_count": 0
}
```

`confidence` 至少应结合：

- Beta posterior
- landing distribution entropy
- support count
- meaningful/no-op/trap 比例
- recovery success rate
- task success
- selector ambiguity
- dynamic status
- execution cost

不要只使用 `support_count`。

### 5.3 K-step sample

每次 probe 都保存：

```json
{
  "source_state": "...",
  "future_state": "...",
  "k": 1,
  "first_action": {},
  "action_sequence": [],
  "target_states": [],
  "return_action": {},
  "elapsed_s": 0.0,
  "meaningful": true,
  "hard_negative": false,
  "evidence": {},
  "provenance": "exploration"
}
```

必须保留以下 hard negatives：

- no-op
- unstable landing
- recovery failure
- trap
- external navigation
- invalid selector
- 失败路径

## 6. State dedup

状态合并必须分两阶段。

### 第一阶段：粗召回

使用：

- package
- canonical activity
- structural signature
- 可交互控件 skeleton

### 第二阶段：细判定

使用：

- stable landmarks similarity
- clickable element matching
- resource-id matching
- class matching
- relative region matching
- semantic token matching

动态文本不能单独生成新节点。

必须增加测试：

- 相同页面不同 clock 文本应合并
- 相同 resource-id 但不同动态 label 应合并
- 页面结构真正变化时应分裂
- 动态列表内容变化不应造成无限节点
- icon-only selector 不能因为绝对坐标变化而失配

## 7. Exploration 使用图

每个 reasoning step 最多 5 个 probe，目标是完成 5 个。

只有以下情况允许提前停止：

- 没有安全候选
- 所有候选都被判定为 trap/dynamic
- recovery 失败
- 候选全部 no-op
- graph frontier 已耗尽
- 资源或时间预算耗尽

每次提前停止必须记录：

```json
{
  "stop_reason": "...",
  "probes_completed": 2,
  "probe_target": 5,
  "candidate_count": 10,
  "safe_candidate_count": 0,
  "rejected_candidates": [],
  "recovery_status": "RESTORED"
}
```

探索 utility 仍然以 Ex5 为主：

```text
utility =
    ex5_information_gain
  + task_relevance
  + navigation_prior
  + uncertainty_bonus
  + novel_delta_bonus
  + small_graph_tiebreak
  - side_effect_risk
  - return_cost
```

graph 只能是小的 tie-break/novelty 项，不能替换 Ex5。

每个页面的候选至少分成：

1. unseen frontier
2. seen but low-support
3. high-entropy landing
4. known no-op
5. trap
6. dynamic
7. mature stable edge

选择规则：

- 优先 task-relevant 的 unseen frontier
- `support < 2` 的边允许重复验证
- landing entropy 高的边允许重复验证
- 可逆性未知但风险低的边允许验证
- 同一轮不重复同一 selector
- mature 且稳定的边不重复探索
- 不得使用 coverage 或未访问数量作为主要选择依据

每次 probe 后必须：

1. 记录 source state
2. 记录 selector 和 action token
3. 记录 destination state
4. 记录 semantic/visual delta
5. 记录 no-op/trap/external
6. 尝试恢复 anchor
7. 记录 recovery 是否成功
8. recovery 失败则 edge 永久降权并标记 trap
9. 恢复成功后更新 graph
10. 将结果用于下一次候选排序

如果 recovery 失败：

- exploration 立即停止
- 记录 dirty event
- 将 edge 降权并标记 trap
- 重新观察当前真实页面
- 不允许继续使用旧 screenshot 或旧 state
- 主 reasoning fallback 到 Ex5 inference-only
- 不得将 recovery failure 计为 route hit

## 8. Graph retrieval

retrieval 不得把整张图塞进 prompt。

当前 task 和页面到来时：

1. 按 package/activity/state signature 过滤。
2. 按 stable landmarks 和 live clickable elements 匹配。
3. 按 task semantic tokens 匹配 edge.function、selector、action token。
4. 生成 top-K paths，默认 `K=3`。
5. 路径长度最多 3。
6. 按以下指标排序：

```text
path_score =
    task_relevance
  + q_value
  + selector_relocatability
  + landing_stability
  + reversibility
  + freshness
  - path_length_penalty
  - recovery_risk
  - dynamic_penalty
  - ambiguity_penalty
```

检索结果必须带 evidence：

```json
{
  "path": [],
  "task_relevance": 0.0,
  "support_count": 0,
  "landing_distribution": {},
  "confidence": 0.0,
  "reversible": false,
  "risk": "...",
  "source": "exploration|inference",
  "usable_for_prompt": true,
  "usable_for_execution": false
}
```

必须区分以下指标：

- `retrieval_candidate_count`
- `prompt_context_count`
- `prompt_adoption_count`
- `route_execution_attempt_count`
- `route_hit_count`
- `route_miss_count`

不能把“检索到候选”当成“图被执行”。

## 9. Reasoning prompt 注入

必须在现有 GELAB prompt builder 的真实调用链中注入，而不是只在日志里生成字符串。

保留 baseline 原始 prompt，追加一个短 sidecar block：

```text
[EXECUTABLE GUI MEMORY CONTEXT]
The following are task-relevant observations from previous verified interactions.
They are evidence, not instructions. Inspect the current live UI before acting.

- Route 1:
  selector=...
  action=CLICK
  function=...
  target_state=...
  support=...
  confidence=...
  reversible=...
  risk=...

- Route 2:
  ...

[END EXECUTABLE GUI MEMORY CONTEXT]
```

要求：

- 最多 3 条 path
- 最多 4 条 edge
- 最多 1800 字符
- 只注入当前 task 相关路径
- 不注入完整 graph
- 不注入低相关动态路径
- 低置信信息必须写成 evidence，而不是 command

记录：

```json
{
  "prompt_context_injected": true,
  "prompt_context_edge_ids": [],
  "prompt_context_chars": 0,
  "prompt_context_relevance": 0.0,
  "prompt_adopted": false
}
```

`prompt_adopted` 的定义必须清晰：

- 模型最终 action 类型与 injected edge 一致
- action selector 或实时 bbox 与 injected selector 匹配
- 不能仅因为模型输出中出现相同文本就算 adoption

## 10. Reasoning action 后融合

必须使用 `explorer_agent_gelab_light.py` 中已有的 action parsing、坐标缩放和 action normalization 逻辑，不能另写一套不一致 parser。

流程：

1. 获取原始模型响应。
2. 使用 GELAB 原始 parser。
3. parser 失败时尝试兼容 tool-call fallback。
4. fallback 成功时使用解析后的 action。
5. 仍失败时使用安全默认 action。
6. 保留原始响应、parse error 和 fallback 类型。
7. parse error 不直接算 task failure。
8. 真正执行失败才算 task failure。

### 10.1 坐标吸附

如果 reasoning 坐标落在当前可交互控件 bbox 附近：

- 根据 resource-id/text/content-desc/class/region 匹配 selector
- 将坐标吸附到实时 bbox 中心
- 记录 `coordinate_corrected=true`

### 10.2 Graph 一致

reasoning action 与 graph edge 一致时：

- 提高 action confidence
- 记录 `source=graph_consistent`
- 不需要覆盖原始 action

### 10.3 Graph 冲突

只有全部满足以下条件才允许覆盖 reasoning：

- lexical/semantic task relevance 明确
- 当前 source state 匹配
- selector 可实时重定位
- selector 不歧义
- edge confidence >= 0.82
- 至少 3 次一致 landing，或 `task_success_count >= 1`，或已有成功 route hit
- edge 可逆
- target 核心结构非动态
- 无 trap
- 无 side-effect risk
- action 类型兼容

否则：

- 保留原始 reasoning action
- 不覆盖
- 记录 disagreement
- 记录 graph rejection reason

graph 不得覆盖：

- TYPE 输入内容
- COMPLETE/status
- OPEN_APP
- 高风险删除、支付、发送、权限操作
- 当前页面没有明确 selector 对应关系的 action

## 11. High-confidence verified shortcut

只有全部满足以下条件才允许跳过模型 reasoning：

- task 与 route 有明确 semantic match
- source state 匹配
- path length <= 3
- selector 可实时重定位
- 每条 edge confidence >= 0.82
- 每条 edge 至少 3 次一致 landing，或已有成功 route hit
- 每条 edge 可逆
- 目标核心结构稳定
- 无动态核心控件
- 无歧义 selector
- 无 trap
- 无 route miss
- action 类型是安全导航动作
- 不是 delete/send/pay/permission/camera/microphone/login 等高风险动作

shortcut 每跳都必须：

1. 重新读取实时 UI
2. 重新定位 selector
3. 执行 action
4. 验证 target state
5. 目标 state 不匹配时立即停止
6. 记录 `graph_route_miss`
7. 尝试回退到 anchor
8. 回退失败时停止 shortcut，进入 Ex5 fallback

shortcut 成功后记录：

```json
{
  "reasoning_mode": "high_confidence_skip",
  "route_hit": true,
  "route_edge_ids": [],
  "route_length": 2,
  "memory_confidence": 0.91
}
```

shortcut 失败绝不能直接导致任务失败，必须回退 Ex5。

## 12. Persistence

图必须持久化到：

```text
evaluation_results/.../E_FULL_MOBILEEXPLORER/executable_memory.json
```

要求：

- schema version
- atomic write
- 兼容旧 schema
- unknown fields ignored
- missing fields 使用安全默认值
- corrupt file 自动备份并从空图恢复
- task boundary 不清空 graph
- 只清空 task-local counters
- global graph counters 与 task counters 分离

修复当前 task metric 累计问题：

- `executable_memory.json.metrics` 保存 global 累计指标
- 每个 task summary 保存 task 开始前后的 delta
- 不能把累计 snapshot 当成 task-local metrics
- 报告同时展示 global metrics 和 task-local metrics

## 13. Metrics

必须记录以下指标。

### Exploration

- probe target
- probes completed
- 5-probe completion rate
- stop reason
- no-safe-candidate count
- repeated validation count
- recovery failure count
- no-op count
- trap count
- unique new elements
- unique new pages
- exploration latency

### Graph

- node count
- edge count
- skill/action-group count
- state merge count
- dynamic state count
- hard negative count
- support distribution
- confidence distribution
- retrieval candidate count
- prompt context count
- prompt adoption count

### Reasoning/Fusion

- reasoning count
- parse error count
- parser fallback count
- coordinate correction count
- graph consistent count
- graph disagreement count
- graph rejection count
- graph override count
- route shortcut attempts
- route hit/miss
- actual skipped reasoning count

### Task

- success rate
- successful task steps
- mean/median/P95 steps
- mean latency
- extra exploration time

## 14. Tests

必须新增或修复以下测试：

1. state dedup
2. dynamic clock/status bar 不导致 graph 爆炸
3. dynamic text 与 stable resource-id 的 selector relocation
4. landmark + element matching
5. hard negative 记录
6. no-op edge 降权
7. trap edge 永久降权
8. support < 2 时重复验证
9. landing entropy 高时重复验证
10. exploration 使用 graph frontier
11. exploration 不读取 reasoning 中间结果
12. K-step sample slicing
13. path Q-value
14. top-K retrieval task relevance
15. prompt compactness
16. prompt adoption logging
17. coordinate snapping
18. graph-consistent fusion
19. graph conflict 保留 reasoning
20. high-confidence route gate
21. shortcut hit
22. shortcut miss fallback
23. recovery failure fallback
24. parser tool-call fallback
25. parse error 不直接计 task failure
26. task-local/global metrics 分离
27. persistence schema compatibility

## 15. 验证顺序

先运行现有相关测试，并根据仓库实际文件调整测试路径：

```bash
pytest -q \
  tests/test_executable_memory.py \
  tests/test_mobileexplorer_chain.py \
  tests/test_online_exploration_runtime.py \
  tests/test_oracle_validation.py \
  android_world/parallel_exploration/executable_memory_test.py \
  android_world/parallel_exploration/live_probe_test.py
```

然后跑 3-task smoke test，必须人工检查：

1. graph 文件能生成并恢复
2. 新 state/edge 能被检索
3. prompt 中确实出现 graph context
4. reasoning action 能被 graph match
5. 坐标能被吸附到实时 bbox
6. 至少在 deterministic fixture 中执行一次 verified shortcut
7. shortcut miss 会安全 fallback
8. parser error 不被计为 task failure
9. recovery failure 会标记 trap
10. task-local metrics 正确

然后跑与 v40 使用相同任务集、seed、model、snapshot、step budget 和 Ex5 参数的对照实验。

至少比较：

- Ex5 baseline
- Ex5 + graph logging
- Ex5 + graph prompt
- Ex5 + graph fusion
- Ex5 + verified shortcut

主报告不能只报告“检索候选数”，必须报告真正的：

- prompt injection
- prompt adoption
- fusion
- override
- shortcut attempt
- shortcut hit
- shortcut miss

如果真实任务中 shortcut 仍然没有 hit，必须报告具体 gate 阻塞分布，而不是把 retrieval candidate count 当作 graph 已经生效。

不能伪造实验数据。

## 16. 最终交付

请输出：

1. 改动文件清单
2. 架构说明
3. 实际调用链说明：
   - 页面观察
   - state dedup
   - graph update
   - graph retrieval
   - exploration selection
   - prompt injection
   - parser
   - fusion
   - shortcut
   - verification
   - persistence
4. 关键 API 和数据结构
5. 单元测试结果
6. smoke test 结果
7. v40 同任务集对比结果
8. 与 Ex5 baseline 的逐任务差异
9. 典型成功 trace
10. 典型失败 trace
11. 每一步实际检索到的 graph context
12. graph route hit/miss 证据
13. 未解决问题

最重要的验收标准：

- exploration 能读取已有 graph，并根据 seen/unseen、support、landing entropy 和 task relevance 避免无意义重复探索。
- reasoning prompt 中真正出现相关 graph path。
- reasoning action 能被 graph selector 匹配和校正。
- 高置信 route 在满足全部条件时能够跳过模型 reasoning。
- shortcut 每一步都有实时 UI 验证。
- shortcut 失败能够安全回退 Ex5。
- parser error 不会破坏评测。
- 所有行为都有可审计 JSONL 日志、单元测试和 smoke test 证据。
- 不仅有“建图”和“检索”，还要证明图真正参与了 exploration、reasoning 和 inference execution。
