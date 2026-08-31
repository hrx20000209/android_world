# MobileExplorer 设计实现核验

## 结论

当前仓库没有一条执行路径完整实现论文中的端到端 MobileExplorer。

仓库里实际上存在三套不同层次的代码：

1. `parallel_exploration/processes.py`：早期 Phase-1 生命周期骨架。它有双进程、事件协议和 pluggable inference backend，但 Explorer 是 mock，不执行真实 UI probe。
2. `explorer_agent_gelab_light.py` / `explorer_agent_gelab_bandit.py`：较完整的旧 MobileExplorer 原型。它包含 reasoning prior、session memory、depth-2、replay、evidence score/type/injection 等代码，但大量逻辑来自固定英文关键词和 app/task 特例；并行方式是同进程线程，而且 VLM 返回后无条件等待 exploration future，没有 inference-authoritative ABORT。
3. `parallel_exploration/live_probe.py` + `run_one_parallel_exploration_task.py`：最近实测所用路径。它是真实双进程并行、真实 Android UI probe、真实 vLLM 和细粒度 latency trace，但只实现单层 feasibility probing，没有 reasoning prior、session utility、adaptive depth、trajectory replay或 evidence injection。

因此，“基础可行性 harness”大部分已经实现；“论文宣称的完整 MobileExplorer”没有集成完成。

## 论文设计逐项核验

| 设计项 | 旧 MobileExplorer agent | 当前真实并行 runner | 判断 |
|---|---|---|---|
| 推理与 UI exploration 并行 | 同进程 ThreadPool 并行 | 独立 Process B，与真实 vLLM 请求并行 | 已实现，但两条路径不同 |
| 推理进程是时间权威，结束即 ABORT | VLM 返回后直接等待 future；没有强制 ABORT | 有 ABORT，并测 `critical_path_extension_ms` | 仅 live runner 实现 |
| FIRST_TOKEN / DECODE_PROGRESS 生命周期 | Phase-1 backend 有 | live runner 只读取 vLLM metrics，没有逐 token 事件 | 未集成 |
| Previous-step reasoning prior | 有 template/regex parser 和 delayed prior | 无 | 原型存在，实测路径未使用 |
| Session-local exploration memory | 有 branch/session stats 和更新 | 只有当前窗口 `probed` set | 原型存在，实测路径未使用 |
| 四信号 node weight | 有多个 score component，但大量为手工启发式 | 只有 token overlap + role bonus | 未按论文完整落地到 live path |
| Adaptive breadth/depth | 旧 agent 有多种策略和可选 LB-MCTS/EWMA budget | baseline 候选单层循环，直到 ABORT 或候选耗尽 | 未集成 |
| 打开页面后重新解析并做 depth-2 | bandit/light agent 有 | live runner 只记录 post-state，不重新选后页候选 | 未实现于实测路径 |
| 风险识别和不可逆动作处理 | 原先主要依赖英文词和 app/task 特例；现默认关闭 | 原先依赖固定词/标签规则；现已移除 | 泛化安全机制缺失 |
| Level-1 depth-bounded Back | 有 | 有 INVERSE/BACK_N | 已实现 |
| Level-2 deterministic trajectory replay | 有 home/app anchor + logged action replay | 只有 `am start` DEEPLINK，不是 action-prefix replay | live path 未实现 |
| 恢复状态验证 | activity + pHash；部分 anchor overlap | activity + (struct 或 pHash)，三种信号均记录 | 屏幕对齐有实现，latent side effect 无法验证 |
| State-aligned evidence selection | 有 activity/pHash/anchor-label matching | 无 | 未集成 |
| Certainty score和 evidence type | 有 ANSWER/ACTION/AVOID/SCHEMA/RISK 等类型及固定阈值 | 无 | 原型存在，实测路径未使用 |
| Evidence 注入下一步 prompt | 有 pending trace 和 prompt context | summary/evidence 均未注入 | 未集成 |
| Prompt token overhead测量 | 有部分 prompt trace | 无 evidence，因此无对应 trade-off | 未完成论文所需实验 |
| Transition completion detector | 旧 agent 使用 AndroidWorld stability wait | live runner 主要是固定 100–150 ms settle + state capture | 未形成统一、自适应 detector |
| Thermal / MemAvailable 2 秒采样 | 未发现完整 live 集成 | 无 | 未实现 |
| 多 app config trial runner + reset | 有大量实验脚本，但不是统一 feasibility runner | 一次只跑一个 AndroidWorld task | 部分实现 |
| 同一手机上的推理/UI资源竞争 | 当前 Jetson + emulator/手机分离 | 当前 Jetson + emulator | 未验证 |

## 固定词汇规则处理

### 当前 live runner

已经删除：

- `BLOCKED_TERMS` 风险词表；
- `delete/send/pay/...` 等文本直接拒绝；
- `More options/menu/expand` 等标签直接准入；
- “Launcher 才允许普通导航、应用内只允许菜单”的固定规则；
- `More fields`、`Start` 等依靠字符串和 class 的临时修补。

`probe_type` 现在只依赖可观测节点状态和 role：checkable -> `EXPAND`，ImageButton/ImageView -> `TAP_MENU`，其他 clickable -> `TAP_NAV`。

仍保留的非词汇测量保护：

- 无 text/content-desc/resource-id 的节点不执行，因为当前扁平 A11y 无法建立可解释身份；
- scroll probe 暂停，因为实测证明等距离反向 swipe 不能精确恢复滚动位置；
- 恢复失败立即停止 trial 并标 dirty。

### 旧 MobileExplorer agent

新增 `ANDROID_WORLD_LIGHT_EXPLORE_LEXICAL_RULES`，默认 `False`。默认关闭：

- risk/commit/noise phrase 词表；
- app 名白名单和 Calendar 专用规则；
- media/account goal-irrelevant 词过滤；
- 基于任务措辞的 DELETE/MEDIA/INFO 等硬模式分类。

设置 `ANDROID_WORLD_LIGHT_EXPLORE_LEXICAL_RULES=1` 只用于复现历史实验。

## 审稿意见是否成立

成立，而且当前代码与实测进一步验证了这些问题：

1. 关键词风险过滤不泛化：icon-only、本地化文本、父子 A11y 语义分离都会绕过或误触规则。
2. pHash/activity 只能验证可见页面，不能验证购物车、已读状态、计时器、草稿、后台数据库等 latent side effect。
3. live runner 能诚实测量 inference 结束后的恢复延长，但没有提前估算可用时间、恢复 reserve 或 prompt evidence token cost。
4. 旧 agent 虽有 EWMA/LB-MCTS budget 代码，但默认策略不是 LB-MCTS，而且 VLM 返回后仍无条件等待 future；这不能证明“不增加关键路径”。
5. 当前 Jetson 与 Android UI 分离，不能回答单手机 GPU/内存/热竞争问题。

## 当前最关键的缺口

去掉关键词规则后，系统没有可泛化的 safety oracle。下一版不应再换一套更长的词表，而应把安全性建模为动作后果和可恢复性：

- 基于 action semantics / Android capability / intent destination 的风险描述，而不是 label；
- 执行前使用 app-state checkpoint、数据库/系统设置可观测 diff 或 sandboxed clone；
- 对无法 checkpoint 的 live app，仅允许经过历史恢复统计证明的 action schema；
- 将未知动作作为 `risk_unknown`，用于排序/测量，而不是用英文字符串直接断言 safe/unsafe；
- Level-2 必须是真正的 stable anchor + deterministic action-prefix replay；
- model action 只有在恢复完成并通过多源验证后才能执行；失败时必须重新 perception/reasoning。

在这套替代机制实现前，关闭词汇规则后的 active probing 只适合 AndroidWorld/emulator 或可重置测试账号，不应直接用于真实支付、消息、邮件或生产账号。

