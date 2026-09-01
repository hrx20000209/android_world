# MobileExplorer 真机版实现说明书

给实现者（Codex）的交接文档。目标：把现在跑在 AndroidWorld 模拟器上的原型，移植成一个**独立的、控制真实手机的程序**，能输入任意自然语言任务，并且可以分别开关"探索"和"建图"。然后跑两个实验。

本文档里凡是提到 `路径:行号` 的地方，都是**现有可直接复用的代码**，不要重写；提到"新写"的地方才需要实现。行号以本仓库当前状态为准，若有偏移按符号名查找。

---

## 0. 一句话说明这个系统在做什么

一个手机 GUI agent 每一步都要：截屏 → 调用视觉语言模型 → 得到一个动作 → 执行。**一次模型调用在慢后端上要 10 秒以上，而 agent 经常在重复解决它十分钟前已经解决过的问题。**

本系统做三件事：

1. **探索**：在模型思考的空档里，去点屏幕上别的控件，看看会跳到哪，然后**撤回原状**。
2. **建图**：把"真实执行过的转移"和"探索试出来的转移"攒成一张状态图。
3. **省推理**：当图对当前处境足够确定时，直接重放记忆中的动作，不调模型；不够确定但有有用信息时，把信息蒸馏成几十个 token 注入 prompt。

三个开关必须**互相独立**：`--exploration on/off`、`--graph on/off`、`--skip on/off`。关掉建图时探索仍可运行（只是结果丢弃），这正是实验二要对比的。

---

## 1. 目标程序的形态

```bash
python real_phone_agent.py \
    --task "Delete the recipes named Lentil Soup and Garlic Butter Shrimp from Broccoli" \
    --serial <adb设备序列号> \
    --api_url http://<host>:<port>/v1/chat/completions \
    --model GELAB-ZERO-4B \
    --max_steps 20 \
    --exploration on|off \
    --graph on|off \
    --skip on|off \
    --profile_memory on|off \
    --out_dir runs/<name>
```

**不要依赖 AndroidWorld。** 现在的原型通过 monkeypatch 挂在 AndroidWorld 的 agent 上（`scripts/run_serial_exploration_task.py:1520` 替换 `GELABAgent.step`），真机版必须把这层剥掉：没有任务成功判定、没有 `run.py` 的 episode 循环、没有模拟器 gRPC。取而代之是一个自己的主循环。

**唯一的设备接口是 adb。** 复用 `android_world/parallel_exploration/adb.py`（83 行，`AdbClient`，带超时和重试一次）。所有设备操作都走它，构造时传 `serial`。

---

## 2. 直接复用的模块（不要重写）

| 模块 | 路径 | 作用 |
|---|---|---|
| adb 封装 | `android_world/parallel_exploration/adb.py` | 带超时/重试的 adb 调用 |
| 屏幕状态与签名 | `android_world/parallel_exploration/state.py` | 见下 |
| 信念图 | `android_world/parallel_exploration/belief_graph.py` | 节点/边/统计/熵 |
| 探索与回滚 | `android_world/parallel_exploration/live_probe.py` | 探测、恢复阶梯、候选筛选 |
| 逆动作 | `android_world/parallel_exploration/recovery.py` | 逆动作规划与轨迹重放 |
| 图蒸馏与三路门 | `android_world/parallel_exploration/graph_distiller.py` | 事实提取、prompt 注入、门控 |
| 候选信息矩阵与打分 | `android_world/parallel_exploration/state_graph_information.py` | 探测候选排序 |
| 推理需求解析 | `android_world/parallel_exploration/information.py` | 从模型上一步输出里提取它在找什么 |

### 2.1 屏幕状态（`state.py`）

- `FastSocketUiTreeProvider`（`state.py:108`）：通过常驻 socket 从设备上的 AccessibilityService 取控件树。**这是性能关键**——比 `uiautomator dump` 快约 125 倍（单次 8ms vs 1000ms）。需要先装 APK 并建立端口转发，见 §3。
- `UiAutomatorDumpProvider`（`state.py:90`）：兜底路径，无需装 APK，但每次约 1 秒。真机首次接入时先用它跑通，再切 socket。
- `AdbScreenshotProvider`（`state.py:326`）：`exec-out screencap -p`，返回灰度图。**真机版要改成彩色**——模型需要彩色截图，现在的实现为了算感知哈希用了 `IMREAD_GRAYSCALE`。请分成两路：感知哈希用灰度，送模型的截图用彩色。
- `StateSignature`（`state.py:41`）与三个签名：
  - `layout_sig`（`state.py:49`）：**宽松的屏幕身份**，只保留可交互控件的 class + resource_id + 粗粒度位置（64px 网格），不含文字，且**重复控件只记种类不记位置**。用于节点 id。
  - `struct_sig` + `phash`：**严格的状态验证**，用于回滚后判断"是不是我离开时的那个状态"。
  - 这两者是**不同的问题**，不要用同一个签名回答。

### 2.2 信念图（`belief_graph.py`）

- 节点 id：`ProgressiveBeliefGraph.make_node_id(activity, layout_signature)`（`:191`）
- 边 id：`canonical_action`（`:34`）+ `hash(源节点, 规范化动作)`，在 `add_speculative_transition`（`:218`）里算。

**两条踩过坑的关键设计，移植时务必保留：**

1. **边的身份不含目的地。** 曾经是 `hash(源, 动作, 目的地)`，结果模型在同一屏三次点同一控件、每次落点不同，被记成三条各执行一次的边，"模型在这里做过这件事吗"永远读出 1。改完后跳过命中率从 18% 升到 86%。目的地存在 `GraphEdge.observed_destinations`。
2. **边的身份不含坐标。** 模型输出归一化坐标会抖（同一按钮记成 `(540,1063)×6` 和 `(540,1068)×5`）。用 `control_key`（`belief_graph.py:63` 的 `control_key_from_identity`）= `resource_id|text|content_desc|class`，**不含 bounds**。把模型给的坐标解析成命中的控件：`scripts/run_serial_exploration_task.py:371` 的 `_control_key_at`（取包含该点的**最小**元素；若该控件除类名外无任何标识，用它自己中心点按 96px 网格锚定）。

- 统计写入统一走 `record_probe` / `record_inference_alignment` / `record_execution_verification` / `record_skip_result` / `record_rollback_result`（`:286`–`:332`），不要在别处直接改字段。

### 2.3 探索（`live_probe.py`）

- 候选筛选：`_safe_candidates`（`:270`）——硬安全过滤，剔除无标识控件、动作按钮、不可逆选择控件等，被剔除的写进 `filtered_elements.jsonl` 并附原因。
- 探测类型判定：`_probe_type`（`:485`），只允许 `TAP_NAV` / `SCROLL`（`TAP_MENU`/`EXPAND` 因回滚失败率 16%/23% 已停用）。
- 执行探测动作：`_adb_action`（`:521`）。
- **回滚阶梯**：`_recover`（`:640`），`NOOP → INVERSE → BACK_N → BACK_OVERLAY → DEEPLINK → TRAJECTORY_REPLAY → FAILED`，每级后重新截图并严格比对。
- 探索进程入口：`spawn_explorer` / `await_explorer_ready` / `run_serial_exploration` / `stop_prepared_explorer`（`:1485`–`:1596`）。现在探索跑在**独立子进程**里，通过一个 config dict 通信。

**真机移植的关键改动：**

- `_recover` 里的 `BACK_N` 按数是**栈深差**，绝不能有 `max(1, ...)` 的下限——从应用根 activity 按 Back 就是退到桌面。任何一级把包名走离目标应用，立刻 `am start` 重新进来再继续（`live_probe.py` 里的 `_left_the_app` / `_reenter_app`）。
- `TRAJECTORY_REPLAY`（`recovery.py:91`）会先按 HOME 再重放已提交动作。若某个 `open_app` 解析不到包名就**必须中止**，否则后面的盲点坐标会落在桌面图标上（实测出现过"给记账应用回滚结果落到 YouTube"）。真机上应用包名表要按实际设备重建。

### 2.4 蒸馏与门控（`graph_distiller.py`）

- `GraphDistiller.distill(...)`：确定性五段 `retrieve → filter → score → compress → template`，**不调用任何模型**。只看当前节点 1 跳出边，最多 3 条事实、≤64 token。
- `ReasoningGate.decide(...)`：返回 `NORMAL_INFERENCE` / `GRAPH_ENHANCED_INFERENCE` / `SKIP_INFERENCE`。
- 输出格式固定：

```
[Memory]
Need: Streaming Services, Pet Supplies.
Verified: Streaming Services -> {Expense Detail, Pet Supplies, Amount, 190.95}.
```

**四条不能违反的措辞约束**（都是实测逼出来的）：陈述而非建议（写成 "Tapping X opens Y" 会被模型当推荐并反复点同一控件）；没有正面事实（VERIFIED/OBSERVED）就完全不注入；坐标不能当控件名；不做覆盖度不支持的否定断言（说"在 X 下未观察到相关证据"，不说"X 里没有电话号码"）。

---

## 3. 设备侧准备

### 3.1 快速 a11y 通道（强烈建议）

`tools/fast_a11y_dumper/` 是一个 AccessibilityService，在设备上开一个 `LocalServerSocket`，宿主机通过 `adb forward` 连上去持续取控件树。

```bash
python tools/fast_a11y_dumper/build_apk.py          # 构建
adb -s <serial> install -r <apk>
# 在设置里手动开启该无障碍服务
adb -s <serial> forward tcp:8765 localabstract:androidworld_fast_a11y   # 注意是 localabstract，不是 tcp:tcp
```

然后 `FastSocketUiTreeProvider(adb, local_port=8765)`。**真机上无障碍服务需要用户手动授权**，程序启动时要检测并给出清晰提示，不要静默退化。

若不装 APK，用 `UiAutomatorDumpProvider` 兜底，但要在日志里明确标注，因为它会让每步多花约 1 秒，会污染实验一的 profile 结果。

### 3.2 真机与模拟器的差异（必须处理）

- **屏幕分辨率不固定**。模型输出的是 0–1000 归一化坐标，要按真机实际分辨率换算。参考 `android_world/agents/gelab_agent.py:320` 的 `_norm_to_abs`。
- **没有 `am start -n <component>` 的通用保证**：部分应用的 activity 不导出，`DEEPLINK` 这一级可能直接失败。要允许它失败并往下走，不要抛异常。
- **应用列表要从设备实时获取**（`adb shell pm list packages -3` + label 解析），不能沿用 `gelab_agent.py:37` 的 `AVAILABLE_APPS` 硬编码表。任务文本里的应用名匹配逻辑见 `scripts/run_serial_exploration_task.py` 的 `_target_app_from_goal`（含单复数归一、最长匹配优先）。
- **真机有通知、来电、锁屏**。状态栏噪声过滤见 `live_probe.py` 的 `_is_app_content`（已处理时钟、电量、信号、`<应用> notification: <内容>` 这类框架固定措辞）。真机上还要额外处理锁屏和权限弹窗。

---

## 4. 主循环规格（新写）

替代 AndroidWorld 的 episode 循环。伪代码，注意**顺序和时序边界**：

```
graph = ProgressiveBeliefGraph(task_id)          # --graph off 时用一个 no-op 实现
guarded = GenerationGuardedGraph(graph)          # 见 §4.1

for step in range(max_steps):
    before = capture()                            # a11y 树 + 截图 + 三个签名
    src_id = node_id_of(before)
    upsert_node(before, src_id, visited=True)

    # (1) 首步应用启动折叠：任务文本点名了应用就直接打开，不单独计一步
    if step == 0 and 在桌面 and 任务文本匹配到已安装应用:
        execute(open_app)
        before = settled_capture()                # 轮询 layout_sig 直到稳定，上限 2s
        src_id = node_id_of(before); upsert(...)
        # 不 return，继续在同一步里推理

    need = parse_reasoning_prior(上一步模型输出, task)   # information.py:104

    # (2) 三路门，只读 step-1 的快照
    snapshot = guarded.snapshot(for_step=step)
    decision = gate.decide(src_id, snapshot, need, reusable_edge, ...)

    if decision.mode == SKIP and --skip on:
        执行记忆中的动作; 立即比对落点; 记 record_skip_result
        guarded.commit_step(); continue           # 计为一步，未调模型

    # (3) 先起探索子进程（它的启动开销藏在推理里）
    if --exploration on and 允许探测(见 §4.2):
        pending = spawn_explorer(config)

    # (4) 推理。注入内容来自 step-1 快照
    prompt = build_prompt(task, history, screenshot, decision.graph_context)
    output = call_vlm(prompt)                     # 见 §5
    action = parse_action(output)
    execute(action)

    # (5) 探索在推理之后跑，但它的基线状态是 spawn 时抓的 S_i
    if pending: outcome = run_serial_exploration(pending); ingest(outcome)

    # (6) 记录权威边，然后让本步的写入对下一步可见
    after = capture()
    if node_id_of(after) != src_id:
        edge_action = dict(action); edge_action["control_key"] = _control_key_at(before, action)
        edge = graph.add_speculative_transition(src_id, edge_action, node_id_of(after), ...)
        graph.record_execution_verification(edge.edge_id, True)
        edge.nodes_at_last_execution = len(graph.nodes)
    guarded.commit_step()
```

### 4.1 时序边界（必须实现）

`GenerationGuardedGraph`（`scripts/run_serial_exploration_task.py` 内，约 220–290 行）：图带世代号，**step *i* 只能读到 step *i-1* 结束时的快照**，越界抛 `GenerationError`。

这不是防御性编程。并行设计里探索和推理**同时**发生，模型从构造上看不到当前步的探索结果；串行执行会把结果放进内存、就在推理调用之前，此时唯一阻止泄漏的只有自觉。**移植时必须把这个守卫一起带过去**，否则所有"探索有没有用"的结论都不成立。

快照是**拷贝**，不能事后长出新边。探索子进程拿到的也是序列化副本。

### 4.2 探索的准入条件（都是实测得出，逐条保留）

- 前两步不探测（开局图为空，扰动却落在模型即将读的屏幕上）
- **只在 `node_visits >= 2` 的屏幕上探测**（探索学到的东西只在该屏幕再次出现时才值钱）
- **屏幕上有已输入文本时不探测**（Back 还原不了输入内容，整个阶梯都救不回来）
- `TAP_NAV` 需要该屏幕已有一次**精确恢复**（`NOOP`/`INVERSE`）才允许；`SCROLL` 随时可用
- **本局一旦出现"未复原"或"被甩出应用"，此后不再探测**
- 单 episode 探测总量上限（原型是 12）

依据（模拟器实测）：完全不探测的 episode 成功率 89%，探测且每轮都复原 89%，**探测且有一轮未复原 67%**。

### 4.3 跳过的准入条件

- 屏幕访问过 ≥2 次
- 模型在该屏幕上**只选过一个动作**（出度 1、决策熵 0）
- 该动作**被模型执行过 ≥2 次**
- **两次执行之间图长出过新节点**（任务在迭代，不是原地打转）
- 该边本局重放不超过 2 次
- 落点不在最近 4 个节点内（对模型自己重复过的边豁免此条）
- 连续跳过 < 3；本局跳过命中率 ≥ 50%

执行后**立即比对落点**，校验对象是 `observed_destinations` **集合**（重复动作本就会落到不同屏幕）。不符则作废该边及子树、停止跳过、交还模型。

> **重要提醒**：在模拟器上，跳过机制目前是**净亏损**——三次独立跑测里，有跳过的任务成功率明显低于对照（1/17 vs 6/17）。探测和注入是中性的。所以 `--skip` 默认应为 **off**，真机实验先不要开，除非专门做这一项的对比。

---

## 5. 模型调用（新写）

原型通过 AndroidWorld 的 agent 间接调用，真机版要自己实现一个薄封装：

- OpenAI 兼容的 `POST {api_url}` ，body 里 `model`、`messages`、`max_tokens`
- messages 里图片用 data URL（`gelab_agent.py:295` 的 `_image_to_data_url`）
- 动作解析：`gelab_agent.py:210` 的 `_normalize_tool_call` 及其周边（`_extract_tool_call_payload`、`_parse_point`、`_norm_to_abs`）可以直接搬，它处理了模型输出里的 code fence、坐标格式、方向推断等一堆脏情况
- 支持的动作至少要有：`click(x,y)`、`input_text`、`scroll(direction)`、`navigate_back`、`navigate_home`、`open_app(name)`、`long_press`、`wait`、`status(complete/infeasible)`

注入的图信息拼在 prompt 里的位置参考 `scripts/run_serial_exploration_task.py` 的 `build_with_briefing`（约 1000 行处），追加在用户消息末尾。

---

## 6. 实验一：内存占用 profile

**要回答的问题**：模型后台推理、建图、探索各自占多少内存，同时跑会不会互相挤。

**测量对象分四组，每组跑同一个任务 3 次**：

| 组 | exploration | graph | skip | 说明 |
|---|---|---|---|---|
| A 纯推理 | off | off | off | 只有主循环 + 模型调用 |
| B 推理+建图 | off | on | off | 加上图的构建与维护 |
| C 推理+探索 | on | off | off | 探索跑但结果丢弃 |
| D 全开 | on | on | off | |

**要采集的指标**（宿主机侧，采样间隔 200ms，全程时间序列 + 峰值 + 均值）：

- 主进程 RSS / USS（`psutil.Process().memory_full_info()`）
- **探索子进程的 RSS**（单独统计，`spawn_explorer` 起的那个进程）
- Python 堆对象数（可选，`gc.get_objects()` 计数，或 `tracemalloc` 快照）
- **图本身的内存**：`len(graph.nodes)`、`len(graph.edges)`、`sys.getsizeof` 无意义，用 `pympler.asizeof` 或序列化后字节数（`json.dumps(graph.to_dict())` 长度）随步数的增长曲线
- 截图与 a11y 树的驻留：每步的 `pixels` 数组字节数、控件数

**如果模型跑在本机**（vLLM/llama.cpp），另外采：GPU 显存（`nvidia-smi --query-gpu=memory.used`）或 Jetson 上的 `tegrastats`；模型进程的 RSS。**如果模型在远端**，明确记录，这一项标为 N/A。

**设备侧也采**（每步一次）：
```bash
adb -s <serial> shell dumpsys meminfo <目标应用包名>
adb -s <serial> shell cat /proc/meminfo
```
因为探索会反复启停 activity，可能推高被测应用的内存。

**输出**：`memory_profile.jsonl`（每个采样点一行）+ 一张四组对比图（时间 × RSS，探索子进程单独一条线）+ 一张表（峰值、均值、图序列化大小随步数）。

**关键是要能回答**：图的内存是不是随步数无界增长？探索子进程的峰值有多高？四组之间主进程的差值是多少？

---

## 7. 实验二：端到端真机运行

**要回答的问题**：三个部件同时跑，会不会互相影响——相比于单独跑推理、单独跑探索、单独建图。

**同样四组（A/B/C/D），每组跑同一批任务**。任务集自己挑 10–15 个真机上可复现的，覆盖三类：单步查询型、多步表单型、**重复迭代型**（比如"删掉这三条记录"——这是渐进式记忆唯一能兑现的场景）。

**每一步都要落盘的结构化日志**（`run_events.jsonl`，一行一个事件）：

| 事件 | 字段 |
|---|---|
| `step` | 步号、当前节点 id、activity、模型动作、耗时分解（截屏 / a11y / 推理 / 执行） |
| `gate` | 三路模式、理由、决策时刻的边统计（`execution_hit_count`、`skip_replays`、`node_visits`、`node_entropy`） |
| `explore` | 本轮探测次数、每次的探测类型/深度/目标控件、**回滚用到的级别**、是否复原、耗时 |
| `graph` | 节点数、边数、本步新增节点/边、序列化字节数 |
| `graph_context` | 是否注入、token 数、原文、选中的事实类型与边 id |
| `skip` | 边 id、是否命中、第几跳、本局累计命中率 |
| `repair` | 是否被甩出应用、用什么方式捞回来（back / relaunch）、搁浅在哪 |

**汇总指标**：

- **成功率**（真机上需要人工判定或写任务专属校验，务必说清判定方式）
- **步数**：总步数、成功任务的步数
- **延迟**：每步端到端耗时，以及推理 / 探索 / 截屏 / a11y 的分解
- **探索**：总探测次数、每任务、每步；回滚成功率按 `(探测类型, 深度)` 分开统计；未复原轮数
- **建图**：最终节点数/边数、重访节点比例、图序列化大小
- **注入**：触发次数、平均 token
- **互相影响**：这是核心。至少要能回答——
  - C 组（开探索）相比 A 组，**每步端到端延迟**增加多少？增量是不是落在探索之后的那次截屏上？
  - D 组相比 B 组，探索是否让图长得更大（节点/边多多少）？多出来的是不是投机边？
  - 开探索后，**主循环的推理耗时**有没有变化（如果模型在本机，探索抢 CPU 会拖慢推理；如果在远端，不应该有影响）——这是判断"探索能不能藏进推理窗口"的直接证据
  - 探索导致的**设备侧扰动**：`repair` 事件次数、未复原轮数，以及它们和该任务成败的关系

**必须做的对照**：同一批任务在 A 组跑一遍作为基线。**不要拿别的环境/别的时间跑出来的历史结果做基线**——在模拟器上我们量到，同一份代码跑两次成功率会差 7 个任务（116 个任务里翻转 9 个），单次跑测分辨不出 ±7 以内的差异。真机上噪声只会更大，所以**每组至少跑 2 遍**，报告里给出两次的数值而不是只给均值。

---

## 8. 落盘与目录约定

```
runs/<name>/
  config.json              # 完整参数，含 git commit
  run_events.jsonl         # §7 的结构化事件
  memory_profile.jsonl     # §6 的采样
  belief_graph.json        # 结束时的图（graph.to_dict()）
  graph_steps/stepNN.json  # 每步的图快照（可选，用于复现生长过程）
  screens/<node_id>.png    # 每块屏幕首次见到时的截图
  probe_trace.jsonl        # 每次探测一行
  filtered_elements.jsonl  # 被安全门剔除的候选及原因
  prompts/stepNN.txt       # 送进模型的完整 prompt（调试用）
```

---

## 9. 移植时最容易踩的坑（都真实发生过）

1. **改完代码不冒烟就起批量跑**。有一次引用了未赋值的变量，每个任务第一步就抛异常、1.1 秒结束，几分钟刷出 33 个"完成"的空结果。**每次改动后先跑单个任务确认，再起批量。**
2. **批量跑到一半改源码**。新起的任务会用新代码，旧的用旧代码，数据被污染且事后无法分辨。**跑测期间冻结代码。**
3. **端口转发是 `localabstract` 不是 `tcp:tcp`**。写错了 socket 连得上但读不到东西。
4. **死掉的端口转发会静默无限超时**：端口还在监听但上游已断，请求永远挂着。启动时先探活一次。
5. **`BACK_N` 的按数不能有下限**，从应用根 activity 按 Back 就退到桌面了。
6. **`TRAJECTORY_REPLAY` 解析不到应用就必须中止**，否则坐标会点在桌面图标上。
7. **回滚"成功"不等于没事**：恢复阶梯可能自报成功而设备实际已经在别的应用里。要**独立地检查前台包名**。
8. **注入的措辞会改变模型行为**。建议句会让它反复点同一个控件。

---

## 10. 现有实测结论（作为真机实验的先验，不是结论）

模拟器 AndroidWorld 116 任务，同 session 同 runner 的对照臂（机制全关）作基线：

| | 成功 | 成功任务平均步数 |
|---|---|---|
| 机制全开 | 42 / 49 / 46（三次） | 6.2–6.5 |
| 对照（机制全关） | 55 | 7.82 |

- **步数收益稳固**：比对照少约 1 步，逐任务约 30 快 / 3 慢。**收益几乎全部来自首步应用启动折叠**（96/96 零失误），它把 `open_app` 和第一个真实决策折叠进同一个计数步。
- **图跳过是净亏损**：有跳过的任务 1/17，对照 6/17。原因是"落点命中"只证明动力学没记错，不证明这个动作此刻是对的。
- **探索与注入是中性的**：探测 24% vs 22%，注入 100% vs 100%（样本小）。
- **前瞻几乎从未成立**：模型下一步真正点的控件只有 16% 出现在探索候选集里，出现时平均排第 9 位。这是整个"投机前瞻"路线的瓶颈。

真机上这些数字都需要重新测，但**准入条件和安全规则不要重新发明**，它们每一条背后都是一次实测失败。
