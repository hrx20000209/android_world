# MobileExplorer 当前实现摘要（2026-09-14 巡检）

供消融实验设计使用。以下全部来自实际代码，不是从设计文档推断的。

## 入口与主循环

| 位置 | 作用 |
|---|---|
| `scripts/run_serial_exploration_task.py` | 唯一的运行器（约 3100 行）。所有消融都通过它的 CLI flag 控制，没有复制第二份 agent 实现 |
| `serial_step(self, goal)`（约 2205 行起） | 每一步的主体：预填 → 引导 → 图读写 → 推理 → 执行 |
| `android_world/suite_utils.py:525` | `return int(10 * (task_complexity))` —— **官方每任务步数预算**。`--max_steps 0` 表示交回给它 |

## 图

| 概念 | 实现 |
|---|---|
| 节点 | `belief_graph.GraphNode`；键 = `hash(activity, layout_signature)`。带 `salient_ui_labels`、`ui_elements`、`visit_count`、`semantic_summary` |
| 边 | `belief_graph.GraphEdge`；键 = `control_key` = `resource_id\|text\|content_desc\|class`，**不用坐标** |
| 代际护栏 | `GenerationGuardedGraph` / `GraphSnapshot`：第 *i* 步只能读到第 *i−1* 步写入的内容。**硬约束：exploration 不得使用本轮推理结果** |
| 每任务落盘 | `progressive_belief_graph.json`（nodes + edges） |

## 跨任务记忆（历史信息）

`android_world/parallel_exploration/app_memory.py`

| 结构 | 内容 |
|---|---|
| `AppMemoryStore` | 每个 package 一个 JSON 文件，`--app_memory <dir>` 指定 |
| `RememberedScreen` | `layout_signature`、`labels`、`description`、`tasks_seen`、`transitions` |
| `RememberedTransition` | `control_key`、`action`、`dst_layout_signature`、`dst_activity`、`executed_count`、`tasks_executed`、**`goals`（执行过它的任务目标原文，≤8 条）** |
| 写入点 | `observe_screen()` / `observe_execution()`，后者只记录真实执行过、且落点仍在同一 app 内的转移 |
| 读取点 | `retrieve_control()` —— 唯一性门 → 目标条件 kNN，两条路都过通过率门与同类门 |

**历史 vs 新鲜的实现边界**：store 是跨进程持久的（历史），belief graph 是每任务新建的（新鲜）。
一个进程 = 一个任务，`_seen_this_task` / `_executed_this_task` 用于把本任务自己的观测从
"别的任务怎么做"的统计里剔除。

## 证据如何进入推理

当前主臂 **`--graph_context off`**：不向 prompt 注入任何图信息。
（`--graph_context` 支持 `edges/path/distill_path/llm/store/store_path/elements/elements_store`，
六种注入形式在 09-12/09-13 全部实测为无收益，故默认关闭。）

图对推理的作用走的是**另一条路**：预填（prefill）在推理**之前**替模型执行一个已知动作，
然后照常跑这一步的推理 —— 即"移除一次决策"，而不是"增加上下文"。

## 预填路径（核心机制）

`_prefill_from(state_start, node_start, step)`（约 2397 行起），每步调用两次：
推理前一次，step 0 的 `open_app` 引导之后一次。

逐道门：

| 门 | flag | 作用 |
|---|---|---|
| 来源 | `--prefill_sources store\|episode` | 默认只用 store（episode 线上仅 30% 命中） |
| 唯一性 | `--prefill_min_tasks` | 该屏只有一个控件被 ≥N 个任务执行过 |
| 目标条件检索 | `--prefill_retrieval`, `--prefill_k`, `--prefill_sim_thr`, `--prefill_margin` | 目标最相似的 k 个先例加权投票，冠亚军边际不足则拒答 |
| 通过率 | `--prefill_pass_rate` | `tasks_executed / tasks_seen ≥ 0.7`，排除岔路口 |
| 同类 | `--prefill_any_kind` | 问答类目标不复用执行类先例 |
| 风险 | `--prefill_risk probe\|loose` | loose 只拒 checked 态控件 |
| 控件匹配 | `--prefill_by_resource_id` | 四段键失配时回退到同屏唯一 resource id |
| 链长 | `--max_prefill` | 一步内最多几跳，逐跳核对落点 |
| 落点核对 | （无 flag） | **按 destination activity**，不按 layout 签名 |
| 落点不符 | `--prefill_rollback` | 按 Back 回退并撤回 summary；落点等于出发点则不回退 |

## 跳过 / 回滚

| 机制 | 位置 | 说明 |
|---|---|---|
| `app_bootstrap` | `--bootstrap`（默认开） | step 0 从桌面 `open_app`，**不消耗一步**，随后照常推理。每任务省 1 次决策 |
| `SKIP_INFERENCE` | `--enable_skip`，`ReasoningGate` | 三态门 NORMAL / GRAPH_ENHANCED / SKIP_INFERENCE。实测全程只触发 6 次 |
| 预填回退 | `--prefill_rollback` | 落点 activity 不符时 Back |
| 探测回滚 | `observe_rollback()` | 仅探测路径用，主臂 `--probes_per_step 0` 已关闭 |

## 已有的事件日志（每任务 `serial_events.jsonl`）

| kind | 字段 |
|---|---|
| `inference` | `step`, `inference_s`, `exploration_s`, `action`, `done` |
| `gate` | `step`, `mode`, `reason`, `node_visits`, `node_entropy` |
| `skip` | `kind_detail`(`app_bootstrap`/`store_prefill`), `matched`, `source`, `control`, `expected_activity`, `landed_activity`, `sig_matched`, `by_resource_id`, `latency_s` |
| `prefill_refused` | `why`, **`gate`（哪道门拒的）**, `best_sim`, `pass_seen`, `pass_ran`, `screen_known`, `resource_id_seen` |
| `prefill_rollback` | `restored`, `landed` |
| `prefill_unmoved` | 点击未改变屏幕 |
| `app_memory_seed` / `app_memory_load` / `description_backfill` | store 读写 |

## 已能测 / 不能测

**已能直接测**：成功率、步数、成功任务步数、推理调用数、预填跳数与命中率、
按来源/按门的拆分、回退次数与复位率、图节点数、墙钟与每步延迟、
`app_bootstrap` 与 `SKIP_INFERENCE` 的显式跳过数。

**当前测不了**：CPU/GPU 利用率、RSS/峰值内存、功耗 —— 运行器没有任何采样点，
且模型跑在远端 vLLM（labserver），设备侧只有 adb 驱动的模拟器，
"端侧资源干扰"在当前部署形态下无法测量。前台探索/后台推理并发也不存在：
主臂 `--probes_per_step 0`，探索与推理是串行的。
