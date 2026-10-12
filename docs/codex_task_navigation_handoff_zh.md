# 交给 Codex 的执行 prompt：MobileExplorer 任务图导航的实测验证

你在 `/Users/huangrunxi/Projects/android_world`（macOS，Python 3.11.5 = `~/anaconda3/bin/python`）工作。目标：在本机 AndroidWorld 模拟器上**实际跑实验**，验证任务条件化的在线图导航能否**稳住成功率、减少模型推理步数**，并写出可复查报告。基线先用 `results/4B_2.txt`。不要只写计划。

完整背景和三个研究问题见 `docs/claude_code_sensys27_task_graph_prompt.md` 与 `docs/sensys27_task_graph_design_zh.md`，先读这两份。下面是已经完成的部分、现状事实和你需要做的事。

## 硬性约束

- 保留所有未提交修改，不 `git reset`/`checkout`，不覆盖 `results/` 下历史结果；新结果只写到 `results/sensys27_task_navigation/<timestamp>/`。
- **API key 不得打印、写入日志、报告或提交。** 不要自行去远端查 key；如请求返回 401，停下来告诉用户。
- 不调小 visit/entropy/confidence 阈值来制造 skip；routes 为零时输出拒绝瀑布（rejection waterfall）。
- 不读 evaluator 的正确答案来生成导航 intent；checker 可读 benchmark 状态，但策略只拿到 invariance verdict。
- 不写“已获得 X% 加速”“保证接收”之类结论；成功率差 < 8 个任务（n≈115）属于噪声（同配置重跑可差 ±10，见下）。
- 先小规模 paired pilot，routes=0 或全是错误 landing 时不要盲跑全套。

## 环境（已核实）

- 模拟器：`emulator-5554`，Android 13，1080x2400，console port 5554；`adb` 在 PATH，runner 默认 adb 路径 `/Users/huangrunxi/Library/Android/sdk/platform-tools/adb`。FastA11y 服务（`com.androidworld.fasta11y`）已开启。`adb root` 可用（checker 依赖它）。
- VLM：远端 AutoDL 上的 vLLM，用户已经开了 SSH 隧道 `127.0.0.1:8000 → 远端 127.0.0.1:8000`（主机实为 `connect.westc.seetacloud.com -p 17139`，用户原文写成 `west.c` 是笔误）。`/health` 返回 200，但 `/v1/*` 返回 **401 Unauthorized**——服务启用了 `--api-key`。
  - 已在 `android_world/agents/infer.py` 的 `LlamaCppWrapper.call_llama` 加了可选 `Authorization: Bearer`，从环境变量 `ANDROID_WORLD_LLM_API_KEY` 或 `VLLM_API_KEY` 读取，未设置时行为不变。
  - **第一件事**：确认用户已把正确 key 放进你进程的环境变量（不要回显它），然后用一个最小请求验证 `/v1/models` 返回 200，记录模型 id、`max_model_len`。若仍 401，停下来请用户处理（或让用户重启 vLLM 不带 `--api-key`）。
  - API URL：`http://127.0.0.1:8000/v1/chat/completions`。串行 runner 用 `--api_url`；并行 runner 用 `ANDROID_WORLD_LLM_API_URL`，其 `--metrics_url` 必须指向同一实例（`http://127.0.0.1:8000/metrics`）。
- 模型输出契约：历史日志显示 GELAB 模型输出 `<THINK>…</THINK>` + `explain:…\taction:CLICK\tpoint:x,y\tsummary:…`，**坐标是 0–1000 归一化**（如 `point:66,857`），`gelab_agent.py` 的缩放与之匹配。`gelab_agent_test.py` 的两个既有失败（`test_parse_tool_call_json_fallback`、`test_tool_call_click_keeps_absolute_coordinate`）针对 tool_call 格式，该模型不用；拿到 API 后用 2–3 个真实请求再确认一次输出格式，并在报告中写明 adapter 契约，不要改测试期望掩盖。
- 当前 `gelab_agent.py` 与产出 4B_2 的提交 `71b87b14` 字节一致（提示词无回退）；`MAX_AGENT_STEPS = 20`，与 4B_2 的步数上限一致。
- 4B_2：`results/4B_2.txt`，约 47/116 成功（按文档口径），成功任务均步约 6.4。注意它是另一环境的单次跑测，**每次比较都必须带同 session、同 runner 的 base 臂**；4B_2 只作参考。

## 已完成（本 session，未提交）

1. `android_world/parallel_exploration/app_state_checker.py`（新）：独立只读持久状态检查器。
   - 以 root 经 `adb shell` 对 SQLite 做 `sqlite3 -readonly`、每库单事务（含 WAL）的逐表逻辑 dump 取 digest；文档目录逐文件 md5；shared_prefs 逐文件 md5（显式排除 Markor 的 `DOCUMENT_MOD_TIMES.xml`、`cache.xml`、`WebViewChromiumPrefs.xml` 这类浏览历史缓存）。
   - 覆盖 Calendar(pro)、Expense、Broccoli、Contacts(pro + contacts2.db)、Tasks、Markor（`/storage/emulated/0/Documents`）、SMS（mmssms.db）。无 spec / 非 root / dump 出错 → `covered=False`，绝不当作“未变”。
   - 实测：每次快照 0.1–0.7s；在模拟器上能检出文件写入，复原后判 invariant。
   - 局限（报告要写）：窗口级证书（同一窗口内先改后复原不可见）；不观测网络/云同步/其他 app。
2. `scripts/run_serial_exploration_task.py`：
   - 参数化 `--serial`、`--console_port`、`--adb_path`（默认值保持原硬编码，兼容旧脚本）。
   - `--state_checker on`：每个探索窗口前后快照；只有 snapshot 一致 + `restore_status == RESTORED` + 实际有 probe 才给该窗口所有 witness 写 `state_effect_verified=true`，并把 verdict 摘要写进 witness 的 `state_check`；事件 `state_check` 入 `serial_events.jsonl`。
   - `--task_navigation_history`：把 agent 自己执行的点击按同一 schema 记录为 `origin="execution"`、`probe_type="EXEC_NAV"` 的 witness（selector 取点击点处最小的 clickable 元素；target 用下一步开始时的 settled capture；状态证书比较动作前与落定后的快照）。用于 `history_nav` 对照臂。
   - 每次模型 HTTP 请求写 `api_calls.jsonl`（step、latency、prompt/completion tokens、是否含 `<NAV_INTENT>`）。注意：`call_llama` 内部的重试循环只记一次，需要的话在 `requests.post` 层再细化。
   - shadow 模式下额外跑一份 `allow_ui_only=True` 的影子规划，`task_navigation_plan` 事件同时记录 `route`/`rejections`（严格）和 `route_ui_only`/`rejections_ui_only`，以及 `intent_valid`、`already_at_target`、`witnesses`、`current` 等字段。
3. `android_world/parallel_exploration/task_navigation.py`：planner 接受 `origin="execution"` 的 witness——不要求 inverse，但仍要求 `state_effect_verified`（ui_only 消融除外），且 probe_type 必须是 `EXEC_NAV`。
4. 测试：`tests/test_app_state_checker.py`（新）与 `tests/test_task_navigation.py` 新增用例。下列集合当前 **273 passed**：
   ```bash
   python -m pytest -q tests/test_task_navigation.py tests/test_app_state_checker.py \
     android_world/parallel_exploration/serial_generation_test.py \
     tests/test_semantic_prefix_memory.py tests/test_online_exploration_runtime.py \
     android_world/parallel_exploration/state_graph_information_test.py \
     android_world/parallel_exploration/live_probe_test.py \
     android_world/parallel_exploration/state_test.py \
     android_world/parallel_exploration/two_system_test.py
   ```
5. `scripts/run_task_navigation_pilot.sh`（新）：`OUT_ROOT SEED ARM TASK...`，可续跑。各臂共享 `--max_steps 0 --no_decision_constraints --graph_context off --exploration_budget_s auto --max_depth 2`，bootstrap 全开（与历史比较口径一致）。臂：`base`、`intent_only`、`shadow_frontier`、`history_nav`、`online_random`、`online_current`、`online_frontier`、`full_strict`、`full_ui_only`。API URL 用 `API_URL` 环境变量覆盖。

**以上串行 runner 改动尚未在真实 episode 上跑过**（API 一直 401）。第一次跑时重点检查：`state_check` 事件是否出现、`history_witness` 是否写入、`task_navigation_plan` 字段是否齐全、`api_calls.jsonl` 是否有 usage。

## 已知事实 / 风险（来自历史测量，写报告时引用）

- 串行 runner 的“步数” = `len(agent._actions)` = 模型决策次数；bootstrap 折叠与导航 macro 都不进 `_actions`，所以它们减少的是**推理调用**，物理操作要单独统计（macro 动作在 `nav_actions` 和 `task_navigation_execute` 事件里）。
- 探索资格：`may_probe` 硬性要求 `step >= 2`、`min_node_visits`（默认 2）、`unrecovered_once` 一次失败即永久停探。pilot 用 `--min_node_visits 1`（这是被测变量 H1，不是为了凑 skip），报告里分层给出首次访问/重访。
- 时序上的根本风险：窗口 t 的 probe 从屏幕 S_t 出发，而模型在 S_t 同时做决定、离开 S_t；witness 只在下一代可用。因此可执行路线主要来自**沿模型实际路径的 depth-2 probe** 或重访。历史上 prefix 对齐只有 ~2.5%，“图来得太晚”是此前所有 null 结果的统一解释（85% 的成功来自 <10 步的短 episode）。务必先用 shadow 量化 route 出现率与首次可用时刻，再决定是否值得 execute。
- 4B 微调模型是否会输出可选的 `<NAV_INTENT>` 是最大未知数；历史上改 prompt 曾掉 12 个任务（`_decision_constraints`）。`intent_only` 臂必须与 `base` 配对，单独衡量契约 prompt 对成功率的影响。若 intent 出现率接近 0，如实报告，不要用任务文本猜 intent 代替。
- `checked` 字段：fast a11y 路径仅 checkable 元素非 None；若某处走 uiautomator 解析，`checked` 永远是 bool，planner 会把所有 witness 拒为 `stateful_or_dynamic_control`、relocate 也全拒——若看到这个拒绝原因占满，先查 a11y 方法。
- 同配置重跑成功数可差 ±10（115 任务）；用 `configs/volatile_suite.json` / `results/mobileexplorer_runs/tasks_volatile.txt`（37 个易翻转任务）放大信号；机制读数（calls、route 数、landing 正确率）比成功率稳定得多。
- SMS 类任务受模拟器默认短信角色漂移影响（`RequestRoleActivity`），pilot 里避开或单独标注。

## 你要做的

### 0. 打通 API（阻塞项）
见上文。确认 200 后，用一个 1 步 episode 冒烟：
```bash
API_URL=http://127.0.0.1:8000/v1/chat/completions \
  scripts/run_task_navigation_pilot.sh results/sensys27_task_navigation/<ts> 30 base MarkorCreateFolder
```
检查 log 末尾的结果表、`api_calls.jsonl`、`serial_events.jsonl`。记录模型 id、量化、`max_model_len`、解码参数（temperature 0、top_p 1、max_tokens 1024）、git HEAD 与 `git diff --stat`。

### 1. 诊断 pilot（先做，约 10 个任务 × 3 臂）
任务（以注册表为准，覆盖读信息 / 多层导航编辑 / 实体列表）：
`SimpleCalendarEventsOnDate, SimpleCalendarNextMeetingWithPerson, SimpleCalendarLocationOfEvent, SimpleCalendarDeleteOneEvent, MarkorEditNote, MarkorAddNoteHeader, MarkorCreateFolder, ContactsAddContact, ExpenseDeleteMultiple, RecipeDeleteMultipleRecipes, RecipeDeleteSingleWithRecipeWithNoise`
臂：`base`、`intent_only`、`shadow_frontier`，seed 30。

写 `scripts/analyze_task_navigation_pilot.py`，逐 episode 输出 CSV + summary JSON：
- 官方成功（log 结果表）、模型调用数（`api_calls.jsonl` 行数）、prompt/output tokens、`_actions` 步数、macro/probe/inverse 物理操作数、episode wall-clock。
- intent 出现率、parser 失败率、合法率；target/entity 是否合理（离线人工/规则审计，不进策略）。
- route 出现率（严格 vs ui_only）、首次可用 step、`already_at_target` 次数、拒绝原因瀑布（按 `rejections` 汇总）。
- 探索：`may_probe` 被哪条条件挡住（需要的话给 runner 加一条 `explore_gate` 日志，记录每个条件的布尔值）、`state_check` 的 verified/changed/not_covered 分布与 checker 耗时。
- 与 4B_2 同任务对照（参考）和与同 session `base` 的配对差。

若 routes=0：给出瀑布，定位是 no intent / no witness / no connected route / unverified state / ambiguous selector，然后针对最大的一层改设计（例如：让 frontier 优先沿模型声明的 target 做 depth-2，或把 intent 的消费时刻提前），**而不是降阈值**。

### 2. 实现剩余设计
- **路径补全 scorer**（`task_navigation.py`）：在 `frontier_scores` 之外新增 budgeted path-completion：优先“差一条已验证连接即可把已有 witness 段连到 intent target”的候选；日志里同时记录预计解锁 hop 数与实际解锁 hop 数。标签重叠只能叫 proxy。
- **并行 runner** `scripts/run_one_parallel_exploration_task.py`：同样参数化 serial/console/adb（目前硬编码），接入 checker（窗口：推理开始前 / 授权动作前），并给 `ReperceptionRequired` 路径写故障注入测试（恢复失败 → 丢弃旧动作 → 重新抓图推理 → 不再在脏状态上 probe；两次请求都计入 calls/tokens）。该修复要在所有对比臂一致生效。
- adapter 测试：frame/epoch 不错配、每跳动作插回 recovery prefix、macro 耗时计入 E2E、恢复失败时旧 action 未执行。

### 3. Controlled pilot（诊断通过后）
臂：`base, intent_only, history_nav, online_random, online_current, online_frontier, full_strict`（+ 单列 `full_ui_only` 消融）。同任务、同 seed，先 seed 30；有信号再扩到 37 任务 volatile suite 与 seeds 30/31/32。warm memory 不混入 cold 臂。旧 `--enable_skip`/`--store_prefill`/`--em_shortcut` 全关。

### 4. 分析与报告
- paired bootstrap（同 task-seed 差值），失败/timeout 计入主结果；共同成功子集只作辅表。hop 数只能叫“机会估计”，真实节省只看与 paired base 的 calls 差。
- 至少一条成功轨迹（base 逐页推理 vs 图在哪个窗口产生证据 → 下一代用契约执行两跳 → terminal 仍调用模型）和一条 rejected/mismatch 轨迹。
- 模型在远端 GPU、UI 在本机模拟器：只能称 **edge-assisted**；单手机资源竞争实验标为未完成。
- 报告写到 `docs/sensys27_task_navigation_validation_zh.md`，逐一回答三个问题（在线任务图如何建/存什么/与离线 UTG/文档/历史复用的区别；如何定位探索缺口；如何省 navigation 推理并在决策前停下），并列出：checker 是否接通及覆盖、契约覆盖率、首次访问是否有收益、frontier 是否优于 random、导航适用性与 landing 是否一致、calls/E2E/TSR trade-off、哪些假设成立/不成立、下一步最有证据支撑的设计选择。

完成后汇报：代码变更、测试结果、真实实验数值、关键失败原因。不要提交 commit，除非用户要求。
