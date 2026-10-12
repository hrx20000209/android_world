# MobileExplorer 任务导航：实测验证报告（2026-10-03）

结果目录：`results/sensys27_task_navigation/20261003_pilot/`。本报告只陈述实际测得的数值；所有结论都基于 **11 个任务、单个评估 seed（30）的小规模 pilot**，成功率差异全部低于噪声地板，不能当作成功率结论。

## 0. 实验环境与口径

| 项 | 值 |
|---|---|
| 模型 | `GELab-Zero-4B-preview`（vLLM，`max_model_len=8192`，prefix caching 开），AutoDL GPU，SSH 隧道 `127.0.0.1:8000` |
| 解码 | temperature 0，top_p 1，max_tokens 1024；基础 prompt = `gelab_agent.py`（与产出 4B_2 的 71b87b14 字节一致） |
| 设备 | 本机 `emulator-5554`，Android 13，1080x2400，FastA11y |
| 部署形态 | **edge-assisted**（模型与 UI 不在同一设备）。单手机资源竞争实验**未做** |
| Runner | `scripts/run_serial_exploration_task.py`（串行；延迟数字不能用于论证并行加速） |
| 共同配置 | `--max_steps 0`（agent 上限 20 步）、`--no_decision_constraints --graph_context off`、bootstrap（app 启动折叠）全组开启 |
| 坐标契约 | 模型输出 `point:x,y`，为 0–1000 归一化；`gelab_agent` 的缩放与之匹配。`gelab_agent_test.py` 的两个既有失败针对 tool_call 格式，本模型不使用 |
| 计数 | “步数”= 模型决策数（`agent._actions`）；模型调用另由 `api_calls.jsonl` 逐 HTTP 请求记录（含 token） |
| 代码状态 | `git_head.txt` + `working_tree_final.diff` + `code_snapshot/` |

任务（11 个，覆盖 Markor / Calendar / Contacts / Expense / Broccoli；读信息、多层导航编辑、实体列表批量）：
ContactsAddContact, ExpenseDeleteMultiple, MarkorAddNoteHeader, MarkorCreateFolder, MarkorEditNote, RecipeDeleteMultipleRecipes, RecipeDeleteSingleWithRecipeWithNoise, SimpleCalendarDeleteOneEvent, SimpleCalendarEventsOnDate, SimpleCalendarLocationOfEvent, SimpleCalendarNextMeetingWithPerson。

## 1. 本次实现

1. **独立持久状态检查器** `android_world/parallel_exploration/app_state_checker.py`：以 root 经 `adb shell` 只读；SQLite 逐表单事务逻辑 dump（含 WAL），文档目录（**目录树 + 文件 md5**），shared_prefs（显式排除 Markor 浏览历史缓存）。7 个 app 有 spec；无 spec / 非 root / 读取失败 → `covered=False`，不发证书。单次快照 0.09–0.7 s。
   - 实跑发现并修复两个真实缺陷：① 只列普通文件，看不见新建的空目录，使 Markor 的 FOLDER 被误判 invariant；② BLOB 行导致 UTF-8 解码失败。均有回归测试与真机验证。
   - 局限：窗口级证书（同窗口先改后复原不可见）；不观测网络、云同步、其他 app。
2. **串行 runner**：设备参数化（`--serial/--console_port/--adb_path`，默认不变）；`--state_checker on`（探索窗口前后快照）；`--task_navigation_history`（agent 实际点击按同一 schema 记为 `origin=execution` 的 witness，证书比较动作前与落定后）；`api_calls.jsonl`；shadow 模式同时记录严格与 UI-only 规划及每步 planner 可见的 witness 快照（`nav_snapshots/`）。
3. **warm 记忆的冻结与门控**（复用已有 `app_memory` + `--store_prefill`，没有另建 memory）：`--app_memory_readonly`（评估不回写，消除任务顺序依赖）；执行转移记录检查器判定（`state_verified_count/state_changed_count`）；`--prefill_require_state_cert`；`--prefill_strict_selector`。
4. **分析**：`scripts/analyze_task_navigation_pilot.py`（逐 episode CSV、汇总、paired bootstrap）、`scripts/hindsight_navigation_bound.py`（后见之明上界）。
5. `infer.py` 可选 Bearer 认证（未设置环境变量时行为不变）。
6. 测试：相关集合 **323 passed**。

## 2. 结果

### 2.1 冷启动、episode 内任务导航（原设计）

| 组 | 成功 | 平均步数 | 模型调用 | prompt tokens | `<NAV_INTENT>` |
|---|---|---|---|---|---|
| base | 8/11 | 6.73 | 74 | 239.5k | — |
| intent_only | 8/11 | 6.73 | 74 | 250.7k（+4.7%） | **0/74** |
| shadow_frontier | 8/11 | 6.73 | 74 | 251.1k | **0/74** |

- **模型契约覆盖率 0%**：4B 微调模型从不输出可选的 `<NAV_INTENT>`。它没有损害成功率，但多付 4.7% prompt token。
- **后见之明上界 = 0**：把模型几步后实际到达的屏幕当作完美契约，用真实 planner 在当时的 witness 快照上搜路线，严格门与 UI-only 门下替代的决策数都是 0。
- 原因（拒绝瀑布与机会诊断）：
  - 严格签名下的重访只占 4/74 步（5%），路线必须从当前屏出发；
  - 探索受 `step >= 2` 门限制，29 个探索窗口中 23 个零 probe（合计 9 次 probe）；
  - planner 拒绝原因：`weak_selector` 217、`uncommitted_or_stale` 133、`unverified_app_state_effect` 44、`not_navigation_probe` 38、`editable_control` 22、`no_exact_inverse` 19；
  - 执行来源 witness 52 条，其中 36 条拿到证书，但都描述已走过的边。

### 2.2 冻结 warm 记忆（历史复用，不是在线图）

语料：同 11 个任务在**训练 seed 0、1** 上以 base 配置运行（成功 8/11、9/11），store 冻结（文件只读）：76 屏、43 条执行转移，17 条被 ≥2 个任务执行，30 条有“不改状态”证书，6 条观测到改状态。评估在 **held-out seed 30**。

| 组 | 成功 | 成功任务均步 | 与 base 共同成功的配对步数差 | 预填（落点命中） |
|---|---|---|---|---|
| base | 8/11 | 6.75 | — | — |
| base 重复 | 8/11 | 7.25 | +0.14（n=7） | — |
| warm_prefill（原有门） | 8/11 | 6.25 | **−0.50**（n=8） | 10（6） |
| warm_prefill_cert | 8/11 | 6.62 | −0.12（n=8） | 5（2） |
| warm_strict（证书+强 selector） | 7/11 | 7.29 | 0.00（n=7） | 0 |

warm_prefill 的 10 次预填逐条审计：

| 类型 | 次数 | 例子 | 是否省步 |
|---|---|---|---|
| 提交类动作（证书显示会改持久状态） | 3 | FOLDER（0 次不变 / 2 次改变）、DELETE ×2（0 / 8） | 是，3 个任务各 −1 |
| 弱 selector，落点命中 | 3 | Expense 行容器 `cd_expense`、`btn_delete` 图标（只弹确认框，不改状态）、Markor `icon` | 是，Expense −2 |
| 弱 selector，落点未命中 | 4 | Calendar 月视图容器 `month_view_background` ×4（训练时模型点的是网格里某个日期，selector 塌缩成整个网格） | 否，错误动作 |
| 强 selector 且有“不改状态”证书 | **0** | — | — |

结论：原有预填省下的 0.5 步，**全部来自替模型做本应由模型做的判断**：提交，或者没有绑定实体的选择。两道门加上后收益归零。warm_strict 少的 1 个成功（SimpleCalendarEventsOnDate）无任何预填，base 重复在同一任务上以同样的 2 步失败，属于噪声。

### 2.3 噪声

同一配置（base）跑两次，11 个任务中 3 个不同：MarkorEditNote 成败翻转（0/4 → 1/6），SimpleCalendarEventsOnDate 翻转（1/3 → 0/2），RecipeDeleteSingle 步数 6 → 7。temperature 0 下仍有非确定性。在 n=11 时，任何 ≤2 个任务的成功率差都不可读。

### 2.4 典型轨迹

- **证书门拦下提交（MarkorCreateFolder，seed 30）**：warm_prefill 在 step 2 替模型按 FOLDER（store 记录 0 次不变 / 2 次改变），3 步成功；warm_prefill_cert 拒绝（`no_state_certificate`），模型自己按下，4 步成功。
- **实体串用 / mismatch（SimpleCalendar*，seed 30）**：预填点击 `month_view_background`，落在网格中心某个随机日期，landing 判 mismatch；warm_strict 以 `weak_selector` 拒绝。
- **规划器零路线**：`shadow_frontier/*/nav_snapshots/` 中每步的 witness 快照与 `hindsight_bound.json`。

## 3. 对三个问题的回答（以证据为准）

**问题一：在线、任务特定的图怎样建立、存什么、与离线 UTG / 文档 / 历史复用如何区别？**
已实现的 witness 存 exact source/target 契约（activity + structural signature + 标签）、selector、generation、恢复与独立状态证书；任务特定性原定来自模型契约。实测：契约覆盖率为 0，且即使契约完美，episode 内 witness 也无法支撑任何路线。因此在本 backbone 和本 benchmark 上，**“在线、任务特定的图”没有可测的贡献**。能产生步数收益的只有冻结的跨 episode 记忆，必须称为历史复用，与 AutoDroid / SwiftAgent 类工作同类。

**问题二：怎样利用图定位最值得补齐的探索缺口？**
frontier scorer 已接入，但缺少可执行目标（契约为 0）且探索窗口大多为零 probe，没有可比较的数据。frontier vs random vs 旧 scorer 的比较**未完成**，不能下结论。从拒绝瀑布看，真实瓶颈是 selector 强度（列表行与容器没有自身文字）和重访稀少，不是候选排序。

**问题三：怎样省 navigation 推理，并在复杂判断前停下？**
在复杂判断前停下这条边界可以实现，而且能测：状态证书 + 强 selector 准确拦下了提交动作和实体不明的选择。但在 11 个任务上，满足边界的纯导航预填为 **0 次**；之前测到的步数收益恰好来自越过这条边界的动作。

## 4. 假设检验汇总

| 假设 | 结果 |
|---|---|
| 4B 模型能在原有推理中给出导航契约 | **不成立**（0/74） |
| H1：冷启动 online witness 能在任务结束前产生可消费路径 | **不成立**（后见上界 0） |
| H2：frontier 优于 random / 旧 scorer | **未检验**（无可执行目标、探索供给近零） |
| H3：多跳导航减少调用且不降成功率 | 冷启动：**无机会**。warm：原有门 −0.5 步、成功率持平，但收益来自提交 / 实体选择；严格门下收益为 0 |
| 独立状态检查器能区分导航与副作用 | **成立**（FOLDER、DELETE 被正确判为改变；对话框打开判为不变；发现并修复 2 个漏检） |
| 导航适用性与 landing 一致 | 不一致：landing 命中 6/10，其中 3 次命中的是提交动作，命中不等于适用 |

## 5. 实现与设计的差距

- 状态检查器：串行 runner 已接通；**并行 runner 未接**；窗口级证书，不是逐 probe。
- 并行 runner：设备参数仍硬编码；`ReperceptionRequired` 故障注入测试**未写**。
- 7 组 controlled pilot（history_nav / online_random / online_current / online_frontier / full_strict）**未跑**。原因：诊断阶段后见上界为 0，按 prompt 规定不在 routes=0 时盲跑全套。
- 样本：11 个任务 × 1 个评估 seed，未扩到 37 个任务的易翻转套件与 seeds 30/31/32。
- 延迟：串行 runner，不报告 E2E 加速。

## 6. 下一步最有证据支撑的选择

1. **把“在决策边界停下”作为贡献**，不再主张“在线图省推理”：检查器和两道门可以作为对 reviewer A/B/C/D（pHash 相同不代表状态恢复）的直接回应，并配上本报告的逐条审计。
2. 如果仍要做步数收益，需要**实体绑定的 selector**：列表行按子节点文字绑定实体、日期网格按日期标签。只有在 selector 能区分实体之后，warm 的纯导航预填才可能出现；但从冷启动 episode 内拿 witness 的路已经被测量否定。
3. 任何成功率主张都需要 37 个任务 × 3 个 seed 以上的重复，并配同 session 的 base 对照。

## 7. 后续：v2 设计（任务实体参数化的在线流程图）

基于本报告第 2 节的否定结果，设计改为以“本 episode 内、以任务实体为参数的决策片段”为图的单位，详见 `docs/sensys27_task_graph_design_v2_zh.md`。实测（27 个任务，seed 30，同 session 交错配对；结果目录 `results/sensys27_task_navigation/20261003_proc_final/`）：

- 成功 18/27 对 18/27；共同成功任务的平均步数 7.76 → 6.47（−17%）。
- 宏触发的 4 个多实体任务：每个任务 −5.5 步、−4.5 次调用，成功 4/4；3 次重复 12/12 成功，步数完全稳定（13→7、10→6、13→11、19→9）。
- 20 跳回放全部停在提交之前。
- 代价：串行 runner 每步约 +1.0 秒（检查器快照），宏未触发的任务整体慢约 27%。另外，只在出现提交时才发起一次约 105 token 的实体抽取。
- 开发过程中两次**由机制造成的失败**（回放失败的实体被误标为完成；越过任务顺序回放打乱了模型的进度叙述），都已定位、修复并补了回归测试。
