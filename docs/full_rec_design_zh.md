# full_rec 设计详解：两张在线图、探索如何用图、推理如何用图

本文描述 `scripts/run_task_navigation_pilot2.sh` 里的 `full_rec` 这一组，也就是当前在 AndroidWorld 上跑的完整设计。所有说法都对应到源码；代码片段经过删节（去掉注释和日志），行号以 2026-10-04 的工作区为准。

```bash
# scripts/run_task_navigation_pilot2.sh
full_rec) flags=(--probes_per_step 3 --max_depth 2 --enable_skip --graph_context distill
                 --procedure_replay execute --state_checker on --procedure_records
                 --probe_only_certifiable --probe_entity_rows) ;;
```

| 开关 | 属于哪一层 | 作用 |
|---|---|---|
| `--probes_per_step 3 --max_depth 2` | 原设计（v29） | 每个推理窗口最多 3 次探测，最深 2 跳 |
| `--enable_skip` | 原设计 | 图跳步：满足条件时整步不调用模型 |
| `--graph_context distill` | 原设计 | 把当前节点的一跳事实蒸馏成 `[Memory]`，注入 prompt |
| `--procedure_replay execute` | 流程图（v2） | episode 内按任务实体回放已演示的导航片段 |
| `--state_checker on` | 流程图 | 独立的持久状态检查器，给每次点击和每个探测窗口发证书 |
| `--procedure_records` | full_rec 新增 | 契约里同时抽取每个实体的属性值，作为“记录槽位” |
| `--probe_only_certifiable` | full_rec 新增 | 只在检查器覆盖的 app 里探测 |
| `--probe_entity_rows` | full_rec 新增 | 允许探测“包含未处理任务实体”的列表行 |

full_rec 和 base 共用的设置：同一 backbone（GELab-Zero-4B），temperature 0，bootstrap（第 0 步直接打开目标 app），步数上限 20。

---

## 0. 一步之内发生了什么

`serial_step()`（`scripts/run_serial_exploration_task.py`）对第 i 步依次执行：

```text
 1. finalize_procedure_hop        把上一步模型的点击连同检查器判定记成流程图的一“跳”
 2. capture S_i                    a11y 树 + 截图签名，upsert 成屏幕图的节点
 3. bootstrap（仅 step 0）         按目标里的 app 名直接启动，不花推理
 4. procedure_plan / execute       流程图：有模板、有下一个任务实体时，回放导航段，停在提交之前
 5. skip gate                      屏幕图：有可复用边且通过门控 → 执行它、整步不推理
 6. graph context                  屏幕图：蒸馏当前节点的一跳事实 → [Memory] 注入 prompt
 7. exploration（满足 may_probe）  在 S_i 上探测候选控件，结果写入屏幕图的“下一代”
 8. inference                      模型在（回放后的）当前屏幕上决策并执行动作
 9. record transition              模型的真实转移写成屏幕图的边；准备第 1 步要用的 pending hop
```

**世代隔离**：第 i 步所有对屏幕图的读取都来自第 i−1 步结束时的快照（`GenerationGuardedGraph.snapshot(for_step=i)`）；第 i 步探测得到的东西写入新一代，第 i+1 步才可见。这模拟的是并行设计里“探测和推理同时进行”，第 i 步的探测不能影响第 i 步的决策。

---

## 1. 建图：两张在线图

full_rec 同时维护两张图，都**从空图开始，只在本 episode 内存在**（没有传 `--app_memory`，所以没有跨任务记忆）。

### 1.1 屏幕图 `ProgressiveBeliefGraph`（原设计）

**节点 = 一个“屏幕类型”**。节点 id 只由 activity 和交互骨架决定，不包含屏幕上的文字：

```python
# belief_graph.py:243
def make_node_id(activity, layout_signature, _unused=""):
  raw = f"{activity}\0{layout_signature}".encode()
  return hashlib.sha256(raw).hexdigest()[:20]
```

```python
# state.py:277  交互骨架：只看可交互控件的 class + resource-id + 粗网格位置
def layout_signature(elements, grid=64, ignore_overlays=True):
  actionable = [e for e in elements
                if (e.clickable or e.scrollable or e.checked is not None)
                and not (ignore_overlays and _is_overlay(e))]   # 键盘、状态栏不算
  kind_counts = Counter((e.class_name, e.resource_id) for e in actionable)
  records = sorted({(e.class_name, e.resource_id,
                     () if kind_counts[(e.class_name, e.resource_id)] > 1   # 重复的行只记种类
                     else tuple((c // grid) * grid for c in e.bounds))     # 单例控件记位置
                    for e in actionable})
  return sha256(json.dumps(records)).hexdigest()[:20]
```

这样设计的原因：
- **文字被排除**：文字是数据，不是“哪个屏幕”。
- **重复控件只记种类**：3 行和 4 行的列表算同一个屏幕。
- **键盘和状态栏被排除**：打开键盘不会产生一个新节点。

节点里还保存两份“严格签名”，**只用来校验，不用来做身份**：`structural_signature`（含文字、16 像素网格）和 `visual_signature`（pHash）。

| 节点字段 | 含义 |
|---|---|
| `activity`, `package`, `layout_signature` | 身份 |
| `structural_signature`, `visual_signature` | 严格校验（回滚是否恢复、落点是否一致） |
| `salient_ui_labels` | 屏幕上可交互控件的标签 |
| `ui_elements` | 控件清单：`{label, control_key, role, clicks, probes}`，记录每个控件在这里被按过、被探测过多少次 |
| `visit_count` | 本 episode 实际站上这个屏幕的次数 |
| `decision_entropy` | 这个屏幕上“还剩多少可选的”，见 §5.3 |
| `status` | COMMITTED（模型真实到过）/ SPECULATIVE（只被探测到过） |

**边 = 在某个屏幕上对某个控件做的某个动作**。边 id 不含目的地，也不含点击坐标：

```python
# belief_graph.py:270  add_speculative_transition
key = f"{src_node}\0{canonical_action(action)}"   # canonical_action 用 control_key，不用 (x,y)
edge_id = sha256(key)[:24]
edge.observed_destinations += (dst_node,)        # 目的地是统计量：同一动作可能落在不同屏幕
```

| 边字段 | 含义 |
|---|---|
| `status` | `UNEXPLORED → SPECULATIVE → INFERENCE_ALIGNED → VERIFIED / REUSABLE`，失效为 `STALE / INVALID` |
| `observed_destinations`, `discovered_labels` | 落在哪些屏幕、落点上新出现了哪些标签 |
| `path_probability`, `confidence` | 模型会走这条边的估计 |
| `inference_alignment_count` | 探测过的这条边，后来被模型自己选中的次数（前缀对齐） |
| `execution_hit/miss_count` | 真实执行或跳步执行时，落点是否符合记录 |
| `probe_count`, `rollback_success/failure_count`, `inverse_level` | 探测次数、回滚成败、用哪一级回滚恢复的 |
| `realized_information_gain`, `exploration_cost` | 探测实际带来的新信息、花的时间 |
| `skip_attempt/success_count` | 被跳步复用的次数和成功次数 |
| `risk_level`, `dynamic_content` | 风险等级；是否内容动态（不可复用） |

**边的三个来源**：
1. **模型的真实转移**：第 9 步，`add_speculative_transition` 后立即 `record_execution_verification(..., True)`。落点标签只保留**相对源屏幕新增的**部分，避免把 app 的固定工具栏当成落点内容。
2. **探测转移**：第 7 步，探测结果作为 SPECULATIVE 边写入，带回滚信息和代价。
3. **前缀对齐提升**：探测过的边后来被模型选中 → `INFERENCE_ALIGNED`；它的目的地下面已经探测过、并且能回滚的子边 → `REUSABLE`（`belief_graph.py:409`）。

### 1.2 流程图 `ProcedureGraph`（v2 + full_rec 的记录槽位）

屏幕图在短 episode 里几乎不会被重访（严格签名下重访只占 5% 的步）。真正重复的是**按实体重复的流程**：删 A、B、C 时，每个实体走的是同一段决策。流程图的单位就是这种片段。

**“跳”（Hop）= 模型真实执行的一次按压或输入**，在下一步开始、画面落定之后记录（`finalize_procedure_hop`，runner 第 2838 行）：

```python
# procedure_replay.py:136（字段，删节）
@dataclass(frozen=True)
class Hop:
  step: int
  source_activity: str
  resource_id: str; class_name: str
  own_label: str                 # 控件自己的 text / content-desc
  inner: tuple[str, ...]         # 控件 bounds 内所有子节点的标签（列表行的名字在这里）
  entity: str                    # 被按控件包含、或输入文字恰好等于的任务实体 → 实体槽位
  target_activity: str
  target_labels: tuple[str, ...]
  state_ok: bool | None          # 检查器：True 不改持久状态 / False 改了（提交）/ None 未覆盖
  barrier: bool = False          # 滑动、作答、输入非实体内容 → 流程边界
  action_type: str = "click"     # click / long_press / input_text，按演示的方式回放
  field: str = ""                # full_rec：这一跳输入或选择的是该实体的哪个属性（记录槽位）
```

**状态证书**（`app_state_checker.py`）：动作执行前，用 root 权限对该 app 的 SQLite 表（单个事务、含 WAL）、文档目录（目录树 + 文件 md5）、shared_prefs 各做一次只读快照；动作落定后再做一次。两次完全一致记 `state_ok=True`，有任何变化记 `False`，app 不在覆盖范围或读取失败记 `None`。证书只看“持久状态变没变”，不看界面像不像。

**任务契约**（`obligation_entities`，runner 第 2816 行）：图里第一次出现“若干跳之后接一次改状态的跳”时，才发起一次纯文本模型调用。full_rec 用的是带属性的版本：

```python
# procedure_replay.py:47
OBLIGATION_RECORDS_PROMPT = (
  "... For each item give its name and every attribute value the task states for it, "
  "copying all text EXACTLY as written in the task. Answer with JSON only: "
  '{"items": [{"name": "...", "fields": {"attribute": "value"}}]}.\nTask: ')

# procedure_replay.py:55  parse_obligation_records：名字和每个值都必须在目标原文里逐字出现
if len(_norm(name)) < 3 or _norm(name) not in goal_n: continue
if not isinstance(value, (str, int, float)) or _norm(value) not in goal_n: continue
value = re.sub(r"^[^\w]+|[^\w]+$", "", str(value).strip())   # "$67.41" → "67.41"
```

契约发起之后，已经记录的跳会**重新绑定**（`set_entities`、`set_records`），因为标签和输入文字都存下来了。

**模板 = 最近一次完整演示的导航部分**（`template()`，第 292 行）：

```python
# 从最近一次提交（state_ok=False）往回走，直到上一个边界
start = commit
while start > 0 and not self._is_boundary(self.hops[start - 1]):
  start -= 1
run = self.hops[start:commit]                  # 不含提交本身
if len({h.entity for h in run if h.entity}) != 1: continue   # 必须恰好绑定一个实体
run = self._with_fields(run, demonstrated)     # full_rec：输入值 / 按下的选项 = 该实体的某个字段 → 记录槽位
if any(h.action_type == "input_text" and not h.entity and not h.field for h in run):
  return None                                  # 输入了既不是实体也不是字段的内容 → 不可回放
return self._validated(run)                    # 实体跳之前的不可回放跳裁掉；之后的直接作废
```

边界的定义（`_is_boundary`）：屏障跳、提交跳、以及输入的文字既不是实体、也不是契约中任何字段值的跳。

**例子**：RecipeDeleteMultipleRecipesWithNoise 第一个实体的演示与模板：

```text
hop0  click  Search                      fixed       state_ok=True
hop1  input  "Chicken Alfredo Pasta"     entity      state_ok=True
hop2  click  Recipe photo（过滤后唯一）   fixed       state_ok=True
hop3  click  More options                fixed       state_ok=True
hop4  click  Delete                      fixed       state_ok=True
hop5  click  DELETE                      ← commit（state_ok=False），模板到此为止，不含它
```

### 1.3 两张图对比

| | 屏幕图 | 流程图 |
|---|---|---|
| 单位 | 屏幕类型（activity + 交互骨架） | 模型执行过的一跳（控件 + 实体 / 字段槽位） |
| 何时有用 | 屏幕被重访、深度 2 探测和模型路径对齐 | 同一任务里多个实体走同一流程 |
| 写入来源 | 真实转移 + 探测 + 对齐提升 | 只有真实转移（附检查器证书） |
| 被谁消费 | 探测打分、跳步门、上下文蒸馏 | 回放执行器、探测的实体前沿 |

---

## 2. 图节点如何匹配到当前页面

| 场景 | 匹配方式 | 代码 |
|---|---|---|
| 当前屏幕 → 屏幕图节点 | `node_id_of(sig) = make_node_id(activity, layout_sig)`，精确哈希匹配 | runner 第 1879 行 |
| 跳步落点校验 | 落点节点 ∈ `observed_destinations`（或者有多个目的地时，离开了源节点） | runner 第 3486 行之后 |
| 探测回滚校验 | 严格签名：structural / pHash / activity，分级比较 | `live_probe._recover` |
| 流程图模板起点 | 当前 activity == 模板某跳的 `source_activity` | `ProcedureGraph.plan` |
| 实体行 | 可点击、resource-id 和 class 与模板一致、**内部标签恰好包含下一个实体**、在屏幕上唯一 | `_entity_controls` |
| 固定控件 | resource-id + class + 命名标签（自身或内部），或在屏幕上唯一的 resource-id；要求唯一 | `relocate_fixed` |
| 输入框 | resource-id + class 唯一；回放时先清空（`clear_text=True`） | `relocate_field` |
| 记录字段选项 | 内部标签等于该实体对应字段值的控件（如类别 “Food”） | `execute()` 的 `hop.field` 分支 |
| 落点 | activity 一致；实体跳的落点上必须出现**新**实体的名字 | `execute()` |
| 屏幕外控件 | 边界超出屏幕尺寸的不算，重新截图最多 3 次（等动画结束） | `_on_screen` |

---

## 3. 探索如何利用图（第 7 步）

### 3.1 什么时候探

```python
# runner 第 3702 行（删节）
may_probe = (args.probes_per_step > 0
             and not _unprobeable(package)                      # 系统设置等不探
             and not (args.probe_only_certifiable                # full_rec：检查器不覆盖的 app 不探
                      and not state_checker.covers(package))
             and not dirty_from_last_step                        # 上一步没能恢复 → 这一步不探
             and step >= 2                                       # 开头两步不探
             and probe_total < args.probe_budget                 # 每个 episode 最多 12 次
             and not unrecovered_once                            # 恢复失败过一次 → 本 episode 停探
             and node_visits[src_id] >= args.min_node_visits     # full_rec 用默认值 2：只探重访屏
             and not _has_unsaved_input(before))
```

`--probe_only_certifiable` 的理由（见代码注释）：r3 里 Files app 一次探测没能恢复，又在重新启动时卡住，整个任务因此失败。检查器不覆盖的 app，探测有没有副作用无法验证，就不去探。

### 3.2 候选集与安全门

候选是屏幕上的可交互控件，按类型分成 `TAP_NAV / TAP_MENU / EXPAND / SCROLL`，只用这四种，因为点击和滚动的逆操作最容易建立。列表行这类“动态集合项”平时会被过滤掉；`--probe_entity_rows` 例外放行**内部标签包含某个未处理任务实体**的行，目的是让探测去定位、并预先认证下一个实体：

```python
# live_probe._safe_candidates：entity_row_targets = 流程图里还没处理的实体
entity_targets = {norm(x) for x in context.get("entity_row_targets") or ()}
```

`SafetyGate`（`state_graph_information.py:361`）是排序之前的硬过滤：元素被封锁、这类探测在这个 activity 上恢复失败过、历史上离开过包、风险是 HIGH/IRREVERSIBLE、**估计可恢复率低于 0.55**，这些候选都不进入排序。

### 3.3 打分：每个候选一行“状态-图信息矩阵”

`StateGraphInformationMatrix.build` 为每个候选填一行 `CandidateInformationRow`，约 60 个字段，分 5 组：

| 组 | 主要字段 | 来自图的哪里 |
|---|---|---|
| 节点 | `node_visit_count`, `node_decision_entropy`, `outgoing_edge_count`, `exploration_coverage` | 当前节点 |
| 精确历史 | `edge_status`, `probe_count`, `inference_alignment_count`, `execution_hit/miss`, `rollback_*`, `mean_realized_ig`, `mean_exploration_cost` | 这个控件在这个节点上的边 |
| 目的地 | `destination_known`, `destination_visit_count`, `destination_entropy`, `destination_subtree_size`, `destination_in_recent_path` | 这条边的目的节点 |
| 需求匹配 | `target_match`, `unresolved_information_match`, `expected_affordance_match`, `risk_conflict` | 候选标签与“信息需求”的词重叠，见 §5.5 |
| 安全与代价 | `estimated_recoverability`, `expected_total_exploration_cost` | 回滚先验 + 本 episode 实测 |

`PredictiveElementScorer`（第 822 行）用显式公式把一行压成一个分数，不需要训练，每一项都单独记日志：

```text
utility = path_probability × expected_IG / expected_cost
```

```python
# 1) 模型会走这里的概率
prior = 0.30 + 0.45 × max(target_match, unresolved_match, affordance_match)
prior = 0.5·prior + 0.5·contextual_alignment_rate        # 同类上下文里的对齐率（如有）
prior ×= 1 + (0.5 − norm_y)                               # 位置先验：越靠上越可能
prior ×= 1.5 if destination_known; ×= 1.5 if destination_in_recent_path
p = shrink(inference_alignment_count, aligned+missed, prior, strength=3)   # 有精确历史时向实测收缩

# 2) 探了能新增多少“图信息”（不是新颖界面）
base = mean_realized_ig 或 0.5（先验）
gain = base × (0.5 + relevance) + 0.25 × ui_novelty
gain ×= 0.25 if destination_known          # 已知目的地，再探只能确认
gain ×= 0.5  if 探过 ≥2 次、无关、且没有新标签
gain ×= 0.5  if destination_in_recent_path

# 3) 代价：探测时间 + 回滚时间 + 回滚失败的期望惩罚
cost = probe_latency + 1.4 + (1 − recoverability) × 12 s
```

`realized_IG` 的定义很朴素：一次探测在落点上新发现的元素数除以 10，上限 1（runner 第 2222 行）。它只是 UI 新颖度的代理指标，**不是**真正的信息增益。

### 3.4 任务前沿：让探测去补流程图的缺口

流程图一旦有了契约，探测器的目标就换成**还没处理的任务实体**（runner 第 3783 行）：

```python
"task_navigation_frontier": ... or bool(procedure_targets()),
"task_navigation_targets":  procedure_targets() or ...,   # 未完成、未失败的实体
```

`frontier_scores`（`task_navigation.py:434`）给每个安全候选加一个前沿分，加到 utility 上（除以代价）：

```python
direct     = |targets ∩ 候选标签| / |targets|                 # 候选本身就带着目标实体
downstream = max(value[dst])                                   # 它通向的屏幕离目标有多近（沿边反向传播 4 轮，每跳 ×0.8）
unknown    = 1 / (1 + probe_count)                             # 没探过的出口更值得一探
score      = direct + downstream + 0.25 × unknown ；已知 no-op 记 0
```

### 3.5 执行、恢复与写回

探测在推理之前、从 S_i 出发执行（串行 runner），每次探测后回滚。恢复按逐级升级的阶梯进行：`NOOP / INVERSE / INVERSE_ANCHOR / BACK_N / BACK_OVERLAY / DEEPLINK / TRAJECTORY_REPLAY`，最后一级是按模型已提交的动作从主屏重放一遍。

结果写回屏幕图的规则：
- 回滚成功 → SPECULATIVE 边，带 `rollback_success`、`inverse_level`、代价、`realized_IG`；
- 回滚失败 → 封锁这个 `(activity, probe_type)`，本 episode 停探；
- 检查器在窗口前后各做一次快照，窗口级证书写进 witness；
- 如果模型下一步正好选了探测过的那个控件（前缀对齐），它下面已探测的子边提升为 REUSABLE，供 §5.3 跳步使用。

---

## 4. 推理如何利用图：四条路径

按第 0 节的顺序，一步里最多发生三件事：先流程回放，再跳步或注入上下文，最后正常推理。

### 4.1 流程回放（流程图 → 替模型做导航，停在提交前）

```python
# procedure_replay.py:350  plan()，删节
undone = [e for e in entities if e not in done and e not in failed][:1]   # 只取任务顺序里的下一个
values = records.get(undone[0])                                           # full_rec：它的字段值
if {h.field for h in template if h.field} - set(values): return None      # 缺字段值 → 不回放
# 起点 1：下一个实体已作为模板的“实体按压”出现在屏幕上，且唯一 → 从那一跳开始（最短）
# 起点 2：从后往前找第一个可行起点：控件在屏、并且实体会在被按之前先被输入（搜索子流程）
```

`execute()`（第 497 行）逐跳执行：重新定位（实体行 / 固定控件 / 输入框 / 字段选项）→ 执行 → 截图 → 校验落点 activity，实体跳还要校验新实体出现在落点上 → 任何一跳不符就停，从真实屏幕交还给模型。

**停止点 = 演示中第一个改持久状态的跳之前**。提交、确认、COMPLETE 永远由模型自己做。

回放时写进模型历史的内容（不改变模型对进度的认知）：

```text
Now handling 'Garlic Butter Shrimp' the same way as 'Zucchini Noodles with Pesto': clicked 'More options'.
The final confirmation for 'Garlic Butter Shrimp' is still pending.
...（模型确认 DELETE 之后，检查器看到持久状态改变）
Confirmed for 'Garlic Butter Shrimp'. Items from the task handled so far: 'Zucchini ...', 'Garlic ...'.
Still NOT handled: 'Lentil Soup'.
```

回放失败的实体会进入 `failed` 集合：不再自动重试，**但也不算完成**；同时告诉模型 “'X' is NOT done yet”。

**full_rec 的记录槽位**让“新增多条记录”的任务也能复用模板。第一条记录的演示是 `FAB → 输入名称(实体) → 输入金额(字段 amount) → 选择类别(字段 category) → SAVE(提交)`；回放第二条时，名称、金额、类别都换成契约里它自己的值，仍然停在 SAVE 之前。

### 4.2 记忆复用：提交控件拦截（两张图的交叉）

流程图记录下的所有提交控件（`state_ok=False` 的 control_key）会进入 `proc["commit_controls"]`。屏幕图的跳步路径一旦包含其中任何一个，就在它之前截断（runner 第 3534 行）。这样旧的跳步机制也不会替模型按下提交。

### 4.3 图跳步（屏幕图 → 整步不调用模型）

`reusable_edge`（runner 第 152 行）只接受两种来源：

1. **REUSABLE 边**：前缀对齐提升出来的前瞻边（§3.5）。
2. **“已定”屏幕**：`decision_entropy ≈ 0`、被访问 ≥2 次、只剩一条 VERIFIED/REUSABLE 出边、这条边被模型自己执行过 ≥2 次、被跳步复用不到 2 次，并且自上次执行以来图里有新节点出现（防止在原地打转）。

决策熵（`belief_graph.py:576`）：

```text
H_i = − Σ_j p_j log p_j ，p_j ∝ 当前节点各条可行出边的 path_probability
```

只剩一个可行延续的屏幕，H = 0，在这里推理没有可选的。完全没有可行出边的屏幕，H = ∞，永远不会被当成“已定”。

`ReasoningGate._skip_blocked`（`graph_distiller.py:493`）在以下情况拒绝跳步：
- 连续跳步达到 3 次；
- 目的地在最近 4 个节点里，并且执行次数 < 2（防循环）；
- 边风险高；
- 出边多于 1 条并且 H > 0.4；
- 路径中含提交控件（§4.2）。

在这之前，runner 还有一道配额（第 3516 行）：每个 episode 最多尝试 2 次图跳步，之前的跳步只要有一次落点没命中就不再跳。代码里还写了“前两次都命中后允许最多 `--max_path 3` 跳”，但这个条件（已尝试 ≥ 2 次）和配额（已尝试 < 2 次）互斥，**所以实际上每次跳步只走 1 跳，`max_path` 在这个配置下从不生效**。

执行时逐跳校验落点，命中与否写回 `execution_hit/miss` 和 `skip_success`。

### 4.4 图上下文注入（屏幕图 → 1 跳事实 → prompt）

不跳步时，`GraphDistiller.distill` 只取**当前节点的出边**（一跳），每条边转成一条事实：

```python
# graph_distiller.py:198  _to_fact（删节）
certainty, weight = {"REUSABLE": 1.0, "VERIFIED": .95, "INFERENCE_ALIGNED": .75,
                     "SPECULATIVE": .45, "UNEXPLORED": .3}[status]
need_match  = |need_tokens ∩ tokens(边标签 + 落点新标签)| / |need_tokens|
historical  = hits / (hits + misses)   或 skip 成功率 或 0.5
freshness   = exp(−age_generations / 6)
fact_type   = DONE（本屏已走过） | VERIFIED | OBSERVED | NO_RELEVANT_EVIDENCE（探过、落点没有新标签）
utility     = (0.25 + need_match) × weight × (0.5 + 0.5·historical) × freshness
              （DONE 记 0.4·freshness，NO_RELEVANT_EVIDENCE 记 0.3·freshness）
```

保留 utility ≥ 0.15 的事实，按分数取前 3 条，总长度不超过 64 个 token，否定事实最多 1 条。如果一条正面事实（VERIFIED 或 OBSERVED）都没有，就整段不注入。渲染格式：

```text
[Memory]
Need: Garlic Butter Shrimp
Verified: More options -> {Edit, Delete, Share}.
Done: Search already used.
```

### 4.5 “信息需求”从哪来、怎样总结

探测打分和上下文蒸馏都要知道“模型现在想找什么”。这个信息**不额外调用模型**，而是从模型上一步已经付过成本的输出里解析出来（`information.py:104`，`parse_reasoning_prior`）：

```python
reasoning = <THINK>…</THINK> + explain: + summary:
target_entity            = 引号里或首字母大写的短语（第一个）
required_information_slots = 这些短语 + “need to / looking for / must …” 后面的从句（最多 6 个）
expected_affordances     = 文本里出现的动作词（open, edit, delete…）
risk_keywords            = 文本里出现的风险词（delete, send, pay…）
```

**总结的方式**：
- 探测时，这些词与候选控件标签的重叠，进入 `target_match` 等字段。重叠按单个需求项计算，并按较短的一方归一化。`risk_conflict > 0` 的候选直接标为 HIGH 风险，被安全门过滤。
- 注入时，`Need:` 行会去掉句子碎片和重复的子串，最多 4 项。
- 流程回放不使用这条线索，它的任务条件来自 §1.2 的契约，比从推理文字里抽出来的更可靠。

---

## 5. 记录了哪些指标

**图结构上的指标**（每步更新，写入 `progressive_belief_graph.json`、`serial_events.jsonl`）：

| 指标 | 定义 | 用在哪 |
|---|---|---|
| `visit_count` | 节点被真实访问的次数 | 探测门（重访）、跳步“已定”判定 |
| `decision_entropy` H | 出边 path_probability 的熵 | 跳步门 |
| `alignment_rate` | 对齐次数 /（对齐 + 未命中） | 探测打分的 path_probability |
| `execution_hit_rate` | 落点命中率 | 上下文注入的 historical、跳步资格 |
| `rollback_success_rate`, `estimated_recoverability` | 回滚成功率（向类型先验收缩，强度 3） | 安全门（≥0.55）、代价 |
| `mean_realized_ig` | 平均新元素数 / 10 | 探测 IG |
| `mean_exploration_cost` | 平均探测耗时 | 代价 |
| `skip_success_rate` | 跳步成功率 | 上下文注入、跳步配额 |
| freshness | exp(−代数 / 6) | 上下文注入 |
| 状态证书 | 持久状态是否改变 | 流程模板的终点、提交控件 |

**每个候选和每条事实**（`scored_candidates.jsonl`、`graph_context` 事件）：path_probability、expected_IG、expected_cost、recoverability、ui_novelty、utility、前沿分；以及事实的 need_match、certainty、historical、freshness、utility_score。

**流程图**：`procedure_plan` 的拒绝原因（`no_entity_demonstration`、`demonstration_not_completed`、`uncertified_hop`、`typed_value_not_a_field`、`record_missing_field`、`entity_not_visible`、`entity_ambiguous`、`selector_missing_or_ambiguous`…）；`procedure_execute` 的 attempted / completed 跳数和结束原因（应当全部是 `stopped_before_commit`）；进度账。

**实验层**（`run_full_paired.py` → `pairs.jsonl`）：官方 evaluator 给出的成功与否、`agent._actions` 步数、实际 HTTP 模型调用数（含实体抽取）、prompt / completion tokens、每个请求的截图和输入输出指纹（用于复现性审计）、总耗时、宏次数和跳数，以及四条停止规则的计数。

---

## 6. 设计边界与需要注意的点

1. **记录槽位越过了“只回放导航”的边界**：`--procedure_records` 会替模型输入金额、选择类别这类内容字段，只是仍然停在保存之前。这些值逐字来自任务原文（解析器强制要求），但输入内容属于“编辑”，不属于“导航”。是否可以接受，要在论文里明确表态。
2. **两种复用机制叠加**：`--enable_skip`（屏幕图跳步）和流程回放同时开着。省下的步数需要分别从 `skip` 事件和 `procedure_execute` 事件里统计，才能归因。
3. **探测门比较保守**：`min_node_visits=2`、`step>=2`、每个 episode 最多 12 次、一次恢复失败就停探。在短 episode 上，屏幕图的探测和跳步很少有机会触发（与此前的测量一致：重访只占 5%）。
4. **图跳步实际被限制为每个 episode 最多 2 次单跳**（§4.3 配额的两个条件互斥），多跳跳步是死代码。
5. **检查器是窗口级的**：同一窗口里先改后复原的情况看不见；只观测 7 个 app 的本地数据，不观测网络和云端。
6. **全部是 edge-assisted 串行 runner 的结果**：模型在远端 GPU，UI 在本机模拟器，探测在关键路径上。延迟数字不能代表并行设计或手机本地部署。
