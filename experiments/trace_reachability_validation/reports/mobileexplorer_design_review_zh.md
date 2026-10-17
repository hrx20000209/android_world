# MobileExplorer 设计评审与重构方案（AndroidWorld）

> 范围：仅 AndroidWorld；不涉及端侧资源调度 / 模型服务 / 硬件。
> 本文严格区分：**[已实现]**（代码中存在且在全量实验中运行过）、**[部分实现]**（代码存在但触发极少或只在部分臂中）、**[文档中计划/未实现]**、**[本文提议]**。
> 路径相对仓库根目录 `experiments/`。所有数字来自本仓库的全量单轮实验（114 个任务，seed 30，步数 = 模型推理调用次数，含重新推理），除非注明为离线分析。

---

## A. 现状总结（Current Design Summary）

### A.1 机制清单与代码位置

| # | 问题 | 现状 | 代码 | 状态 |
|---|---|---|---|---|
| 1 | UI 状态表示与匹配 | a11y 树解析为 `UIState`；四级签名：`exact_sig`（全部节点含 bounds）、`struct_sig`（activity + IME + 骨架 + 非列表锚文本）、`layout_sig`（activity + IME + 骨架）、`content_sig`（列表行文本、输入框内容、勾选状态）。`same_state` = struct 且 content 相同。联合图页面键 = (package, activity, layout_sig) | `runtime_graph_study/scripts/rgs_ui.py:116,162-186`；`rgs_run.py:190 same_state`；`trace_reachability_validation/scripts/joint_graph.py:20 page_key` | [已实现] |
| 2 | 动作与转移存储 | 观测追加写入 `graph.jsonl`：`{origin, instance, slot_id, selector, src摘要, src_cid, dst摘要, dst_cid, outcome, rollback}`；索引键 `(src.struct_sig, slot_id)`。agent 边与探索边的 slot_id 格式不同（agent 用控件族 `E|…`，探索用 slot），联合图把两者统一为控件族 | `rgs_run.py:195 Graph, 240 add`；`full_run.py ingest_agent`；`joint_graph.py:26 JointGraph` | [已实现] |
| 3 | 历史轨迹记录 | 每个 episode 的 `steps.jsonl`（模型动作、函数/复用动作、pre 截图摘要）；成功 episode 的自身序列追加到 `store.json`；成功 episode 经 `fgraph.extract` 抽取为参数化函数，`build_cache` 汇总为 `function_cache.jsonl` | `full_run.py:888 _run`；`runtime_graph_exploration/scripts/fgraph.py:141 extract, 201 build_cache` | [已实现] |
| 4 | 探索轨迹的生成与存储 | 与推理并行的探测窗口：E1（预测下一步）、FX（未知控件，统一价值排序）、D2（经已知可回退跨 activity 边的二层）、RV（按"边+页面内容"重新观测）；每次推理最多 5 次；探测记录与 agent 观测写进同一 `graph.jsonl`（origin=explore） | `full_run.py:488 window`；`rgs_run.py:387 probe`；`full_run.py:723 _depth2_target, 774 _probe_depth2` | [已实现] |
| 5 | 是否区分探索证据与执行证据 | 只区分 **来源**（`origin`）。**不区分"是否被成功任务验证"**：图中的边没有 task 成败字段；成败只体现在 `store.json`（成功序列）和 `function_cache`（成功轨迹抽取的函数）里，二者与图是分开的 | `joint_graph.py:28`（origins 集合）；`full_run.py:888` | [部分实现] |
| 6 | 探索目标选择 | 非 best-first 搜索，而是每窗口的打分选择：统一图价值 `GraphValue`（rel = 目标操作词 × 证据文本的 TF-IDF 余弦；证据分层 L1 已知边目的页 / L2 同标签近邻 / L3 自身文本；词权重 = 全局界面 IDF × 当前图 app 局部 IDF 的几何平均；价值回传 V=max(rel, 0.7·maxV(子页))）+ 页面级去重 + 可回退性门槛 P_rev≥0.7（角色×位置先验 + 容器 Beta 后验）+ 列表行的参数匹配 | `scripts/explore_value.py:193 GraphValue`；`full9.py:729 p_reversible, 687 container_key, 645 page_family_index` | [已实现] |
| 7 | 探索后回滚 | 恢复阶梯：retap（原地变化）→ 重选兄弟 tab → Back×≤2 → `restore_via_graph`（≤2 跳探索边）→ 联合图安全路径（≤3 跳）→ force-stop + 重启 + 重放自启动以来的动作（仅当路径无提交）→ 内容校验（变了即记为副作用）；Back 失败的控件所在容器整体记为不可回退（跨任务持久） | `rgs_run.py:357,387`；`joint_graph.py:115 restore`；`full9.py:525 relaunch_restore, 518 replayable`；`full_run.py:690-700` | [已实现] |
| 8 | 新任务如何检索图知识 | (a) app 名 → 启动；(b) 函数库按"动词 + 入口 activity + 字段值齐全 + 首步可落地"检索（跨模板留一法）；(c) 表单函数按字段名 ↔ 输入框标签落地；(d) 联合图 ≤3 跳路由到函数可落地的页；(e) 探索价值打分读图 | `full7.py:256 resolve_app`；`fgraph.py:241 Library.candidates`；`full9.py:127 try_apply, 314 _form_plan, 344 try_form`；`joint_graph.py:132 try_route` | (a)(b)(c)[已实现]；(d)[部分实现，全量中 ≤3 次] |
| 9 | 何时跳过 LLM 推理 | `open_app`（桌面 + 目标中 app 名且图见过其包）；`confirm`（模型自身提交后的纯确认对话框）；`fill_record`（库函数首步落地）；`form`（≥1 个本屏空字段与目标字段名匹配）；`graph_reuse`（本局同页两次同一导航且离开本页）；另：决策有效性放行 / 决策重定位 / 恢复后执行原决定（避免重新推理，FULL22） | `full_run.py:136,153,177-199,279-300,799 _try_reuse` | [已实现] |
| 10 | 跨任务复用 | 跨任务：函数库（不同模板同动词）、open_app、图边用于价值/去重/恢复/路由；**跨任务的多步导航直接执行几乎没有**（路由 ≤3 次/轮，执行后函数落地 0 次；同任务多跳跳转 FULL9 起关闭，因跨任务跳转导致 Calendar 失败）。跨 run：`function_cache` 来自先前 run；图在一个臂内跨任务持久，臂之间不共享 | `full_run.py:378 plan_jump`（已停用） | 函数级[已实现]；导航级[部分实现] |
| 11 | 支撑实验 | FULL8–FULL22 全量单轮；FULL19、FULL21 各有同配置重复；离线：`eval_explore_value.py`（探索排序）、`joint_matrix.py`（来源×结果矩阵）、`function_view.py`、`rev_prior.py` | `scripts/*.py`；`reports/full11_13_results_zh.md` §1–16 | — |

### A.2 当前最好结果（FULL21，两轮复现）

| | 成功率 | 平均推理步数 | 成功任务平均推理步数 |
|---|---|---|---|
| FULL21 ×2 | 101/228（44.3%） | 9.28 | 6.24 |
| BASE | 51/114（44.7%） | 10.05 | 7.41 |
| 4B_2 | 54/114（47.4%） | 10.75 | 7.85 |

推理节省的来源（每轮）：open_app 94、confirm ~23、fill_record ~28 次（~87 动作）、form 9 次（27 字段）、本局复用 7–14 次，合计 ≈17.6% 决策；代价：重新推理 ~32 次。

### A.3 局限（有代码与数据支撑）

1. **节省几乎全部来自"过程级"复用，而非"图级"复用。** 图里大量边从未被直接执行：FULL21 中模型 431 次跨页导航，有 201 次（47%）走的边已被其他任务观测过，落地预测准确率 91–100%（`joint_matrix.py`），但这些步全部仍由模型推理。
2. **边上没有"任务语义"。** 图只回答"点这里会到哪"（可达性），不回答"这一步对哪类目标有用"。路由只能把"函数首步可落地"当目标，而函数覆盖仅 ~20/114 个任务。
3. **探索证据与执行证据没有统一的可信度模型。** 只有 origin 标记；成功任务对边的"验证"信息存在于 store/函数库，图本身不累积。
4. **状态身份粒度不对。** `layout_sig` 随内容变化会碎片化（同一详情页因记录不同而成不同节点；离线：记录循环用 layout 键 70% 正确，activity 键仅 39%），而 activity 又过粗（单 activity app 的 Markor）。隐藏上下文（Markor 当前文件夹、Calendar 当前选中日期）完全不在状态里——这就是 FULL18 "上一个月"翻过头、FULL17 路由进 `.app` 文件夹的根因。
5. **探索的作用以"间接"为主。** 探索贡献约 24% 的落地页、仅探索边的落地预测 100% 准确，但被直接执行利用的极少；加大探索量（FULL17）只增加重新推理。

---

## B. 核心研究缺口（Core Research Gap）

**图中存储的是"可达性"（reachability），而跳过推理需要的是"对当前目标的进展"（goal progress）。**

现有系统（以及大多数 state-action graph / trajectory cache 工作）把这两件事混在一起：要么只在"完全相同的任务"上回放轨迹（progress 由同任务保证），要么在图上做搜索但不知道哪条路服务于当前目标。跨任务复用失败的根本原因不是"没见过这条边"（47% 的模型导航边已在图中），而是**缺少一个能把不同来源的边按"对当前目标的作用"打分、并判断何时可以不经模型执行的统一模型**。

单一最重要的缺口：**边的目标条件化效用（goal-conditioned edge utility）——由探索提供"这条边打开了什么内容"（unlock 证据），由成功执行提供"这条边曾为什么目标服务"（use 证据），在同一个节点/边上合并，用来做跨任务的多步路径组合与跳过决策。**

离线证据表明这条路可行但必须带校验：用 TF-IDF（目标操作词 + 参数词 × 目的页内容）给当前页的跨任务已知边排序，top-1 命中模型选择 55%（136/248）；只取最有把握的 20%（占全部导航 11%），准确率 88%；当前页只有一条跨任务已知边时 80%。**88% 不足以盲目跳过，但足以成为"执行 + 落地校验 + 立即刹车"的候选。**

---

## C. 提议的系统架构（3 个组件）

核心思想（一句话）：**一张两层的统一图——可达层记录"点什么会到哪"（探索与执行同权），意图层记录"这条边打开/服务了什么"（探索给 unlock 证据，成功执行给 use 证据）；导航决策在这张图上做目标条件化的多步组合，只把"导航性"的前缀交给图执行，把依赖任务参数或会提交的动作留给函数或模型。**

### C1. Unified Evidence Graph（统一证据图）

- **输入**：模型动作转移（pre/post 截图摘要）、探测观测、episode 成败。
- **输出**：页面节点、边、每条边的可达统计与意图画像。
- **内部算法**：见 D1。节点 = 角色级页面（activity + layout + 标题锚文本，见下）；边 = (页面, 控件族) → 落地分布。每条边维护：
  - 可达统计：`n_obs[origin]`、落地分布（众数与熵）、可回退性（Back 是否撤销）、风险（提交 / 副作用 / 离开 app）。
  - 意图画像：`unlock`（目的页的 TF-IDF 词向量，来自任何观测，探索为主）；`use`（成功经过这条边的任务的目标操作词向量，来自执行）。
  - 新鲜度：最近一次观测时间与该次目的页 content_sig。
- **与其他组件**：C2 读它做路径组合；C3 读它决定探什么，并把探测结果写回。
- **为什么必要**：现在图只有可达层；成功信息散在 store/函数库，无法与探索边拼接。

### C2. Goal-conditioned Path Composer（目标条件化路径组合器）

- **输入**：当前页、目标（操作词向量 g、参数词向量 p）、C1 的图。
- **输出**：三选一的决策——直接执行一段导航宏（Mode 1）/ 交给模型（Mode 2，可附图证据）/ 无可用知识（纯模型）。
- **内部算法**：见 D2、D4。在可达层上做 ≤3 跳的路径搜索，每跳用意图层打分；路径在"语义前沿"处截断：第一个依赖任务参数（选择某个实体行、输入值）或会提交的动作前停下——参数动作交给函数/表单（已有），提交交给模型。
- **为什么必要**：把"导航可达"与"任务完成语义"分开，是避免把 "Delete Contact" 这类动作错误复用到别的请求的关键。

### C3. Information-seeking Explorer（信息驱动的探索器）

- **输入**：当前页、目标向量、C1 的覆盖与不确定性、可回退性模型。
- **输出**：本窗口的探测目标（最多 K 个）与观测结果（写回 C1）。
- **内部算法**：见 D3。目标不是"新状态数"，而是 C2 做决策所缺的信息：(i) 目标相关但未知的边的 unlock 内容；(ii) **动态信息**——目标参数（实体名、标题、文件名）在哪个页面出现（按 TF-IDF：参数词在界面语料中 IDF 高、在本 app 页面间 DF 低，即"内容型"而非"导航型"词）；(iii) 过期的动态内容（列表随任务改变）用 RV 刷新；(iv) 能把 C2 中断开的路径连起来的边。
- **为什么必要**：现有探索按"目标相关性"排序已有效（FX 选中控件后被模型点击的比例 4.7% → 12.8%），但其产物没有被转成跳过；C3 让探索直接服务于 C2 的置信度。

（与现有代码的关系：C1 可由 `graph.jsonl` + `joint_graph.JointGraph` 扩展而来；C2 的路径搜索与执行可复用 `joint_graph.paths/walk`、函数执行 `full9.FuncExec`；C3 复用 `explore_value.GraphValue`、`full9.p_reversible`、恢复阶梯。）

---

## D. 算法

### D1. 统一图更新

```
# 角色级页面身份：解决 layout 碎片化与隐藏上下文
page_id(ui) = (pkg, activity, layout_sig, title_anchor(ui))
  # title_anchor = 工具栏/标题区的非数字文本（如 Markor 当前文件夹名、Calendar 月份标题）
  # —— 隐藏上下文进入身份，"上一个月"后的页面不再与当前页相同

on_transition(src_ui, control, dst_ui, origin, episode):
  e = G.edge(page_id(src_ui), family(control))
  e.n_obs[origin] += 1
  e.land[page_id(dst_ui)] += 1                 # 落地分布；熵高 = 依赖隐藏上下文
  e.reversible.update(rollback_ok)  if origin == explore
  e.risk = max(e.risk, risk(control, dst_ui))  # 提交/副作用/离开 app/改了输入框
  e.unlock = ema(e.unlock, tfidf(page_tokens(dst_ui)))   # 目的页"打开了什么"
  e.fresh = (now, content_sig(dst_ui))

on_episode_end(episode, success):
  if success:
    for e in edges_traversed_by_agent_or_graph(episode):
      e.use += goal_vector(episode.goal)       # 这条边曾为哪类目标服务（执行验证）
      e.n_valid += 1                           # 探索边一旦被成功任务经过即"验证"
  else:
    for e in edges_executed_by_graph(episode) whose next model action was Back:
      e.n_misuse += 1                          # 图执行后被模型撤销 = 误用
```

合并与冲突：同一 (页面, 控件族) 的多次观测合并为落地分布，不覆盖；结果矛盾（熵高）的边在 D4 中不允许直接执行。动态内容只影响 `unlock` 与 `fresh`，不改变身份。

### D2. 跨任务路径检索与组合

```
compose(ui, g, p):                       # g=目标操作词向量, p=参数词向量
  best = None
  for path in G.paths(page_id(ui), max_hops=3, edge_ok=navigational):   # 来自任何来源、任何任务
    score, ok = 0, True
    for hop in path:
      e = hop.edge
      if e.risk or entropy(e.land) > H_max: ok = False; break
      intent = λ·cos(g, e.use) + (1-λ)·cos(g ⊕ p, e.unlock)   # 执行证据优先，探索证据补充
      score += log(max(intent, ε)) + log(P_land(e))
    if ok and (best is None or score > best.score): best = (path, score)
  # 语义前沿：路径终点是"参数动作或提交动作所在的页"——到这里为止只做导航
  return truncate_at_semantic_frontier(best)

navigational(edge) = 非提交 ∧ 非副作用 ∧ 非列表实体行（除非由参数 p 落地，交函数处理）∧ 不离开 app
```

校验（执行时，每跳）：重新按控件族在实时屏上定位（找不到即停）→ 点击 → 等待稳定 → 落地页面 id 与预测一致（至少 activity + title_anchor 一致）→ 否则立即停止并交还模型，记 misuse。

### D3. 自适应探索选择

```
explore_targets(ui, g, p, K=5):
  C = safe_controls(ui) with P_rev ≥ τ_rev (或 τ_risky 且可重启重放)        # 现有
  for c in C:
    e = G.edge(page_id(ui), family(c))
    if e is None:                                   # 未知：价值来自文本证据（L2 近邻 / L3 自身）
      gain = rel(g ⊕ p, evidence_text(c))
    elif stale(e) and e.unlock has dynamic terms:   # 已知但动态内容过期：重新观测（RV）
      gain = rel(p, dynamic_terms(e.unlock))        # 参数词是否在其目的页出现
    elif connects_disjoint_components(e):           # 能接上 C2 中断开的路径
      gain = bridge_bonus
    else: gain = 0
    cost = (1 - P_rev(c)) · requery_cost + t_probe
    priority(c) = gain / cost
  return top_K(priority), one per container, gain > 0

dynamic_terms(text) = { t ∈ tokens(text) : idf_ui(t) 高 ∧ df_local(t) 低 }   # TF-IDF 定义的"动态信息"
update after each episode: 若某边因探索获得的 unlock 证据在后续被 D4 用于直接执行且未被撤销 → 该页/容器类型的探索优先级 ×(1+η)
```

计算开销：每窗口 O(|C|·k) 次稀疏余弦（k≤3 份目的页证据），图规模 ~2–3k 观测；目的页截图按需读取并缓存（`explore_value.load_cap`）。未单独测量打分耗时；与之相比，每次探测本身（点击 + 截图 + 回滚）约 1.5–3 秒，是主要开销，且与推理并行。

### D4. 知识感知的导航决策（每步）

```
step(ui):
  1. 函数层（已有）：open_app / confirm / fill_record / form —— 参数化、首步落地即执行
  2. path = compose(ui, g, p)
     if path and DIRECT_OK(path): execute path hop-by-hop with verification; return    # Mode 1
  3. 后台：explore_targets(...) 并行；前台：LLM 推理（可附 path 作为证据，见下）    # Mode 3 ∥ Mode 2
  4. 推理返回后：若屏幕被探测改变 → 决策有效性 / 重定位 / 恢复后执行，否则放弃（FULL22，已实现）

DIRECT_OK(path) :=
    len(path) ≥ 1 ∧ 每跳 navigational
  ∧ 每跳 P_land ≥ 0.9 ∧ entropy(land) ≤ H_max ∧ n_obs ≥ 2（任何来源）
  ∧ intent margin（与次优路径/次优出边的分差）≥ m*      # 离线：top-20% margin → 88%
  ∧ 至少一跳 n_valid ≥ 1 且 cos(g, e.use) ≥ θ           # 有执行验证的同类目标
  ∧ 本 episode 未触发过 misuse 刹车
Fallback：任何一项不满足 → Mode 2（模型推理）；执行中落地不符 → 立即停止，交还模型，记 misuse。
```

关于 Mode 2（把图证据给模型）：本仓库已有测得的负面结论——文字提示模型会照做但图提示正确率约 50% 低于模型自身约 90%，净负（`project-hint-compliance-vs-precision`）；高亮无效。因此**默认不注入**，仅在 path 满足除 margin 外的全部条件时作为候选动作的"预填"实验（prefill 曾优于 injection，`project-prefill-beats-injection`）。

---

## E. 实例（Broccoli 菜谱 app，三个 AndroidWorld 任务 + 一次探索）

页面（角色级 id 简写）：`M`=MainActivity 菜谱列表，`S`=搜索态列表，`D`=RecipeDetailsActivity（标题锚 = 菜谱名），`E`=CreateAndEditRecipeActivity，`O`=详情页溢出菜单（同 activity，弹层）。

1. **任务 A：RecipeAddSingleRecipe（历史执行，成功）**：`M -[New Recipe]→ E`（字段填写由 form/fill_record 完成）`-[SAVE]→ D -[Navigate up]→ M`。
   更新：边 `M·NewRecipe→E` 的 `use += {add, recipe}`、`n_valid=1`；`E·SAVE` 风险=提交；`D·NavigateUp→M` use += {add, recipe}。
2. **任务 B：RecipeDeleteSingleRecipe（历史执行，成功）**：`M -[Search]→ S`（输入"Classic Margherita Pizza"）`-[行:参数]→ D -[More options]→ O -[Delete]→ 确认`。
   更新：`M·Search→S` use += {delete, recipe}；`D·MoreOptions→O` use += {delete, recipe}；`O·Delete` 风险=提交；`S·行` 为参数动作（不进入导航宏）。
3. **探索（任务 B 的推理窗口中，FULL16+ 的 RV/FX）**：在 `D` 上探测 `Edit`→`E`（Back 可回退，`unlock` = {title, servings, ingredients, directions, …}），在 `M` 上重新观测列表内容（RV），得到当前存在的菜谱名（动态信息：菜谱名在界面语料 IDF 高、在本 app 各页 DF 低）。
4. **新任务 C：RecipeDeleteSingleWithRecipeWithNoise（删除"Spicy Tuna Wraps"）**——从未执行过这个模板：
   - 第 0 步（已实现）：`open_app` 进入 `M`（0 次推理）。
   - D2 组合：从 `M` 出发，`M·Search→S`（use={delete, recipe}，与 g={delete, recipe} 余弦高，n_valid=1，P_land=1.0）→ 语义前沿：`S` 上的下一动作是"输入参数"+"选参数行"，截断。DIRECT_OK 成立 → 图直接点 Search（省 1 次推理）。
   - 参数动作：输入"Spicy Tuna Wraps"与点该行由函数层（fill_record 的 slot 步，或 form）完成（已实现机制，省 1–2 次推理）。
   - 在 `D`（标题锚 = Spicy Tuna Wraps）：候选出边 `MoreOptions→O`（use 含 delete，任务 B 验证）与 `Edit→E`（仅探索，unlock 是编辑字段，与 g 的 delete 不相关）。margin 足够 → 直接点 More options（省 1 次推理）。
   - `O·Delete` 是提交 → 交给模型（1 次推理），随后 `confirm()` 按确定（已实现，0 次推理），最后模型终止（1 次推理）。
   - 合计：约 2 次推理。实测对照：BASE 在该任务上 20 次推理且失败；FULL19/FULL21/FULL21R 已能以 3–5 次成功（来自 open_app + 函数层），导航宏的增量是剩下的 Search / More options 两步。
   - **安全性要点**：`D·MoreOptions` 能被复用，是因为它被"删除"类成功任务验证过且在语义前沿之前；`O·Delete` 永远不被复用，因为它是提交；`Edit` 边虽然是真实观测的转移（可达性可信），但其意图与当前目标无关，所以不会被选中——这正是"可达性 ≠ 任务完成语义"的区分。

---

## F. 新颖性与局限

| 对比对象 | 本方案的实际技术差异 |
|---|---|
| 精确轨迹复用（同任务回放） | 复用单位是"边 + 意图画像"，可在从未执行过的任务上组合；同任务回放无法覆盖新模板。证据：47% 的模型导航边已被**其他任务**观测过。 |
| 标准 state-action graph | 普通图边只有可达性；本方案边上带 provenance-typed 的 unlock/use 意图证据，决策是目标条件化的，而不是可达即可用。 |
| 基础图搜索（最短路 / BFS） | 搜索目标不是"到达某状态"，而是"到达语义前沿"，且每跳用意图打分 + 落地分布熵约束；隐藏上下文进入页面身份以避免别名。 |
| ToMe / BPE 式轨迹压缩 | 压缩把高频子序列合并为宏，但宏的适用性只靠出现频次；本方案的宏是在线组合的，适用性由意图 margin + 执行验证决定，且在提交/参数动作处强制截断。 |
| 现有 GUI agent memory / exploration（如 AppAgent 式探索文档、AutoDroid 式 UI 记忆） | 它们把记忆作为提示上下文给 LLM；本方案把记忆用于**不调用 LLM 的执行**，并用探索直接降低该执行的不确定性（unlock 证据、动态信息定位、RV 刷新）。本仓库数据显示提示注入净负，这是选择执行而非注入的依据。 |

**不应夸大之处**：
- 函数层（fill_record/form/open_app/confirm）本身不依赖统一图；统一图的增益只在"多步导航宏"和"探索为宏提供证据"上，需要单独度量。
- 离线 88%@11% 覆盖意味着即使全部做对，额外省下的推理上限约为导航步的 10% 量级（每任务 ~0.4 次）；论文的主张应是"在不降成功率的前提下，把跨任务导航的一部分从推理中移走"，而非数量级提升。
- 隐藏上下文只能部分通过 title_anchor 捕获（例如选中但未显示的状态无法捕获）；落地分布熵是事后的防线。
- 单轮成功数波动 ±10 题：所有成功率结论需要至少 2 轮重复；主指标用"双方都成功任务上的配对推理次数"和机制读数（误用率、刹车次数）。

---

## G. 实施优先级

### P0（最小改动、最高研究价值）

| 项 | 内容 | 复杂度 | 改动位置 | 可能失败 | 验证实验 |
|---|---|---|---|---|---|
| P0-1 | 边的执行验证与 use 画像：episode 结束时把成功任务经过的边写入 `use`/`n_valid`；图执行后被模型撤销记 `misuse` | 低（~80 行） | `full_run._run`、`joint_graph.JointGraph` 读取 | 成功判定只在 episode 末，跨任务顺序影响积累速度 | 来源×结果矩阵新增"已验证"行；验证边的落地准确率 vs 未验证 |
| P0-2 | 意图打分的导航宏（D2+D4 的最小版：1–2 跳、语义前沿截断、margin 门槛、逐跳校验与刹车） | 中（~200 行） | `joint_graph.py` 新增 `compose/execute_macro`；`full_run` 在函数层之后调用 | 误用（aliasing）；margin 阈值过松 | 离线先在 FULL21 日志上回放得到 精度–覆盖 曲线选阈值；在线跑 2 轮，报告 跳过数、误用数、刹车数、配对推理 |
| P0-3 | 探索目标改为"信息增益/代价"：动态信息（参数词 TF-IDF）+ 过期 RV + 桥接边，替换现有 FX/RV 的固定配额 | 中（~120 行） | `full_run.window`、`explore_value` | 增益估计为 0 的窗口变多 → 探测减少 | RQ1：有用发现数（被 P0-2 使用的边 / 被函数落地的页）每次探测的产出 vs FULL21 |

### P1（更强的算法改进）

| 项 | 内容 | 复杂度 | 失败情形 | 验证 |
|---|---|---|---|---|
| P1-1 | 角色级页面身份（+title_anchor）与落地分布熵 | 中 | 标题含动态内容导致碎片化 | 记录循环/复用的离线准确率（现 layout 70%） |
| P1-2 | ≤3 跳组合与跨来源拼接（agent 段 + 探索段） | 中 | 长路径误差累积 | 组合路径中含探索段的比例及其落地准确率 |
| P1-3 | 探索优先级按后续复用反馈自适应（η 更新） | 低 | 反馈稀疏 | 跨 2 轮的探索产出率变化 |

### P2（可延后）

- 跨臂/跨 run 的图持久化（需要处理 AndroidWorld 每任务重置后的数据差异与留一法的泄漏控制）。
- 基于 prefill 的 Mode 2（只在高 margin 但未验证时预填候选动作）。
- 按 app 学习阈值。

---

## H. 评估方案（AndroidWorld 上可执行）

**任务分组**（同 app、共享中间状态、目标不同）：Broccoli（Recipe* 10 个）、Pro Expense（Expense* 8 个）、Markor（Markor* 约 12 个）、Simple Calendar（SimpleCalendar* 约 15 个）。全集 114 个任务照常跑，分组用于分析。

**统一指标**：成功率（≥2 轮重复）、平均推理次数、双方都成功任务上的配对推理次数（主指标）、图直接执行步数、误用（图执行后模型撤销或落地不符）、恢复/刹车次数、端到端时长。

- **RQ1 历史知识是否让探索更高效**：FULL21 探索策略 vs P0-3。度量：每次探测产生"后续被使用"的边/页的比例（用于宏、函数落地、模型选择的落地页由探索首先发现）、冗余探测（同一页面同一内容下重复观测）、Back 失败率、重新推理次数。
- **RQ2 统一知识是否提升跨任务泛化**：五个条件——无记忆（BASE）；仅历史（函数 + 宏只用 agent 边，探索关）；仅探索（宏只用探索边，函数关）；两者独立（函数用历史、宏只用探索边，不合并验证信息）；统一图（P0 全部）。在上述同 app 分组上比较配对推理次数与误用率。注：此前已决定不做"去掉探索"的主线消融，此处作为论文 RQ2 的必要对照单独跑。
- **RQ3 组合路径能否在不降成功率的情况下减少推理**：P0-2 开/关，各 2 轮；报告跳过数、误用数、恢复频次、成功率的合并置信区间、端到端时长。
- **RQ4 探索何时值得**：同一 app 的任务按执行顺序分前/后半，比较探索在当次任务（即时收益：重新推理代价 vs 宏/函数增益）与后续任务（摊销收益：后半段跳过数增量）上的净效果。

**AndroidWorld 约束说明**：每个任务初始化会重置 app 数据，因此"动态信息"（列表内容）跨任务失效，只有结构/导航知识跨任务有效——这正是要把动态内容放进 `unlock/fresh` 而非身份、并用 RV 在当前任务中刷新的原因。跨查询（同模板不同参数）：环境支持，函数层已利用；跨任务（不同模板共享页面）：环境支持，导航宏是新增；跨 run（臂之间）：需要额外的图持久化与泄漏控制，当前不支持。

---

## I. 正在运行与已有证据

- FULL22（FULL21 + 不重新推理：恢复后执行原决定，否则放弃该决定）正在全量运行，结果将补充到 `full11_13_results_zh.md`。
- 本文 §B 的离线数字来自会话临时目录中的分析脚本 `intent_pred.py`（尚未放入仓库；P0-2 实施时应作为 `scripts/intent_pred.py` 正式加入）。
