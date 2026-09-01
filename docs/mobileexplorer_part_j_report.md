# MobileExplorer 交付报告（PART J，2026-09-01，v16）

对应 PART A–I 的实现。所有数字来自本机真实跑测，测量方法在第 13、14、15 节。

---

## 1. 改动的文件

| 文件 | 性质 | 内容 |
|---|---|---|
| `android_world/parallel_exploration/state_graph_information.py` | **新增** | `StateGraphInformationMatrix` / `CandidateInformationRow` / `PredictiveElementScorer` / `ContextualHistoryTable` / `SafetyGate` / `ScoringConfig` |
| `android_world/parallel_exploration/graph_distiller.py` | **新增** | `GraphFact` / `GraphDistiller` / `ReasoningGate` / `DistillerConfig` / `GateConfig` |
| `android_world/parallel_exploration/state_graph_information_test.py` | **新增** | 41 个单测 |
| `scripts/analyze_exploration_metrics.py` | **新增** | PART F 的 F1/F2 指标离线计算 |
| `android_world/parallel_exploration/belief_graph.py` | 修改 | `GraphEdge` 累计统计 + `record_probe/record_skip_result/record_rollback_result` |
| `android_world/parallel_exploration/live_probe.py` | 修改 | 候选选择改为矩阵→安全门→打分器；系统 UI 噪声过滤；回滚阶梯的 Back 越界修复；全量候选落盘 |
| `android_world/parallel_exploration/recovery.py` | 修改 | `TRAJECTORY_REPLAY` 在无法解析启动动作时中止，不再对着桌面重放坐标 |
| `scripts/run_serial_exploration_task.py` | 修改 | 三路门接入、蒸馏器接入、搁浅修复、探索停止规则、消融开关、bootstrap 稳定截屏 |
| `android_world/agents/gelab_agent.py` | **未改动** | 与 `reference_gelab_agent.py` 逐字节相同 |

---

## 2. 新架构

```
                        当前 UI (S_i)
                             |
              +--------------+--------------+
              |                             |
              v                             v
   探索候选抽取 _safe_candidates      渐进式信念图（第 i-1 代快照）
              |                             |
              v                             |
   SafetyGate（硬可行性过滤）  <------------+
              |                             |
              v                             |
   StateGraphInformationMatrix  <-----------+
              |                             |
              v                             v
   PredictiveElementScorer            ReasoningGate（三路）
              |                     /        |         \
              v              NORMAL   GRAPH_ENHANCED   SKIP
        投机探测 + 逆动作回滚          |                |
              |                  GraphDistiller    重放并验证落点
              v                       |                |
        观测 + 回滚验证                 v                |
              |               [Memory] 紧凑上下文        |
              +---------------> 图更新 <-----------------+
                             （下一代可见）
```

时序不变量：step *i* 的一切读取都来自 step *i-1* 的快照。探索进程拿到的是序列化副本，从结构上无法读到当前代。

---

## 3. 候选矩阵 schema

`CandidateInformationRow`，八组。**未观测的统计一律为 `None`，不是 `0.0`**——把"对齐率 0%"存给从没试过的候选，会让它排在已被证明无用的候选之后，正好反了。

| 组 | 字段 |
|---|---|
| A1.1 当前 UI | `element_identity` `text` `content_desc` `role` `probe_type` `norm_x` `norm_y` `clickable` `scrollable` `enabled` `selected` `checked` `nearby_text` |
| A1.2 节点成熟度 | `node_visit_count` `node_decision_entropy` `outgoing_edge_count` `valid_outgoing_edge_count` `explored_element_count` `candidate_element_count` `exploration_coverage` `graph_generation` |
| A1.3 精确边历史 | `has_exact_history` `edge_id` `edge_status` `confidence` `probe_count` `inference_alignment_count` `execution_hit_count` `execution_miss_count` `alignment_rate` `execution_hit_rate` `skip_attempt_count` `skip_success_count` `skip_success_rate` `mean_realized_ig` `rollback_success_count` `rollback_failure_count` `rollback_success_rate` `mean_exploration_cost` `known_inverse_level` `destination_known` `destination_node_id` |
| A1.4 目的地 | `destination_visit_count` `destination_entropy` `destination_out_degree` `destination_valid_out_degree` `destination_subtree_size` `destination_known_label_count` `destination_in_recent_path` `discovered_labels` |
| — | `session_alignment_hits`（本 episode 现场观测到的模型选择，与图里的 `inference_alignment_count` 分开存，两者来源不同不能相加） |
| A1.5 上下文历史 | `contextual_probe_count` `contextual_alignment_rate` `contextual_execution_hit_rate` `contextual_mean_realized_ig` `contextual_rollback_success_rate` |
| A1.6 信息需求 | `target_match` `expected_affordance_match` `unresolved_information_match` `candidate_action_type_match` `risk_conflict` |
| A1.7 安全 | `blocked_element` `blocked_recovery_context` `historical_rollback_success_rate` `historical_deep_recovery_rate` `cross_package_history` `risk_level` `estimated_recoverability` |
| A1.8 代价 | `expected_probe_latency` `expected_rollback_latency` `expected_total_exploration_cost` |
| 导出 | `ui_novelty_score` `path_probability` `expected_information_gain` `predictive_value` `utility` |

A1.5 按 `(probe_type, role, need_type)` 聚合，**不按控件文字**——"列表屏上的菜单探测有 84% 能回滚"是关于交互种类的陈述，可以跨应用迁移；按文字聚合就变成了应用专属规则。

---

## 4. 第一版探索打分公式

```
predictive_value(e) = path_probability(e) × expected_IG(e)
utility(e)          = predictive_value(e) / max(0.5, expected_cost(e))

约束（硬门，在打分之前执行）：
    estimated_recoverability(e) ≥ 0.55
    且 e 未被拉黑、未跨包、risk_level ∉ {HIGH, IRREVERSIBLE}
```

各项：

```
path_probability:
    prior = min(0.95, 0.30 + 0.45 × max(target_match, slot_match, affordance_match))
    if 有上下文对齐率:  prior = 0.5·prior + 0.5·contextual_alignment_rate
    if 本局现场观测过:  prior = max(prior, min(0.95, 0.60 + 0.10 × session_hits))
    if 有精确历史:      Beta-Binomial 收缩 (hits + 3·prior) / (n + 3)

expected_IG:
    base = mean_realized_ig            （有精确历史）
         | 0.5·contextual_mean + 0.25  （有上下文历史）
         | 0.5                          （都没有）
    gain = base × (0.5 + relevance) + 0.25 × ui_novelty_score
    if 目的地已知:        gain × 0.25
    if 探过≥2次且零相关:  gain × 0.5
    if 目的地在最近路径:  gain × 0.5

expected_cost:
    probe_latency（按探测类型，或历史均值）
  + rollback_latency 1.4s
  + (1 − recoverability) × 12.0s
```

先验来自实测（2026-08-31，452 次探测的回滚成功率：TAP_NAV 93%、SCROLL 90%、TAP_MENU 84%、EXPAND 77%）。所有权重集中在 `ScoringConfig` 一个 dataclass 里。

**`new_element_count/10` 不再被当作信息增益**，只保留为 `ui_novelty_score`：满屏陌生标签不等于回答了模型卡住的问题。

---

## 5. 图历史如何改变元素排序

| 情形 | 效果 | 实现 |
|---|---|---|
| 从未探过 | 只用当前 UI + 需求 + 上下文历史的先验 | `has_exact_history=False`，统计为 `None` |
| 反复与模型真实动作对齐 | `path_probability` 上升 | 对齐次数经 Beta-Binomial 收缩 |
| 反复产出有用目的地 | `expected_IG` 上升 | `mean_realized_ig` |
| 反复回滚失败 | **判为不可行**，打分前剔除 | `estimated_recoverability` < 0.55 → `SafetyGate` |
| 反复通向无关分支 | 预测价值下降（不封禁） | `irrelevant_branch_penalty` 0.5 |
| 当前节点覆盖率已高 | 转向未探候选 | 目的地已知 → IG × 0.25 |

全部由图统计涌现，没有任何按应用/按任务的规则。

---

## 6. InformationNeed 如何改变排序

`parse_reasoning_prior(上一步模型输出, 目标)` 产出 `P_{i-1} = {K_target, A_expect, K_risk, U_miss}`，**零额外模型调用**。四个匹配度分别保留（`target_match` / `expected_affordance_match` / `unresolved_information_match` / `risk_conflict`），不提前塌缩成单一相似度，因为要区分的是"UI 新奇度"和"任务相关的预测信息"。它进入 `path_probability` 的先验项，权重 0.45；`risk_conflict > 0` 直接把 `risk_level` 抬到 HIGH，由硬门剔除。

---

## 7. 可恢复性如何改变排序

**它不改变排序——它决定候选是否存在。** `SafetyGate` 在打分之前剔除。理由是排序是相对的：当所有候选都很差时，最不差的那个仍会被探测，这对"有用性"是对的，对安全是错的；一次无法撤销的探测代价不是选了个次优，而是整个 episode 走不下去。

`estimated_recoverability` 的取值顺序：本边的回滚成功率（Beta-Binomial 收缩）→ 同 `(probe_type, role, need_type)` 桶的上下文成功率 → 按探测类型的实测先验。若图里记着该转移有 `NOOP`/`INVERSE` 级逆动作，直接抬到 0.9——那是"无需走恢复阶梯即可撤销"的直接证据。

可恢复性还进入代价项：回滚失败按 12 秒定价，让代价本身（而不只是硬门）偏好可靠可逆的候选。

---

## 8. GraphFact schema

```python
GraphFact:
    fact_type: VERIFIED | OBSERVED | DONE | NO_RELEVANT_EVIDENCE
    action_label:      str      # 人能在截图里找到的控件名
    source_edge_id:    str
    evidence_labels:   tuple    # 目的地上观察到的标签，至多 4 个
    edge_status:       str
    certainty:         Verified | Observed | Tentative
    certainty_weight:  float    # 内部用，不进 prompt
    need_match:        float
    historical_utility:float
    freshness:         float    # exp(-代龄 / 6)
    already_taken:     bool
    risk_level:        str
    utility_score:     float
```

`REUSABLE`→Verified(1.0)，`VERIFIED`→Verified(0.95)，`INFERENCE_ALIGNED`→Observed(0.75)，`SPECULATIVE`→Tentative(0.45)。**不向模型暴露 `confidence=0.63` 这类原始数字**——那会诱使它对一个它看不到标定的数值做算术。

---

## 9. GraphDistiller 算法

确定性五段，**不调用任何模型**：

1. **retrieve** — 只取当前节点的出边（1 跳）。再远的东西模型这一步也用不上。
2. **filter** — 丢弃 `INVALID`/`STALE`；丢弃只能用坐标称呼的控件；未真正探测过（`probe_count = 0`）的边不得作否定断言；`open_app` 边不得作否定断言。
3. **score** — `U = (0.25 + need_match) × certainty × (0.5 + 0.5 × historical) × freshness`；`DONE` 固定 `0.4 × freshness`（它的价值在于阻止重复，不该被 need 匹配乘没），`NO_RELEVANT_EVIDENCE` 固定 `0.3 × freshness`。低于 0.15 丢弃。
4. **compress** — 按 U 贪心取前 K，`max_graph_facts=3`、`max_graph_context_tokens=64`；否定事实至多 1 条。
5. **template** — 固定 `[Memory] / Need / Verified / Observed / Done` 结构。

**若选出的事实里没有一条正面事实（VERIFIED/OBSERVED），则完全不注入。** 依据：2026-08-31 的 12 任务实测中，20 次注入没有一条带正面事实，全是 `Need:` 复述模型自己的话加一句"没观察到证据"，该臂丢了 4 个任务。

`Need:` 行去重并按名词短语形状过滤——`parse_reasoning_prior` 故意过量产出，照抄会得到 `Need: Zucchini Noodles with Pesto, Zucchini Noodles with Pesto, Zucchini, Noodles, Pesto.`；同时滤掉从 `<THINK>` 里挖出的半句话（`s detail page. I will click on the`、`Therefore`）。

---

## 10. 真实任务上的紧凑上下文示例

`RecipeDeleteMultipleRecipes`，修复前后对比。

修复前（实际注入过的，44 token，零信息）：

```
[Memory]
Need: Garlic Butter Shrimp, Garlic Butter Shrimp, Lentil Soup, Zucchini Noodles
with Pesto, Garlic, Butter, Shrimp, Lentil, Soup, Zucchini, Noodles, Pesto,
first open its details page, select the 'garlic butter shrimp' recipe to access
its detai.
Observed: no relevant evidence observed under Broccoli.
```

修复后的目标形态（≤64 token，需有正面事实才注入）：

```
[Memory]
Need: Garlic Butter Shrimp.
Verified: Details -> {Phone, Address}.
Done: Filter already used.
```

---

## 11. 三种模式的确切触发条件

**SKIP_INFERENCE** —— 全部满足：

- 当前 UI 匹配图中节点，且存在可复用边
- 该边 `status == REUSABLE`（前缀对齐晋升的前瞻边），**或**该节点满足全部三条：决策熵为 0、访问过 ≥2 次、且**模型在它上面只选过一个动作且执行过 ≥2 次**
- 目的地非空、非自环、`risk_level ∈ {SAFE, LOW}`
- **不在最近 4 个节点内**——但对模型自己执行过 ≥2 次的转移豁免（那是任务在重复，不是图在绕圈）
- 节点出度 > 1 时，决策熵 ≤ 0.4
- 连续跳过 < 3
- 本 episode 跳过命中率 ≥ 50%（前 2 次免检）
- 多跳仅在本 episode 已有 ≥2 次跳过且全部命中时允许

执行后立即比对落点，**校验对象是 `observed_destinations` 集合**：只观察到一个目的地时严格校验；多个时要求落到集合内、或至少离开了原节点（重复动作本就会落到不同屏幕）。命中则记 `record_skip_result(True)` 并可继续；未命中则作废该边及子树、停止跳过、交还模型。

**"重访屏幕上的权威边"这一来源已删除。** 按来源拆开统计 28 次图跳过（2026-08-31）：bootstrap 80/80，这一类 3/17（18%），且每个步数变差的离群任务都由它造成。屏幕看起来一样，不等于处境一样。

**GRAPH_ENHANCED_INFERENCE** —— 不满足跳过条件，但蒸馏器产出了含正面事实的非空上下文。

**NORMAL_INFERENCE** —— 其余情况。**不注入任何图信息**，避免用弱相关内容污染 prompt。

---

## 12. 执行反馈如何回写图

**边的身份是 `(源节点, 规范化动作)`，目的地不参与。** 原先包含目的地，把一次重复的决策拆成多条各自只执行过一次的边（`RecipeDeleteMultipleRecipes`：同一个控件点了三次，每次删完剩下的列表不同、落点节点不同，于是三条边各 `hits=1`）。所有"模型在这里做过这件事吗"的统计因此永远读出 1，重复型任务——渐进式记忆最该服务的那一类——恰恰是机制永远无法触发的。合并后目的地进入 `observed_destinations`：只有一个说明转移确定、可严格校验；多个说明动作仍然对，但落点取决于图没建模的状态。

| 事件 | 方法 | 写入 |
|---|---|---|
| 一次投机探测 | `record_probe` | `probe_count` `rollback_success/failure_count` `cumulative_exploration_cost` `cumulative_realized_ig` `last_updated_generation` |
| 前缀对齐（探测猜测与模型落点一致） | `record_inference_alignment` + `promote_children_of_aligned_prefix` | `inference_alignment_count`，子边升 `REUSABLE` |
| 真实执行后落点比对 | `record_execution_verification` | `execution_hit_count` / `execution_miss_count`，状态升降 |
| 跳过后落点比对 | `record_skip_result` | `skip_attempt_count` / `skip_success_count` |
| 回滚结果 | `record_rollback_result` | `rollback_*_count`、`inverse_level` |

`record_skip_result` 与 `record_execution_verification` 刻意分开：前者问"信图而不信模型这次划算吗"，后者问"这条转移的动力学还成立吗"。一条边可以如实描述 UI，同时在任务的这个位置是错误的选择。

单值字段记录最近一次观察，累计量记录历史——"回滚成功过一次、失败过两次"与"回滚成功过一次"是完全不同的处境，而两者的 `rollback_success` 都是 `True`。

---

## 13. 新增日志字段

`scored_candidates.jsonl`（**每个候选一行，不只是选中的那个**）：矩阵全部字段 + `task_id` `step` `node_id` `rank` `selected_for_probe` `final_exploration_score` `path_probability` `expected_information_gain` `predictive_value` `expected_cost` `recoverability` `ui_novelty_score`。

之所以记录全部候选而非仅选中项：PART F 的核心指标是模型的真实动作后来排在第几位，而这从"选了谁"的日志里恢复不出来——**把正确元素排在第二位的选择器，和从未把它纳入考虑的选择器，是完全不同的两回事**。

`filtered_elements.jsonl`：安全门剔除的候选 + `reason` + `estimated_recoverability` + `risk_level`。

`serial_events.jsonl`：
- `gate`：`mode` `reason` `had_reusable` `context_tokens`
- `graph_context`：`graph_mode` `injected` `chars` `tokens` `text` `raw_fact_count` `candidate_fact_count` `selected_fact_count` `selected_fact_types` `selected_edge_ids`
- `skip`：`edge_id` `matched` `hop` `episode_accuracy` `kind_detail`
- `explore`：`probes_completed` `restore_status` `max_depth_reached` `unexpected_error`
- `repair`：`ok` `via` `stranded` `landed`
- `prefix_aligned`：`edge_id` `promoted`

---

## 14. 探索侧消融命令

```bash
# 排序策略
--exploration_policy information_need      # 原 InformationNeedRanker
--exploration_policy graph_matrix          # 矩阵 + 预测式打分器（默认）

# 特征组
--disable_exact_history
--disable_contextual_history
--disable_information_need
--disable_cost
--disable_recovery_history

# 探索本身
--probes_per_step 0                        # 关闭探索
--allow_icon_only_probes                   # 放开无障碍标签缺失的图标控件
```

---

## 15. 推理侧消融命令

```bash
--graph_reasoning off                # 不用图
--graph_reasoning briefing           # 旧的 screen_briefing
--graph_reasoning distill            # 蒸馏器紧凑上下文
--graph_reasoning skip_only          # 只跳过，不注入
--graph_reasoning distill_and_skip   # 两者都开
```

指标由 `scripts/analyze_exploration_metrics.py <run_dir>` 计算，输出 F1（`rank_of_future_real_action`、`top1_future_action_hit_rate`、`top3_future_action_coverage`、`graph_history_usage_rate`）与 F2（`inference_skip_rate`、`skip_hit_rate`、`context_injected`、`mean_context_tokens`、`repeated_action_rate`、`unrestored_rounds`）。

---

## 16. 仍然是启发式的部分

| 部分 | 现状 | 可替换为 |
|---|---|---|
| `path_probability` 的先验形式 | 线性 + Beta-Binomial 收缩 | `SmallModelPredictiveEstimator`（接口已留） |
| `expected_IG` 的相关性项 | token 重叠 | 学习到的相关性 |
| 事实效用 `U(fact)` | 四项相乘 | 学习到的效用 |
| 代价模型 | 按探测类型的实测均值 | 在线回归 |
| 蒸馏预算 3 事实 / 64 token | 固定 | 按剩余步数自适应 |
| `_role_of_class` 的角色划分 | 按框架控件类名 | 不变（框架级事实） |
| `_is_entity_like` 的连接词表 | 英语语法固定表 | 不变（语言级事实） |

`ScoringConfig` / `DistillerConfig` / `GateConfig` 三个 dataclass 集中了全部阈值与权重。

---

## 17. 现在已具备的、可训练未来预测器的数据

`scored_candidates.jsonl` 每行是一个 `(状态, 候选)` 对的完整特征向量（约 60 维），可与以下标签联接：

- **`rank_of_future_real_action`** —— 由 `serial_events.jsonl` 的 `inference.action` 坐标与候选的归一化中心联接得到；这是训练"预测模型下一步会点哪个"的直接监督信号
- `rollback_success` / `recovery_level` —— 来自 `probe_trace.jsonl`
- `realized_information_gain` —— 探测后的新元素集合
- `inference_alignment` / `execution_hit` —— 后续步骤回填
- `skip_success` —— `skip` 事件的 `matched`

也就是说，一次全量跑测就能产出一个带监督标签的候选级数据集，直接支持把 `PredictiveElementScorer` 换成学习模型。

---

## 关键研究问题的当前答案（全量 116，2026-09-01）

**Q1：累积的探索历史能让后续探索的元素选择更有预测性、更安全吗？**

安全——能。可恢复性硬门、在线拉黑、"一旦有一轮未复原就停止探测"三条合起来，把探索的代价压到了可忽略：全量 116 里探测 63 次、10 次未复原，只在 1 个任务上造成损失。

预测性——**不能，瓶颈已定位。** `rank_of_future_real_action` 显示模型下一步真正点击的控件只有 **16%** 出现在候选集里，出现时平均排第 **9.2** 位，`graph_history_usage_rate = 0`。瓶颈在候选集不在排序器：安全过滤剔除的 673 个候选中 55% 是 `icon_only_no_accessible_label`，正是模型最常点的一类。这直接解释了 116 个任务里前缀对齐只发生 1 次。

**Q2：置信度不足以跳过时，探索信息还能有用吗？**

机制已实现且不再有害（"没有正面事实就不注入"之后，注入从每 12 任务 20 次降到 2 次，那 20 次经核实全是噪声）。但**收益仍不可测**，因为正面事实需要探测边带回目的地标签且模型回到同一节点，而 Q1 的瓶颈使这种情况极少发生。

**Q3：确定性、有 token 预算的蒸馏器能在不增加模型调用的前提下改善推理吗？**

蒸馏器代价可忽略（纯字符串处理），格式约束经实测确立。**改善效果尚未可测**，原因同 Q2。

**Q4：同一张渐进式图能同时支撑"去哪探索"和"是否/如何推理"吗？**

结构上能，且已经在跑：两条路径读同一份边统计、同一个世代守卫、同一个 `ContextualHistoryTable`，`record_*` 是唯一写入口。v16 更进一步证明了这个统一是必要的——修好边身份（一个纯粹的图表示问题）之后，跳过命中率从 18% 跳到 86%，这是"图的质量直接决定推理侧收益"的直接证据。

---

## 与基线的对比（全量 AndroidWorld 116）

**必须用同环境对照臂。** 这套 agent 的单次跑测方差本身就很大：`4B_2`（54/116）与同期另一个纯 baseline `base116_cap15`（57/116）**交集只有 45**，各自独有 12 和 9——两个都不含本设计机制的跑测彼此翻转了 21 个任务。

| | 完成 | 成功 | 成功率 | 成功任务平均步数 |
|---|---|---|---|---|
| **本设计 v16** | 114 | **49** | 43.0% | **6.86** |
| 同环境对照（机制全关） | 115 | 55 | 47.8% | 7.82 |
| `base116_cap15` | 114 | 57 | 50.0% | — |
| `4B_2` | 116 | 54 | 46.6% | 7.85 |

两臂都成功的 45 个任务：**6.38 vs 7.31 步**（−0.93），**逐任务 30 快 / 2 慢 / 13 平**。

机制表现：bootstrap **96/96**，图跳过 **42/49（86%）**，探测 63 次 / 10 次未复原。

**结论的边界**：

- 步数成立（116 任务，30 快对 2 慢）。
- 机制本身成立（图跳过从 v12 的 3/17 做到 42/49）。
- **成功率无法断言**：−6 落在两个纯 baseline 之间 ±10 的波动内。逐任务归属核对显示 10 个掉分里 **8 个是零机制活动**（设计根本没运行），真正归属机制的 2 个，而另有 4 个任务只有本设计拿下。分层看差距在"跳过触发的 14 个任务"（−2）与"未触发的 100 个"（−4）上均匀分布，不集中在机制作用处。

要得到可发表的成功率结论，需两臂各重复 3 次取平均。

---

## v16 让机制成立的三处修复

1. **边身份去掉目的地**（根因）。`hash(源, 动作, 目的地)` 把"模型三次选同一控件"拆成三条 `hits=1` 的边，因为每次操作后剩余列表不同、落点节点不同。改成 `hash(源, 动作)`、目的地存为 `observed_destinations` 后，重复型任务的节点立刻呈现 `hits=3、出度 1、熵 0`。
2. **落点验证按观测集合**。重复动作本就落到不同屏幕，按单一目的地校验会把机制最好的情形判成失配。
3. **环路保护对模型自己重复过的边豁免**。两道守卫（`reusable_edge` 与 `ReasoningGate`）都把"删完回列表再删下一个"当成绕圈；绕圈保护是为投机边设的（`CameraTakeVideo` 那 13 次），模型亲自执行过 ≥2 次的边是任务在重复。
