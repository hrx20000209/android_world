# MobileExplorer 两系统设计：实验结论与修复记录（2026-08-29）

本文记录一夜实验的完整结论。所有数字来自真实 AndroidWorld 跑测，原始数据在 `results/` 下。

## 一、方法论修正：此前的 baseline 是错的

此前所有对比都以 `results/4B_2.txt`（2026-03）为 baseline。该表由不同 agent 配置产生，**与当前 `gelab_agent` 已不可比**。

本次建立了真正的对照：同 agent、同 seed(30)、同 a11y(fast_provider)、完全关闭探索，直接跑 `run.py`。

| 任务集 | 旧表 4B_2 | 实测纯 baseline |
|---|---|---|
| 5 任务 | 5/5 | **4/5** |
| 15 任务 | 7/15 | **4/15** |

具体反例：`ExpenseDeleteMultiple2`、`RecipeDeleteMultipleRecipesWithNoise` 旧表标记成功，实测纯 baseline 同样失败。

**结论：此前所有"two-system 落后 baseline"的结论都被这个错误基准夸大了。** 后续实验一律以实测纯 baseline 为准。

复现：`/private/tmp/.../scratchpad/pure_baseline_15.sh`，结果在 `results/pure_gelab_baseline_{5,15}tasks/`。

## 二、核心结论：投机探测是唯一的净损失来源

5 任务集，5 种探索强度的消融：

| 配置 | 成功率 | 回滚失败 | 推理调用 |
|---|---|---|---|
| 实测纯 baseline | 4/5 | — | — |
| **零探测 + 证据注入** | **4/5** | **0** | 44 |
| **零探测** | **4/5** | **0** | 43 |
| 轻量探测（3 探测/1 层） | 3/5 | 9 | 55 |
| 重度探测 + 深层恢复补全 | 2/5 | 8 | 50 |
| 重度探测 + 黑名单修复 | 2/5 | 11 | 56 |
| 重度探测 + 历史污染修复 | 3/5 | 7 | 49 |

最强的因果证据：`RecipeDeleteMultipleRecipes` 与 `ContactsAddContact` 在**所有开启探测的配置下无一例外失败**，关闭探测后**双双成功且步数与 baseline 完全一致**（14 步 / 9 步）。

代价收益：探测节省约 4% 推理调用（50 次省 2 次），代价是 20–40% 成功率。**投入产出严重倒挂，且对探索强度不敏感**（20 探测降到 3 探测，成功率无实质改善）。

失败元素分析显示：翻车的几乎都是"第一次遇到的新元素"（`Share` / `fab_recipes` / `floating_action_button` / `nav_manage` 各失败一次），**事后学习的黑名单在原理上防不住第一次**。

15 任务集最终：实测纯 baseline **4/15** vs 零探测 two-system **5/15**（唯一差异 `ClockStopWatchRunning`：baseline 5 步失败、two-system 4 步成功）。注意 5/15 vs 4/15 仅差一个任务，而同配置重跑存在翻转，**该领先尚不具统计显著性**。

## 三、成立的部分：渐进式图 + 跳过推理

零探测配置并非退化为 baseline，它仍然提供：

- 15 任务中 **14 次跳过推理**，推理调用数为所有配置最低
- **0 次回滚失败**
- 从真实执行积累的 state-action 图（本次新增能力，见下）

**用户设计的核心（渐进式 state-action 图 + 跳过推理）是成立的；不成立的是"靠猜逆动作恢复设备"这个前提。**

排除的替代方案：模拟器快照。实测 `adb emu avd snapshot save` 需 **50 秒**，而推理窗口仅 7–8 秒；`load` 需 3.6–5.7 秒但只能恢复到旧快照，非当前提交状态。

## 四、本次修复的 9 个真实缺陷

每处修复在代码中都有注释说明证据来源。

1. **节点身份退化**（`state.py` / `belief_graph.py`）
   节点 ID = `activity + 结构签名 + pHash`，而结构签名含全部元素文字、pHash 为像素级。秒表读数变化、列表少一行、滚动几像素都产生全新节点 —— `ExpenseAddMultiple` 中一个 activity 被拆成 **40 个节点**，图退化为不重复的链，渐进式记忆无法积累。
   修复：新增 `layout_signature`（仅可交互元素的 class + resource_id + 粗粒度位置，不含文字与像素哈希）作为节点身份；原严格签名保留用于回滚验证。**将"节点身份（宽松）"与"状态验证（严格）"两个概念分离**。已加单测 `test_layout_signature_ignores_volatile_text_but_keeps_skeleton`。

2. **历史污染**（`run_one_parallel_exploration_task.py`）
   内部工程日志（`"Skipped model inference using verified lookahead edge c291..."`）被当作动作历史喂给模型。`ContactsAddContact` 因此点击一次后凭空宣布"任务已完成"（baseline 需 9 步）。
   修复：模型可见历史统一为普通动作格式，工程细节移入 `skip_detail` 字段。修复后该任务从 4 步→7 步不再过早终止，`ExpenseDeleteMultiple2` 一度从失败翻为成功。

3. **黑名单深度盲区**：只记录 `depth==1` 且 `notes=="RESTORE_FAILED"` 的失败，深层失败标记为 `NESTED_RESTORE_FAILED`/`depth==2`，永远进不了黑名单 —— 同一个不可恢复的菜谱卡片被重复踩 3 次（8 次失败中 4 次为此元素）。

4. **深层恢复阶梯残缺**（`live_probe.py`）：深层探测调用 `_recover` 时传 `committed_actions=None`，永远够不到最可靠的 `TRAJECTORY_REPLAY`，而 **52%（12/23）的回滚失败正是深层探测**。修复后深层失败 12→6。

5. **预算地板 `max(1, ...)`**（`resources.py`）：让"完全关闭探索"在架构上无法表达（配 0 会被强行改为 1），违反 spec §20"每个组件可独立关闭做消融"。**此前所有号称"关闭探索"的对比其实都未真正关闭。**

6. **循环检测覆盖面**：原本仅在"两次重复动作之间夹着跳过"时触发，但 15 任务批次中多数循环（`RecipeDeleteMultipleRecipes`、`MarkorEditNote` 等）中间**没有任何跳过**，是模型自身卡死。已放宽为任意重复即触发。

7. **置信度日志时机**：`skip_events.jsonl` 记录的是失败后被清零的置信度，掩盖了决策时刻的真实值。

8. **晋升门槛过松**（`belief_graph.py`）：一次投机验证 + ranker 的先验打分即可跨过复用门槛（`0.82 + 0.16 * path_probability`）。`ExpenseDeleteMultiple` 中一条 `inference_alignment_count=0, execution_hit_count=0` 的边被直接信任并跳错。已将首次证据置信度上限压到 0.85（低于常用阈值 0.86–0.90），必须经真实执行验证才能跨过。

9. **探索子进程静默崩溃**：单次 adb 超时会让子进程直接崩溃且不上报状态，主进程干等 170 秒后拖垮整个任务。已加兜底上报。

## 五、新增能力：权威转移入图（零风险记忆）

此前图**只从 bootstrap 边和投机探测边**积累，agent 每一步真实执行的 `状态A --动作--> 状态B` 这一**零风险、100% 可靠**的转移被完全丢弃。

现已记录为 `VERIFIED` 边（真实执行的证据强于任何投机探测），并附带目标页面的显著标签，使证据注入在零探测模式下也能工作。这正是"UI state-action 图"这一设计的最低风险实现形式。

15 任务跑测中该机制稳定工作，`authoritative_edges.jsonl` 可查。

## 六、其他实测数据

- **跳过推理仍计入 step**（`episode_runner.py:83`，每次 `agent.step()` 即一步）。这是合理的：跳过的那步仍真实执行了 GUI 动作，节省的是 LLM 调用而非动作。**正确指标是推理调用次数，而非步数。**
- 证据注入的 prompt 开销可忽略（推理调用 44 vs 43），但本次未观察到成功率增益（4/5，与不注入持平），样本过小，需更大任务集验证。

## 七、遗留问题与建议方向

1. **图跨 episode / 跨任务复用**：同一 App 的界面结构稳定，这才是记忆真正摊薄成本的场景。当前图仅 task-local。
2. **仅对结构上可证明可逆的动作做投机探测**（滚动、开关类），放弃对未知导航的投机。
3. **证据注入需在更大任务集上验证**。
4. **5 任务集方差较大**：同配置重跑存在成功↔失败翻转，任何小于 2 个任务的差异都应视为噪声。
5. 已知风险（按用户要求保留未处理）：设备变脏后每步重新启用探索，可能让单次失误级联 —— 2026-08-28 曾观察到探索经由支出编辑页进入 Android 分享面板并暴露真实登录账号。

## 八、复现命令

```bash
# 实测纯 baseline（无任何探索机制）
ANDROID_WORLD_A11Y_METHOD=fast_provider python run.py \
  --suite_family=android_world --agent_name=gelab_agent \
  --tasks=<TASK> --n_task_combinations=1 --fixed_task_seed \
  --task_random_seed=30 --max_n_steps=<N> --console_port=5554 \
  --output_path=results/<OUT>

# 零探测 two-system（推荐配置）
python scripts/run_two_system_baseline_gate.py \
  --manifest configs/two_system_repetitive_15tasks.json \
  --output results/<OUT> --agent_name gelab_agent --resume \
  --max_probes 0 --max_depth 1

# 加证据注入
#   追加 --inject_evidence

# 开启投机探测（当前证据表明净负面）
#   --max_probes 20 --max_depth 3
```

单测：`python -m pytest -q android_world/parallel_exploration/`（35 passed）
