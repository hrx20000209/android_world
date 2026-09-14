# 消融开关校验（Part 9）

校验方式：**不额外跑任务**，直接从已完成运行的 `serial_events.jsonl` 验证
每个开关确实改变了目标行为。比"跑 3-5 个任务肉眼看 trace"覆盖面大得多
（每臂 113-116 个任务的全部事件），且不消耗设备时间。

## 结果

| 臂 | 预填跳 | 来源分布 | 回退 | 被通过率门拒 | 被目标投票拒 | 判定 |
|---|---|---|---|---|---|---|
| Baseline 不建图 | **0** | — | **0** | 0 | 0 | ✅ 无任何图机制 |
| MobileExplorer 完整 | 79 | goal 55 / unique 24 | 1 | —¹ | —¹ | ✅ 两条检索路都工作 |
| w/o 目标条件检索 | 34 | **unique 34（100%）** | 0 | 0 | **985** | ✅ 目标投票关闭，纯唯一性门 |
| w/o 通过率门 | **100** | goal 78 / unique 22 | 1 | **0**² | 0 | ✅ 门关后开火 +27% |
| w/o 落点回退 | 18³ | goal 11 / unique 7 | **0** | 12 | 0 | ✅ 回退归零，其余门照常 |

¹ 完整臂跑在门级别拒绝日志加入之前（运行 09-13 19:35，仪表 09-14 14:27 加入），
  其 `prefill_refused` 事件没有 `gate` 字段。不是缺陷，是时间顺序。
² `--prefill_pass_rate 0` 时该门不可能拒绝，0 是正确值而非未触发。
³ 该臂当时仅跑到 77/116。

## 逐条对照 Part 9 的检查项

| 检查项 | 结论 |
|---|---|
| 1. Base 确实不调用额外探索 | ✅ 预填 0 跳、回退 0、无 gate 事件 |
| 2. 历史-only 不产生新鲜探索 | ✅ `--prefill_sources store` 下 `episode` 来源计数为 0 |
| 3. 在线-only 拿不到历史 | 待跑（`--prefill_sources episode` + 空 store） |
| 4. 完整同时用两者 | 当前主臂默认 **只用 store**（episode 线上仅 30% 命中，已关） |
| 5. 记录的推理/动作计数与 trace 一致 | ✅ `inference` 事件数 = 日志里的 step 数（见 summary.csv 的 `avg_reasoning_calls` 对 `avg_steps`，差值恰为 `app_bootstrap` 的显式跳过数） |
| 6. 图前/图后统计正确 | 每任务落盘 `progressive_belief_graph.json`；图**后**已记录。图**前**需新增仪表（当前 store 是跨任务累积的，任务开始时的快照没有单独保存） |
| 7. 新鲜/历史证据未混淆 | ✅ 每个预填事件带 `source` 字段（`store` 侧再分 `unique`/`goal`，`episode` 为本轮新鲜） |
| 8. 条件间应用重置正确 | ✅ 每臂的 `run_arm.sh` 逐任务重新 `adb forward` 并重设 SMS role；AndroidWorld 自身负责任务级重置 |

## 未通过 / 待补

- 检查项 3、4 需要 `--prefill_sources episode` 臂，尚未运行。
- 检查项 6 的"图前节点数"需新增仪表：任务开始时 dump 一次 store 规模。
  当前只能报"图后"与"本任务新建节点数"。
