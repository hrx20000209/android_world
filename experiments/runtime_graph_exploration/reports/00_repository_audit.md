# 00 仓库与基础设施审计（2026-10-14）

## 1. 基础设施（实测）
| 项 | 状态 |
|---|---|
| vLLM | LabServer 8095 / 8096 / 8097 都在服务 **GELAB-ZERO-4B**（max_model_len 65536）。本机只隧道了 8095（127.0.0.1:18095，直连 SSH 保活脚本 `runtime_graph_study/scripts/tunnel_keepalive.sh`） |
| 采样 | `LlamaCppWrapper(temperature=0.0)`，GELABAgent 原生 prompt 与解析（`rgs_run.decide`） |
| 设备 | AndroidWorld 模拟器 `emulator-5554`，只有 1 个。另有一台 adb 设备 `10.79.170.66:33219`，用途不明，不使用。**GUI episode 只能串行**；多个 vLLM 端口只能用于离线推理类实验 |
| 磁盘 | 剩余约 11 GiB。每 232 个 episode 约 0.4 GB，完整评测所需空间足够，但不能再保存全分辨率截图的冗余副本 |

## 2. 可复用代码
- `runtime_graph_study/scripts/rgs_ui.py`：a11y 解析、三级状态签名（exact / struct / layout）、按 z-order 取最顶层可点击节点的 `element_at`、`enumerate_slots` 安全分类（deny-list、对话框按钮、输入框、开关、弹出列表）、`match_element`。
- `runtime_graph_study/scripts/rgs_run.py`：Device（截图 + a11y、settle）、Explorer（探测、回滚梯子：none / retap / reselect / BACK / graph restore，struct + content 校验）、Graph（不可变转移日志、不可回滚记忆）、Episode 主循环。
- `trace_reachability_validation/scripts/{seq_run,full_run,proc_replay}.py`：续写计数 Store、验证跳步、episode 内实体流程复用、模型提交后的确认、统一历史写入（`agent._actions/_summaries`）。
- 历史轨迹：`trace_reachability_validation/raw/full/{BASE,FULL,FULL2,FULL3,FULL4,FULL5,FULLNE}`，共 1262 个 episode，覆盖 116 个模板，seed 30/31。每一步都有 a11y（gz）与截图，可离线重建。

## 3. 已知风险的现状
| 风险 | 现状 |
|---|---|
| 动作绑到大的父节点 | 已修：`element_at` 取最顶层可点击节点（FAB 压在列表行上的情况已覆盖） |
| 探索资格过严 | 已放宽到 deny-list + 回滚可验证；剩余限制是安全必需 |
| 探索器多选 Scroll | 当前 Explorer 不做 scroll（本轮也禁止滚动） |
| 回滚失败 | FULL v1 有 123 次，去掉覆盖式探索后 FULL3–5 为 15–22 次 / 232 个 episode |
| 探索证据与后续决策对不上 | E1 预测下一步约 44% 被用上，但只节省等待时间 |
| 图证据进 prompt | 6 种形式 + 安慰剂：没有净收益（Pilot B） |

## 4. 已有结论（本轮设计必须面对）
- 跨任务计数作为"相关性"有幸存者偏差：FULL3 的跨任务导航跳步造成 7 vs 2 个任务的损失。
- 已执行函数遇到的真正运行时缺口（G1 / G2）在 464 个 episode 中为 0。
- 增益只来自模型授权的执行：episode 内流程复用、模型提交后的确认（FULL5：触发的 9 个 episode 调用 −19%）。

## 5. 本轮协议与上一轮的差别
- 初始记忆按**留一模板**构建：只用其他模板的历史轨迹（来自上面 1262 个 episode），等价模板（名字去掉末尾数字后相同的变体）一并排除。每个 episode 从不可变的快照开始，episode 之间不相互泄漏。
- 测试用新 seed（32），与历史实例不同。
