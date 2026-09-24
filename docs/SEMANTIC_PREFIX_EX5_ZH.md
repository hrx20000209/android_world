# Semantic Prefix Memory + Ex5 Conservative Route Assist

这是一个接在现有 `run_sensys30_online_task.py` 上的 sidecar 原型，不是第二个
AndroidWorld runner。Ex5 的 reasoning、action parser、evaluator、恢复梯子和安全
过滤仍是主路径。

## 条件

| 条件 | 配置 |
|---|---|
| A | Ex5 baseline；不启用长期 executable graph 或 Prefix |
| B | 之前的 exploration + executable graph：graph prompt、post-fusion、K-step、hard negatives、action groups |
| C | B + Prefix logging，不让 Prefix 影响决策 |
| D | B + Prefix top-K prompt context |
| E | 完整 MobileExplorer：B + Prefix prompt/route assist + verified graph skip |
| F | B + coverage-only exploration ranker，作为负对照 |

B–F 都使用 `mobileexplorer_executable`，并在 arm 内共享版本化
`executable_memory.json`；C–F 额外共享 `semantic_prefix_memory.json`。因此 Prefix
是完整 exploration/graph 设计之上的压缩执行 sidecar，不是替代 graph 的简化 agent。

Prefix 文件是版本化 JSON：`schema_version=1`。它保存 state、edge、落点分布、
可逆性、trap/dynamic/ambiguity 标记、route hit/miss 和解析错误计数。Selector
每次执行前按 resource-id、content-desc、text 和 class 在实时 UI 中唯一重定位；
绝对坐标不作为身份。

D 条件只有同时满足 lexical match、唯一 selector、至少三次一致落点或历史 route
hit、confidence >= 0.82、路径不超过 3 跳、可逆且非 dynamic/trap 时才执行。执行
后用新的 state signature 验证，失败会记录 `prefix_route_miss` 并返回 Ex5。

所有 Prefix 条件把 5 设为页面 probe target；候选为空、没有安全候选或恢复失败时
可以提前停止，但 `inference_windows.jsonl` 会记录 `stop_reason` 和完成情况。

## 运行

先建立到空闲 LabServer vLLM 的 tunnel（本次使用远端 8084）：

```bash
ssh -N -T -L 18084:127.0.0.1:8084 LabServer
```

短 smoke：

```bash
python scripts/run_semantic_prefix_ex5_experiment.py \
  --output_root evaluation_results/semantic_prefix_ex5_<timestamp>/arms_smoke \
  --limit 1 --max_steps 2 \
  --api_url http://127.0.0.1:18084/v1/chat/completions \
  --metrics_url http://127.0.0.1:18084/metrics \
  --model GELAB-ZERO-4B
```

完整固定 40-task 矩阵使用同一任务顺序、seed、模型、probe 和 step 参数；省略
`--limit`，并保留默认的五个 arm：

```bash
python scripts/run_semantic_prefix_ex5_experiment.py \
  --output_root evaluation_results/semantic_prefix_ex5_<timestamp>/full_40 \
  --api_url http://127.0.0.1:18084/v1/chat/completions \
  --metrics_url http://127.0.0.1:18084/metrics \
  --model GELAB-ZERO-4B
```

每个 task 目录包含原 runner 的 JSONL 日志；每个 arm 另外包含
`semantic_prefix_memory.json`、`semantic_prefix_events.jsonl`、
`semantic_prefix_summary.json` 和 `summary.json`。根目录的 `summary_zh.md` 是
汇总入口。脚本串行运行 task，因为它们共享 Android 设备。

## 已验证结果

`evaluation_results/full_mobileexplorer_prefix_20260922/smoke_2/` 和 `smoke_3/`
已完成完整组合条件的 1-task、2-step smoke；所有 runner return code 为 0。
graph arms 均完成 2 个真实 reasoning step并产生 executable graph transition/fusion
事件，Prefix arms 同时产生 Prefix nodes/edges。由于人为设置了 `max_steps=2`，
这些数据只用于验证编排、日志和异常隔离，不能当作成功率结论。

`smoke_notes_logging_final/` 还验证了 LabServer 返回 `AWAKE requires value` 时：
记录了 2 个 `parse_error`，使用安全 WAIT fallback，任务继续执行；Prefix sidecar
正常持久化，probe stop reason 为 `no_safe_candidate`，没有把解析异常升级为任务
失败。离线单元 smoke 验证了 3 次一致落点、selector 重定位、trap/dynamic 拒绝、
route hit/miss 和 schema reload。

完整 40-task 结果只有在上面的 `full_40` 命令实际完成后才可报告；不得用 smoke
结果替代它。
