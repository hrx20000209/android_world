# e2e116nA 替换式消融最终报告（Orin，116 题配对）

- **设置**：Orin vLLM（GELab-Zero-4B-preview，n-gram8 投机解码 + CUDA graph），模拟器内后台运行 Qwen2-VL-2B 负载，与 e2e116n 的 FULL45 setting 完全相同。
- **数据来源**：BASE 和 FULL45 沿用 e2e116n 的结果（本轮没有重跑）。三个消融组来自 e2e116nA，均把 FULL45 的一个组件替换成朴素版本：
  - **w/ random search（RAND）**：探索候选改为随机顺序，去掉 E1/RV 目标，其余与 FULL45 相同。
  - **w/ prompt injection（INJ）**：关闭全部图执行（bootstrap、confirm、function/form/route、reuse、macro、R1 跳转），只把当前页面在图中的出边和目的页文本写进 prompt。
  - **w/o pipelined execution（NOSYS）**：关闭流式早动作、自适应等待、像素检查和 R2 复用，改为等待完整解码后执行，动作后固定等待 1 s。
- **错误处理**：两条因 Orin 掉线中断的 episode 已补跑（均为 SimpleCalendarAnyEventsOnDate，INJ 和 NOSYS 各一条），最终 116/116 题完整配对，其中五组都成功的有 28 题。
- **脚本与输出**：`scripts/ablation_perf.py` → `figures/understand/ablation_summary.json`；`scripts/fig_ablation.py` → `g_ablation.*`、`h_ablation_breakdown.*`。

## 1. 三个口径

| 组 | 成功 | 全部 116 题：调用 / e2e | 成功题：调用 / e2e | 五组共同成功 28 题：调用 / e2e |
|---|---|---|---|---|
| BASE | 53 | 10.14 / 151.9 s | 8.06 / 130.2 s | 6.86 / 115.1 s |
| **FULL45** | **54** | 9.16 / 135.2 s | 6.69 / 107.7 s | **5.11 / 88.0 s** |
| w/ random search | 50 | 8.89 / 129.2 s | 6.48 / 105.7 s | 5.21 / 90.1 s |
| w/ prompt injection | 50 | 11.60 / 147.9 s | 9.08 / 124.9 s | 8.04 / 113.1 s |
| w/o pipelined execution | 47 | 9.25 / 147.2 s | 6.87 / 122.8 s | 6.25 / 114.8 s |

**成功率的差异（47–54）都落在同配置重复实验 ±10 的波动范围内**，所以不把它当作结论，结论以调用次数和 e2e 为准。逐题看，各组相对 FULL45 输赢的题互有进出，见第 4 节。

## 2. 每组 e2e 由哪几部分组成（五组共同成功的 28 题，每题平均）

| 组 | 模型决策 | 动作执行与 UI 等待 | 图执行 | 启动与收尾 | 合计 | 模型调用 | 每次调用的决策时间 |
|---|---|---|---|---|---|---|---|
| BASE | 101.3 | 3.1 | 0 | 10.7 | 115.1 | 6.86 | 14.8 s |
| FULL45 | 66.9 | 6.5 | 3.7 | 10.9 | 88.0 | 5.11 | 13.1 s |
| w/ random search | 67.8 | 7.5 | 3.7 | 11.1 | 90.1 | 5.18 | 13.1 s |
| w/ prompt injection | 89.1 | 12.8 | 0 | 11.2 | 113.1 | 7.96 | 11.2 s* |
| w/o pipelined execution | 91.7 | 6.1 | 5.8 | 11.2 | 114.8 | 6.14 | 14.9 s |

- 表中单位均为秒。“合计”与第 1 节的 e2e 一致。
- “模型调用”一列按 steps 统计，与第 1 节的 calls 字段有约 0.1 的差异。
- \*INJ 每次调用较短，是因为其中包含大量解析失败的短输出（约 29 个 token），见 3.2。

FULL45 相对 BASE 共省 27.1 s。拆开看：决策时间省 34.4 s，图执行和动作执行多花约 7 s。

## 3. 各组件的贡献

### 3.1 图执行贡献了调用次数的节省（w/ prompt injection）

- **调用次数：** 把图执行换成“只注入 prompt”后，五组共同成功题上调用从 5.11 升到 8.04，比 BASE 的 6.86 还多。全部 116 题上是 11.60，同样高于 BASE 的 10.14。e2e 也退回到 BASE 水平（113 s 对 115 s）。
- **图本身仍然在构建：** INJ 的探索照常进行（probe 总数 163，与 FULL45 相同），prompt 注入了 821 次。区别只在于图里的信息交给模型自己用，还是由系统直接执行。
- **注入不但没帮上忙，还会伤害 4B 模型的输出格式：** INJ 有 147 步出现“Parser fallback wait”（输出无法解析），分布在 68 个 episode 中；FULL45 只有 16 步（9 个 episode），BASE 有 28 步（17 个 episode）。
- **扣掉解析失败后结论不变：** 扣掉这些步后，INJ 每题约 10.3 次调用，BASE 约 9.9 次，仍然没有节省。
- **结论：** 节省的来源是“由系统执行图里已经验证过的转移，从而移除一次模型决策”，而不是“给模型更多上下文”。这与之前 prefill 优于 injection 的结论一致。
- **典型题目：**
  - ExpenseAddSingle、RecipeAddSingleRecipe、RecipeDeleteMultipleRecipes：FULL45 比 INJ 少 9–12 次调用、100–120 s。
  - ExpenseAddMultiple、RecipeAddMultipleRecipes：FULL45 成功，INJ 跑满 20 步后失败。

### 3.2 流水线执行贡献了约一半的 e2e 节省（w/o pipelined execution）

- **全部 116 题：** NOSYS 的调用次数与 FULL45 几乎相同（9.25 对 9.16），但 e2e 是 147.2 s 对 135.2 s，FULL45 少 12 s。
- **五组共同成功题：** NOSYS 的 e2e 为 114.8 s，基本等于 BASE 的 115.1 s，比 FULL45 多 26.8 s。这 26.8 s 大致由三部分组成：
  - **每次调用变慢（约 11 s）：** 每次决策 14.9 s 对 13.1 s，多出约 1.8 s，就是等待完整解码、动作后固定等待带来的时间，乘以 6.1 次调用约 11 s。
  - **调用变多（约 14 s）：** 五组共同成功题上多 1.0–1.1 次调用，按每次约 13 s 计约 14 s。原因是 NOSYS 同时关闭了 R2 复用（图复用 33 次，FULL45 为 46 次），且窗口更长时探索更多。不过全部 116 题上两组调用几乎一样，所以这部分有一定的题目依赖。
  - **图执行步更慢（约 2 s）。**
- **结论：** 流水线执行（流式早动作、自适应等待、像素检查）不减少调用次数，但单独贡献每次调用约 12% 的决策时延。配对比较的 40 道双方都成功的题上，NOSYS 平均比 FULL45 慢 16.9 s/题，调用只多 0.5 次。

### 3.3 探索的排序策略对时延几乎没有影响（w/ random search）

- 把 FULL45 的启发式排序换成随机后，调用（5.21 对 5.11）、e2e（90 s 对 88 s）、时延分解都几乎不变。在 41 道双方都成功的题上，调用差 +0.07 次，e2e 差 +0.5 s。
- 随机排序下 probe 总数从 163 降到 83（因为去掉了 E1/RV 目标），但图执行次数基本不变（例如 bootstrap 都是 76 次）。
- **结论：** 时延节省主要来自同任务内的图执行（bootstrap、function、form、reuse），它们依赖的是 agent 自己走过的转移，而不是探索发现的转移。这与之前 expansion reuse bound 的结论（瓶颈在复用而非发现）一致。
- **论文写法建议：** 如实写成“exploration ordering 对 latency 影响很小，主要作用是在新状态下提供恢复能力”。不要声称启发式搜索带来加速。
- **成功题差异：** RAND 少 4 道成功题，其中 Calendar 写入类较多（AddOneEvent、AddRepeatingEvent、DeleteOneEvent 等），但同时多赢了 9 道，在 ±10 的噪声以内。

## 4. 异常题的轨迹核查

| 题目 | 现象 | 核查结论 |
|---|---|---|
| SimpleCalendarLocationOfEvent | BASE 和 FULL45 成功，三个消融组都失败 | RAND 和 NOSYS 的动作序列与 FULL45 完全相同（bootstrap 加 3 次点击，再回答），失败在于最终答案的内容，属于模型回答的随机性。INJ 开头两步都是解析失败，之后走了 10 步。与组件无关。 |
| SystemBrightnessMinVerify | FULL45 用 4 次调用，三个消融组都要 9–10 次 | 三组都卡在亮度滑条上反复 swipe，属于模型操作的随机性。FULL45 这次是一次拖到位。这是 RAND 相对 FULL45 差距最大的一题（+106 s）。 |
| SimpleCalendarEventsInTimeRange | RAND 和 NOSYS 要 10 次调用 | 两组都在周视图里来回翻页，FULL45 和 INJ 是直接定位。属于模型随机性。 |
| RetroCreatePlaylist | 只有 BASE 成功 | INJ 没有图执行也失败了，所以不是 form_function 造成的。几组都在建完歌单后的加歌步骤上走偏，属于 BASE 单次碰巧成功。 |
| ClockStopWatchPausedVerify | BASE 和 RAND 第 0 步就直接回答（1 次调用） | 初始状态下模型直接判定任务已完成，属于环境和模型判断的随机性，与组件无关。 |

## 5. 给论文 “Understanding MobileExplorer's performance” 的结论

1. **调用次数的节省来自图执行。** 只注入 prompt 不仅不省调用（8.0 对 6.9），还会把解析失败率提高约 9 倍。
2. **e2e 节省约一半来自流水线执行。** 去掉后，五组共同成功题上的 e2e 从 88 s 回到 115 s，每次决策慢约 1.8 s。
3. **探索的排序策略不是加速来源**（随机排序 90 s 对 88 s）。探索的作用是给图提供可执行的转移，并负责新状态下的恢复。
4. **两类机制互补，任意替换一个都会让 e2e 退回 BASE 附近：** 图执行减少的是“需要多少次决策”，流水线执行减少的是“每次决策要等多久”。

**可直接用的图：** `figures/understand/g_ablation.pdf`（调用次数和 e2e 两个子图，各自单独的坐标轴）和 `h_ablation_breakdown.pdf`（每组的时延分解）。

**遗留问题：** e2e raw 目录结构下，`EV.load_cap` 找不到目的页截图，所以 TF-IDF 目的页证据在 e2e 运行中一直没有生效（FULL45 和三个消融组都一样，所以对比本身是公平的）。论文 5.3.1 节如果描述了这一证据，需要修正代码重跑，或者调整文字。
