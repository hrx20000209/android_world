# 给 LabServer Codex 的 AndroidWorld 实验与优化 Prompt

复制下面代码块，交给能访问 LabServer 的 Codex。它以仓库里的 posterior-coverage 实验为主线，要求先检查环境、跑小规模 smoke，再在安全条件允许时扩展配对评测并做单因素优化。

```text
你要在 LabServer 上独立运行 AndroidWorld，评测并逐步改进本仓库的 posterior-coverage 在线探索方法。所有代码、进程、模拟器和结果都必须留在服务器；不要依赖开发者 Mac 的文件、ADB、模拟器或 vLLM。

## 目标与实验对象

核心实现：
- android_world/parallel_exploration/posterior_coverage.py
- scripts/run_posterior_coverage_task.py

这个 runner 在原始 GELAB agent 的完整推理请求进行时，为安全可点击控件计算候选动作 posterior，并尝试 live probe。比较以下五臂：
- none：不做 probe；端到端无探索基线。
- current：现有 GraphKeywordRanker 的排序。
- random：固定种子的均匀随机选择。
- probability：按 posterior 概率从高到低选择，不根据预算优化。
- posterior_coverage：以候选概率质量为收益、在线测得的 probe 耗时为成本，在当前推理剩余时间内解 0/1 knapsack，并在每次 probe 后重新规划。

注意：此 posterior runner 是受控实验 harness，不代表方法已经自动接入默认 AndroidWorld/MobileExplorer 路径。它的 probe 会在同一任务设备上直接点击，然后根据主模型最终动作决定提交该点击或恢复；这不是独立 shadow emulator。只允许使用专用、可丢弃、已快照的 AndroidWorld 模拟器，绝不能对个人手机或有用户数据的设备运行。探测/覆盖指标不是任务成功率，必须单独报告。

## 1. 服务器、Git 和数据安全

- 仓库预期路径：`/data/rxhuang/android_world_repo`
- 结果预期路径：`/data/rxhuang/android_world_server/runs/`
- 后者是运行数据目录，不是 Git 仓库；不要对它 `git init`、清理或 reset。
- 先检查 hostname、当前路径、Git 分支/commit/status、磁盘、GPU、ADB 设备、现存 AndroidWorld/vLLM 进程和监听端口。仓库缺失时再 clone `git@github.com:hrx20000209/android_world.git`；仓库干净时可 fetch 并 fast-forward 到远端。若有未提交修改，不覆盖、不 stash、不 reset；记录现状并在独立工作区继续或先报告阻塞。
- 已有运行和输出一律保留。每个新实验使用带日期/commit/实验名的唯一目录；不续写已有实验目录，不重复已完成的 task×method 组合。
- 不上传模型、APK、模拟器镜像、截图、checkpoint、日志、JSONL 结果或其他大文件到 GitHub。不要在命令、日志或报告里打印口令、token、私钥；使用服务器现有 SSH agent/凭据，勿把秘密写进脚本或仓库。
- 禁止 `git reset --hard`、删除用户数据、杀掉不属于本实验的进程或重启已有服务。任何资源归属不清楚时先保持原状。

## 2. 固定采样设置

所有对比臂和优化臂固定，不可为了提升分数而改变：
- 完整回答：`temperature=0.0`、`top_p=1.0`（runner 的 reasoning 请求使用 agent wrapper temperature，并明确发送 top_p=1）。
- 候选 posterior scoring：代码固定 `temperature=0.0`、`max_tokens=1`，使用 vLLM `prompt_logprobs`。
- vLLM server seed：`0`；只有确需启动新服务时才启动，并在启动参数和日志中确认 seed=0。
- AndroidWorld task seed：`34`，并启用 `--fixed_task_seed`（runner 已设置）。
- 所有 arm 使用完全相同的模型、task manifest、初始 app/device 状态、step budget、`max_probes=10`、`score_workers=8` 和评测版本。

不要改温度、seed、任务顺序、模型、预算或多个算法因素来制造收益。若要研究其中一个变量，另建明确标记的诊断实验，不混入主对比。

## 3. 运行前检查（不满足就先修环境，不要启动大批次）

1. 盘点 `nvidia-smi`、`ps`、`ss -ltnp`、`adb devices -l`、可用内存/磁盘、容器和 AVD。不能只凭端口号猜服务用途；确认每个 endpoint 的 `/v1/models`、served model、GPU 映射、日志和负载。优先复用确认空闲且兼容的 GELAB-ZERO-4B OpenAI-compatible vLLM；绝不停止或覆盖现有服务。若确实需要启动新服务，先确认有空闲 GPU/显存，复用仓库 `scripts/start_labserver_vllm.sh` 的 seed=0 设置，并将端口、GPU、模型路径、启动日志写入实验 manifest。
2. posterior 方法要求 endpoint 支持本代码用到的 `prompt_logprobs`、`continue_final_message`、`add_generation_prompt`，且与 chat/completions 接口兼容。用一个无副作用的小请求验证返回包含可读的 `prompt_logprobs`；随后再用单个候选验证分数解析和归一化。若不支持、报错或有效候选比例为 0，停止 posterior 矩阵，记录请求/服务端错误；不要把它记为 agent 失败。
3. 确认 ADB 中有专用 `emulator-5554`，因为当前 runner 固定使用该 serial 和 console port 5554。通过 `command -v adb` 找到服务器 adb，并设置 `ANDROID_WORLD_ADB_PATH`，避免代码默认的 Mac adb 路径。检查 AndroidWorld 依赖、a11y/fast provider 端口及应用初始化是否可用。`current` arm 的 `GraphKeywordRanker` 可能还依赖 semantic service（默认 port 8766）；依赖不可用时先诊断并单独记录，不要静默改变 arm。
4. runner 会真实操作同一 emulator，且当前 CLI 不提供可配置 ADB serial/console port 的选项。因此默认串行运行 task×method。只有每个并发 runner 拥有完全独立的容器/ADB namespace、AVD userdata snapshot、emulator serial/console port、a11y 端口、vLLM 配额、stats 和输出目录，且确认不会共用状态时才允许并行；否则绝不并发。
5. 每个 task×method 开始前，用 AndroidWorld 自身 reset 加独立快照恢复到同一干净初态；确认恢复成功并记录快照 ID/hash。不可只假设固定 seed 会清理先前运行留下的 app/device 状态。发生 recovery failure、`dirty=true`、runner 仍在运行但 AVD 状态不明时，暂停该设备上的后续运行，先恢复/核验快照。

## 4. 实验流程

### A. Smoke

- 从仓库已有固定 manifest 或服务器上可复现的既有 AndroidWorld baseline 选择 1 个有效、包含可点击 UI 的任务；固定写入 manifest，不临时换任务。
- 先对五个 arm 各跑一次，确认 AndroidWorld evaluator 能结束、日志/checkpoint 可读、完整回答请求成功、posterior 解析正常、探测恢复安全。每次用全新的输出目录及同 seed 初态。
- `none` 不做 posterior scoring；其他四个探索 arm 都计算候选分数。因此，`posterior_coverage` 与 `current/random/probability` 的对比用于判断选择策略；与 `none` 的对比还要计入 posterior scoring 和探索开销。明确报告这一点。
- posterior 数据至少核查：`posterior_errors`、有效候选数、`posterior_sum`（有候选且无错误时应接近 1）、posterior latency、probe cost、reasoning slack、`rollback_ok`、`dirty`、`speculative_commit` 及 AndroidWorld 最终任务结果。
- 若 smoke 中出现可疑状态污染、未恢复、posterior 接口不兼容、连续请求失败或应用/基础设施错误，先停，不要扩成多任务矩阵。

### B. 配对 pilot 与完整评测

- Smoke 通过后，从同一固定 manifest 中选 5 个具有代表性的任务做 pilot；同一 task 的五个 arm 逐一恢复到同一初始快照。保留 task 顺序和 seed 34。
- Pilot 没有系统性基础设施错误且设备恢复可靠时，继续该 manifest 的固定 30-task 子集；若仓库/服务器已有相同口径的 30-task manifest，优先复用并记录来源。若没有，先从 AndroidWorld 可用任务中固定并保存一个分层 task list，再开始任何完整臂。不要在看过中途分数后换题。
- 每个 arm 使用独立 `stats.json`：给所有 arm 相同代码默认 latency prior（reasoning 4.0s、probe 1.5s），之后只允许各自用本 arm 历史在线更新，避免不同 arm 互相泄漏成本统计。每个 task 使用独立 output 目录。任务状态必须按 AndroidWorld evaluator 判定；进程退出/产生 JSONL 不能替代成功判定。
- 推荐单任务命令模板（在服务器 shell 中先设定实际检查过的 `API_URL`、`MODEL_ID`、`RUN_ROOT`、`TASK` 和 method；不要原样猜端口/模型）：

  `ANDROID_WORLD_ADB_PATH="$(command -v adb)" python scripts/run_posterior_coverage_task.py --task "$TASK" --method posterior_coverage --output "$RUN_ROOT/posterior_coverage/$TASK" --stats "$RUN_ROOT/posterior_coverage/stats.json" --seed 34 --api_url "$API_URL" --model "$MODEL_ID" --max_probes 10 --score_workers 8`

  其他 arm 只把 `--method` 和隔离的 output/stats 子目录改为 `none`、`current`、`random` 或 `probability`。runner 额外输出 `steps.jsonl`、`filtered_elements.jsonl`、`run_args.json`；AndroidWorld 自己的 evaluator/checkpoint 也要一并归档在该任务目录。

## 5. 结果读取与判断

每一轮都保存运行 manifest（commit、hostname、UTC 时间、task 列表、每臂命令、seed/temperature/top_p、API/model、GPU/端口、AVD snapshot ID、依赖版本、输出目录）。报告至少包括：

- 任务数、成功/失败/基础设施错误和 evaluator success rate；按 task 配对列出分歧，不把 infrastructure error 算成 agent failure。
- 每任务端到端时间、step latency、full reasoning 时间/TTFT/output tokens、posterior latency、probe 数与实际 forward/settle/recovery 耗时、总 probe overhead、final action 是否在推理结束前已 probe、commit/rollback/recovery 成功率。
- posterior 错误率、候选数、候选概率分布/覆盖率；CPU、内存、GPU/显存和 vLLM 请求/队列负载（若指标可用）。
- 平均数之外同时报告中位数、分位数和逐任务明细；只在共同完成的 task 上做配对 latency 比较。请求数少、探测覆盖高、或单一任务成功都不是设计有效的充分证据。
- 从 `steps.jsonl` 读取机制数据，从 AndroidWorld 官方 evaluator 输出读取任务成败；如果 evaluator 结果缺失，就标为未确定，不补猜。

## 6. 单因素优化循环

1. 先完成并分析 pilot/当前完整臂，找出具体瓶颈（例如 posterior 请求延迟、候选错误、成本预测偏差、回滚时间或有 probe 无收益的任务）。
2. 提出一个可证伪假设，一轮只改一个因素；温度、top_p、seed、模型、任务 manifest 和初态保持不变。
3. 修改前检查现有工作树，使用独立分支/工作区，不动正在运行的代码和结果；增加或更新小范围单测，并运行相关单测及一个隔离 smoke。不要通过测试后再覆盖旧实验。
4. 新方案用新 commit 标识和新结果目录，与同一固定 task manifest 做 paired comparison。没有完整或足够配对证据时，将其标为探索性结果；有退化、恢复失败或异常时停止推广并保留证据，不删除旧结果。
5. 仅在完整 evaluator 结果与配对开销均支持时建议保留优化。无法证明有益时，明确报告无结论/负结果，不把覆盖率当成功率。

## 7. 交付

最后用简洁中文给出：
- 服务器 hostname、仓库 commit/status、服务/GPU/AVD 映射和确定性参数；
- task manifest、所有实验臂完整命令、每臂完成情况及结果表；
- 成功率、配对变化、端到端/推理/posterior/probe/recovery 开销、posterior 错误和基础设施失败；
- 有代表性的失败及服务器日志/JSONL/checkpoint 路径；
- 本轮改动的假设、代码 diff/commit、测试与评测证据、限制和下一步。

只在服务器保留运行产物；不要把大文件或结果树 push 到 GitHub。状态有疑问或恢复不安全时，暂停对应实验并先说明问题，不要盲目重启。
```
