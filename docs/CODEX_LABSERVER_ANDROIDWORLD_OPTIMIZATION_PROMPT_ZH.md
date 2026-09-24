# LabServer AndroidWorld 运行与设计优化 Prompt

把下面这段直接交给运行在 LabServer 上的 Codex。仓库地址是：
`git@github.com:hrx20000209/android_world.git`

```text
你现在运行在 LabServer 上，目标是在服务器端独立运行 AndroidWorld，并持续验证和优化 MobileExplorer 设计。不要依赖开发者 Mac 的本地文件、进程、ADB、模拟器或 vLLM；所有源码、结果和运行状态放在服务器自己的工作目录中。

## 1. 工作目录与安全边界

- 工作目录：`/data/rxhuang/android_world_server`
- 如果目录不存在，从 `git@github.com:hrx20000209/android_world.git` clone；否则先检查 git status，再按需 pull。
- 不要上传或提交模型权重、APK、模拟器镜像、截图、日志、checkpoint、JSONL 结果、`evaluation_results/`、`tmp/` 或任何大于 50 MB 的文件。
- 结果写入 `/data/rxhuang/android_world_server/runs/`，源码修改留在 git 分支或清晰的 commit 中。
- 不要使用 `git reset --hard`、删除用户数据或覆盖已有实验结果。新实验必须使用新的输出前缀和独立 worker。

## 2. 确定性设置

所有 baseline、ablation 和优化实验必须保持：

- client `temperature=0.0`
- client `top_p=1.0`
- vLLM `seed=0`
- AndroidWorld `--task_random_seed=34`
- `--fixed_task_seed`

不要通过改变 temperature、seed、task 顺序或随机采样来制造收益。报告中同时记录模型 API 地址、served model、GPU、代码 commit 和任务列表。

## 3. vLLM 与模拟器

- 使用 OpenAI-compatible GELAB-ZERO-4B vLLM。
- 每个 vLLM 实例只绑定一个 GPU，端口和 GPU 映射必须在日志中记录。
- 使用 `scripts/start_labserver_vllm.sh` 或等价命令启动服务；启动后检查 `/v1/models`、`/metrics`，确认日志中的 `seed=0`。
- 每个 AndroidWorld worker 使用独立容器、独立 AVD 和独立 `/runs/<worker>` 目录。
- 运行前确认 `adb devices`、a11y forwarding、fast_a11y_dumper、`openai`、`matplotlib==3.6.1`、`ImageHash` 和 `numpy==1.26.3` 都可用。
- 如果某个 vLLM 端口已有非 vLLM 服务，不要停止或改写它。

## 4. 必须保留的实验臂

### A. 原始 baseline

使用 `run.py --agent_name=gelab_agent`，不启用 MobileExplorer、two-system、probe、memory 或 semantic prefix。

### B. 当前设计

使用：

- `mobileexplorer_executable`
- `--variant full`
- `--two_system`
- executable memory
- semantic prefix assist
- online live probe

所有共享 memory/prefix 必须只属于该实验臂，不能跨 worker 或跨 ablation 污染。

### C. 优化设计

每次优化只改变一个明确因素，例如 probe budget、触发门控或 parser 修复。不要同时改模型、seed、任务集合和多个核心算法。每个优化臂必须有单独目录，例如：

`runs/design_v2_<date>_<change_name>/`

## 5. 当前已知问题与优先级

先检查并修复这些问题，再声称设计有收益：

1. `FilesDeleteFile` 曾出现 `ValueError: No app name provided`，检查 action parser 到 `execute_adb_action` 的字段完整性。
2. 当前完整设计的 semantic-prefix `route_hit_count` 很低或为 0，不能把已生成 memory 当作真实复用收益。
3. 统计必须区分 primary VLM request、exploration request、recovery、prompt token、端到端 step latency 和额外探索时间。
4. task-local metrics 与 shared cumulative metrics 分开报告，不能把累计 snapshot 当成单任务结果。
5. a11y、容器依赖、目录冲突和 runner 导入错误不能计入模型失败；必须作为 infrastructure error 单独统计。

## 6. 评测方法

- 先跑固定的 30-task AndroidWorld 子集。
- baseline 与设计臂使用相同任务、相同 seed、相同初始 AVD 状态和相同 step budget。
- 至少完成一轮完整 30-task 后，才比较 success rate。
- 报告：完成数、成功数、失败数、infrastructure error 数、成功率、平均/中位 step latency、总运行时间、vLLM request/token 统计、probe 数、recovery 成功率、额外内存和额外时间。
- 优先做 paired task-level comparison；只在双方都完成的任务上比较步数和延迟。
- 如果结果为负，不要继续堆加模块；先定位失败任务和开销来源。

## 7. 优化循环

每次循环按以下顺序执行：

1. 读已有日志和结构化结果，定位失败最多或开销最高的任务。
2. 提出一个可证伪的单因素改动。
3. 写测试，运行相关单测和最小 smoke task。
4. 在独立 worker 上跑 ablation，不覆盖旧结果。
5. 只有完整任务集和配对分析都支持时，才保留改动；否则回到上一个稳定 commit。
6. 每轮结束写一份简短报告，包含 commit、命令、设置、结果和下一步。

不要把“请求数减少”直接等同于“端到端更快”，也不要把“探索收集到证据”直接等同于“任务成功率提高”。

## 8. 最终交付

最终回答必须给出：

- 当前 git commit 和工作树状态
- 每个实验臂的完整命令和 vLLM/GPU 映射
- 30-task 结果表
- 失败分类和至少三个代表性失败日志位置
- latency、probe、recovery、token 和 memory overhead
- 当前设计是否值得保留，以及证据和限制
- 下一步最小可行优化

完成后不要上传大文件或实验结果树；只提交必要源码、测试、运行脚本、依赖和文档。
```
