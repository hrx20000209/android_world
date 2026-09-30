# 给 LabServer Codex 的 AndroidWorld 运行与优化 Prompt

将下方 prompt 交给能访问 LabServer 的 Codex。它要求先核实服务器现状，再沿用当前确定性配置验证“任务条件化探索图”，并以强历史复用基线为参照做小步、可归因的优化。

```text
你要在 LabServer 上运行 AndroidWorld，验证并改进仓库中的 MobileExplorer / Executable Exploration Memory。源码、Android worker、vLLM 请求和结果都在服务器端；不要依赖开发者 Mac 的文件、进程、ADB、模拟器或 vLLM。

## 目标：必须分别回答三个研究问题

1. 建图：建立与 Android GUI 交互结构有关、又由当前任务条件化的图，而不是只做截图去重或积累通用页面访问次数。状态是稳定 UI schema（activity、可访问性控件/结构），边是 selector/action/落点及支持数、成功率、成本、可逆性、失败/无效/副作用证据和 provenance。当前任务只以归一化、不可读回的语义签名参与边排序；不要把原始任务、用户输入、答案、联系人/日期/电话等值写入跨任务图。
2. 探索：先由既有 Ex5 安全层决定可执行候选，再让图做有界 tie-break。区分已知成熟边、需要复核的弱边和当前 UI 上任务相关的未探索 frontier；在有安全 frontier 时少重复成熟边，目标是取得有用的新状态/证据，而非单纯增加覆盖率。记录每次选择为何与任务有关、是否到达新节点、probe/recovery 成本及副作用。图绝不提升不安全控件的权限。
3. 推理：只把有限的任务相关信息注入 prompt：当前 UI、相关且已观察的导航路线、未验证 frontier（明确标注不是事实/指令）、以及相关 trap/no-op。不要倾倒整图。历史图只能指导导航；只有短、稳定、任务对象匹配且多次验证的导航前缀可跳过 VLM。每跳都要对当前 UI 重新定位 selector 并验证落点；抵达可编辑表单或遇到 Save/Submit/Confirm/Delete 等复杂/提交动作前必须停止，让 VLM 基于当前页面重新推理。不能靠图直接代填当前任务值或自动提交。

研究边界：AutoDroid 的 UTG 与 task/function memory 已经证明 GUI 图、自动探索和任务记忆注入不是新颖点（原文：[AutoDroid](https://arxiv.org/abs/2308.15272)）。OmniFlow 已做成功历史轨迹的可参数化 function/transition reuse 与 live re-grounding；SwiftAgent 已做有序 reuse points、gap action 与局部恢复。不要声称“我们用了图”“图是 GUI-specific”或“能重放历史路径”本身是贡献。要检验的差异只能是：图是否在当前任务中识别了有价值的未探索控件/证据缺口、提前获取新状态信息、并且相对 validated replay/UTG 基线带来更好的端到端结果。若没有增益，诚实报告并收窄结论。

## 1. 服务器安全与工作区

- GitHub：`git@github.com:hrx20000209/android_world.git`
- 现有主 checkout：`/data/rxhuang/android_world_repo`
- 运行数据：`/data/rxhuang/android_world_server/runs/`
- 先记录 hostname、UTC 时间、`git status`/commit、磁盘、GPU/显存、ADB 设备、容器、监听端口、AndroidWorld/vLLM 进程和 `/v1/models`。只用服务器已有 SSH 凭据；绝不索要、显示或写入密码/token/私钥。
- 现有 checkout 可能正被 AndroidWorld `run.py` 使用。若检测到活动进程，不在该目录 pull、切分支、覆盖代码或安装依赖；从远端最新 commit 创建一个新、独立 worktree（先确认目标不存在），在新 worktree 改代码。保留活动进程及其代码版本。
- 结果必须写入唯一目录 `/data/rxhuang/android_world_server/runs/<worker>/<日期>_<commit>_<arm>/`。不要覆盖、清理、reset、挪动任何旧结果、checkpoint、AVD 文件或运行目录。不要杀不属于本轮且归属不清楚的进程，不要重启已有服务。
- 不要向 GitHub 上传模型权重、APK、模拟器镜像、截图、checkpoint、日志、JSONL 结果、`evaluation_results/`、`tmp/` 或大文件。提交只包含必要源码、测试、脚本和本文档。
- 若目录冲突、工作树 dirty、设备/服务归属不明，先保留现场并报告；禁止 `git reset --hard`、强推、删除式清理或直接改活动 checkout。

## 2. 延续已有 AndroidWorld 实验，不重复已完成任务

先检查以下既有运行及其真实 checkpoint/evaluator 状态（路径仅作定位，不代表应重跑）：

- worker-a：`/data/rxhuang/android_world_server/runs/worker-a/baseline_20260924.log`，原始 GELAB baseline，原记录使用 vLLM 8083。
- worker-b：`/data/rxhuang/android_world_server/runs/worker-b/design_current_20260924_orchestrator.log`，MobileExplorer + two_system + executable memory + semantic prefix，原记录使用 vLLM 8084。
- worker-c：`/data/rxhuang/android_world_server/runs/worker-c/baseline2_20260924.log`，独立 baseline 复现，原记录使用 vLLM 8085。
- worker-d：`/data/rxhuang/android_world_server/runs/worker-d/design_v2c_20260924_orchestrator.log`，V2 `max_probes=2, min_probes=1`，原记录使用 vLLM 8085；只统计 `design_v2c_20260924`，忽略 `invalid_v2*` 诊断目录。

旧记录曾报告上述四路各完成 30 个任务；不要仅凭这段文字认定完成或失败。逐一读 orchestrator、task manifest、run_args、checkpoint、evaluator 输出及进程。已完成的 task×arm 不得重复计数或无故重跑。若进程停止，先确认是否正常完成及最后一个可靠 checkpoint；只有 runner 明确支持 resume 且输出/设备状态可验证时才续跑。不能安全恢复就保留现场并报告，不要盲目从头启动。

新实验先做 1 个隔离 smoke，再做配对 pilot，确认 AVD 状态、恢复和 evaluator 正常后才扩到预先冻结的 task 子集。baseline 和优化臂必须使用相同 task、初始状态、step budget 和 evaluator。若进行多个并发臂，每臂必须独占 AVD/userdata snapshot、ADB namespace/serial/console port、a11y 端口、输出目录和必要的 vLLM/GPU 配额；共用设备则串行。单纯开多个 AndroidWorld 进程不叫隔离。

## 3. 确定性：保持既有设置，不得为结果改采样

- 首先从有效 `run_args.json`、启动脚本、vLLM 命令行和服务日志读取每个已有实验实际使用的 temperature、top_p、vLLM seed、AndroidWorld task seed、模型和 task 顺序；结果报告引用这些已验证值，而不是猜测或套用另一条 runner 的默认值。
- 当前项目既有对照规范是完整回答 `temperature=0.0, top_p=1.0`、vLLM `seed=0`、AndroidWorld seed `34` 并启用 `--fixed_task_seed`；posterior scoring 请求也固定 temperature 0。只有经文件/日志确认这正是该实验臂的生效设置时才按此值运行。
- 对延续中的实验，精确保留其生效设置；任何 baseline/优化臂不得更改 temperature、top_p、seed、模型、task manifest/order 来制造差异。若两个旧臂设置不同，先标记不可直接配对，不能偷偷改成一致后把旧结果混比。新实验若缺少明确设置，先暂停并报告，不擅自选择温度。

## 4. vLLM、ADB 与任务状态核验

- 不要按端口猜服务用途。检查每个端点 `/v1/models`、served model、GPU 映射、日志和负载。只有确认空闲且兼容时才复用服务；posterior runner 还须核实 `prompt_logprobs`、chat template 参数与 scoring parser。
- 先 `command -v adb`，并为服务器进程指定服务器上的 adb；确认 `adb devices -l`、a11y/fast dumper、forwarding、AndroidWorld app 初始化正常。绝不使用 Mac adb。
- 使用专用、可恢复快照的 AndroidWorld emulator/AVD；绝不在个人手机、有用户数据设备或共享 AVD 上试验。固定 seed 不等于重置设备。每个 task/arm 开始前记录 AVD/snapshot 标识并核验恢复；发生 `dirty=true`、restore failure、状态不明或 app 数据污染时暂停该 worker。
- evaluator 的任务判定是唯一成功依据。进程退出、动作数、checkpoint 行数、探测覆盖和“生成了 memory”都不能代替任务成功。

## 5. 评测臂与单因素优化

按现有实现和资源能力选择，但至少保留：

1. `gelab_agent` 原始 baseline（无本设计）。
2. 完整当前设计（当前实验臂实际启用的 two-system、memory、prefix 等模块需逐项核实）。
3. 单因素图消融：图仅记录；图引导 exploration；任务相关 prompt summary；保守 navigation-prefix skip。分别开关，不叠加多个变化。
4. 强历史复用对照：live selector re-grounding 的 validated trajectory/action replay（OmniFlow/SwiftAgent 风格）；并尽量做 AutoDroid-style UTG/task memory 对照。比较图与 replay 的互补收益，而非只对比弱的纯文本轨迹 prompt。

一次优化只写一个可证伪假设，例如“task-relevant unseen frontier 能减少成熟边的重复 probe，且不降低 evaluator 成功率”。先补针对性单测，再运行隔离 smoke 和同一冻结 task 集配对 pilot。改变温度或 seed 不属于可接受的算法优化。

针对当前代码，重点验证：

- 图建构/检索是否随当前 activity 的节点数和 outgoing degree 增长，而不是每步扫描整张图；state 合并是否不会把动态值/表单数据写入长期 memory。
- Ex5 safe candidate admission 不变；graph frontier 的 tiebreak 有界、成熟 transition 在存在安全 frontier 时避免重复探索、hard-negative/no-op 能抑制重试。
- prompt 只包含少量当前任务相关的路线/frontier/trap，并将 frontier 明确标成“未验证”；提示不会泄露其他 task 的值或把图的动作误当成已执行。
- navigation-prefix skip 在起点和每跳重新 grounding；每个 landing 都验证，分歧后安全回滚；到可编辑字段或提交/副作用控件之前收口。route failure、rollback failure 和错误 skip 必须单独计数。
- graph/history/replay 相比 baseline 是否真的节省 VLM calls、端到端时间或提升任务成功率。selector 命中、route candidate、coverage 或 probe count 都只是机制指标。

## 6. 日志分析与决策

每个 worker 定期检查：PID/进程状态、当前任务与 checkpoint、最后日志时间、任务成功/失败/infrastructure error、每任务耗时/step、primary 与 exploration VLM 请求/token、probe 数及耗时、rollback/restore、route hit/miss/block reason、memory 节点/边/frontier、vLLM 错误/队列、GPU 利用率/显存、AVD 状态和异常堆栈。异常要归类为 agent、任务/evaluator、模型服务、设备/依赖基础设施，不得混算。

报告需给出：task×arm 配对表、成功/失败/infra 数及 success rate、共同完成任务上的端到端/step latency（中位数及分位数）、primary VLM calls、probe/recovery overhead、错误 skip 与路线 block 原因、GPU/服务状态。列出代表性失败和对应日志/checkpoint 路径；结果不足时标“探索性/无结论”。

若一臂停止，先保存当前证据并核对可恢复点，不能重复任务；若出现恢复失败、误跳过、成功率回归、共享设备冲突或采样设置漂移，暂停该臂并报告。所有旧数据只追加解释，不覆盖。

## 7. 最终交付

最后用中文交付：服务器 hostname、代码 commit/status、新 worktree 路径、GPU/vLLM/AVD 映射、核实过的确定性设置、task manifest、每臂命令和已完成任务；配对结果和失败分类；机制开销与错误；代码修改假设、测试/评测证据、适用限制及下一步。只保留运行产物在服务器，不向 GitHub 推送运行数据或大文件。
```
