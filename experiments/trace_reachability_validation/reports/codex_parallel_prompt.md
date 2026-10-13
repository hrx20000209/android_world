# 给 Codex 的并行任务（与正在运行的 FULL3 同时进行）

## 最终目标：一份有实测依据的完整设计
下面的任务 A/B/C 都不是独立的实验。它们是为了补齐一份完整设计里缺的证据。最后要交付 `experiments/parallel_offline/reports/design_final_zh.md`，回答下面三个问题。**每个设计模块都必须写明"动机（实测数字 + 出处文件）→ 技术设计"**，不能写成启发式规则的罗列。

### Q1 动机：为什么要图 + 探索
- 出发点是 **runtime dynamics**：同一 app、同一任务在不同运行之间会变化。
  - 实测：截图 62%、a11y 50% 会变，但 rid 只有 2.9% 会变，转移明确不同的只有 1.4%。也就是说，变化主要在内容层，页面结构和转移基本稳定。
  - 直接回放历史轨迹会**悄悄失败**（24/36）。
  - 见 `experiments/runtime_graph_study/reports/final_report_zh.md`。
- 希望达到的效果：利用"结构稳定、内容变化"这一点，用图 + 探索**加速推理、减少推理步数（模型调用），让推理的动作更准确**，同时不降低成功率。
- 你需要把这条动机链写完整：变化在哪一层 → 为什么不能直接回放 → 图需要抽象到哪一层 → 运行时必须验证什么。

### Q2 探索什么信息？哪些信息、以什么方式帮助推理？
- **已有结论**：把探索信息注入 prompt，基本都是负面或无效的。
  - 6 种注入形式加安慰剂对照，全部无净收益，见 `TRV/reports/final_report_zh.md`、`TRV/analysis/pilot_b/`；
  - 仓库更早的实验也是"预填优于注入"。
- 因此，探索信息应主要作用在**推理完成之后的系统侧**，不是作为上下文交给模型。具体有三种：
  - **验证**模型选中的动作会到达哪里；
  - **预观测**：提前拿到执行后的下一屏，省掉等待；
  - **跳步 / 预填**：直接替模型执行已经验证过的下一步，省掉一次推理。
- 已有证据：
  - E1"按轨迹预测下一步再去探"：约 50% 的探测被 agent 用上；
  - E2"空隙页面覆盖式探索"：在线只有 5% 被用上，却带来了大部分回滚失败；
  - 深度 2 有害：第二跳回滚失败 40%。
- 你要回答：
  - 探索应该瞄准哪些信息：下一动作的结果？后继页面？可回滚性？
  - 每种信息分别通过哪一种系统侧机制帮助推理，以及帮助有多大？用任务 B、C 的数字。
  - 是否还有任何形式的 prompt 注入值得保留？没有就明确写"不注入"。

### Q3 图怎么设计？存什么？怎么用来增强推理和探索？
- 从 Q2 中"真正被用上的信息"反推图的设计：
  - 页面身份分几级：layout 用于跨变体匹配，struct/content 用于验证；
  - 动作身份怎么定义；
  - 续写计数怎么组织：(页面, 上下文) → 下一动作，包含所有动作类型，按同任务 / 跨任务模板分开统计；
  - 边上存哪些属性：后继分布、回滚方法、是否不可恢复。
- 图的两个用途：
  - **增强推理**：跳步门槛怎么定（用任务 B 的数字）、提交类动作怎么保证安全、验证用哪一级签名；
  - **指导探索**：先探什么、哪些控件不探（用任务 C 的数字）、空隙页面要不要放到离线或空闲时段补图。
- 最后用任务 A 的同时段对照，给出完整 AndroidWorld 上的成功率、步数和调用次数，作为整个设计的端到端结果。

现有设计稿 `TRV/reports/full_design_zh.md` 是起点。最终文档要在它的基础上，用 A/B/C 的新证据**确认、修改或删除**每一个模块，并写明理由。

## 背景（先读）
- 仓库：`/Users/huangrunxi/Projects/android_world`。实验目录：`experiments/trace_reachability_validation`（下称 TRV），底层运行时在 `experiments/runtime_graph_study/scripts`（`rgs_run.py`、`rgs_ui.py`）。
- 主模型：LabServer vLLM 上的 GELAB-ZERO-4B，通过本机 SSH 隧道 `http://127.0.0.1:18095/v1/chat/completions` 访问（`/v1/models` 返回里含 "GELAB" 即为可用）。不要启动或改用任何其他模型（例如 MAI-UI），也不要用 Claude 或其他 VLM 做探索。
- 设计和已有结果：`TRV/reports/full_design_zh.md`（含 FULL2、FULL3 修订）、`TRV/reports/full_results_zh.md`。
- **当前正在运行**：`python scripts/full_driver.py FULL3`（PID 30532，cwd 为 TRV），在 `emulator-5554` 上跑 116 个 AndroidWorld 任务 × 2 轮（seed 30/31）。它和已有的 `raw/full/BASE`（几天前、另一个 session 跑的）做配对比较。
- 已知问题：同一配置重复跑，成功率能相差 ±10/116；**只有同一时段跑的两组才有可比性**，跨天的差异多次被证明是噪声。所以 FULL3 和旧 BASE 比，可信度不够。

## 硬性约束（违反任何一条都会毁掉正在运行的实验）
1. **不要碰 `emulator-5554`**：不对它执行任何 adb 命令，也不 kill、重启或 snapshot 它。不要碰 adb 设备 `10.79.170.66:33219`。
2. 不要 kill、暂停或修改 FULL3 的任何进程（`full_driver.py`、`full_run.py`）。要停你自己的进程时，按你自己的输出路径匹配（`pgrep -f <你的输出目录>`），停之前先确认 PID。
3. **不要编辑 TRV/scripts 和 runtime_graph_study/scripts 下的任何已有文件。** FULL3 每个任务都会新起一个 Python 进程重新 import 这些文件，改一个字都会改变正在跑的实验。需要改代码时，先把整个目录复制到新位置，在副本里改（见任务 A）。
4. 不要写入以下路径：`TRV/raw/full/`、`TRV/graph_snapshots/full/`、`TRV/logs/full.log`、`TRV/logs/full_driver_full3.log`、`TRV/reports/execution_log.md`。你的日志单独写一个文件。
5. 不要用带变量的 `rm -rf`。需要"清空"时，换一个新目录名。
6. 不要 commit 或 push，做完告诉我改了哪些文件。
7. 不要在代码或日志里硬编码或打印任何凭据。

---

## 任务 A（最重要）：在第二个模拟器上跑同时段的 BASE 对照

**目的**：让 FULL3 有一个同时段的 BASE 对照（BASE_S），并顺便得到"同配置跨时段"的噪声估计（BASE_S 对比旧 BASE）。

1. **起第二个模拟器**
   - 可用的 AVD：`AndroidWorldAVD_2`、`AndroidWorldAvd`、`AndroidWorldShadow`（`~/Library/Android/sdk/emulator/emulator -list-avds`）。先确认哪个没在被 5554 使用：`adb -s emulator-5554 emu avd name`，这是唯一允许对 5554 执行的只读命令。
   - 选一个空闲的 AVD，用 `-port 5556` 启动，并加 `-grpc 8556` 之类的独立 gRPC 端口，避免和 5554 冲突。机器配置是 24 GB 内存、10 核，限制它的内存（例如 `-memory 4096`）。
   - 检查这个 AVD 是否已经完成 AndroidWorld 的初始化（应用都已安装、fast a11y 服务 APK 已安装）。如果没有，只对 5556 跑 `env_launcher.load_and_setup_env(console_port=5556, emulator_setup=True, ...)`。
   - 检查短信默认应用（SMS role）。已知它会悄悄漂移，导致所有 SimpleSms* 任务失败。确认它和 5554 上的设置一致（对比方式见 `android_world` 里的 SMS 相关任务初始化）。
2. **复制代码并参数化（只改副本）**
   - 复制到 `experiments/parallel_emu2/`，保持相对目录结构，让 `ROOT`、`REPO` 的推导仍然正确：
     - `experiments/parallel_emu2/runtime_graph_study/scripts/`
     - `experiments/parallel_emu2/trace_reachability_validation/{scripts,configs}/`
     - 检查 `REPO = Path(__file__).resolve().parents[3]` 这类写法在新位置仍然指向仓库根目录；如果不对，就只在副本里改。
   - 在副本里把设备参数化（环境变量，默认值保持原样）：
     - `rgs_run.py` 第 527 行 `console_port=5554`；
     - 第 953 行 `subprocess_forward` 里写死的 `emulator-5554` 和 `tcp:8765`；
     - `ANDROID_WORLD_FAST_A11Y_SOCKET_PORT`：第二台用 8766；
     - 再 `grep -rn "5554\|8765"`，把副本里所有写死的地方都找出来，包括 dump 包装里重新做 forward 的那段。
   - 副本中 `full_driver.py` 的日志写到副本自己的 `reports/execution_log_emu2.md` 和 `logs/`。确认 `ROOT` 指向副本，这样 raw 和 graph 都落在 `parallel_emu2` 下。
3. **冒烟**：在 5556 上跑 2 个 BASE 任务（例如 `ClockStopWatchRunning`、`MarkorCreateNote`，seed 30），确认以下几点：
   - 截图和 a11y 来自 5556：在副本里打印 serial 和端口；
   - 5554 上的 FULL3 日志在这期间继续正常推进，没有出错；
   - `result.json` 的字段与 `raw/full/BASE` 的一致。
4. **正式运行**：在副本中运行 `full_driver.py BASE_S`，arm 名用 `BASE_S`（`full_run.py` 中 `FULL_ARM=BASE_S` 必须走与 BASE 完全相同的纯 agent 路径。检查一下：任何不是 FULL*/G* 的 arm 是否都等同于 BASE？如果不是，在副本里把 BASE_S 映射到 BASE 的代码路径）。任务顺序和 seed 与 FULL3 相同（30/31）。用 `nohup` 后台运行，等待时只按日志内容判断是否完成，不要用会匹配到自身的 `pgrep` 循环。
5. **分析**：在副本中写 `compare_same_session.py`，读取 `TRV/raw/full/{BASE,FULL3}`（只读）和 `parallel_emu2/.../raw/full/BASE_S`，输出：
   - 每一轮的成功数；
   - 三组配对：BASE_S vs FULL3（主比较）、BASE vs FULL3、BASE vs BASE_S（噪声）。每组给出不一致对的数量和 McNemar 精确检验的 p 值；
   - 两边都成功的配对上的步数、调用次数、墙钟时间；
   - "被基准决定的任务"：BASE、BASE_S 两轮都成功或都失败的任务数，以及剩下的"易变任务"列表；
   - 结果写到 `parallel_emu2/analysis/same_session.json` 和一份中文简报 `parallel_emu2/reports/same_session_zh.md`。

**注意**：两台模拟器的速度可能不同。在简报里报告两边平均墙钟时间的差异。如果 5556 明显更慢（超过 1.5 倍），先告诉我，不要直接下结论。

---

## 任务 B（纯离线，不用模拟器）：跳步门槛的离线标定

**目的**：不在线跑，用已有轨迹回答两个问题：FULL3 的跳步门槛在完整 AndroidWorld 上**有多少触发机会**？**触发时有多准**？同时比较几种门槛变体，为下一版设计提供依据。

1. **数据**（只读）：`TRV/raw/full/{BASE,FULL}/<task>/p{0,1}/` 下的 `steps.jsonl`、`meta.json`、`result.json`。只用 agent 自己的步（`kind == "model"`），不用跳步、探测或回放的步。
2. **复现在线过程**：按 `full_driver.py` 的顺序（第 1 轮按任务字母序，然后第 2 轮）逐个 episode 回放。在每个 episode 开始前，store 里只包含**此前已经成功的** episode（与在线版本一致：成功才入库）。直接 import TRV 的 `seq_run.Store` 和 `full_run` 里的门槛逻辑（`plan_jump_invariant`），**只读 import，不改原文件**。如果需要改动，复制一份到 `experiments/parallel_offline/` 再改。
3. **分母不能有偏**：对每个 episode 的**每一步**（不能只统计 store 能匹配上的步）判断门槛是否触发，以及预测的动作 token 是否等于 agent 实际的下一动作；如果有下一屏的 layout 签名，也检查到达页面是否一致。
4. **比较的门槛变体**：
   - V1：FULL v1（多数票，n≥2、占比≥0.8）；
   - V3：FULL3（同任务门槛，或 ≥2 个模板一致 + agent 已表态 + 上下文阶数≥1）；
   - V3-same：只用同任务；
   - V3-cross：只用跨任务一致；
   - 门槛参数扫描：n∈{2,3,5}，占比∈{0.8,0.9,1.0}。
5. **输出**（全部按来源拆开：同任务 / 跨任务、导航 / 提交、第 1 轮 / 第 2 轮、按 app）：
   - 触发率 = 触发步数 / 全部 agent 步数；
   - 精度 = 预测正确 / 触发步数；
   - 理论可省调用 = 正确触发的步数（连续链按 4 跳上限计）占总调用的比例；
   - 错误触发的例子清单（task、step、预测 vs 实际），以及错误触发集中在哪些 app、哪些动作；
   - 在 BASE 成功的 episode 和失败的 episode 上分别统计。
   - 结果写到 `experiments/parallel_offline/analysis/gate_calibration.json` 和中文简报 `experiments/parallel_offline/reports/gate_calibration_zh.md`。简报开头先用一句话回答：FULL3 门槛的省调用上限是多少、精度多少、哪个变体的精度和覆盖权衡最好。

---

## 任务 C（可选，离线）：E1 探测回滚失败的归因

FULL v1 中有 123 次回滚失败后的重新请求，其中大部分来自 E2。FULL3 去掉了 E2，但 E1 仍会探测。

请从 `TRV/raw/full/FULL/*/p*/` 的探测记录中（字段名以 `steps.jsonl` 和 probe 相关 jsonl 为准，先看文件里有什么），**只看 E1 的探测**，统计：
- 回滚失败率；
- 按控件类型、app、回滚梯子在哪一级失败分别统计；
- 失败前后页面签名的变化类型。

目标是给出一个可以直接写进 deny-list 或"探测前判定"的具体规则，并估计它能挡掉多少比例的失败、会误伤多少本来有用的 E1 探测（E1 被 agent 用上的判定以 `m_e1_hit` 的逻辑为准）。写到 `experiments/parallel_offline/reports/e1_rollback_zh.md`。

---

## 交付
- **主交付**：`experiments/parallel_offline/reports/design_final_zh.md`，按上面的 Q1/Q2/Q3 组织。每个模块写明：动机（数字 + 出处）→ 技术设计 → 本轮证据是确认、修改还是删除它。
  - 末尾附一张表：完整 AndroidWorld 上 BASE_S vs FULL3 的成功率、步数、调用次数（同时段配对），以及噪声基线（BASE vs BASE_S）。
  - 再附一张"机制 → 省下的调用 / 带来的伤害"漏斗表，按来源拆开。
- 每个任务一份中文简报，数字都标明来源文件。
- 列出新建或修改的所有文件。
- 任务 A 运行期间，每完成一轮（116 个任务）汇报一次，并附上 FULL3 当时是否仍在正常推进（只读 `TRV/logs/full_driver_full3.log` 的最后几行）。

---

## 附：在 LabServer 上做 B/C 时的数据位置
- B/C 所需的记录已经随仓库同步：`experiments/trace_reachability_validation/data_sync/full_records/{BASE,FULL}/<task>/p{0,1}/`（json/jsonl，含页面签名和动作 token；不含截图和 a11y）。FULL v1 的最终全局图在 `data_sync/graph_snapshots_full/FULL/`。说明见 `data_sync/README.md`。
- 在服务器上，把上文的 `TRV/raw/full/{BASE,FULL}` 换成 `TRV/data_sync/full_records/{BASE,FULL}`。服务器上没有模拟器，所以任务 A 不在服务器上做。
- 服务器上的约束 1–4 不适用（那里没有 FULL3），但仍然**不要直接改原脚本**：改动放到 `experiments/parallel_offline/`，方便和本机合并。
