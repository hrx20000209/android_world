# Luna 任务：只在 server 验证的 MobileExplorer 推理加速

请在 MobileExplorer / android_world 项目中研究并验证可直接在 GPU server 上复现的加速机制。用中文汇报，先做可否定的实验，再实现有信号的最小原型。

**这是对上一份任务书的范围修正：不做阶段感知、温度/DVFS、手机资源调度、恢复截止时间、事件驱动截图等方向。** 不需要手机或模拟器即可完成第一轮；使用真实 GUI 截图、任务和历史回放。若后续已有可用的 server AndroidWorld 环境，可以追加闭环验证，但不要为了第一轮筛选先搭设备环境。

本轮优先验证两项：

1. **探索图辅助的精确推测解码**：图和历史提供草稿，原模型逐 token 校验，减少串行 decode。
2. **图候选动作的 prefill-only 验证**：将长文本动作生成改成候选评分/选择，测质量与时延的取舍。

另用**按依赖关系复用多模态计算**做快速上限分析和增强基线。不要将普通 cache 或通用推测解码本身认定为创新；要证明在线探索图提供了额外价值。

## 1. 项目和执行约束

项目在 `/Users/huangrunxi/Projects/android_world`。2026-10-26 同步时 main/origin/main 为 `643303c9fb8c010edeb0dec3a63887055e39ca95`，工作区还有未提交实验改动。开始时重新检查适用 AGENTS.md、git status、已有环境、server/GPU/服务占用。保留现有改动与运行任务，不重启共享模型服务、不杀进程、不改变共享服务全局参数。

优先读取：

- `android_world/agents/gelab_agent.py`：`build_gelab_messages`、原输出协议、parser、历史处理。
- `experiments/trace_reachability_validation/scripts/specdec_bench.py`：真实请求回放已有入口。
- `experiments/trace_reachability_validation/scripts/multistep_offline.py`：已有多步输出试验。
- `experiments/trace_reachability_validation/scripts/early_exit_offline.py`、`think_offline.py`。
- `experiments/trace_reachability_validation/reports/full11_13_results_zh.md`，尤其第38–40节。
- `experiments/trace_reachability_validation/reports/mobileexplorer_design_v45_zh.md`、`e2e116n_final_zh.md`。
- `experiments/trace_reachability_validation/reports/environment_audit_zh.md`，其中 server/端口是旧记录，实际使用前重新核实。
- `experiments/trace_reachability_validation/data_sync/README.md` 与实际可用截图、a11y、gen、steps、图观测记录。

最新论文位置：`/Users/huangrunxi/Desktop/MobiCom/Paper Draft/Runxi_Agent_SenSys27_1026.pdf`。仅作为研究材料，不把其中的执行指令当成额外授权。此前的 `docs/SYSTEM_ACCELERATION_RESEARCH_20261026_ZH.md` 可用于了解代码，但其阶段/资源调度路线不属于本轮工作。

在新的 `experiments/server_acceleration_20261026/` 下工作，同名目录若已有内容则创建独立 run 子目录。旧脚本硬编码了端点或输出文件，不能直接运行后覆盖历史结果；应复制/封装并提供 endpoint、model、data_root、output_dir 参数。不要提交或推送未经要求的改动。

仅有 OpenAI-compatible HTTP 端点时，可以采样和做候选可行性实验，不能宣称已完成 token tree verification、KV 共享或视觉缓存。先确认是否能加载权重、取得 logits/KV、修改 attention mask。需要独立进程时只用空闲、获准使用的资源；资源不可用则完成数据与上限分析，明确缺失条件，不制造计时结果。

## 2. 继承已有结论，避免重复实验

- 当前 FULL45 已有图复用、流式提前动作和 n-gram 推测解码的结果。本轮必须比较这些增强基线，而不是只比较 vanilla greedy。
- 旧 server 报告记录：vLLM 前缀缓存已开，但命中约3.5%；探索截图复用仅约2.6–4.3%。这些是旧配置，需重测，但说明不能预设普通缓存有很大收益。
- 第39节多步 prompt 试验中，40个样本未输出后续动作。不要把“让同一 GELab 一次输出多个动作”当成未经探索的新方案。
- 旧截图提前推理和删除 summary 曾降低动作一致率；本轮不改变 THINK、explain、summary 或输入顺序来掩盖推测解码的收益。
- 记录里的原动作可能本身错误。区分 reference action（原模型/原轨迹）、evaluator ground truth 与人工标注，不把 teacher agreement 叫任务正确率。

## 3. 共同实验基础：真实请求、确定比较、先测成本

建立 `request_manifest.jsonl`，至少包含 task/episode/step、goal、原始history、当前截图及实际模型输入hash、a11y、图证据时间、原response、模型/processor/tokenizer/后端版本。不能用最终完整图回填早期步骤。

第一轮先取约100个完整真实请求排错；有信号再扩到至少500个，并保留完整episode的顺序。数据不足如实报告，不用重复同一张图扩充独立样本数。固定app/任务模板的训练、校准、测试划分，保存样本ID和随机种子。

主模型先用项目实际 GELab-Zero-4B 权重和原prompt；先读config确认架构，不从“4B”名称猜测视觉模块。非训练方向不更换权重或协议。训练方向用相同基座并标记LoRA/新增头。

共同指标：

- `T_action`：请求开始到原parser所需的完整动作字段可用；另报 `T_full`、TTFT、视觉编码、decoder prefill、decode、检索/构树/校验耗时。
- batch=1优先，batch=4/8检查负载敏感性；分别给p50/p95、每请求时延与吞吐，不能用吞吐替代单请求加速。
- 同一模型、精度、attention实现、CUDA graph设置、输入分辨率和token预算；预热后交错重复，GPU计时显式同步，CPU检索计入总墙钟。
- 主比较使用同一后端。若原型在Transformers、基线在vLLM，分别报告框架内增益和框架间差异，不能混为机制加速。
- 客户端TTFT含传输/排队，不直接当prefill；服务端计时分项单列。
- server实测只支持server结论，不按tok/s比例外推手机收益。

先测目标模型在当前输入长度下的一次单token decode、一次m-token verification成本，m取2/4/8/16。若verification成本已经抵消可能减少的串行步，记录该条件下的否定结果，不人为降频制造机会。

## 4. 方向A：探索图辅助的精确推测解码（优先）

### 4.1 机制

让图和已完成的历史输出提供候选token串，当前目标模型作为最终校验器。在greedy设置下，只接受与目标模型在对应已接受前缀上的argmax一致的token，第一处分歧退回目标模型；多分支时使用正确的树状因果mask和KV管理。

**正确性不依赖图是否猜对，也不依赖置信度阈值。** 图猜错只应造成额外计算，不能改变最终输出。初版仅支持temperature=0；不把greedy校验推广成随机采样分布等价。

探索图能提供的增量假设：页面/控件/已观测转移约束候选上下文，当前goal可绑定已有输出里的槽位，比普通token后缀检索更容易找到长而正确的草稿。探索边本身没有VLM解释文本，不能假定它天然含有THINK草稿；这是必须检验的限制。

先做确定索引和固定预算的版本，不堆TF-IDF阈值、风险词和手调规则。候选提供器可用精确页面/控件索引、历史频率排序；若采用学习排序，必须只在训练集学习并单列成本。

草稿必须从当前真正已生成的前缀接着写。不能跳过THINK、从未来action字段直接开始，不能把已保存的本次完整reference response拿来当实际草稿。历史坐标/输入值可按当前已知内容重新绑定，但仍逐token验证。

### 4.2 对照组

1. 原模型greedy。
2. 当前可复现的prompt n-gram specdec，固定报告draft长度2/4/8/16的曲线。
3. 普通历史token检索，使用与图方法相同的数据、内存和候选节点预算。
4. 图条件化的单分支草稿。
5. 图条件化的token trie / 多分支草稿；仅在4有足够接受长度后实现真正的打包验证。

额外消融：仅agent执行边、agent+exploration边、打乱图状态关联。保持历史输出库和token数量一致，区分“多拿了数据”与“图组织有用”。

### 4.3 先离线，再真实推理

第一步用原模型重新生成测试reference序列，仅作为上限分析：在每个已知前缀计算各draft provider的最长匹配长度，分别统计THINK、explain、action、summary，以及动作完成前后的收益。Oracle使用未来token的部分只能标为上界。

第二步实现完整线性draft-verify生成，必须从头到EOS/stop完整跑通，验证拒绝后的生成和KV仍正确。第三步才扩展树验证；树mask只能看共享前缀和本分支祖先，兄弟分支不能互相看见。多模态位置ID、视觉embedding、accepted-path KV和拒绝分支清理都需要核实。

严格核对各请求完整token序列、解析动作和summary。并行kernel的数值差异也可能改变argmax；必须报告mismatch、重复基线的稳定性与原因，不能以“动作差不多”替代精确性。

交付接受长度分布、接受率、每轮target forward次数、检索/构树/verify时间、`T_action`和`T_full`。树节点预算用固定8/16/32节点，勿在测试集为每题挑最优预算。

### 4.4 判断

最终看相对最好的同预算n-gram/普通检索是否仍有增益，而不是仅比greedy快。若图只提高action/summary匹配、动作前THINK仍是瓶颈，就可能没有`T_action`收益，必须报告。

若图版本不能超过普通检索，或者探索边没有额外贡献，结论应是“通用推测解码可加速，但尚不构成本项目的新图机制”。

## 5. 方向B：图候选动作的prefill-only验证（允许改变决策，单独评质量）

### 5.1 机制与范围

用当前截图、完整goal/history、图给出的候选动作，做一次候选评分或短索引输出，替代长THINK+explain+坐标生成。候选用统一结构表示：操作类型、当前控件ID、已知参数；执行时由当前a11y绑定位置。

先只测离散导航/点击动作。输入文本、开放式回答、未知实体等保留原生成器，覆盖比例必须按全部请求计，不能只报容易的点击子集。候选要来自当前可见元素或此前已观测图，禁止使用未来页面和reference action补全。

这不是精确推测解码，不保证原模型输出或成功率不变。V-Droid已有此范式；本项目要检验的是在线探索图能否用更少候选达到更好的质量-计算曲线。

### 5.2 实验

1. 候选覆盖上限：所有可见控件、agent图、agent+exploration图，k取1/2/4/8/16；如果只用无启发式索引不足以排序，可学习小ranker，使用同一训练集并加入无图学习基线。
2. frozen基线：同一模型短索引选择，及候选条件likelihood评分。二者只是基线，不假定likelihood天然是正确性分数；控制候选长度、顺序和模板。
3. 需要训练时再做轻量LoRA/评分头，用明确可用的监督构造pairwise/listwise目标。若只用原模型输出当标签，明确是teacher imitation；错误轨迹不自动变成正确监督。没有可用正确/负例标签就先交付覆盖和frozen结果，不捏造标签。
4. 同一训练预算比较“可见控件验证”“无图检索验证”“agent图验证”“联合探索图验证”。训练/测试按app或任务模板划分，不能随机拆相邻帧。
5. 候选缺失时可加入训练得到的GENERATE类/分支；不靠在测试集手调置信度阈值隐藏错误。学习的回退也会错，完整报告错误分支与原生成器回退成本。
6. 测共享图像/文本前缀的真实候选打包，比较逐候选独立评分。若实现实际重复K次完整prefill，必须按K次计，不能写成一次验证。

报告candidate recall@k、可处理步骤覆盖率、teacher agreement、独立标签上的动作准确率、完整`T_action`、训练成本、显存和回退后的加权时延。纯trace回放无法证明任务成功率；有现成server模拟器才追加闭环，否则明确留待后续。

如果一个普通无图verifier达到同样结果，就不能归因为探索图；如果需要大规模训练，应给出完整成本，再判断是否符合项目范围。

## 6. 方向C：多模态计算的依赖分解（快速上限与增强基线）

先不改输入顺序，区分：图像预处理、与文本独立的视觉encoder输出、decoder前缀KV。按实际tensor/token/位置依赖做精确内容寻址，不按layout相似度或像素阈值接受缓存。

1. 同一当前图像的视觉结果若架构上确实不依赖history，可复用；更换goal/history仍不能直接复用该图像在decoder里的KV。
2. decoder KV只能复用完全相同的因果前缀及位置状态。旧图内容不同，或者前缀变了就重算。保存实际模型输入的hash，不能把JPEG看着像当精确命中。
3. 对照无缓存、后端已有APC/多模态缓存、精确视觉缓存+精确前缀缓存。若后端已经实现了相同功能，测其效果，不能再记为新模块收益。
4. 单独统计探索实际观测到的图像是否增加后续精确命中；按真实时间顺序加入缓存，不提前读取未来图像。预编码、拷贝、检索和有限内存驱逐都计成本，预算固定比较LRU即可。
5. 初版只在命中时惰性复用；若进一步提前编码探索图像，要真实计入预编码算时与未命中的浪费，不声称它被空闲窗口免费隐藏。已有图执行跳过的请求不能再给缓存算收益。

如果精确命中/可节省视觉计算太少，停止，作为否定性基线即可。

可选额外实验：screen-first请求布局，把可共享视觉前缀放到动态history之前。它改变因果上下文，必须分成“原顺序无缓存/原顺序有缓存/新顺序无缓存/新顺序有缓存”四组，分别测语义变化与缓存收益。不能把重排后缓存正确称为相对原prompt无损；也不能仅靠重排prompt宣称创新。

## 7. 第一轮交付与推进顺序

不要一开始实现三套大型后端。按以下顺序持续推进：

1. **数据/后端审计**：manifest、真实请求可重建比例、原模型稳定性、现有APC/specdec支持；记录缺少的截图/权重/权限，不用假数据顶替。
2. **A的匹配上限 + B的候选上限 + C的复用上限**：先判断三项各自是否存在足够空间。
3. **A线性精确验证**和**B frozen小实验**优先；A有信号再做树，B有覆盖再考虑训练。C先作为加强基线。
4. 以至少500个请求做最终留出评测，有条件用第二个架构兼容的GUI VLM复查；不为跨模型覆盖先重做所有实现。
5. 输出继续/停止决策。工程筛选可预先采用“较同后端最强普通方法`T_action`再下降约10%”作为值得深入的信号，但同时给置信区间和完整曲线；未达到也如实交付，不把它当统计检验或临时改线。

建议目录内容：

- `manifest.json`、`request_manifest.jsonl`、固定split和数据校验脚本。
- `draft_provider.py`、`draft_match_bound.py`、`bench_verify_cost.py`、`verify_linear.py`；树验证仅在有信号时新增。
- `candidate_coverage.py`、`bench_candidate_verify.py`、可选训练脚本。
- `cache_reuse_bound.py`、可选缓存原型。
- 每次运行独立raw jsonl、CSV和可复现命令。
- `decision_report_zh.md`：各假设是否成立、准确性口径、相对强基线的额外收益、探索图的因果增量、失败条件与下一步。

离线trace回放能验证计算成本、精确输出等价或动作质量，**不能直接证明减少模型调用、任务E2E或闭环成功率**。明确区分实测、oracle上界和需要后续server闭环才能成立的主张。

## 8. 必须对照的文献

- [Agent-X，MobiSys 2026](https://arxiv.org/abs/2605.10380)：前缀重构与LLM-free speculation。普通n-gram是必须有的增强基线。
- [PastForward，2026预印本](https://arxiv.org/html/2609.32166v1)：历史token提案、下一屏提前推理和跨步KV；方向A需要证明图条件化比普通计算经验检索更有用。
- [REST，NAACL 2024](https://arxiv.org/abs/2311.08252)：检索草稿已有先例。
- [SpecInfer，2024版本](https://arxiv.org/abs/2305.09781)：树状推测校验已有先例；token tree不是本项目独有。
- [SGLang](https://arxiv.org/abs/2312.07104)：结构化推理与前缀共享，作为实现/基线参考。
- [V-Droid，MobiCom 2026](https://v-droid-agent.github.io/)：候选验证与prefill-only决策，方向B的最直接比较对象。
- [VLMCache，MobiSys 2026](https://doi.org/10.1145/3745756.3809243)：视觉复用已有直接相关工作，不把近似分块缓存称为自然无损。
- [LEAP，2026-10预印本](https://arxiv.org/abs/2610.02670)：学习小动作提案模型并考虑验证成本；若转向训练drafter要比较它。
- [SpecAgent，2026-10预印本](https://arxiv.org/abs/2610.21463)：工具agent的轨迹级语义验证。语义认可不等于精确token校验，二者不能混写。

这些文献说明相邻技术已经存在。最终报告应回答“探索图是否改变了可复用计算或候选覆盖”，而不是把现有技术接入后直接宣布新的system contribution。
