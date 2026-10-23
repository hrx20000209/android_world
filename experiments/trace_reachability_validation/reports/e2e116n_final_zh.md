# e2e116n 最终报告（BASE vs FULL45，Orin vLLM n-gram8 + CUDA graph，模拟器内 Qwen2-VL-2B 背景负载）

116 题全部配对（FULL45 两条出错 episode 已重跑：ExpenseAddSingle 成功 3 次推理；SystemCopyToClipboard 失败）。无 Orin 拥塞标记。

| 口径 | BASE 成功 / 推理 / e2e | FULL45 成功 / 推理 / e2e |
|---|---|---|
| 全部 116 | 53 / 10.14 / 151.9 s | 54 / 9.16 / 135.2 s (−11.0%) |
| 成功任务 | 53 / 8.06 / 130.2 s | 54 / 6.69 / 107.7 s (−17.3%) |
| 共同成功 47 | 7.98 / 129.3 s | 6.34 / 102.0 s (−21.1%) |

与上一轮（原 Orin 配置，FULL44）对比：BASE 194.5→151.9 s（e2e/推理 18.55→14.98 s，推理数相同的 75 题 182.8→146.0 s，−20%，即 n-gram+CUDA graph 收益）；FULL44→FULL45 成功 49→54，163.2→135.2 s。

不一致任务 13 题：FULL45 赢 7（AudioRecorderRecordAudio, ClockStopWatchPausedVerify, ExpenseAddMultiple, OsmAndFavorite, RecipeAddMultipleRecipes, SimpleCalendarAnyEventsOnDate, SimpleDrawProCreateDrawing），输 6（AudioRecorderRecordAudioWithFileName, NotesRecipeIngredientCount, RetroCreatePlaylist, SimpleCalendarFirstEventAfterStartTime, SystemBluetoothTurnOff, TurnOffWifiAndTurnOnBluetooth）。

System 设置页问题：探索点到“内含开关的整行”(Bluetooth / Connection preferences) → in_place_content_change 回滚失败；Network & internet 探索回滚失败；开关连点疑似截图滞后。另：模型输出中文 app 名(剪贴板) → adb monkey 异常导致 episode 崩溃；流式 summary 偶尔被写成 action 字典。

机制：FULL45 early_actions 935/1062, func_steps 87, form_steps 67, probes 163, requery 11, bootstrap 76, stale_pixels 242, jump_commit 13。生成：ttft 4.65 s, decode 17.1 tok/s（BASE 17.7）, 平均 114 tokens, THINK 58%。
