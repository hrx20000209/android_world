# 每个 app 第一道题：是否有跨 app 的泛化（2026-11-02）

数据：`success_by_task_3orders.csv`（三种顺序的逐题成功），`reports/e2e116n_runs.jsonl`（默认顺序的时延）。
random_apps 按 order_microbench_REPORT 的“app 内倒序”处理：每个 app 的第一道题 = 默认顺序里该 app 的最后一道。
random_tasks 的题目顺序不在仓库里，无法确定其中每个 app 的第一道题；P/V 两轮的逐题时延也不在仓库里。

## 汇总

| | 默认顺序 | random_apps |
|---|---|---|
| 第一道题成功（基线 / MobileExplorer） | 12/22 vs 12/22 | 7/22 vs 4/22 |
| 其余题成功 | 41/94 vs 42/94 | 45/94 vs 47/94 |
| 第一道题 E2E（全部 22 道） | 134.0 s vs 134.4 s（1.00×） | 无数据 |
| 第一道题 E2E（两边都成功，9 道） | 110.4 s vs 98.8 s（1.12×，95% CI [0.98, 1.25]） | 报告：1.09×（4 道） |

- 两种顺序合并：第一道题只有基线成功 6 道、只有 MobileExplorer 成功 3 道，符号检验 p = 0.51。
- 噪声参照：基线没有记忆、与顺序无关，但同一道题在任意两轮之间有 13–20 道（116 道中）结果不同。
- 同一道题作为第一道 vs. 作为后面的题（19 个多题 app）：MobileExplorer − 基线的成功数，作为第一道时为 −1 / −3，作为后面的题时为 +1 / 0。

## 逐 app

| app | 题数 | 默认顺序第一道 | 成功（基线/ME） | E2E 基线 → ME | random_apps 第一道 | 成功（基线/ME） |
|---|---|---|---|---|---|---|
| AudioRecorder | 2 | AudioRecorderRecordAudioWithFileName | ✓ / ✗ | 176 → 280 s | AudioRecorderRecordAudio | ✗ / ✗ |
| Browser | 3 | BrowserDraw | ✗ / ✗ | 181 → 96 s | BrowserMultiply | ✗ / ✗ |
| Camera | 2 | CameraTakeVideo | ✗ / ✗ | 181 → 165 s | CameraTakePhoto | ✗ / ✗ |
| Clock | 3 | ClockStopWatchPausedVerify | ✗ / ✓ | 22 → 154 s | ClockTimerEntry | ✗ / ✗ |
| Contacts | 2 | ContactsAddContact | ✓ / ✓ | 161 → 111 s | ContactsNewContactDraft | ✓ / ✓ |
| Expense | 9 | ExpenseDeleteSingle | ✓ / ✓ | 90 → 82 s | ExpenseAddMultipleFromGallery | ✗ / ✗ |
| Files | 2 | FilesDeleteFile | ✓ / ✓ | 171 → 154 s | FilesMoveFile | ✗ / ✗ |
| Markor | 14 | MarkorCreateFolder | ✓ / ✓ | 105 → 74 s | MarkorMergeNotes | ✗ / ✗ |
| Notes | 4 | NotesTodoItemCount | ✓ / ✓ | 69 → 78 s | NotesRecipeIngredientCount | ✓ / ✗ |
| OpenApp | 1 | OpenAppTaskEval | ✓ / ✓ | 35 → 56 s | （同左） | ✓ / ✓ |
| OsmAnd | 3 | OsmAndFavorite | ✗ / ✓ | 238 → 169 s | OsmAndTrack | ✗ / ✗ |
| Recipe | 13 | RecipeDeleteMultipleRecipesWithConstraint | ✗ / ✗ | 62 → 59 s | RecipeDeleteMultipleRecipesWithNoise | ✗ / ✗ |
| Retro | 4 | RetroCreatePlaylist | ✓ / ✗ | 221 → 159 s | RetroSavePlaylist | ✗ / ✗ |
| SaveCopy | 1 | SaveCopyOfReceiptTaskEval | ✓ / ✓ | 129 → 117 s | （同左） | ✓ / ✓ |
| SimpleCalendar | 17 | SimpleCalendarEventsInNextWeek | ✗ / ✗ | 146 → 144 s | SimpleCalendarAddRepeatingEvent | ✓ / ✗ |
| SimpleDraw | 1 | SimpleDrawProCreateDrawing | ✗ / ✓ | 227 → 180 s | （同左） | ✗ / ✗ |
| SimpleSms | 6 | SimpleSmsResend | ✓ / ✓ | 179 → 154 s | SimpleSmsSendReceivedAddress | ✗ / ✗ |
| Sports | 6 | SportsTrackerActivitiesCountForWeek | ✗ / ✗ | 28 → 32 s | SportsTrackerTotalDistanceForCategoryOverInterval | ✗ / ✗ |
| System | 13 | SystemWifiTurnOnVerify | ✓ / ✓ | 56 → 62 s | SystemCopyToClipboard | ✗ / ✗ |
| Tasks | 6 | TasksDueNextWeek | ✗ / ✗ | 36 → 31 s | TasksHighPriorityTasksDueOnDate | ✓ / ✗ |
| Turn | 2 | TurnOffWifiAndTurnOnBluetooth | ✓ / ✗ | 173 → 297 s | TurnOnWifiAndOpenApp | ✓ / ✓ |
| Vlc | 2 | VlcCreatePlaylist | ✗ / ✗ | 263 → 300 s | VlcCreateTwoPlaylists | ✗ / ✗ |

## 结论

- 第一道题上没有看到跨 app 的收益：成功率持平或略低（差异在噪声范围内），全部第一道题的时延 1.00×，两边都成功的子集 1.09–1.12× 且置信区间包含 1。
- 收益从同一 app 的第 2 道题开始出现（加速比 1.19× → 1.24×），与图按 (app, activity, layout) 组织、复用主要发生在同一 app 内的设计一致。
- 论文中不宜声称跨 app 泛化；可以写“新 app 冷启动无需准备、第一道题不受明显影响，收益随同一 app 的任务累积而增长”。
- 若要检验跨 app 迁移：需要 P/V 两轮的逐题时延与题目顺序；统计第一道题上图执行步骤所用边的来源 app；或做消融（每个 app 开始前清空其他 app 的知识）。
