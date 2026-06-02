# Frozen 30-task strategy selection

- benchmark: AndroidWorld
- split: variant-sharded; each assigned strategy runs the full 30-task set
- note: task shard A/B below is only a task bucket label, not the machine split
- variants: B0_BASELINE_RERUN, S1_BFS_BUDGET12, S2_DFS_BUDGET12, S3_BEAM_BUDGET12, S4_MCTS_BUDGET12, S5_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12
- budget: 12 safe candidates per exploration step, depth 2
- selection rule: app-stratified, includes answer/query, mutation, verification, web, recording, drawing, and map tasks

| task_id | task | app | task_mode | shard |
| ---: | --- | --- | --- | --- |
| 1 | NotesRecipeIngredientCount | Joplin | answer_entity | A |
| 2 | NotesIsTodo | Joplin | answer_boolean | A |
| 3 | FilesDeleteFile | Files | delete_boundary | A |
| 4 | FilesMoveFile | Files | navigation_mutation | A |
| 5 | SportsTrackerActivityDuration | OpenTracks | answer_stats | A |
| 6 | SportsTrackerTotalDistanceForCategoryOverInterval | OpenTracks | answer_stats | A |
| 7 | SimpleCalendarNextMeetingWithPerson | Calendar | answer_entity | A |
| 8 | SimpleCalendarEventsInTimeRange | Calendar | answer_entity | A |
| 9 | ClockStopWatchRunning | Clock | state_verify | A |
| 10 | ClockTimerEntry | Clock | state_verify | A |
| 11 | MarkorCreateNote | Markor | create_note | A |
| 12 | MarkorEditNote | Markor | edit_note | A |
| 13 | MarkorCreateFolder | Markor | create_folder | A |
| 14 | ExpenseDeleteSingle | Expense | delete_boundary | A |
| 15 | ExpenseDeleteMultiple2 | Expense | delete_boundary | A |
| 16 | ExpenseAddSingle | Expense | create_record | B |
| 17 | BrowserMaze | Browser | web_interaction | B |
| 18 | BrowserMultiply | Browser | web_interaction | B |
| 19 | BrowserDraw | Browser | web_interaction | B |
| 20 | ContactsNewContactDraft | Contacts | draft_data_entry | B |
| 21 | ContactsAddContact | Contacts | data_entry | B |
| 22 | AudioRecorderRecordAudio | AudioRecorder | record_audio | B |
| 23 | AudioRecorderRecordAudioWithFileName | AudioRecorder | record_audio | B |
| 24 | SystemWifiTurnOnVerify | System | state_verify | B |
| 25 | SystemWifiTurnOffVerify | System | state_verify | B |
| 26 | SystemBrightnessMinVerify | System | state_verify | B |
| 27 | SimpleDrawProCreateDrawing | SimpleDraw | create_drawing | B |
| 28 | CameraTakePhoto | Camera | capture_photo | B |
| 29 | OsmAndFavorite | OsmAnd | map_marker | B |
| 30 | OsmAndMarker | OsmAnd | map_marker | B |
