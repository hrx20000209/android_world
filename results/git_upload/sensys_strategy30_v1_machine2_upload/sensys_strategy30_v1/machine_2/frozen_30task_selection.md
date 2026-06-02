# Frozen 30-task strategy selection

- benchmark: AndroidWorld
- split: variant-sharded; every assigned variant runs the same frozen 30 tasks
- variants: B0/S1/S2/S3/S4/S5 budget12
- depth: 2
- per-step exploration budget: 12 safe candidates
- active t+2 shortcut: disabled; shadow traces only
- note: the `shard` column below is only a task bucket label, not the machine split

| task_id | task | app | task_mode | shard |
| ---: | --- | --- | --- | --- |
| 1 | `NotesRecipeIngredientCount` | Joplin | answer_entity | A |
| 2 | `NotesIsTodo` | Joplin | answer_boolean | A |
| 3 | `FilesDeleteFile` | Files | delete_boundary | A |
| 4 | `FilesMoveFile` | Files | navigation_mutation | A |
| 5 | `SportsTrackerActivityDuration` | OpenTracks | answer_stats | A |
| 6 | `SportsTrackerTotalDistanceForCategoryOverInterval` | OpenTracks | answer_stats | A |
| 7 | `SimpleCalendarNextMeetingWithPerson` | Calendar | answer_entity | A |
| 8 | `SimpleCalendarEventsInTimeRange` | Calendar | answer_entity | A |
| 9 | `ClockStopWatchRunning` | Clock | state_verify | A |
| 10 | `ClockTimerEntry` | Clock | state_verify | A |
| 11 | `MarkorCreateNote` | Markor | create_note | A |
| 12 | `MarkorEditNote` | Markor | edit_note | A |
| 13 | `MarkorCreateFolder` | Markor | create_folder | A |
| 14 | `ExpenseDeleteSingle` | Expense | delete_boundary | A |
| 15 | `ExpenseDeleteMultiple2` | Expense | delete_boundary | A |
| 16 | `ExpenseAddSingle` | Expense | create_record | B |
| 17 | `BrowserMaze` | Browser | web_interaction | B |
| 18 | `BrowserMultiply` | Browser | web_interaction | B |
| 19 | `BrowserDraw` | Browser | web_interaction | B |
| 20 | `ContactsNewContactDraft` | Contacts | draft_data_entry | B |
| 21 | `ContactsAddContact` | Contacts | data_entry | B |
| 22 | `AudioRecorderRecordAudio` | AudioRecorder | record_audio | B |
| 23 | `AudioRecorderRecordAudioWithFileName` | AudioRecorder | record_audio | B |
| 24 | `SystemWifiTurnOnVerify` | System | state_verify | B |
| 25 | `SystemWifiTurnOffVerify` | System | state_verify | B |
| 26 | `SystemBrightnessMinVerify` | System | state_verify | B |
| 27 | `SimpleDrawProCreateDrawing` | SimpleDraw | create_drawing | B |
| 28 | `CameraTakePhoto` | Camera | capture_photo | B |
| 29 | `OsmAndFavorite` | OsmAnd | map_marker | B |
| 30 | `OsmAndMarker` | OsmAnd | map_marker | B |
