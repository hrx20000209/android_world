# Transition-validation pilot protocol — 2026-10-13

> **Historical pilot record:** this file preserves the earlier protocol and diagnostic run. Its task/probe outcome counts are not the canonical full-profile totals. Use `final_design_validation_zh.md` and `interim_progress_and_design_20261013_zh.md` for the latest consolidated results and definitions.

**Status:** complete; all 40 frozen tasks ran on GELAB-ZERO-4B via vLLM 8095/8096. These results are an engineering pilot and must not be merged with MAI-UI 2B.

## Cohort

Use the frozen 40-task metadata cohort at:

`experiments/info_value_profile_20261012/cohort40_information_intervention_pre_registration_20261012.json` (private LabServer data; not included in this branch)

It has 20 historical baseline successes and 20 failures, 18 apps, and low/middle/high source-complexity bands. Historical outcome is a stratum, not the pilot outcome. Keep task ID, seed 34, task parameters, and official evaluator fixed. Never replace a slow or failing task after observing results.

## Pilot mode

- One probe maximum per eligible inference window; max depth 1; existing safe-control filtering and recovery; no VLM/LLM call by Explorer.
- Preserve the original GELAB decision policy. Disable graph prompt injection, skip/replay, and action substitution in this first validation pass.
- Record the probe trace and `transition_verification.jsonl` after the model's actual action lands. Keep successful, failed, unmatched, and new-branch events; do not drop negative cases.
- First run one task per worker on separate emulator containers and separate vLLM endpoints. Measure wall time and throughput before expanding to the whole 40-task cohort. Then adjust only concurrency/batching to measured available time; do not change the frozen cohort or per-task treatment.
- Tasks selected for the two-worker throughput check: `MarkorDeleteNote` (historical success) and `FilesMoveFile` (historical failure), from different apps.

## Primary readout

1. Probe and recovery: attempts, completed probes, rollback success, latency, deadline/preemption, UI dirtiness.
2. State/action alignment: stable action-slot match between probe and authoritative action; distinguish unmatchable actions from true destination mismatch.
3. Outcome validation: exact runtime destination match, known alternative, same-family new runtime branch, new family branch, unmatchable. Exact structural hashes are treated as strict evidence and layout signatures as family labels; report false-split/false-merge review separately.
4. Cost and task outcome: wall time, inference latency/tokens, probe/recovery time, regular actions, official evaluator result. Do not attribute task-success improvement to validation because this arm does not change actions or prompts.

## Gate to reasoning experiment

After the pilot, count decisions that actually have a prior verified transition for the current state and stable action slot before the prompt is built. If same-source evidence is rarely available because the UI has advanced, do not claim prompt enhancement. Use only the eligible, leakage-audited decision-state pairs for a separate no-evidence vs fresh-transition-context test, with stale destination as a negative control. Keep model, screenshot/history, decoding, task/seed, and token budget paired.

Scale to all 116 tasks after the 40-task pilot reports runtime feasibility and transition-match quality. Re-run the primary claim on MAI-UI 2B when available; 4B is screening only.


# Preflight results — 2026-10-13

These are engineering observations before the frozen 40-task pilot. GELAB-ZERO-4B is the only model used in these runs; none is a gold trajectory or a MAI-UI 2B result.

| Cohort task | Baseline stratum | Inference windows | Probe/recovery | Main action join | Task evaluator |
|---|---|---:|---|---|---|
| MarkorDeleteNote | historical success | 7 | 0 probe; safe candidate absent on 3 windows | no probe edges | not used for design claim |
| SimpleCalendarAddOneEvent | historical success | 19 | 0 probe; safe candidate absent on 18 windows | no probe edges | smoke stopped after repeated no-candidate windows |
| FilesMoveFile retry4 | historical failure | 3 windows / 10 main actions | 2 SCROLL probes, 2/2 recovered | old coordinate binding yielded 0/10 matchable actions | failed |
| FilesMoveFile retry5 | historical failure | 3 windows / 7 main actions | 2 SCROLL probes, 2/2 recovered; one changed layout, one no-op | after binding fix 6/7 matchable; 0/7 had a matching verified probe edge | failed |
| FilesMoveFile alignment diagnostic max2 | historical failure | 3 windows / 7 main actions | despite budget 2, only 2 SCROLL probes; 2/2 recovered | 6/7 actions matchable; 0/7 verified-edge comparisons; one swipe unmatchable on its source screen | failed |

Separate from the cohort and from all GELAB pilot results, a manually selected Settings navigation candidate was replayed on the same seed-34 runtime instance: click `Network & internet` at (380,860), recover via Navigate up, then repeat the same click. Stable source layout matched before/after recovery (`43acda25ee7a8c52f97`) and the probe/replay successors matched (`319d5259ca0cb97a9ac0`), a family/slot-level `KNOWN_OUTCOME`. The full structural source hash changed because a dynamic Storage percentage changed from 47% to 48%; A11y also exposed no clickable parent for the row, so this is runtime-coordinate-only evidence. It validates the experiment's causal loop in a controlled manual case, not the online selector/verifier integration or cross-instance control relocation. `model_calls=0`; do not pool it with the model pilot or claim task success.

The action-binding bug was that the verifier used the first A11y node enclosing a coordinate, often the screen-wide FrameLayout. It now chooses the smallest compatible clickable/editable target; a unique scrollable container may be bound for exact-instance verification. The probe trace now records the fixed upward swipe direction. Ambiguous targets remain unmatched. Action-binding plus online-evidence/runtime tests: 15 passed.

Retry5 confirmed that coordinate-to-control binding is necessary but not sufficient: the randomized probes were SCROLLs while the agent's visible actions were mostly clicks, so no source/action edge could be compared. The max2 mechanism diagnostic shows that increasing per-window budget does not fix alignment: the safe-navigation gate still yields only SCROLL probes and no comparison. The next smoke must validate an actual candidate action slot, under strict reversibility and recovery checks; it is not a further random-probe budget sweep. This diagnostic deviation is not part of the frozen 40-task arm and will not be pooled with it.

A separate direct-action harness has been exercised without VLM calls (model_calls=0). A manual FilesMoveFile attempt has 7 actions but no completed evaluator pass; the target file was not visible after opening Movies, so validated_reference=false and it is excluded from gold references. Two other manual references did pass evaluator: `TurnOffWifiAndTurnOnBluetooth` (10 actions, score 1.0, validated reference) and `SystemBrightnessMinVerify` (5 actions, score 1.0, validated reference but initial state already satisfied the goal, therefore pre-satisfied and not evidence of a state-changing path). Only evaluator-validated, manually selected trajectories count as expert traces; report pre-satisfied references separately.

## Gate before the 40-task run

The manual replay smoke confirmed that a family/slot transition could be observed, recovered, and independently reproduced on a real UI. That justified the frozen 40-task **coverage/matchability pilot**, not a reasoning claim. The max2 GELAB diagnostic had zero verified-edge comparisons; the cohort tested how often the online mechanism could bind probes to authoritative actions. The manual smoke was not counted among the 40 tasks.

## Final results — frozen 40-task pilot

All 40 tasks completed with return code 0: 20 historical baseline successes and 20 failures, 18 apps, and low/middle/high complexity 17/17/6. The GELAB-ZERO-4B task evaluator passed 18/40 (45%): 13/20 in the historical-success stratum and 5/20 in the historical-failure stratum. Since this treatment did not modify model prompts or actions, task success is only a separate pilot outcome, not an effect of exploration.

The controller dispatched 38 tasks across two workers in 33m38s (67.8 tasks/hour); including the initial two-worker throughput check, 40 tasks took 38m33s end to end (62.2 tasks/hour). Per-task runtime averaged 103s, with a 98s median. Worker load was balanced at 19 controller tasks each; the dynamic queue gave the next task to whichever worker became free.

Across the 40 tasks, 44 probes were attempted. Recovery succeeded for 41/44 (93.2%); three recovery failures occurred in `NotesIsTodo`, `NotesRecipeIngredientCount`, and `RetroCreatePlaylist`, all on SCROLL. The failed probes remain audit records and ended exploration for their episodes. Twenty-three probes changed the layout and 21 were no-ops. The 268 inference windows stopped for `no_safe_candidate` 128 times, `candidate_rejected_or_no_progress` 43, `budget_exhausted` 41, `reasoning_finished` 53, and `recovery_failed` 3.

Of 410 authoritative actions, 368 (89.8%) could be bound to a control/action slot. However, **0/368** matched a prior source-restored probe observation on the same source/action. “Source-restored” means recovery returned the emulator to its anchor state; it does not mean the observed successor was independently replayed. The 44 `KNOWN_OUTCOME` and 11 `NEW_BRANCH` verdicts came from the authoritative path's own prior history, not a matching exploratory edge. This is the decisive negative result: random probes provide some recoverable transition coverage, but the current selector does not close the evidence-to-reasoning loop. More random probe budget is not the next experiment.

## Required next experiment

Implement and smoke-test a candidate-conditioned protocol: first let the reasoner propose an action slot; then let Explorer probe that exact slot only if it passes the reversible/recovery gate; restore and verify the source; give the observed/verified successor to a second reasoning decision; then execute the selected action and compare its actual successor against the graph. Keep the graph branch-preserving. If a candidate is unsafe or cannot be bound to a live control, record it as ineligible/unmatchable instead of broadening the safety filter.

## Full-registry coverage extension — complete

The remaining 76 tasks finished on the same seed-34 GELAB-ZERO-4B endpoint and random-selector configuration. All 76 process runs returned 0; the controller recorded no timeout and no task substitution. Together with the frozen 40-task pilot, this gives 116/116 registry task artifacts and official evaluator checkpoints across 20 apps. The pilot evaluator passed 50/116 and failed 66/116; this is a model pilot outcome, not a causal effect of exploration.

Across all 116 tasks there were 838 inference windows, 1,206 agent steps, and 139 probes. Source recovery passed for 131/139 probes (94.2%); eight failed and remain in the audit trail. Seventy-nine probes changed the layout and 60 were no-ops. Of 1,108 authoritative actions, 945 (85.3%) were bound to an action slot. **Zero of those 945** aligned to a prior recovered probe on the same source/action. The authoritative verdict counts were 997 `UNMATCHABLE`, 85 `KNOWN_OUTCOME`, and 26 `NEW_BRANCH`; because no probe aligned, the latter two counts reflect authoritative-path history, not exploratory validation. Stop reasons across 838 windows were `no_safe_candidate` 472, `candidate_rejected_or_no_progress` 99, `budget_exhausted` 131, `reasoning_finished` 128, and `recovery_failed` 8.

The 76-task extension took 61m39s across the two workers (73.9 tasks/hour; task wall-time mean 97s, median 86s, maximum 259s). Full aggregate: `experiments/info_value_profile_20261012/private_voi8-live-20261012-fd89ba/androidworld116_remaining76_run_20261013/aggregate_summary_116.json` (private LabServer data; not included in this branch).

This completes random-selector transition coverage/runtime profiling only. Do not present it as a reasoning intervention. The next treatment is still candidate-conditioned: propose an action slot, safely probe that exact slot, restore and validate source, supply the transition evidence to a second reasoning pass, commit the selected action, and compare its actual successor. Run that version as a separate treatment, first with a small safety smoke, then paired `fresh / none / stale-or-wrong` conditions. GELAB-ZERO-4B remains engineering screening; repeat the final primary claim on MAI-UI 2B.
