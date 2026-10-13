# MobileExplorer design validation (2026-10-13)

This folder contains the pilot code, aggregate results, Chinese reports, and figures for the MobileExplorer design review. The code changes in `android_world/parallel_exploration/` implement live action-to-control binding and verified multi-outcome transition evidence; `scripts/expert_trace_harness.py` records evaluator-checked traces through direct actions without constructing a model client.

`reports/final_design_validation_zh.md` is the canonical design/result report. `reports/interim_progress_and_design_20261013_zh.md` has the latest count snapshot. Earlier protocol and audit notes are retained as research history; where their counts differ, use the canonical report and its explicit data cutoff.

## Reproduce the committed figures

From the repository root, run:

```bash
python experiments/mobileexplorer_design_validation/generate_final_validation_figures.py
```

The plotting inputs are intentionally minimized: `plot_inputs/` contains aggregate gate fields and paired outcome indicators only. `topk_evidence_pilot_20261013/summary.json` contains the condition-level aggregates. The actual screenshot composite is committed as a figure; its original private capture directory is not included.

## Re-run the evidence pilot

`topk_evidence_pilot_20261013.py` needs the completed profiling run and a live GELAB-ZERO-4B vLLM endpoint on localhost ports 8095 and 8096. Supply the profiling root and, if needed, a custom output location through environment variables:

```bash
MOBILEEXPLORER_PROFILE_ROOT=/path/to/private_voi8-profile \
MOBILEEXPLORER_PILOT_OUTPUT_DIR=/path/to/private-pilot-output \
python experiments/mobileexplorer_design_validation/topk_evidence_pilot_20261013.py --prepare-only
```

Use `--run` only when both services are ready and the candidate dataset has been audited. The pilot is K=1 and tests a path-agreement proxy; it does not measure AndroidWorld task success or establish that evidence improves reasoning.

The full interim report generator reads the 40-task and 76-task profiling directories, P0 smoke audit, and live-smoke traces. Set `MOBILEEXPLORER_PROFILE_ROOT` to the directory that contains those experiment runs and `MOBILEEXPLORER_REPORT_OUT` to choose a report output directory before running it. Those per-task traces and private captures are kept off this branch.

## Scope and interpretation

- The report labels results as GELAB-ZERO-4B pilot results; they are not MAI-UI 2B results.
- P0 validates the probe, rollback, graph retrieval, and replay mechanisms on three screens. It is an engineering smoke, not an efficacy estimate.
- P1 compares 24 decision states per evidence condition offline. Its action metric is successful-run-path agreement, not ground-truth action correctness or task completion.
- The 116-task profile describes runtime behavior and outcomes; it does not establish that dynamic UI caused baseline failures.

Raw model responses, task-level UI traces, full manifests, and per-task screenshots are excluded. The committed report, aggregate JSON, minimized plotting inputs, and figures preserve the claims and chart inputs needed for review without publishing those run-level artifacts.
