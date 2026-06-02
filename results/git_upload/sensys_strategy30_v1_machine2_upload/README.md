# sensys_strategy30_v1 machine2 upload package

This directory contains machine_2 results for variants:

- S2_DFS_BUDGET12
- S4_MCTS_BUDGET12

Merge-ready layout:

```text
sensys_strategy30_v1/
  machine_2/
    S2_DFS_BUDGET12/
    S4_MCTS_BUDGET12/
```

Each variant directory includes non-empty `per_task_results.csv`, `per_step_metrics.csv`, key trace JSONL files, `variant_manifest.json`, logs, and `report/` with figures.

On the aggregation computer, put `machine_1` and this `machine_2` under the same `sensys_strategy30_v1` directory, then run:

```powershell
python scripts/merge_strategy_30task_results.py --run_group <path-to-sensys_strategy30_v1>
```
