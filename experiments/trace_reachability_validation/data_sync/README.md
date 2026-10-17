# data_sync：给离线分析（任务 B/C）用的小体积实验记录

- `full_records/{BASE,FULL}/<task>/p{0,1}/`：从 `raw/full/` 复制，只包含 `meta.json`、`result.json`、`steps.jsonl`、`probes.jsonl`、`agent_transitions.jsonl`、`windows.jsonl` 等 json/jsonl 文件。不含截图（`img/`）和 a11y dump（`a11y/`）；`steps.jsonl` 里已有页面签名（exact/struct/layout/content）和动作 token。
- 不含 `_partial_*` 目录。
- `graph_snapshots_full/FULL/`：FULL v1 两轮跑完后的全局图（`graph.jsonl`、`store.json`、`unrestorable.jsonl`）。
- 本机（emulator）上的原路径是 `raw/full/` 和 `graph_snapshots/full/`。在服务器上运行时，把脚本里的 `RAW` 指向 `data_sync/full_records`。
- FULL3 正在本机运行，不在这里。

## full_results（2026-10-17 追加）

- `full_results/<ARM>_p0.json`：各实验臂单轮（seed 30）的每题结果摘要，`{task: {goal, result}}`，`result` 即 `raw/full/<ARM>/<task>/p0/result.json`（含 `is_successful`、`model_calls`、`steps`、各机制计数 `m_*`）。覆盖 BASE、FULL7–FULL21R；FULL22 运行中未包含。
- 完整逐步记录（steps/windows/probes 等，各臂约 10–40MB）与全局图快照未提交，仍在本机 `raw/full/` 与 `graph_snapshots/full/`。
