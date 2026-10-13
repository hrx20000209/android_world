# data_sync：给离线分析（任务 B/C）用的小体积实验记录

- `full_records/{BASE,FULL}/<task>/p{0,1}/`：从 `raw/full/` 复制，只包含 `meta.json`、`result.json`、`steps.jsonl`、`probes.jsonl`、`agent_transitions.jsonl`、`windows.jsonl` 等 json/jsonl 文件。不含截图（`img/`）和 a11y dump（`a11y/`）；`steps.jsonl` 里已有页面签名（exact/struct/layout/content）和动作 token。
- 不含 `_partial_*` 目录。
- `graph_snapshots_full/FULL/`：FULL v1 两轮跑完后的全局图（`graph.jsonl`、`store.json`、`unrestorable.jsonl`）。
- 本机（emulator）上的原路径是 `raw/full/` 和 `graph_snapshots/full/`。在服务器上运行时，把脚本里的 `RAW` 指向 `data_sync/full_records`。
- FULL3 正在本机运行，不在这里。
