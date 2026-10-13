# Runtime GUI Dynamics + Model-Free Graph Exploration Study

完整中文报告：[reports/final_report_zh.md](reports/final_report_zh.md)

## 目录
| 路径 | 内容 |
|---|---|
| `scripts/rgs_ui.py` | a11y 解析、四种状态签名、安全 action slot 枚举、元素对应（无加权相似度） |
| `scripts/rgs_run.py` | 单 episode runner：agent（可选推理窗口探索）、strict / structure-aware 回放（+ 单独标注的 recovery） |
| `scripts/rgs_driver.py` | 批量驱动（可断点续跑，拒绝与其他 runner 并行） |
| `scripts/rgs_analyze.py` | 回放步、对齐、截图/A11y/转移差异（全部从 raw 重算） |
| `scripts/rgs_graph.py` | 从不可变观测重建图：exact/struct/layout × shared/per-run |
| `scripts/rgs_summary.py` | 带分母的汇总 → `analysis/summary.json`、`reports/experiment_summary.csv` |
| `scripts/rgs_review.py` | 状态匹配人工复核样本 |
| `scripts/rgs_utility.py` | 单步推理效用对照（A/B/C） |
| `scripts/rgs_figures.py` | 图 1–10 |
| `configs/batch{1,2}_manifest.json` | 14 个任务 × 5 个 seed |
| `raw/main/<task>/<episode>/` | 每个 episode：`meta.json` `result.json` `steps.jsonl` `probes.jsonl` `windows.jsonl` `replay.jsonl` `outcomes.jsonl` `img/*.jpg` `a11y/*.json.gz`（330 MB，不入 git） |
| `raw/old_*` | 修复前的运行（保留作证据，不进入统计） |
| `graphs/main/<task>.explore.jsonl` | 不可变转移观测；`*.{exact,struct,layout}.graph.json` 为重建快照；`_app_unrestorable.jsonl` 为 app 级不可恢复记忆 |
| `analysis/` | 所有 CSV/JSON、人工复核、效用实验 |
| `figures/` | PNG + PDF |
| `reports/` | 环境报告、任务清单、汇总、执行日志、最终报告 |

## 复现
```bash
scripts/tunnel_keepalive.sh &          # 127.0.0.1:18095 -> LabServer:8095 (GELAB-ZERO-4B)
python scripts/rgs_driver.py --manifest configs/batch1_manifest.json --raw raw/main --graphs graphs/main
python scripts/rgs_driver.py --manifest configs/batch2_manifest.json --raw raw/main --graphs graphs/main
python scripts/rgs_analyze.py raw/main analysis && python scripts/rgs_graph.py graphs/main analysis
python scripts/rgs_summary.py && python scripts/rgs_figures.py analysis figures
python scripts/rgs_utility.py select && python scripts/rgs_utility.py run && python scripts/rgs_utility.py sheet
```
