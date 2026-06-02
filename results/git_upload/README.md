# 可上传实验结果目录

这个目录用于保存可在 Git 间同步的“精选”实验结果。

建议放入内容（按任务场景增量提交）：
- summary 报告（如 phaseA/phaseB/phaseC 的 `*.md`）
- 聚合指标 CSV/JSON（如 `summary.json`, `per_task_results.csv`）
- 关键截图（若需复现实验证据）

提交前建议先确认已剔除非必要文件（raw trace、全量 a11y dump、冗长日志），以减少仓库体积。
