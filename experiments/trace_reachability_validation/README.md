# Trace-Guided Exploration & Backward Reachability Validation

最终报告：[reports/final_report_zh.md](reports/final_report_zh.md)；Pilot 决策门：[reports/pilot_findings_zh.md](reports/pilot_findings_zh.md)

| 路径 | 内容 |
|---|---|
| `scripts/common.py` | 读取上一轮原始证据；slot/行族身份；动作 token |
| `scripts/pa_trace.py`, `pa_report.py` | Pilot A：轨迹续写索引、P0–P4 排序、warm/loto/loao/cold 四种历史设定 |
| `scripts/pb_states.py`, `pb_run.py`, `pb_sheets.py` | Pilot B：决策状态、M0–M4 记忆构造、受控推理与打分 |
| `configs/pilot_b_labels.json` | 人工核对的动作标签（E/N/C 集合） |
| `scripts/tr_run.py`, `tr_driver.py`, `tr_analyze.py` | Phase II 在线：TG/RR × 深度 1/2，成对运行 |
| `analysis/pilot_a`, `analysis/pilot_b`, `analysis/phase2` | 所有逐决策 / 逐步数据与汇总 |
| `reachability_memory/pilot_b_memory.json` | 每个决策状态的 M1–M4 记忆（可复查） |
| `raw/phase2/<task>/s<seed>_<arm>/` | 在线 episode 的原始截图、a11y、probe、窗口日志 |
| `graph_snapshots/phase2/` | 每个 arm 的图（以上一轮的图为起点） |
| `scripts/p3_offline.py`, `pf_run.py`, `pf_driver.py`, `pf_analyze.py` | Phase III：离线阶段/页面分析，闭环预填实验 |
| `figures/` | fig1–fig10（PNG+PDF） |

复现：
```bash
python scripts/pa_trace.py && python scripts/pa_report.py
python scripts/pb_states.py && python scripts/pb_run.py run && python scripts/pb_run.py score
python scripts/tr_driver.py --manifest configs/phase2_manifest.json --raw raw/phase2 --arms TG,RR
python scripts/tr_driver.py --manifest configs/phase2_manifest.json --raw raw/phase2 --arms TG1
python scripts/tr_driver.py --manifest configs/phase2_manifest.json --raw raw/phase2 --arms RR1
python scripts/tr_analyze.py && python scripts/figs.py
```
