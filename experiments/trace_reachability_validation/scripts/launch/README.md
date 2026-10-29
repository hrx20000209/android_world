# Launch scripts

Run from anywhere; each script changes to `experiments/trace_reachability_validation`. Logs go to `logs/`, per-episode
results to `reports/<tag>_runs.jsonl`, raw episodes to `raw/<tag>/`.

| Step | Command |
|---|---|
| Orin inference server (on the Orin) | `orin_vllm_start_ngram8_graph.sh` (copy of `~/Projects/gelab-server/start_ngram8_graph.sh`) |
| Orin tunnel (localhost:18097) | `nohup sh scripts/launch/orin_tunnel_keeper.sh &` |
| LabServer tunnel (localhost:18126) | `nohup sh scripts/launch/labserver_tunnel_keeper.sh > logs/lab18126_keeper.log 2>&1 &` |
| Seen-task warm-up (LabServer) | `sh scripts/launch/run_seen_warmup.sh seenW1` |
| Seen/unseen eval (LabServer, steps only) | `sh scripts/launch/run_seen_eval_labserver.sh seenE2 seenW1 FULL45,FULL45+SEENR` |
| BASE vs FULL45 (Orin, e2e) | `sh scripts/launch/run_orin_e2e.sh BASE,FULL45 e2e116n` |
| Replacement ablations (Orin) | `sh scripts/launch/run_orin_e2e.sh FULL45+RAND,FULL45+INJ,FULL45+NOSYS e2e116nA` |
| Latest design, seen/unseen (Orin) | `sh scripts/launch/run_orin_e2e.sh FULL45+SEENR e2e116S2` |
| Same-session BASE vs seenE2 rules (Orin) | `sh scripts/launch/run_orin_e2e.sh BASE,FULL45+SEENR0 e2e116B` |
| Re-run episodes hit by an Orin hang | `python scripts/launch/mark_e2e_hang.py e2e116S2 FULL45+SEENR`, then the same `run_orin_e2e.sh` again |

Arm flags (set by the drivers): `+SEENR` = SAME_TPL, AGGR_COMMIT, TPL_DEMOS, EXPLORE_FIT, NAV_UNAMBIG;
`+SEENR0` = SAME_TPL, AGGR_COMMIT, TPL_DEMOS; `+RAND` / `+INJ` / `+NOSYS` = RAND_EXPLORE / GRAPH_PROMPT / NO_SYS.
The Orin runs need the emulator `emulator-5554` with the on-device load model at `/data/local/tmp/llm`.
