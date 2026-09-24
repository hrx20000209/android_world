# Online Exploration Gate

This directory defines the two experiments that must pass before investing in
a learned selector or a same-phone runtime.

## Experiment A: current-information oracle

Run all five arms from `protocol.json` on exactly the same task, seed and
decision point. Write one `OracleTrialResult.to_dict()` JSON object per line.
The loader rejects incomplete five-arm pairs, duplicate arms, current-seed
leakage, fresh evidence in non-online arms, stale evidence in the fresh arm,
and evidence bound to another state.

The oracle chooses a safe branch and extracts an answer slot; it does not get
the task answer directly. `evidence_eligible` means the arm had a valid online
opportunity, not that the final answer was correct. Record calls/actions from
the selected decision point through termination.

The no-opportunity decision points from the same runs are negative controls.
They must remain in the file with `online_opportunity=false`; they are excluded
from the per-family go/no-go test but retained in aggregate overhead results.

## Experiment B: dual-emulator isolation

Primary is authoritative. Shadow starts from the same task seed and replays
the hash-checked authoritative prefix. A probe is accepted only when:

1. activity, UI structure, evaluator-visible state and prefix match before it;
2. all speculative actions execute through the shadow-only callback;
3. primary remains byte-for-byte equal under the same four fingerprints;
4. shadow rebuilds from the authoritative prefix;
5. evidence extraction and rebuild finish before the inference deadline.

Write `ProbeOutcome.to_dict()` records as JSONL. Include successful probes,
shadow database mutations, shadow crashes, deliberate pre-probe mismatches,
and cancellations caused by early inference completion.

## Report command

```bash
python scripts/analyze_online_exploration_study.py \
  --oracle-jsonl results/oracle_trials.jsonl \
  --isolation-jsonl results/probe_outcomes.jsonl \
  --output-dir results/online_gate
```

The command validates the paired design, produces bootstrap confidence
intervals, applies the preregistered go/no-go rule, and creates the two meeting
figures. Do not replace missing observations or unsuccessful arms with
synthetic values.
