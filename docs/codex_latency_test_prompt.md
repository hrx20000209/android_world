# Codex task: measure the real-device latency cost of the semantic exploration ranker

## Context you need

This repo is **MobileExplorer**, a mobile GUI agent for AndroidWorld. It runs a
VLM (GELAB-ZERO-4B via vLLM on `localhost:8084`) that picks one UI action per
step, and **in parallel** with that inference it runs a short "exploration"
window that probes other on-screen controls and records what they lead to, into
a belief graph.

The whole design rests on one invariant:

> **Exploration is free only while it fits inside the window that inference is
> already occupying.** If a round overruns, exploration stops being hidden and
> starts adding wall-clock to every step.

Measured on the emulator: inference is **3.42 s median / 4.81 s p90**, and an
exploration round is 0.14 s median. So the budget is roughly **3.4 s per step**,
of which a scoring call may take at most ~150 ms before it eats into probing.

## What was just added, and why it needs a latency test

A **22M-parameter sentence encoder** (`sentence-transformers/paraphrase-MiniLM-L6-v2`,
CPU-only) now ranks exploration candidates.

- `android_world/parallel_exploration/semantic_service.py` — a persistent
  process that loads the model once and answers over a TCP socket
  (newline-delimited JSON). It exists *because* the explorer is a fresh Python
  interpreter per step (`spawn_explorer`, multiprocessing "spawn") and loading
  MiniLM costs **7.4 s**; per-step loading would blow the budget outright.
- `GraphKeywordRanker` in `android_world/parallel_exploration/rankers.py` —
  scores `sim(candidate, need) - 0.5 * max_j sim(candidate, discovered_label_j)`.
  Two stage: a cheap ranker narrows ~108 on-screen candidates to 32, then the
  encoder re-ranks those.
- Enabled by `--exploration_policy semantic --semantic_port 8766`.

**The open question is entirely about latency under load**, and there is already
one measurement that says it matters:

| condition | one call: 1 need + 23 labels x 32 candidates |
|---|---|
| host idle | **46 ms** |
| host busy (a 116-task arm running, vLLM serving) | **195 ms median, 131-244 ms range** |

195 ms already exceeds the ~150 ms allowance, on an emulator, on a Mac. Nobody
has measured it on a real phone, where the a11y dump and the probe taps are
slower and the inference window may be a different size.

## What to measure

Run on a **physical Android device** (not the emulator), with the agent's normal
stack up: vLLM on 8084, the fast a11y socket forwarded, and the encoder service
started with

```
python3 -m android_world.parallel_exploration.semantic_service --port 8766
```

Report these, with distributions (median / p90 / max), not just means:

1. **Encoder call latency**, isolated. Drive `semantic_service.query_many`
   directly with realistic payloads — 32 candidates, 1 need plus 0/8/24 known
   labels — under three host conditions: idle, vLLM serving, and vLLM serving
   while a task is running. This is the number that decides whether the
   two-stage design holds.
2. **Inference latency on the real device**, from `serial_events.jsonl`
   (`kind=="inference"`, field `inference_s`). The budget is whatever this is;
   on the emulator it was 3.42 s median. If the phone is faster, the encoder has
   *less* room, not more.
3. **Exploration round latency**, `kind=="explore"`, field `exploration_ms`,
   split by `probes_completed`. Compare `--exploration_policy semantic` against
   `--exploration_policy graph_matrix` (the default, no encoder) on the same
   tasks. The delta is the encoder's real cost in situ.
4. **Rounds that overran the window**: count rounds where
   `exploration_ms/1000 > ` the median `inference_s` of that episode. On the
   emulator this was 4.8% of rounds with the old ranker. If the semantic ranker
   pushes this up materially, the "free" claim needs qualifying in the paper.
5. **Memory and CPU** of the encoder service (RSS, and CPU% while answering),
   to confirm it stays inside the "<50 MB, <150 ms, CPU-only" envelope the
   design claims. Note the model itself is 22.7M parameters.

## How to run the agent for (2)-(4)

Use a small task set; there is a 37-task suite in
`configs/graph_active_suite.json` chosen because graph mechanisms actually fire
on it. Two arms, same session, back to back:

```
--exploration_policy semantic --semantic_port 8766 \
  --probes_per_step 3 --exploration_budget_s auto --max_depth 2 --enable_skip \
  --graph_context distill --no_decision_constraints --depth2_needs_known_inverse \
  --max_steps 15
```

and the same with `--exploration_policy graph_matrix` (drop `--semantic_port`).

`--exploration_budget_s auto` caps a round at the running median of that
episode's observed `inference_s`, so an overrun is bounded by one probe. Every
run writes `run_args.json` with the full argv and git HEAD — check it to confirm
the flags actually took effect rather than trusting the command line.

## Things that will bite you

- **The encoder service must be running before the agent starts.** If it is not,
  `GraphKeywordRanker` silently falls back to the cheap ranker and you will
  measure the wrong thing. Check for a live process on 8766 and assert that
  `semantic_service.query(8766, "x", ["y"])` returns a list, not `None`.
- **One call per known label is a trap.** An earlier version issued a request per
  label (24 round trips) and took a probe round to **6.9 s**. The encoder is
  dominated by how many strings it sees, and the candidates repeat across
  queries, so they must be encoded once — that is what `query_many` is for. If
  you add any new call site, batch it.
- **`ContactsNewContactDraft` crashes inside AndroidWorld's own
  `contacts.py:211`** on every arm. It is not your bug; expect N=115, not 116.
- Occasional `adb_controller` subprocess timeouts kill 1-3 tasks per run. Re-run
  those tasks rather than treating them as failures.

## What a useful answer looks like

A short table of the five measurements with distributions, plus a one-line
verdict on this question:

> **On a real device, does the encoder call fit inside the inference window, or
> does the semantic ranker make exploration stop being free?**

If it does not fit, say what would: a smaller candidate cap (currently 32), a
smaller label cap (currently 24), quantising the encoder, or moving the scoring
onto the GPU alongside vLLM.
