# Lightweight GUI Exploration Selector (V0)

This directory contains a deliberately small feasibility prototype. It scores
each clickable UI element with exactly:

```text
score(element) = w^T multi_hot(tokens(text, content_desc, resource_id)) + bias
```

There is no neural network, embedding, LLM, ONNX Runtime, or PyTorch on the
phone. The Android executable is a statically linked arm64 C++ program using a
small built-in JSON parser and the C++ standard library.

## Files

- `vocabulary.txt`: the explicit 81-token vocabulary.
- `train.py`: dependency-free logistic-regression training and weight export.
- `explorer_selector.cpp`: standalone inference and benchmark executable.
- `weights.txt`: manually initialized weights for testing before a dataset exists.
- `example_screen.json`, `example_train.json`: small examples.
- `build_android.sh`: NDK arm64 cross-compilation.

The weight format is intentionally plain text. A model contains `bias` and one
weight per vocabulary token, so it is easy to inspect, edit, version, and push
to a device.

## Train offline

```bash
python3 train.py \
  --train example_train.json \
  --output weights.trained.txt

# Or start from the manually initialized model.
python3 train.py \
  --train example_train.json \
  --weights-in weights.txt \
  --output weights.trained.txt
```

Each training row needs `text`, `content_desc`, `resource_id`, and `label`.
The script also accepts an optional nested `element` object. Tokenization is
lowercase alphanumeric word splitting; features are presence bits, not counts.

## Build and run on a Galaxy S24 or any arm64 Android device

The host needs an Android NDK. `build_android.sh` uses `ANDROID_NDK_HOME`,
`ANDROID_NDK_ROOT`, `ANDROID_HOME/ndk/*`, or the standard macOS SDK location.

```bash
./build_android.sh
adb push explorer_selector /data/local/tmp/
adb push example_screen.json /data/local/tmp/
adb push weights.txt /data/local/tmp/
adb shell chmod +x /data/local/tmp/explorer_selector
adb shell "/data/local/tmp/explorer_selector /data/local/tmp/example_screen.json /data/local/tmp/weights.txt"
```

Example output is sorted by score and ends with:

```text
SELECTED: index=0 text="Search"
```

The input JSON is a list of objects. Only `index`, `text`, `content_desc`, and
`resource_id` are needed; unknown fields are ignored.

## Benchmark

The benchmark excludes JSON/model loading and measures the repeated scoring and
sorting pass over the same already-loaded screen. It reports candidate count,
average, p50, p95, model bytes, executable bytes, and the process's approximate
`VmHWM` peak memory.

```bash
./explorer_selector example_screen.json weights.txt --benchmark 1000
```

For a device measurement:

```bash
adb shell "/data/local/tmp/explorer_selector \
  /data/local/tmp/example_screen.json \
  /data/local/tmp/weights.txt --benchmark 1000"
```

One measured SM-S9210 run is recorded in `benchmark_s24.json`. It is a device
microbenchmark, not an AndroidWorld task success result.

## AndroidWorld integration

The existing AndroidWorld shadow explorer can select this ranker after its
existing safety filter:

```bash
python scripts/run_sensys30_online_task.py \
  --output results/lightweight_selector/clock \
  --task ClockStopWatchRunning \
  --seed 30 \
  --max_steps 8 \
  --ranker LightweightLinearRanker \
  --selector_weights gui_exploration_selector/weights.txt
```

This does not weaken the exploration safety gate or recovery verification. The
tiny model only orders the safe candidates. The existing Executable Memory also
writes a bounded compressed graph sidecar named `executable_memory.compact.json`
and injects a compact state/edge summary into low-confidence reasoning prompts.

The AndroidWorld run still needs the repository's normal model inference
service and two-device setup. Results are not fabricated when that service is
unavailable.

For this first experiment, `weights.txt` is intentionally manually initialized
with near-ground-truth-like priors. It is not presented as a trained model.
