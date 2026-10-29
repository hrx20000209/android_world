#!/usr/bin/env bash
# Runs ON the Jetson AGX Orin (~/Projects/gelab-server/start_ngram8_graph.sh): GELab-Zero-4B on vLLM with n-gram
# speculative decoding (8 tokens, prompt lookup 2-6) and CUDA graphs (no --enforce-eager). Serves 127.0.0.1:8000.
#   ssh Jetson.AGX.Orin 'cd ~/Projects/gelab-server && setsid nohup bash start_ngram8_graph.sh > server_ngram8_graph.log 2>&1 < /dev/null &'
# If the server hangs in D state (kill -9 has no effect), only a reboot of the Orin frees the GPU.
set -euo pipefail
export PATH="$HOME/miniconda3/envs/agent/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/agent/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec python -m vllm.entrypoints.openai.api_server \
  --model "$HOME/Projects/models/gelab_zero_4B" \
  --served-model-name GELab-Zero-4B-preview \
  --host 127.0.0.1 --port 8000 \
  --dtype half --max-model-len 8192 \
  --gpu-memory-utilization 0.60 --max-num-seqs 1 \
  --mm-processor-kwargs '{"max_pixels":2097152}' \
  --speculative-config '{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":6,"prompt_lookup_min":2}'
