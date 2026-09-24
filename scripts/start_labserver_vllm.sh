#!/usr/bin/env bash

# Start one isolated GELAB vLLM endpoint for the LabServer evaluation matrix.
# The client also sends temperature=0 and top_p=1; seed=0 makes the server
# side RNG setting explicit instead of relying on vLLM's default.
set -euo pipefail

gpu_id="${1:?usage: $0 GPU_ID PORT [MODEL_DIR]}"
port="${2:?usage: $0 GPU_ID PORT [MODEL_DIR]}"
model_dir="${3:-/home/rxhuang/Projects/models/gelab_zero_4B}"

exec env CUDA_VISIBLE_DEVICES="${gpu_id}" \
  /home/rxhuang/anaconda3/envs/agent/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model "${model_dir}" \
  --served-model-name GELAB-ZERO-4B \
  --host 0.0.0.0 \
  --port "${port}" \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --seed 0
