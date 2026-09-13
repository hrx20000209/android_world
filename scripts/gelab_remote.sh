#!/bin/bash
# AutoDL 上的 GELab-Zero-4B (vLLM) —— 隧道 + 环境变量
#
#   source scripts/gelab_remote.sh          # 建隧道并导出环境变量
#   source scripts/gelab_remote.sh restart  # 顺便重启远端 vLLM（实例重启后需要）
#
# 远端已就绪的东西（装在数据盘 /root/autodl-tmp，系统盘只有 30G）：
#   miniconda3/envs/vllm   python3.12 + vllm 0.29.0 + torch 2.13.0+cu130
#   models/GELab-Zero-4B-preview   8.3G, qwen3_vl 架构
#   /root/serve.sh         启动脚本（含 LD_LIBRARY_PATH 修复）

REMOTE_PORT=37516
REMOTE_HOST=root@connect.westc.seetacloud.com
LOCAL_PORT=8084

if [ "${1:-}" = "restart" ]; then
  echo "重启远端 vLLM ..."
  ssh -o BatchMode=yes -p $REMOTE_PORT $REMOTE_HOST 'bash /root/serve.sh'
  until ssh -o BatchMode=yes -p $REMOTE_PORT $REMOTE_HOST \
      'curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1'; do sleep 10; done
  echo "远端就绪"
fi

# 隧道：没有就建，有就复用
if ! curl -sf http://localhost:$LOCAL_PORT/v1/models >/dev/null 2>&1; then
  pkill -f "$LOCAL_PORT:127.0.0.1:8000" 2>/dev/null
  nohup ssh -N -T -o BatchMode=yes -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
      -L $LOCAL_PORT:127.0.0.1:8000 -p $REMOTE_PORT $REMOTE_HOST \
      > /tmp/gelab_tunnel.log 2>&1 &
  until curl -sf http://localhost:$LOCAL_PORT/v1/models >/dev/null 2>&1; do sleep 2; done
fi

export ANDROID_WORLD_LLM_API_URL=http://localhost:$LOCAL_PORT/v1/chat/completions
# 必须设：LlamaCppWrapper 默认不发 model 字段，vLLM 会拒绝（infer.py:433）
export ANDROID_WORLD_LLAMACPP_MODEL=GELAB-ZERO-4B
echo "就绪  $ANDROID_WORLD_LLM_API_URL  model=$ANDROID_WORLD_LLAMACPP_MODEL"
