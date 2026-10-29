#!/bin/sh
# Keeps localhost:18097 -> Orin 127.0.0.1:8000 alive (SSH host alias Jetson.AGX.Orin; campus network or VPN needed).
#   nohup sh scripts/launch/orin_tunnel_keeper.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
L=logs/orin_tunnel.log
while :; do
  if ! curl -s --noproxy '*' -m 8 http://127.0.0.1:18097/v1/models | grep -q GELab; then
    pkill -f "18097:127.0.0.1:8000"
    ssh -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ConnectTimeout=15 -o ExitOnForwardFailure=yes \
      -f -N -L 18097:127.0.0.1:8000 Jetson.AGX.Orin >> $L 2>&1 && echo "$(date '+%m-%d %H:%M:%S') orin tunnel reconnected" >> $L
  fi
  sleep 30
done
