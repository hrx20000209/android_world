#!/bin/sh
# Keeps localhost:18126 -> LabServer 127.0.0.1:18126 (GELAB-ZERO-4B vLLM, RTX 4090 D) alive. -F /dev/null avoids the
# RemoteForward in ~/.ssh/config that fails on the LabServer.
#   nohup sh scripts/launch/labserver_tunnel_keeper.sh > logs/lab18126_keeper.log 2>&1 &
while :; do
  if ! curl -s --noproxy '*' -m 8 localhost:18126/v1/models | grep -qi gelab; then
    pkill -f "18126:127.0.0.1:18126"; sleep 1
    ssh -F /dev/null -o HostName=143.89.191.15 -o User=rxhuang -o BatchMode=yes -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 -o ConnectTimeout=20 -f -N -L 18126:127.0.0.1:18126 tunnel-labserver
    echo "$(date) reconnected"
  fi
  sleep 30
done
