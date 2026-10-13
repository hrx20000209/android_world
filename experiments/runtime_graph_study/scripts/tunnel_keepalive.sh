#!/bin/bash
# Keeps a local forward 18095 -> LabServer:8095 (GELAB-ZERO-4B vLLM) alive.
while true; do
  ssh -N -T -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      -o HostName=143.89.191.15 -o User=rxhuang -F /dev/null \
      -o StrictHostKeyChecking=yes -o IdentitiesOnly=no \
      -L 18095:127.0.0.1:8095 tunnel-labserver
  echo "$(date) tunnel exited $?, restarting" >&2
  sleep 5
done
