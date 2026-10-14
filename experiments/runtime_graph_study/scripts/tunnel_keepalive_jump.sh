#!/bin/bash
# Keeps 18095 -> LabServer:8095 alive via jump host ras.cse.ust.hk.
# The jump hop reuses a ControlMaster socket the user opens once by hand
# (password entered interactively), so this loop never handles credentials.
CM=$HOME/.ssh/cm-ras
while true; do
  if ssh -S "$CM" -O check rhuangbj@ras.cse.ust.hk 2>/dev/null; then
    ssh -N -T -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
        -o HostName=143.89.191.15 -o User=rxhuang -F /dev/null \
        -o StrictHostKeyChecking=yes -o IdentitiesOnly=no \
        -o ProxyCommand="ssh -S $CM -W %h:%p rhuangbj@ras.cse.ust.hk" \
        -L 18095:127.0.0.1:8095 tunnel-labserver
    echo "$(date) tunnel exited $?, restarting" >&2
  else
    echo "$(date) jump master not open" >&2
  fi
  sleep 5
done
