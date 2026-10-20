#!/bin/bash
# usage: chain_next.sh PREV NEXT "description"  -> starts NEXT (single pass) once PREV's pass is done
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
P=$1; N=$2; D=$3
pl=$(echo $P | tr A-Z a-z); nl=$(echo $N | tr A-Z a-z)
until grep -q "$P SINGLE PASS DONE" logs/full_driver_$pl.log; do sleep 30; done
while pgrep -f "full_run.py" > /dev/null || pgrep -f "full_driver.py $P" > /dev/null; do sleep 10; done
sed "s/FULL28/$N/g; s/full28/$nl/g" scripts/stop_after_pass0_full28.sh > scripts/stop_after_pass0_$nl.sh
echo "- $(date '+%m-%d %H:%M:%S') $N single pass started ($D)" >> reports/execution_log.md
nohup python scripts/full_driver.py $N > logs/full_driver_$nl.log 2>&1 &
sleep 2
nohup bash scripts/stop_after_pass0_$nl.sh > logs/stop_$nl.log 2>&1 &
