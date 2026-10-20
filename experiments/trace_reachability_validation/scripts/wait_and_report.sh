#!/bin/bash
# usage: wait_and_report.sh ARM N  -> waits until N episodes of ARM are done (or the pass ends), prints the report
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
A=$1; N=$2; L=logs/full_driver_$(echo $A | tr A-Z a-z).log
until [ $(ls raw/full/$A/*/p0/result.json 2>/dev/null | wc -l) -ge $N ] || grep -q "SINGLE PASS DONE\|Traceback" $L 2>/dev/null; do sleep 60; done
python scripts/progress20.py $A FULL37 FULL34 FULL33 BASE
python scripts/full_table.py BASE FULL33 FULL34 FULL37 $A | tail -6
