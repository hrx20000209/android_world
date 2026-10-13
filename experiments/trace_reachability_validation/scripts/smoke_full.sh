#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
SM=/private/tmp/claude-501/-Users-huangrunxi-Projects-android-world/1dd310ad-f151-42bc-a619-4cc3aef047ac/scratchpad/fullsmoke2
for t in "ExpenseDeleteSingle 30" "ExpenseAddSingle 30" "ExpenseDeleteSingle 31"; do
  set -- $t
  FULL_ARM=FULL SEQ_APPDIR=$SM python scripts/full_run.py --task $1 --seed $2 --variant smk$2 --mode agent --out raw/smoke/full2/$1_$2 > logs/smoke_full.log 2>&1
  grep RESULT logs/smoke_full.log | cut -c1-520
  grep -A8 Traceback logs/smoke_full.log | head -10
done
