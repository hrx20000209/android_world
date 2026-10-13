#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
SM=/private/tmp/claude-501/-Users-huangrunxi-Projects-android-world/1dd310ad-f151-42bc-a619-4cc3aef047ac/scratchpad/full2smokec
for s in 30 31 32; do
  FULL_ARM=FULL2 SEQ_APPDIR=$SM python scripts/full_run.py --task ExpenseDeleteSingle --seed $s --variant smk$s --mode agent --out raw/smoke/full2c/ExpenseDeleteSingle_$s > logs/smoke_full2.log 2>&1
  grep RESULT logs/smoke_full2.log | cut -c1-430
  grep -A8 Traceback logs/smoke_full2.log | head -10
done
