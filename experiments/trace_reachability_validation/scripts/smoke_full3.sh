#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
SM=/private/tmp/claude-501/-Users-huangrunxi-Projects-android-world/1dd310ad-f151-42bc-a619-4cc3aef047ac/scratchpad/full3smoke
for t in "MarkorCreateNote 30" "MarkorCreateFolder 30" "MarkorCreateNote 31" "MarkorCreateFolder 31" "MarkorDeleteNote 30"; do
  set -- $t
  FULL_ARM=FULL3 SEQ_APPDIR=$SM python scripts/full_run.py --task $1 --seed $2 --variant smk$2 --mode agent --out raw/smoke/full3/$1_$2 > logs/smoke_full3.log 2>&1
  grep RESULT logs/smoke_full3.log | cut -c1-300
  grep -A6 Traceback logs/smoke_full3.log | head -8
done
