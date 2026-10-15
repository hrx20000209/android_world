#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/runtime_graph_exploration
B=../trace_reachability_validation/raw/full
run() { # var arm task ref
  o=raw/var/$1/$3_$2; [ -f $o/result.json ] && return
  python scripts/variation_run.py --task $3 --ref $B/$4 --var $1 --arm $2 --out $o > logs/var_$1_$3_$2.log 2>&1
  echo "VAR $1 $2 $3 $(python -c "import json;r=json.load(open('$o/result.json'));print('succ',r.get('is_successful'),'exec',r.get('executed'),'/',r.get('ref_steps'),'gaps',[g.get('reason') for g in r.get('gaps',[])],'repairs',[(x['ok'],x.get('path'),x.get('why')) for x in r.get('repairs',[])],'probes',r.get('probes'),'rbfail',r.get('rollback_failures'),r.get('error','')[:60])" 2>&1 | tail -1)"
}
declare -a V2T=("SimpleCalendarAddOneEvent BASE/SimpleCalendarAddOneEvent/p0" "SimpleCalendarDeleteOneEvent BASE/SimpleCalendarDeleteOneEvent/p0" "SimpleCalendarEventsOnDate FULL4/SimpleCalendarEventsOnDate/p0")
declare -a V1T=("MarkorCreateFolder BASE/MarkorCreateFolder/p0" "ExpenseDeleteSingle BASE/ExpenseDeleteSingle/p0" "RecipeDeleteSingleRecipe BASE/RecipeDeleteSingleRecipe/p0" "TasksHighPriorityTasks BASE/TasksHighPriorityTasks/p0")
declare -a V4T=("MarkorDeleteNote BASE/MarkorDeleteNote/p0" "ExpenseDeleteSingle BASE/ExpenseDeleteSingle/p0" "ContactsAddContact BASE/ContactsAddContact/p0")
for v in V2 V1 V4; do
  eval "L=(\"\${${v}T[@]}\")"
  for x in "${L[@]}"; do set -- $x; run V0 plain $1 $2; run $v plain $1 $2; run $v repair $1 $2; done
done
echo "VAR ALL DONE"
