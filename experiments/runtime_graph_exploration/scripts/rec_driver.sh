#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/runtime_graph_exploration
for t in RetroCreatePlaylist RetroSavePlaylist SaveCopyOfReceiptTaskEval MarkorChangeNoteContent MarkorTranscribeReceipt ExpenseDeleteMultiple2 TasksCompletedTasksForDate SimpleCalendarAddOneEventTomorrow; do
  for arm in REC; do
    o=raw/hl/$arm/$t; [ -f $o/result.json ] && continue
    HL_ARM=$arm python scripts/hl_run.py --task $t --seed 32 --out $o > logs/hl_${arm}_$t.log 2>&1
    echo "REC $arm $t $(python -c "import json;r=json.load(open('$o/result.json'));print('succ',r.get('is_successful'),'steps',r.get('steps'),'calls',r.get('m_calls'),'hl',r.get('m_hl_steps'),'hl_rt',r.get('m_hl_from_runtime'),'clicked_hl',r.get('m_clicked_highlight'),'probes',r.get('m_probes'),'rbfail',r.get('m_rollback_fail'),'rq',r.get('m_requery'),r.get('error','')[:60])" 2>&1 | tail -1)"
  done
done
echo "REC ALL DONE"
