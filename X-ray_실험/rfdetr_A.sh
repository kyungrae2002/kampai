#!/bin/bash
# RF-DETR-S를 A 구성으로 재학습 → 예측 → [검증셋 전용] 2차 판정. 예상 약 1.5시간.
set -u
cd "$(dirname "$0")"; ROOT="$PWD"; mkdir -p logs
LOG="$ROOT/logs/rfdetr_A.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== rfdetr_A start $(date)"
caffeinate -dimsu -w $$ &
PY="$ROOT/.venv/bin/python"
step(){ local n="$1"; shift
  if [ -f "$ROOT/logs/$n.done" ]; then echo "--- skip $n (done)"; return 0; fi
  echo "--- $n start $(date)"; local t=$SECONDS
  if "$@" > "$ROOT/logs/$n.log" 2>&1; then touch "$ROOT/logs/$n.done"; echo "--- $n OK ($((SECONDS-t))s)"; else echo "!!! $n FAILED ($((SECONDS-t))s) — see logs/$n.log"; touch "$ROOT/logs/$n.failed"; exit 1; fi; }
cd scripts
step 71_rfdetr_A_train "$PY" rfdetr_A_run.py
"$PY" compare_A2_val.py | tee "$ROOT/logs/rfdetr_A_판정.txt"
echo "=== rfdetr_A all done $(date)  (테스트셋은 아직 평가하지 않았습니다)"
