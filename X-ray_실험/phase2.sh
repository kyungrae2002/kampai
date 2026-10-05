#!/bin/bash
# 2단계: 데이터 조합 비교(A~F) + 스트레스 예측. 1단계(run_all.sh)가 끝날 때까지 기다렸다가 자동 시작.
set -u
cd "$(dirname "$0")"; ROOT="$PWD"; mkdir -p logs
LOG="$ROOT/logs/phase2.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== phase2 start $(date)"
caffeinate -dimsu -w $$ &
while pgrep -f "run_all.sh" > /dev/null; do echo "1단계 진행 중... 대기 $(date +%H:%M)"; sleep 300; done
PY="$ROOT/.venv/bin/python"
step(){ local n="$1"; shift
  if [ -f "$ROOT/logs/$n.done" ]; then echo "--- skip $n (done)"; return 0; fi
  echo "--- $n start $(date)"; local t=$SECONDS
  if "$@" > "$ROOT/logs/$n.log" 2>&1; then touch "$ROOT/logs/$n.done"; echo "--- $n OK ($((SECONDS-t))s)"; else echo "!!! $n FAILED ($((SECONDS-t))s) — see logs/$n.log"; touch "$ROOT/logs/$n.failed"; return 1; fi; }
cd scripts
A="$ROOT/ablation"
step 50_stress_preds "$PY" p2_predict_stress.py
step 51_p2_A "$PY" p2_yolo_ablation.py A_실제만 "$A/A_실제만.yaml" 30 640 0
step 52_p2_B "$PY" p2_yolo_ablation.py B_실제+이물합성 "$A/B_실제+이물합성.yaml" 30 640 0
step 53_p2_C "$PY" p2_yolo_ablation.py C_실제+이물제거 "$A/C_실제+이물제거.yaml" 30 640 0
step 54_p2_D "$PY" p2_yolo_ablation.py D_전체 "$A/D_전체.yaml" 30 640 0
if [ ! -f "$ROOT/logs/55_p2_E.done" ]; then EY=$("$PY" p2_select_E.py); echo "E 데이터: $EY"; step 55_p2_E "$PY" p2_yolo_ablation.py E_선택+960 "$EY" 30 960 0; fi
step 56_p2_F "$PY" p2_yolo_ablation.py F_전체+동료합성 "$A/F_전체+동료합성.yaml" 30 640 0
step 57_p2_A_seed1 "$PY" p2_yolo_ablation.py A_실제만_seed1 "$A/A_실제만.yaml" 30 640 1
step 58_p2_D_seed1 "$PY" p2_yolo_ablation.py D_전체_seed1 "$A/D_전체.yaml" 30 640 1
for f in "$ROOT"/preds/p2_*.csv "$ROOT"/preds/*.csv; do case "$f" in *_smoke.csv|*/stress_*) continue;; esac
  n=$(basename "$f" .csv); [ -f "$ROOT/preds/${n}_eval.json" ] || { "$PY" evaluate.py "$f" --out "$ROOT/preds/${n}_eval.json" > /dev/null && echo "eval $n"; }; done
echo "=== phase2 all done $(date)"
