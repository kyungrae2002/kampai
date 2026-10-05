#!/bin/bash
# X-ray 이물 탐지 — 맥(MPS) 전체 실행 스크립트. 중간에 멈춰도 다시 실행하면 끝난 단계는 건너뜀.
# 실행:  bash ~/Desktop/kamp_ai/X-ray_실험/run_all.sh
set -u
cd "$(dirname "$0")"; ROOT="$PWD"; mkdir -p logs preds runs
LOG="$ROOT/logs/run_all.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== start $(date)"
# 잠자기 방지 (이 스크립트가 끝날 때까지)
caffeinate -dimsu -w $$ &
PY312=/Users/kyungrae/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12
[ -x "$PY312" ] || PY312=$(command -v python3.12 || command -v python3.11 || command -v python3)
step(){ # name, command...
  local n="$1"; shift
  if [ -f "$ROOT/logs/$n.done" ]; then echo "--- skip $n (done)"; return 0; fi
  echo "--- $n start $(date)"; local t=$SECONDS
  if "$@" > "$ROOT/logs/$n.log" 2>&1; then touch "$ROOT/logs/$n.done"; echo "--- $n OK ($((SECONDS-t))s)"; else echo "!!! $n FAILED ($((SECONDS-t))s) — see logs/$n.log"; touch "$ROOT/logs/$n.failed"; return 1; fi
}
# 1) 환경
if [ ! -x .venv/bin/python ]; then "$PY312" -m venv .venv; fi
PY="$ROOT/.venv/bin/python"
step 00_install bash -c "$PY -m pip install -U pip && $PY -m pip install torch torchvision ultralytics 'rfdetr[train]' transformers peft accelerate pillow opencv-python-headless numpy" || exit 1
step 01_env "$PY" -c "import torch,ultralytics,transformers,peft,platform;print(platform.platform(),torch.__version__,'mps',torch.backends.mps.is_available(),ultralytics.__version__,transformers.__version__,peft.__version__)" || exit 1
cd scripts
# 2) 스모크 테스트 (각 모델 1epoch 소량) — 하나라도 실패하면 본 학습 전 중단
S=0
step 10_smoke_yolo "$PY" yolo_run.py yolov8n.pt yolov8n 1 --smoke || S=1
step 11_smoke_rfdetr "$PY" rfdetr_run.py 1 --smoke || S=1
step 12_smoke_vlm "$PY" vlm_run.py smoke || S=1
if [ $S -ne 0 ]; then echo "스모크 실패 — 본 학습을 시작하지 않습니다. logs 폴더를 확인하세요."; exit 1; fi
# 3) 본 학습 (가벼운 것부터)
step 20_yolov8n "$PY" yolo_run.py yolov8n.pt yolov8n 80
step 21_yolo11s "$PY" yolo_run.py yolo11s.pt yolo11s 70
step 30_vlm_zeroshot "$PY" vlm_run.py zeroshot
step 31_vlm_lora "$PY" vlm_run.py train
step 40_rfdetr_s "$PY" rfdetr_run.py 15
# 4) 채점
for f in "$ROOT"/preds/*.csv; do case "$f" in *_smoke.csv) continue;; esac
  n=$(basename "$f" .csv); [ -f "$ROOT/preds/${n}_eval.json" ] || "$PY" evaluate.py "$f" --out "$ROOT/preds/${n}_eval.json" > /dev/null && echo "eval $n"; done
echo "=== all done $(date)"
