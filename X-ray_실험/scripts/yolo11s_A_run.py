"""YOLO11s를 A 구성(실제 영상만)으로 재학습 → 검증·테스트·정상합성·스트레스셋 예측 저장.
학습 조건은 기존 yolo11s(D 구성)와 동일: 640px, 최대 70ep, patience 20, batch 16, degrees 10, seed 0.
출력: runs/yolo11s_A, preds/yolo11s_A.csv, preds/stress_yolo11s_A.csv, preds/yolo11s_A_meta.json"""
import os, glob, time, json
from common import *
from ultralytics import YOLO
yaml = f'{ROOT}/ablation/A_실제만.yaml'; name = 'yolo11s_A'
t0 = time.time()
YOLO('yolo11s.pt').train(data=yaml, imgsz=640, epochs=70, patience=20, batch=16, device=DEVICE, degrees=10,
                         project=RUNS, name=name, exist_ok=True, seed=0, deterministic=False, workers=4, plots=True, verbose=True)
train_sec = time.time() - t0
m = YOLO(f'{RUNS}/{name}/weights/best.pt')
def pred(dirs):
    rows = []; lat = []
    for sp, d in dirs.items():
        for p in images(d):
            t = time.time(); r = m.predict(p, imgsz=640, conf=0.001, iou=0.6, device=DEVICE, max_det=100, verbose=False)[0]; lat.append(time.time() - t)
            for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                rows.append([stem(p), sp, f'{x1:.2f}', f'{y1:.2f}', f'{x2:.2f}', f'{y2:.2f}', f'{s:.5f}'])
    return rows, lat
rows, lat = pred(EVAL_DIRS); write_preds(f'{PRED}/{name}.csv', rows)
srows, _ = pred({os.path.basename(d): d for d in sorted(glob.glob(f'{ROOT}/eval_sets/stress_*'))}); write_preds(f'{PRED}/stress_{name}.csv', srows)
ntrain = sum(1 for l in open(yaml.replace('.yaml', '.txt')) if l.strip())
json.dump({'model': name, 'yaml': yaml, 'train_images': ntrain, 'train_seconds': train_sec, 'epochs_requested': 70,
           'ms_per_image_mps_batch1': 1000 * sum(lat[5:]) / max(1, len(lat) - 5)}, open(f'{PRED}/{name}_meta.json', 'w'), ensure_ascii=False, indent=1)
print('DONE', name)
