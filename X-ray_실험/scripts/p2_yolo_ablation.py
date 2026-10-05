"""2단계 데이터 조합 비교: YOLOv8n을 같은 조건(30epoch, 조기종료 없음)으로 학습 → 검증·테스트·정상합성·스트레스셋 예측
사용: python p2_yolo_ablation.py <실험이름> <yaml> <epochs> <imgsz> <seed>"""
import sys, os, glob, time, json
from common import *
from ultralytics import YOLO
name, yaml, epochs, imgsz, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
run = f'{RUNS}/p2_{name}'
t0 = time.time()
YOLO('yolov8n.pt').train(data=yaml, imgsz=imgsz, epochs=epochs, patience=epochs, batch=16, device=DEVICE, degrees=10,
                         project=RUNS, name=f'p2_{name}', exist_ok=True, seed=seed, workers=4, plots=True, verbose=False)
train_sec = time.time() - t0
m = YOLO(f'{run}/weights/best.pt'); rows = []
splits = dict(EVAL_DIRS); splits.update({os.path.basename(d): d for d in sorted(glob.glob(f'{ROOT}/eval_sets/stress_*'))})
for sp, d in splits.items():
    for p in images(d):
        r = m.predict(p, imgsz=imgsz, conf=0.001, iou=0.6, device=DEVICE, max_det=100, verbose=False)[0]
        for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            rows.append([stem(p), sp, f'{x1:.2f}', f'{y1:.2f}', f'{x2:.2f}', f'{y2:.2f}', f'{s:.5f}'])
write_preds(f'{PRED}/p2_{name}.csv', rows)
ntrain = sum(1 for l in open(yaml.replace('.yaml', '.txt')) if l.strip()) if os.path.exists(yaml.replace('.yaml', '.txt')) else None
json.dump({'exp': name, 'yaml': yaml, 'epochs': epochs, 'imgsz': imgsz, 'seed': seed, 'train_images': ntrain, 'train_seconds': train_sec},
          open(f'{PRED}/p2_{name}_meta.json', 'w'), ensure_ascii=False, indent=1)
print('DONE', name)
