"""본 비교 모델(YOLOv8n·YOLO11s·RF-DETR-S)로 스트레스셋(대비 k배 축소) 예측 → preds/stress_<모델>.csv"""
import sys, os, glob
from common import *
from PIL import Image
def main():
    splits = {os.path.basename(d): d for d in sorted(glob.glob(f'{ROOT}/eval_sets/stress_*'))}
    from ultralytics import YOLO
    for name in ['yolov8n', 'yolo11s']:
        w = f'{RUNS}/{name}/weights/best.pt'
        if not os.path.exists(w): print('skip', name); continue
        m = YOLO(w); rows = []
        for sp, d in splits.items():
            for p in images(d):
                r = m.predict(p, imgsz=640, conf=0.001, iou=0.6, device=DEVICE, max_det=100, verbose=False)[0]
                for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    rows.append([stem(p), sp, f'{x1:.2f}', f'{y1:.2f}', f'{x2:.2f}', f'{y2:.2f}', f'{s:.5f}'])
        write_preds(f'{PRED}/stress_{name}.csv', rows); print('done', name)
    out = f'{RUNS}/rfdetr_s'
    ck = [f for f in ['checkpoint_best_total.pth', 'checkpoint_best_ema.pth', 'checkpoint_best_regular.pth'] if os.path.exists(f'{out}/{f}')]
    if ck:
        from rfdetr import RFDETRSmall
        m = RFDETRSmall(pretrain_weights=f'{out}/{ck[0]}', num_classes=1); rows = []
        for sp, d in splits.items():
            for p in images(d):
                det = m.predict(Image.open(p).convert('RGB'), threshold=0.001)
                for (x1, y1, x2, y2), s in zip(det.xyxy, det.confidence):
                    rows.append([stem(p), sp, f'{x1:.2f}', f'{y1:.2f}', f'{x2:.2f}', f'{y2:.2f}', f'{s:.5f}'])
        write_preds(f'{PRED}/stress_rfdetr_s.csv', rows); print('done rfdetr')
if __name__ == '__main__':
    main()
