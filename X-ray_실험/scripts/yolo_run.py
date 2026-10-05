"""YOLO 학습(MPS) → 검증/테스트/정상합성 예측 CSV 저장.  사용: python yolo_run.py yolov8n.pt yolov8n 80 [--smoke]"""
import sys, time, json, os
from common import *
from ultralytics import YOLO
weights, name, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]); smoke='--smoke' in sys.argv
yaml=os.path.join(ROOT,'data_전체+약라벨.yaml')
run=f'{RUNS}/{name}{"_smoke" if smoke else ""}'
m=YOLO(weights)
t0=time.time()
m.train(data=yaml, imgsz=640, epochs=1 if smoke else epochs, patience=20, batch=16, device=DEVICE, degrees=10,
        project=RUNS, name=os.path.basename(run), exist_ok=True, seed=0, deterministic=False, workers=4,
        fraction=0.03 if smoke else 1.0, plots=not smoke, verbose=True)
train_sec=time.time()-t0
best=YOLO(f'{run}/weights/best.pt')
rows=[]; lat=[]
for sp,d in EVAL_DIRS.items():
    ims=images(d)[:8] if smoke else images(d)
    for p in ims:
        t=time.time(); r=best.predict(p, imgsz=640, conf=0.001, iou=0.6, device=DEVICE, max_det=100, verbose=False)[0]; lat.append(time.time()-t)
        for (x1,y1,x2,y2),s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            rows.append([stem(p),sp,f'{x1:.2f}',f'{y1:.2f}',f'{x2:.2f}',f'{y2:.2f}',f'{s:.5f}'])
out=f'{PRED}/{name}{"_smoke" if smoke else ""}.csv'; write_preds(out,rows)
json.dump({'model':name,'weights':weights,'train_seconds':train_sec,'epochs_requested':epochs,
           'ms_per_image_mps_batch1':1000*sum(lat[5:])/max(1,len(lat)-5),'pred_csv':out},
          open(f'{PRED}/{name}{"_smoke" if smoke else ""}_meta.json','w'),indent=1)
print('DONE',name,out)
