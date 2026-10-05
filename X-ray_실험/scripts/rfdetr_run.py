"""RF-DETR-S 학습(MPS) → 예측 CSV. 사용: python rfdetr_run.py 15 [--smoke]"""
import sys, os, json, time, shutil
from common import *
from PIL import Image
def main():
    epochs=int(sys.argv[1]); smoke='--smoke' in sys.argv
    CO=f'{ROOT}/rfdetr_data{"_smoke" if smoke else ""}'
    def build(split, dirs, limit=None):
        o=f'{CO}/{split}'; os.makedirs(o,exist_ok=True)
        js={'images':[],'annotations':[],'categories':[{'id':0,'name':'defects','supercategory':'none'},{'id':1,'name':'defect','supercategory':'defects'}]}
        aid=0
        for d in dirs:
            ims=images(d)[:limit] if limit else images(d)
            for p in ims:
                s=stem(p); fn=s+'.png'; dst=f'{o}/{fn}'
                if not os.path.lexists(dst): os.symlink(os.path.relpath(p,o),dst)
                W,H=Image.open(p).size; iid=len(js['images']); js['images'].append({'id':iid,'file_name':fn,'width':W,'height':H})
                for l in open(f'{d}/labels/{s}.txt'):
                    if not l.strip(): continue
                    c,x,y,w,h=map(float,l.split()); bw,bh=w*W,h*H
                    js['annotations'].append({'id':aid,'image_id':iid,'category_id':1,'bbox':[x*W-bw/2,y*H-bh/2,bw,bh],'area':bw*bh,'iscrowd':0}); aid+=1
        json.dump(js,open(f'{o}/_annotations.coco.json','w'))
    lim=12 if smoke else None
    build('train',TRAIN_DIRS,lim); build('valid',[EVAL_DIRS['val']],lim); build('test',[EVAL_DIRS['test']],lim)
    from rfdetr import RFDETRSmall
    out=f'{RUNS}/rfdetr_s{"_smoke" if smoke else ""}'; os.makedirs(out,exist_ok=True)
    kw=dict(num_workers=2, dataset_dir=CO, epochs=1 if smoke else epochs, batch_size=4, grad_accum_steps=4, lr=1e-4, output_dir=out)
    m=RFDETRSmall(); t0=time.time()
    try: m.train(device=DEVICE, **kw)
    except TypeError as e:
        print('device kwarg not accepted, retrying without it:',e); m.train(**kw)
    train_sec=time.time()-t0
    ck=[f for f in ['checkpoint_best_total.pth','checkpoint_best_ema.pth','checkpoint_best_regular.pth','checkpoint.pth'] if os.path.exists(f'{out}/{f}')]
    print('checkpoint used:',ck[0])
    best=RFDETRSmall(pretrain_weights=f'{out}/{ck[0]}', num_classes=1)
    try: best.optimize_for_inference()
    except Exception as e: print('optimize skipped',e)
    rows=[];lat=[]
    for sp,d in EVAL_DIRS.items():
        for p in (images(d)[:8] if smoke else images(d)):
            im=Image.open(p).convert('RGB'); t=time.time(); det=best.predict(im, threshold=0.001); lat.append(time.time()-t)
            for (x1,y1,x2,y2),s in zip(det.xyxy, det.confidence):
                rows.append([stem(p),sp,f'{x1:.2f}',f'{y1:.2f}',f'{x2:.2f}',f'{y2:.2f}',f'{s:.5f}'])
    name='rfdetr_s'+('_smoke' if smoke else ''); write_preds(f'{PRED}/{name}.csv',rows)
    json.dump({'model':'RF-DETR-S','checkpoint':ck[0],'train_seconds':train_sec,'epochs_requested':epochs,'ms_per_image_mps_batch1':1000*sum(lat[5:])/max(1,len(lat)-5)},open(f'{PRED}/{name}_meta.json','w'),indent=1)
    print('DONE',name)


if __name__=='__main__':  # macOS는 DataLoader 워커가 spawn 방식이라 메인 가드 필요
    main()
