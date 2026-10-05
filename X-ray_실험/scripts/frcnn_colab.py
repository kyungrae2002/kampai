"""Faster R-CNN R50-FPN v2 (COCO 사전학습) — Colab GPU용. 같은 학습셋·같은 채점기 사용.
사용: python frcnn_colab.py --epochs 12 --out /content/drive/MyDrive/kamp_colab/out [--smoke]"""
import os, sys, json, time, argparse, random, numpy as np, torch, torchvision
from PIL import Image
from common import *
import evaluate as E
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torchvision.transforms.functional as TF
a=argparse.ArgumentParser(); a.add_argument('--epochs',type=int,default=12); a.add_argument('--out',default=f'{ROOT}/colab_out'); a.add_argument('--smoke',action='store_true'); A=a.parse_args()
os.makedirs(A.out,exist_ok=True); dev='cuda' if torch.cuda.is_available() else 'cpu'; print('device',dev, torch.cuda.get_device_name(0) if dev=='cuda' else '')
torch.manual_seed(0); random.seed(0); np.random.seed(0)
items=[(p,f'{d}/labels/{stem(p)}.txt') for d in TRAIN_DIRS for p in images(d)]
if A.smoke: random.shuffle(items); items=items[:64]
class DS(torch.utils.data.Dataset):
    def __len__(s): return len(items)
    def __getitem__(s,i):
        p,l=items[i]; im=TF.to_tensor(Image.open(p).convert('RGB')); _,H,W=im.shape; b=[]
        for t in open(l):
            if t.strip(): c,x,y,w,h=map(float,t.split()); b.append([(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H])
        b=torch.tensor(b,dtype=torch.float32).reshape(-1,4)
        if random.random()<0.5: im=im.flip(-1); b[:,[0,2]]=W-b[:,[2,0]]
        if random.random()<0.5: im=im.flip(-2); b[:,[1,3]]=H-b[:,[3,1]]
        return im,{'boxes':b,'labels':torch.ones(len(b),dtype=torch.int64)}
dl=torch.utils.data.DataLoader(DS(),batch_size=8,shuffle=True,num_workers=2,collate_fn=lambda x:tuple(zip(*x)))
m=fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,min_size=800,max_size=1333,box_detections_per_img=100,box_score_thresh=0.001)
m.rpn.anchor_generator=AnchorGenerator(((16,),(32,),(64,),(128,),(256,)),((0.5,1.0,2.0),)*5)  # 작은 이물(~10px) 대응
m.roi_heads.box_predictor=FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,2); m.to(dev)
params=[p for p in m.parameters() if p.requires_grad]
opt=torch.optim.SGD(params,lr=0.02,momentum=0.9,weight_decay=1e-4)
EP=1 if A.smoke else A.epochs; total=EP*len(dl); warm=min(300,total//5)
sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s: (s+1)/warm if s<warm else 0.5*(1+np.cos(np.pi*(s-warm)/max(1,total-warm))))
scaler=torch.amp.GradScaler(enabled=dev=='cuda')
@torch.no_grad()
def predict(splits,limit=None):
    m.eval(); rows=[]; lat=[]
    for sp in splits:
        for p in (images(EVAL_DIRS[sp])[:limit] if limit else images(EVAL_DIRS[sp])):
            im=TF.to_tensor(Image.open(p).convert('RGB')).to(dev)
            if dev=='cuda': torch.cuda.synchronize()
            t=time.time(); o=m([im])[0]
            if dev=='cuda': torch.cuda.synchronize()
            lat.append(time.time()-t)
            for (x1,y1,x2,y2),s in zip(o['boxes'].cpu().numpy(),o['scores'].cpu().numpy()):
                rows.append([stem(p),sp,f'{x1:.2f}',f'{y1:.2f}',f'{x2:.2f}',f'{y2:.2f}',f'{s:.5f}'])
    m.train(); return rows, 1000*float(np.mean(lat[5:])) if len(lat)>5 else None
gt_val=E.load_gt('val'); best=(-1,-1); hist=[]; t0=time.time(); step=0
for ep in range(EP):
    m.train(); tl=0
    for ims,tg in dl:
        ims=[i.to(dev) for i in ims]; tg=[{k:v.to(dev) for k,v in t.items()} for t in tg]
        with torch.autocast(device_type='cuda',enabled=dev=='cuda'):
            loss=sum(m(ims,tg).values())
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); step+=1; tl+=loss.item()
        if step%50==0: print(f'ep{ep} step{step}/{total} loss {loss.item():.4f} {time.time()-t0:.0f}s',flush=True)
    rows,_=predict(['val'],limit=8 if A.smoke else None)
    tmp=f'{A.out}/_val_tmp.csv'; write_preds(tmp,rows); ap50=E.ap(gt_val,E.load_preds(tmp,'val'),0.5)
    hist.append({'epoch':ep,'train_loss':tl/len(dl),'val_AP50':ap50,'sec':time.time()-t0}); print(hist[-1],flush=True)
    if ap50>best[0]: best=(ap50,ep); torch.save(m.state_dict(),f'{A.out}/frcnn_best.pt')
train_sec=time.time()-t0
m.load_state_dict(torch.load(f'{A.out}/frcnn_best.pt',map_location=dev))
rows,ms=predict(list(EVAL_DIRS),limit=8 if A.smoke else None)
name='faster_rcnn'+('_smoke' if A.smoke else ''); write_preds(f'{A.out}/{name}.csv',rows)
json.dump({'model':'Faster R-CNN R50-FPN v2','device':torch.cuda.get_device_name(0) if dev=='cuda' else dev,'best_epoch':best[1],'best_val_AP50':best[0],
           'history':hist,'train_seconds':train_sec,'ms_per_image_gpu_batch1':ms,'note':'속도는 Colab GPU 기준이라 맥 MPS 모델과 직접 비교 불가'},open(f'{A.out}/{name}_meta.json','w'),indent=1,ensure_ascii=False)
if not A.smoke: json.dump(E.evaluate(f'{A.out}/{name}.csv'),open(f'{A.out}/{name}_eval.json','w'),indent=1,ensure_ascii=False)
print('DONE',name)
