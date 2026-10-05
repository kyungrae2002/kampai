"""공통 채점기: 모든 모델의 예측 CSV(image,split,x1,y1,x2,y2,score; 픽셀 좌표)를 같은 기준으로 평가.
split ∈ {val, test, val_normal, test_normal}. 임계값은 val에서 F1 최대가 되는 값으로 고정해 test에 그대로 적용."""
import os,sys,json,glob,csv,argparse,numpy as np
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,'..'))
DATA=os.path.abspath(os.path.join(ROOT,'..','X-ray_데이터셋_통합'))
SPLITS={'val':(f'{DATA}/검증','decoys_val.json'),'test':(f'{DATA}/테스트','decoys_test.json'),
        'val_normal':(f'{ROOT}/eval_sets/검증_정상합성',None),'test_normal':(f'{ROOT}/eval_sets/테스트_정상합성',None)}
def load_gt(split):
    d,_=SPLITS[split]; gt={}
    for f in sorted(glob.glob(f'{d}/labels/*.txt')):
        s=os.path.basename(f)[:-4]; W,H=Image.open(f'{d}/images/{s}.png').size; b=[]
        for l in open(f):
            if l.strip():
                c,x,y,w,h=map(float,l.split()); b.append([(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H])
        gt[s]=np.array(b).reshape(-1,4)
    return gt
def load_decoys(split):
    _,j=SPLITS[split]
    if not j: return {}
    dj=json.load(open(f'{DATA}/metadata/원본_README/{j}'))
    return {k[:-4]:np.array([[x,y,x+w,y+h] for x,y,w,h in v]) for k,v in dj.items()}
def load_preds(path,split):
    p={}
    for r in csv.DictReader(open(path,encoding='utf-8-sig')):
        if r['split']!=split: continue
        p.setdefault(r['image'],[]).append([float(r[k]) for k in ('x1','y1','x2','y2','score')])
    return {k:np.array(v) for k,v in p.items()}
def iou(a,b):
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)))
    x1=np.maximum(a[:,None,0],b[None,:,0]); y1=np.maximum(a[:,None,1],b[None,:,1])
    x2=np.minimum(a[:,None,2],b[None,:,2]); y2=np.minimum(a[:,None,3],b[None,:,3])
    i=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None)
    aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]); bb=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return i/(aa[:,None]+bb[None,:]-i+1e-9)
def match(gt,pred,thr_iou,min_score=-1):
    """greedy by score; returns list of (score, is_tp) and number of gt"""
    out=[]; ngt=0
    for s,g in gt.items():
        ngt+=len(g); p=pred.get(s,np.zeros((0,5))); p=p[p[:,4]>=min_score] if len(p) else p
        if len(p)==0: continue
        p=p[np.argsort(-p[:,4])]; M=iou(p[:,:4],g); used=np.zeros(len(g),bool)
        for i in range(len(p)):
            j=-1
            if len(g):
                cand=np.where((M[i]>=thr_iou)&~used)[0]
                if len(cand): j=cand[np.argmax(M[i,cand])]
            if j>=0: used[j]=True; out.append((p[i,4],1))
            else: out.append((p[i,4],0))
    return out,ngt
def ap(gt,pred,thr_iou):
    m,ngt=match(gt,pred,thr_iou)
    if not m or ngt==0: return 0.0
    m=sorted(m,key=lambda t:-t[0]); tp=np.cumsum([t[1] for t in m]); fp=np.cumsum([1-t[1] for t in m])
    rec=tp/ngt; prec=tp/np.maximum(tp+fp,1e-9)
    prec=np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean([prec[rec>=r].max() if (rec>=r).any() else 0 for r in np.linspace(0,1,101)]))
def counts_at(gt,pred,thr,thr_iou=0.5):
    m,ngt=match(gt,pred,thr_iou,thr); tp=sum(t[1] for t in m); fp=len(m)-tp
    return tp,fp,ngt-tp
def f1(tp,fp,fn): return 2*tp/max(2*tp+fp+fn,1e-9)
def best_threshold(gt,pred):
    scores=sorted({float(x[4]) for v in pred.values() for x in v})
    best=(-1,0.5)
    for t in scores:
        v=f1(*counts_at(gt,pred,t))
        if v>best[0]+1e-12: best=(v,t)
    return best[1]
def decoy_hits(pred,decoys,thr):
    n=0
    for s,d in decoys.items():
        p=pred.get(s,np.zeros((0,5)));p=p[p[:,4]>=thr] if len(p) else p
        if len(p)==0 or len(d)==0: continue
        cx=(p[:,0]+p[:,2])/2; cy=(p[:,1]+p[:,3])/2
        n+=int(sum(((cx>=x1)&(cx<=x2)&(cy>=y1)&(cy<=y2)).any() for x1,y1,x2,y2 in d))
    return n, int(sum(len(d) for d in decoys.values()))
def img_max(pred,names): return np.array([pred[s][:,4].max() if s in pred and len(pred[s]) else 0.0 for s in names])
def auroc(pos,neg):
    s=np.concatenate([pos,neg]); y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    o=np.argsort(s,kind='mergesort'); r=np.empty(len(s)); 
    # average ranks for ties
    ss=s[o]; i=0; rk=np.arange(1,len(s)+1,dtype=float)
    while i<len(s):
        j=i
        while j+1<len(s) and ss[j+1]==ss[i]: j+=1
        rk[i:j+1]=(i+j+2)/2; i=j+1
    r[o]=rk; return float((r[y==1].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def evaluate(pred_csv, boot=1000, fixed_thr=None):
    R={}; P={sp:load_preds(pred_csv,sp) for sp in SPLITS}; G={sp:load_gt(sp) for sp in SPLITS}
    thr=fixed_thr if fixed_thr is not None else best_threshold(G['val'],P['val'])
    R['threshold_from_val']=thr
    for sp in ['val','test']:
        g,p=G[sp],P[sp]; tp,fp,fn=counts_at(g,p,thr)
        dh,dn=decoy_hits(p,load_decoys(sp),thr)
        pos=img_max(p,list(g)); neg=img_max(P[sp+'_normal'],list(G[sp+'_normal']))
        R[sp]=dict(AP50=ap(g,p,0.5),AP50_95=float(np.mean([ap(g,p,t) for t in np.arange(0.5,0.96,0.05)])),
            TP=tp,FP=fp,FN=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),F1=f1(tp,fp,fn),
            decoy_FP=dh,decoys=dn,image_detect_rate=float((pos>=thr).mean()),
            normal_false_alarm_rate=float((neg>=thr).mean()) if len(neg) else None,
            normal_FP_boxes=int(sum((v[:,4]>=thr).sum() for v in P[sp+'_normal'].values())),
            image_AUROC=auroc(pos,neg) if len(neg) else None, n_images=len(g), n_normal=len(neg))
    if boot:
        rng=np.random.default_rng(0); names=list(G['test']); f=[];a=[]
        for _ in range(boot):
            pick=rng.choice(len(names),len(names)); gg={};pp={}
            for k,i in enumerate(pick):
                key=f'{names[i]}#{k}'; gg[key]=G['test'][names[i]]
                if names[i] in P['test']: pp[key]=P['test'][names[i]]
            f.append(f1(*counts_at(gg,pp,thr))); a.append(ap(gg,pp,0.5))
        R['test']['F1_CI95']=[float(np.percentile(f,2.5)),float(np.percentile(f,97.5))]
        R['test']['AP50_CI95']=[float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]
    return R
if __name__=='__main__':
    ap_=argparse.ArgumentParser(); ap_.add_argument('pred_csv'); ap_.add_argument('--out'); ap_.add_argument('--boot',type=int,default=1000)
    a=ap_.parse_args(); r=evaluate(a.pred_csv,a.boot)
    s=json.dumps(r,ensure_ascii=False,indent=1); print(s)
    if a.out: open(a.out,'w').write(s)
