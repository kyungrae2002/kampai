import csv,os,hashlib,unicodedata,re,collections,json,datetime,glob
import numpy as np,cv2
K=os.path.expanduser('~/mnt/kamp_ai'); RAW=K+'/dataset/dataset/test1/yolov3'
EXP=K+'/X-ray_실험'
rep=list(csv.DictReader(open('8_검증_테스트_설정파일_README/X-ray_데이터셋/split_report.csv',encoding='utf-8-sig')))
info={}
for r in rep:
    p=r['source'].replace('/content/work/src',RAW)
    if not os.path.exists(p): p=unicodedata.normalize('NFD',p)
    raw=open(p,'rb').read()
    m=re.search(r'_(\d{8})_(\d{6})',r['name']); t=datetime.datetime.strptime(m.group(1)+m.group(2),'%Y%m%d%H%M%S')
    info[r['name']]=dict(split=r['split'].replace('train_pseudo','train'),hogi=r['hogi'],group=r['group'],sha=hashlib.sha256(raw).hexdigest(),path=p,t=t,label=r['label'])
# 1) which raw images feed the ACTUAL training lists (D + F)
used=set(); unknown=[]
names=set(info)
for lst in ['ablation/F_전체+동료합성.txt','train_list.txt']:
    for l in open(f'{EXP}/{lst}'):
        s=os.path.basename(l.strip())[:-4]
        if not s: continue
        if s.endswith('_col08'): continue
        base=re.sub(r'(_syn_random\d|_syn_edge_rot\d|_clean|_partial_[ox]{3}|_weak)$','',s)
        if base in names: used.add(base)
        else: unknown.append(s)
# colleague subset parents: map via curated output id -> sha
oid2sha={r['output_id']:r['sha256'] for r in csv.DictReader(open(K+'/processed/processing_log.csv',encoding='utf-8-sig'))}
sha2name={v['sha']:n for n,v in info.items()}
col_parents=set()
for f in glob.glob(f'{EXP}/extra_colleague_08/images/*.png'):
    oid=os.path.basename(f).split('__')[0]+'__'+os.path.basename(f).split('__')[1]
    col_parents.add(sha2name.get(oid2sha.get(oid),'UNMAPPED:'+oid))
print('train-list base images:',len(used),'unknown stems:',len(unknown),unknown[:3])
print('used by split:',collections.Counter(info[n]['split'] for n in used))
print('colleague parents by split:',collections.Counter(info[n]['split'] if n in info else 'unmapped' for n in col_parents))
ev=[n for n,v in info.items() if v['split'] in('val','test')]
trn=[n for n in used|{n for n in col_parents if n in info}]
# exact raw duplicate
trsha={info[n]['sha'] for n in trn}
print('val/test raw sha found in training raws:',sum(info[n]['sha'] in trsha for n in ev))
# 2) near-duplicate: NCC on 64x64 grayscale
def vec(n):
    g=cv2.imread(info[n]['path'],0); g=cv2.resize(g,(64,64),interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    g-=g.mean(); return g/(np.linalg.norm(g)+1e-9)
V={n:vec(n) for n in set(ev)|set(trn)}
res=[]
for h in '123':
    tr=[n for n in trn if info[n]['hogi']==h]; T=np.stack([V[n] for n in tr])
    for n in [x for x in ev if info[x]['hogi']==h]:
        s=T@V[n]; i=int(np.argmax(s)); dt=min(abs((info[n]['t']-info[m]['t']).total_seconds()) for m in tr)
        res.append((n,info[n]['split'],h,float(s[i]),tr[i],dt))
# baseline: within-train, nearest neighbour in a DIFFERENT session
base=[]
for h in '123':
    tr=[n for n in trn if info[n]['hogi']==h]; T=np.stack([V[n] for n in tr])
    for idx in range(0,len(tr),7):
        n=tr[idx]; s=T@V[n]; mask=np.array([ (info[m]['group']==info[n]['group']) for m in tr]); s[mask]=-1
        base.append(float(s.max()))
# within-eval session redundancy
wi=[]
for sp in ('val','test'):
    g=collections.defaultdict(list)
    for n in ev:
        if info[n]['split']==sp: g[(info[n]['hogi'],info[n]['group'])].append(n)
    sizes=[len(v) for v in g.values()]
    for v in g.values():
        for a in range(len(v)):
            for b in range(a+1,len(v)): wi.append(float(V[v[a]]@V[v[b]]))
    print(sp,'sessions',len(g),'images per session: min/med/max',min(sizes),int(np.median(sizes)),max(sizes))
a=np.array([r[3] for r in res]); b=np.array(base); w=np.array(wi)
print('eval->train max NCC: med %.4f p10 %.4f p90 %.4f max %.4f'%(np.median(a),np.percentile(a,10),np.percentile(a,90),a.max()))
print('train->other-session train max NCC (baseline): med %.4f p10 %.4f p90 %.4f'%(np.median(b),np.percentile(b,10),np.percentile(b,90)))
print('within eval same-session NCC: med %.4f'%np.median(w))
dts=np.array([r[5] for r in res]); print('time gap eval->nearest train frame (same hogi) sec: min %.0f p10 %.0f med %.0f'%(dts.min(),np.percentile(dts,10),np.median(dts)))
print('eval frames with a training frame within 5 min:',int((dts<300).sum()),'within 1 h:',int((dts<3600).sum()))
top=sorted(res,key=lambda r:-r[3])[:5]; print('most similar pairs',[(r[0],r[4],round(r[3],4),int(r[5])) for r in top])
# dates
for sp in ('train','val','test'):
    ds=sorted({info[n]['t'].date() for n in info if info[n]['split']==sp}); print(sp,'dates',len(ds),ds[0],ds[-1])
json.dump([dict(name=r[0],split=r[1],hogi=r[2],max_ncc=r[3],nearest_train=r[4],nearest_train_time_gap_s=r[5]) for r in res],open('leak_ncc.json','w'),ensure_ascii=False,indent=0)
