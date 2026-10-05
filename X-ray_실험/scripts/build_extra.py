# 기존 원본 BMP의 색 박스를 약라벨로 사용해, 새 분할의 학습 세션(train_pseudo) 영상만 통합본과 같은 방식으로 가공
import hashlib,csv,os,unicodedata,cv2,numpy as np,sys,json
sys.path.insert(0,os.path.dirname(__file__)); from xproc import *
K=os.path.expanduser('~/mnt/kamp_ai'); RAW=K+'/dataset/dataset/test1/yolov3'
META=os.path.expanduser('~/xw/8_검증_테스트_설정파일_README/X-ray_데이터셋/split_report.csv')
OUT=K+'/X-ray_데이터셋_통합/학습/6_약라벨_추가영상'
SHRINK=8; MIN=5; SIGMA=0.6
os.makedirs(OUT+'/images',exist_ok=True); os.makedirs(OUT+'/labels',exist_ok=True)
rows=[r for r in csv.DictReader(open(META,encoding='utf-8-sig')) if r['split']=='train_pseudo']
EXCL=set(l.strip() for l in open(os.path.join(os.path.dirname(__file__),'excluded_no_box.txt')))
rows=[r for r in rows if r['name'] not in EXCL]
start,end=int(sys.argv[1]),int(sys.argv[2])
man=[]
for r in rows[start:end]:
    stem=r['name']+'_weak'
    rng=np.random.default_rng(int(hashlib.md5(r['name'].encode()).hexdigest()[:8],16))
    p=r['source'].replace('/content/work/src',RAW)
    if not os.path.exists(p): p=unicodedata.normalize('NFD',p)
    raw=cv2.imread(p); H,W=raw.shape[:2]
    img,comps,full,dec=process(raw,rng,SIGMA)
    lines=[]
    for x,y,w,h in comps:
        cx,cy=x+w/2,y+h/2; w2,h2=max(MIN,w-SHRINK),max(MIN,h-SHRINK)
        lines.append(f'0 {cx/W:.6f} {cy/H:.6f} {w2/W:.6f} {h2/H:.6f}')
    cv2.imwrite(f'{OUT}/images/{stem}.png',img)
    open(f'{OUT}/labels/{stem}.txt','w').write('\n'.join(lines)+('\n' if lines else ''))
print('done',start,end)
