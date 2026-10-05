# 검증·테스트 영상의 라벨된 이물을 지운 '정상 합성' 평가 세트 생성 (영상 단위 오경보 측정용, 학습에 사용 금지)
import os,glob,cv2,numpy as np,hashlib,json,sys
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','X-ray_데이터셋_통합')
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','eval_sets')
def hp(g): g=g.astype(np.float32); return g-cv2.GaussianBlur(g,(5,5),1.2)
stats=[]
for sp in ['검증','테스트']:
    o=f'{OUT}/{sp}_정상합성'; os.makedirs(o+'/images',exist_ok=True); os.makedirs(o+'/labels',exist_ok=True)
    for f in sorted(glob.glob(f'{D}/{sp}/labels/*.txt')):
        s=os.path.basename(f)[:-4]; g=cv2.imread(f'{D}/{sp}/images/{s}.png',0); H,W=g.shape
        rng=np.random.default_rng(int(hashlib.md5(s.encode()).hexdigest()[:8],16))
        mask=np.zeros_like(g,np.uint8)
        for l in open(f):
            c,x,y,w,h=map(float,l.split()); x1=max(0,int(np.floor((x-w/2)*W))-2); x2=min(W,int(np.ceil((x+w/2)*W))+2)
            y1=max(0,int(np.floor((y-h/2)*H))-2); y2=min(H,int(np.ceil((y+h/2)*H))+2); mask[y1:y2,x1:x2]=1
        # Telea 인페인팅(주변 슬롯·제품 밝기로 메움) + 제품 잡음 수준의 노이즈 재주입 (모폴로지 닫힘은 슬롯 끝이 밝게 떠서 폐기)
        out=cv2.inpaint(g,mask,5,cv2.INPAINT_TELEA).astype(np.float32)
        ring=(cv2.dilate(mask,np.ones((7,7),np.uint8))>0)&(mask==0); m=mask>0
        sd=hp(g)[ring].std(); out[m]+=rng.normal(0,sd*0.6,m.sum())
        stats.append(float(out[m].mean()-out[ring].mean()))
        res=np.clip(np.round(out),0,255).astype(np.uint8)
        cv2.imwrite(f'{o}/images/{s}_normal.png',res); open(f'{o}/labels/{s}_normal.txt','w').close()
print('patch-ring mean diff: median %.2f, p90 %.2f'%(np.median(stats),np.percentile(stats,90)), 'n',len(stats))
