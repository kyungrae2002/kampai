import os, csv, glob
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,'..'))
DATA=os.path.abspath(os.path.join(ROOT,'..','X-ray_데이터셋_통합'))
TRAIN_DIRS=[f'{DATA}/학습/{m}' for m in ['1_원본영상','2_이물합성_제품안임의위치','3_이물합성_경계강화+모양변형','4_이물전부제거_정상영상','5_이물일부제거','6_약라벨_추가영상']]
EVAL_DIRS={'val':f'{DATA}/검증','test':f'{DATA}/테스트','val_normal':f'{ROOT}/eval_sets/검증_정상합성','test_normal':f'{ROOT}/eval_sets/테스트_정상합성'}
PRED=f'{ROOT}/preds'; RUNS=f'{ROOT}/runs'; LOGS=f'{ROOT}/logs'
for d in (PRED,RUNS,LOGS): os.makedirs(d,exist_ok=True)
_EX=f'{ROOT}/excluded_train_images.csv'
EXCLUDE={r['stem'] for r in csv.DictReader(open(_EX,encoding='utf-8-sig'))} if os.path.exists(_EX) else set()
def images(d): return sorted(p for p in glob.glob(f'{d}/images/*.png') if os.path.splitext(os.path.basename(p))[0] not in EXCLUDE)
def stem(p): return os.path.splitext(os.path.basename(p))[0]
def write_preds(path, rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['image','split','x1','y1','x2','y2','score']); w.writerows(rows)
DEVICE=os.environ.get('XRAY_DEVICE','mps')
