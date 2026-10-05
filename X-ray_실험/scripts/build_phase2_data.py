"""2단계 실험 데이터 준비 (맥 학습 전에 1회 실행, 파일만 생성)
- ablation/<실험>.yaml + 학습 목록: A(실제만) B(+이물합성) C(+이물제거) D(전체) F(D+동료 08 조합 부분집합)
- 동료 08(상·중·하 유지/제거 조합): 원본이 새 분할 '학습' 세션인 것만, 원본당 1개 조합, 흑백 변환, 라벨 8px 축소
- 스트레스 평가셋: 검증·테스트 실제 이물의 대비를 k배로 낮춘 영상 (평가 전용, 학습 금지)"""
import os, sys, csv, glob, hashlib, unicodedata, random, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
MAC_KAMP = '/Users/kyungrae/Desktop/kamp_ai'
KAMP = os.path.abspath(os.path.join(ROOT, '..'))
def mac(p): return p.replace(KAMP, MAC_KAMP)
T = {os.path.basename(d): d for d in TRAIN_DIRS}
GROUPS = {
 'A_실제만':      ['1_원본영상', '6_약라벨_추가영상'],
 'B_실제+이물합성': ['1_원본영상', '6_약라벨_추가영상', '2_이물합성_제품안임의위치', '3_이물합성_경계강화+모양변형'],
 'C_실제+이물제거': ['1_원본영상', '6_약라벨_추가영상', '4_이물전부제거_정상영상', '5_이물일부제거'],
 'D_전체':        list(T),
}
# ---- 동료 08 부분집합 (F) ----
COL = f'{ROOT}/extra_colleague_08'
if not os.path.exists(f'{COL}/labels'):
    os.makedirs(f'{COL}/images', exist_ok=True); os.makedirs(f'{COL}/labels', exist_ok=True)
    RAW = f'{KAMP}/dataset/dataset/test1/yolov3'
    oid2sha = {r['output_id']: r['sha256'] for r in csv.DictReader(open(f'{KAMP}/processed/processing_log.csv', encoding='utf-8-sig'))}
    rep = f'{DATA}/metadata/원본_README/split_report.csv'
    sha2new = {}
    for r in csv.DictReader(open(rep, encoding='utf-8-sig')):
        p = r['source'].replace('/content/work/src', RAW)
        if not os.path.exists(p): p = unicodedata.normalize('NFD', p)
        sha2new[hashlib.sha256(open(p, 'rb').read()).hexdigest()] = r['split'].replace('train_pseudo', 'train')
    TP = f'{KAMP}/processed/three_position_combinations_v1'
    by = {}
    for r in csv.DictReader(open(f'{TP}/manifest.csv', encoding='utf-8-sig')):
        if sha2new.get(oid2sha.get(r['source_id'])) == 'train': by.setdefault(r['source_id'], []).append(r)
    rnd = random.Random(0); n = 0
    for sid, rs in sorted(by.items()):
        r = rnd.choice(rs); im = cv2.imread(f"{TP}/{r['image']}", cv2.IMREAD_GRAYSCALE); H, W = im.shape
        cstem = os.path.basename(r['image'])[:-4] + '_col08'
        lines = []
        for l in open(f"{TP}/{r['label']}"):
            if not l.strip(): continue
            c, x, y, w, h = map(float, l.split()); w2 = max(5, w*W-8)/W; h2 = max(5, h*H-8)/H
            lines.append(f'0 {x:.6f} {y:.6f} {w2:.6f} {h2:.6f}')
        cv2.imwrite(f'{COL}/images/{cstem}.png', im); open(f'{COL}/labels/{cstem}.txt', 'w').write('\n'.join(lines) + ('\n' if lines else '')); n += 1
    print('colleague 08 subset images:', n)
GROUPS['F_전체+동료합성'] = list(T) + ['__COL__']
for name, folders in GROUPS.items():
    paths = []
    for f in folders:
        d = COL if f == '__COL__' else T[f]
        paths += [mac(p) for p in images(d)]
    open(f'{ROOT}/ablation/{name}.txt', 'w').write('\n'.join(paths) + '\n')
    open(f'{ROOT}/ablation/{name}.yaml', 'w').write(f"path: {MAC_KAMP}/X-ray_데이터셋_통합\ntrain: {MAC_KAMP}/X-ray_실험/ablation/{name}.txt\nval: 검증/images\ntest: 테스트/images\nnames:\n  0: defect\n")
    print(name, len(paths))
# ---- 스트레스 평가셋 ----
KS = [0.75, 0.5, 0.35, 0.25]
for sp, sub in [('val', '검증'), ('test', '테스트')]:
    d = EVAL_DIRS[sp]
    for k in KS:
        o = f'{ROOT}/eval_sets/stress_{sp}_k{int(k*100):03d}'
        if os.path.exists(f'{o}/labels') and len(os.listdir(f'{o}/labels')) == 80: continue
        os.makedirs(f'{o}/images', exist_ok=True); os.makedirs(f'{o}/labels', exist_ok=True)
        for p in images(d):
            s = stem(p); g = cv2.imread(p, 0); H, W = g.shape; mask = np.zeros_like(g)
            for l in open(f'{d}/labels/{s}.txt'):
                c, x, y, w, h = map(float, l.split())
                x1=max(0,int((x-w/2)*W)-2); x2=min(W,int((x+w/2)*W)+3); y1=max(0,int((y-h/2)*H)-2); y2=min(H,int((y+h/2)*H)+3); mask[y1:y2, x1:x2] = 1
            bg = cv2.inpaint(g, mask, 5, cv2.INPAINT_TELEA).astype(np.float32)   # 이물이 없을 때의 배경 추정
            out = g.astype(np.float32); m = mask > 0
            out[m] = bg[m] + k * (out[m] - bg[m])                                    # 배경 대비 차이만 k배로 축소 (잡음 포함 축소)
            cv2.imwrite(f'{o}/images/{s}.png', np.clip(np.round(out), 0, 255).astype(np.uint8))
            open(f'{o}/labels/{s}.txt', 'w').write(open(f'{d}/labels/{s}.txt').read())
print('stress sets ready')
