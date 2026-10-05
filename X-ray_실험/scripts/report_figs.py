import os, sys, json, glob, csv
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from common import *
plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'savefig.dpi': 200, 'axes.unicode_minus': False})
FIG = f'{ROOT}/report/figs'; os.makedirs(FIG, exist_ok=True)
C = {'rfdetr_s': '#1f7a74', 'yolo11s': '#c47a12', 'yolov8n': '#5b6670', 'qwen25vl3b_lora': '#7a4fa0', 'qwen25vl3b_zeroshot': '#b9a6cc'}
N = {'rfdetr_s': 'RF-DETR-S', 'yolo11s': 'YOLO11s', 'yolov8n': 'YOLOv8n', 'qwen25vl3b_lora': 'Qwen2.5-VL LoRA', 'qwen25vl3b_zeroshot': 'Qwen2.5-VL 제로샷'}
S1 = json.load(open(f'{ROOT}/analysis_stage1.json')); RA = json.load(open(f'{ROOT}/report/report_analysis.json'))
def gray3(p): return cv2.cvtColor(cv2.imread(p, 0), cv2.COLOR_GRAY2RGB)
def boxes(lp, W, H):
    out = []
    for l in open(lp):
        if l.strip(): c, x, y, w, h = map(float, l.split()); out.append(((x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H))
    return out
def draw(im, bs, col, t=1):
    for x1, y1, x2, y2 in bs: cv2.rectangle(im, (int(x1)-2, int(y1)-2), (int(x2)+2, int(y2)+2), col, t)
    return im
# 1 pipeline
name = 'h1_002_20200622_203053(2)'
raw = glob.glob('/mnt/user-data/uploads/kamp_ai/dataset/**/002_20200622_203053(2).bmp', recursive=True)[0]
r = cv2.cvtColor(cv2.imread(raw), cv2.COLOR_BGR2RGB); pp = f'{DATA}/학습/1_원본영상/images/{name}.png'; pr = gray3(pp); H, W = pr.shape[:2]
diff = np.abs(cv2.cvtColor(r, cv2.COLOR_RGB2GRAY).astype(int) - pr[:, :, 0].astype(int)); dm = (diff > 0).astype(np.uint8)
ov = pr.copy(); ov[dm > 0] = (230, 120, 20)
lab = draw(pr.copy(), boxes(f'{DATA}/학습/1_원본영상/labels/{name}.txt', W, H), (0, 170, 80))
fig, ax = plt.subplots(1, 4, figsize=(10, 2.9))
for a, im, t in zip(ax, [r, pr, ov, lab], ['① 원본 (장비 색 박스 표시)', '② 색 박스 복원 + 미끼 박스 + 잡음', '③ 수정된 픽셀 (주황): 실제 3 + 미끼 3', '④ 정답 라벨 (사람)']):
    a.imshow(im[60:290, 110:330]); a.set_title(t, fontsize=8.5, fontweight='normal'); a.axis('off')
plt.tight_layout(); plt.savefig(f'{FIG}/f1_pipeline.png'); plt.close()
# 2 methods for one parent
cands = {}
for f in glob.glob(f'{DATA}/학습/5_이물일부제거/images/*.png'):
    s = os.path.basename(f)[:-4]; parent = s.rsplit('_partial_', 1)[0]; cands[parent] = f
par = sorted(cands)[5]
items = [('원본', f'{DATA}/학습/1_원본영상', par), ('합성: 제품 안 임의 위치', f'{DATA}/학습/2_이물합성_제품안임의위치', par+'_syn_random0'),
         ('합성: 경계 강화+모양 변형', f'{DATA}/학습/3_이물합성_경계강화+모양변형', par+'_syn_edge_rot0'), ('이물 전부 제거 (정상)', f'{DATA}/학습/4_이물전부제거_정상영상', par+'_clean'),
         ('이물 일부 제거', f'{DATA}/학습/5_이물일부제거', os.path.basename(cands[par])[:-4])]
fig, ax = plt.subplots(1, 5, figsize=(10, 2.5))
for a, (t, d, s) in zip(ax, items):
    im = gray3(f'{d}/images/{s}.png'); H, W = im.shape[:2]; im = draw(im, boxes(f'{d}/labels/{s}.txt', W, H), (220, 40, 40))
    ys, xs = np.where(im[:, :, 0] < 150); y0, y1, x0, x1 = max(0, ys.min()-10), ys.max()+10, max(0, xs.min()-10), xs.max()+10
    a.imshow(im[y0:y1, x0:x1]); a.set_title(t, fontsize=8, fontweight='normal'); a.axis('off')
plt.tight_layout(); plt.savefig(f'{FIG}/f2_methods.png'); plt.close()
# 3 composition
comp = [('1 원본(사람 라벨)', 340, '실제'), ('6 약라벨 실제 영상', 1880, '실제'), ('2 이물 합성: 임의 위치', 680, '합성'), ('3 이물 합성: 경계·변형', 680, '합성'),
        ('4 이물 전부 제거', 340, '합성'), ('5 이물 일부 제거', 452, '합성')]
fig, ax = plt.subplots(figsize=(7, 2.4))
y = np.arange(len(comp))[::-1]
ax.barh(y, [c[1] for c in comp], color=['#1f7a74' if c[2] == '실제' else '#c9b38a' for c in comp])
for yi, c in zip(y, comp): ax.text(c[1]+20, yi, f'{c[1]:,}', va='center', fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([c[0] for c in comp]); ax.set_xlabel('영상 수 (학습 4,372장)'); ax.set_xlim(0, 2150)
ax.set_title('학습 데이터 구성 (청록: 실제 촬영, 황토: 원본 340장에서 만든 합성)', fontweight='normal')
plt.tight_layout(); plt.savefig(f'{FIG}/f3_composition.png'); plt.close()
# 4 model compare
ms = ['rfdetr_s', 'yolo11s', 'yolov8n', 'qwen25vl3b_lora', 'qwen25vl3b_zeroshot']
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
for a, sp in zip(ax[:2], ['val', 'test']):
    mets = [('재현율', 'recall'), ('정밀도', 'precision'), ('mAP50', 'AP50'), ('mAP50-95', 'AP50_95')]; x = np.arange(len(mets)); w = 0.16
    for i, m in enumerate(ms):
        v = [S1[m]['eval'][sp][k] for _, k in mets]; a.bar(x + (i-2)*w, v, w, color=C[m], label=N[m])
    a.set_xticks(x); a.set_xticklabels([m[0] for m in mets]); a.set_ylim(0, 1.05); a.set_title('검증 80장' if sp == 'val' else '테스트 80장')
fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=5, fontsize=7.5, frameon=False); plt.subplots_adjust(bottom=0.2)
a = ax[2]; x = np.arange(3); w = 0.35
for i, sp in enumerate(['val', 'test']):
    fn = [S1[m]['eval'][sp]['FN'] for m in ms[:3]]; a.bar(x + (i-0.5)*w, fn, w, color=[C[m] for m in ms[:3]], alpha=1 if sp == 'val' else 0.55)
    for xi, v in zip(x + (i-0.5)*w, fn): a.text(xi, v+0.2, str(v), ha='center', fontsize=8)
a.set_xticks(x); a.set_xticklabels([N[m] for m in ms[:3]]); a.set_title('미탐(FN) 수: 진한색 검증 / 연한색 테스트')
plt.tight_layout(rect=(0,0.08,1,1)); plt.savefig(f'{FIG}/f4_models.png'); plt.close()
# 5 threshold sweep
fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
for a, m in zip(ax, ms[:3]):
    sw = S1[m]['sweep']; t = [s['thr'] for s in sw]
    a.plot(t, [s['val']['FN'] for s in sw], color='#c2410c', lw=2, label='FN(미탐)')
    a.plot(t, [s['val']['FP'] for s in sw], color='#2f6f9f', lw=2, ls='--', label='FP(오탐)')
    a.plot(t, [s['val']['normal_FA']*100 for s in sw], color='#777', lw=1.2, ls=':', label='정상 오경보율(%)')
    a.axvline(S1[m]['thr'], color=C[m], lw=1.2); a.set_ylim(0, 60); a.set_title(N[m]); a.set_xlabel('confidence 임계값')
ax[0].set_ylabel('검증셋 개수 / %'); ax[0].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(f'{FIG}/f5_threshold.png'); plt.close()
# 6 failure by factor
fac = ['크기', '대비', '배경 밝기', '호기']
fig, ax = plt.subplots(1, 4, figsize=(10, 2.6), sharey=True)
for a, f in zip(ax, fac):
    rows = [r for r in RA['fail'][f] if r['n'] > 0]; x = np.arange(len(rows)); w = 0.26
    for i, m in enumerate(ms[:3]):
        a.bar(x + (i-1)*w, [r[m]['FN'] for r in rows], w, color=C[m], label=N[m])
    a.set_xticks(x); a.set_xticklabels([f"{r['level']}\n(n={r['n']})" for r in rows], fontsize=7.5); a.set_title(f)
ax[0].set_ylabel('FN (검증+테스트)'); ax[0].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(f'{FIG}/f6_failure.png'); plt.close()
# 7 stress
fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), gridspec_kw={'width_ratios': [1, 1, 1.25]})
for a, key, t in [(ax[0], 'center_recall', '이물 중심 적중률'), (ax[1], 'recall', '재현율 (IoU≥0.5)')]:
    for m in ms[:3]:
        for sp, ls in [('val', '-'), ('test', '--')]:
            rr = [r for r in RA['stress'][m] if r['split'] == sp]
            a.plot([r['k'] for r in rr], [r[key] for r in rr], ls, color=C[m], marker='o', ms=3, label=f'{N[m]} {"검증" if sp == "val" else "테스트"}')
    a.set_xlabel('이물 대비 배율 k (1.0 = 원본)'); a.set_title(t); a.invert_xaxis(); a.set_ylim(0, 1.03)
ax[0].legend(fontsize=6.5, frameon=False, loc='lower left')
s0 = sorted(G := E.load_gt('val'))[16]; g = G[s0][0]; tiles = []
for kk in ['', '075', '050', '035', '025']:
    p = f'{EVAL_DIRS["val"]}/images/{s0}.png' if not kk else f'{ROOT}/eval_sets/stress_val_k{kk}/images/{s0}.png'
    im = cv2.imread(p, 0); cx, cy = int((g[0]+g[2])/2), int((g[1]+g[3])/2); c = im[cy-18:cy+18, cx-18:cx+18]; tiles.append(cv2.resize(c, (90, 90), interpolation=cv2.INTER_NEAREST))
ax[2].imshow(np.hstack(tiles), cmap='gray', vmin=0, vmax=255); ax[2].axis('off'); ax[2].set_title('k = 1.0 / 0.75 / 0.5 / 0.35 / 0.25 (평가 전용 합성)', fontweight='normal', fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIG}/f7_stress.png'); plt.close()
# 8 FN gallery (RF-DETR, YOLO11s)
thr = RA['t_high']; tiles = []
for d in RA['fn_detail']:
    if d['model'] not in ('rfdetr_s', 'yolo11s'): continue
    sp = d['split']; im = gray3(f"{EVAL_DIRS[sp]}/images/{d['image']}.png"); H, W = im.shape[:2]
    gt = G2 = E.load_gt(sp)[d['image']]; pr = E.load_preds(f"{PRED}/{d['model']}.csv", sp).get(d['image'], np.zeros((0, 5))); pr = pr[pr[:, 4] >= thr[d['model']]]
    for b in gt: cv2.rectangle(im, (int(b[0]), int(b[1])), (int(np.ceil(b[2])), int(np.ceil(b[3]))), (0, 190, 80), 1)
    for b in pr: cv2.rectangle(im, (int(b[0]), int(b[1])), (int(np.ceil(b[2])), int(np.ceil(b[3]))), (230, 40, 40), 1)
    # crop around the missed GT (closest by size match)
    gsel = min(gt, key=lambda b: abs((b[2]-b[0]) - d['gt_w']) + abs((b[3]-b[1]) - d['gt_h']))
    cx, cy = int((gsel[0]+gsel[2])/2), int((gsel[1]+gsel[3])/2); c = im[max(0, cy-20):cy+20, max(0, cx-20):cx+20]
    c = cv2.resize(c, (120, 120), interpolation=cv2.INTER_NEAREST)
    tiles.append((c, f"{N[d['model']]} · {'검증' if sp == 'val' else '테스트'}\nGT {d['gt_w']:.0f}×{d['gt_h']:.0f}px · IoU {d['best_iou']:.3f}"))
n = len(tiles); cols = 6; rows_ = int(np.ceil(n/cols))
fig, ax = plt.subplots(rows_, cols, figsize=(10, 2.0*rows_))
for i, a in enumerate(np.array(ax).ravel()):
    a.axis('off')
    if i < n: a.imshow(tiles[i][0]); a.set_title(tiles[i][1], fontsize=6.5, fontweight='normal')
plt.tight_layout(); plt.savefig(f'{FIG}/f8_fn_gallery.png'); plt.close()
print('FN tiles', n)
# 9 VLM examples
raw = {(r['image'], r['split']): r['text'] for r in json.load(open(f'{PRED}/qwen25vl3b_lora_raw.json'))}
zs = E.load_preds(f'{PRED}/qwen25vl3b_zeroshot.csv', 'val'); lo = E.load_preds(f'{PRED}/qwen25vl3b_lora.csv', 'val'); G = E.load_gt('val')
picks = sorted(G)[::27][:3]
fig, ax = plt.subplots(1, 6, figsize=(10, 2.1))
for i, s in enumerate(picks):
    for j, (pp_, t) in enumerate([(zs, '제로샷'), (lo, 'LoRA')]):
        im = gray3(f"{EVAL_DIRS['val']}/images/{s}.png")
        for b in G[s]: cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 190, 80), 1)
        for b in pp_.get(s, []): cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (230, 40, 40), 1)
        ys, xs = np.where(im[:, :, 0] < 150); a = ax[2*i+j]
        a.imshow(im[max(0, ys.min()-5):ys.max()+5, max(0, xs.min()-5):xs.max()+5]); a.axis('off'); a.set_title(f'{t} #{i+1}', fontsize=8, fontweight='normal')
plt.tight_layout(); plt.savefig(f'{FIG}/f9_vlm.png'); plt.close()
print('done')
