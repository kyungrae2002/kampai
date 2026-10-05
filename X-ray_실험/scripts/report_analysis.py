"""보고서용 분석: 실패조건(크기·대비·위치·배경·형상·호기), 스트레스(대비 축소) 곡선, 3단계 판정, 세션 단위 신뢰구간"""
import os, sys, json, csv, collections
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from common import *
OUT = f'{ROOT}/report'; os.makedirs(OUT, exist_ok=True)
MODELS = {'rfdetr_s': 'RF-DETR-S', 'yolo11s': 'YOLO11s', 'yolov8n': 'YOLOv8n'}
EV = {k: json.load(open(f'{PRED}/{k}_eval.json')) for k in MODELS}
THR = {k: EV[k]['threshold_from_val'] for k in MODELS}
G = {sp: E.load_gt(sp) for sp in E.SPLITS}
P = {k: {sp: E.load_preds(f'{PRED}/{k}.csv', sp) for sp in E.SPLITS} for k in MODELS}
rep = {r['name']: r for r in csv.DictReader(open(f'{DATA}/metadata/원본_README/split_report.csv', encoding='utf-8-sig'))}
def hp(g): g = g.astype(np.float32); return g - cv2.GaussianBlur(g, (5, 5), 1.2)
def prodmask(g):
    b = cv2.GaussianBlur(g, (7, 7), 0); _, t = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(t); k = 1 + np.argmax(st[1:, 4]); return (lab == k).astype(np.uint8)
# ---------- per-GT attributes ----------
attrs = []
for sp in ('val', 'test'):
    d = EVAL_DIRS[sp]
    for s, gt in G[sp].items():
        g = cv2.imread(f'{d}/images/{s}.png', 0); H, W = g.shape; pm = prodmask(g)
        dist = cv2.distanceTransform(pm, cv2.DIST_L2, 5)
        for j, (x1, y1, x2, y2) in enumerate(gt):
            X1, Y1, X2, Y2 = int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))
            box = g[Y1:Y2, X1:X2].astype(np.float32)
            ring = np.zeros_like(g, bool); ring[max(0, Y1-6):Y2+6, max(0, X1-6):X2+6] = True; ring[Y1:Y2, X1:X2] = False
            bg = np.median(g[ring]); noise = hp(g)[ring].std() + 1e-6
            core = np.sort(box.ravel())[:max(1, box.size // 4)].mean()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            w, h = x2 - x1, y2 - y1
            attrs.append(dict(split=sp, image=s, j=j, hogi=s[1], session=f"{s[1]}-{rep[s]['group']}", size=float(np.sqrt(w * h)),
                              aspect=float(max(w, h) / max(1e-6, min(w, h))), contrast=float((bg - core) / noise), contrast_raw=float(bg - core),
                              bg=float(bg), edge_dist=float(dist[min(cy, H-1), min(cx, W-1)]), W=W))
def match_flags(k, sp, thr):
    """GT별: IoU0.5 매칭 여부, 중심 적중 여부, 최고 IoU, 매칭 점수"""
    out = {}
    for s, g in G[sp].items():
        p = P[k][sp].get(s, np.zeros((0, 5))); p = p[p[:, 4] >= thr] if len(p) else p
        p = p[np.argsort(-p[:, 4])] if len(p) else p
        M = E.iou(p[:, :4], g) if len(p) else np.zeros((0, len(g))); used = np.zeros(len(g), bool)
        for i in range(len(p)):
            c = np.where((M[i] >= 0.5) & ~used)[0]
            if len(c): used[c[np.argmax(M[i, c])]] = True
        for j, b in enumerate(g):
            ctr = False
            if len(p):
                cx = (p[:, 0] + p[:, 2]) / 2; cy = (p[:, 1] + p[:, 3]) / 2
                ctr = bool(((cx >= b[0]-2) & (cx <= b[2]+2) & (cy >= b[1]-2) & (cy <= b[3]+2)).any())
            out[(s, j)] = dict(tp=bool(used[j]), ctr=ctr, best_iou=float(M[:, j].max()) if len(p) else 0.0,
                               best_score=float(p[np.argmax(M[:, j]), 4]) if len(p) else 0.0)
    return out
F = {k: {sp: match_flags(k, sp, THR[k]) for sp in ('val', 'test')} for k in MODELS}
# bins (pooled val+test, tertiles computed on pooled GT)
def tert(key):
    v = np.array([a[key] for a in attrs]); q = np.percentile(v, [33.3, 66.7]); return q
QS, QC, QB = tert('size'), tert('contrast'), tert('bg')
def lab_size(a): return '작음' if a['size'] < QS[0] else ('중간' if a['size'] < QS[1] else '큼')
def lab_con(a): return '낮음' if a['contrast'] < QC[0] else ('중간' if a['contrast'] < QC[1] else '높음')
def lab_bg(a): return '어두운 배경' if a['bg'] < QB[0] else ('중간 배경' if a['bg'] < QB[1] else '밝은 배경')
def lab_edge(a): return '가장자리(<25px)' if a['edge_dist'] < 25 else '내부'
def lab_shape(a): return '길쭉함(≥1.4)' if a['aspect'] >= 1.4 else '정방형'
def lab_hogi(a): return f"{a['hogi']}호기"
FACT = [('크기', lab_size, ['작음', '중간', '큼']), ('대비', lab_con, ['낮음', '중간', '높음']), ('배경 밝기', lab_bg, ['어두운 배경', '중간 배경', '밝은 배경']),
        ('위치', lab_edge, ['가장자리(<25px)', '내부']), ('형상', lab_shape, ['정방형', '길쭉함(≥1.4)']), ('호기', lab_hogi, ['1호기', '2호기', '3호기'])]
fail = {}
for fname, fn, levels in FACT:
    rows = []
    for lv in levels:
        sel = [a for a in attrs if fn(a) == lv]; r = {'level': lv, 'n': len(sel)}
        for k in MODELS:
            fl = [F[k][a['split']][(a['image'], a['j'])] for a in sel]
            r[k] = {'FN': sum(not f['tp'] for f in fl), 'recall': float(np.mean([f['tp'] for f in fl])) if fl else None,
                    'center_recall': float(np.mean([f['ctr'] for f in fl])) if fl else None}
        rows.append(r)
    fail[fname] = rows
bins = {'size_px': QS.tolist(), 'contrast': QC.tolist(), 'bg': QB.tolist()}
# ---------- FN detail table ----------
fn_detail = []
for k in MODELS:
    for sp in ('val', 'test'):
        for a in [x for x in attrs if x['split'] == sp]:
            f = F[k][sp][(a['image'], a['j'])]
            if not f['tp']:
                g = G[sp][a['image']][a['j']]
                fn_detail.append(dict(model=k, split=sp, image=a['image'], gt_w=float(g[2]-g[0]), gt_h=float(g[3]-g[1]), best_iou=f['best_iou'], score=f['best_score'],
                                      center_hit=f['ctr'], size=lab_size(a), contrast=lab_con(a), edge=lab_edge(a), hogi=a['hogi']))
# ---------- stress curves ----------
stress = {}
for k in MODELS:
    rows = []
    for sp in ('val', 'test'):
        for kk in ['100', '075', '050', '035', '025']:
            split = sp if kk == '100' else f'stress_{sp}_k{kk}'
            preds = P[k][sp] if kk == '100' else E.load_preds(f'{PRED}/stress_{k}.csv', split)
            tp, fp, fn = E.counts_at(G[sp], preds, THR[k])
            ctr = 0; n = 0
            for s, g in G[sp].items():
                p = preds.get(s, np.zeros((0, 5))); p = p[p[:, 4] >= THR[k]] if len(p) else p
                for b in g:
                    n += 1
                    if len(p):
                        cx = (p[:, 0]+p[:, 2])/2; cy = (p[:, 1]+p[:, 3])/2
                        ctr += bool(((cx >= b[0]-2) & (cx <= b[2]+2) & (cy >= b[1]-2) & (cy <= b[3]+2)).any())
            img_rate = float(np.mean([(s in preds and len(preds[s]) and (preds[s][:, 4] >= THR[k]).any()) for s in G[sp]]))
            rows.append(dict(split=sp, k=int(kk)/100, FN=fn, FP=fp, recall=tp/(tp+fn), center_recall=ctr/n, image_detect=img_rate))
    stress[k] = rows
# ---------- session-level bootstrap CI (test) ----------
def sess_boot(k, sp='test', B=2000):
    sess = collections.defaultdict(list)
    for s in G[sp]: sess[f"{s[1]}-{rep[s]['group']}"].append(s)
    keys = list(sess); rng = np.random.default_rng(0); rec = []; f1s = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys)); gg = {}; pp = {}
        for t, i in enumerate(pick):
            for s in sess[keys[i]]:
                gg[f'{s}#{t}'] = G[sp][s]
                if s in P[k][sp]: pp[f'{s}#{t}'] = P[k][sp][s]
        tp, fp, fn = E.counts_at(gg, pp, THR[k]); rec.append(tp/max(1, tp+fn)); f1s.append(E.f1(tp, fp, fn))
    return dict(sessions=len(keys), recall_CI95=[float(np.percentile(rec, 2.5)), float(np.percentile(rec, 97.5))],
                F1_CI95=[float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))])
ci = {k: sess_boot(k) for k in MODELS}
# ---------- 3-tier decision (image level) ----------
# 규칙은 검증셋에서만 정함: T_high = 모델별 검증 F1 최대 임계값, T_low = 검증 정상합성 경보율이 10% 이하가 되는 가장 낮은 임계값(0.05 단위)
def max_score(k, sp, s):
    p = P[k][sp].get(s); return float(p[:, 4].max()) if p is not None and len(p) else 0.0
def tlow(k):
    for t in np.arange(0.05, 0.96, 0.05):
        neg = E.img_max(P[k]['val_normal'], list(G['val_normal']))
        if (neg >= t).mean() <= 0.10: return round(float(t), 2)
    return THR[k]
TL = {k: tlow(k) for k in MODELS}
def decide(sp, s, primary='rfdetr_s', second='yolo11s'):
    m1 = max_score(primary, sp, s); m2 = max_score(second, sp, s)
    if m1 >= THR[primary] and m2 >= THR[second]: return 'REJECT'
    if m1 >= THR[primary] or m2 >= THR[second]: return 'RE-INSPECTION'   # 두 모델 판단 불일치
    if m1 >= TL[primary] or m2 >= TL[second]: return 'RE-INSPECTION'      # 애매한 점수대
    return 'PASS'
def decide_single(k, sp, s):
    m = max_score(k, sp, s)
    return 'REJECT' if m >= THR[k] else ('RE-INSPECTION' if m >= TL[k] else 'PASS')
policy = {}
for name, fn in [('RF-DETR-S 단독', lambda sp, s: decide_single('rfdetr_s', sp, s)), ('YOLO11s 단독', lambda sp, s: decide_single('yolo11s', sp, s)),
                 ('RF-DETR-S + YOLO11s', decide)]:
    r = {}
    for sp in ('val', 'test'):
        r[sp] = dict(collections.Counter(fn(sp, s) for s in G[sp]))
        r[sp + '_normal'] = dict(collections.Counter(fn(sp + '_normal', s) for s in G[sp + '_normal']))
    policy[name] = r
json.dump(dict(bins=bins, fail=fail, fn_detail=fn_detail, stress=stress, session_ci=ci, t_low=TL, t_high=THR, policy=policy,
               n_gt=len(attrs)), open(f'{OUT}/report_analysis.json', 'w'), ensure_ascii=False, indent=1)
json.dump(attrs, open(f'{OUT}/gt_attributes.json', 'w'), ensure_ascii=False)
print('bins', bins); print('TL', TL, 'THR', THR); print(json.dumps(policy, ensure_ascii=False)); print(json.dumps(ci, ensure_ascii=False))
for fname in fail: print(fname, [(r['level'], r['n'], {k: r[k]['FN'] for k in MODELS}) for r in fail[fname]])
for k in MODELS: print(k, [(r['split'], r['k'], r['FN'], round(r['center_recall'], 3), round(r['image_detect'], 3)) for r in stress[k]])
