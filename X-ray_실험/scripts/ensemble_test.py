"""앙상블 테스트 확인: 검증에서 정한 방식·임계값을 그대로 테스트에 1회 적용 (재조정 없음)"""
import os, sys, json, csv, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from common import *
import ensemble_val as V   # 검증 결과(V.res)와 함수 재사용
G = E.load_gt('test'); GN = E.load_gt('test_normal')
rep = {r['name']: r for r in csv.DictReader(open(f'{DATA}/metadata/원본_README/split_report.csv', encoding='utf-8-sig'))}
def metrics(pred, thr, normal=None):
    tp, fp, fn = E.counts_at(G, pred, thr); ctr = 0; n = 0
    for s, g in G.items():
        p = pred.get(s, np.zeros((0, 5))); p = p[p[:, 4] >= thr] if len(p) else p
        for b in g:
            n += 1
            if len(p):
                cx = (p[:, 0]+p[:, 2])/2; cy = (p[:, 1]+p[:, 3])/2; ctr += bool(((cx >= b[0]-2) & (cx <= b[2]+2) & (cy >= b[1]-2) & (cy <= b[3]+2)).any())
    r = dict(TP=tp, FP=fp, FN=fn, precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn), F1=E.f1(tp, fp, fn), center_recall=ctr/n, AP50=E.ap(G, pred, 0.5),
             AP50_95=float(np.mean([E.ap(G, pred, t) for t in np.arange(0.5, 0.96, 0.05)])), decoy_FP=E.decoy_hits(pred, E.load_decoys('test'), thr)[0])
    if normal is not None: r['normal_FA'] = float((E.img_max(normal, list(GN)) >= thr).mean())
    # 세션 단위 부트스트랩 재현율 CI
    sess = collections.defaultdict(list)
    for s in G: sess[f"{s[1]}-{rep[s]['group']}"].append(s)
    keys = list(sess); rng = np.random.default_rng(0); rec = []
    for _ in range(2000):
        pick = rng.choice(len(keys), len(keys)); gg = {}; pp = {}
        for t, i in enumerate(pick):
            for s in sess[keys[i]]:
                gg[f'{s}#{t}'] = G[s]
                if s in pred: pp[f'{s}#{t}'] = pred[s]
        a, b, c = E.counts_at(gg, pp, thr); rec.append(a/max(1, a+c))
    r['recall_CI95'] = [float(np.percentile(rec, 2.5)), float(np.percentile(rec, 97.5))]
    return r
out = {}
for k in ('rfdetr_s', 'yolo11s', 'yolov8n'):
    thr = V.THR[k]; r = metrics(V.load(k, 'test'), thr, normal=V.load(k, 'test_normal')); r['thr'] = thr
    r['stress'] = {kk: metrics(V.load(k, f'stress_test_k{kk}', f'stress_{k}'), thr)['center_recall'] for kk in ('075', '050', '035')}
    out[k] = r
for key, vr in V.res.items():
    if '|' not in key: continue
    pair, mode = key.split('|'); a, b = pair.split('+'); thr = vr['thr']
    r = metrics(V.combine(a, b, mode, 'test'), thr, normal=V.combine(a, b, mode, 'test_normal')); r['thr'] = thr
    r['stress'] = {kk: metrics(V.combine(a, b, mode, f'stress_test_k{kk}'), thr)['center_recall'] for kk in ('075', '050', '035')}
    out[key] = r
json.dump({'val': V.res, 'test': out}, open(f'{ROOT}/report/ensemble_val_test.json', 'w'), ensure_ascii=False, indent=1)
for k, r in out.items():
    print(f"{k:28s} thr {r['thr']:.3f} FN {r['FN']:2d} FP {r['FP']:2d} R {r['recall']:.3f} P {r['precision']:.3f} AP50 {r['AP50']:.3f} AP95 {r['AP50_95']:.3f} ctr {r['center_recall']:.3f} nFA {r['normal_FA']:.3f} decoy {r['decoy_FP']} CI {r['recall_CI95'][0]:.3f}-{r['recall_CI95'][1]:.3f} | stress {r['stress']['075']:.3f} {r['stress']['050']:.3f} {r['stress']['035']:.3f}")
