import json, os, sys, numpy as np, glob
sys.path.insert(0, '.'); import evaluate as E
from common import *
EXPS = ['A_실제만', 'A_실제만_seed1', 'B_실제+이물합성', 'C_실제+이물제거', 'D_전체', 'D_전체_seed1', 'E_선택+960', 'F_전체+동료합성']
G = {sp: E.load_gt(sp) for sp in ('val', 'test')}
attrs = json.load(open(f'{ROOT}/report/gt_attributes.json')) if os.path.exists(f'{ROOT}/report/gt_attributes.json') else None
def flags(preds, gt, thr):
    out = {}
    for s, g in gt.items():
        p = preds.get(s, np.zeros((0, 5))); p = p[p[:, 4] >= thr] if len(p) else p
        p = p[np.argsort(-p[:, 4])] if len(p) else p
        M = E.iou(p[:, :4], g) if len(p) else np.zeros((0, len(g))); used = np.zeros(len(g), bool)
        for i in range(len(p)):
            c = np.where((M[i] >= 0.5) & ~used)[0]
            if len(c): used[c[np.argmax(M[i, c])]] = True
        for j, b in enumerate(g):
            ctr = False
            if len(p):
                cx = (p[:, 0]+p[:, 2])/2; cy = (p[:, 1]+p[:, 3])/2
                ctr = bool(((cx >= b[0]-2) & (cx <= b[2]+2) & (cy >= b[1]-2) & (cy <= b[3]+2)).any())
            out[(s, j)] = (bool(used[j]), ctr)
    return out
res = {}
for e in EXPS:
    f = f'{PRED}/p2_{e}.csv'; ev = json.load(open(f'{PRED}/p2_{e}_eval.json')); thr = ev['threshold_from_val']; r = {'eval': ev, 'meta': json.load(open(f'{PRED}/p2_{e}_meta.json'))}
    for sp in ('val', 'test'):
        fl = flags(E.load_preds(f, sp), G[sp], thr); r[sp+'_center'] = float(np.mean([v[1] for v in fl.values()]))
        st = {}
        for k in ('075', '050', '035'):
            fk = flags(E.load_preds(f, f'stress_{sp}_k{k}'), G[sp], thr)
            st[k] = dict(center=float(np.mean([v[1] for v in fk.values()])), recall=float(np.mean([v[0] for v in fk.values()])))
        r[sp+'_stress'] = st
        if sp == 'val': r['val_flags'] = fl
    res[e] = r
# small-object FN (size tertile from report attrs, pooled val+test)
if attrs:
    q = np.percentile([a['size'] for a in attrs], [33.3, 66.7])
    for e in EXPS:
        thr = res[e]['eval']['threshold_from_val']; f = f'{PRED}/p2_{e}.csv'; cnt = {'작음': [0, 0], '중간': [0, 0], '큼': [0, 0]}
        F = {sp: flags(E.load_preds(f, sp), G[sp], thr) for sp in ('val', 'test')}
        for a in attrs:
            lv = '작음' if a['size'] < q[0] else ('중간' if a['size'] < q[1] else '큼')
            cnt[lv][1] += 1; cnt[lv][0] += (not F[a['split']][(a['image'], a['j'])][0])
        res[e]['size_fn'] = cnt
for e in res: res[e].pop('val_flags', None)
json.dump(res, open(f'{ROOT}/report/p2_analysis.json', 'w'), ensure_ascii=False, indent=1)
for e, r in res.items():
    v, t = r['eval']['val'], r['eval']['test']
    print(f"{e:16s} valFN {v['FN']:2d} FP {v['FP']:2d} ctr {r['val_center']:.3f} | testFN {t['FN']} FP {t['FP']} | k.5 ctr val {r['val_stress']['050']['center']:.3f} test {r['test_stress']['050']['center']:.3f} | k.35 val {r['val_stress']['035']['center']:.3f} test {r['test_stress']['035']['center']:.3f} | small {r.get('size_fn')}")
