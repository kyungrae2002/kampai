"""[보고용] D vs A 비교표. 채택 여부는 검증셋 판정(compare_A_val / compare_A2_val)으로 이미 확정된 뒤 실행.
임계값은 각 구성의 검증 F1 최대값, 테스트에는 그대로 1회 적용. 출력: report/a_vs_d.json"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from common import *
def load(k, split): return E.load_preds(f'{PRED}/{"stress_"+k if split.startswith("stress") else k}.csv', split)
def s1(a, b, split):
    A = load(a, split); B = load(b, split); out = {}
    for s in set(A) | set(B):
        pa = A.get(s, np.zeros((0, 5))); pb = B.get(s, np.zeros((0, 5)))
        out[s] = pa if (pa[:, 4].max() if len(pa) else -1) >= (pb[:, 4].max() if len(pb) else -1) else pb
    return out
def center(gt, pred, thr):
    c = n = 0
    for s, g in gt.items():
        p = pred.get(s, np.zeros((0, 5))); p = p[p[:, 4] >= thr] if len(p) else p
        for b in g:
            n += 1
            if len(p):
                cx = (p[:, 0]+p[:, 2])/2; cy = (p[:, 1]+p[:, 3])/2
                c += bool(((cx >= b[0]-2) & (cx <= b[2]+2) & (cy >= b[1]-2) & (cy <= b[3]+2)).any())
    return c / n
GT = {sp: E.load_gt(sp) for sp in ('val', 'test', 'val_normal', 'test_normal')}
def run(get):
    G = GT['val']; thr = E.best_threshold(G, get('val')); out = {'thr': thr}
    for sp in ('val', 'test'):
        g = GT[sp]; p = get(sp); tp, fp, fn = E.counts_at(g, p, thr)
        out[sp] = dict(TP=tp, FP=fp, FN=fn, recall=tp/max(1, tp+fn), center=center(g, p, thr),
                       normal_FA=float((E.img_max(get(sp+'_normal'), list(GT[sp+'_normal'])) >= thr).mean()),
                       k050=center(g, get(f'stress_{sp}_k050'), thr), k035=center(g, get(f'stress_{sp}_k035'), thr))
    return out
C = {'YOLO11s (D)': lambda sp: load('yolo11s', sp), 'YOLO11s (A)': lambda sp: load('yolo11s_A', sp),
     'RF-DETR-S (D)': lambda sp: load('rfdetr_s', sp), 'RF-DETR-S (A)': lambda sp: load('rfdetr_s_A', sp),
     'S1: RF-DETR-S(D)+YOLO11s(D) [최종]': lambda sp: s1('rfdetr_s', 'yolo11s', sp),
     'S1: RF-DETR-S(D)+YOLO11s(A)': lambda sp: s1('rfdetr_s', 'yolo11s_A', sp),
     'S1: RF-DETR-S(A)+YOLO11s(A)': lambda sp: s1('rfdetr_s_A', 'yolo11s_A', sp)}
R = {k: run(f) for k, f in C.items()}
json.dump(R, open(f'{ROOT}/report/a_vs_d.json', 'w'), ensure_ascii=False, indent=1)
for k, r in R.items():
    v, t = r['val'], r['test']
    print(f"{k:36s} thr {r['thr']:.3f} | FN {v['FN']}/{t['FN']} R {v['recall']:.3f}/{t['recall']:.3f} ctr {v['center']:.3f}/{t['center']:.3f} nFA {v['normal_FA']:.3f}/{t['normal_FA']:.3f} k.5 {v['k050']:.2f}/{t['k050']:.2f} k.35 {v['k035']:.2f}/{t['k035']:.2f}")
