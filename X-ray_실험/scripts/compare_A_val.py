"""[검증셋 전용] YOLO11s(A) 채택 여부 판정. 테스트셋은 읽지 않음.
비교: 기존 S1 = RF-DETR-S + YOLO11s(D)  vs  새 S1 = RF-DETR-S + YOLO11s(A)
채택 기준(학습 전 확정): ① 검증 FN ≤ 기존 ② 검증 k=0.5 중심 적중률 +5%p 이상 ③ 합성 정상 오경보 증가 ≤ 5%p ④ 미끼 오탐 0
출력: report/compare_A_val.json"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from common import *
G = E.load_gt('val'); GN = E.load_gt('val_normal'); DEC = E.load_decoys('val')
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
def report(get):
    pv = get('val'); thr = E.best_threshold(G, pv); tp, fp, fn = E.counts_at(G, pv, thr)
    return dict(thr=thr, TP=tp, FP=fp, FN=fn, recall=tp/max(1, tp+fn), precision=tp/max(1, tp+fp),
                AP50=E.ap(G, pv, 0.5), AP50_95=float(np.mean([E.ap(G, pv, t) for t in np.arange(0.5, 0.96, 0.05)])),
                center=center(G, pv, thr), normal_FA=float((E.img_max(get('val_normal'), list(GN)) >= thr).mean()),
                decoy_FP=E.decoy_hits(pv, DEC, thr)[0],
                stress={k: center(G, get(f'stress_val_k{k}'), thr) for k in ('075', '050', '035')})
R = {'yolo11s(D) 단독': report(lambda sp: load('yolo11s', sp)),
     'yolo11s(A) 단독': report(lambda sp: load('yolo11s_A', sp)),
     '기존 S1: RF-DETR-S + YOLO11s(D)': report(lambda sp: s1('rfdetr_s', 'yolo11s', sp)),
     '새 S1: RF-DETR-S + YOLO11s(A)': report(lambda sp: s1('rfdetr_s', 'yolo11s_A', sp))}
o, n = R['기존 S1: RF-DETR-S + YOLO11s(D)'], R['새 S1: RF-DETR-S + YOLO11s(A)']
crit = {'① 검증 FN ≤ 기존': n['FN'] <= o['FN'], '② k=0.5 적중률 +5%p 이상': n['stress']['050'] - o['stress']['050'] >= 0.05 - 1e-9,
        '③ 정상 오경보 증가 ≤ 5%p': n['normal_FA'] - o['normal_FA'] <= 0.05 + 1e-9, '④ 미끼 오탐 0': n['decoy_FP'] == 0}
R['채택기준'] = crit; R['채택'] = all(crit.values())
json.dump(R, open(f'{ROOT}/report/compare_A_val.json', 'w'), ensure_ascii=False, indent=1)
print(f"{'구성':34s} {'thr':>6s} {'FN':>3s} {'FP':>3s} {'재현율':>6s} {'AP50-95':>7s} {'중심':>5s} {'정상FA':>6s} {'미끼':>3s} | k.75 / .50 / .35")
for k, r in R.items():
    if isinstance(r, dict) and 'thr' in r:
        print(f"{k:34s} {r['thr']:6.3f} {r['FN']:3d} {r['FP']:3d} {r['recall']:6.3f} {r['AP50_95']:7.3f} {r['center']:5.3f} {r['normal_FA']:6.3f} {r['decoy_FP']:3d} | {r['stress']['075']:.2f} / {r['stress']['050']:.2f} / {r['stress']['035']:.2f}")
for k, v in crit.items(): print(('통과 ' if v else '미달 ') + k)
print('=> 채택' if R['채택'] else '=> 기존 구성 유지')
