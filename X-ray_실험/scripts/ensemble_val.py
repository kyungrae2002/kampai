"""두 모델 혼용(앙상블) 실험 — 검증셋 전용.
방식
 S1 영상 단위 선택: 영상마다 최고 confidence가 더 높은 모델의 예측만 사용 (요청하신 방식 그대로)
 S2 박스 단위 선택: 두 모델 박스를 모아, 같은 이물(IoU≥0.3)로 겹치면 confidence가 높은 박스만 남김
 S3 박스 단위 선택 + 점수 보정: 각 모델 점수를 자기 검증 임계값으로 나눈 값(score/thr)으로 비교
각 방식의 최종 임계값은 검증 F1 최대값으로 정함 (검증 안에서 정하고 검증으로 평가 → 낙관적 수치)"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from common import *
OUT = f'{ROOT}/report/ensemble_val.json'
G = E.load_gt('val'); GN = E.load_gt('val_normal')
def load(k, split, csvname=None): return E.load_preds(f'{PRED}/{csvname or k}.csv', split)
THR = {k: json.load(open(f'{PRED}/{k}_eval.json'))['threshold_from_val'] for k in ('rfdetr_s', 'yolo11s', 'yolov8n')}
def nms_merge(pa, pb, iou_thr=0.3):
    if len(pa) == 0: return pb
    if len(pb) == 0: return pa
    P = np.vstack([pa, pb]); P = P[np.argsort(-P[:, 4])]; keep = []
    used = np.zeros(len(P), bool)
    for i in range(len(P)):
        if used[i]: continue
        keep.append(P[i]); ov = E.iou(P[i:i+1, :4], P[:, :4])[0]; used |= ov >= iou_thr
    return np.array(keep)
def combine(a, b, mode, split):
    A = load(a, split, a if not split.startswith('stress') else f'stress_{a}'); B = load(b, split, b if not split.startswith('stress') else f'stress_{b}')
    out = {}
    for s in set(A) | set(B):
        pa = A.get(s, np.zeros((0, 5))).copy(); pb = B.get(s, np.zeros((0, 5))).copy()
        if mode == 'S3':
            if len(pa): pa[:, 4] = pa[:, 4] / THR[a]
            if len(pb): pb[:, 4] = pb[:, 4] / THR[b]
        if mode == 'S1':
            ma = pa[:, 4].max() if len(pa) else -1; mb = pb[:, 4].max() if len(pb) else -1
            out[s] = pa if ma >= mb else pb
        else:
            out[s] = nms_merge(pa, pb)
    return out
def metrics(pred, thr, gt=G, normal=None):
    tp, fp, fn = E.counts_at(gt, pred, thr); ctr = 0; n = 0
    for s, g in gt.items():
        p = pred.get(s, np.zeros((0, 5))); p = p[p[:, 4] >= thr] if len(p) else p
        for b in g:
            n += 1
            if len(p):
                cx = (p[:, 0]+p[:, 2])/2; cy = (p[:, 1]+p[:, 3])/2; ctr += bool(((cx >= b[0]-2) & (cx <= b[2]+2) & (cy >= b[1]-2) & (cy <= b[3]+2)).any())
    r = dict(TP=tp, FP=fp, FN=fn, precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn), F1=E.f1(tp, fp, fn), center_recall=ctr/n, AP50=E.ap(gt, pred, 0.5),
             AP50_95=float(np.mean([E.ap(gt, pred, t) for t in np.arange(0.5, 0.96, 0.05)])))
    if normal is not None:
        neg = E.img_max(normal, list(GN)); r['normal_FA'] = float((neg >= thr).mean())
    dec = E.load_decoys('val'); r['decoy_FP'] = E.decoy_hits(pred, dec, thr)[0]
    return r
res = {}
def single(k):
    pv = load(k, 'val'); thr = THR[k]
    r = metrics(pv, thr, normal=load(k, 'val_normal')); r['thr'] = thr
    r['stress'] = {kk: metrics(load(k, f'stress_val_k{kk}', f'stress_{k}'), thr)['center_recall'] for kk in ('075', '050', '035')}
    return r
for k in ('rfdetr_s', 'yolo11s', 'yolov8n'): res[k] = single(k)
for a, b in (('rfdetr_s', 'yolo11s'), ('rfdetr_s', 'yolov8n')):
    for mode in ('S1', 'S2', 'S3'):
        pv = combine(a, b, mode, 'val'); thr = E.best_threshold(G, pv)
        r = metrics(pv, thr, normal=combine(a, b, mode, 'val_normal')); r['thr'] = thr
        r['stress'] = {kk: metrics(combine(a, b, mode, f'stress_val_k{kk}'), thr)['center_recall'] for kk in ('075', '050', '035')}
        if mode == 'S1':
            A = load(a, 'val'); B = load(b, 'val'); win = sum(1 for s in G if (A.get(s, np.zeros((0, 5)))[:, 4].max() if s in A and len(A[s]) else -1) >= (B.get(s, np.zeros((0, 5)))[:, 4].max() if s in B and len(B[s]) else -1))
            r['images_won_by_first'] = win
        res[f'{a}+{b}|{mode}'] = r
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(f"{'구성':28s} {'thr':>6s} {'TP':>4s} {'FN':>3s} {'FP':>3s} {'정밀도':>7s} {'재현율':>7s} {'F1':>6s} {'AP50':>6s} {'AP50-95':>7s} {'중심':>6s} {'정상FA':>6s} {'미끼':>3s} | 저대비중심 k.75/.5/.35")
for k, r in res.items():
    print(f"{k:28s} {r['thr']:6.3f} {r['TP']:4d} {r['FN']:3d} {r['FP']:3d} {r['precision']:7.3f} {r['recall']:7.3f} {r['F1']:6.3f} {r['AP50']:6.3f} {r['AP50_95']:7.3f} {r['center_recall']:6.3f} {r['normal_FA']:6.3f} {r['decoy_FP']:3d} | {r['stress']['075']:.3f} {r['stress']['050']:.3f} {r['stress']['035']:.3f}" + (f"  (첫 모델 선택 {r['images_won_by_first']}/80장)" if 'images_won_by_first' in r else ''))
