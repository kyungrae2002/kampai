"""[보고용] 두 모델 판단 불일치 시 보류(HOLD) 규칙 평가. 임계값은 기존 검증 결정값을 그대로 사용(새로 튜닝하지 않음).
규칙 U: 각 모델의 확정선(검증 F1 최대) 이상 박스를 서로 대응(중심이 상대 박스 안 또는 IoU≥0.3).
  - 두 모델 모두 박스 0개 → PASS
  - 박스 개수가 같고 모든 박스가 1:1 대응 → REJECT(합의)
  - 그 외(개수 차이·한쪽만 찾은 박스) → HOLD(품질 담당자 확인 또는 정밀 모델 재판독)
규칙 U+: U에 더해, 어느 모델이든 재검사 하한 이상 후보가 있으면 PASS 대신 HOLD
출력: report/disagree_policy.json"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from common import *
RA = json.load(open(f'{ROOT}/report/report_analysis.json'))
TH = {'rfdetr_s': RA['t_high']['rfdetr_s'], 'yolo11s': RA['t_high']['yolo11s']}
TL = {'rfdetr_s': RA['t_low']['rfdetr_s'], 'yolo11s': RA['t_low']['yolo11s']}
def load(k, sp): return E.load_preds(f'{PRED}/{"stress_"+k if sp.startswith("stress") else k}.csv', sp)
def inside(b, g): cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2; return g[0]-2 <= cx <= g[2]+2 and g[1]-2 <= cy <= g[3]+2
def same(a, b): return inside(a, b) or inside(b, a) or E.iou(a[None, :4], b[None, :4])[0][0] >= 0.3
def decide(pa, pb, plus):
    A = pa[pa[:, 4] >= TH['rfdetr_s']] if len(pa) else pa; B = pb[pb[:, 4] >= TH['yolo11s']] if len(pb) else pb
    if len(A) == 0 and len(B) == 0:
        low = (len(pa) and pa[:, 4].max() >= TL['rfdetr_s']) or (len(pb) and pb[:, 4].max() >= TL['yolo11s'])
        return 'HOLD' if (plus and low) else 'PASS'
    if len(A) == len(B) and all(any(same(a, b) for b in B) for a in A) and all(any(same(b, a) for a in A) for b in B): return 'REJECT'
    return 'HOLD'
def s1(pa, pb):
    ma = pa[:, 4].max() if len(pa) else -1; mb = pb[:, 4].max() if len(pb) else -1
    return (pa, TH['rfdetr_s']) if ma >= mb else (pb, TH['yolo11s'])
def hit(p, thr, g):
    p = p[p[:, 4] >= thr] if len(p) else p
    return any(inside(r, g) for r in p)
OUT = {}
for sp, gtsp in [('val', 'val'), ('test', 'test'), ('stress_val_k050', 'val'), ('stress_test_k050', 'test'), ('stress_val_k035', 'val'), ('stress_test_k035', 'test'), ('val_normal', None), ('test_normal', None)]:
    A = load('rfdetr_s', sp); B = load('yolo11s', sp)
    G = E.load_gt(gtsp) if gtsp else {s: np.zeros((0, 4)) for s in E.load_gt(sp)}
    r = {'images': len(G)}
    for plus in (False, True):
        cnt = {'PASS': 0, 'HOLD': 0, 'REJECT': 0}; miss_img_s1 = 0; miss_img_s1_pass = 0; obj_s1_miss = 0; obj_s1_miss_flag = 0; obj_n = 0; obj_union = 0
        for s, g in G.items():
            stem_ = s
            pa = A.get(s, np.zeros((0, 5))); pb = B.get(s, np.zeros((0, 5)))
            dcs = decide(pa, pb, plus); cnt[dcs] += 1
            P, thr = s1(pa, pb); missed = 0
            for b in g:
                obj_n += 1; m = not hit(P, thr, b)
                u = hit(pa, TH['rfdetr_s'], b) or hit(pb, TH['yolo11s'], b); obj_union += u
                if m: obj_s1_miss += 1; missed += 1; obj_s1_miss_flag += (dcs != 'PASS')
            if missed: miss_img_s1 += 1; miss_img_s1_pass += (dcs == 'PASS')
        r['U+' if plus else 'U'] = dict(**cnt, objects=obj_n, s1_center_miss=obj_s1_miss, s1_miss_sent_to_hold_or_reject=obj_s1_miss_flag,
                                       union_center_hit=obj_union, imgs_with_s1_miss=miss_img_s1, imgs_with_s1_miss_passed=miss_img_s1_pass)
    OUT[sp] = r
json.dump({'TH': TH, 'TL': TL, 'res': OUT}, open(f'{ROOT}/report/disagree_policy.json', 'w'), ensure_ascii=False, indent=1)
for sp, r in OUT.items():
    for k in ('U', 'U+'):
        x = r[k]; print(f"{sp:18s} {k:3s} PASS {x['PASS']:3d} HOLD {x['HOLD']:3d} REJ {x['REJECT']:3d} | 이물 {x['objects']} S1중심놓침 {x['s1_center_miss']} 그중보류/거부 {x['s1_miss_sent_to_hold_or_reject']} 두모델합집합적중 {x['union_center_hit']} | S1놓침영상 {x['imgs_with_s1_miss']} 그중PASS {x['imgs_with_s1_miss_passed']}")
