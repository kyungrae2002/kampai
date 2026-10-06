"""[검증셋 전용] 2차 판정: RF-DETR-S(A) + YOLO11s(A) 채택 여부. 테스트셋은 읽지 않음.
2차 기준(RF-DETR-S(A) 결과를 보기 전 확정, 10/5):
 ① 검증 FN ≤ 기존 S1 + 1  ② 이물 중심 적중률 100%  ③ k=0.5 중심 적중률 +5%p 이상  ④ 정상 오경보 증가 ≤ 5%p, 미끼 오탐 0
출력: report/compare_A2_val.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare_A_val.py'), encoding='utf-8').read().split('R = {')[0]
exec(src)
mx = lambda P, s: P[s][:, 4].max() if s in P and len(P[s]) else -1
R = {'RF-DETR-S(D) 단독': report(lambda sp: load('rfdetr_s', sp)),
     'RF-DETR-S(A) 단독': report(lambda sp: load('rfdetr_s_A', sp)),
     '기존 S1: RF-DETR-S(D) + YOLO11s(D)': report(lambda sp: s1('rfdetr_s', 'yolo11s', sp)),
     '1차 S1: RF-DETR-S(D) + YOLO11s(A)': report(lambda sp: s1('rfdetr_s', 'yolo11s_A', sp)),
     '2차 S1: RF-DETR-S(A) + YOLO11s(A)': report(lambda sp: s1('rfdetr_s_A', 'yolo11s_A', sp))}
A = load('rfdetr_s_A', 'val'); B = load('yolo11s_A', 'val')
R['2차 S1에서 RF-DETR-S(A) 선택 영상 수'] = int(sum(mx(A, s) >= mx(B, s) for s in G))
o, n = R['기존 S1: RF-DETR-S(D) + YOLO11s(D)'], R['2차 S1: RF-DETR-S(A) + YOLO11s(A)']
crit = {'① 검증 FN ≤ 기존+1': n['FN'] <= o['FN'] + 1, '② 중심 적중률 100%': n['center'] >= 1 - 1e-9,
        '③ k=0.5 적중률 +5%p 이상': n['stress']['050'] - o['stress']['050'] >= 0.05 - 1e-9,
        '④ 정상 오경보 증가 ≤ 5%p · 미끼 0': (n['normal_FA'] - o['normal_FA'] <= 0.05 + 1e-9) and n['decoy_FP'] == 0}
R['채택기준'] = crit; R['채택'] = all(crit.values())
json.dump(R, open(f'{ROOT}/report/compare_A2_val.json', 'w'), ensure_ascii=False, indent=1)
print(f"{'구성':38s} {'thr':>6s} {'FN':>3s} {'FP':>3s} {'재현율':>6s} {'AP50-95':>7s} {'중심':>5s} {'정상FA':>6s} {'미끼':>3s} | k.75 / .50 / .35")
for k, r in R.items():
    if isinstance(r, dict) and 'thr' in r:
        print(f"{k:38s} {r['thr']:6.3f} {r['FN']:3d} {r['FP']:3d} {r['recall']:6.3f} {r['AP50_95']:7.3f} {r['center']:5.3f} {r['normal_FA']:6.3f} {r['decoy_FP']:3d} | {r['stress']['075']:.2f} / {r['stress']['050']:.2f} / {r['stress']['035']:.2f}")
print('2차 S1에서 RF-DETR-S(A) 선택:', R['2차 S1에서 RF-DETR-S(A) 선택 영상 수'], '/ 80장')
for k, v in crit.items(): print(('통과 ' if v else '미달 ') + k)
print('=> 채택 (A 구성으로 통일)' if R['채택'] else '=> 기존 구성 유지')
