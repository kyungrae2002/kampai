"""사람 라벨 500장으로 색 박스 기반 자동 라벨(원래 크기 / 8px 축소 규칙) 품질 재확인 + 보고서 그림 생성. 데이터셋은 읽기만 함.
출력: report/label_audit.json, report/figs/f12_label_audit.png"""
import os, csv, json, glob, unicodedata, cv2, numpy as np
from xproc import color_mask
K = os.path.expanduser('~/mnt/kamp_ai'); RAW = K + '/dataset/dataset/test1/yolov3'; D = K + '/X-ray_데이터셋_통합'; R = K + '/X-ray_실험/report'
rows = [r for r in csv.DictReader(open(D + '/metadata/원본_README/split_report.csv', encoding='utf-8-sig')) if r['label'] == 'human']
lab_dir = {'train': '학습/1_원본영상', 'val': '검증', 'test': '테스트'}
def comps(raw):
    cm = color_mask(raw); n, lab, st, _ = cv2.connectedComponentsWithStats(cm)
    return [(st[i, 0], st[i, 1], st[i, 2], st[i, 3]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 8 and 8 <= st[i, 2] <= 60 and 8 <= st[i, 3] <= 60]
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)
def match(G, P):  # greedy 1:1 by IoU
    pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(G) for j, p in enumerate(P)), reverse=True); ug = set(); up = set(); m = []
    for v, i, j in pairs:
        if v >= 0.5 and i not in ug and j not in up: ug.add(i); up.add(j); m.append(v)
    return m
def cdist(G, P):
    used = set(); n = 0; ds = []
    for g in G:
        c = ((g[0]+g[2])/2, (g[1]+g[3])/2); best = None
        for j, p in enumerate(P):
            if j in used: continue
            d = np.hypot((p[0]+p[2])/2 - c[0], (p[1]+p[3])/2 - c[1])
            if best is None or d < best[0]: best = (d, j)
        if best and best[0] <= 3: used.add(best[1]); ds.append(best[0])
    return ds
S = {k: dict(gt=0, pred=0, iou_match=0, ious=[], ctr=0, cd=[]) for k in ('outline', 'shrink8')}
per_split = {}; ex = []
for r in rows:
    p = r['source'].replace('/content/work/src', RAW)
    if not os.path.exists(p): p = unicodedata.normalize('NFD', p)
    raw = cv2.imread(p); H, W = raw.shape[:2]
    G = []
    for l in open(f"{D}/{lab_dir[r['split']]}/labels/{r['name']}.txt"):
        if l.strip(): c, x, y, w, h = map(float, l.split()); G.append((( x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H))
    C = comps(raw)
    P = {'outline': [(x, y, x+w, y+h) for x, y, w, h in C],
         'shrink8': [(x+w/2-max(5, w-8)/2, y+h/2-max(5, h-8)/2, x+w/2+max(5, w-8)/2, y+h/2+max(5, h-8)/2) for x, y, w, h in C]}
    for k, PP in P.items():
        m = match(G, PP); cds = cdist(G, PP); s = S[k]
        s['gt'] += len(G); s['pred'] += len(PP); s['iou_match'] += len(m); s['ious'] += m; s['ctr'] += len(cds); s['cd'] += cds
        if k == 'shrink8': ps = per_split.setdefault(r['split'], [0, 0]); ps[0] += len(m); ps[1] += len(G)
    ex.append((len(G) - len(match(G, P['shrink8'])), r, raw, G, P))
out = {k: dict(gt=s['gt'], pred=s['pred'], iou50_match=s['iou_match'], iou50_rate=s['iou_match']/s['gt'], center3px_match=s['ctr'], center3px_rate=s['ctr']/s['gt'],
               mean_center_err=float(np.mean(s['cd'])), mean_iou_matched=float(np.mean(s['ious']))) for k, s in S.items()}
out['shrink8_by_split'] = per_split
json.dump(out, open(R + '/label_audit.json', 'w'), ensure_ascii=False, indent=1); print(json.dumps(out, ensure_ascii=False, indent=1))
# figure: 2 good + 1 mismatch example, crop around objects, scale x3
def panel(raw, G, P, title):
    g = cv2.cvtColor(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    allb = G + P['outline']; x0 = int(max(0, min(b[0] for b in allb) - 18)); y0 = int(max(0, min(b[1] for b in allb) - 18))
    x1 = int(min(raw.shape[1], max(b[2] for b in allb) + 18)); y1 = int(min(raw.shape[0], max(b[3] for b in allb) + 18))
    crop = cv2.resize(g[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    def rect(b, col, t): cv2.rectangle(crop, (int((b[0]-x0)*4), int((b[1]-y0)*4)), (int((b[2]-x0)*4), int((b[3]-y0)*4)), col, t)
    for b in P['outline']: rect(b, (0, 220, 255), 2)
    for b in P['shrink8']: rect(b, (255, 0, 255), 2)
    for b in G: rect(b, (0, 200, 0), 2)
    crop = cv2.copyMakeBorder(crop, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
    return cv2.resize(crop, (int(crop.shape[1] * 520 / crop.shape[0]), 520))
ex.sort(key=lambda e: e[0]); picks = [ex[0], ex[len(ex)//2], ex[-1]]
pans = [panel(e[2], e[3], e[4], '') for e in picks]
fig = cv2.hconcat([cv2.copyMakeBorder(p, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in pans])
cv2.imwrite(R + '/figs/f12_label_audit.png', fig); print('picks', [(e[0], e[1]['name'], e[1]['split']) for e in picks], fig.shape)
