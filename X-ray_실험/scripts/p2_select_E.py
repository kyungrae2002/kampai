"""A~D 중 '검증' 결과만으로 E(입력 960)의 학습 데이터를 고른다. 규칙(사전 고정): 검증 FN 최소 → 동률이면 검증 AP50 최대.
선택된 yaml 경로를 출력."""
import json, os, sys
from common import *
import evaluate as E
best = None
for n in ['A_실제만', 'B_실제+이물합성', 'C_실제+이물제거', 'D_전체']:
    f = f'{PRED}/p2_{n}.csv'
    if not os.path.exists(f): continue
    j = f'{PRED}/p2_{n}_eval.json'
    if not os.path.exists(j): json.dump(E.evaluate(f, boot=1000), open(j, 'w'), ensure_ascii=False, indent=1)
    v = json.load(open(j))['val']; key = (v['FN'], -v['AP50'])
    print(n, 'val FN', v['FN'], 'AP50 %.4f' % v['AP50'], file=sys.stderr)
    if best is None or key < best[0]: best = (key, n)
print(f'{ROOT}/ablation/{best[1]}.yaml')
open(f'{ROOT}/ablation/E_선택근거.txt', 'w').write(f'E 학습 데이터 = {best[1]} (검증 FN 최소, 동률 시 검증 AP50 최대 규칙)\n')
