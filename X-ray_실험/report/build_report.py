import json, os, base64, datetime
R = '/home/claude/kamp/X-ray_실험'
S1 = json.load(open(f'{R}/analysis_stage1.json')); RA = json.load(open(f'{R}/report/report_analysis.json')); P2 = json.load(open(f'{R}/report/p2_analysis.json')); ENS = json.load(open(f'{R}/report/ensemble_val_test.json')); BL = json.load(open(f'{R}/report/baseline_yolov3_valtest.json')); LA = json.load(open(f'{R}/report/label_audit.json')); DP = json.load(open(f'{R}/report/disagree_policy.json'))['res']; CP = json.load(open(f'{R}/report/complement.json')); AD = json.load(open(f'{R}/report/a_vs_d.json'))
def img(p, w='100%'):
    b = base64.b64encode(open(p, 'rb').read()).decode(); ext = 'png' if p.endswith('png') else 'jpeg'
    return f'<img src="data:image/{ext};base64,{b}" style="width:{w}">'
F = lambda n: img(f'{R}/report/figs/{n}')
def pct(x, d=1): return '-' if x is None else f'{x*100:.{d}f}%'
N = {'rfdetr_s': 'RF-DETR-S', 'yolo11s': 'YOLO11s', 'yolov8n': 'YOLOv8n', 'qwen25vl3b_lora': 'Qwen2.5-VL-3B LoRA', 'qwen25vl3b_zeroshot': 'Qwen2.5-VL-3B 제로샷', 'yolov8n_4403장': 'YOLOv8n (라벨 수정 전 4,403장)'}
def ms(k):
    m = S1[k]['meta']; v = m.get('ms_per_image_mps_batch1') or m.get('ms_per_image_mps'); return f'{v:,.0f}' if v >= 100 else f'{v:.1f}'
def main_table(keys, hl=('rfdetr_s','yolo11s')):
    h = '<table><thead><tr><th>모델</th><th>세트</th><th>정밀도</th><th>재현율</th><th>F1</th><th>mAP50</th><th>mAP50-95</th><th>TP</th><th>FN</th><th>FP</th><th>중심 적중률</th><th>정상 오경보</th><th>미끼 오탐</th></tr></thead><tbody>'
    for k in keys:
        for sp, lab in (('val', '검증'), ('test', '테스트')):
            v = S1[k]['eval'][sp]
            B = (lambda x: f'<b>{x}</b>') if k in hl else (lambda x: x)
            h += f"<tr>{f'<td rowspan=2 class=l>{B(N[k])}</td>' if sp=='val' else ''}<td>{lab}</td><td>{pct(v['precision'])}</td><td>{B(pct(v['recall']))}</td><td>{B(f"{v['F1']:.3f}")}</td><td>{B(f"{v['AP50']:.3f}")}</td><td>{B(f"{v['AP50_95']:.3f}")}</td><td>{v['TP']}</td><td>{B(v['FN'])}</td><td>{v['FP']}</td><td>{pct(S1[k][sp+'_center_recall'])}</td><td>{pct(v['normal_false_alarm_rate'])}</td><td>{v['decoy_FP']}</td></tr>"
    return h + '</tbody></table>'
def sweep_table(k):
    h = '<table><thead><tr><th>임계값</th><th>검증 TP</th><th>검증 FN</th><th>검증 FP</th><th>검증 정밀도</th><th>검증 재현율</th><th>검증 정상 오경보</th><th>테스트 FN</th><th>테스트 FP</th><th>테스트 정상 오경보</th></tr></thead><tbody>'
    for s in S1[k]['sweep']:
        if round(s['thr']*100) % 10 != 0 and round(s['thr'], 2) != 0.05: continue
        v, t = s['val'], s['test']
        h += f"<tr><td>{s['thr']:.2f}</td><td>{v['TP']}</td><td>{v['FN']}</td><td>{v['FP']}</td><td>{pct(v['P'])}</td><td>{pct(v['R'])}</td><td>{pct(v['normal_FA'])}</td><td>{t['FN']}</td><td>{t['FP']}</td><td>{pct(t['normal_FA'])}</td></tr>"
    return h + '</tbody></table>'
def hogi_table(k):
    h = '<table><thead><tr><th>세트</th><th>1호기 FN/이물</th><th>2호기 FN/이물</th><th>3호기 FN/이물</th></tr></thead><tbody>'
    for sp, lab in (('val', '검증'), ('test', '테스트')):
        hz = S1[k][sp+'_hogi']; h += f"<tr><td>{lab}</td>" + ''.join(f"<td>{hz[x]['FN']} / {hz[x]['TP']+hz[x]['FN']}</td>" for x in '123') + '</tr>'
    return h + '</tbody></table>'
def fnt_table(k):
    a, b = S1[k]['val_fn_types'], S1[k]['test_fn_types']
    return f"<table><thead><tr><th>세트</th><th>박스 불일치 (IoU 0.1~0.5)</th><th>확신도 미달</th><th>후보 없음</th></tr></thead><tbody><tr><td>검증</td><td>{a['box_mismatch']}</td><td>{a['below_threshold']}</td><td>{a['no_candidate']}</td></tr><tr><td>테스트</td><td>{b['box_mismatch']}</td><td>{b['below_threshold']}</td><td>{b['no_candidate']}</td></tr></tbody></table>"
CI = RA['session_ci']; TH = RA['t_high']; TL = RA['t_low']; POL = RA['policy']
fail = RA['fail']
FL={'크기':'이물 크기','대비':'이물 대비','배경 밝기':'이물 주변 배경 밝기','호기':'X-ray 장비 번호(호기)','형상':'이물 형상','위치':'제품 경계와 이물의 거리'}
def fail_rows(f):
    return ''.join(f"<tr><td class=l>{FL.get(f,f)}</td><td class=l>{r['level']}</td><td>{r['n']}</td>" + ''.join(f"<td>{r[k]['FN']} ({pct(r[k]['recall'])})</td>" for k in ('rfdetr_s','yolo11s','yolov8n')) + '</tr>' for r in fail[f])
stress = RA['stress']
def st(k, sp, kk, key='center_recall'): return [r for r in stress[k] if r['split']==sp and abs(r['k']-kk)<1e-6][0][key]
def pol_rows():
    out = ''
    for name, r in POL.items():
        for sp, lab in (('val','검증'),('test','테스트')):
            d = r[sp]; n = r[sp+'_normal']
            out += f"<tr>{f'<td rowspan=2 class=l>{name}</td>' if sp=='val' else ''}<td>{lab}</td><td>{d.get('REJECT',0)}</td><td>{d.get('RE-INSPECTION',0)}</td><td><b>{d.get('PASS',0)}</b></td><td>{n.get('REJECT',0)}</td><td>{n.get('RE-INSPECTION',0)}</td><td>{n.get('PASS',0)}</td></tr>"
    return out
css = """
@page { size: A4; }
body { word-break: keep-all; font-family: 'Noto Serif CJK KR', 'Noto Serif KR', serif; font-size: 14pt; line-height: 1.6; color: #111; margin:0 }
.cover h1 { font-size: 19pt; text-align:center; margin: 14mm 0 10mm; letter-spacing: .01em; word-break: keep-all }
.cover table td { font-size: 13pt; padding: 8px 12px; vertical-align: top }
.cover .k { width: 26mm; background:#f2f2f2; font-weight:700; text-align:center }
.cover .sum { font-size: 10.5pt; line-height:1.6; text-align: justify }
.cover .stmt { font-size: 13pt; margin: 10mm 0 6mm; }
.cover .date { text-align:right; font-size:13pt; margin-bottom: 4mm }
.cover .sig { text-align:right; font-size:13pt; line-height:2.0 }
.cover .to { text-align:center; font-size: 16pt; font-weight:700; margin-top: 10mm }
h2 { font-size: 16pt; margin: 0 0 6mm; break-after: avoid; border-bottom: 1.5px solid #222; padding-bottom: 2mm }
h2.chap { break-before: page }
.o { font-size: 14pt; font-weight: 700; margin: 6mm 0 2mm; break-after: avoid }
.d { font-size: 14pt; margin: 1.5mm 0 1.5mm 6mm; text-indent: -4mm; padding-left: 4mm }
.s { font-size: 10pt; margin: 1mm 0 1mm 14mm; text-indent: -3mm; padding-left: 3mm; color:#222 }
table { border-collapse: collapse; width: 100%; font-size: 9pt; line-height: 1.4; margin: 2mm 0 2mm; font-family: 'Noto Sans CJK KR', sans-serif; break-inside: avoid }
th, td { border: 0.6px solid #888; padding: 1.4mm 1.6mm; text-align: center; vertical-align: middle }
th { background: #eceff1; font-weight: 700 }
td.l { text-align: left }
figure { margin: 3mm 0 4mm; break-inside: avoid; text-align:center }
figcaption, .cap { font-size: 10pt; color: #333; margin-top: 1.5mm; text-align:center }
.tcap { font-size: 10pt; font-weight:700; margin: 3mm 0 0; break-after: avoid }
.box { border: 1px solid #999; padding: 3mm 4mm; margin: 3mm 0; font-size: 11pt; background:#fafafa; break-inside: avoid }
pre { font-size: 8.5pt; line-height:1.35; background:#f5f5f5; padding: 3mm; border:0.6px solid #bbb; white-space: pre-wrap; break-inside: avoid; font-family: 'Noto Sans Mono CJK KR', monospace }
.pb { break-before: page }
"""
P = []; a = P.append
o = lambda t: a(f'<div class=o>◦ {t}</div>')
d = lambda t: a(f'<div class=d>- {t}</div>')
s = lambda t: a(f'<div class=s>* {t}</div>')
rf, y11, y8 = S1['rfdetr_s']['eval'], S1['yolo11s']['eval'], S1['yolov8n']['eval']
# ---------------- COVER ----------------
summary = f"""포장공정 마지막 품질검사인 X선 이물검사의 미검·과검을 AI 영상 판정으로 보완하는 과제입니다. KAMP X선 이물검출기 데이터는 장비가 이물 위치에 색 표시 박스를 그린 영상이 기본이며, 이 데이터로 학습한 기존 모델(KAMP 제공 YOLOv3)은 표시 박스가 있을 때만 이물을 찾습니다. 같은 테스트 영상에서 표시 박스를 지우자 기존 모델의 이물 적중률은 {BL['test_raw']['center_recall']*100:.0f}%에서 0%가 됐습니다. 본 팀은 표시 박스를 지우고, 이물이 없는 자리에도 같은 지운 흔적(미끼 박스)을 남겨 모델이 표시나 흔적이 아니라 이물 자체를 학습하게 했습니다. 그 결과 표시 박스가 없는 영상에서 RF-DETR-S와 YOLO11s를 혼용한 최종 모델이 테스트셋의 실제 이물 {rf['test']['TP']+rf['test']['FN']}개 중 {rf['test']['TP']}개를 일반 객체탐지 기준으로 검출했으며({pct(rf['test']['recall'])}), 이물 위치 자체는 모두 찾아냈습니다. 미끼 흔적에 대한 오탐은 0건이었습니다. 혼용 방식은 대비를 절반으로 낮춘 이물의 적중률도 84%에서 90%로 높였습니다. 기준상 미탐 2개도 위치는 맞혔으나 박스 크기가 정답과 달랐던 경우이며, 현장에서는 두 모델이 각각 판독해 이물 개수·위치가 다르거나 확신도가 애매하면 제품을 보류하고 품질 담당자가 확인하는 PASS / RE-INSPECTION / REJECT 3단계 판정을 제안합니다."""
a(f"""<section class=cover><h1>제6회 K-인공지능 제조데이터 분석 경진대회 보고서</h1>
<table><tr><td class=k>프로젝트명</td><td>&nbsp;</td></tr><tr><td class=k>팀명</td><td>&nbsp;</td></tr><tr><td class=k>내용요약</td><td class=sum>{summary}</td></tr></table>
<div class=stmt>&nbsp;&nbsp;상기 본인(팀)은 위의 내용과 같이 제6회 K-인공지능 제조데이터 분석 경진대회 결과 보고서를 제출합니다.</div>
<div class=date>2026년 &nbsp;&nbsp;&nbsp;&nbsp; 월 &nbsp;&nbsp;&nbsp;&nbsp; 일</div>
<div class=sig>팀장 : &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (서명)<br>팀원 : &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (서명)<br>팀원 : &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (서명)</div>
<div class=to>(사)중소기업기술혁신협회 귀중</div></section>""")
# ---------------- CH1 ----------------
a('<h2 class=chap>□ 제1장. 데이터 이해 및 진단</h2>')
o('공정 배경과 현장 문제')
d('대상 공정은 식품제조업체 H사의 분말유크림(분무건조공법) 생산 라인 중 포장공정의 마지막 단계인 X선 이물검사입니다. X선 이물검출기는 완제품을 개봉하지 않고 투과 영상으로 내부의 금속·이물 포함 여부를 판정하며, 소비자에게 제품이 전달되기 전 마지막 품질검사입니다.')
d('현장의 핵심 문제는 X-ray 검사장비의 기술적 한계로 발생하는 미검(불량을 양품으로 판정)과 과검(양품을 불량으로 판정)입니다. 현재는 이중·삼중 검사로 대응하지만 중복 작업 비용이 들고, 같은 장비로 반복해도 장비의 근본 한계는 남습니다.')
d('본 팀은 이를 보완하기 위해 기존 장비의 표시나 판정 결과에 의존하지 않고 X-ray 영상 자체에서 이물을 독립적으로 판별하는 AI 모델을 개발하고자 했습니다.')
s('데이터 출처: 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), X-ray 검사장비 AI 데이터셋, KAIST(㈜임픽스, 한양대학교 산학협력단, ㈜아큐라소프트), 2020.12.14., www.kamp-ai.kr')
o('연구 질문과 선행연구에서 얻은 설계 근거')
d('본 연구는 라벨이 적고(500장) 같은 시험편을 연속 촬영한 영상이 많다는 데이터 특성에서 출발했습니다. 그래서 새 모델 구조를 만드는 것보다, 기존 탐지 모델에 데이터 처리와 평가·운영 규칙을 연결하는 데 초점을 두고 다음 세 가지를 확인했습니다.')
a("""<div class=box>① 장비 표시와 라벨 부족 문제를 데이터 처리로 해결할 수 있는가? (1장)<br>② 모델은 어떤 조건의 이물을 놓치며, 두 모델은 서로의 오류를 얼마나 보완하는가? (2·3장)<br>③ 모델이 틀릴 수 있다는 전제에서, 현장에서 놓친 이물이 출하되지 않게 하는 판정 규칙은 무엇인가? (4장)</div>""")
a("""<table><thead><tr><th style="width:30%">선행연구</th><th>핵심 내용</th><th>본 연구에 적용한 점</th></tr></thead><tbody>
<tr><td class=l>Kim 등 (2021) [2]</td><td class=l>부족한 식품 X-ray 이물 자료를 합성으로 보완하고 정확도와 처리 시간을 함께 평가</td><td class=l>합성·제거 가공의 효과를 실험으로 따로 확인(표 3), 처리 시간 함께 보고</td></tr>
<tr><td class=l>Andriiashen 등 (2024) [3]</td><td class=l>X-ray 촬영 조건과 대비에 따라 검출 성능이 달라짐을 분석</td><td class=l>평균 성능 외에 크기·대비·호기별 미탐과 저대비 평가(3장)</td></tr>
<tr><td class=l>Wang 등 (2021) [4]</td><td class=l>작은 물체는 박스가 조금만 어긋나도 IoU가 크게 떨어짐</td><td class=l>IoU 기준 FN과 실제 위치 미탐(중심 적중률)을 구분해 해석(3장)</td></tr>
<tr><td class=l>Geifman·El-Yaniv (2019) [5]</td><td class=l>확신이 낮은 예측은 자동 처리하지 않고 보류하는 개념(reject option)</td><td class=l>두 모델의 판단이 다르거나 애매하면 제품을 보류하는 판정 규칙(4장)</td></tr></tbody></table>""")
s('타 연구의 성능 수치는 본 데이터의 성능 근거로 쓰지 않았습니다. 방법을 고른 이유와 평가 설계를 뒷받침하는 데만 참고했습니다.')
o('데이터가 나타내는 공정 상태와 분석 목적')
d('KAMP 제공 X선 이물검출기 영상(1·2·3호기)입니다. 장비의 SD 카드에 자동 저장된 영상을 PC로 옮겨 수집했으며, 장비가 불량으로 판정한 경우에만 저장되므로 수집 주기는 불규칙합니다.')
s('원본은 BMP 2,809장(1호기 1,031·2호기 920·3호기 858장, 415MB)이며, 이 중 500장에 사람이 이물 위치를 라벨링한 YOLO 좌표(txt)가 함께 제공됩니다. 촬영일은 파일명 기준 2020.06.22~09.22입니다.')
d('각 영상은 슬롯 3개가 있는 시험편을 투과 촬영한 흑백 영상이며, 장비가 이물로 판정한 위치에 색 박스를 그려 저장한 NG 영상입니다.')
d('이물은 슬롯 끝에 놓인 지름 약 10px(5~21px)의 어두운 점입니다. 분석 목적은 영상만 보고 이물의 위치를 찾아 해당 제품을 보류할지 판단하는 것입니다.')
s('촬영 시각을 보면 같은 시험편을 약 4시간 간격으로 반복 촬영한 데이터입니다(아래 데이터 진단 결과 참고). 따라서 본 데이터의 성능은 "알려진 시험편을 다른 시각에 다시 촬영했을 때"의 탐지 성능입니다.')
o('문제 정의: 표시 박스에 의존하는 기존 모델')
d('KAMP 원본 영상에는 장비가 이물로 판정한 위치에 빨강·초록 등의 색 박스가 그려져 있습니다. 이 영상으로 학습한 기존 모델(KAMP 제공 YOLOv3-SPP, 806 epoch 체크포인트)은 이물이 아니라 표시 박스를 단서로 학습했습니다.')
d(f"같은 테스트 80장에서 기존 모델은 표시 박스가 있는 원본에서 이물 {BL['test_raw']['gt']}개 중 {BL['test_raw']['center_hits']}개({pct(BL['test_raw']['center_recall'],0)})의 위치를 맞혔지만, 표시 박스만 지운 영상에서는 하나도 찾지 못했습니다(0%). 검증 80장도 {pct(BL['val_raw']['center_recall'],0)} → 0%로 같았습니다.")
s(f"참고: KAMP 실습 자료의 YOLOv3 성능(정밀도 약 0.92, 재현율 약 0.95)은 표시 박스가 있는 영상에서 측정한 값입니다. 같은 KAMP 제공 체크포인트를 본 팀의 테스트 원본(표시 있음)에 적용하면 이물 중심 적중률은 {pct(BL['test_raw']['center_recall'],0)}였으며, 평가 영상과 라벨 기준이 달라 직접 비교하지는 않습니다.")
d('실제 생산 라인에서 판정해야 하는 영상은 표시 박스가 그려지기 전의 영상입니다. 표시 박스가 있어야만 동작하는 모델은 기존 장비의 판정을 따라 할 뿐, 새 영상에서 이물을 스스로 찾지 못합니다.')
d('따라서 본 과제의 목표를 "표시 박스가 없는 X선 영상에서 이물을 직접 판별하는 모델"로 정했습니다.')
a(f'<figure>{F("f11_shortcut.png")}<figcaption>그림 1. 기존 모델과 우리 모델의 비교. ①의 파란 박스는 기존 모델 예측, 빨간 박스는 장비 표시. ③의 빨간 박스는 우리 모델 예측, 초록 박스는 정답</figcaption></figure>')
o('주요 변수 정의')
a("""<table><thead><tr><th>변수</th><th>형식</th><th>의미 / 값</th></tr></thead><tbody>
<tr><td>영상</td><td>PNG, 8bit 흑백</td><td class=l>1호기 352×332 또는 316×332, 2호기 316×332, 3호기 576×444 px</td></tr>
<tr><td>이물 라벨</td><td>YOLO txt</td><td class=l>한 줄 = 이물 1개 (class 0, 중심 x·y, 너비·높이, 0~1 정규화). 빈 파일 = 이물 없음</td></tr>
<tr><td>호기</td><td>1·2·3</td><td class=l>검출기 장비 번호. 장비마다 영상 크기·밝기가 다름</td></tr>
<tr><td>촬영 시각·세션</td><td>파일명</td><td class=l>60초 이내 연속 촬영을 한 세션으로 묶음. 같은 세션은 한 분할에만 배치</td></tr>
<tr><td>라벨 출처</td><td>사람 / 약라벨</td><td class=l>사람 라벨 500장(KAMP 제공, 학습 340·검증 80·테스트 80), 라벨 없는 영상에 자동 생성한 약라벨 1,880장(아래 '라벨 확보' 참고)</td></tr>
<tr><td>가공 방법</td><td>폴더</td><td class=l>원본·이물 합성·이물 제거 등 6종 (아래 표)</td></tr></tbody></table>""")
o('전처리 내역')
d('장비 색 박스 제거: X-ray 영상은 사실상 흑백이므로, 주변과 달리 강한 색을 띠는 화소를 장비 표시 박스로 판단했습니다. 해당 영역을 제거한 뒤 주변 밝기 정보를 이용해 자연스럽게 복원했습니다.')
s('세부 기준: 화소의 RGB 최댓값과 최솟값 차이가 80 이상이고 최댓값이 180 이상이면 색 표시로 보고, 1px 넓힌 영역을 Telea 인페인팅(주변 화소로 빈 곳을 메우는 복원 방법)으로 채웠습니다.')
d('미끼 박스: 복원 흔적이 이물 위치에만 남으면 모델이 흔적을 외웁니다. 이물이 없는 제품 영역에도 같은 모양의 박스를 그렸다 지워 흔적을 위치와 무관하게 만들었습니다.')
d('잡음 재주입: 복원 부위가 매끈해지지 않도록 주변 잡음 세기에 맞춘 가우시안 잡음을 더했습니다.')
d('중복 제거와 세션 단위 분할: 내용이 같은 중복 영상을 제거했습니다. 60초 이내 연속 촬영된 영상은 같은 촬영 묶음으로 보고 하나의 세션으로 정의했습니다. 같은 세션의 영상이 학습셋과 검증·테스트셋에 동시에 들어가지 않도록 세션 전체를 한쪽에만 배정했으며, 호기별로 세션의 15%씩을 검증·테스트에 배정했습니다.')
s('영상을 무작위로 나누면 거의 같은 연속 촬영 영상이 학습과 평가에 함께 들어가 성능이 실제보다 좋게 보입니다. 이를 막기 위해 세션 단위로 나눴습니다.')
a(f'<figure>{F("f1_pipeline.png")}<figcaption>그림 2. 전처리 과정. ③의 주황 화소가 수정된 영역이며, 실제 이물 3곳과 미끼 3곳이 같은 방식으로 수정됩니다.</figcaption></figure>')
o('학습 데이터 구성과 각 가공의 목적')
a("""<table><thead><tr><th>폴더</th><th>장수</th><th>무엇을 개선하려는 가공인가</th></tr></thead><tbody>
<tr><td class=l>1 원본(사람 라벨)</td><td>340</td><td class=l>기준 데이터</td></tr>
<tr><td class=l>2 이물 합성: 제품 안 임의 위치</td><td>680</td><td class=l>이물이 늘 슬롯 끝에만 있어 생기는 위치 암기 방지</td></tr>
<tr><td class=l>3 이물 합성: 경계 강화+모양 변형</td><td>680</td><td class=l>회전·비율 변형과 제품 가장자리 위치로 형상·위치 다양화</td></tr>
<tr><td class=l>4 이물 전부 제거</td><td>340</td><td class=l>실제 정상 영상이 없어 "이물 없음" 사례를 합성 (오경보 감소 목적)</td></tr>
<tr><td class=l>5 이물 일부 제거</td><td>452</td><td class=l>같은 영상 안에 이물 있는/없는 슬롯 끝을 함께 보여 위치가 아닌 이물 자체를 학습</td></tr>
<tr><td class=l>6 약라벨 실제 영상</td><td>1,880</td><td class=l>사람 라벨이 없는 학습 세션 영상에 장비 색 박스를 라벨로 사용해 실제 영상 수 확대</td></tr></tbody></table>""")
s('약라벨 영상도 사람 라벨 영상과 같은 전처리(표시 박스 제거·미끼 박스·잡음 재주입)를 거쳤습니다. 라벨 생성 방법과 품질 점검은 아래 "라벨 확보"에 정리했습니다.')
s('색 박스가 없는 노란 선 영상 105장, 장비 화면의 T 아이콘이 라벨로 들어간 6장, 박스 두 개가 붙어 하나로 잡힌 25장은 제외했습니다.')
a(f'<figure>{F("f2_methods.png")}<figcaption>그림 3. 원본 1장과 그로부터 만든 가공 영상 (빨간 박스 = 학습 라벨)</figcaption></figure>')
a(f'<figure>{img(R+"/report/figs/f3_composition.png","82%")}<figcaption>그림 4. 학습 데이터 4,372장의 구성. 검증 80장(이물 183개)·테스트 80장(이물 172개)은 모두 사람 라벨 원본입니다.</figcaption></figure>')
o('라벨 확보: 라벨 없는 영상의 자동 라벨링과 품질 감사')
d('라벨링은 AI가 배울 정답을 표시하는 작업으로, 영상마다 이물 위치를 네모 박스로 표시해 텍스트 파일로 저장합니다. KAMP 원본 2,809장(중복 제거 후 2,516장) 중 사람이 라벨링한 영상은 500장뿐이고, 나머지 2,016장은 라벨이 없었습니다. 라벨 없는 영상은 모두 학습 세션에 속합니다.')
a("""<table><thead><tr><th>구분</th><th>장수</th><th>라벨 출처</th><th>사용처</th></tr></thead><tbody>
<tr><td class=l>사람 라벨</td><td>500</td><td class=l>KAMP 제공 라벨 그대로 사용 (500장 모두 좌표 일치 확인)</td><td class=l>학습 340 · 검증 80 · 테스트 80</td></tr>
<tr><td class=l>라벨 없던 영상</td><td>2,016</td><td class=l>자동 라벨링 대상</td><td class=l>학습 세션만</td></tr>
<tr><td class=l>└ 부적합 제외</td><td>136</td><td class=l>색 박스 없는 노란 선 105 · T 아이콘 6 · 붙은 박스 25</td><td class=l>사용 안 함</td></tr>
<tr><td class=l><b>└ 약라벨</b></td><td><b>1,880</b></td><td class=l>장비 색 표시 박스를 코드로 검출 → 박스 8px 축소</td><td class=l><b>학습에만 사용</b></td></tr></tbody></table>""")
d('약라벨(weak label)은 사람이 직접 그리지 않고 규칙으로 자동 생성해 사람 라벨보다 덜 정확할 수 있는 라벨입니다. 원본 영상에는 검출기가 이물로 판정한 위치에 색 표시 박스가 있으므로, ① 색 화소의 연결 영역을 찾아 박스 외곽을 얻고, ② 사람 라벨보다 박스가 큰 점을 보정해 가로·세로를 8px씩 줄였으며, ③ 표시 모양을 전수 분류해 라벨로 쓰면 안 되는 영상을 제외했습니다.')
d('이 작업은 AI 코딩 에이전트(Claude, Codex)에게 위임해 규칙 코드 작성·실행·감사를 맡기고, 팀이 결과를 확인하는 방식으로 진행했습니다. 에이전트가 영상을 한 장씩 보고 박스를 그린 것이 아니며, 학습된 모델의 예측을 라벨로 되돌려 쓰지도 않았습니다.')
a("""<pre># 약라벨 파일 예시: h1_002_20200629_082648(6)_weak.txt  (한 줄 = 이물 1개)
# 클래스  중심 x    중심 y    너비      높이   (영상 크기를 1로 둔 비율)
0 0.553977 0.412651 0.028409 0.030120
0 0.565341 0.478916 0.028409 0.030120
0 0.582386 0.575301 0.028409 0.030120</pre>""")
o2=LA['outline']; s8=LA['shrink8']; bs=LA['shrink8_by_split']
a(f"""<div class=tcap>표 1-1. 자동 라벨 규칙의 품질 감사 (사람 라벨 500장, 이물 {s8['gt']:,}개 대조)</div><table><thead><tr><th>자동 라벨 방식</th><th>IoU ≥ 0.5 일치</th><th>중심 3px 이내 일치</th><th>평균 중심 오차</th><th>일치쌍 평균 IoU</th></tr></thead><tbody>
<tr><td class=l>색 박스 외곽 그대로</td><td>{o2['iou50_match']:,} ({pct(o2['iou50_rate'])})</td><td>{o2['center3px_match']:,} ({pct(o2['center3px_rate'])})</td><td>{o2['mean_center_err']:.2f}px</td><td>{o2['mean_iou_matched']:.2f}</td></tr>
<tr><td class=l><b>8px 축소 규칙 (실제 약라벨)</b></td><td><b>{s8['iou50_match']:,} ({pct(s8['iou50_rate'])})</b></td><td>{s8['center3px_match']:,} ({pct(s8['center3px_rate'])})</td><td>{s8['mean_center_err']:.2f}px</td><td>{s8['mean_iou_matched']:.2f}</td></tr></tbody></table>""")
d(f"그래서: 색 박스는 이물 위치를 정확히 가리키지만(중심 오차 평균 {s8['mean_center_err']:.1f}px) 박스가 커서 그대로 쓰면 IoU 기준 일치가 {pct(o2['iou50_rate'],0)}에 그칩니다. 8px 축소 규칙을 적용하면 {pct(s8['iou50_rate'])}가 사람 라벨과 일치해, 약라벨로 학습 데이터를 늘릴 근거가 됩니다(세트별 학습 {bs['train'][0]}/{bs['train'][1]}, 검증 {bs['val'][0]}/{bs['val'][1]}, 테스트 {bs['test'][0]}/{bs['test'][1]}).")
a(f'<figure>{F("f12_label_audit.png")}<figcaption>그림 4-1. 자동 라벨과 사람 라벨 비교 (초록 = 사람 라벨, 노랑 = 색 박스 외곽, 자홍 = 8px 축소 약라벨). 오른쪽은 아래쪽 이물에서 박스 위치가 어긋난 불일치 사례</figcaption></figure>')
s('한계: 감사는 같은 규칙을 사람 라벨이 있는 500장에 적용해 추정한 것이며, 약라벨 1,880장 자체에는 사람 정답이 없습니다. 남은 약 5%의 불일치는 주로 박스 크기·경계 차이이며, 약라벨은 크기가 10~12px로 거의 고정이라 사람 라벨보다 크기 다양성이 적습니다. 검증·테스트 라벨은 감사에 읽기만 했고 수정하거나 학습에 쓰지 않았습니다.')
o('데이터 진단 결과')
d('파일 무결성: 전체 2,652개 영상과 라벨을 검사한 결과, 파일 누락이나 형식 오류는 없었으며 학습·검증·테스트셋 사이에 완전히 동일한 영상도 발견되지 않았습니다(3px 미만의 아주 작은 박스 1건만 확인).')
d('학습·평가 데이터 중복 점검: 학습에 쓴 가공 영상은 모두 학습 분할의 원본에서만 만들었고, 원본 파일·세션이 학습과 평가에 겹치는 경우는 없었습니다. 검증·테스트 영상과 시간적으로 가장 가까운 학습 영상도 촬영 시각이 최소 63분 이상 떨어져 있어, 같은 연속 촬영 장면이 학습과 평가에 동시에 포함된 경우는 없었습니다.')
d('같은 시험편 반복: 검증·테스트 160장 중 137장(86%)은 이물 배치가 거의 같은(제품 크기의 5% 이내) 학습 영상이 있습니다. 처음 보는 제품에 대한 성능은 이 데이터로 측정할 수 없습니다.')
d('라벨 누락 1건: 검증 영상 h2_002_20200623_123055(5)는 장비 표시가 3개인데 사람 라벨이 2개입니다. 탐지 모델 3종 모두 누락된 이물을 찾았고 오탐으로 집계됐습니다. 평가 기준의 일관성을 위해 라벨은 수정하지 않았습니다.')
d('박스 크기 불일치: IoU는 사람이 표시한 정답 박스와 AI가 예측한 박스가 얼마나 겹치는지를 나타내는 값입니다. 1에 가까울수록 두 박스가 잘 일치하며, 본 평가에서는 0.5 이상을 탐지 성공으로 인정했습니다.')
s('사람 라벨 크기가 7~15px로 넓게 퍼져 있어, 지름 10px 정도의 작은 이물에서는 박스가 2px만 달라도 IoU가 0.5 아래로 떨어질 수 있습니다(3장 참고).')
d('분포 편중: 이물은 모두 제품 가장자리에서 25px 이상 안쪽(슬롯 끝)에 있고, 실제 정상 제품 영상은 없습니다(원본 데이터에 양품 영상은 수집되지 않음). 가장자리 이물과 실제 오경보율은 이 데이터로 검증할 수 없습니다.')
a(f'<figure>{img(R+"/report/figs_extra/label_missing.jpg","78%")}<figcaption>그림 5. 검증셋 라벨 누락 사례. 왼쪽 장비 표시 3개, 오른쪽 사람 라벨 2개(가운데 슬롯 누락)</figcaption></figure>')
# ---------------- CH2 ----------------
a('<h2 class=chap>□ 제2장. AI 예측모델 개발 및 성능평가</h2>')
o('평가 설계')
d('모든 모델은 같은 학습 데이터 4,372장으로 학습하고, 같은 채점 코드로 평가했습니다. 정답과 예측은 IoU(두 박스의 겹침 정도) 0.5 이상일 때 같은 이물로 대응시켰습니다.')
d('이물 확신도(confidence score)는 모델이 검출한 후보를 이물이라고 판단하는 정도를 0~1로 나타낸 값이며, 값이 클수록 이물일 가능성을 높게 판단한 것입니다. 이 값이 임계값 이상인 후보만 이물로 판정합니다. 단, 이 값은 모델 내부의 판단 강도이며 정답과 비교해 잰 정확도나 실제 불량 확률이 아닙니다.')
d('이물 확신도 임계값은 검증셋 F1이 최대인 값으로 정하고, 테스트셋에는 그 값을 바꾸지 않고 한 번만 적용했습니다.')
a('<div class=tcap>평가 지표 설명</div><table><thead><tr><th style="width:24%">지표</th><th>의미</th></tr></thead><tbody>'
  '<tr><td>정밀도 (Precision)</td><td class=l>AI가 이물이라고 판단한 것 중 실제 이물이었던 비율</td></tr>'
  '<tr><td><b>재현율 (Recall)</b></td><td class=l><b>실제 이물 중 AI가 찾아낸 비율</b></td></tr>'
  '<tr><td>F1-score</td><td class=l>정밀도와 재현율을 함께 고려한 종합 지표 (두 값의 조화평균)</td></tr>'
  '<tr><td>FN</td><td class=l>실제 이물을 평가 기준(IoU 0.5)상 놓친 수</td></tr>'
  '<tr><td>FP</td><td class=l>실제 이물이 아닌 곳을 이물이라고 잘못 탐지한 수</td></tr>'
  '<tr><td>mAP50</td><td class=l>예측 박스와 정답 박스가 50% 이상 겹치면 성공으로 보는 탐지 성능</td></tr>'
  '<tr><td>mAP50-95</td><td class=l>겹침 기준을 50%부터 95%까지 점점 엄격하게 적용해 평균낸 더 까다로운 지표</td></tr>'
  '<tr><td>이물 중심 적중률</td><td class=l>예측 박스의 중심이 정답 박스 안에 들어간 비율. 박스 크기와 상관없이 이물 위치를 찾았는지를 봄</td></tr>'
  '<tr><td>합성 정상 오경보</td><td class=l>검증·테스트 영상에서 이물을 지워 만든 정상 영상 중 AI가 경보를 낸 영상 비율 (과검의 대리 지표)</td></tr>'
  '<tr><td>미끼 오탐</td><td class=l>표시 박스를 지운 흔적(미끼)을 이물로 잘못 탐지한 수</td></tr></tbody></table>')
d('식품 이물검사에서는 이물이 든 제품을 통과시키는 미검을 막는 것이 가장 중요하므로, 재현율과 FN을 주지표로 보고 정밀도·F1·mAP·FP를 보조지표로 삼았습니다.')
s('현장 용어와의 대응: FN은 미검, 합성 정상 영상의 오경보는 과검에 해당합니다. 성능 구간의 신뢰도는 세션 단위 부트스트랩 95% 신뢰구간으로 함께 확인했습니다.')
o('비교 모델과 학습 조건')
a(f"""<table><thead><tr><th>모델</th><th>선정 이유</th><th>구조</th><th>학습 조건</th><th>학습 시간</th><th>1장 처리</th></tr></thead><tbody>
<tr><td class=l>YOLOv8n (기준선)</td><td class=l>가볍고 널리 쓰이는 기준 모델</td><td class=l>CNN 1단계 탐지, 3.0M</td><td class=l>640px, 최대 80ep, patience 20 → 31ep 종료</td><td>{S1['yolov8n']['meta']['train_seconds']/3600:.1f} h</td><td>{ms('yolov8n')} ms</td></tr>
<tr><td class=l>YOLO11s</td><td class=l>최신 YOLO 계열 성능 확인</td><td class=l>CNN 1단계 탐지, 9.4M</td><td class=l>640px, 최대 70ep, patience 20 → 55ep 종료</td><td>{S1['yolo11s']['meta']['train_seconds']/3600:.1f} h</td><td>{ms('yolo11s')} ms</td></tr>
<tr><td class=l>RF-DETR-S</td><td class=l>CNN 계열과 다른 Transformer 기반 탐지 모델 비교</td><td class=l>트랜스포머(DINOv2 기반), 31.8M</td><td class=l>512px, 15ep, 배치 4×누적 4</td><td>{S1['rfdetr_s']['meta']['train_seconds']/3600:.1f} h</td><td>{ms('rfdetr_s')} ms</td></tr>
<tr><td class=l>Qwen2.5-VL-3B 제로샷</td><td class=l rowspan=2>범용 비전-언어 모델이 전문 탐지 모델을 대체할 수 있는지 확인</td><td class=l>비전-언어 모델</td><td class=l>학습 없음, 박스 JSON 출력 지시</td><td>-</td><td>{ms('qwen25vl3b_zeroshot')} ms</td></tr>
<tr><td class=l>Qwen2.5-VL-3B LoRA</td><td class=l>비전-언어 모델 + LoRA(r=16)</td><td class=l>1,300장 × 2ep, 영상 1.5배 확대</td><td>{S1['qwen25vl3b_lora']['meta']['train_seconds']/3600:.1f} h</td><td>{ms('qwen25vl3b_lora')} ms</td></tr></tbody></table>""")
s('비전-언어 모델(VLM): 이미지와 문장을 함께 이해하는 범용 AI 모델. LoRA: 전체 모델을 다시 학습하지 않고 일부 추가 파라미터만 학습하는 경량 미세조정 방식.')
s('장치: Apple M5 Pro (MPS), Python 3.12.14, PyTorch 2.14.1. 사전학습 가중치는 공개 COCO 가중치만 사용했습니다.')
o('동일 조건 성능 비교')
a('<div class=tcap>표 1. 모델별 성능 (학습 데이터 D 구성 4,372장, 검증 80장·테스트 80장, 임계값은 검증에서 결정)</div>' + main_table(['rfdetr_s','yolo11s','yolov8n','qwen25vl3b_lora','qwen25vl3b_zeroshot']))
a(f'<figure>{F("f4_models.png")}<figcaption>그림 6. 모델별 지표와 미탐 수</figcaption></figure>')
s('표 읽는 법: mAP50은 박스가 50% 이상 겹치면 성공으로 보는 탐지 성능, mAP50-95는 겹침 기준을 95%까지 엄격하게 올려 평균낸 값이며, 두 값 모두 높을수록 탐지 위치와 박스 크기가 정답과 잘 맞는다는 뜻입니다. 굵은 글씨는 상위 후보(RF-DETR-S, YOLO11s)의 주요 수치입니다.')
d(f"계산 예시: 최종 모델(RF-DETR-S 기준)의 테스트 결과는 TP {rf['test']['TP']}개(맞힌 이물), FP {rf['test']['FP']}개(잘못 표시), FN {rf['test']['FN']}개(평가상 놓친 이물)입니다. 정밀도 = {rf['test']['TP']}/({rf['test']['TP']}+{rf['test']['FP']}) = {pct(rf['test']['precision'])}, 재현율 = {rf['test']['TP']}/({rf['test']['TP']}+{rf['test']['FN']}) = {pct(rf['test']['recall'])}, F1 = 2×{rf['test']['TP']}/(2×{rf['test']['TP']}+{rf['test']['FP']}+{rf['test']['FN']}) = {rf['test']['F1']:.3f}입니다. 모두 이물 단위 점수이며 제품(영상) 단위 정확도와는 다릅니다.")
d(f"참고로 KAMP 제공 기존 모델(YOLOv3)은 표시 박스가 없는 같은 테스트 영상에서 이물을 하나도 찾지 못했습니다(중심 적중률 0%, 1장 참고). 아래 모델은 모두 표시 박스가 없는 영상으로 학습·평가했습니다.")
d(f"RF-DETR-S는 검증·테스트 모두 미탐이 가장 적었고(FN {rf['val']['FN']}·{rf['test']['FN']}) mAP50·mAP50-95도 가장 높았습니다.")
d(f"테스트 재현율의 세션 단위 95% 신뢰구간은 RF-DETR-S {pct(CI['rfdetr_s']['recall_CI95'][0])}~{pct(CI['rfdetr_s']['recall_CI95'][1])}, YOLO11s {pct(CI['yolo11s']['recall_CI95'][0])}~{pct(CI['yolo11s']['recall_CI95'][1])}, YOLOv8n {pct(CI['yolov8n']['recall_CI95'][0])}~{pct(CI['yolov8n']['recall_CI95'][1])}입니다. 상위 두 모델의 차이는 통계적으로 구분되지 않습니다.")
d('탐지 모델 3종 모두 이물 중심 적중률이 100%입니다. 즉 IoU 기준으로 집계된 FN도 이물 위치는 모두 찾았고, 박스 크기 차이 때문에 미탐으로 집계된 것입니다(3장).')
d('비전-언어 모델(VLM)은 LoRA 미세조정으로 재현율이 3.8%에서 80.3%(검증)로 올랐지만, 박스 정확도가 낮고 이물 확신도를 내지 않아 임계값 조정이 불가능하며 처리 시간이 탐지 모델의 50배 이상입니다.')
o('상위 후보 모델 선정')
d('표 1 결과로 RF-DETR-S와 YOLO11s를 상위 후보로 정했습니다. 이후 ① 학습 데이터 구성을 검증하고, ② 두 후보를 함께 쓰는 혼용 방식을 비교해 최종 모델 구성을 정했습니다.')
d(f"RF-DETR-S: 실제 검증·테스트에서 미탐이 가장 적고(FN 5·2), 1장 {ms('rfdetr_s')}ms로 실시간 검사에 쓸 수 있습니다.")
d(f"YOLO11s: 원래 대비에서는 RF-DETR-S와 비슷하고, 흐린 이물(저대비)에서 더 강해 RF-DETR-S의 약점을 보완할 수 있습니다. 처리 시간은 1장 {ms('yolo11s')}ms로 가장 빠른 축입니다.")
s('YOLOv8n은 기준선으로 남기고, 비전-언어 모델은 박스 정확도와 처리 속도 문제로 후보에서 제외했습니다.')
o('학습 데이터 구성 검증 ① YOLOv8n으로 6가지 구성 비교')
d('목적: 학습 데이터 4,372장 중 원본에서 만든 가공 영상이 실제 영상의 미탐을 줄이는지 확인했습니다. YOLOv8n을 같은 조건(30 epoch, 조기 종료 없음)으로 학습하고, 실제 검증셋으로만 판정했습니다.')
d('판정 기준: 실험 전, FN이 3건 이상 변하고 반복 학습에서도 같은 방향이 확인될 때 의미 있는 변화로 판단하기로 정했습니다. A와 D는 학습 시드를 바꿔 두 번 학습했습니다.')
s('학습 시드(seed)는 데이터 순서와 초기값 등 학습 과정의 무작위성을 고정하는 값입니다. 같은 조건에서도 시드에 따라 결과가 달라질 수 있어 일부 실험은 시드를 바꿔 반복했습니다.')
d('저대비 평가: k는 이물과 주변 배경의 명암 차이를 얼마나 줄였는지를 나타냅니다. k=0.5는 원래 이물 대비를 절반 수준으로 낮춰 더 흐리게 만든 조건입니다. 흐린 이물에 대한 대응력을 보기 위한 평가 전용 합성 영상이며 학습에는 쓰지 않았습니다.')
EXN={'A_실제만':('A','실제 영상만'),'A_실제만_seed1':('A′','A, 시드 1'),'B_실제+이물합성':('B','A + 이물 합성'),'C_실제+이물제거':('C','A + 이물 제거'),'D_전체':('D','전체'),'D_전체_seed1':('D′','D, 시드 1'),'E_선택+960':('E','B + 입력 960px'),'F_전체+동료합성':('F','D + 별도 합성 762장')}
h='<div class=tcap>표 3. 학습 데이터 조합별 결과 (YOLOv8n, 30 epoch)</div><table><thead><tr><th>실험</th><th>학습 데이터</th><th>장수</th><th>검증 FN</th><th>검증 FP</th><th>검증 재현율</th><th>검증 mAP50</th><th>검증 mAP50-95</th><th>저대비 k=0.5<br>중심 적중률</th><th>저대비 k=0.35<br>중심 적중률</th><th>테스트 FN</th><th>학습 시간</th></tr></thead><tbody>'
for e,(code,lab) in EXN.items():
    r=P2[e]; v=r['eval']['val']; t=r['eval']['test']; m=r['meta']
    k5=(r['val_stress']['050']['center']+r['test_stress']['050']['center'])/2; k35=(r['val_stress']['035']['center']+r['test_stress']['035']['center'])/2
    h+=f"<tr><td>{code}</td><td class=l>{lab}</td><td>{m['train_images']:,}</td><td><b>{v['FN']}</b></td><td>{v['FP']}</td><td>{pct(v['recall'])}</td><td>{v['AP50']:.3f}</td><td>{v['AP50_95']:.3f}</td><td><b>{pct(k5,0)}</b></td><td>{pct(k35,0)}</td><td>{t['FN']}</td><td>{m['train_seconds']/60:.0f}분</td></tr>"
a(h+'</tbody></table><div class=cap>저대비 적중률은 검증·테스트 평균. 저대비 영상은 평가 전용 합성이며 학습에 쓰지 않았습니다.</div>')
a(f'<figure>{F("f10_ablation.png")}<figcaption>그림 7. 학습 데이터 조합별 미탐 수와 저대비 대응력</figcaption></figure>')
a('<div class=box><b>결론:</b> YOLOv8n에서는 합성 데이터를 많이 추가하는 것이 일반 탐지 성능을 개선하지 않았으며, 실제 영상만 사용한 A가 저대비 조건에서 가장 강했습니다(k=0.5 적중률 A 79%, 전체 결합 D 두 시드 평균 47%).</div>')
a("""<table><thead><tr><th>가공·조건</th><th>왜 했나</th><th>결과</th><th>결론</th></tr></thead><tbody>
<tr><td class=l>이물 합성 (B)</td><td class=l>위치 암기 방지, 형상·위치 다양화</td><td class=l>검증 FN 5 (A: 5·7). 저대비 k=0.5 적중률 79% → 60%</td><td class=l>미탐 감소 효과 확인 안 됨. 저대비 대응력은 오히려 낮아짐</td></tr>
<tr><td class=l>이물 제거 (C)</td><td class=l>이물 없는 사례로 오탐·위치 암기 감소</td><td class=l>검증 FN 5, FP 6 (A: FP 5). 적중률 73%</td><td class=l>오탐·미탐 모두 차이 없음</td></tr>
<tr><td class=l>전체 결합 (D)</td><td class=l>모든 가공을 합친 효과</td><td class=l>검증 FN 9·8 (두 시드), 적중률 37%·58% (평균 47%)</td><td class=l>미탐이 평균 2.5개 늘었으나 판정 기준(3개) 미만. 저대비 대응력이 가장 낮음</td></tr>
<tr><td class=l>입력 960px (E)</td><td class=l>작은 이물의 박스 불일치 감소</td><td class=l>검증 FN 9 (B: 5), 작은 이물 FN 10 (B: 7), 학습 2.1배</td><td class=l>개선 없음. 원본이 316~576px라 확대해도 정보가 늘지 않음</td></tr>
<tr><td class=l>별도 생성 합성 (F)</td><td class=l>다른 방식의 합성 추가 효과</td><td class=l>검증 FN 9, 적중률 66%</td><td class=l>개선 없음</td></tr></tbody></table>""")
d('가능한 원인(가설): 이물 제거·합성 영상에는 지운 자리의 희미한 흔적이 "이물 없음"으로 라벨되어 있어, 모델이 희미한 어두운 점을 정상으로 학습했을 수 있습니다.')
d('다만 이 비교는 YOLOv8n 한 모델의 결과이므로, 최종 후보 모델에서도 같은 결과가 나오는지 아래에서 다시 확인했습니다.')
o('학습 데이터 구성 검증 ② 상위 후보 모델로 A·D 재확인')
d('YOLO11s와 RF-DETR-S를 A 구성(실제 영상 2,220장)으로 다시 학습해, 기존 D 구성(4,372장) 학습 결과와 비교했습니다. 학습 조건은 데이터만 바꾸고 나머지는 같게 했습니다.')
d('채택 여부는 각 학습 전에 정한 기준으로 검증셋에서 먼저 판정했습니다. 공통 기준은 저대비 k=0.5 적중률이 5%p 이상 오르고, 합성 정상 오경보가 5%p 넘게 늘지 않으며, 미끼 오탐이 없는 것입니다. 미탐 기준은 YOLO11s 단계에서 "검증 FN이 늘지 않을 것", RF-DETR-S 추가 단계에서 "FN 1개 이내 증가는 허용하되 실제로 놓친 이물이 없을 것"으로 두었습니다. 두 단계 모두 기준을 통과하지 못했습니다.')
d('아래 테스트 결과는 판정이 끝난 뒤 기록용으로 한 번 계산한 값이며, 선택에는 쓰지 않았습니다.')
AK=[('YOLO11s (D)','YOLO11s','D'),('YOLO11s (A)','YOLO11s','A'),('RF-DETR-S (D)','RF-DETR-S','D'),('RF-DETR-S (A)','RF-DETR-S','A'),('S1: RF-DETR-S(D)+YOLO11s(D) [최종]','혼용 S1','D + D (최종)'),('S1: RF-DETR-S(A)+YOLO11s(A)','혼용 S1','A + A')]
h='<div class=tcap>표 3-2. 상위 후보 모델의 학습 데이터 구성 비교 (임계값은 검증에서 결정)</div><table><thead><tr><th>모델</th><th>학습 데이터</th><th>검증 FN</th><th>테스트 FN</th><th>검증 재현율</th><th>테스트 재현율</th><th>F1<br>검증 / 테스트</th><th>저대비 k=0.5<br>검증 / 테스트</th><th>합성 정상 오경보<br>검증 / 테스트</th></tr></thead><tbody>'
for key,mod,dat in AK:
    r=AD[key]; v=r['val']; te=r['test']
    h+=f"<tr><td class=l>{mod}</td><td>{dat}</td><td><b>{v['FN']}</b></td><td><b>{te['FN']}</b></td><td>{pct(v['recall'])}</td><td>{pct(te['recall'])}</td><td>{2*v['TP']/(2*v['TP']+v['FP']+v['FN']):.3f} / {2*te['TP']/(2*te['TP']+te['FP']+te['FN']):.3f}</td><td>{pct(v['k050'],0)} / {pct(te['k050'],0)}</td><td>{pct(v['normal_FA'])} / {pct(te['normal_FA'])}</td></tr>"
a(h+'</tbody></table>')
d(f"저대비: A 구성은 두 모델 모두 저대비 적중률을 크게 높였습니다(검증 k=0.5 기준 YOLO11s {pct(AD['YOLO11s (D)']['val']['k050'],0)} → {pct(AD['YOLO11s (A)']['val']['k050'],0)}, RF-DETR-S {pct(AD['RF-DETR-S (D)']['val']['k050'],0)} → {pct(AD['RF-DETR-S (A)']['val']['k050'],0)}). YOLOv8n에서 본 경향이 그대로 재현됐습니다.")
d(f"실제 영상: 테스트 FN은 A 구성에서 오히려 늘었습니다(YOLO11s {AD['YOLO11s (D)']['test']['FN']} → {AD['YOLO11s (A)']['test']['FN']}, RF-DETR-S {AD['RF-DETR-S (D)']['test']['FN']} → {AD['RF-DETR-S (A)']['test']['FN']}). 혼용 시 검증에서 실제로 놓친 이물이 1개 생겼습니다(3호기, 한 변 8px).")
d(f"오경보: RF-DETR-S(A)는 합성 정상 영상의 {pct(AD['RF-DETR-S (A)']['val']['normal_FA'],0)}(검증)에 경보를 냈습니다. 경보 46개 중 45개가 원래 이물을 지운 자리에서 나왔습니다. A 구성에는 이물을 지운 영상이 없어, 모델이 지운 흔적 같은 희미한 어두운 점도 이물로 보게 된 것입니다.")
a('<div class=box><b>결론:</b> A 구성은 저대비 이물에는 더 민감했지만 실제 영상의 미탐과 정상 영상의 오경보가 증가하여, 최종 학습 데이터는 D 구성을 유지했습니다. YOLOv8n에서는 A가 유리했지만 상위 후보 모델에서는 D가 더 안정적이었으며, 데이터 구성의 효과는 모델 구조에 따라 달랐습니다.</div>')
s('이 결과는 앞서 가설로 둔 원인(이물 제거 영상이 희미한 흔적을 "정상"으로 가르쳐 저대비 대응력을 낮춤)을 뒷받침합니다. 즉 저대비 이물 탐지와 정상 제품 오경보는 서로 맞바꾸는 관계이며, 어느 쪽에 맞출지는 실제 흐린 시험편과 실제 정상 제품으로만 정할 수 있습니다.')
o('최종 모델 구성: 두 모델 혼용 방식 결정')
d('목적: 원래 대비에서 가장 좋은 RF-DETR-S와 저대비에 강한 YOLO11s를 함께 쓰면 두 장점을 모두 얻을 수 있는지 확인했습니다. 다시 학습하지 않고 저장된 예측을 조합했습니다.')
d('S1(영상 단위 선택): 영상마다 두 모델의 최고 이물 확신도를 비교해 더 높은 모델의 예측만 사용합니다. S2(박스 단위 선택): 같은 이물(IoU ≥ 0.3)로 겹치는 박스 중 이물 확신도가 높은 박스를 남깁니다. S3: S2와 같되 각 모델의 이물 확신도를 자기 검증 임계값으로 나눠 비교합니다.')
d('방식과 임계값은 검증셋에서만 정하고, 테스트셋에는 그대로 한 번 적용했습니다.')
ENN={'rfdetr_s':'RF-DETR-S 단독','yolo11s':'YOLO11s 단독','rfdetr_s+yolo11s|S1':'RF-DETR-S + YOLO11s · S1','rfdetr_s+yolo11s|S2':'RF-DETR-S + YOLO11s · S2','rfdetr_s+yolo11s|S3':'RF-DETR-S + YOLO11s · S3'}
h='<div class=tcap>표 4. 두 모델 혼용 결과 (임계값은 검증에서 결정)</div><table><thead><tr><th>구성</th><th>세트</th><th>FN</th><th>FP</th><th>재현율</th><th>정밀도</th><th>mAP50</th><th>mAP50-95</th><th>정상 오경보</th><th>저대비 k=0.5<br>중심 적중률</th><th>저대비 k=0.35<br>중심 적중률</th></tr></thead><tbody>'
for k,lab in ENN.items():
    for sp,sl in (('val','검증'),('test','테스트')):
        r=ENS[sp][k]
        h+=f"<tr>{f'<td rowspan=2 class=l><b>{lab}</b></td>' if sp=='val' else ''}<td>{sl}</td><td><b>{r['FN']}</b></td><td>{r['FP']}</td><td>{pct(r['recall'])}</td><td>{pct(r['precision'])}</td><td>{r['AP50']:.3f}</td><td>{r['AP50_95']:.3f}</td><td>{pct(r['normal_FA'])}</td><td><b>{pct(r['stress']['050'],0)}</b></td><td>{pct(r['stress']['035'],0)}</td></tr>"
a(h+'</tbody></table>')
d(f"S1(RF-DETR-S + YOLO11s)은 원래 대비에서 RF-DETR-S 단독과 같은 성능(테스트 FN {ENS['test']['rfdetr_s+yolo11s|S1']['FN']}, 재현율 {pct(ENS['test']['rfdetr_s+yolo11s|S1']['recall'])})을 유지하면서, 대비를 절반으로 낮춘 이물의 중심 적중률을 검증 {pct(ENS['val']['rfdetr_s']['stress']['050'],0)} → {pct(ENS['val']['rfdetr_s+yolo11s|S1']['stress']['050'],0)}, 테스트 {pct(ENS['test']['rfdetr_s']['stress']['050'],0)} → {pct(ENS['test']['rfdetr_s+yolo11s|S1']['stress']['050'],0)}로 높였습니다. 검증에서 본 경향이 테스트에서도 같은 방향으로 확인됐습니다.")
d('원래 대비의 미탐은 줄지 않았습니다. 남은 미탐은 두 모델 모두 위치는 맞혔으나 박스 크기가 달라 생긴 것이어서, 어느 모델의 박스를 골라도 해결되지 않습니다. 검증 영상 80장 중 71장은 RF-DETR-S의 점수가 더 높아 RF-DETR-S 예측이 쓰였습니다.')
d('S2·S3는 S1과 미탐이 같거나 많고 저대비 적중률이 낮아, 가장 단순한 S1을 선택했습니다.')
d('YOLOv8n과의 조합도 비교했으나 개선이 없어 최종 후보에서 제외했습니다.')
s('S1은 영상마다 한 모델의 예측만 쓰므로, 흐린 이물 하나를 다른 모델만 찾았더라도 그 영상에서 이물 확신도가 낮은 쪽이면 버려집니다. 또 이물 확신도 척도가 다른 모델을 넣으면 어느 모델이 선택되는지가 크게 바뀝니다(D 구성 YOLO11s와 조합 시 RF-DETR-S 선택 71/80장, A 구성 YOLO11s와 조합 시 23/80장). 이 한계는 4장의 "두 모델 판단 불일치 시 보류" 규칙으로 보완합니다.')
s('검증 결과는 같은 데이터에서 방식을 고르고 평가한 값이라 낙관적일 수 있습니다. 저대비 평가는 평가 전용 합성 영상 기준입니다.')
a(f"<div class=box><b>최종 모델:</b> D 구성으로 학습한 RF-DETR-S와 YOLO11s를 영상 단위로 혼용(S1)합니다. 영상마다 두 모델 중 이물 확신도가 높은 모델의 예측을 사용하고, 두 모델의 판단이 엇갈리면 재검사로 보냅니다(4장). YOLOv8n 기준선 대비 테스트 FN 7 → 2, 정밀도 95.4% → 98.8%이며, 두 모델 처리 시간 합은 1장 약 45ms입니다.</div>")

# ---------------- CH3 ----------------
a('<h2 class=chap>□ 제3장. 영향요인 및 오류분석</h2>')
o('IoU 기준 미탐(FN)의 원인 분석')
d('여기서 미탐(FN)은 이물 위치를 전혀 찾지 못했다는 뜻이 아니라, 예측 박스와 정답 박스의 겹침 정도(IoU)가 0.5 미만이어서 평가상 미탐으로 집계된 경우를 포함합니다.')
a('<table><thead><tr><th>모델</th><th>박스 불일치<br>(이물 위 예측, IoU 0.1~0.5)</th><th>확신도 미달</th><th>후보 없음</th><th>이물 중심 적중률</th></tr></thead><tbody>' + ''.join(f"<tr><td class=l>{N[k]}</td><td>{S1[k]['val_fn_types']['box_mismatch']} / {S1[k]['test_fn_types']['box_mismatch']}</td><td>{S1[k]['val_fn_types']['below_threshold']} / {S1[k]['test_fn_types']['below_threshold']}</td><td>{S1[k]['val_fn_types']['no_candidate']} / {S1[k]['test_fn_types']['no_candidate']}</td><td>{pct(S1[k]['val_center_recall'])} / {pct(S1[k]['test_center_recall'])}</td></tr>" for k in ('rfdetr_s','yolo11s','yolov8n','qwen25vl3b_lora')) + '</tbody></table><div class=cap>값은 검증 / 테스트</div>')
d('탐지 모델에서 발생한 FN은 모두 이물 위치에는 예측이 있었으나, 정답 박스와 예측 박스의 크기 차이로 IoU가 0.31~0.50에 그친 사례였습니다. 정답 박스가 유난히 작거나(한 변 5~8px) 크거나 길쭉한(한 변 11px 이상) 이물에서, 모델은 학습 라벨의 대표 크기인 약 10px 박스를 그렸습니다. 작은 물체일수록 같은 위치 차이에도 IoU가 크게 떨어진다는 선행연구[4]와 같은 현상입니다.')
d('따라서 임계값을 낮춰도 FN이 줄지 않습니다. 일부 이물은 박스 크기 차이로 FN으로 집계됐지만, 이물이 포함된 영상을 정상 제품으로 잘못 통과시킨 사례는 없었습니다.')
a(f'<figure>{F("f8_fn_gallery.png")}<figcaption>그림 8. RF-DETR-S·YOLO11s의 미탐 전체 (초록 = 정답, 빨강 = 예측)</figcaption></figure>')
o('조건별 미탐 (검증+테스트 이물 355개)')
a('<table><thead><tr><th>요인</th><th>구간</th><th>이물 수</th><th>RF-DETR-S<br>FN (재현율)</th><th>YOLO11s<br>FN (재현율)</th><th>YOLOv8n<br>FN (재현율)</th></tr></thead><tbody>' + ''.join(fail_rows(f) for f in ('크기','대비','배경 밝기','호기','형상','위치')) + '</tbody></table>')
s(f"구간 기준: 이물 크기 = 박스 면적의 제곱근 3분위({RA['bins']['size_px'][0]:.1f}, {RA['bins']['size_px'][1]:.1f}px), 대비 = (주변 밝기 중앙값 − 이물 어두운 25% 평균) ÷ 주변 잡음 3분위, 배경 밝기 = 주변 밝기 3분위, 가장자리 = 제품 경계까지 25px 미만")
a(f'<figure>{F("f6_failure.png")}<figcaption>그림 9. 조건별 미탐 수</figcaption></figure>')
s('그래서: FN은 주로 작은 이물과 영상이 큰 3호기에서 나오며, 박스 크기를 맞추기 어려운 조건에 몰려 있습니다.')
d('크기: 미탐이 작은 이물(9.5px 미만)에 집중됩니다(RF-DETR-S 7개 중 6개). 작은 이물에서 2px 박스 차이가 IoU를 크게 낮추기 때문입니다.')
d('호기: 2호기는 세 모델 모두 미탐 0개이며, 미탐은 1·3호기에서만 나왔습니다. 3호기는 영상이 커서 같은 이물이 입력에서 더 작아집니다.')
d('대비·배경: 실제 데이터 범위에서는 뚜렷한 경향이 없습니다. 위치: 평가셋에 가장자리 이물이 0개라 분석할 수 없습니다.')
o('저대비 이물 취약성 (평가 전용 합성)')
d('검증·테스트 실제 이물의 배경 대비를 k배(0.75, 0.5, 0.35, 0.25)로 낮춘 영상으로 같은 모델을 평가했습니다. 학습에는 쓰지 않았습니다.')
a(f'<figure>{F("f7_stress.png")}<figcaption>그림 10. 이물 대비를 낮췄을 때의 탐지율 (실선 검증, 점선 테스트)</figcaption></figure>')
d(f"k = 0.75까지는 세 모델 모두 이물 중심 적중률 93% 이상을 유지하지만, k = 0.5에서 RF-DETR-S {pct(st('rfdetr_s','val',0.5),0)}·YOLO11s {pct(st('yolo11s','val',0.5),0)}·YOLOv8n {pct(st('yolov8n','val',0.5),0)}(검증)로 떨어지고, k = 0.35에서는 절반 이상을 놓칩니다.")
d('원래 대비에서 가장 좋은 RF-DETR-S보다 YOLO11s가 저대비에서 더 강합니다. 이 결과가 두 모델을 함께 쓰는 근거이며, 실제로 혼용(S1) 시 저대비 적중률이 단일 모델보다 높았습니다(표 4). 합성 조건이므로 실제 흐릿한 이물의 탐지율로 해석하지 않습니다.')
d('실제 영상만 사용한 A 구성은 흐린 이물을 더 잘 탐지했지만, 동시에 정상에 가까운 흔적까지 이물로 판단해 오경보가 증가했습니다(표 3-2). 즉 저대비 이물 탐지 성능을 높일수록 정상 제품을 잘못 경고할 위험도 함께 커지는 경향이 확인됐습니다.')
o('두 모델이 서로 보완한 오류와 함께 놓친 오류')
d('두 모델은 같은 영상을 각각 따로 검사합니다. 같은 데이터로 학습했으므로 오류가 완전히 독립적이지는 않아, 함께 놓친 이물은 확률을 곱해 추정하지 않고 같은 정답 이물에서 직접 셌습니다.')
a('<table><thead><tr><th>세트</th><th>이물 수</th><th>RF-DETR-S FN</th><th>YOLO11s FN</th><th>RF-DETR-S가 보완<br>(YOLO11s만 놓침)</th><th>YOLO11s가 보완<br>(RF-DETR-S만 놓침)</th><th>함께 놓친 FN</th></tr></thead><tbody>' + ''.join(f"<tr><td>{lab}</td><td>{CP[sp]['objects']}</td><td>{CP[sp]['rf_FN']}</td><td>{CP[sp]['y11_FN']}</td><td>{CP[sp]['rf_covers_y11']}</td><td>{CP[sp]['y11_covers_rf']}</td><td><b>{CP[sp]['common_FN']}</b></td></tr>" for sp, lab in (('val','검증'),('test','테스트'))) + '</tbody></table>')
d(f"그래서: 검증·테스트에서 두 모델이 각각 IoU 기준으로 놓친 FN {CP['val']['rf_FN']+CP['val']['y11_FN']+CP['test']['rf_FN']+CP['test']['y11_FN']}건 중 {CP['val']['rf_covers_y11']+CP['val']['y11_covers_rf']+CP['test']['rf_covers_y11']+CP['test']['y11_covers_rf']}건은 다른 모델이 맞혔습니다. 함께 놓친 FN(검증 {CP['val']['common_FN']}개, 테스트 {CP['test']['common_FN']}개)도 두 모델 모두 이물 위치는 찾았고 박스 크기만 달랐습니다. 두 모델을 함께 쓰고 판단이 다르면 사람이 확인하는 4장의 구조가 이 결과에 근거합니다.")
s('정답 라벨로 계산한 사후 진단이며, 두 모델을 결합한 탐지기의 F1이 아닙니다.')
o('오탐 분석')
d('오탐은 대부분 박스 불일치로 생긴 같은 이물의 중복 집계입니다. 이물이 없는 곳의 오탐은 세 모델 합쳐 1건(YOLOv8n 테스트), 라벨 누락 이물을 찾은 경우가 각 모델 1건입니다.')
d('미끼 박스(복원 흔적)에 반응한 오탐은 모든 탐지 모델에서 0건으로, 모델이 장비 표시 흔적에 의존하지 않음을 확인했습니다.')
o('이물 확신도(Confidence) 임계값 분석')
d('Confidence 임계값은 모델이 어느 정도 이상 확신했을 때 이물로 인정할지를 정하는 기준입니다. 임계값을 낮추면 작은 의심까지 잡아 미검을 줄일 수 있지만 오탐이 늘고, 높이면 오탐은 줄지만 실제 이물을 놓칠 수 있습니다.')
a(f'<figure>{F("f5_threshold.png")}<figcaption>그림 11. 임계값별 FN·FP·합성 정상 오경보율 (검증셋, 세로선 = 선택 임계값)</figcaption></figure>')
a('<div class=tcap>표 2. RF-DETR-S 임계값별 결과</div>' + sweep_table('rfdetr_s'))
d('미탐은 임계값 0.05~0.5 구간에서 거의 변하지 않고 0.6을 넘으면 급증합니다. 반대로 0.1 이하에서는 오탐과 정상 오경보가 급증합니다.')
s('그래서: 0.1~0.5 사이에서는 미검과 오탐이 모두 안정적이므로, 이 구간 안에서 확정 판정선과 재검사 하한을 따로 둘 수 있습니다.')
d(f"안전 측면의 임계값 후보: 확정 판정선은 검증 F1 최대값(RF-DETR-S {TH['rfdetr_s']:.3f}, YOLO11s {TH['yolo11s']:.3f}), 재검사 하한은 합성 정상 경보율 10% 이하가 되는 가장 낮은 값(RF-DETR-S {TL['rfdetr_s']:.2f}, YOLO11s {TL['yolo11s']:.2f})입니다. 0.5를 그대로 쓰지 않고 검증 결과로 정했습니다.")
# ---------------- CH4 ----------------
a('<h2 class=chap>□ 제4장. 현장 활용방안</h2>')
o('표시 박스 없이 동작하는 독립 판정기')
a('''<figure><svg viewBox="0 0 720 470" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Noto Sans CJK KR, sans-serif"><defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs><g stroke="#444" stroke-width="1.6" fill="none" marker-end="url(#ar)"><path d="M360,64 L360,84 L185,84 L185,104"/><path d="M360,64 L360,84 L535,84 L535,104"/><path d="M185,176 L185,206 L330,206"/><path d="M535,176 L535,206 L390,206" /><path d="M360,232 L360,262"/><path d="M360,300 L360,320 L120,320 L120,342"/><path d="M360,300 L360,342"/><path d="M360,300 L360,320 L600,320 L600,342"/><path d="M360,402 L360,424"/></g><rect x="240" y="16" width="240" height="48" rx="6" fill="#eceff1" stroke="#555"/><text x="360" y="37" text-anchor="middle" font-size="18.3" font-weight="700">X-ray 원본 영상</text><text x="360" y="55" text-anchor="middle" font-size="14.0" fill="#333">장비 표시 박스가 그려지기 전</text><rect x="70" y="104" width="230" height="72" rx="6" fill="#fff" stroke="#555"/><text x="185" y="133" text-anchor="middle" font-size="17.1" font-weight="700">기존 검사장비 판정</text><text x="185" y="153" text-anchor="middle" font-size="14.0" fill="#333">장비 자체 알고리즘 (OK / NG)</text><rect x="420" y="104" width="230" height="72" rx="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/><text x="535" y="128" text-anchor="middle" font-size="17.1" font-weight="700">AI 2차 판독</text><text x="535" y="147" text-anchor="middle" font-size="14.0">RF-DETR-S · YOLO11s 각각 판독</text><text x="535" y="166" text-anchor="middle" font-size="14.6" font-weight="700" fill="#1565c0">약 45 ms/장</text><rect x="280" y="190" width="160" height="42" rx="21" fill="#fff8e1" stroke="#a07800"/><text x="360" y="216" text-anchor="middle" font-size="17.1" font-weight="700">두 결과 비교</text><rect x="180" y="262" width="360" height="38" rx="6" fill="#fff" stroke="#555"/><text x="360" y="286" text-anchor="middle" font-size="15.2">두 AI 모델의 이물 개수·위치 + 장비 판정 일치 여부</text><rect x="30" y="342" width="180" height="60" rx="6" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.8"/><text x="120" y="367" text-anchor="middle" font-size="18.3" font-weight="700" fill="#2e7d32">PASS</text><text x="120" y="388" text-anchor="middle" font-size="14.0">둘 다 이물 없음 → 통과</text><rect x="270" y="342" width="180" height="60" rx="6" fill="#fff3e0" stroke="#e65100" stroke-width="1.8"/><text x="360" y="367" text-anchor="middle" font-size="18.3" font-weight="700" fill="#e65100">RE-INSPECTION</text><text x="360" y="388" text-anchor="middle" font-size="14.0">개수·위치 불일치 등</text><rect x="510" y="342" width="180" height="60" rx="6" fill="#ffebee" stroke="#c62828" stroke-width="1.8"/><text x="600" y="367" text-anchor="middle" font-size="18.3" font-weight="700" fill="#c62828">REJECT</text><text x="600" y="388" text-anchor="middle" font-size="14.0">확실한 이물 → 보류·격리</text><rect x="250" y="424" width="220" height="38" rx="6" fill="#fff" stroke="#e65100" stroke-dasharray="4 3"/><text x="360" y="448" text-anchor="middle" font-size="15.2">품질 담당자 확인 또는 재촬영</text></svg><figcaption>그림 12. X-ray 이물검사 현장 적용 흐름(제안)</figcaption></figure>''')
s('기존 장비와 AI의 병렬 판정 구조는 현장 적용 제안이며, 현재 KAMP 데이터로 장비+AI 결합 성능까지 실증한 것은 아닙니다.')
d('최종 모델은 장비 표시가 그려지기 전의 X선 영상만 보고 판정합니다. 따라서 기존 검사 장비의 판정을 따라 하는 것이 아니라, 같은 영상을 독립적으로 한 번 더 검사하는 2차 판정기로 쓸 수 있습니다.')
d('기존 장비와 모델의 판정이 다르면 재검사 대상으로 분류해, 장비 알고리즘이 놓친 이물과 모델이 놓친 이물을 서로 보완합니다.')
o('PASS / RE-INSPECTION / REJECT 3단계 판정: 두 모델 판단 불일치 시 보류')
d('S1(이물 확신도가 높은 모델의 예측만 사용)은 화면에 이물 위치를 표시하는 데만 쓰고, 합격·불합격 판정은 두 모델이 각각 내린 결과를 비교해 정합니다. S1만으로 판정하면 한 모델만 찾은 이물이 그 모델의 확신도가 낮다는 이유로 버려질 수 있기 때문입니다.')
d('아래 표의 이물 확신도(confidence score)는 모델이 검출한 후보를 이물이라고 판단하는 정도를 0~1로 나타낸 값이며, 값이 클수록 이물일 가능성을 높게 판단한 것입니다. "최고 이물 확신도"는 한 영상에서 나온 후보 중 가장 높은 값입니다.')
a(f"""<table><thead><tr><th>판정</th><th>조건 (검사 시점에 관찰 가능한 정보만 사용)</th><th>조치</th></tr></thead><tbody>
<tr><td><b>REJECT</b></td><td class=l>두 모델 모두 확정선 이상(RF-DETR-S 이물 확신도 ≥ {TH['rfdetr_s']:.3f}, YOLO11s ≥ {TH['yolo11s']:.3f})으로 이물을 찾고, 찾은 이물의 개수와 위치가 일치</td><td class=l>제품 보류·격리. 재촬영에서 안 보여도 자동 취소하지 않음</td></tr>
<tr><td><b>RE-INSPECTION</b></td><td class=l>두 모델의 판단이 다름(예: 모델 A는 이물 2개, 모델 B는 3개 / 한 모델만 이물을 찾음 / 위치가 다름), 또는 한 모델이라도 최고 이물 확신도가 재검사 하한(RF-DETR-S {TL['rfdetr_s']:.2f}, YOLO11s {TL['yolo11s']:.2f}) 이상</td><td class=l>제품 보류 → 품질 담당자가 두 모델의 박스를 모두 띄운 화면으로 확인하거나 새로 재촬영. 정밀하지만 느린 판독 모델에 넘기는 방식은 향후 과제</td></tr>
<tr><td><b>PASS</b></td><td class=l>두 모델 모두 최고 이물 확신도가 재검사 하한 미만</td><td class=l>통과</td></tr></tbody></table>""")
s('"작은 이물이면 재검사"처럼 모델이 놓치면 알 수 없는 정보는 기준으로 쓰지 않았습니다.')
s('영상 품질 이상(촬영 실패, 포화, 기준 시험편 이탈)은 모델 판정 전에 보류·재촬영합니다.')
s('이 방식은 이물 탐지가 한 모델의 성능이나 확신도 척도에 좌우될 가능성을 줄입니다. 두 모델이 서로 다른 판단을 내리는 경우만 사람에게 넘기므로, 모든 제품을 이중·삼중 검사하지 않아도 됩니다.')
o('S1 단독 판정과 불일치 보류 규칙 비교')
d('같은 저장 예측으로 판정 규칙만 바꿔 비교했습니다. 임계값은 기존 검증 결정값을 그대로 썼고 새로 조정하지 않았습니다. 실제 영상에서는 두 모델이 모두 이물을 찾아 차이가 드러나지 않으므로, 이물을 흐리게 만든 저대비 평가셋(k = 이물 대비를 줄인 배율)에서 차이를 확인했습니다.')
def _row(lab, sp):
    u=DP[sp]['U']; up=DP[sp]['U+']; lone=u['union_center_hit']-(u['objects']-u['s1_center_miss'])
    return f"<tr><td class=l>{lab}</td><td>{lone}</td><td>{u['imgs_with_s1_miss_passed']}</td><td>{up['imgs_with_s1_miss_passed']}</td><td>{up['HOLD']}</td></tr>"
a('<div class=tcap>표 4-1. 판정 규칙별 결과 (각 80장)</div><table><thead><tr><th>평가셋</th><th>S1이 버린 이물<br>(다른 모델만 찾음)</th><th>이물이 있는데 통과된 영상<br>S1 단독</th><th>이물이 있는데 통과된 영상<br>불일치 보류 규칙</th><th>보류 영상 수<br>(불일치 보류 규칙)</th></tr></thead><tbody>'
  + _row('저대비 k=0.5 검증','stress_val_k050') + _row('저대비 k=0.5 테스트','stress_test_k050') + _row('저대비 k=0.35 검증','stress_val_k035') + _row('저대비 k=0.35 테스트','stress_test_k035')
  + f"<tr><td class=l>실제 검증 · 테스트</td><td>0 · 0</td><td>0 · 0</td><td>0 · 0</td><td>{DP['val']['U+']['HOLD']} · {DP['test']['U+']['HOLD']}</td></tr>"
  + f"<tr><td class=l>합성 정상 검증 · 테스트</td><td>-</td><td>-</td><td>-</td><td>{DP['val_normal']['U+']['HOLD']} · {DP['test_normal']['U+']['HOLD']}</td></tr></tbody></table>")
d('S1은 흐린 이물 평가에서 다른 모델만 찾은 이물을 세트마다 7~19개 버렸고, 이물이 있는데 통과시킨 영상이 최대 27장이었습니다. 불일치 보류 규칙은 k=0.5에서 통과 영상을 0장으로, k=0.35에서도 4~5장으로 줄였습니다.')
d('실제 검증·테스트 영상에서는 두 모델의 이물 개수와 위치가 모두 일치해 추가 보류가 생기지 않았습니다. 합성 정상 영상의 보류는 12.5~15%로, 재검사 하한에서 생기는 비용입니다.')
s('개수·위치 비교만으로는 두 모델이 모두 확정선 아래인 흐린 이물을 잡을 수 없어, 재검사 하한(낮은 확신도 후보 보류)을 함께 써야 합니다. 저대비 결과는 평가 전용 합성 영상 기준입니다.')
o('판정 결과 (영상 단위)')
a('<table><thead><tr><th rowspan=2>판정 방식</th><th rowspan=2>세트</th><th colspan=3>이물 있는 영상 (80장)</th><th colspan=3>합성 정상 영상 (80장)</th></tr><tr><th>REJECT</th><th>RE-INSP.</th><th>PASS(놓침)</th><th>REJECT</th><th>RE-INSP.</th><th>PASS</th></tr></thead><tbody>' + pol_rows() + '</tbody></table>')
d('세 방식 모두 이물 있는 영상 160장을 모두 REJECT로 판정해 놓친 제품이 없었습니다. 표의 "두 모델" 행은 위의 불일치 보류 규칙과 같은 결과입니다.')
d('두 모델 조합은 정상 영상의 REJECT를 줄이는 대신(검증 4 → 1장) 재검사를 늘립니다(12.5~15%). 잘못된 폐기를 재검사로 돌리는 효과입니다.')
s('정상 영상은 이물을 지운 합성 영상이므로 실제 공장의 재검사율은 실제 정상 제품으로 다시 측정해야 합니다.')
o('이중·삼중 검사를 대상 선별 재검사로')
d('현재 현장은 미검·과검을 막기 위해 모든 제품을 이중·삼중으로 검사합니다. 3단계 판정을 쓰면 REJECT는 바로 격리하고, 재검사는 RE-INSPECTION 판정 제품에만 하면 됩니다.')
d('합성 정상 영상 기준으로 재검사 대상은 12.5~15%였고, 이물 있는 영상은 모두 REJECT였습니다. 실제 재검사 비율은 실제 정상 제품으로 다시 확인해야 합니다.')
o('사후 분석에서 실시간 판정으로')
d('가이드북의 분석은 SD 카드에 저장된 영상을 PC로 옮겨 분석하는 사후 분석이며, 가이드북도 실시간 적용을 위해 영상 수집과 탐지를 통합한 모듈이 필요하다고 밝힙니다.')
d('본 모델은 장비 표시가 그려지기 전의 영상만 있으면 되므로, 장비 영상 출력에 바로 연결해 판정할 수 있습니다. 두 모델 합산 처리 시간은 1장 약 45ms(Apple M5 Pro, MPS)입니다.')
s('실제 라인 속도와 장비 영상 출력 방식(저장 형식·전송 지연)은 현장에서 확인해야 합니다.')
o('유사 현장 적용 시 고려사항')
d('다른 비전검사 장비나 제품은 영상 크기·밝기·이물 형태가 달라, 해당 장비 데이터로 다시 학습하고 같은 방식(세션 단위 분할, 검증셋에서 임계값 결정)으로 평가해야 합니다.')
d('장비가 판정 표시를 영상에 덧그려 저장하는 경우에는 본 팀의 표시 제거·미끼 박스 전처리를 그대로 적용해, 모델이 장비 표시를 따라 하지 않는지 먼저 점검할 수 있습니다.')
d('재검사 하한·확정선 같은 판정 기준은 현장 품질 담당자와 목표 탐지율·허용 재검사율을 정한 뒤 확정합니다.')
o('놓친 경고가 사라지지 않게 하는 운영 규칙')
d('재검사는 공장 품질 담당자가 의심 제품을 따로 다시 확인하는 현장 절차입니다. 모델 평가용 정답 라벨을 다시 만드는 작업과는 다릅니다.')
d('① 제품 식별자, 촬영 시각, 호기와 영상을 하나로 묶어 기록합니다. 영상과 제품의 대응이 불명확하거나 영상 품질이 나쁘면 판정하지 않고 보류·재촬영합니다.')
d('② 두 모델의 결과가 다르면 둘 다 보존하고 재검사로 보냅니다. 한 모델이 아무것도 찾지 못했다고 다른 모델의 경고를 지우지 않습니다.')
d('③ 추론 시간 초과, 영상 누락, 통신 오류, 배출 확인 실패는 정상 판정으로 바꾸지 않고 보류 절차를 적용해 작업자에게 알립니다. 기존 검사 장비의 보호 기능과 최종 조치 권한은 그대로 유지합니다.')
d('④ 재검사 화면에는 영상 전체, 의심 부분 확대, 두 모델의 박스, 장비 판정과 AI 판정의 차이를 함께 보여 주고, 작업자가 실제 이물 여부와 조치를 기록합니다. 쌓인 기록은 호기·시간대·제품별 반복 오류 점검과 모델 개선에 씁니다.')
s('설계 참고: METTLER TOLEDO X-ray Inspection Guidance[6]의 배출 확인·기록 관리 사례. 위 흐름은 본 팀의 제안이며 실제 설비 연동은 검증하지 않았습니다.')
o('단계적 도입과 효과를 확인하는 방법')
a("""<table><thead><tr><th style="width:18%">단계</th><th>실행 범위</th><th>확인할 항목</th></tr></thead><tbody>
<tr><td>1. 병행 기록</td><td class=l>기존 검사 유지, AI 결과만 별도 저장. AI가 장비 판정을 취소하거나 자동 PASS를 내리지 않음</td><td class=l>새 시험편·실제 정상 제품에서 장비와 AI 판정의 일치</td></tr>
<tr><td>2. 재검사 지원</td><td class=l>작업자에게 의심 위치·확대 영상과 우선순위 제공</td><td class=l>재검사 소요 시간, 놓친 이물, 정상 제품 오경보</td></tr>
<tr><td>3. 제한적 자동 연계</td><td class=l>검증된 조건 안에서 배출 장치와 연결</td><td class=l>처리 지연, 제품 추적, 배출 확인, 장애 대응</td></tr></tbody></table>""")
d('도입 효과는 네 가지 지표로 확인합니다: ① 이물 제품 중 경보 없이 통과한 비율, ② 정상 제품 중 경보한 비율, ③ 전체 제품 중 재검사로 보낸 비율, ④ 촬영부터 조치까지의 처리 시간. 라인 속도와 허용 재검사량은 제공되지 않았으므로 현장에서 정해야 하며, 재검사량 감소나 미탐 감소가 이미 실현됐다고 주장하지 않습니다.')
o('현장 도입 전 검증 계획')
d('현재 평가 데이터는 같은 시험편을 반복 촬영한 영상이 많고 실제 정상 제품 영상도 포함되어 있지 않습니다. 따라서 현재 성능을 실제 생산라인 성능으로 그대로 해석할 수 없습니다.')
d('현장 적용 전에는 학습에 사용하지 않은 새로운 시험편과 실제 정상 제품을 별도로 촬영하여, ① 실제 이물을 얼마나 놓치지 않는지, ② 정상 제품을 얼마나 자주 재검사로 보내는지를 다시 확인해야 합니다.')
d('예를 들어 실제 이물 299개를 독립적으로 검사해 한 건도 놓치지 않는다면 95% 신뢰수준에서 탐지율 99% 이상이라는 근거를 확보할 수 있습니다. 같은 시험편의 반복 촬영이나 증강 영상은 독립 표본으로 계산하지 않습니다.')
a('<div class=tcap>표 5. 목표 탐지율을 확인하기 위해 필요한 독립 시험 수</div>')
a("""<table><thead><tr><th>확인하려는 탐지율</th><th>한 건도 놓치지 않을 때</th><th>1건 놓칠 때</th><th>2건 놓칠 때</th></tr></thead><tbody>
<tr><td>90%</td><td>29</td><td>46</td><td>61</td></tr><tr><td>95%</td><td>59</td><td>93</td><td>124</td></tr><tr><td>99%</td><td>299</td><td>473</td><td>628</td></tr></tbody></table>""")
s('95% 신뢰수준 기준(정확 이항 계산). 정상 제품도 같은 방식으로, 재검사 비율이 5% 이하·1% 이하임을 보이려면 각각 59개·299개의 정상 제품을 한 건도 재검사로 보내지 않아야 합니다.')
# ---------------- CH5 ----------------
a('<h2 class=chap>□ 제5장. 창의성 및 차별성</h2>')
o('핵심 차별점: 표시 박스가 없는 영상에서도 이물 판별')
d(f"기존 데이터와 기존 모델은 장비 표시 박스가 있는 상황을 전제로 합니다. 기존 모델은 표시를 지우면 이물 적중률이 {pct(BL['test_raw']['center_recall'],0)}에서 0%로 떨어졌습니다.")
d(f"본 팀의 모델은 표시 박스가 없는 영상에서 테스트 재현율 {pct(rf['test']['recall'])}, 이물 중심 적중률 100%를 기록했습니다. 모델이 이물 자체를 보고 판단한다는 뜻입니다.")
d('이를 위해 표시 박스를 지우는 것에 그치지 않고, 이물이 없는 자리에도 같은 지운 흔적(미끼 박스)과 잡음을 넣었습니다. 지운 흔적만으로는 이물 여부를 알 수 없게 만들어, 모델이 흔적을 새 단서로 외우는 것을 막았습니다.')
d('검증: 미끼 흔적에 반응한 오탐이 모든 모델·모든 실험에서 0건이었습니다. 모델이 표시 박스도, 지운 흔적도 단서로 쓰지 않음을 확인했습니다.')
s('한계: 실제로 표시 박스가 그려지기 전의 원본 영상은 제공되지 않아, 표시를 지운 영상으로 이를 대신했습니다. 현장 적용 전에는 표시 없는 원본 영상으로 다시 확인해야 합니다.')
o('KAMP 가이드북 접근과의 차이')
a('''<table><thead><tr><th style="width:16%">항목</th><th style="width:38%">KAMP 가이드북</th><th>본 팀</th></tr></thead><tbody>
<tr><td>입력 영상</td><td class=l>장비 표시 박스가 그려진 영상</td><td class=l>표시를 지우고 미끼 흔적·잡음을 넣은 영상</td></tr>
<tr><td>탐지 모델</td><td class=l>YOLOv3-SPP</td><td class=l>RF-DETR-S + YOLO11s 영상 단위 혼용</td></tr>
<tr><td>데이터 분할</td><td class=l>영상 무작위 8:2</td><td class=l>촬영 세션 단위, 임계값은 검증셋에서만 결정</td></tr>
<tr><td>평가 지표</td><td class=l>정밀도·재현율·F1·mAP50</td><td class=l>위 지표 + 이물 중심 적중률, 미끼 오탐, 합성 정상 오경보, 저대비 평가, 세션 단위 신뢰구간</td></tr>
<tr><td>결과 활용</td><td class=l>결함 위치 표시·잘라낸 영상</td><td class=l>PASS / RE-INSPECTION / REJECT 3단계 판정</td></tr>
<tr><td>라벨 수 실험</td><td class=l>15~400장, 200장이 효율적</td><td class=l>가공 데이터 조합 A~F 비교 후 상위 후보 모델로 재확인(D 유지)</td></tr></tbody></table>''')
o('학습·평가 데이터 중복이 없는 평가 설계')
d('원본 해시·촬영 세션·촬영 간격·이물 배치까지 점검해 학습과 평가 영상 사이의 직접 유출이 없음을 확인했고, 같은 시험편 반복이라는 구조적 한계를 수치(86%)로 밝혔습니다.')
o('미탐을 원인별로 분해한 평가')
d('IoU 기준 미탐을 박스 불일치·확신도 미달·후보 없음으로 나눠, 남은 미탐이 위치를 놓친 것이 아니라 박스 크기 차이임을 보였습니다(중심 적중률 100%). 이 결과로 임계값을 낮추는 대신 재검사 기준을 설계했습니다.')
o('저대비 스트레스 평가로 모델 조합 근거 확보')
d('실제 데이터의 평균 성능만으로는 보이지 않던 차이(저대비에서 YOLO11s 우세)를 평가 전용 합성으로 드러내, 구조가 다른 두 모델의 판단 불일치를 재검사 신호로 쓰는 근거로 삼았습니다.')
o('약점이 다른 두 모델의 혼용과 불일치 보류')
d('평균 성능이 비슷한 두 모델의 조건별 강점(원래 대비 vs 저대비)을 찾아, 이물 확신도가 높은 모델을 쓰는 간단한 규칙만으로 원래 성능을 유지하면서 저대비 대응력을 높였습니다(테스트 k=0.5 적중률 84% → 90%).')
d('나아가 두 모델의 이물 개수·위치가 다르면 제품을 보류해 사람이 확인하도록 해, 한 모델만 찾은 흐린 이물이 버려지는 문제를 막았습니다(저대비 k=0.5에서 이물이 있는데 통과된 영상 5장 → 0장). 확신이 낮은 예측을 자동 처리에서 빼는 보류(reject option) 개념[5]을, 별도 모델 학습 없이 프로그램 규칙으로 구현한 것입니다.')
o('가공 데이터 효과의 사전 기준 검증')
d('합성·제거 영상을 "많을수록 좋다"고 가정하지 않고, 실험 전 판정 기준과 반복 학습으로 효과를 확인했습니다. 그 결과 가공 데이터가 저대비 대응력을 낮추는 대신 오경보를 줄인다는 맞바꿈 관계를 찾았고, 최종 후보 모델로 다시 학습해 이를 확인했습니다(표 3-2).')
o('라벨 확보와 라벨 품질 감사')
d(f"사람 라벨이 500장뿐인 한계를 장비 색 표시를 이용한 자동 라벨링(약라벨 1,880장)으로 보완하고, 같은 규칙을 사람 라벨 500장에 적용해 {pct(LA['shrink8']['iou50_rate'])}가 일치함을 확인했습니다. 표시 모양 전수 분류로 부적합 영상(T 아이콘 6장, 붙은 박스 25장 등)을 제외하고, 검증 라벨 누락 1건을 찾아 오탐 해석에 반영했습니다.")
# ---------------- CH6 ----------------
a('<h2 class=chap>□ 제6장. 코드 구성 및 재현성</h2>')
o('실행 흐름')
d('① 환경 설치: Python 가상환경과 필요한 라이브러리를 설치합니다.')
d('② 데이터 전처리: 장비 표시 박스 제거·미끼 박스·잡음 재주입과 세션 단위 분할, 합성 정상·저대비 평가셋을 만듭니다.')
d('③ 모델 학습: YOLOv8n·YOLO11s·RF-DETR-S·Qwen2.5-VL을 같은 데이터로 학습하고, 학습 데이터 구성(A~F, A·D 재확인)을 비교합니다.')
d('④ 성능 평가: 공통 채점 코드로 검증셋에서 임계값을 정하고 테스트셋에 한 번만 적용합니다. 두 모델 혼용도 같은 방식으로 정합니다.')
d('⑤ 보고서용 결과 생성: 분석 수치·그림·비교표를 만들고 보고서를 생성합니다.')
o('핵심 실행 명령')
a("""<pre>bash run_all.sh                    # ①~④ 설치 · 학습 · 예측 · 채점
bash phase2.sh                     # 학습 데이터 구성 비교 (A~F)
python scripts/a_vs_d_report.py    # 상위 후보 모델 A·D 비교표</pre>""")
s('세부 스크립트와 폴더 구성은 부록 B에 정리했습니다.')
o('실행 환경과 재현 조건')
d('macOS 26.6 (Apple M5 Pro, MPS), Python 3.12.14, torch 2.14.1, ultralytics 8.4.172, rfdetr 1.11.1, transformers 5.18.0, peft 0.21.2. 시드는 0입니다.')
d('예측은 confidence 0.001 이상을 모두 저장한 뒤 채점 단계에서 임계값을 적용하므로, 임계값 분석에 재학습이 필요 없습니다.')
s('MPS 연산의 비결정성 때문에 재학습 시 수치가 소폭 달라질 수 있습니다.')
o('한계와 향후 과제')
d('표시 박스가 그려지기 전의 실제 원본 영상이 없어, 표시를 지운 영상으로 대신 평가했습니다.')
d('같은 시험편 반복 촬영 데이터라 처음 보는 제품에 대한 성능은 검증하지 못했습니다.')
d('실제 정상 제품이 없어 실제 오경보율을 측정하지 못했고, 가장자리 이물은 평가셋에 없습니다.')
d('평가 표본이 각 80장(16·22세션)으로 작아 상위 모델 간 차이는 통계적으로 구분되지 않습니다.')
d('저대비 이물 탐지와 정상 제품 오경보의 균형점(학습 데이터 구성, 박스 단위 결합 방식)은 실제 흐린 시험편과 실제 정상 제품으로 다시 정해야 합니다.')
a('<h2 class=chap>□ 참고문헌</h2>')
for r_ in ["[1] 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), X-ray 검사장비 AI 데이터셋, KAIST(㈜임픽스, 한양대학교 산학협력단, ㈜아큐라소프트), 2020.12.14., www.kamp-ai.kr (분석실습 가이드북 포함)",
           "[2] Kim, K. et al. (2021). Real-Time Anomaly Detection in Packaged Food X-Ray Images Using Supervised Learning. Computers, Materials &amp; Continua, 67(2). https://doi.org/10.32604/cmc.2021.014642",
           "[3] Andriiashen, V. et al. (2024). X-Ray Image Generation as a Method of Performance Prediction for Real-Time Inspection: a Case Study. Journal of Nondestructive Evaluation. https://arxiv.org/abs/2401.16847",
           "[4] Wang, J. et al. (2021). A Normalized Gaussian Wasserstein Distance for Tiny Object Detection. https://arxiv.org/abs/2110.13389",
           "[5] Geifman, Y., El-Yaniv, R. (2019). SelectiveNet: A Deep Neural Network with an Integrated Reject Option. ICML, PMLR 97.",
           "[6] METTLER TOLEDO. X-ray Inspection Guidance. https://www.mt.com",
           "[7] Roboflow. RF-DETR 공식 구현. https://github.com/roboflow/rf-detr · Ultralytics YOLO11 공식 문서. https://docs.ultralytics.com"]:
    a(f'<div class=d style="font-size:11pt">{r_}</div>')
# ---------------- APPENDIX ----------------
a('<h2 class=chap>□ 부록 A. 모델별 상세 분석 자료</h2>')
for k in ['rfdetr_s','yolo11s','yolov8n','yolov8n_4403장']:
    a(f'<div class=o>◦ {N[k]}</div>')
    a(f"<div class=s>* 검증 F1 최대 임계값 {S1[k]['thr']:.4f}, 학습 {S1[k]['meta']['train_seconds']/3600:.2f} h, 1장 처리 {ms(k)} ms (MPS, 배치 1)</div>")
    a('<div class=tcap>성능</div>' + main_table([k]))
    a('<div class=tcap>임계값별 결과</div>' + sweep_table(k) if 'sweep' in S1[k] else '')
    a('<div class=tcap>호기별 미탐</div>' + hogi_table(k))
    a('<div class=tcap>미탐 유형</div>' + fnt_table(k))
for k in ['qwen25vl3b_lora','qwen25vl3b_zeroshot']:
    a(f'<div class=o>◦ {N[k]}</div>')
    a('<div class=tcap>성능 (이물 확신도를 내지 않아 임계값 분석 없음)</div>' + main_table([k]))
    a('<div class=tcap>호기별 미탐</div>' + hogi_table(k))
    a('<div class=tcap>미탐 유형</div>' + fnt_table(k))
a(f'<figure>{F("f9_vlm.png")}<figcaption>그림 A1. 비전-언어 모델 예측 비교 (초록 = 정답, 빨강 = 예측)</figcaption></figure>')
a('<div class=o>◦ 대비 축소 평가 상세 (이물 중심 적중률 / 재현율 / 영상 검출률)</div>')
h = '<table><thead><tr><th>모델</th><th>세트</th>' + ''.join(f'<th>k={x}</th>' for x in ('1.0','0.75','0.5','0.35','0.25')) + '</tr></thead><tbody>'
for k in ('rfdetr_s','yolo11s','yolov8n'):
    for sp, lab in (('val','검증'),('test','테스트')):
        rr = [r for r in stress[k] if r['split']==sp]
        h += f"<tr>{f'<td rowspan=2 class=l>{N[k]}</td>' if sp=='val' else ''}<td>{lab}</td>" + ''.join(f"<td>{pct(r['center_recall'],0)} / {pct(r['recall'],0)} / {pct(r['image_detect'],0)}</td>" for r in rr) + '</tr>'
a(h + '</tbody></table>')
# satisfaction page
a('<h2 class=chap>□ 부록 B. 코드 폴더 구성</h2>')
a("""<pre>kamp_ai/
├─ X-ray_데이터셋_통합/          학습 6개 가공 폴더 · 검증 · 테스트 · metadata · manifest.csv
├─ X-ray_실험/
│  ├─ run_all.sh / 학습시작.command   1단계: 설치 → 스모크 → 모델 학습 → 채점 (중단 시 이어서 실행)
│  ├─ phase2.sh / 2단계시작.command  2단계: 학습 데이터 조합 비교 A~F
│  ├─ data_전체+약라벨.yaml, train_list.txt, excluded_train_images.csv
│  ├─ eval_sets/                    합성 정상 · 대비 축소 평가셋 (평가 전용)
│  ├─ scripts/
│  │   ├─ common.py                 경로·제외 목록
│  │   ├─ build_extra.py, xproc.py  약라벨 영상 생성 (색 박스 복원·미끼·잡음)
│  │   ├─ make_normal_eval.py       합성 정상 평가셋
│  │   ├─ build_phase2_data.py      조합 실험 목록·대비 축소 평가셋
│  │   ├─ yolo_run.py, rfdetr_run.py, vlm_run.py   학습·예측
│  │   ├─ evaluate.py               공통 채점 (검증 임계값을 테스트에 고정 적용)
│  │   ├─ ensemble_val.py, ensemble_test.py   두 모델 혼용(검증 결정 → 테스트 1회)
│  │   ├─ yolo11s_A_run.py, rfdetr_A_run.py   최종 후보 모델 A 구성 재학습
│  │   ├─ compare_A_val.py, compare_A2_val.py  A 구성 채택 판정(검증 전용)
│  │   ├─ a_vs_d_report.py           A·D 구성 비교표(판정 후 기록용)
│  │   └─ report_analysis.py, report_figs.py, leak_check.py         분석·그림
│  ├─ preds/                        모델별 예측 CSV · 채점 JSON · 메타
│  └─ report/                       보고서 생성 코드·그림·분석 JSON</pre>""")
a("""<pre># 전체 재현 순서
bash X-ray_실험/run_all.sh
bash X-ray_실험/phase2.sh
bash X-ray_실험/yolo11s_A.sh && bash X-ray_실험/rfdetr_A.sh
python scripts/a_vs_d_report.py
python scripts/report_analysis.py && python scripts/report_figs.py</pre>""")
a('<h2 class=chap>□ 경진대회 만족도 조사 완료 페이지 캡쳐 화면(필수)</h2><div class=d>- 만족도 조사 url : https://naver.me/FetWt7SN</div><div class=box style="height:140mm;display:flex;align-items:center;justify-content:center;color:#888">만족도 조사 완료 화면 캡처를 여기에 첨부</div>')
html = f'<!doctype html><html lang=ko><head><meta charset=utf-8><title>제6회 K-인공지능 제조데이터 분석 경진대회 보고서</title><style>{css}</style></head><body>{"".join(P)}</body></html>'
open(f'{R}/report/report.html', 'w').write(html); print(len(html))
