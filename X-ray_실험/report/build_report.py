import json, os, base64, datetime
R = '/home/claude/kamp/X-ray_실험'
S1 = json.load(open(f'{R}/analysis_stage1.json')); RA = json.load(open(f'{R}/report/report_analysis.json')); P2 = json.load(open(f'{R}/report/p2_analysis.json')); ENS = json.load(open(f'{R}/report/ensemble_val_test.json')); BL = json.load(open(f'{R}/report/baseline_yolov3_valtest.json'))
def img(p, w='100%'):
    b = base64.b64encode(open(p, 'rb').read()).decode(); ext = 'png' if p.endswith('png') else 'jpeg'
    return f'<img src="data:image/{ext};base64,{b}" style="width:{w}">'
F = lambda n: img(f'{R}/report/figs/{n}')
def pct(x, d=1): return '-' if x is None else f'{x*100:.{d}f}%'
N = {'rfdetr_s': 'RF-DETR-S', 'yolo11s': 'YOLO11s', 'yolov8n': 'YOLOv8n', 'qwen25vl3b_lora': 'Qwen2.5-VL-3B LoRA', 'qwen25vl3b_zeroshot': 'Qwen2.5-VL-3B 제로샷', 'yolov8n_4403장': 'YOLOv8n (라벨 수정 전 4,403장)'}
def ms(k):
    m = S1[k]['meta']; v = m.get('ms_per_image_mps_batch1') or m.get('ms_per_image_mps'); return f'{v:,.0f}' if v >= 100 else f'{v:.1f}'
def main_table(keys):
    h = '<table><thead><tr><th>모델</th><th>세트</th><th>정밀도</th><th>재현율</th><th>mAP50</th><th>mAP50-95</th><th>TP</th><th>FN</th><th>FP</th><th>중심 적중률</th><th>정상 오경보</th><th>미끼 오탐</th></tr></thead><tbody>'
    for k in keys:
        for sp, lab in (('val', '검증'), ('test', '테스트')):
            v = S1[k]['eval'][sp]
            h += f"<tr>{f'<td rowspan=2 class=l><b>{N[k]}</b></td>' if sp=='val' else ''}<td>{lab}</td><td>{pct(v['precision'])}</td><td><b>{pct(v['recall'])}</b></td><td>{v['AP50']:.3f}</td><td>{v['AP50_95']:.3f}</td><td>{v['TP']}</td><td><b>{v['FN']}</b></td><td>{v['FP']}</td><td>{pct(S1[k][sp+'_center_recall'])}</td><td>{pct(v['normal_false_alarm_rate'])}</td><td>{v['decoy_FP']}</td></tr>"
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
    return f"<table><thead><tr><th>세트</th><th>박스 불일치 (IoU 0.1~0.5)</th><th>점수 미달</th><th>후보 없음</th></tr></thead><tbody><tr><td>검증</td><td>{a['box_mismatch']}</td><td>{a['below_threshold']}</td><td>{a['no_candidate']}</td></tr><tr><td>테스트</td><td>{b['box_mismatch']}</td><td>{b['below_threshold']}</td><td>{b['no_candidate']}</td></tr></tbody></table>"
CI = RA['session_ci']; TH = RA['t_high']; TL = RA['t_low']; POL = RA['policy']
fail = RA['fail']
def fail_rows(f):
    return ''.join(f"<tr><td class=l>{f}</td><td class=l>{r['level']}</td><td>{r['n']}</td>" + ''.join(f"<td>{r[k]['FN']} ({pct(r[k]['recall'])})</td>" for k in ('rfdetr_s','yolo11s','yolov8n')) + '</tr>' for r in fail[f])
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
pre { font-size: 8.5pt; line-height:1.35; background:#f5f5f5; padding: 3mm; border:0.6px solid #bbb; white-space: pre-wrap; font-family: 'Noto Sans Mono CJK KR', monospace }
.pb { break-before: page }
"""
P = []; a = P.append
o = lambda t: a(f'<div class=o>◦ {t}</div>')
d = lambda t: a(f'<div class=d>- {t}</div>')
s = lambda t: a(f'<div class=s>* {t}</div>')
rf, y11, y8 = S1['rfdetr_s']['eval'], S1['yolo11s']['eval'], S1['yolov8n']['eval']
# ---------------- COVER ----------------
summary = f"""포장공정 마지막 품질검사인 X선 이물검사의 미검·과검을 AI 영상 판정으로 보완하는 과제입니다. KAMP X선 이물검출기 데이터는 장비가 이물 위치에 색 표시 박스를 그린 영상이 기본이며, 이 데이터로 학습한 기존 모델(KAMP 제공 YOLOv3)은 표시 박스가 있을 때만 이물을 찾습니다. 같은 테스트 영상에서 표시 박스를 지우자 기존 모델의 이물 적중률은 {BL['test_raw']['center_recall']*100:.0f}%에서 0%가 됐습니다. 본 팀은 표시 박스를 지우고, 이물이 없는 자리에도 같은 지운 흔적(미끼 박스)을 남겨 모델이 표시나 흔적이 아니라 이물 자체를 학습하게 했습니다. 그 결과 표시 박스가 없는 영상에서 RF-DETR-S와 YOLO11s를 혼용한 최종 모델이 테스트 재현율 {pct(rf['test']['recall'])}(미탐 {rf['test']['FN']}개), 이물 중심 적중률 100%를 기록했고, 미끼 흔적에 대한 오탐은 0건이었습니다. 혼용 방식은 대비를 절반으로 낮춘 이물의 적중률도 84%에서 90%로 높였습니다. 남은 미탐은 위치는 맞혔으나 박스 크기가 달랐던 경우이며, 판단이 애매한 제품은 재검사로 보내는 PASS / RE-INSPECTION / REJECT 3단계 판정을 제안합니다."""
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
d('현장 문제는 장비 자체의 기술적 한계로 생기는 미검(불량을 양품으로 판정)과 과검(양품을 불량으로 판정)입니다. 현재는 이를 막기 위해 이중·삼중 검사를 하지만, 생산 흐름과 작업 동선이 바뀌고 중복 작업 비용이 들며, 같은 장비로 반복 검사해도 장비의 근본 한계는 남습니다.')
d('KAMP 분석실습 가이드북은 이 문제를 장비가 수집한 X선 영상의 AI 객체 탐지로 보완하는 방향을 제시합니다. 본 팀도 같은 목표를 따르되, 아래 문제 정의에서 보듯 장비 판정에 기대지 않는 독립 판정을 목표로 삼았습니다.')
s('데이터 출처: 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), X-ray 검사장비 AI 데이터셋, KAIST(㈜임픽스, 한양대학교 산학협력단, ㈜아큐라소프트), 2020.12.14., www.kamp-ai.kr')
o('데이터가 나타내는 공정 상태와 분석 목적')
d('KAMP 제공 X선 이물검출기 영상(1·2·3호기)입니다. 장비의 SD 카드에 자동 저장된 영상을 PC로 옮겨 수집했으며, 장비가 불량으로 판정한 경우에만 저장되므로 수집 주기는 불규칙합니다.')
s('원본은 BMP 2,809장(1호기 1,031·2호기 920·3호기 858장, 415MB)이며, 이 중 500장에 사람이 이물 위치를 라벨링한 YOLO 좌표(txt)가 함께 제공됩니다. 촬영일은 파일명 기준 2020.06.22~09.22입니다(가이드북 표기 06.23~09.22).')
d('각 영상은 슬롯 3개가 있는 시험편을 투과 촬영한 흑백 영상이며, 장비가 이물로 판정한 위치에 색 박스를 그려 저장한 NG 영상입니다.')
d('이물은 슬롯 끝에 놓인 지름 약 10px(5~21px)의 어두운 점입니다. 분석 목적은 영상만 보고 이물의 위치를 찾아 해당 제품을 보류할지 판단하는 것입니다.')
s('촬영 시각을 보면 같은 시험편을 약 4시간 간격으로 반복 촬영한 데이터입니다(아래 데이터 진단 결과 참고). 따라서 본 데이터의 성능은 "알려진 시험편을 다른 시각에 다시 촬영했을 때"의 탐지 성능입니다.')
o('문제 정의: 표시 박스에 의존하는 기존 모델')
d('KAMP 원본 영상에는 장비가 이물로 판정한 위치에 빨강·초록 등의 색 박스가 그려져 있습니다. 이 영상으로 학습한 기존 모델(KAMP 제공 YOLOv3-SPP, 806 epoch 체크포인트)은 이물이 아니라 표시 박스를 단서로 학습했습니다.')
d(f"같은 테스트 80장에서 기존 모델은 표시 박스가 있는 원본에서 이물 {BL['test_raw']['gt']}개 중 {BL['test_raw']['center_hits']}개({pct(BL['test_raw']['center_recall'],0)})의 위치를 맞혔지만, 표시 박스만 지운 영상에서는 하나도 찾지 못했습니다(0%). 검증 80장도 {pct(BL['val_raw']['center_recall'],0)} → 0%로 같았습니다.")
d('KAMP 가이드북도 붉은 표시 상자의 중심에 있는 어두운 부분을 결함으로 설명하고, 표시가 그려진 영상에 라벨링해 YOLOv3를 학습합니다. 가이드북은 라벨 200장 학습 시 정밀도 약 0.92, 재현율 약 0.95, F1 약 0.93, 1장 0.02~0.03초를 보고합니다.')
s(f"이 수치는 표시 박스가 있는 영상에서 측정한 값입니다. 같은 KAMP 제공 체크포인트를 본 팀의 테스트 원본(표시 있음)에 적용하면 이물 중심 적중률은 {pct(BL['test_raw']['center_recall'],0)}였습니다. 평가 영상과 라벨 기준이 달라 가이드북 수치와 직접 비교하지는 않습니다.")
d('실제 생산 라인에서 판정해야 하는 영상은 표시 박스가 그려지기 전의 영상입니다. 표시 박스가 있어야만 동작하는 모델은 기존 장비의 판정을 따라 할 뿐, 새 영상에서 이물을 스스로 찾지 못합니다.')
d('따라서 본 과제의 목표를 "표시 박스가 없는 X선 영상에서 이물을 직접 판별하는 모델"로 정했습니다.')
a(f'<figure>{F("f11_shortcut.png")}<figcaption>그림 1. 기존 모델과 우리 모델의 비교. ①의 파란 박스는 기존 모델 예측, 빨간 박스는 장비 표시. ③의 빨간 박스는 우리 모델 예측, 초록 박스는 정답</figcaption></figure>')
o('주요 변수 정의')
a("""<table><thead><tr><th>변수</th><th>형식</th><th>의미 / 값</th></tr></thead><tbody>
<tr><td>영상</td><td>PNG, 8bit 흑백</td><td class=l>1호기 352×332 또는 316×332, 2호기 316×332, 3호기 576×444 px</td></tr>
<tr><td>이물 라벨</td><td>YOLO txt</td><td class=l>한 줄 = 이물 1개 (class 0, 중심 x·y, 너비·높이, 0~1 정규화). 빈 파일 = 이물 없음</td></tr>
<tr><td>호기</td><td>1·2·3</td><td class=l>검출기 장비 번호. 장비마다 영상 크기·밝기가 다름</td></tr>
<tr><td>촬영 시각·세션</td><td>파일명</td><td class=l>60초 이내 연속 촬영을 한 세션으로 묶음. 같은 세션은 한 분할에만 배치</td></tr>
<tr><td>라벨 출처</td><td>사람 / 약라벨</td><td class=l>사람 라벨 500장(학습 340·검증 80·테스트 80), 장비 색 박스 기반 약라벨 1,880장</td></tr>
<tr><td>가공 방법</td><td>폴더</td><td class=l>원본·이물 합성·이물 제거 등 6종 (아래 표)</td></tr></tbody></table>""")
o('전처리 내역')
d('장비 색 박스 제거: 각 화소의 max(RGB)−min(RGB) ≥ 80 이고 max ≥ 180 인 화소를 색 표시로 보고, 1px 확장한 영역을 주변 밝기로 복원했습니다(Telea 인페인팅).')
d('미끼 박스: 복원 흔적이 이물 위치에만 남으면 모델이 흔적을 외웁니다. 이물이 없는 제품 영역에도 같은 모양의 박스를 그렸다 지워 흔적을 위치와 무관하게 만들었습니다.')
d('잡음 재주입: 복원 부위가 매끈해지지 않도록 주변 잡음 세기에 맞춘 가우시안 잡음을 더했습니다.')
d('중복 제거와 분할: 내용이 같은 중복 영상을 제거하고, 세션 단위로 호기별 15%씩 검증·테스트에 배정했습니다.')
s('가이드북 실습은 영상을 무작위 8:2로 나눕니다. 본 데이터는 같은 시험편을 연속 촬영한 영상이 많아, 무작위로 나누면 거의 같은 영상이 학습과 평가에 함께 들어갑니다. 이를 막기 위해 촬영 세션 단위로 나눴습니다.')
a(f'<figure>{F("f1_pipeline.png")}<figcaption>그림 2. 전처리 과정. ③의 주황 화소가 수정된 영역이며, 실제 이물 3곳과 미끼 3곳이 같은 방식으로 수정됩니다.</figcaption></figure>')
o('학습 데이터 구성과 각 가공의 목적')
a("""<table><thead><tr><th>폴더</th><th>장수</th><th>무엇을 개선하려는 가공인가</th></tr></thead><tbody>
<tr><td class=l>1 원본(사람 라벨)</td><td>340</td><td class=l>기준 데이터</td></tr>
<tr><td class=l>2 이물 합성: 제품 안 임의 위치</td><td>680</td><td class=l>이물이 늘 슬롯 끝에만 있어 생기는 위치 암기 방지</td></tr>
<tr><td class=l>3 이물 합성: 경계 강화+모양 변형</td><td>680</td><td class=l>회전·비율 변형과 제품 가장자리 위치로 형상·위치 다양화</td></tr>
<tr><td class=l>4 이물 전부 제거</td><td>340</td><td class=l>실제 정상 영상이 없어 "이물 없음" 사례를 합성 (오경보 감소 목적)</td></tr>
<tr><td class=l>5 이물 일부 제거</td><td>452</td><td class=l>같은 영상 안에 이물 있는/없는 슬롯 끝을 함께 보여 위치가 아닌 이물 자체를 학습</td></tr>
<tr><td class=l>6 약라벨 실제 영상</td><td>1,880</td><td class=l>사람 라벨이 없는 학습 세션 영상에 장비 색 박스를 라벨로 사용해 실제 영상 수 확대</td></tr></tbody></table>""")
s('약라벨은 같은 전처리를 거쳤고, 사람 라벨과 비교해 색 박스가 가로·세로 중앙값 8px 크고 중심 오차가 1px이어서 박스를 8px 줄였습니다.')
s('색 박스가 없는 노란 선 영상 105장, 장비 화면의 T 아이콘이 라벨로 들어간 6장, 박스 두 개가 붙어 하나로 잡힌 25장은 제외했습니다.')
a(f'<figure>{F("f2_methods.png")}<figcaption>그림 3. 원본 1장과 그로부터 만든 가공 영상 (빨간 박스 = 학습 라벨)</figcaption></figure>')
a(f'<figure>{img(R+"/report/figs/f3_composition.png","82%")}<figcaption>그림 4. 학습 데이터 4,372장의 구성. 검증 80장(이물 183개)·테스트 80장(이물 172개)은 모두 사람 라벨 원본입니다.</figcaption></figure>')
o('데이터 진단 결과')
d('파일 무결성: 2,652개 영상·라벨을 전수 검사해 누락·형식 오류·분할 간 동일 영상은 0건이었습니다(3px 미만 박스 1건).')
d('검증·테스트 유출: 학습에 쓴 모든 영상의 원본이 학습 분할에 속하고, 원본 해시·세션이 겹치는 경우는 0건입니다. 가장 가까운 학습 영상과의 촬영 간격은 최소 63분입니다.')
d('같은 시험편 반복: 검증·테스트 160장 중 137장(86%)은 이물 배치가 거의 같은(제품 크기의 5% 이내) 학습 영상이 있습니다. 처음 보는 제품에 대한 성능은 이 데이터로 측정할 수 없습니다.')
d('라벨 누락 1건: 검증 영상 h2_002_20200623_123055(5)는 장비 표시가 3개인데 사람 라벨이 2개입니다. 탐지 모델 3종 모두 누락된 이물을 찾았고 오탐으로 집계됐습니다. 평가 기준의 일관성을 위해 라벨은 수정하지 않았습니다.')
d('박스 크기 불일치: 사람 라벨 크기가 7~15px로 넓게 퍼져 있어, 10px 물체에서 2px 차이만으로도 IoU가 0.5 아래로 떨어집니다(3장).')
d('분포 편중: 이물은 모두 제품 가장자리에서 25px 이상 안쪽(슬롯 끝)에 있고, 실제 정상 제품 영상은 없습니다(가이드북: 양품 영상은 따로 수집되지 않음). 가장자리 이물과 실제 오경보율은 이 데이터로 검증할 수 없습니다.')
a(f'<figure>{img(R+"/report/figs_extra/label_missing.jpg","78%")}<figcaption>그림 5. 검증셋 라벨 누락 사례. 왼쪽 장비 표시 3개, 오른쪽 사람 라벨 2개(가운데 슬롯 누락)</figcaption></figure>')
# ---------------- CH2 ----------------
a('<h2 class=chap>□ 제2장. AI 예측모델 개발 및 성능평가</h2>')
o('평가 설계')
d('모든 모델은 같은 학습 데이터 4,372장으로 학습하고, 같은 채점 코드로 평가했습니다. 정답과 예측은 IoU ≥ 0.5로 1:1 대응시켰습니다.')
d('confidence 임계값은 검증셋 F1이 최대인 값으로 정하고, 테스트셋에는 그 값을 바꾸지 않고 한 번만 적용했습니다.')
d('이물 미탐이 가장 중요하므로 재현율과 FN을 주지표로, mAP·정밀도·FP를 보조지표로 삼았습니다.')
s('현장 용어와의 대응: FN(놓친 이물)은 미검, 합성 정상 영상에서의 오경보는 과검에 해당합니다.')
s('보조 평가: 이물 중심 적중률(예측 박스 중심이 정답 박스 안), 미끼 박스 오탐, 합성 정상 영상(검증·테스트 영상의 이물을 지운 160장) 오경보율, 세션 단위 부트스트랩 95% 신뢰구간')
o('비교 모델과 학습 조건')
a(f"""<table><thead><tr><th>모델</th><th>구조</th><th>학습 조건</th><th>학습 시간</th><th>1장 처리</th></tr></thead><tbody>
<tr><td class=l>YOLOv8n (기준선)</td><td class=l>CNN 1단계 탐지, 3.0M</td><td class=l>640px, 최대 80ep, patience 20 → 31ep 종료</td><td>{S1['yolov8n']['meta']['train_seconds']/3600:.1f} h</td><td>{ms('yolov8n')} ms</td></tr>
<tr><td class=l>YOLO11s</td><td class=l>CNN 1단계 탐지, 9.4M</td><td class=l>640px, 최대 70ep, patience 20 → 55ep 종료</td><td>{S1['yolo11s']['meta']['train_seconds']/3600:.1f} h</td><td>{ms('yolo11s')} ms</td></tr>
<tr><td class=l>RF-DETR-S</td><td class=l>트랜스포머(DINOv2 기반), 31.8M</td><td class=l>512px, 15ep, 배치 4×누적 4</td><td>{S1['rfdetr_s']['meta']['train_seconds']/3600:.1f} h</td><td>{ms('rfdetr_s')} ms</td></tr>
<tr><td class=l>Qwen2.5-VL-3B 제로샷</td><td class=l>비전-언어 모델</td><td class=l>학습 없음, 박스 JSON 출력 지시</td><td>-</td><td>{ms('qwen25vl3b_zeroshot')} ms</td></tr>
<tr><td class=l>Qwen2.5-VL-3B LoRA</td><td class=l>비전-언어 모델 + LoRA(r=16)</td><td class=l>1,300장 × 2ep, 영상 1.5배 확대</td><td>{S1['qwen25vl3b_lora']['meta']['train_seconds']/3600:.1f} h</td><td>{ms('qwen25vl3b_lora')} ms</td></tr></tbody></table>""")
s('장치: Apple M5 Pro (MPS), Python 3.12.14, PyTorch 2.14.1. 사전학습 가중치는 공개 COCO 가중치만 사용했습니다.')
o('동일 조건 성능 비교')
a('<div class=tcap>표 1. 모델별 성능 (검증 80장·테스트 80장, 임계값은 검증에서 결정)</div>' + main_table(['rfdetr_s','yolo11s','yolov8n','qwen25vl3b_lora','qwen25vl3b_zeroshot']))
a(f'<figure>{F("f4_models.png")}<figcaption>그림 6. 모델별 지표와 미탐 수</figcaption></figure>')
d(f"참고로 KAMP 제공 기존 모델(YOLOv3)은 표시 박스가 없는 같은 테스트 영상에서 이물을 하나도 찾지 못했습니다(중심 적중률 0%, 1장 참고). 아래 모델은 모두 표시 박스가 없는 영상으로 학습·평가했습니다.")
d(f"RF-DETR-S는 검증·테스트 모두 미탐이 가장 적었고(FN {rf['val']['FN']}·{rf['test']['FN']}) mAP50·mAP50-95도 가장 높았습니다.")
d(f"테스트 재현율의 세션 단위 95% 신뢰구간은 RF-DETR-S {pct(CI['rfdetr_s']['recall_CI95'][0])}~{pct(CI['rfdetr_s']['recall_CI95'][1])}, YOLO11s {pct(CI['yolo11s']['recall_CI95'][0])}~{pct(CI['yolo11s']['recall_CI95'][1])}, YOLOv8n {pct(CI['yolov8n']['recall_CI95'][0])}~{pct(CI['yolov8n']['recall_CI95'][1])}입니다. 상위 두 모델의 차이는 통계적으로 구분되지 않습니다.")
d('탐지 모델 3종 모두 이물 중심 적중률이 100%입니다. 놓친 이물은 0개이고, IoU 기준 미탐은 전부 박스 크기 차이에서 나왔습니다(3장).')
d('비전-언어 모델은 LoRA 학습으로 재현율이 3.8%에서 80.3%(검증)로 올랐지만, 박스 정확도가 낮고 confidence 점수를 내지 않아 임계값 조정이 불가능하며 처리 시간이 탐지 모델의 50배 이상입니다.')
o('최종 모델 선택과 이유')
d('최종 구성: RF-DETR-S + YOLO11s를 영상 단위로 혼용(S1)합니다. 영상마다 두 모델 중 confidence가 높은 모델의 예측을 사용하고, 두 모델의 판단이 엇갈리면 재검사로 보냅니다(4장).')
d('RF-DETR-S: 실제 검증·테스트에서 미탐이 가장 적고(FN 5·2), 1장 28ms로 실시간 검사에 쓸 수 있습니다.')
d('YOLO11s: 원래 대비에서는 RF-DETR-S와 비슷하고, 저대비에서 더 강해 RF-DETR-S의 약점을 보완합니다. 혼용 시 원래 성능은 유지되고 저대비 적중률이 높아졌습니다(표 4).')
s('YOLOv8n 기준선 대비 개선: 테스트 FN 7 → 2, 정밀도 95.4% → 98.8%. 두 모델 처리 시간 합은 1장 약 45ms입니다.')
o('학습 데이터 조합 비교 (가공·증강의 효과 검증)')
d('목적: 학습 데이터 4,372장 중 원본에서 만든 가공 영상이 실제 영상의 미탐을 줄이는지 확인했습니다. YOLOv8n을 같은 조건(30 epoch, 조기 종료 없음)으로 학습하고, 실제 검증셋으로만 판정했습니다.')
d('판정 기준(실험 전 고정): 검증 FN이 3개 이상 줄고 시드를 바꿔도 같은 방향일 때만 효과가 있다고 봅니다. A와 D는 시드 0·1로 두 번 학습해 학습마다 생기는 흔들림을 쟀습니다.')
EXN={'A_실제만':('A','실제 영상만'),'A_실제만_seed1':('A′','A, 시드 1'),'B_실제+이물합성':('B','A + 이물 합성'),'C_실제+이물제거':('C','A + 이물 제거'),'D_전체':('D','전체'),'D_전체_seed1':('D′','D, 시드 1'),'E_선택+960':('E','B + 입력 960px'),'F_전체+동료합성':('F','D + 별도 합성 762장')}
h='<div class=tcap>표 3. 학습 데이터 조합별 결과 (YOLOv8n, 30 epoch)</div><table><thead><tr><th>실험</th><th>학습 데이터</th><th>장수</th><th>검증 FN</th><th>검증 FP</th><th>검증 재현율</th><th>검증 mAP50</th><th>검증 mAP50-95</th><th>저대비 k=0.5<br>중심 적중률</th><th>저대비 k=0.35<br>중심 적중률</th><th>테스트 FN</th><th>학습 시간</th></tr></thead><tbody>'
for e,(code,lab) in EXN.items():
    r=P2[e]; v=r['eval']['val']; t=r['eval']['test']; m=r['meta']
    k5=(r['val_stress']['050']['center']+r['test_stress']['050']['center'])/2; k35=(r['val_stress']['035']['center']+r['test_stress']['035']['center'])/2
    h+=f"<tr><td>{code}</td><td class=l>{lab}</td><td>{m['train_images']:,}</td><td><b>{v['FN']}</b></td><td>{v['FP']}</td><td>{pct(v['recall'])}</td><td>{v['AP50']:.3f}</td><td>{v['AP50_95']:.3f}</td><td><b>{pct(k5,0)}</b></td><td>{pct(k35,0)}</td><td>{t['FN']}</td><td>{m['train_seconds']/60:.0f}분</td></tr>"
a(h+'</tbody></table><div class=cap>저대비 적중률은 검증·테스트 평균. 저대비 영상은 평가 전용 합성이며 학습에 쓰지 않았습니다.</div>')
a(f'<figure>{F("f10_ablation.png")}<figcaption>그림 7. 학습 데이터 조합별 미탐 수와 저대비 대응력</figcaption></figure>')
a("""<table><thead><tr><th>가공·조건</th><th>왜 했나</th><th>결과</th><th>결론</th></tr></thead><tbody>
<tr><td class=l>이물 합성 (B)</td><td class=l>위치 암기 방지, 형상·위치 다양화</td><td class=l>검증 FN 5 (A: 5·7). 저대비 k=0.5 적중률 79% → 60%</td><td class=l>미탐 감소 효과 확인 안 됨. 저대비 대응력은 오히려 낮아짐</td></tr>
<tr><td class=l>이물 제거 (C)</td><td class=l>이물 없는 사례로 오탐·위치 암기 감소</td><td class=l>검증 FN 5, FP 6 (A: FP 5). 적중률 73%</td><td class=l>오탐·미탐 모두 차이 없음</td></tr>
<tr><td class=l>전체 결합 (D)</td><td class=l>모든 가공을 합친 효과</td><td class=l>검증 FN 9·8 (두 시드), 적중률 37%·58% (평균 47%)</td><td class=l>미탐이 평균 2.5개 늘었으나 판정 기준(3개) 미만. 저대비 대응력이 가장 낮음</td></tr>
<tr><td class=l>입력 960px (E)</td><td class=l>작은 이물의 박스 불일치 감소</td><td class=l>검증 FN 9 (B: 5), 작은 이물 FN 10 (B: 7), 학습 2.1배</td><td class=l>개선 없음. 원본이 316~576px라 확대해도 정보가 늘지 않음</td></tr>
<tr><td class=l>별도 생성 합성 (F)</td><td class=l>다른 방식의 합성 추가 효과</td><td class=l>검증 FN 9, 적중률 66%</td><td class=l>개선 없음</td></tr></tbody></table>""")
d('정리하면, 원본에서 만든 가공 영상은 실제 검증 영상의 미탐을 줄이지 못했고, 실제 영상만으로 학습한 A가 미탐은 같거나 적으면서 저대비 이물에 가장 강했습니다(k=0.5에서 79%, 전체 결합 D는 두 시드 평균 47%).')
d('가능한 원인: 이물 제거·합성 영상에는 지운 자리의 희미한 흔적이 "이물 없음"으로 라벨되어 있어, 모델이 희미한 어두운 점을 정상으로 학습했을 수 있습니다. 이는 가설이며 추가 검증이 필요합니다.')
d('이에 따라 최종 모델은 실제 영상(A 구성)으로 다시 학습하는 것을 권장합니다. 본 보고서의 모델 비교(표 1)는 전체 결합(D 구성)으로 학습한 결과입니다.')
o('두 모델 혼용(앙상블) 실험')
d('목적: 원래 대비에서 가장 좋은 RF-DETR-S와 저대비에 강한 YOLO11s를 함께 쓰면 두 장점을 모두 얻을 수 있는지 확인했습니다. 다시 학습하지 않고 저장된 예측을 조합했습니다.')
d('S1(영상 단위 선택): 영상마다 두 모델의 최고 confidence를 비교해 더 높은 모델의 예측만 사용합니다. S2(박스 단위 선택): 같은 이물(IoU ≥ 0.3)로 겹치는 박스 중 confidence가 높은 박스를 남깁니다. S3: S2와 같되 각 모델 점수를 자기 검증 임계값으로 나눠 비교합니다.')
d('방식과 임계값은 검증셋에서만 정하고, 테스트셋에는 그대로 한 번 적용했습니다.')
ENN={'rfdetr_s':'RF-DETR-S 단독','yolo11s':'YOLO11s 단독','rfdetr_s+yolo11s|S1':'RF-DETR-S + YOLO11s · S1','rfdetr_s+yolo11s|S2':'RF-DETR-S + YOLO11s · S2','rfdetr_s+yolo11s|S3':'RF-DETR-S + YOLO11s · S3','rfdetr_s+yolov8n|S1':'RF-DETR-S + YOLOv8n · S1','rfdetr_s+yolov8n|S3':'RF-DETR-S + YOLOv8n · S3'}
h='<div class=tcap>표 4. 두 모델 혼용 결과 (임계값은 검증에서 결정)</div><table><thead><tr><th>구성</th><th>세트</th><th>FN</th><th>FP</th><th>재현율</th><th>정밀도</th><th>mAP50</th><th>mAP50-95</th><th>정상 오경보</th><th>저대비 k=0.5<br>중심 적중률</th><th>저대비 k=0.35<br>중심 적중률</th></tr></thead><tbody>'
for k,lab in ENN.items():
    for sp,sl in (('val','검증'),('test','테스트')):
        r=ENS[sp][k]
        h+=f"<tr>{f'<td rowspan=2 class=l><b>{lab}</b></td>' if sp=='val' else ''}<td>{sl}</td><td><b>{r['FN']}</b></td><td>{r['FP']}</td><td>{pct(r['recall'])}</td><td>{pct(r['precision'])}</td><td>{r['AP50']:.3f}</td><td>{r['AP50_95']:.3f}</td><td>{pct(r['normal_FA'])}</td><td><b>{pct(r['stress']['050'],0)}</b></td><td>{pct(r['stress']['035'],0)}</td></tr>"
a(h+'</tbody></table>')
d(f"S1(RF-DETR-S + YOLO11s)은 원래 대비에서 RF-DETR-S 단독과 같은 성능(테스트 FN {ENS['test']['rfdetr_s+yolo11s|S1']['FN']}, 재현율 {pct(ENS['test']['rfdetr_s+yolo11s|S1']['recall'])})을 유지하면서, 대비를 절반으로 낮춘 이물의 중심 적중률을 검증 {pct(ENS['val']['rfdetr_s']['stress']['050'],0)} → {pct(ENS['val']['rfdetr_s+yolo11s|S1']['stress']['050'],0)}, 테스트 {pct(ENS['test']['rfdetr_s']['stress']['050'],0)} → {pct(ENS['test']['rfdetr_s+yolo11s|S1']['stress']['050'],0)}로 높였습니다. 검증에서 본 경향이 테스트에서도 같은 방향으로 확인됐습니다.")
d('원래 대비의 미탐은 줄지 않았습니다. 남은 미탐은 두 모델 모두 위치는 맞혔으나 박스 크기가 달라 생긴 것이어서, 어느 모델의 박스를 골라도 해결되지 않습니다. 검증 영상 80장 중 71장은 RF-DETR-S의 점수가 더 높아 RF-DETR-S 예측이 쓰였습니다.')
d('YOLOv8n과의 조합은 이득이 없었고, 점수를 보정한 S3는 오히려 미탐이 늘었습니다(검증 FN 9). 모델별 점수 척도가 달라도 이 데이터에서는 원래 점수 비교(S1)가 가장 좋았습니다.')
s('검증 결과는 같은 데이터에서 방식을 고르고 평가한 값이라 낙관적일 수 있습니다. 저대비 평가는 평가 전용 합성 영상 기준입니다.')

# ---------------- CH3 ----------------
a('<h2 class=chap>□ 제3장. 영향요인 및 오류분석</h2>')
o('미탐의 원인: 박스 크기 불일치')
a('<table><thead><tr><th>모델</th><th>박스 불일치<br>(이물 위 예측, IoU 0.1~0.5)</th><th>점수 미달</th><th>후보 없음</th><th>이물 중심 적중률</th></tr></thead><tbody>' + ''.join(f"<tr><td class=l>{N[k]}</td><td>{S1[k]['val_fn_types']['box_mismatch']} / {S1[k]['test_fn_types']['box_mismatch']}</td><td>{S1[k]['val_fn_types']['below_threshold']} / {S1[k]['test_fn_types']['below_threshold']}</td><td>{S1[k]['val_fn_types']['no_candidate']} / {S1[k]['test_fn_types']['no_candidate']}</td><td>{pct(S1[k]['val_center_recall'])} / {pct(S1[k]['test_center_recall'])}</td></tr>" for k in ('rfdetr_s','yolo11s','yolov8n','qwen25vl3b_lora')) + '</tbody></table><div class=cap>값은 검증 / 테스트</div>')
d('탐지 모델의 미탐은 모두 이물 위에 박스를 그렸지만 IoU가 0.31~0.50에 그친 경우입니다. 정답 박스가 유난히 작거나(한 변 5~8px) 크거나 길쭉한(한 변 11px 이상) 이물에서, 모델은 학습 라벨의 대표 크기인 약 10px 박스를 그렸습니다.')
d('따라서 임계값을 낮춰도 미탐이 줄지 않고, 제품 단위 판정(보류 여부)에서는 놓친 제품이 없습니다.')
a(f'<figure>{F("f8_fn_gallery.png")}<figcaption>그림 8. RF-DETR-S·YOLO11s의 미탐 전체 (초록 = 정답, 빨강 = 예측)</figcaption></figure>')
o('조건별 미탐 (검증+테스트 이물 355개)')
a('<table><thead><tr><th>요인</th><th>구간</th><th>이물 수</th><th>RF-DETR-S<br>FN (재현율)</th><th>YOLO11s<br>FN (재현율)</th><th>YOLOv8n<br>FN (재현율)</th></tr></thead><tbody>' + ''.join(fail_rows(f) for f in ('크기','대비','배경 밝기','호기','형상','위치')) + '</tbody></table>')
s(f"구간 기준: 크기 = 박스 면적의 제곱근 3분위({RA['bins']['size_px'][0]:.1f}, {RA['bins']['size_px'][1]:.1f}px), 대비 = (주변 밝기 중앙값 − 이물 어두운 25% 평균) ÷ 주변 잡음 3분위, 배경 밝기 = 주변 밝기 3분위, 가장자리 = 제품 경계까지 25px 미만")
a(f'<figure>{F("f6_failure.png")}<figcaption>그림 9. 조건별 미탐 수</figcaption></figure>')
d('크기: 미탐이 작은 이물(9.5px 미만)에 집중됩니다(RF-DETR-S 7개 중 6개). 작은 이물에서 2px 박스 차이가 IoU를 크게 낮추기 때문입니다.')
d('호기: 2호기는 세 모델 모두 미탐 0개이며, 미탐은 1·3호기에서만 나왔습니다. 3호기는 영상이 커서 같은 이물이 입력에서 더 작아집니다.')
d('대비·배경: 실제 데이터 범위에서는 뚜렷한 경향이 없습니다. 위치: 평가셋에 가장자리 이물이 0개라 분석할 수 없습니다.')
o('저대비 이물 취약성 (평가 전용 합성)')
d('검증·테스트 실제 이물의 배경 대비를 k배(0.75, 0.5, 0.35, 0.25)로 낮춘 영상으로 같은 모델을 평가했습니다. 학습에는 쓰지 않았습니다.')
a(f'<figure>{F("f7_stress.png")}<figcaption>그림 10. 이물 대비를 낮췄을 때의 탐지율 (실선 검증, 점선 테스트)</figcaption></figure>')
d(f"k = 0.75까지는 세 모델 모두 이물 중심 적중률 93% 이상을 유지하지만, k = 0.5에서 RF-DETR-S {pct(st('rfdetr_s','val',0.5),0)}·YOLO11s {pct(st('yolo11s','val',0.5),0)}·YOLOv8n {pct(st('yolov8n','val',0.5),0)}(검증)로 떨어지고, k = 0.35에서는 절반 이상을 놓칩니다.")
d('원래 대비에서 가장 좋은 RF-DETR-S보다 YOLO11s가 저대비에서 더 강합니다. 이 결과가 두 모델을 함께 쓰는 근거이며, 실제로 혼용(S1) 시 저대비 적중률이 단일 모델보다 높았습니다(표 4). 합성 조건이므로 실제 흐릿한 이물의 탐지율로 해석하지 않습니다.')
d('학습 데이터 조합 비교(표 3)에서 같은 평가를 하면, 실제 영상만으로 학습한 모델이 저대비에 가장 강했습니다. 저대비 취약성의 일부는 모델 구조가 아니라 가공 학습 데이터에서 온 것으로 보입니다.')
o('오탐 분석')
d('오탐은 대부분 박스 불일치로 생긴 같은 이물의 중복 집계입니다. 이물이 없는 곳의 오탐은 세 모델 합쳐 1건(YOLOv8n 테스트), 라벨 누락 이물을 찾은 경우가 각 모델 1건입니다.')
d('미끼 박스(복원 흔적)에 반응한 오탐은 모든 탐지 모델에서 0건으로, 모델이 장비 표시 흔적에 의존하지 않음을 확인했습니다.')
o('Confidence 임계값 분석')
a(f'<figure>{F("f5_threshold.png")}<figcaption>그림 11. 임계값별 FN·FP·합성 정상 오경보율 (검증셋, 세로선 = 선택 임계값)</figcaption></figure>')
a('<div class=tcap>표 2. RF-DETR-S 임계값별 결과</div>' + sweep_table('rfdetr_s'))
d('미탐은 임계값 0.05~0.5 구간에서 거의 변하지 않고 0.6을 넘으면 급증합니다. 반대로 0.1 이하에서는 오탐과 정상 오경보가 급증합니다.')
d(f"안전 측면의 임계값 후보: 확정 판정선은 검증 F1 최대값(RF-DETR-S {TH['rfdetr_s']:.3f}, YOLO11s {TH['yolo11s']:.3f}), 재검사 하한은 합성 정상 경보율 10% 이하가 되는 가장 낮은 값(RF-DETR-S {TL['rfdetr_s']:.2f}, YOLO11s {TL['yolo11s']:.2f})입니다. 0.5를 그대로 쓰지 않고 검증 결과로 정했습니다.")
# ---------------- CH4 ----------------
a('<h2 class=chap>□ 제4장. 현장 활용방안</h2>')
o('표시 박스 없이 동작하는 독립 판정기')
d('최종 모델은 장비 표시가 그려지기 전의 X선 영상만 보고 판정합니다. 따라서 기존 검사 장비의 판정을 따라 하는 것이 아니라, 같은 영상을 독립적으로 한 번 더 검사하는 2차 판정기로 쓸 수 있습니다.')
d('기존 장비와 모델의 판정이 다르면 재검사 대상으로 분류해, 장비 알고리즘이 놓친 이물과 모델이 놓친 이물을 서로 보완합니다.')
o('PASS / RE-INSPECTION / REJECT 3단계 판정')
a(f"""<table><thead><tr><th>판정</th><th>조건 (검사 시점에 관찰 가능한 정보만 사용)</th><th>조치</th></tr></thead><tbody>
<tr><td><b>REJECT</b></td><td class=l>RF-DETR-S 최고 점수 ≥ {TH['rfdetr_s']:.3f} 이고 YOLO11s 최고 점수 ≥ {TH['yolo11s']:.3f}</td><td class=l>제품 보류·격리. 재촬영에서 안 보여도 자동 취소하지 않음</td></tr>
<tr><td><b>RE-INSPECTION</b></td><td class=l>두 모델 중 하나만 확정선을 넘음(판단 불일치), 또는 한 모델이라도 재검사 하한(RF-DETR-S {TL['rfdetr_s']:.2f}, YOLO11s {TL['yolo11s']:.2f}) 이상</td><td class=l>새로 X선 재촬영 또는 작업자 판독. 같은 영상을 확대해 다시 추론하는 것은 재검사로 보지 않음</td></tr>
<tr><td><b>PASS</b></td><td class=l>두 모델 모두 재검사 하한 미만</td><td class=l>통과</td></tr></tbody></table>""")
s('"작은 이물이면 재검사"처럼 모델이 놓치면 알 수 없는 정보는 기준으로 쓰지 않았습니다.')
s('영상 품질 이상(촬영 실패, 포화, 기준 시험편 이탈)은 모델 판정 전에 보류·재촬영합니다.')
s('REJECT 판정의 점수는 혼용 방식(S1)에 따라 두 모델 중 높은 confidence를 사용합니다. 아래 결과는 이 규칙과 동일합니다.')
o('판정 결과 (영상 단위)')
a('<table><thead><tr><th rowspan=2>판정 방식</th><th rowspan=2>세트</th><th colspan=3>이물 있는 영상 (80장)</th><th colspan=3>합성 정상 영상 (80장)</th></tr><tr><th>REJECT</th><th>RE-INSP.</th><th>PASS(놓침)</th><th>REJECT</th><th>RE-INSP.</th><th>PASS</th></tr></thead><tbody>' + pol_rows() + '</tbody></table>')
d('세 방식 모두 이물 있는 영상 160장을 모두 REJECT로 판정해 놓친 제품이 없었습니다.')
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
o('현장 도입 전 검증 계획')
d('목표 탐지율과 허용 재검사율을 먼저 정하고, 실제 흐릿한 시험편과 정상 제품을 독립 촬영해 확인합니다.')
a("""<table><thead><tr><th>목표 탐지율 (95% 신뢰하한)</th><th>미탐 0개일 때 필요 표본</th><th>미탐 1개</th><th>미탐 2개</th></tr></thead><tbody>
<tr><td>90%</td><td>29</td><td>46</td><td>61</td></tr><tr><td>95%</td><td>59</td><td>93</td><td>124</td></tr><tr><td>99%</td><td>299</td><td>473</td><td>628</td></tr></tbody></table>""")
s('정확 이항(Clopper–Pearson) 단측 95% 기준. 같은 시험편의 반복 촬영·증강본은 독립 표본으로 세지 않습니다. 정상 제품 재검사율 상한 5%·1%를 보이려면 각각 59·299개가 필요합니다.')
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
<tr><td>라벨 수 실험</td><td class=l>15~400장, 200장이 효율적</td><td class=l>가공 데이터 조합 A~F 비교, 실제 영상 위주 구성 권고</td></tr></tbody></table>''')
o('누출 없는 평가 설계')
d('원본 해시·촬영 세션·촬영 간격·이물 배치까지 점검해 학습과 평가 영상 사이의 직접 유출이 없음을 확인했고, 같은 시험편 반복이라는 구조적 한계를 수치(86%)로 밝혔습니다.')
o('미탐을 원인별로 분해한 평가')
d('IoU 기준 미탐을 박스 불일치·점수 미달·후보 없음으로 나눠, 남은 미탐이 위치를 놓친 것이 아니라 박스 크기 차이임을 보였습니다(중심 적중률 100%). 이 결과로 임계값을 낮추는 대신 재검사 기준을 설계했습니다.')
o('저대비 스트레스 평가로 모델 조합 근거 확보')
d('실제 데이터의 평균 성능만으로는 보이지 않던 차이(저대비에서 YOLO11s 우세)를 평가 전용 합성으로 드러내, 구조가 다른 두 모델의 판단 불일치를 재검사 신호로 쓰는 근거로 삼았습니다.')
o('약점이 다른 두 모델의 혼용')
d('평균 성능이 비슷한 두 모델의 조건별 강점(원래 대비 vs 저대비)을 찾아, 높은 confidence를 낸 모델을 쓰는 간단한 규칙만으로 원래 성능을 유지하면서 저대비 대응력을 높였습니다(테스트 k=0.5 적중률 84% → 90%).')
o('가공 데이터 효과의 사전 기준 검증')
d('합성·제거 영상을 "많을수록 좋다"고 가정하지 않고, 실험 전 판정 기준과 반복 학습으로 효과를 확인했습니다. 그 결과 가공 데이터가 저대비 대응력을 낮춘다는 위험을 찾아 학습 데이터 구성 권고로 연결했습니다.')
o('라벨 품질 감사')
d('장비 표시 모양을 전수 분류해 T 아이콘 라벨 6장, 합쳐진 박스 25장을 찾아 제외하고, 검증 라벨 누락 1건을 찾아 오탐 해석에 반영했습니다.')
# ---------------- CH6 ----------------
a('<h2 class=chap>□ 제6장. 코드 구성 및 재현성</h2>')
o('폴더 구성')
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
│  │   └─ report_analysis.py, report_figs.py, leak_check.py         분석·그림
│  ├─ preds/                        모델별 예측 CSV · 채점 JSON · 메타
│  └─ report/                       보고서 생성 코드·그림·분석 JSON</pre>""")
o('실행 순서')
a("""<pre># 1) 학습·예측·채점 (맥 MPS)
bash X-ray_실험/run_all.sh
# 2) 학습 데이터 조합 비교
bash X-ray_실험/phase2.sh
# 3) 단일 모델 채점
python scripts/evaluate.py preds/rfdetr_s.csv --out preds/rfdetr_s_eval.json
# 4) 보고서 분석·그림
python scripts/report_analysis.py && python scripts/report_figs.py</pre>""")
o('실행 환경과 재현 조건')
d('macOS 26.6 (Apple M5 Pro, MPS), Python 3.12.14, torch 2.14.1, ultralytics 8.4.172, rfdetr 1.11.1, transformers 5.18.0, peft 0.21.2. 시드는 0입니다.')
d('예측은 confidence 0.001 이상을 모두 저장한 뒤 채점 단계에서 임계값을 적용하므로, 임계값 분석에 재학습이 필요 없습니다.')
s('MPS 연산의 비결정성 때문에 재학습 시 수치가 소폭 달라질 수 있습니다.')
o('한계와 향후 과제')
d('표시 박스가 그려지기 전의 실제 원본 영상이 없어, 표시를 지운 영상으로 대신 평가했습니다.')
d('같은 시험편 반복 촬영 데이터라 처음 보는 제품에 대한 성능은 검증하지 못했습니다.')
d('실제 정상 제품이 없어 실제 오경보율을 측정하지 못했고, 가장자리 이물은 평가셋에 없습니다.')
d('평가 표본이 각 80장(16·22세션)으로 작아 상위 모델 간 차이는 통계적으로 구분되지 않습니다.')
d('실제 영상 중심 구성(표 3의 A·C)으로 최종 모델을 다시 학습해 표 1과 비교하는 작업이 남아 있습니다.')
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
    a('<div class=tcap>성능 (점수를 내지 않아 임계값 분석 없음)</div>' + main_table([k]))
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
a('<h2 class=chap>□ 경진대회 만족도 조사 완료 페이지 캡쳐 화면(필수)</h2><div class=d>- 만족도 조사 url : https://naver.me/FetWt7SN</div><div class=box style="height:140mm;display:flex;align-items:center;justify-content:center;color:#888">만족도 조사 완료 화면 캡처를 여기에 첨부</div>')
html = f'<!doctype html><html lang=ko><head><meta charset=utf-8><title>제6회 K-인공지능 제조데이터 분석 경진대회 보고서</title><style>{css}</style></head><body>{"".join(P)}</body></html>'
open(f'{R}/report/report.html', 'w').write(html); print(len(html))
