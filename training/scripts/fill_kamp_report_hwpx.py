#!/usr/bin/env python3
"""Fill the official KAMP HWPX result-report template without altering the original."""

from __future__ import annotations

import copy
import csv
import io
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT = Path("/Users/kyungrae/Desktop/kamp_ai")
TEMPLATE = Path("/Users/kyungrae/Downloads/경진대회 결과보고서 양식_일반국민,대학(원)생 부문.hwpx")
OUTPUT = PROJECT / "training" / "reports" / "KAMP_Xray_5fold_결과보고서_초안.hwpx"
CV_ROOT = PROJECT / "training" / "experiments" / "point_decoy_roi_5fold"
HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"


def text_of(node: ET.Element) -> str:
    return "".join(t.text or "" for t in node.iter(f"{{{HP}}}t"))


def clear_layout(paragraph: ET.Element) -> None:
    for child in list(paragraph):
        if child.tag == f"{{{HP}}}linesegarray":
            paragraph.remove(child)


def make_bullet(template: ET.Element, content: str) -> ET.Element:
    paragraph = copy.deepcopy(template)
    texts = list(paragraph.iter(f"{{{HP}}}t"))
    if len(texts) < 2:
        raise ValueError("Unexpected bullet template")
    texts[1].text = content
    for extra in texts[2:]:
        extra.text = ""
    clear_layout(paragraph)
    paragraph.set("id", "0")
    return paragraph


def fill_cell(table: ET.Element, row: int, col: int, value: str) -> None:
    for cell in table.iter(f"{{{HP}}}tc"):
        addr = cell.find(f"{{{HP}}}cellAddr")
        if addr is None or addr.get("rowAddr") != str(row) or addr.get("colAddr") != str(col):
            continue
        paragraph = next(cell.iter(f"{{{HP}}}p"))
        run = paragraph.find(f"{{{HP}}}run")
        if run is None:
            raise ValueError("Missing cover run")
        text = run.find(f"{{{HP}}}t")
        if text is None:
            text = ET.SubElement(run, f"{{{HP}}}t")
        text.text = value
        clear_layout(paragraph)
        return
    raise ValueError(f"Cover cell {row},{col} not found")


def report_sections(summary: dict) -> list[list[tuple[str, str]]]:
    c, p, d, r = (summary[name] for name in ("control", "point", "decoy", "roi"))
    fmt = lambda item: f"{item['mean']:.4f}±{item['sd']:.4f}"
    return [
        [
            ("main", "분석 대상과 제조공정 의미"),
            ("detail", "X-ray 검사장비 1·2·3호기의 완제품 검사 프레임을 분석했다. 한 장은 검사 시점의 제품 내부 투과영상이며, 작은 이물 후보의 위치를 박스로 표시하는 것이 목표다. 영상은 연속 촬영 자료에서 추출된 프레임으로 보이며 독립 표본으로 취급하지 않았다."),
            ("detail", "정확 중복을 제거한 고유 영상은 2,532장이다. 개발 세트 2,157장(545개 촬영 세션), 고정 test 375장(95개 세션)으로 분리했다. 개발 세트 약한 정답 박스는 3,779개, test는 673개다."),
            ("main", "라벨·결측·불균형 진단"),
            ("detail", "원본 이미지에 그려진 색상 사각형에서 YOLO 좌표를 자동 추출했다. 따라서 이 좌표는 실제 이물을 사람이 다시 확인한 정답이 아니라 약한 라벨이다. 색상 표시 픽셀은 마스크로 분리하고 주변 RGB를 이용한 Telea 보간으로 정제 영상을 만들었다."),
            ("detail", "무라벨 고유 영상 105장은 잠정 음성으로 사용했다. 개발 세트 89장, test 16장이며 모두 1호기 자료다. 무라벨은 수동으로 정상 판정을 확인한 자료가 아니므로 정상 특이도, 특히 2·3호기 정상 특이도의 근거가 부족하다."),
            ("main", "전처리와 누출 방지"),
            ("detail", "중복 영상은 해시로 제거하고, 호기·촬영 시각을 기준으로 가까운 프레임을 촬영 세션으로 묶었다. 5-fold는 이미지가 아니라 세션 단위로 나눴고 train/validation/test의 세션 교집합은 0이다. fold 검증 장수는 429~433장이다."),
            ("detail", "모든 변형 데이터는 각 fold의 학습 원본에서만 생성했다. 검증 변형은 해당 fold의 미사용 세션에서 따로 생성했으며, test 영상은 5-fold 모델 선택에 사용하지 않았다."),
        ],
        [
            ("main", "비교 모델 및 공통 학습 조건"),
            ("detail", "YOLOv3-SPP(입력 320×320, 단일 이물 클래스)를 사용했다. 기존 세션 분리 5-fold 기본 모델의 fold별 가중치에서 네 데이터 구성 모델을 각각 2 epoch 미세조정했다. 각 fold는 원본 학습 영상 약 1,724~1,728장과 동일하게 선택한 추가 예시 600장으로 구성해 학습량을 맞췄다."),
            ("detail", "대조군은 원본 600장을 반복했다. 합성 점은 제품 영역 우선 위치에 반경 3~5px의 어두운 점을 삽입하고 생성 좌표의 정확한 박스를 추가했다. 미끼 방식은 정답과 떨어진 빨강·노랑 사각형을 넣고 정답은 추가하지 않았다. ROI 방식은 기존 박스 주변 160×160px을 잘라 라벨 좌표를 변환했다."),
            ("detail", "Apple M5 Pro 내장 GPU(MPS), batch 4, gradient accumulation 4, learning rate 0.0001, seed 20261001을 사용했다. 운영 후보의 confidence 0.07을 모든 fold에 고정했다. 예측 중심이 약한 정답 박스에 들어가는 F1을 주지표, IoU 0.5 AP를 보조지표로 사용했다."),
            ("main", "동일 세션 5-fold 성능: 평균±표본표준편차"),
            ("detail", f"원본 반복학습 대조군: 원본 중심점 F1 {fmt(c['original_f1'])}, AP@0.5 IoU {fmt(c['original_ap50_iou'])}. 합성 점 발견율 {fmt(c['point_recall'])}, 미끼 위치 검출률 {fmt(c['decoy_hit_rate'])}."),
            ("detail", f"합성 점 학습: 원본 F1 {fmt(p['original_f1'])}, 합성 점 발견율 {fmt(p['point_recall'])}, 미끼 검출률 {fmt(p['decoy_hit_rate'])}. 새 점은 찾지만 원본 성능과 교란 안정성은 악화됐다."),
            ("detail", f"미끼 박스 학습: 원본 F1 {fmt(d['original_f1'])}, AP@0.5 IoU {fmt(d['original_ap50_iou'])}, 미끼 검출률 {fmt(d['decoy_hit_rate'])}. 원본 F1은 5개 fold 중 4개에서 대조군보다 높았고, 미끼 검출은 5개 모두에서 감소했다."),
            ("detail", f"ROI crop 학습: 원본 F1 {fmt(r['original_f1'])}, 정답 위치를 알고 자른 ROI F1 {fmt(r['roi_f1'])}. 원본 F1은 5개 fold 모두에서 대조군보다 낮았다."),
            ("main", "선정 근거와 최종 가중치의 지위"),
            ("detail", f"미끼 방식은 원본 F1 평균 {c['original_f1']['mean']:.4f}→{d['original_f1']['mean']:.4f}로 거의 유지하며 미끼 위치 검출을 총 {c['decoy_hits_total']}→{d['decoy_hits_total']}건(약 69% 감소)으로 줄였다. 평균 원본 F1 차이 +0.0015는 작은 변화이므로 통계적으로 확정된 향상이라고 주장하지 않는다."),
            ("detail", "전체 개발 데이터에 미끼를 추가 학습한 별도 잠정 가중치는 기존 최강 모델 대비 고정 test 원본 F1 0.9933→0.9926, 미끼 영상 F1 0.9666→0.9889였다. 이 test는 이전 모델 비교에 반복 사용됐으므로 독립 최종 성능이 아닌 탐색적 수치다."),
        ],
        [
            ("main", "호기별 오차 집중"),
            ("detail", "5-fold 원본 검증을 합산하면 대조군의 3호기 재현율은 0.9294(미탐 88건), 원본 오탐 168건이었다. 1·2호기보다 오류가 집중됐다. 미끼 모델의 3호기 재현율도 0.9294(미탐 88건)로 같고 오탐은 158건으로 소폭 줄었다."),
            ("detail", "합성 점 모델은 3호기 재현율을 0.9463으로 높였지만 3호기 원본 오탐도 337건으로 증가했다. ROI 모델은 3호기 재현율 0.8974와 오탐 330건으로 전체 영상 목적에는 불리했다."),
            ("main", "박스 교란과 오류 조건"),
            ("detail", f"미끼 박스가 놓인 검증 영상 2,157장에서 대조군은 미끼 위치를 {c['decoy_hits_total']}번, 미끼 학습 모델은 {d['decoy_hits_total']}번 검출했다. 모델이 색상·직사각형 교란에 취약하다는 진단이나, 미끼를 무시한다는 사실이 실제 이물 특징을 학습했다는 증명은 아니다."),
            ("detail", "원본 색상 박스의 보간 흔적이 남았을 가능성과 가까운 촬영 프레임의 유사성이 주요 편향 위험이다. 현재 약한 라벨로는 이물 크기·제품 가장자리 거리·실제 밀도 대비에 따른 미탐 조건을 정확히 분류할 수 없다."),
            ("detail", "후속 오류분석에서는 사람 검수 정답으로 이물 크기, 제품 경계까지의 거리, 국소 명암 대비를 구간화하고, 각 구간의 FN/FP와 세션 간 편차를 평가해야 한다. 특히 2·3호기 정상 영상을 확보해 오경보율을 따로 계산해야 한다."),
        ],
        [
            ("main", "검사 우선순위와 재검사 흐름"),
            ("detail", "모델이 제시한 좌표와 confidence를 검사자 화면에 함께 표시한다. 현재 0.07은 약한 라벨 기반 검증에서 정한 연구용 후보 임계값이며, 이를 넘지 않았다고 자동 합격 판정해서는 안 된다."),
            ("detail", "임계값 이상 검출은 제품 분리·재검사 대상으로 보내고, 낮은 신뢰도 또는 제품 가장자리·작은 이물 의심 프레임은 작업자 확인 대상으로 둔다. 재촬영 가능 여부와 검사 비용을 고려한 운영 기준은 수동 정답 자료에서 놓침 비용과 오경보 비용을 비교해 결정한다."),
            ("detail", "연속 영상에서는 인접 프레임의 좌표·시간을 연결해 같은 제품의 반복 검출과 순간 오탐을 구분하는 후처리를 제안한다. 이번 실험은 추출 프레임 평가이므로 프레임 결합 효과와 시간당 오경보 수는 아직 측정하지 않았다."),
            ("main", "도입 전 통과 조건"),
            ("detail", "각 호기의 정상·불량 신규 세션을 확보하고 사람이 직접 표시한 이물 정답을 구축한다. 호기별 재현율, 크기·가장자리 조건별 미탐, 정상 1,000장당 오경보, 연속 영상 재검사량을 확인한 뒤 임계값을 승인한다."),
        ],
        [
            ("main", "현장 취약성을 겨냥한 데이터 실험"),
            ("detail", "사각형 표시를 제거해 학습하는 데 그치지 않고, 정답이 아닌 미끼 박스를 직접 그려 음성 예시로 학습시켰다. 원본 반복학습 대조군을 두고 같은 초기 가중치·영상 수·epoch에서 비교해 단순 추가 학습 효과와 미끼 효과를 분리했다."),
            ("detail", "합성 점에 생성 좌표 정답을 부여해 점 검출 가능성을 확인했으나, 원본 F1이 0.9628→0.9429로 떨어지고 미끼 위치 검출률이 0.0849→0.2278로 올라 실제 이물 일반화와 구별했다."),
            ("detail", "ROI 확대는 oracle crop F1을 0.2861→0.6137로 높였지만, 원본 영상 F1은 0.9628→0.9373으로 떨어졌다. 정답 위치를 먼저 아는 평가를 실제 배포 성능으로 제시하지 않는 점이 본 실험의 중요한 구분이다."),
            ("detail", "추가 탐색 실험에서 학습하지 않은 파랑·초록색, 더 두껍고 큰 미끼에 대해 기존 최강 모델 47/375장, 미끼 추가 학습 모델 8/375장에서 미끼 위치 검출이 발생했다. test 반복 사용을 감안하면 이 결과는 보조 진단이다."),
        ],
        [
            ("main", "재현 가능한 처리 순서"),
            ("detail", "① 원본 중복 제거·표시 박스 추출·표시 픽셀 마스킹 및 보간 → ② 세션 단위 5-fold 분할 → ③ fold별 원본·합성 점·미끼·ROI 학습 목록 생성 → ④ 같은 조건으로 YOLOv3-SPP 학습 → ⑤ 신뢰도 0.07에서 네 검증 화면 평가 → ⑥ 평균·표준편차·호기별 지표 합산 순서로 실행한다."),
            ("detail", "주요 실행 파일은 training/scripts/experiment_variants.py, generate_cv_variants.py, run_cv_variant_models.py, evaluate_variants.py, summarize_cv_variants.py다. 각 fold의 매니페스트·학습 목록·설정·로그·가중치·JSON 지표는 training/experiments/point_decoy_roi_5fold에 저장했다. 1번 fold는 선행 동일 설정 결과를 재사용했다."),
            ("detail", "Python 3.12, PyTorch 2.14, OpenCV 5.0, Apple MPS를 사용했다. 원본 자료와 정제 자료의 경로, 중복 제거 기준, 세션 배정, 고정 임계값을 함께 제공해야 다른 환경에서 같은 평가가 가능하다."),
            ("main", "제출 전 확인 사항"),
            ("detail", "공식 제출에는 보고서 PDF, 소스코드 ZIP(의존성 목록·학습 데이터·README·테스트 예측결과), 발표자료 PDF/PPT, 만족도 조사 완료 화면이 요구된다. 본 HWPX는 결과보고서 초안이며 팀명·서명·설문 캡처 및 실제 제출용 테스트 예측 파일은 참가자가 확인·보완해야 한다."),
        ],
    ]


def main() -> None:
    summary = json.loads((CV_ROOT / "summary.json").read_text())["summary"]
    with zipfile.ZipFile(TEMPLATE) as archive:
        section = archive.read("Contents/section0.xml")
        namespaces = dict(ET.iterparse(io.BytesIO(section), events=("start-ns",)))
        for prefix, uri in namespaces.items():
            if prefix not in ("xml", "xmlns"):
                ET.register_namespace(prefix or "", uri)
        root = ET.fromstring(section)
        original = list(root)
        cover = original[0]
        table = next(cover.iter(f"{{{HP}}}tbl"))
        fill_cell(table, 1, 1, "X-ray 영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석")
        fill_cell(table, 3, 1,
                  "세션 분리 5-fold에서 YOLOv3-SPP의 데이터 구성 4종을 비교했다. 미끼 박스 음성학습은 원본 중심점 F1 0.9628→0.9643을 유지하며 미끼 위치 검출을 183→56건 줄였다. 단, 자동 추출된 약한 라벨 기준이며 독립 수동 정답과 2·3호기 정상 영상 검증이 필요하다.")
        headings = [original[index] for index in (1, 8, 16, 24, 32, 40)]
        main_template = original[2]
        detail_template = original[5]
        children = [cover]
        sections = report_sections(summary)
        for heading, body in zip(headings, sections, strict=True):
            clear_layout(heading)
            children.append(heading)
            for level, content in body:
                children.append(make_bullet(main_template if level == "main" else detail_template, content))
        survey_heading, survey_url = original[58], original[59]
        clear_layout(survey_heading)
        clear_layout(survey_url)
        children.extend((survey_heading, survey_url,
                         make_bullet(detail_template, "설문 완료 화면은 미첨부 상태입니다. 제출 전 참가자가 실제 완료 화면 캡처를 이 위치에 삽입해야 합니다.")))
        root[:] = children
        updated = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + ET.tostring(root, encoding="utf-8")
        ET.fromstring(updated)
        preview_lines = ["제6회 K-인공지능 제조데이터 분석 경진대회 결과보고서",
                         "X-ray 영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석"]
        for child in children[1:]:
            s = text_of(child).strip()
            if s:
                preview_lines.append(s)
        preview = "\n".join(preview_lines).encode("utf-8")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(OUTPUT, "w") as out:
            for info in archive.infolist():
                data = updated if info.filename == "Contents/section0.xml" else (
                    preview if info.filename == "Preview/PrvText.txt" else archive.read(info.filename))
                out.writestr(info, data)
    with zipfile.ZipFile(OUTPUT) as check:
        assert check.testzip() is None
        section_check = ET.fromstring(check.read("Contents/section0.xml"))
        full_text = text_of(section_check)
        assert all(f"제{i}장" in full_text for i in range(1, 7))
        assert "작성 요령" not in full_text and "휴먼명조 14" not in full_text
    print(OUTPUT)


if __name__ == "__main__":
    main()
