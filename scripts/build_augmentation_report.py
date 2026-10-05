#!/usr/bin/env python3
"""Create the standalone PDF record of KAMP X-ray preprocessing/augmentation."""

from __future__ import annotations

import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle


PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "processed"
OUT = PROCESSED / "AUGMENTATION_WORKFLOW_REPORT.pdf"
FONT = Path("/Applications/Hancom Office HWP Viewer.app/Contents/Resources/Hnc/Shared/TTF/Install")
pdfmetrics.registerFont(TTFont("HanBatang", str(FONT / "HANBatang.ttf")))
pdfmetrics.registerFont(TTFont("HanBatangBold", str(FONT / "HANBatangB.ttf")))
pdfmetrics.registerFontFamily("HanBatang", normal="HanBatang", bold="HanBatangBold")


def p(value: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(html.escape(value).replace("\n", "<br/>"), style)


def styles() -> dict[str, ParagraphStyle]:
    base = {"textColor": colors.HexColor("#17212b"), "wordWrap": "CJK"}
    return {
        "title": ParagraphStyle("title", **base, fontName="HanBatangBold", fontSize=22, leading=31, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("subtitle", **base, fontName="HanBatang", fontSize=11, leading=17, alignment=TA_CENTER),
        "h1": ParagraphStyle("h1", **base, fontName="HanBatangBold", fontSize=16, leading=24, spaceAfter=13),
        "h2": ParagraphStyle("h2", **base, fontName="HanBatangBold", fontSize=12, leading=19, spaceBefore=11, spaceAfter=5),
        "body": ParagraphStyle("body", **base, fontName="HanBatang", fontSize=10.6, leading=17, spaceAfter=7),
        "small": ParagraphStyle("small", fontName="HanBatang", fontSize=9.1, leading=14,
                                textColor=colors.HexColor("#4d5b67"), wordWrap="CJK", spaceAfter=5),
        "cell": ParagraphStyle("cell", **base, fontName="HanBatang", fontSize=9, leading=13),
        "cellhead": ParagraphStyle("cellhead", **base, fontName="HanBatangBold", fontSize=9, leading=13),
        "code": ParagraphStyle("code", **base, fontName="HanBatang", fontSize=9.1, leading=15, leftIndent=10,
                               backColor=colors.HexColor("#f3f6f8"), borderPadding=8, spaceAfter=8),
    }


def table(data: list[list[str]], widths: list[int], st: dict[str, ParagraphStyle]) -> Table:
    rows = [[p(cell, st["cellhead" if i == 0 else "cell"]) for cell in row] for i, row in enumerate(data)]
    result = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7edf2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafb")]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.65, colors.HexColor("#637784")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#bdc8cf")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return result


def scaled_image(path: Path, max_width: float = 491, max_height: float = 300) -> Image:
    from PIL import Image as PilImage
    with PilImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def footer(canvas, document) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#d4dce2"))
    canvas.line(52, 45, width - 52, 45)
    canvas.setFont("HanBatang", 8.5)
    canvas.setFillColor(colors.HexColor("#63717b"))
    canvas.drawString(52, 30, "KAMP X-ray 데이터 전처리 및 증강 작업 기록 | 2026-10-02")
    canvas.drawRightString(width - 52, 30, str(document.page))
    canvas.restoreState()


def main() -> None:
    st = styles()
    prep = json.loads((PROCESSED / "processing_summary.json").read_text())
    curated = json.loads((PROCESSED / "curated/split_summary.json").read_text())
    noise = json.loads((PROCESSED / "augmentation_noise_faint_v1/summary.json").read_text())
    combo = json.loads((PROCESSED / "three_position_combinations_v1/summary.json").read_text())
    cv = json.loads((PROJECT / "training/experiments/point_decoy_roi_5fold/summary.json").read_text())["summary"]
    story: list = []

    story += [Spacer(1, 45), p("KAMP X-ray 데이터 전처리·증강 작업 기록", st["title"]), Spacer(1, 15),
              p("실행 결과, 품질 한계, 재현 절차 | 2026년 10월 2일", st["subtitle"]), Spacer(1, 47)]
    story.append(table([
        ["항목", "확인된 결과"],
        ["원본·정제", f"원본 {prep['images']:,}장 전체 정제, 색상 표시 제거본·YOLO 라벨·마스크·비교 미리보기 생성"],
        ["중복·분할", f"정확 중복 {curated['duplicate_copies_excluded']:,}장 제외, 고유 {curated['unique_images']:,}장, 세션 단위 개발 2,157장 / 고정 테스트 375장"],
        ["노이즈·저대비", f"4개 처리 폴더에 각각 {noise['generated_per_variant']:,}장과 대응 라벨 생성; 오류 {len(noise['errors'])}건"],
        ["3위치 조합", f"3박스 개발 영상 {combo['source_candidates']}장 중 {combo['accepted_sources']}장 품질 통과; 7조합 총 {combo['generated_images']:,}장 생성"],
    ], [103, 390], st))
    story += [Spacer(1, 28), p("결론", st["h2"]),
              p("새 증강 데이터는 실험 후보로 준비되었다. 기존 5-fold에서는 미끼 박스 학습이 오검출을 줄였지만, 이번 노이즈·저대비·3위치 조합은 아직 모델에 학습시키지 않았다. 데이터 파일 수 증가는 독립 표본 수 증가나 현장 성능 향상을 뜻하지 않는다.", st["body"]),
              Spacer(1, 18), p("데이터 사용 원칙", st["h2"]),
              p("원본은 보존한다. 중복, 같은 촬영 세션, 해당 fold의 검증 영상, 고정 테스트 영상이 학습에 섞이지 않도록 매니페스트로 선별한다. 자동 추출 박스는 약한 라벨이므로 사람의 정답 검수가 필요하다.", st["body"])]

    story += [PageBreak(), p("1. 원본 정제와 누출 방지", st["h1"])]
    story.append(table([
        ["단계", "수행 내용", "확인 사항"],
        ["표시 탐지", "색상 차이·밝기로 원본의 빨강/노랑/파랑 테두리 픽셀 검출", "색상 픽셀만 분리"],
        ["박스 라벨", "사각형 성분을 추출해 YOLO 중심 x·y·너비·높이로 정규화", "수동 확정 정답이 아닌 약한 라벨"],
        ["표시 제거", "검출 픽셀과 1픽셀 주변만 마스킹한 뒤 OpenCV Telea 보간(반경 3)", "마스크 외부 픽셀은 유지"],
        ["중복·분할", "원본 SHA-256 완전 중복 제외; 호기·날짜·15초 이내 프레임을 세션으로 묶음", "한 세션은 한 분할에만 배치"],
    ], [79, 270, 144], st))
    story += [Spacer(1, 14), p(f"정제 출력은 이미지 {prep['images']:,}장, 최초 추출 박스 {prep['labels']:,}개다. 정확 중복 {curated['duplicate_copies_excluded']:,}장을 제외하면 고유 영상 {curated['unique_images']:,}장, 약한 박스 {curated['boxes']:,}개가 남는다. 무라벨 고유 영상 {curated['negative_user_confirmed_images']}장은 잠정 음성으로 포함했으며, 실제 정상이라는 수동 판정과 동일하지 않다.", st["body"]),
              p("`processed/images_clean`, `labels`, `masks`, `previews`, `processing_log.csv`가 전수 처리 결과다. `processed/curated`는 중복을 제외하고 세션별 train/validation/test를 기록한다. 이전 문서의 학습 차단 문구는 당시 승인 전 상태의 기록이며, 이후 사용자의 승인으로 별도 학습 실험을 진행했다.", st["body"]),
              p("한계", st["h2"]),
              p("색상 테두리의 원래 내부 픽셀은 관측되지 않았고 Telea는 주변에서 추정한 값만 채운다. 따라서 완전한 물리적 복원이 아니다. 표시 위치에서 만든 박스도 실제 이물 경계보다 클 수 있다.", st["body"])]

    story += [PageBreak(), p("2. 선행 데이터 구성 실험과 실제 5-fold 결과", st["h1"]),
              p("기존 YOLOv3-SPP fold별 가중치에서 대조군·합성 점·미끼 박스·ROI crop을 같은 세션 분할과 추가 학습 조건으로 비교했다. 아래 값은 원본 검증 영상 중심점 F1의 fold 평균이다.", st["body"])]
    t = [["데이터 구성", "원본 F1", "합성 점 발견율", "미끼 위치 검출률"]]
    for key, name in [("control", "원본 반복"), ("point", "합성 점"), ("decoy", "미끼 박스"), ("roi", "ROI crop")]:
        item = cv[key]
        t.append([name, f"{item['original_f1']['mean']:.4f}±{item['original_f1']['sd']:.4f}",
                  f"{item['point_recall']['mean']:.4f}", f"{item['decoy_hit_rate']['mean']:.4f}"])
    story += [table(t, [113, 103, 143, 134], st), Spacer(1, 10),
              p("합성 점은 반경 3-5px의 어두운 점을 삽입하고 정확한 생성 좌표를 라벨로 추가했다. 미끼는 정답이 아닌 색상 사각형을 그려 음성 예시로 사용했다. ROI crop은 기존 박스 주변 160×160 영역을 오리고 라벨을 변환했다.", st["body"]),
              p(f"미끼 방식은 원본 F1 {cv['control']['original_f1']['mean']:.4f}→{cv['decoy']['original_f1']['mean']:.4f}를 거의 유지하면서 미끼 위치 검출을 {cv['control']['decoy_hits_total']}→{cv['decoy']['decoy_hits_total']}건으로 줄였다. 차이가 작으므로 원본 성능 향상이 확정되었다고 주장하지 않는다. 점·ROI는 각 합성 화면의 성능은 높였지만 원본 전체 영상 F1은 대조군보다 낮았다.", st["body"]),
              p("이 실험 결과는 새로 만든 노이즈·저대비·3위치 조합의 효과를 검증한 값이 아니다.", st["small"])]

    story += [PageBreak(), p("3. 노이즈와 저대비 이물 증강", st["h1"]),
              p("기존 표시 제거 이미지 2,809장을 모두 새 디렉터리에 복사했다. 영상마다 8비트 기준 표준편차 2.5-9.5의 평균 0 가우시안 노이즈를 별도로 뽑았다. 기존 약한 박스 안의 작은 어두운 특징은 국소 명암 차이의 40-75%를 줄여 더 옅게 만들었다. 광학적 초점 블러가 아니라 진하기 조절이다.", st["body"])]
    story.append(table([
        ["폴더", "처리", "라벨"],
        ["01_overlay_removed_clean", "표시 제거본 원형 보존", "기존 라벨"],
        ["02_random_gaussian_noise", "무작위 강도 노이즈", "기존 라벨"],
        ["03_fainter_labeled_defect", "박스 안의 어두운 특징만 약화", "기존 라벨"],
        ["04_fainter_defect_plus_noise", "저대비화 후 동일 노이즈 실현값 적용", "기존 라벨"],
    ], [193, 200, 100], st))
    noise_preview = next((PROCESSED / "augmentation_noise_faint_v1/previews").glob("m2_1__*.png"))
    story += [Spacer(1, 12), scaled_image(noise_preview, max_height=190),
              p("비교 미리보기: 왼쪽부터 정제본, 노이즈, 옅은 이물, 둘의 조합. 주황색 박스는 미리보기에만 있다.", st["small"]),
              p(f"자동 처리 오류는 {len(noise['errors'])}건이다. 무라벨 원본 106장은 지울 대상이 없어 저대비 전용 폴더에서는 원형 유지된다. 테스트 영상 6장의 왼쪽 위 문자 표시가 약한 이물 박스로 잡혀 그 위치는 명암 변경에서 제외하고 매니페스트에 표시했다. 라벨 자체는 임의 수정하지 않았다.", st["body"])]

    story += [PageBreak(), p("4. 세 위치의 존재·부재 조합", st["h1"]),
              p("개발 영상 중 약한 박스가 정확히 3개인 고유 영상 843장을 대상으로 했다. 위·중간·아래 박스 순서로 `o`는 유지, `x`는 해당 박스 중앙의 작은 어두운 핵을 국소 Telea 보간으로 제거한다. 원본 `ooo`는 별도 복제하지 않았다.", st["body"])]
    story.append(table([
        ["남는 이물 수", "조합", "생성 수"],
        ["2개", "xoo, oxo, oox", f"{combo['accepted_sources'] * 3:,}장"],
        ["1개", "xxo, xox, oxx", f"{combo['accepted_sources'] * 3:,}장"],
        ["0개", "xxx (빈 YOLO 라벨)", f"{combo['accepted_sources']:,}장"],
    ], [100, 275, 118], st))
    story += [Spacer(1, 10), p(f"전체 843장 중 826장({combo['accepted_sources']/combo['source_candidates']:.1%})이 제거 품질 기준을 통과했고, 17장은 어두운 점이 충분히 지워지지 않아 제외했다. 통과 영상에서 총 {combo['generated_images']:,}개 증강 이미지와 대응 라벨을 생성했다. 고정 테스트와 정확 중복은 생성 대상에서 제외했다.", st["body"])]
    combo_preview = next((PROCESSED / "three_position_combinations_v1/previews").glob("m1_1__*.png"))
    story += [scaled_image(combo_preview, max_height=250),
              p("8칸 비교: ooo 원본과 7개 합성 조합. 주황색 표시는 미리보기에서 남겨 둔 위치만 보여준다.", st["small"])]

    story += [PageBreak(), p("5. 품질 판단과 성능 검증 계획", st["h1"]),
              p("지금 확인한 것은 파일 생성과 영상 복원 품질이다. 새 증강 데이터를 학습한 뒤의 정확도·미탐·오경보 수치는 아직 없다. 조합 7장은 같은 원본에서 나왔으므로 서로 독립된 7개 관측치로 해석하면 과적합과 성능 과대평가 위험이 있다.", st["body"])]
    story.append(table([
        ["확인 항목", "이번 결과", "다음 검증"],
        ["생성 성공", f"3박스 843장 중 {combo['accepted_sources']}장 채택, 17장 제외", "제외 장면 육안 검수"],
        ["라벨 일관성", "패턴의 o 개수와 YOLO 행 수 일치", "가려진 점의 잔상·보간 경계 수동 확인"],
        ["누출 방지", "고정 테스트·정확 중복 제외, fold 열 기록", "각 fold에서 해당 검증 세션과 파생본 전부 차단"],
        ["효과", "미측정", "같은 원본 수·epoch로 대조군과 5-fold 비교"],
    ], [100, 181, 212], st))
    story += [Spacer(1, 12), p("권장 실험", st["h2"]),
              p("원본 기준 모델과 원본+조합 모델을 같은 fold 출발 가중치·epoch·입력 크기·임계값으로 비교한다. 조합 모델은 매 epoch 원본 하나당 생성본을 1-2장만 뽑거나 원본 그룹 가중치를 일정하게 맞춘다. 원본 검증 세션에서 이물 1·2·3개별 재현율, 정상 오경보율, 호기별 미탐을 함께 본다. 합성 조합 영상의 지표는 보조 진단으로만 사용한다.", st["body"]),
              p("인페인팅이 남긴 질감 자체를 모델이 정답 단서로 사용할 수 있다. 따라서 미탐 감소가 원본의 독립 수동 라벨에서도 재현되지 않으면 배포용 데이터로 채택하지 않는다.", st["body"]),
              p("참고 연구: Ghiasi et al., Simple Copy-Paste Is a Strong Data Augmentation Method for Instance Segmentation, CVPR 2021; Winkler et al., Uncovering and Correcting Shortcut Learning in Machine Learning Models for Skin Cancer Diagnosis, 2022. 두 연구는 합성 데이터의 가능성과 인페인팅 단서의 위험을 이해하기 위한 참고이며, 본 X-ray 데이터에서의 효과를 증명하지 않는다.", st["small"])]

    story += [PageBreak(), p("6. 재현 및 파일 위치", st["h1"]),
              p("프로젝트 루트에서 Python 3.10 이상과 `processed/requirements_augmentation.txt`의 NumPy·OpenCV를 설치한다. 이미 정제 데이터와 세션별 매니페스트가 있는 상태라면 아래 명령 한 줄이 두 증강을 새 결과 디렉터리로 생성한다.", st["body"]),
              p("AUGMENT_PYTHON=.venv-augmentation/bin/python bash processed/run_augmentations.sh reproduction_01", st["code"]),
              p("구체적인 환경 생성 명령, 폴더 구조, 다른 코딩 에이전트에 바로 넣을 프롬프트는 `processed/RUN_AUGMENTATIONS.md`에 있다. 스크립트는 기존 출력이 있으면 덮어쓰기를 거부한다.", st["body"]),
              table([
                  ["파일", "용도"],
                  ["scripts/process_dataset.py", "색상 표시 추출·YOLO 약한 라벨·마스크·정제 영상"],
                  ["scripts/prepare_curated_split.py", "완전 중복 제거·세션 분할"],
                  ["scripts/augment_clean_dataset.py", "노이즈와 이물 진하기 조절 재현"],
                  ["scripts/augment_three_positions.py", "7개 존재·부재 조합 재현"],
                  ["processed/augmentation_noise_faint_v1/augmentation_manifest.csv", "영상별 강도·분할·중복·주의 표시"],
                  ["processed/three_position_combinations_v1/manifest.csv", "조합별 출처·fold·라벨·복원 진단"],
              ], [258, 235], st),
              Spacer(1, 16),
              p("최종 주의: 본 PDF는 데이터 생성·검증 기록이다. 새 증강의 모델 성능 보고서가 아니며, 사람 검수 정답과 연속 영상 평가도 아직 없다.", st["body"])]

    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=51, rightMargin=51,
                          topMargin=53, bottomMargin=60,
                          title="KAMP X-ray 데이터 전처리 및 증강 작업 기록", author="KAMP X-ray 프로젝트")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="normal", frames=[frame], onPage=footer))
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
