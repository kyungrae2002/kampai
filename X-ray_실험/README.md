# X-ray 이물 탐지 — 모델 비교 실험

## 실행 (맥)
터미널에서 한 줄:  `bash ~/Desktop/kamp_ai/X-ray_실험/run_all.sh`
- 설치 → 스모크 테스트(모델별 1epoch 소량) → 본 학습 → 채점 순서. 스모크가 하나라도 실패하면 본 학습을 시작하지 않음.
- 중간에 멈추면 같은 명령을 다시 실행: `logs/*.done`이 있는 단계는 건너뜀.
- 실행 중 잠자기 방지(caffeinate). 전원 연결 필요. 로그: `logs/run_all.log`, 단계별 `logs/<단계>.log`.

## 실행 (Colab — Faster R-CNN)
`colab/xray_colab_data.zip`을 Drive `MyDrive/kamp_colab/`에 올리고 `colab/faster_rcnn_colab.ipynb`를 GPU 런타임에서 실행.
결과 `faster_rcnn.csv / _meta.json / _eval.json`을 `preds/`에 넣기.

## 공통 조건
- 학습: `X-ray_데이터셋_통합/학습/` 6개 폴더 (5개 방법 2,492장 + 6_약라벨_추가영상 1,911장 = 4,403장)
- 검증(80): 모델·임계값 선택 / 테스트(80): 최종 1회 평가
- 정상 합성 평가셋(`eval_sets/`, 각 80장): 검증·테스트 영상의 이물을 지운 영상 — 영상 단위 오경보율 측정 전용, 학습 금지
- 채점: `scripts/evaluate.py` (AP50, AP50-95, val에서 F1 최대 임계값을 test에 고정 적용, 미끼 박스 오탐, 정상 오경보율, 영상 AUROC, 테스트 부트스트랩 95% CI)

## 6_약라벨_추가영상 만든 방법 (`scripts/build_extra.py`, `xproc.py`)
- 대상: `split_report.csv`의 train_pseudo 2,016장 중 색 박스가 있는 1,911장 (색 박스 없는 노란 선 영상 105장 제외, `excluded_no_box.txt`)
- 새 분할의 학습 세션에만 속함 → 검증·테스트와 세션 겹침 없음
- 가공: 통합본과 같은 방식 (색 박스 Telea 복원 + 같은 모양 미끼 박스 + 복원 부위 노이즈 재주입, 노이즈 배율은 통합본 질감 비율에 맞춰 0.6)
- 라벨: 색 박스 외곽에서 가로·세로 8px 축소 (사람 라벨 대비 색 박스가 중앙값 8px 크고 중심 오차 1px)
