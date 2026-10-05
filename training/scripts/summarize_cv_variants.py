#!/usr/bin/env python3
"""Aggregate matched five-fold KAMP counterfactual experiments."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


PROJECT = Path("/Users/kyungrae/Desktop/kamp_ai")
ROOT = PROJECT / "training" / "experiments" / "point_decoy_roi_5fold"
VARIANTS = ("control", "point", "decoy", "roi")


def load(fold: int, variant: str) -> dict:
    if fold == 1:
        path = PROJECT / "training" / "experiments" / "point_decoy_roi" / "metrics" / f"{variant}.json"
    else:
        path = ROOT / f"fold_{fold}" / "metrics" / f"{variant}.json"
    return json.loads(path.read_text())


def mean_sd(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values),
            "min": min(values), "max": max(values)}


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main() -> None:
    rows = []
    raw = {}
    for fold in range(1, 6):
        raw[fold] = {}
        for variant in VARIANTS:
            data = load(fold, variant)
            raw[fold][variant] = data
            views = data["views"]
            original = views["original"]["center"]
            decoy = views["decoy"]["center"]
            point = views["point"]
            rows.append({"fold": fold, "variant": variant,
                         "images": original["images"], "original_tp": original["tp"],
                         "original_fp": original["fp"], "original_fn": original["fn"],
                         "original_f1": original["f1"],
                         "original_ap50_iou": views["original"]["ap50_iou"],
                         "original_negative_specificity": original["negative_specificity"],
                         "point_found": point["point_found"],
                         "point_recall": point["point_recall"],
                         "decoy_hits": views["decoy"]["decoy_hits"],
                         "decoy_hit_rate": views["decoy"]["decoy_hit_rate"],
                         "decoy_f1": decoy["f1"],
                         "roi_f1": views["roi"]["center"]["f1"]})
    summary = {}
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        tp = sum(r["original_tp"] for r in selected)
        fp = sum(r["original_fp"] for r in selected)
        fn = sum(r["original_fn"] for r in selected)
        machine = {}
        for m in ("1", "2", "3"):
            counts = [raw[fold][variant]["views"]["original"]["per_machine"][m]
                      for fold in range(1, 6)]
            mtp, mfp, mfn = (sum(c[k] for c in counts) for k in ("tp", "fp", "fn"))
            machine[m] = {"tp": mtp, "fp": mfp, "fn": mfn,
                          "recall": mtp / max(mtp + mfn, 1), "f1": f1(mtp, mfp, mfn)}
        summary[variant] = {
            "original_f1": mean_sd([r["original_f1"] for r in selected]),
            "original_ap50_iou": mean_sd([r["original_ap50_iou"] for r in selected]),
            "original_micro": {"tp": tp, "fp": fp, "fn": fn, "f1": f1(tp, fp, fn)},
            "point_recall": mean_sd([r["point_recall"] for r in selected]),
            "decoy_hit_rate": mean_sd([r["decoy_hit_rate"] for r in selected]),
            "decoy_hits_total": sum(r["decoy_hits"] for r in selected),
            "decoy_f1": mean_sd([r["decoy_f1"] for r in selected]),
            "roi_f1": mean_sd([r["roi_f1"] for r in selected]),
            "per_machine_original": machine,
        }
    paired = {}
    for variant in ("point", "decoy", "roi"):
        paired[variant] = {}
        for key in ("original_f1", "point_recall", "decoy_hit_rate", "decoy_f1", "roi_f1"):
            diffs = [next(r[key] for r in rows if r["fold"] == fold and r["variant"] == variant) -
                     next(r[key] for r in rows if r["fold"] == fold and r["variant"] == "control")
                     for fold in range(1, 6)]
            paired[variant][key] = {"fold_differences": diffs, **mean_sd(diffs),
                                    "positive_folds": sum(x > 0 for x in diffs),
                                    "negative_folds": sum(x < 0 for x in diffs)}
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "fold_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "summary.json").write_text(json.dumps({"summary": summary, "paired": paired}, indent=2))
    fmt = lambda d: f"{d['mean']:.4f} ± {d['sd']:.4f}"
    lines = ["# KAMP 점·미끼·ROI 5-fold 비교", "",
             "동일한 세션 분할·fold별 출발 YOLOv3-SPP 가중치에서 각 모델을 2 epoch 추가 학습했다. ",
             "각 fold는 원본 학습 영상과 동일한 600장 추가 예시를 사용했고, confidence 0.07을 고정했다.",
             "기존 원본 박스는 자동 추출된 약한 라벨이며, 합성 점만 생성 좌표가 정확한 정답이다.", "",
             "## 전체 결과", "",
             "| 모델 | 원본 중심점 F1 | 원본 AP@0.5 IoU | 합성 점 발견율 | 미끼 위치 검출률 | 미끼 영상 F1 | Oracle ROI F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        item = summary[variant]
        lines.append(f"| {variant} | {fmt(item['original_f1'])} | {fmt(item['original_ap50_iou'])} | "
                     f"{fmt(item['point_recall'])} | {fmt(item['decoy_hit_rate'])} | "
                     f"{fmt(item['decoy_f1'])} | {fmt(item['roi_f1'])} |")
    lines += ["", "원본 F1은 fold별 값의 평균±표본표준편차다. 미끼 위치 검출률은 미끼 영상 중 그 박스 위치에 예측 중심이 들어간 영상의 비율이며 **낮을수록 좋다**. ROI 검증은 정답 위치를 알고 자른 상한선 실험이다.", "",
              "## Fold별 원본 F1 / 미끼 위치 검출 건수", "",
              "| Fold | 검증 장수 | 대조군 | 합성 점 | 미끼 | ROI | 대조군 미끼 건수 | 미끼 학습 미끼 건수 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for fold in range(1, 6):
        r = {variant: next(x for x in rows if x["fold"] == fold and x["variant"] == variant)
             for variant in VARIANTS}
        lines.append(f"| {fold} | {r['control']['images']} | {r['control']['original_f1']:.4f} | "
                     f"{r['point']['original_f1']:.4f} | {r['decoy']['original_f1']:.4f} | "
                     f"{r['roi']['original_f1']:.4f} | {r['control']['decoy_hits']} | {r['decoy']['decoy_hits']} |")
    lines += ["", "## 대조군 대비 fold별 변화", ""]
    for variant in ("point", "decoy", "roi"):
        original = paired[variant]["original_f1"]
        hits = paired[variant]["decoy_hit_rate"]
        lines.append(f"- {variant}: 원본 F1 평균 차이 {original['mean']:+.4f} "
                     f"({original['positive_folds']}/5 folds 개선), 미끼 위치 검출률 차이 {hits['mean']:+.4f}.")
    lines += ["", "## 주의", "",
              "이 5-fold 결과는 추가 학습 데이터 구성의 비교다. 기존 5-fold 기반 모델에서 미세조정했으므로 Darknet-53부터 20개 모델을 독립 재학습한 결과는 아니다. 원본의 실제 이물 수동 정답, 2·3호기 정상 영상, 연속 영상 평가가 없어 현장 성능을 확정할 수 없다."]
    (ROOT / "CV_5FOLD_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(rows), "report": str(ROOT / "CV_5FOLD_REPORT.md"),
                      "summary": {k: {"original_f1": v["original_f1"],
                                      "decoy_hit_rate": v["decoy_hit_rate"]}
                                  for k, v in summary.items()}}, indent=2))


if __name__ == "__main__":
    main()
