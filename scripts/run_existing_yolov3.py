#!/usr/bin/env python3
"""Run the repository's YOLOv3-SPP checkpoint on raw and cleaned images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    return image


def write_image(path: Path, image: np.ndarray, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"Could not encode {path}")
    encoded.tofile(path)


def letterbox_fixed(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = int(round(width * ratio)), int(round(height * ratio))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_width, pad_height = size - new_width, size - new_height
    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))
    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))
    output = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return output, ratio, (float(left), float(top))


def scale_boxes(boxes: torch.Tensor, ratio: float, pad: tuple[float, float], shape: tuple[int, int]) -> torch.Tensor:
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    height, width = shape
    boxes[:, [0, 2]].clamp_(0, width)
    boxes[:, [1, 3]].clamp_(0, height)
    return boxes


def load_ground_truth(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        boxes.append(
            [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ]
        )
    return boxes


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def matching_counts(predictions: list[list[float]], ground_truth: list[list[float]], mode: str) -> tuple[int, int, int]:
    matched: set[int] = set()
    true_positives = 0
    for prediction in sorted(predictions, key=lambda row: row[4], reverse=True):
        best_index = -1
        best_score = -1.0
        for index, target in enumerate(ground_truth):
            if index in matched:
                continue
            if mode == "center":
                center_x = (prediction[0] + prediction[2]) / 2
                center_y = (prediction[1] + prediction[3]) / 2
                score = 1.0 if target[0] <= center_x <= target[2] and target[1] <= center_y <= target[3] else 0.0
            else:
                score = box_iou(prediction[:4], target)
            if score > best_score:
                best_index, best_score = index, score
        threshold = 1.0 if mode == "center" else float(mode)
        if best_index >= 0 and best_score >= threshold:
            matched.add(best_index)
            true_positives += 1
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    return true_positives, false_positives, false_negatives


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_manifest(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if limit <= 0 or limit >= len(rows):
        return rows

    # Deterministic coverage across the full date/machine ordered manifest.
    indices = np.linspace(0, len(rows) - 1, limit, dtype=int)
    return [rows[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--yolo-dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", choices=("raw", "clean"), default=("raw", "clean"))
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--conf-thres", type=float, default=0.3)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--annotated-limit", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    processed = args.processed.resolve()
    yolo_dir = args.yolo_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))

    sys.path.insert(0, str(yolo_dir))
    from models import Darknet  # type: ignore
    from utils.utils import non_max_suppression  # type: ignore

    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")
    model = Darknet(str(args.cfg.resolve()), args.img_size)
    checkpoint = torch.load(args.weights.resolve(), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    manifest = load_manifest(processed / "processing_log.csv", args.limit)
    prediction_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    aggregate: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    confidence_values: dict[str, list[float]] = defaultdict(list)
    start_time = time.perf_counter()

    jobs: list[tuple[str, dict[str, str], Path]] = []
    for row in manifest:
        for variant in args.variants:
            path = Path(row["source_path"]) if variant == "raw" else processed / row["clean_image"]
            jobs.append((variant, row, path))

    annotated_counts: dict[str, int] = defaultdict(int)

    with torch.inference_mode():
        for batch_start in range(0, len(jobs), args.batch_size):
            batch_jobs = jobs[batch_start : batch_start + args.batch_size]
            tensors: list[np.ndarray] = []
            originals: list[np.ndarray] = []
            metadata: list[tuple[float, tuple[float, float]]] = []

            for _, _, image_path in batch_jobs:
                original = read_image(image_path)
                boxed, ratio, pad = letterbox_fixed(original, args.img_size)
                tensor = boxed[:, :, ::-1].transpose(2, 0, 1)
                tensors.append(np.ascontiguousarray(tensor, dtype=np.float32) / 255.0)
                originals.append(original)
                metadata.append((ratio, pad))

            inputs = torch.from_numpy(np.stack(tensors)).to(device)
            inference_start = time.perf_counter()
            raw_predictions = model(inputs)[0]
            detections = non_max_suppression(
                raw_predictions,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                multi_label=False,
            )
            batch_seconds = time.perf_counter() - inference_start

            for local_index, ((variant, manifest_row, image_path), original, detection) in enumerate(
                zip(batch_jobs, originals, detections)
            ):
                height, width = original.shape[:2]
                target_boxes = load_ground_truth(processed / manifest_row["label"], width, height)
                predictions: list[list[float]] = []

                if detection is not None and len(detection):
                    detection = detection.clone().cpu()
                    ratio, pad = metadata[local_index]
                    scale_boxes(detection[:, :4], ratio, pad, (height, width))
                    for prediction_index, item in enumerate(detection.tolist()):
                        x1, y1, x2, y2, confidence, class_id = item
                        predictions.append([x1, y1, x2, y2, confidence, class_id])
                        prediction_rows.append(
                            {
                                "variant": variant,
                                "output_id": manifest_row["output_id"],
                                "prediction_index": prediction_index,
                                "class_id": int(class_id),
                                "confidence": f"{confidence:.8f}",
                                "x1": f"{x1:.3f}",
                                "y1": f"{y1:.3f}",
                                "x2": f"{x2:.3f}",
                                "y2": f"{y2:.3f}",
                                "image_width": width,
                                "image_height": height,
                                "source_path": str(image_path),
                            }
                        )
                        confidence_values[variant].append(float(confidence))

                center_tp, center_fp, center_fn = matching_counts(predictions, target_boxes, "center")
                iou10_tp, iou10_fp, iou10_fn = matching_counts(predictions, target_boxes, "0.1")
                iou50_tp, iou50_fp, iou50_fn = matching_counts(predictions, target_boxes, "0.5")

                image_rows.append(
                    {
                        "variant": variant,
                        "output_id": manifest_row["output_id"],
                        "ground_truth_count": len(target_boxes),
                        "prediction_count": len(predictions),
                        "max_confidence": f"{max((p[4] for p in predictions), default=0.0):.8f}",
                        "center_tp": center_tp,
                        "center_fp": center_fp,
                        "center_fn": center_fn,
                        "iou10_tp": iou10_tp,
                        "iou10_fp": iou10_fp,
                        "iou10_fn": iou10_fn,
                        "iou50_tp": iou50_tp,
                        "iou50_fp": iou50_fp,
                        "iou50_fn": iou50_fn,
                        "source_path": str(image_path),
                    }
                )

                metrics = aggregate[variant]
                metrics["images"] += 1
                metrics["images_with_predictions"] += int(bool(predictions))
                metrics["ground_truth"] += len(target_boxes)
                metrics["predictions"] += len(predictions)
                for name, value in (
                    ("center_tp", center_tp), ("center_fp", center_fp), ("center_fn", center_fn),
                    ("iou10_tp", iou10_tp), ("iou10_fp", iou10_fp), ("iou10_fn", iou10_fn),
                    ("iou50_tp", iou50_tp), ("iou50_fp", iou50_fp), ("iou50_fn", iou50_fn),
                ):
                    metrics[name] += value

                if annotated_counts[variant] < args.annotated_limit:
                    annotated = original.copy()
                    for x1, y1, x2, y2, confidence, _ in predictions:
                        cv2.rectangle(annotated, (round(x1), round(y1)), (round(x2), round(y2)), (0, 0, 255), 1)
                        cv2.putText(
                            annotated,
                            f"{confidence:.2f}",
                            (round(x1), max(10, round(y1) - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    for x1, y1, x2, y2 in target_boxes:
                        cv2.rectangle(annotated, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 1)
                    write_image(output / "annotated" / variant / f"{manifest_row['output_id']}.jpg", annotated)
                    annotated_counts[variant] += 1

            completed = min(batch_start + len(batch_jobs), len(jobs))
            if completed % max(args.batch_size * 10, 1) == 0 or completed == len(jobs):
                print(f"Inference {completed}/{len(jobs)} ({batch_seconds:.2f}s last batch)")

    prediction_fields = [
        "variant", "output_id", "prediction_index", "class_id", "confidence",
        "x1", "y1", "x2", "y2", "image_width", "image_height", "source_path",
    ]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(prediction_rows)

    image_fields = [
        "variant", "output_id", "ground_truth_count", "prediction_count", "max_confidence",
        "center_tp", "center_fp", "center_fn", "iou10_tp", "iou10_fp", "iou10_fn",
        "iou50_tp", "iou50_fp", "iou50_fn", "source_path",
    ]
    with (output / "image_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=image_fields)
        writer.writeheader()
        writer.writerows(image_rows)

    summary: dict[str, object] = {
        "checkpoint": str(args.weights.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "cfg": str(args.cfg.resolve()),
        "parameters": {
            "img_size": args.img_size,
            "conf_thres": args.conf_thres,
            "iou_thres": args.iou_thres,
            "batch_size": args.batch_size,
            "images_selected": len(manifest),
            "variants": list(args.variants),
        },
        "elapsed_seconds": time.perf_counter() - start_time,
        "variants": {},
        "metric_note": "Labels are weak boxes extracted from color overlays; center-match is more meaningful than IoU for this audit.",
    }

    for variant in args.variants:
        metrics = aggregate[variant]
        variant_summary: dict[str, object] = {key: int(value) for key, value in metrics.items()}
        values = confidence_values[variant]
        variant_summary["confidence_mean"] = statistics.fmean(values) if values else 0.0
        variant_summary["confidence_median"] = statistics.median(values) if values else 0.0
        variant_summary["image_detection_rate"] = safe_ratio(
            metrics["images_with_predictions"], metrics["images"]
        )
        for prefix in ("center", "iou10", "iou50"):
            tp, fp, fn = metrics[f"{prefix}_tp"], metrics[f"{prefix}_fp"], metrics[f"{prefix}_fn"]
            variant_summary[f"{prefix}_precision"] = safe_ratio(tp, tp + fp)
            variant_summary[f"{prefix}_recall"] = safe_ratio(tp, tp + fn)
        summary["variants"][variant] = variant_summary  # type: ignore[index]

    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
