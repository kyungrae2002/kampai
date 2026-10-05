#!/usr/bin/env python3
"""Evaluate YOLOv3 checkpoints with IoU and weak-label center matching."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

np.int = int  # type: ignore[attr-defined]
_numpy_save = np.save


def _legacy_compatible_numpy_save(file, array, *args, **kwargs):
    if isinstance(array, list):
        array = np.asarray(array, dtype=object)
    return _numpy_save(file, array, *args, **kwargs)


np.save = _legacy_compatible_numpy_save  # type: ignore[assignment]

import torch
from torch.utils.data import DataLoader


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    out = boxes.clone()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not len(a) or not len(b):
        return torch.zeros((len(a), len(b)))
    inter_x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_flags(pred: torch.Tensor, gt: torch.Tensor, mode: str) -> torch.Tensor:
    flags = torch.zeros(len(pred), dtype=torch.bool)
    if not len(pred) or not len(gt):
        return flags
    used: set[int] = set()
    order = pred[:, 4].argsort(descending=True)
    ious = pairwise_iou(pred[:, :4], gt)
    for pi in order.tolist():
        available = [j for j in range(len(gt)) if j not in used]
        if not available:
            break
        if mode == "iou":
            values = ious[pi, available]
            best_pos = int(values.argmax())
            match = available[best_pos]
            valid = float(values[best_pos]) >= 0.5
        else:
            cx = float((pred[pi, 0] + pred[pi, 2]) / 2)
            cy = float((pred[pi, 1] + pred[pi, 3]) / 2)
            candidates = [j for j in available if float(gt[j, 0]) <= cx <= float(gt[j, 2]) and float(gt[j, 1]) <= cy <= float(gt[j, 3])]
            if candidates:
                match = min(candidates, key=lambda j: (cx - float((gt[j, 0] + gt[j, 2]) / 2)) ** 2 + (cy - float((gt[j, 1] + gt[j, 3]) / 2)) ** 2)
                valid = True
            else:
                match, valid = -1, False
        if valid:
            flags[pi] = True
            used.add(match)
    return flags


def average_precision(scores: np.ndarray, tp: np.ndarray, num_gt: int) -> float:
    if num_gt == 0 or scores.size == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = tp[order].astype(float)
    fp = 1.0 - tp
    recall = np.cumsum(tp) / num_gt
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(grid, recall, precision), grid))


def clear_legacy_cache(list_path: Path) -> None:
    lines = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return
    image = Path(lines[0])
    parts = list(image.parts)
    if "images" not in parts:
        return
    parts[parts.index("images")] = "labels"
    cache = Path(str(Path(*parts).with_suffix(".txt").parent) + ".npy")
    if cache.exists():
        cache.unlink()


def threshold_metrics(records: list[dict], threshold: float, mode: str, machine: str | None = None) -> dict:
    selected = [r for r in records if machine is None or r["machine"] == machine]
    tp = fp = fn = 0
    positive_images = detected_positive_images = 0
    negative_images = clean_negative_images = 0
    for record in selected:
        pred = record["pred"]
        keep = pred[:, 4] >= threshold if len(pred) else torch.zeros(0, dtype=torch.bool)
        pred = pred[keep]
        flags = greedy_flags(pred, record["gt"], mode)
        image_tp = int(flags.sum())
        tp += image_tp
        fp += len(pred) - image_tp
        fn += len(record["gt"]) - image_tp
        if len(record["gt"]):
            positive_images += 1
            detected_positive_images += image_tp > 0
        else:
            negative_images += 1
            clean_negative_images += len(pred) == 0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "positive_image_recall": detected_positive_images / max(positive_images, 1),
        "negative_specificity": clean_negative_images / max(negative_images, 1),
        "images": len(selected),
        "positive_images": positive_images,
        "negative_images": negative_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    sys.path.insert(0, str(args.repo))
    from models import Darknet
    from utils.datasets import LoadImagesAndLabels
    from utils.utils import non_max_suppression

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    device = torch.device(args.device)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = Darknet(str(args.cfg), args.img_size)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    clear_legacy_cache(args.data_list)
    dataset = LoadImagesAndLabels(str(args.data_list), args.img_size, args.batch_size, rect=True, single_cls=True, pad=0.5)
    clear_legacy_cache(args.data_list)
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False, num_workers=0, pin_memory=False, collate_fn=dataset.collate_fn)
    records = []
    with torch.no_grad():
        for images, targets, paths, _ in loader:
            images = images.to(device).float() / 255.0
            inference, _ = model(images)
            # torchvision NMS is more reliable on CPU for this legacy implementation.
            outputs = non_max_suppression(inference.cpu(), conf_thres=0.01, iou_thres=0.5, multi_label=False)
            height, width = images.shape[2:]
            for idx, path in enumerate(paths):
                rows = targets[targets[:, 0] == idx, 2:6].clone()
                if len(rows):
                    rows *= torch.tensor([width, height, width, height])
                    gt = xywh_to_xyxy(rows)
                else:
                    gt = torch.zeros((0, 4))
                pred = outputs[idx] if outputs[idx] is not None else torch.zeros((0, 6))
                name = Path(path).name
                machine = name[1] if name.startswith("m") and len(name) > 1 else "unknown"
                records.append({"path": path, "machine": machine, "gt": gt.cpu(), "pred": pred.cpu()})

    scores = np.concatenate([r["pred"][:, 4].numpy() for r in records if len(r["pred"])]) if any(len(r["pred"]) for r in records) else np.array([])
    num_gt = sum(len(r["gt"]) for r in records)
    iou_tp = np.concatenate([greedy_flags(r["pred"], r["gt"], "iou").numpy() for r in records if len(r["pred"])]) if scores.size else np.array([])
    center_tp = np.concatenate([greedy_flags(r["pred"], r["gt"], "center").numpy() for r in records if len(r["pred"])]) if scores.size else np.array([])
    thresholds = (
        [round(float(x), 3) for x in np.arange(0.01, 0.101, 0.005)]
        + [round(float(x), 2) for x in np.arange(0.15, 0.96, 0.05)]
    )
    center_table = [threshold_metrics(records, t, "center") for t in thresholds]
    iou_table = [threshold_metrics(records, t, "iou") for t in thresholds]
    selected_threshold = args.threshold if args.threshold is not None else max(center_table, key=lambda x: (x["f1"], x["negative_specificity"]))["threshold"]
    result = {
        "weights": str(args.weights.resolve()),
        "data_list": str(args.data_list.resolve()),
        "device": str(device),
        "images": len(records),
        "ground_truth_boxes": num_gt,
        "detections_at_0.01": int(scores.size),
        "ap50_iou": average_precision(scores, iou_tp, num_gt),
        "ap_center": average_precision(scores, center_tp, num_gt),
        "selected_threshold": selected_threshold,
        "selected": {
            "iou": threshold_metrics(records, selected_threshold, "iou"),
            "center": threshold_metrics(records, selected_threshold, "center"),
        },
        "per_machine": {
            machine: {
                "iou": threshold_metrics(records, selected_threshold, "iou", machine),
                "center": threshold_metrics(records, selected_threshold, "center", machine),
            }
            for machine in sorted({r["machine"] for r in records})
        },
        "threshold_table": {"iou": iou_table, "center": center_table},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
