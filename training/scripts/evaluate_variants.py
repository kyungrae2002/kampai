#!/usr/bin/env python3
"""Evaluate original and counterfactual KAMP validation views at a fixed threshold."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--threshold", type=float, default=0.07)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--decoy-list", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project
    sys.path.insert(0, str(project / "training" / "scripts"))
    from evaluate_yolov3 import (average_precision, clear_legacy_cache, greedy_flags,
                                 threshold_metrics, xywh_to_xyxy)
    repo = project / "dataset" / "dataset" / "test1" / "yolov3"
    sys.path.insert(0, str(repo))
    from models import Darknet
    from utils.datasets import LoadImagesAndLabels
    from utils.utils import non_max_suppression

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Darknet(str(repo / "yolov3-spp.cfg"), 320)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    root = args.experiment_root or (project / "training" / "experiments" / "point_decoy_roi")
    details = {}
    with (root / "variant_manifest.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            details[row["image"]] = row
    result = {"model": args.model_name, "weights": str(args.weights),
              "threshold": args.threshold, "split": args.split,
              "decoy_list": str(args.decoy_list) if args.decoy_list else None,
              "device": str(device), "views": {}}
    for view in ("original", "point", "decoy", "roi"):
        if args.split == "test":
            list_path = (project / "training" / "cv" / "test.txt") if view == "original" else (root / "lists" / view / "test_variant.txt")
        else:
            list_name = "val_original.txt" if view == "original" else "val_variant.txt"
            list_variant = "control" if view == "original" else view
            list_path = root / "lists" / list_variant / list_name
        if view == "decoy" and args.decoy_list is not None:
            list_path = args.decoy_list
        clear_legacy_cache(list_path)
        dataset = LoadImagesAndLabels(str(list_path), 320, 8, rect=True, single_cls=True, pad=0.5)
        clear_legacy_cache(list_path)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0,
                            pin_memory=False, collate_fn=dataset.collate_fn)
        records = []
        point_found = decoy_hit = 0
        with torch.no_grad():
            for images, targets, paths, shapes in loader:
                images = images.to(device).float() / 255.0
                inference, _ = model(images)
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
                    pred = pred.cpu()
                    selected = pred[pred[:, 4] >= args.threshold]
                    centers = (selected[:, :2] + selected[:, 2:4]) / 2
                    if view == "point":
                        mark = gt[-1]
                        point_found += int(bool(((centers[:, 0] >= mark[0]) &
                                                 (centers[:, 0] <= mark[2]) &
                                                 (centers[:, 1] >= mark[1]) &
                                                 (centers[:, 1] <= mark[3])).any()))
                    if view == "decoy":
                        x1, y1, x2, y2, _ = json.loads(details[path]["detail"])
                        _, ((rw, rh), (padx, pady)) = shapes[idx]
                        decoy = (x1*rw+padx, y1*rh+pady, x2*rw+padx, y2*rh+pady)
                        decoy_hit += int(bool(((centers[:, 0] >= decoy[0]) &
                                               (centers[:, 0] <= decoy[2]) &
                                               (centers[:, 1] >= decoy[1]) &
                                               (centers[:, 1] <= decoy[3])).any()))
                    machine = Path(path).name[1]
                    records.append({"path": path, "machine": machine, "gt": gt.cpu(), "pred": pred})
        scores = np.concatenate([r["pred"][:, 4].numpy() for r in records if len(r["pred"])]) if any(len(r["pred"]) for r in records) else np.array([])
        num_gt = sum(len(r["gt"]) for r in records)
        iou_tp = np.concatenate([greedy_flags(r["pred"], r["gt"], "iou").numpy() for r in records if len(r["pred"])]) if scores.size else np.array([])
        center_tp = np.concatenate([greedy_flags(r["pred"], r["gt"], "center").numpy() for r in records if len(r["pred"])]) if scores.size else np.array([])
        result["views"][view] = {
            "center": threshold_metrics(records, args.threshold, "center"),
            "iou": threshold_metrics(records, args.threshold, "iou"),
            "ap50_iou": average_precision(scores, iou_tp, num_gt),
            "ap_center": average_precision(scores, center_tp, num_gt),
            "point_recall": point_found / len(records) if view == "point" else None,
            "point_found": point_found if view == "point" else None,
            "decoy_hit_rate": decoy_hit / len(records) if view == "decoy" else None,
            "decoy_hits": decoy_hit if view == "decoy" else None,
            "per_machine": {m: threshold_metrics(records, args.threshold, "center", m)
                            for m in sorted({r["machine"] for r in records})},
        }
        print(json.dumps({"model": args.model_name, "view": view,
                          "center": result["views"][view]["center"],
                          "point_found": result["views"][view]["point_found"],
                          "decoy_hits": result["views"][view]["decoy_hits"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
