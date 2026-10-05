#!/usr/bin/env python3
"""Deterministic, session-safe diagnostic datasets for KAMP YOLOv3."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(image: Path) -> list[list[float]]:
    path = label_path(image)
    return [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]


def save_labels(path: Path, labels: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("0 " + " ".join(f"{v:.8f}" for v in row[1:]) + "\n" for row in labels))


def rects_px(labels: list[list[float]], width: int, height: int) -> list[tuple[int, int, int, int]]:
    out = []
    for _, cx, cy, bw, bh in labels:
        x1 = int((cx - bw / 2) * width)
        y1 = int((cy - bh / 2) * height)
        x2 = int((cx + bw / 2) * width)
        y2 = int((cy + bh / 2) * height)
        out.append((x1, y1, x2, y2))
    return out


def find_location(gray: np.ndarray, boxes: list[tuple[int, int, int, int]], rng: random.Random,
                  margin: int = 18) -> tuple[int, int]:
    h, w = gray.shape
    # Only place synthetic marks on dark product pixels, away from annotated defects.
    candidate = np.argwhere((gray > 65) & (gray < 180))
    candidate = candidate[(candidate[:, 0] > margin) & (candidate[:, 0] < h - margin) &
                          (candidate[:, 1] > margin) & (candidate[:, 1] < w - margin)]
    if not len(candidate):
        candidate = np.argwhere(np.ones_like(gray[margin:h-margin, margin:w-margin], dtype=bool)) + margin
    for _ in range(min(300, len(candidate))):
        y, x = map(int, candidate[rng.randrange(len(candidate))])
        if all(not (x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin)
               for x1, y1, x2, y2 in boxes):
            return x, y
    raise ValueError("No nonoverlapping product location for synthetic mark")


def synthetic_point(image: np.ndarray, labels: list[list[float]], rng: random.Random):
    h, w = image.shape[:2]
    boxes = rects_px(labels, w, h)
    x, y = find_location(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), boxes, rng)
    radius = rng.choice((3, 4, 5))
    yy, xx = np.mgrid[-radius-1:radius+2, -radius-1:radius+2]
    d = np.sqrt(xx * xx + yy * yy)
    alpha = np.clip((radius + 0.5 - d) / 1.5, 0, 1) * np.clip(1 - (d / (radius + 1)) ** 2, 0, 1)
    roi = image[y-radius-1:y+radius+2, x-radius-1:x+radius+2].astype(np.float32)
    dark = (40 + rng.randint(-8, 8)) * alpha[..., None]
    image[y-radius-1:y+radius+2, x-radius-1:x+radius+2] = np.clip(roi - dark, 0, 255).astype(np.uint8)
    side = 2 * radius + 8
    return image, labels + [[0.0, x / w, y / h, side / w, side / h]], (x, y, side)


def decoy_box(image: np.ndarray, labels: list[list[float]], rng: random.Random):
    h, w = image.shape[:2]
    boxes = rects_px(labels, w, h)
    x, y = find_location(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), boxes, rng, margin=24)
    sizes = [(max(12, x2-x1), max(12, y2-y1)) for x1, y1, x2, y2 in boxes]
    bw, bh = rng.choice(sizes) if sizes else (rng.randint(14, 22), rng.randint(14, 22))
    bw, bh = min(bw, 34), min(bh, 34)
    x1, y1, x2, y2 = x-bw//2, y-bh//2, x+bw//2, y+bh//2
    # Original images carry red and yellow rectangular overlays; these are non-target decoys.
    color = rng.choice(((0, 0, 255), (0, 255, 255)))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    return image, labels, (x1, y1, x2, y2, color)


def roi_crop(image: np.ndarray, labels: list[list[float]], rng: random.Random, train: bool):
    h, w = image.shape[:2]
    size = min(160, h, w)
    if labels:
        target = rng.choice(labels) if train else labels[0]
        cx, cy = target[1] * w, target[2] * h
    else:
        cx, cy = find_location(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), [], rng)
    jitter = 12 if train else 0
    left = int(np.clip(round(cx - size / 2 + rng.randint(-jitter, jitter)), 0, w-size))
    top = int(np.clip(round(cy - size / 2 + rng.randint(-jitter, jitter)), 0, h-size))
    crop = image[top:top+size, left:left+size].copy()
    crop_labels = []
    for _, gx, gy, gw, gh in labels:
        px, py = gx*w, gy*h
        if not (left <= px < left+size and top <= py < top+size):
            continue
        bx1, bx2 = max(left, (gx-gw/2)*w), min(left+size, (gx+gw/2)*w)
        by1, by2 = max(top, (gy-gh/2)*h), min(top+size, (gy+gh/2)*h)
        if bx2 <= bx1 or by2 <= by1:
            continue
        crop_labels.append([0.0, ((bx1+bx2)/2-left)/size, ((by1+by2)/2-top)/size,
                            (bx2-bx1)/size, (by2-by1)/size])
    return crop, crop_labels, (left, top, size)


def process(source: Path, out: Path, variant: str, seed: int, train: bool) -> dict:
    rng = random.Random(seed)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {source}")
    labels = read_labels(source)
    if variant == "control":
        result, result_labels, detail = image, labels, None
    elif variant == "point":
        result, result_labels, detail = synthetic_point(image, labels, rng)
    elif variant == "decoy":
        result, result_labels, detail = decoy_box(image, labels, rng)
    elif variant == "roi":
        result, result_labels, detail = roi_crop(image, labels, rng, train)
    else:
        raise ValueError(variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    if variant == "control":
        os.link(source, out)
    elif not cv2.imwrite(str(out), result):
        raise ValueError(f"Cannot save {out}")
    dest_label = label_path(out)
    if variant == "control":
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        os.link(label_path(source), dest_label)
    else:
        save_labels(dest_label, result_labels)
    return {"variant": variant, "source": str(source), "image": str(out),
            "source_labels": len(labels), "result_labels": len(result_labels),
            "detail": json.dumps(detail)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--augment-count", type=int, default=600)
    p.add_argument("--seed", type=int, default=20261001)
    a = p.parse_args()
    root = a.project / "training" / "experiments" / "point_decoy_roi"
    source_fold = a.project / "training" / "cv" / f"fold_{a.fold}"
    train = [Path(s) for s in (source_fold / "train.txt").read_text().splitlines() if s.strip()]
    val = [Path(s) for s in (source_fold / "val.txt").read_text().splitlines() if s.strip()]
    selected = random.Random(a.seed).sample(train, min(a.augment_count, len(train)))
    manifest = []
    for variant in ("control", "point", "decoy", "roi"):
        train_list = list(train)
        val_list = []
        for split, sources in (("train", selected), ("val", val)):
            for idx, source in enumerate(sources):
                dest = root / "images" / variant / split / source.name
                try:
                    row = process(source, dest, variant, a.seed + idx + (100000 if split == "val" else 0), split == "train")
                except ValueError as exc:
                    manifest.append({"variant": variant, "source": str(source), "image": "",
                                     "source_labels": "", "result_labels": "", "detail": f"SKIP: {exc}"})
                    continue
                row["split"] = split
                manifest.append(row)
                (train_list if split == "train" else val_list).append(dest)
        list_dir = root / "lists" / variant
        list_dir.mkdir(parents=True, exist_ok=True)
        (list_dir / "train.txt").write_text("\n".join(map(str, train_list)) + "\n")
        (list_dir / "val_variant.txt").write_text("\n".join(map(str, val_list)) + "\n")
        (list_dir / "val_original.txt").write_text("\n".join(map(str, val)) + "\n")
    manifest_path = root / "variant_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("variant", "split", "source", "image", "source_labels", "result_labels", "detail"))
        writer.writeheader()
        writer.writerows(manifest)
    print(json.dumps({"root": str(root), "source_train": len(train), "source_val": len(val),
                      "augment_count": len(selected), "records": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
