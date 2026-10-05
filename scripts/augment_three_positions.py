#!/usr/bin/env python3
"""Create x/o presence combinations from clean X-ray frames with three weak boxes.

`o` retains the originally visible defect; `x` removes one detected dark core.
Only unique development images are eligible. This is experimental synthetic data:
the removed region and weak label must be quality-checked before model training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


PATTERNS = ("xoo", "oxo", "oox", "xxo", "xox", "oxx", "xxx")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def image_read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def image_write(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    buf.tofile(path)


def parse_boxes(path: Path, width: int, height: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = map(float, line.split())
        if cls != 0:
            raise ValueError(f"Unexpected class in {path}")
        x1 = max(0, min(width - 1, round((cx - bw / 2) * width)))
        y1 = max(0, min(height - 1, round((cy - bh / 2) * height)))
        x2 = max(x1 + 1, min(width, round((cx + bw / 2) * width)))
        y2 = max(y1 + 1, min(height, round((cy + bh / 2) * height)))
        boxes.append((line, (x1, y1, x2, y2)))
    return sorted(boxes, key=lambda pair: (pair[1][1] + pair[1][3]) / 2)


def removal_delta(image: np.ndarray, box: tuple[int, int, int, int]):
    """Return a local repair patch, its location, and quality diagnostics."""
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad = 13
    rx1, ry1, rx2, ry2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    patch = image[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # The three real dark dots sit near the centers of their weak boxes, while
    # the darker left edge of each stripe can dominate a broad closing mask.
    # A local median residual plus center support separates the dot from edge.
    baseline = cv2.medianBlur(gray, 9)
    deficit = baseline.astype(np.int16) - gray.astype(np.int16)
    core = np.zeros(gray.shape, dtype=np.uint8)
    center_x, center_y = (x1 + x2) / 2 - rx1, (y1 + y2) / 2 - ry1
    yy, xx = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    support = ((xx - center_x) / max(4.0, bw * 0.34)) ** 2 + ((yy - center_y) / max(4.0, bh * 0.34)) ** 2 <= 1
    support[:y1 - ry1, :] = False
    support[y2 - ry1:, :] = False
    support[:, :x1 - rx1] = False
    support[:, x2 - rx1:] = False
    core[(deficit >= 9) & support] = 1
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(core, connectivity=8)
    if count <= 1:
        raise ValueError("no_dark_core")
    candidates = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 2:
            continue
        cx, cy = centroids[component]
        distance = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
        mean_deficit = float(deficit[labels == component].mean())
        candidates.append((mean_deficit / (distance + 2), component))
    if not candidates:
        raise ValueError("no_component")
    chosen = max(candidates)[1]
    target = (labels == chosen).astype(np.uint8) * 255
    target = cv2.dilate(target, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    area = int(np.count_nonzero(target))
    if area > min(130, int(bw * bh * 0.55)):
        raise ValueError(f"mask_too_large:{area}")
    repaired = cv2.inpaint(patch, target, 3, cv2.INPAINT_TELEA)
    # A narrow feather blends the repair into the natural stripe texture.
    alpha = cv2.GaussianBlur(target.astype(np.float32) / 255.0, (0, 0), 0.75)
    output = np.clip(patch.astype(np.float32) * (1 - alpha[..., None]) +
                     repaired.astype(np.float32) * alpha[..., None], 0, 255).astype(np.uint8)
    delta = output.astype(np.int16) - patch.astype(np.int16)
    changed = int(np.count_nonzero(np.any(delta != 0, axis=2)))
    if changed < 2:
        raise ValueError("negligible_repair")
    before = float(np.mean(deficit[target > 0]))
    after = float(np.mean(np.maximum(baseline.astype(np.int16) - cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.int16), 0)[target > 0]))
    if after >= before * 0.8:
        raise ValueError(f"dark_core_persists:{after:.1f}/{before:.1f}")
    return (rx1, ry1, rx2, ry2), delta, target, {"mask_pixels": area, "changed_pixels": changed,
                                                   "dark_deficit_before": round(before, 2),
                                                   "dark_deficit_after": round(after, 2)}


def compose(image: np.ndarray, repairs: list[tuple], pattern: str) -> np.ndarray:
    out = image.astype(np.int16).copy()
    for bit, (extent, delta, _, _) in zip(pattern, repairs, strict=True):
        if bit == "x":
            x1, y1, x2, y2 = extent
            out[y1:y2, x1:x2] += delta
    return np.clip(out, 0, 255).astype(np.uint8)


def preview(image: np.ndarray, boxes: list[tuple], variants: dict[str, np.ndarray]) -> np.ndarray:
    frames = [("ooo (source)", image)] + [(pattern, variants[pattern]) for pattern in PATTERNS]
    tile_w, tile_h = 430, 325
    canvas = np.full((tile_h * 2, tile_w * 4, 3), 245, dtype=np.uint8)
    for i, (name, frame) in enumerate(frames):
        yoff, xoff = (i // 4) * tile_h, (i % 4) * tile_w
        scale = min((tile_w - 12) / frame.shape[1], (tile_h - 34) / frame.shape[0])
        small = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        tx, ty = xoff + (tile_w - small.shape[1]) // 2, yoff + 26
        canvas[ty:ty + small.shape[0], tx:tx + small.shape[1]] = small
        cv2.putText(canvas, name, (xoff + 10, yoff + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 25, 25), 1, cv2.LINE_AA)
        for pos, (_, box) in enumerate(boxes):
            if name != "ooo (source)" and name[pos] == "x":
                continue
            x1, y1, x2, y2 = box
            cv2.rectangle(canvas, (tx + round(x1 * scale), ty + round(y1 * scale)),
                          (tx + round(x2 * scale), ty + round(y2 * scale)), (29, 139, 245), 1)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-sources", type=int, default=0, help="Pilot only; 0 uses all eligible development images")
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    processed, out = args.processed.resolve(), args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty output: {out}")
    eligible = [r for r in rows(processed / "curated/manifests/all_unique.csv")
                if r["split"] in ("train", "val") and int(r["box_count"]) == 3]
    if args.max_sources:
        rng = random.Random(args.seed)
        by_machine = defaultdict(list)
        for row in eligible:
            by_machine[row["machine"]].append(row)
        selected = []
        for machine, subset in sorted(by_machine.items()):
            selected.extend(rng.sample(subset, min(len(subset), max(1, args.max_sources // 3))))
        eligible = sorted(selected[:args.max_sources], key=lambda r: r["output_id"])
    else:
        eligible.sort(key=lambda r: r["output_id"])
    cv_fold = {r["output_id"]: r["fold"] for r in rows(processed.parent / "training/cv/fold_assignments.csv")}
    for pattern in PATTERNS:
        (out / pattern / "images").mkdir(parents=True)
        (out / pattern / "labels").mkdir(parents=True)
    (out / "previews").mkdir(parents=True)
    manifest, rejected = [], []
    previews_per_machine = Counter()
    for index, row in enumerate(eligible, start=1):
        source_id = row["output_id"]
        try:
            image = image_read(processed / row["clean_image"])
            boxes = parse_boxes(processed / row["label"], image.shape[1], image.shape[0])
            if len(boxes) != 3:
                raise ValueError(f"not_three_boxes:{len(boxes)}")
            repairs = [removal_delta(image, box) for _, box in boxes]
            variants = {pattern: compose(image, repairs, pattern) for pattern in PATTERNS}
            for pattern, result in variants.items():
                name = f"{source_id}__{pattern}"
                image_write(out / pattern / "images" / f"{name}.png", result)
                label_rows = [line for flag, (line, _) in zip(pattern, boxes, strict=True) if flag == "o"]
                (out / pattern / "labels" / f"{name}.txt").write_text(
                    "\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
                manifest.append({"source_id": source_id, "pattern": pattern, "machine": row["machine"],
                                 "session_id": row["session_id"], "source_split": row["split"],
                                 "cv_validation_fold": cv_fold[source_id], "kept_boxes": pattern.count("o"),
                                 "removed_boxes": pattern.count("x"), "image": f"{pattern}/images/{name}.png",
                                 "label": f"{pattern}/labels/{name}.txt",
                                 "repair_diagnostics": json.dumps([repair[3] for repair in repairs])})
            if previews_per_machine[row["machine"]] < 2:
                previews_per_machine[row["machine"]] += 1
                image_write(out / "previews" / f"m{row['machine']}_{previews_per_machine[row['machine']]}__{source_id}.png",
                            preview(image, boxes, variants))
        except Exception as exc:
            rejected.append({"source_id": source_id, "machine": row["machine"],
                             "session_id": row["session_id"], "reason": f"{type(exc).__name__}:{exc}"})
        if index % 100 == 0 or index == len(eligible):
            print(f"Processed {index}/{len(eligible)}; accepted={len(manifest) // 7}, rejected={len(rejected)}", flush=True)
    if manifest:
        with (out / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
    with (out / "rejected.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_id", "machine", "session_id", "reason"))
        writer.writeheader()
        writer.writerows(rejected)
    summary = {"source_candidates": len(eligible), "accepted_sources": len(manifest) // 7,
               "generated_images": len(manifest), "rejected_sources": len(rejected),
               "rejection_reasons": dict(Counter(r["reason"].split(":")[1].split(":")[0] for r in rejected)),
               "patterns": PATTERNS, "source_policy": "unique development only; fixed test excluded",
               "warning": "Not seven independent samples per source. Use fold-aware subsampling and validate on untouched sessions."}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Three-position presence/absence augmentation\n\n"
        "The three weak YOLO boxes are ordered by vertical center: top, middle, bottom.\n"
        "`o` means the original dark dot is retained; `x` means its local dark core is inpainted.\n"
        "The source `ooo` image remains in `processed/images_clean` and is not duplicated here.\n"
        "Each pattern folder contains one PNG and a geometry-preserving YOLO label file for each accepted source.\n"
        "`xxx` has an empty label file. Removed locations are not relabeled as positives.\n\n"
        f"This run accepted {summary['accepted_sources']} of {summary['source_candidates']} unique three-box development images "
        f"and generated {summary['generated_images']} pattern images. See `rejected.csv` for skipped samples.\n"
        "The fixed test split and exact duplicates were excluded from generation.\n"
        "For cross-validation, never train on a pattern image whose `cv_validation_fold` equals the held-out fold.\n"
        "All seven variants of one source are highly correlated; sample a small number or weight source groups equally.\n"
        "These are synthetic weak-label images, not additional independent observations.\n"
        "Visual inspection and a fold-wise baseline comparison are required before claiming a performance gain.\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
