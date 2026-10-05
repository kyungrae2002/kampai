#!/usr/bin/env python3
"""Create reproducible noise and faint-defect variants of every cleaned X-ray frame.

The source clean images and labels are never modified. All variants preserve the
weak YOLO boxes; model training must still use only fold-eligible image lists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


VARIANTS = {
    "01_overlay_removed_clean": "Clean X-ray frame after colored overlay removal; no augmentation",
    "02_random_gaussian_noise": "Same clean frame plus image-level random-strength Gaussian sensor noise",
    "03_fainter_labeled_defect": "Locally attenuated dark detail inside existing weak-label boxes",
    "04_fainter_defect_plus_noise": "Fainter labeled defect followed by the same noise realization as variant 02",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError(f"Could not encode {path}")
    encoded.tofile(path)


def labels_px(path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = (float(value) for value in line.split())
        if cls != 0:
            raise ValueError(f"Unexpected class {cls} in {path}")
        x1 = max(0, min(width - 1, math.floor((cx - bw / 2) * width)))
        y1 = max(0, min(height - 1, math.floor((cy - bh / 2) * height)))
        x2 = max(x1 + 1, min(width, math.ceil((cx + bw / 2) * width)))
        y2 = max(y1 + 1, min(height, math.ceil((cy + bh / 2) * height)))
        boxes.append((x1, y1, x2, y2))
    return boxes


def fade_labeled_dark_detail(
    clean: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    strength: float,
) -> tuple[np.ndarray, int, float]:
    if not boxes:
        return clean.copy(), 0, 0.0
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    delta = np.zeros(gray.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        box_w, box_h = x2 - x1, y2 - y1
        k = int(np.clip(round(0.8 * min(box_w, box_h)), 7, 17)) | 1
        pad = max(8, k)
        px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
        px2, py2 = min(gray.shape[1], x2 + pad), min(gray.shape[0], y2 + pad)
        region = gray[py1:py2, px1:px2]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        background = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)
        deficit = np.maximum(background.astype(np.float32) - region.astype(np.float32), 0)
        core = deficit[y1 - py1:y2 - py1, x1 - px1:x2 - px1]
        if not core.size:
            continue
        # Suppress flat texture. The taper avoids a new rectangular boundary.
        yy, xx = np.mgrid[0:box_h, 0:box_w]
        taper_x = np.sin(np.pi * (xx + 0.5) / box_w) ** 0.55
        taper_y = np.sin(np.pi * (yy + 0.5) / box_h) ** 0.55
        support = np.clip((core - 2.5) / 8.0, 0, 1)
        adjustment = strength * core * support * taper_x * taper_y
        view = delta[y1:y2, x1:x2]
        np.maximum(view, adjustment, out=view)
    result = np.clip(clean.astype(np.float32) + delta[..., None], 0, 255).astype(np.uint8)
    changed = int(np.count_nonzero(np.any(result != clean, axis=2)))
    return result, changed, float(delta.max())


def add_grayscale_noise(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, image.shape[:2]).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)


def preview_panel(images: list[np.ndarray], box: tuple[int, int, int, int] | None) -> np.ndarray:
    names = ("CLEAN", "RANDOM NOISE", "FAINTER DEFECT", "FAINTER + NOISE")
    tile_width = 480
    tile_height = 345
    zoom_size = 158
    panels = []
    for name, image in zip(names, images, strict=True):
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        tile = np.full((tile_height + zoom_size + 43, tile_width, 3), 246, dtype=np.uint8)
        xoff, yoff = (tile_width - resized.shape[1]) // 2, 30
        tile[yoff:yoff + resized.shape[0], xoff:xoff + resized.shape[1]] = resized
        cv2.putText(tile, name, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.59, (20, 20, 20), 1, cv2.LINE_AA)
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(tile, (xoff + round(x1 * scale), yoff + round(y1 * scale)),
                          (xoff + round(x2 * scale), yoff + round(y2 * scale)), (32, 125, 245), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = max(17, int(max(x2 - x1, y2 - y1) * 1.5))
            crop = image[max(0, cy - half):min(image.shape[0], cy + half),
                         max(0, cx - half):min(image.shape[1], cx + half)]
            zoom = cv2.resize(crop, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)
            tile[tile_height + 35:tile_height + 35 + zoom_size, 9:9 + zoom_size] = zoom
            cv2.putText(tile, "ZOOM (same ROI)", (179, tile_height + 61), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (60, 60, 60), 1, cv2.LINE_AA)
        panels.append(tile)
    return cv2.hconcat(panels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--max-images", type=int, default=0, help="Pilot only; 0 processes all")
    parser.add_argument("--overwrite-own-output", action="store_true",
                        help="Regenerate an existing output only if it already contains this tool's manifest")
    args = parser.parse_args()
    processed, out = args.processed.resolve(), args.output.resolve()
    if out.exists() and any(out.iterdir()):
        if not args.overwrite_own_output or not (out / "augmentation_manifest.csv").is_file():
            raise SystemExit(f"Refusing to overwrite nonempty output: {out}")
    source_rows = read_csv(processed / "processing_log.csv")
    source_rows = [row for row in source_rows if row.get("clean_image") and row.get("label")]
    if args.max_images:
        by_machine = defaultdict(list)
        for row in source_rows:
            by_machine[row["machine"]].append(row)
        rng = random.Random(args.seed)
        chosen = []
        for machine, rows in sorted(by_machine.items()):
            positives = [r for r in rows if int(r["label_count"]) > 0]
            chosen.extend(rng.sample(positives, min(math.ceil(args.max_images / 3), len(positives))))
        source_rows = sorted(chosen[:args.max_images], key=lambda r: r["output_id"])
    else:
        source_rows.sort(key=lambda row: row["output_id"])

    unique_rows = read_csv(processed / "curated/manifests/all_unique.csv")
    split_by_id = {row["output_id"]: row["split"] for row in unique_rows}
    session_by_id = {row["output_id"]: row["session_id"] for row in unique_rows}
    fold_path = processed.parent / "training/cv/fold_assignments.csv"
    fold_by_id = {row["output_id"]: row["fold"] for row in read_csv(fold_path)}

    for name in VARIANTS:
        (out / name / "images").mkdir(parents=True, exist_ok=True)
        (out / name / "labels").mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(parents=True, exist_ok=True)
    records = []
    errors = []
    preview_counts = Counter()
    for index, row in enumerate(source_rows, start=1):
        try:
            source = processed / row["clean_image"]
            source_label = processed / row["label"]
            clean = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
            if clean is None:
                raise ValueError(f"Cannot decode {source}")
            boxes = labels_px(source_label, clean.shape[1], clean.shape[0])
            # Some test frames have a colored metadata glyph in the upper-left
            # corner. Its weak box is not an X-ray defect and must not be faded.
            metadata_boxes = [b for b in boxes if b[2] <= 0.12 * clean.shape[1]
                              and b[3] <= 0.12 * clean.shape[0]]
            defect_boxes = [b for b in boxes if b not in metadata_boxes]
            content_digest = row["sha256"]
            item_seed = args.seed ^ int(content_digest[:16], 16)
            params = np.random.default_rng(item_seed)
            sigma = float(params.uniform(2.5, 9.5))
            fade_strength = float(params.uniform(0.40, 0.75))
            noise_seed = item_seed ^ 0x92C43015
            faded, changed_pixels, max_adjustment = fade_labeled_dark_detail(clean, defect_boxes, fade_strength)
            noise = add_grayscale_noise(clean, sigma, noise_seed)
            both = add_grayscale_noise(faded, sigma, noise_seed)
            generated = (clean, noise, faded, both)
            filename = source.name
            label_name = source_label.name
            for (directory, _), image in zip(VARIANTS.items(), generated, strict=True):
                image_dest = out / directory / "images" / filename
                label_dest = out / directory / "labels" / label_name
                if directory == "01_overlay_removed_clean":
                    shutil.copy2(source, image_dest)
                else:
                    write_png(image_dest, image)
                shutil.copy2(source_label, label_dest)
            canonical_id = row["duplicate_of"] or row["output_id"]
            split = split_by_id.get(canonical_id, "unknown")
            records.append({
                "output_id": row["output_id"], "canonical_id": canonical_id,
                "original_sha256": content_digest, "machine": row["machine"],
                "split": split, "cv_validation_fold": fold_by_id.get(canonical_id, ""),
                "session_id": session_by_id.get(canonical_id, ""),
                "duplicate_of": row["duplicate_of"], "label_count": len(boxes),
                "metadata_boxes_skipped": len(metadata_boxes),
                "noise_sigma_8bit": f"{sigma:.4f}", "fade_strength": f"{fade_strength:.4f}",
                "faded_changed_pixels": changed_pixels, "faded_max_adjustment": f"{max_adjustment:.4f}",
                "faint_status": ("no_label_unchanged" if not boxes else
                                 "metadata_label_skipped" if not defect_boxes else
                                 "changed_metadata_label_skipped" if metadata_boxes and changed_pixels else
                                 "changed" if changed_pixels else "no_local_dark_feature"),
                "image_filename": filename, "label_filename": label_name,
            })
            if boxes and preview_counts[row["machine"]] < 2 and changed_pixels:
                preview_counts[row["machine"]] += 1
                panel = preview_panel(list(generated), boxes[0])
                write_png(out / "previews" / f"m{row['machine']}_{preview_counts[row['machine']]}__{row['output_id']}.png", panel)
        except Exception as exc:
            errors.append({"output_id": row["output_id"], "error": f"{type(exc).__name__}: {exc}"})
        if index % 200 == 0 or index == len(source_rows):
            print(f"Processed {index}/{len(source_rows)}; errors={len(errors)}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    if records:
        with (out / "augmentation_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "source_clean_images": len(source_rows),
        "generated_per_variant": len(records),
        "errors": errors,
        "variants": VARIANTS,
        "faint_status_counts": dict(Counter(r["faint_status"] for r in records)),
        "metadata_boxes_skipped": sum(r["metadata_boxes_skipped"] for r in records),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "noise_sigma_range": [2.5, 9.5],
        "fade_strength_range": [0.40, 0.75],
        "seed": args.seed,
        "note": "Labels are unchanged weak labels. Use only canonical, fold-eligible training records; never train on held-out test or validation-fold variants.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Clean-image noise and faint-defect augmentation\n\n"
        + "\n".join(f"- `{name}/`: {description}" for name, description in VARIANTS.items())
        + "\n\nAll 2,809 cleaned frames are retained for audit. The original processed directory is unchanged.\n"
        "Noise is zero-mean grayscale Gaussian with per-image random sigma 2.5-9.5 (8-bit).\n"
        "Faintness attenuates only local dark detail within existing weak-label boxes, with random strength 0.40-0.75.\n"
        "Negative images contain no defect to fade and are unchanged in the faint-only directory.\n"
        "YOLO labels are copied unchanged because geometry is unchanged. These remain weak labels, not manually verified object masks.\n"
        "The same noise realization is used for noise-only and combined variants of each source image.\n"
        "Upper-left metadata-glyph boxes are excluded from fading and flagged in the manifest; original weak labels are kept for audit.\n"
        f"This run flagged {summary['metadata_boxes_skipped']} such boxes; review their labels before any model evaluation.\n"
        "For training, exclude exact duplicates and fixed test images, and in each CV fold use only records whose cv_validation_fold differs from the current fold.\n"
        "Never treat all variant directories as a single training set.\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "variants"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
