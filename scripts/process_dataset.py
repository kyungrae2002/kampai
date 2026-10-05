#!/usr/bin/env python3
"""Build a leakage-reduced X-ray dataset from color-box annotated BMP files.

The color overlays are used once to create weak YOLO labels.  Only overlay pixels
are then repaired with local inpainting; source images are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode the image")
    return image


def write_image(path: Path, image: np.ndarray, params: list[int] | None = None) -> None:
    suffix = path.suffix.lower()
    ok, encoded = cv2.imencode(suffix, image, params or [])
    if not ok:
        raise ValueError(f"OpenCV could not encode {suffix}")
    encoded.tofile(path)


def source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def color_mask(image: np.ndarray, chroma_threshold: int, brightness_threshold: int) -> np.ndarray:
    high = image.max(axis=2).astype(np.int16)
    low = image.min(axis=2).astype(np.int16)
    return (((high - low) >= chroma_threshold) & (high >= brightness_threshold)).astype(np.uint8) * 255


def rectangle_score(component: np.ndarray) -> tuple[float, tuple[float, float, float, float]]:
    height, width = component.shape
    band = max(1, min(3, min(height, width) // 5))
    top = float(component[:band].mean())
    bottom = float(component[-band:].mean())
    left = float(component[:, :band].mean())
    right = float(component[:, -band:].mean())
    coverages = (top, bottom, left, right)
    return min(coverages), coverages


def extract_boxes(mask: np.ndarray) -> tuple[list[tuple[int, int, int, int]], int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    rejected = 0

    for component_id in range(1, count):
        x, y, width, height, area = map(int, stats[component_id])
        if width < 5 or height < 5 or area < 12:
            rejected += 1
            continue

        component = labels[y : y + height, x : x + width] == component_id
        border_score, _ = rectangle_score(component)
        aspect = width / max(height, 1)
        perimeter_density = area / max(2.0 * (width + height), 1.0)

        if border_score < 0.35 or not (0.12 <= aspect <= 8.0) or perimeter_density < 0.35:
            rejected += 1
            continue

        boxes.append((x, y, x + width - 1, y + height - 1))

    boxes.sort(key=lambda box: (box[1], box[0], box[3], box[2]))
    return boxes, rejected


def yolo_rows(boxes: list[tuple[int, int, int, int]], width: int, height: int) -> list[str]:
    rows = []
    for x1, y1, x2, y2 in boxes:
        box_width = x2 - x1 + 1
        box_height = y2 - y1 + 1
        center_x = x1 + box_width / 2.0
        center_y = y1 + box_height / 2.0
        rows.append(
            f"0 {center_x / width:.8f} {center_y / height:.8f} "
            f"{box_width / width:.8f} {box_height / height:.8f}"
        )
    return rows


def machine_id(relative_path: Path) -> str:
    for part in relative_path.parts:
        match = re.match(r"([123])", part)
        if match:
            return match.group(1)
    return "unknown"


def output_id(relative_path: Path) -> str:
    stem = re.sub(r"[^0-9A-Za-z_.()-]+", "_", relative_path.stem)
    short_hash = hashlib.sha1(str(relative_path).encode("utf-8")).hexdigest()[:8]
    return f"m{machine_id(relative_path)}_{stem}__{short_hash}"


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    bar_height = 28
    panel = cv2.copyMakeBorder(image, bar_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
    cv2.putText(panel, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def create_preview(
    original: np.ndarray,
    repair_mask: np.ndarray,
    clean: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    mask_view = original.copy()
    mask_view[repair_mask > 0] = (0, 0, 255)
    clean_view = clean.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(clean_view, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return cv2.hconcat(
        [
            add_title(original, "ORIGINAL"),
            add_title(mask_view, "REMOVAL MASK"),
            add_title(clean_view, "CLEAN + LABEL"),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chroma-threshold", type=int, default=80)
    parser.add_argument("--brightness-threshold", type=int, default=180)
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--inpaint-radius", type=float, default=3.0)
    parser.add_argument("--preview-quality", type=int, default=86)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output}")

    images_dir = output / "images_clean"
    labels_dir = output / "labels"
    masks_dir = output / "masks"
    previews_dir = output / "previews"
    for directory in (images_dir, labels_dir, masks_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images found below {source}")

    rows: list[dict[str, object]] = []
    digest_first_seen: dict[str, str] = {}
    status_counts: Counter[str] = Counter()
    total_boxes = 0
    total_colored_pixels = 0

    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(source)
        item_id = output_id(relative)
        status = "ok"
        warnings: list[str] = []

        try:
            original = read_image(path)
            height, width = original.shape[:2]
            overlay_mask = color_mask(original, args.chroma_threshold, args.brightness_threshold)
            colored_pixels = int(np.count_nonzero(overlay_mask))
            boxes, rejected_components = extract_boxes(overlay_mask)

            if colored_pixels == 0:
                warnings.append("no_colored_overlay")
            if not boxes:
                warnings.append("no_rectangle_detected")
            if rejected_components:
                warnings.append(f"rejected_color_components:{rejected_components}")

            if args.dilate > 0 and colored_pixels:
                kernel = np.ones((3, 3), np.uint8)
                repair_mask = cv2.dilate(overlay_mask, kernel, iterations=args.dilate)
            else:
                repair_mask = overlay_mask

            if colored_pixels:
                clean = cv2.inpaint(original, repair_mask, args.inpaint_radius, cv2.INPAINT_TELEA)
            else:
                clean = original.copy()

            clean_path = images_dir / f"{item_id}.png"
            label_path = labels_dir / f"{item_id}.txt"
            mask_path = masks_dir / f"{item_id}.png"
            preview_path = previews_dir / f"{item_id}.jpg"

            write_image(clean_path, clean, [cv2.IMWRITE_PNG_COMPRESSION, 5])
            write_image(mask_path, repair_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            label_path.write_text("\n".join(yolo_rows(boxes, width, height)) + ("\n" if boxes else ""), encoding="utf-8")
            preview = create_preview(original, repair_mask, clean, boxes)
            write_image(preview_path, preview, [cv2.IMWRITE_JPEG_QUALITY, args.preview_quality])

            digest = source_digest(path)
            duplicate_of = digest_first_seen.get(digest, "")
            if duplicate_of:
                warnings.append(f"exact_duplicate_of:{duplicate_of}")
            else:
                digest_first_seen[digest] = item_id

            if warnings:
                status = "review"
            mask_pixels = int(np.count_nonzero(repair_mask))
            total_boxes += len(boxes)
            total_colored_pixels += colored_pixels

            rows.append(
                {
                    "output_id": item_id,
                    "source_path": str(path),
                    "source_relative_path": str(relative),
                    "machine": machine_id(relative),
                    "width": width,
                    "height": height,
                    "sha256": digest,
                    "duplicate_of": duplicate_of,
                    "colored_pixels": colored_pixels,
                    "repair_mask_pixels": mask_pixels,
                    "repair_mask_fraction": f"{mask_pixels / (width * height):.8f}",
                    "label_count": len(boxes),
                    "rejected_components": rejected_components,
                    "status": status,
                    "warnings": ";".join(warnings),
                    "clean_image": str(clean_path.relative_to(output)),
                    "label": str(label_path.relative_to(output)),
                    "mask": str(mask_path.relative_to(output)),
                    "preview": str(preview_path.relative_to(output)),
                }
            )
        except Exception as exc:  # keep a complete failure manifest
            status = "error"
            rows.append(
                {
                    "output_id": item_id,
                    "source_path": str(path),
                    "source_relative_path": str(relative),
                    "machine": machine_id(relative),
                    "status": status,
                    "warnings": f"{type(exc).__name__}:{exc}",
                }
            )

        status_counts[status] += 1
        if index % 100 == 0 or index == len(paths):
            print(f"Processed {index}/{len(paths)}")

    fieldnames = [
        "output_id",
        "source_path",
        "source_relative_path",
        "machine",
        "width",
        "height",
        "sha256",
        "duplicate_of",
        "colored_pixels",
        "repair_mask_pixels",
        "repair_mask_fraction",
        "label_count",
        "rejected_components",
        "status",
        "warnings",
        "clean_image",
        "label",
        "mask",
        "preview",
    ]
    with (output / "processing_log.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    (output / "classes.names").write_text("defect\n", encoding="utf-8")
    dataset_config = {
        "path": str(output),
        "train": "images_clean",
        "val": "images_clean",
        "names": {0: "defect"},
        "note": "A leakage-safe train/val split must be created by acquisition session before training.",
    }
    (output / "dataset.yaml").write_text(
        "path: " + json.dumps(str(output), ensure_ascii=False) + "\n"
        "train: images_clean\n"
        "val: images_clean\n"
        "names:\n  0: defect\n"
        "# Do not train with this placeholder split; create a session-level split first.\n",
        encoding="utf-8",
    )
    summary = {
        "source": str(source),
        "output": str(output),
        "images": len(paths),
        "unique_content": len(digest_first_seen),
        "exact_duplicates": len(paths) - len(digest_first_seen),
        "labels": total_boxes,
        "colored_pixels": total_colored_pixels,
        "status_counts": dict(status_counts),
        "parameters": {
            "chroma_threshold": args.chroma_threshold,
            "brightness_threshold": args.brightness_threshold,
            "dilate": args.dilate,
            "inpaint_radius": args.inpaint_radius,
        },
    }
    (output / "processing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
