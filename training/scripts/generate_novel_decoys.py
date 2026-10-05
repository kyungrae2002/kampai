#!/usr/bin/env python3
"""Unseen-style decoy challenge: larger, thicker, blue/green rectangles."""

import csv
import json
import random
from pathlib import Path

import cv2

from experiment_variants import find_location, label_path, read_labels, rects_px, save_labels


def main() -> None:
    project = Path("/Users/kyungrae/Desktop/kamp_ai")
    root = project / "training" / "experiments" / "point_decoy_roi"
    list_path = root / "lists" / "novel_decoy" / "test_variant.txt"
    if list_path.exists():
        raise SystemExit("Novel decoy test already generated")
    sources = [Path(x) for x in (project / "training" / "cv" / "test.txt").read_text().splitlines() if x.strip()]
    rows, generated = [], []
    for index, source in enumerate(sources):
        rng = random.Random(20261001 + 300000 + index)
        image = cv2.imread(str(source))
        labels = read_labels(source)
        h, w = image.shape[:2]
        boxes = rects_px(labels, w, h)
        x, y = find_location(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), boxes, rng, margin=40)
        sizes = [(max(20, x2-x1), max(20, y2-y1)) for x1, y1, x2, y2 in boxes]
        bw, bh = rng.choice(sizes) if sizes else (20, 20)
        bw, bh = min(int(bw * 1.5), 48), min(int(bh * 1.5), 48)
        x1, y1, x2, y2 = x-bw//2, y-bh//2, x+bw//2, y+bh//2
        color = (255, 0, 0) if index % 2 else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        dest = root / "images" / "novel_decoy" / "test" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), image)
        save_labels(label_path(dest), labels)
        generated.append(dest)
        rows.append({"variant": "novel_decoy", "split": "test", "source": str(source),
                     "image": str(dest), "source_labels": len(labels), "result_labels": len(labels),
                     "detail": json.dumps((x1, y1, x2, y2, color))})
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(map(str, generated)) + "\n")
    with (root / "variant_manifest.csv").open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("variant", "split", "source", "image", "source_labels", "result_labels", "detail"))
        writer.writerows(rows)
    print(f"Generated {len(rows)} unseen-style test decoys")


if __name__ == "__main__":
    main()
