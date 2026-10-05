#!/usr/bin/env python3
"""Make an auditable side-by-side view of the three dataset interventions."""

import csv
import json
from pathlib import Path

import cv2


def main() -> None:
    root = Path("/Users/kyungrae/Desktop/kamp_ai/training/experiments/point_decoy_roi")
    rows = list(csv.DictReader((root / "variant_manifest.csv").open()))
    sample = next(r["source"] for r in rows if r["split"] == "val" and r["variant"] == "point"
                  and Path(r["source"]).name.startswith("m2_002_20200728_123106"))
    by_variant = {r["variant"]: r for r in rows if r["split"] == "val" and r["source"] == sample}
    views = []
    for name in ("original", "point", "decoy", "roi"):
        path = sample if name == "original" else by_variant[name]["image"]
        image = cv2.imread(path)
        if name == "point":
            x, y, side = json.loads(by_variant[name]["detail"])
            cv2.circle(image, (x, y), side + 7, (0, 0, 255), 2)
        if name == "decoy":
            x1, y1, x2, y2, _ = json.loads(by_variant[name]["detail"])
            cv2.rectangle(image, (x1-5, y1-5), (x2+5, y2+5), (255, 0, 0), 2)
        image = cv2.resize(image, (320, 320))
        panel = cv2.copyMakeBorder(image, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(panel, name.upper(), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 20, 20), 2)
        views.append(panel)
    result = cv2.hconcat(views)
    dest = root / "previews" / "variant_comparison.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), result)
    print(dest)


if __name__ == "__main__":
    main()
