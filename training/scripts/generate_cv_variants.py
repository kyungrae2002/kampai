#!/usr/bin/env python3
"""Build one session-held-out fold of the four controlled KAMP datasets."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from experiment_variants import process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(2, 6), required=True)
    parser.add_argument("--count", type=int, default=600)
    args = parser.parse_args()
    project = args.project
    root = project / "training" / "experiments" / "point_decoy_roi_5fold" / f"fold_{args.fold}"
    if root.exists():
        raise SystemExit(f"Refusing to overwrite existing fold: {root}")
    source_fold = project / "training" / "cv" / f"fold_{args.fold}"
    train = [Path(s) for s in (source_fold / "train.txt").read_text().splitlines() if s.strip()]
    val = [Path(s) for s in (source_fold / "val.txt").read_text().splitlines() if s.strip()]
    seed = 20261001 + args.fold * 1000
    selected = random.Random(seed).sample(train, min(args.count, len(train)))
    rows = []
    for variant in ("control", "point", "decoy", "roi"):
        train_list, val_list = list(train), []
        for split, sources in (("train", selected), ("val", val)):
            for index, source in enumerate(sources):
                dest = root / "images" / variant / split / source.name
                row = process(source, dest, variant, seed + index + (100000 if split == "val" else 0), split == "train")
                row["split"] = split
                rows.append(row)
                (train_list if split == "train" else val_list).append(dest)
        list_dir = root / "lists" / variant
        list_dir.mkdir(parents=True, exist_ok=True)
        (list_dir / "train.txt").write_text("\n".join(map(str, train_list)) + "\n")
        (list_dir / "val_variant.txt").write_text("\n".join(map(str, val_list)) + "\n")
        (list_dir / "val_original.txt").write_text("\n".join(map(str, val)) + "\n")
    with (root / "variant_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("variant", "split", "source", "image", "source_labels", "result_labels", "detail"))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"fold": args.fold, "train": len(train), "val": len(val),
                      "augmented_per_variant": len(selected), "records": len(rows), "root": str(root)}))


if __name__ == "__main__":
    main()
