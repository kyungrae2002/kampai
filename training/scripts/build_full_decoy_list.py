#!/usr/bin/env python3
"""Combine all development originals with session-safe decoy training examples."""

import csv
from pathlib import Path


def main() -> None:
    project = Path("/Users/kyungrae/Desktop/kamp_ai")
    root = project / "training" / "experiments" / "point_decoy_roi"
    originals = [Path(x) for x in (project / "training" / "cv" / "integrated" / "train.txt").read_text().splitlines() if x.strip()]
    test = set((project / "training" / "cv" / "test.txt").read_text().splitlines())
    with (root / "variant_manifest.csv").open(newline="") as f:
        decoys = [row for row in csv.DictReader(f) if row["variant"] == "decoy" and row["split"] in ("train", "val")]
    assert len(originals) == 2157 and len(decoys) == 1030
    assert not (set(map(str, originals)) & test)
    assert all(row["source"] not in test for row in decoys)
    assert all(Path(row["image"]).exists() for row in decoys)
    paths = originals + [Path(row["image"]) for row in decoys]
    dest = root / "lists" / "decoy" / "full_development_train.txt"
    dest.write_text("\n".join(map(str, paths)) + "\n")
    print(f"{dest}: {len(originals)} original + {len(decoys)} decoy = {len(paths)} images")


if __name__ == "__main__":
    main()
