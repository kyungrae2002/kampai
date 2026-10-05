#!/usr/bin/env python3
"""Prepare held-out counterfactual test views after validation-only model selection."""

import csv
from pathlib import Path

from experiment_variants import process


def main() -> None:
    project = Path("/Users/kyungrae/Desktop/kamp_ai")
    root = project / "training" / "experiments" / "point_decoy_roi"
    with (root / "variant_manifest.csv").open(newline="") as f:
        if any(row["split"] == "test" for row in csv.DictReader(f)):
            raise SystemExit("Test views already generated; refusing to append duplicate manifest rows")
    sources = [Path(x) for x in (project / "training" / "cv" / "test.txt").read_text().splitlines() if x.strip()]
    rows = []
    for variant in ("point", "decoy", "roi"):
        generated = []
        for index, source in enumerate(sources):
            dest = root / "images" / variant / "test" / source.name
            row = process(source, dest, variant, 20261001 + 200000 + index, False)
            row["split"] = "test"
            rows.append(row)
            generated.append(dest)
        (root / "lists" / variant / "test_variant.txt").write_text("\n".join(map(str, generated)) + "\n")
    with (root / "variant_manifest.csv").open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("variant", "split", "source", "image", "source_labels", "result_labels", "detail"))
        writer.writerows(rows)
    print(f"Generated {len(rows)} test variants from {len(sources)} session-held-out images")


if __name__ == "__main__":
    main()
