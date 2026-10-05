#!/usr/bin/env python3
"""Resumable matched 5-fold fine-tuning and evaluation for three interventions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


VARIANTS = ("control", "point", "decoy", "roi")


def execute(command: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--fold-start", type=int, default=2)
    parser.add_argument("--fold-end", type=int, default=5)
    args = parser.parse_args()
    project = args.project.resolve()
    repo = project / "dataset" / "dataset" / "test1" / "yolov3"
    scripts = project / "training" / "scripts"
    for fold in range(args.fold_start, args.fold_end + 1):
        if fold not in (2, 3, 4, 5):
            raise ValueError(f"Unexpected fold {fold}")
        root = project / "training" / "experiments" / "point_decoy_roi_5fold" / f"fold_{fold}"
        if not (root / "variant_manifest.csv").exists():
            print(f"DATA fold={fold}", flush=True)
            execute([sys.executable, str(scripts / "generate_cv_variants.py"),
                     "--project", str(project), "--fold", str(fold)], root.parent / f"fold_{fold}_generation.log", project)
        for variant in VARIANTS:
            run = root / "runs" / variant
            weights = run / "last.pt"
            summary = run / "summary.json"
            if not (weights.exists() and summary.exists()):
                print(f"TRAIN fold={fold} variant={variant}", flush=True)
                began = time.time()
                execute([sys.executable, str(scripts / "train_yolov3_mps.py"),
                         "--repo", str(repo), "--cfg", str(repo / "yolov3-spp.cfg"),
                         "--train-list", str(root / "lists" / variant / "train.txt"),
                         "--val-list", str(project / "training" / "cv" / f"fold_{fold}" / "val.txt"),
                         "--init-weights", str(project / "training" / "runs" / "integrated_cv" / f"fold_{fold}" / "best.pt"),
                         "--output", str(run), "--epochs", "2", "--batch-size", "4", "--img-size", "320",
                         "--workers", "0", "--lr", "0.0001", "--accumulate", "4", "--seed", "20261001",
                         "--device", "mps"], root / "logs" / f"{variant}_train.log", project)
                print(f"TRAIN_DONE fold={fold} variant={variant} seconds={time.time()-began:.1f}", flush=True)
            metric = root / "metrics" / f"{variant}.json"
            if not metric.exists():
                print(f"EVAL fold={fold} variant={variant}", flush=True)
                execute([sys.executable, str(scripts / "evaluate_variants.py"),
                         "--project", str(project), "--experiment-root", str(root),
                         "--weights", str(weights), "--model-name", variant,
                         "--threshold", "0.07", "--output", str(metric)],
                        root / "logs" / f"{variant}_eval.log", project)
            record = json.loads(metric.read_text())
            views = record["views"]
            print(json.dumps({"fold": fold, "variant": variant,
                              "original_f1": views["original"]["center"]["f1"],
                              "point_recall": views["point"]["point_recall"],
                              "decoy_hits": views["decoy"]["decoy_hits"],
                              "decoy_f1": views["decoy"]["center"]["f1"],
                              "roi_f1": views["roi"]["center"]["f1"]}), flush=True)


if __name__ == "__main__":
    main()
