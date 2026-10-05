#!/usr/bin/env python3
"""Create leakage-safe 5-fold manifests for the curated KAMP dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_tie(text: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()[:16], 16)


def assign_groups(rows: list[dict[str, str]], folds: int, seed: int) -> dict[str, int]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["session_id"]].append(row)

    # Balance each machine independently, while also balancing boxes and negatives.
    assignment: dict[str, int] = {}
    by_machine: dict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for sid, members in groups.items():
        machines = {r["machine"] for r in members}
        if len(machines) != 1:
            raise ValueError(f"Session {sid} spans machines: {machines}")
        by_machine[next(iter(machines))].append((sid, members))

    for machine, sessions in sorted(by_machine.items()):
        sessions.sort(
            key=lambda item: (
                -len(item[1]),
                -sum(int(r["box_count"]) for r in item[1]),
                -sum(r["sample_type"] == "negative_user_confirmed" for r in item[1]),
                stable_tie(item[0], seed),
            )
        )
        loads = [{"images": 0, "boxes": 0, "negatives": 0, "sessions": 0} for _ in range(folds)]
        totals = {
            "images": sum(len(m) for _, m in sessions) / folds,
            "boxes": sum(sum(int(r["box_count"]) for r in m) for _, m in sessions) / folds,
            "negatives": max(1.0, sum(sum(r["sample_type"] == "negative_user_confirmed" for r in m) for _, m in sessions) / folds),
            "sessions": len(sessions) / folds,
        }
        for sid, members in sessions:
            item = {
                "images": len(members),
                "boxes": sum(int(r["box_count"]) for r in members),
                "negatives": sum(r["sample_type"] == "negative_user_confirmed" for r in members),
                "sessions": 1,
            }
            def score(fold: int) -> tuple[float, int, int]:
                projected = {k: loads[fold][k] + item[k] for k in item}
                imbalance = (
                    4.0 * (projected["images"] / max(totals["images"], 1)) ** 2
                    + 2.0 * (projected["boxes"] / max(totals["boxes"], 1)) ** 2
                    + 1.0 * (projected["negatives"] / totals["negatives"]) ** 2
                    + 0.5 * (projected["sessions"] / max(totals["sessions"], 1)) ** 2
                )
                return imbalance, loads[fold]["images"], stable_tie(f"{sid}:{fold}", seed)
            chosen = min(range(folds), key=score)
            assignment[sid] = chosen
            for key in item:
                loads[chosen][key] += item[key]
    return assignment


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{v}\n" for v in values), encoding="utf-8")


def summarize(rows: list[dict[str, str]]) -> dict:
    return {
        "images": len(rows),
        "sessions": len({r["session_id"] for r in rows}),
        "positives": sum(r["sample_type"] != "negative_user_confirmed" for r in rows),
        "negatives": sum(r["sample_type"] == "negative_user_confirmed" for r in rows),
        "boxes": sum(int(r["box_count"]) for r in rows),
        "machines": dict(sorted(Counter(r["machine"] for r in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()

    project = args.project.resolve()
    curated = project / "processed" / "curated"
    out = project / "training" / "cv"
    manifests = curated / "manifests"
    dev = read_csv(manifests / "train.csv") + read_csv(manifests / "val.csv")
    test = read_csv(manifests / "test.csv")
    assignment = assign_groups(dev, args.folds, args.seed)

    for row in dev:
        row["fold"] = str(assignment[row["session_id"]] + 1)
    fields = list(dev[0].keys())
    out.mkdir(parents=True, exist_ok=True)
    with (out / "fold_assignments.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(dev, key=lambda r: (int(r["fold"]), r["session_id"], r["timestamp"])))

    def image_path(row: dict[str, str]) -> str:
        return str((curated / row["curated_image"]).resolve())

    summary = {
        "seed": args.seed,
        "folds": args.folds,
        "group_key": "session_id",
        "development": summarize(dev),
        "fixed_test": summarize(test),
        "fold_details": {},
    }
    test_paths = [image_path(r) for r in test]
    write_lines(out / "test.txt", test_paths)

    for fold in range(args.folds):
        val_rows = [r for r in dev if assignment[r["session_id"]] == fold]
        train_rows = [r for r in dev if assignment[r["session_id"]] != fold]
        fold_dir = out / f"fold_{fold + 1}"
        write_lines(fold_dir / "train.txt", [image_path(r) for r in train_rows])
        write_lines(fold_dir / "val.txt", [image_path(r) for r in val_rows])
        if {r["session_id"] for r in train_rows} & {r["session_id"] for r in val_rows}:
            raise RuntimeError(f"Session leakage detected in fold {fold + 1}")
        summary["fold_details"][str(fold + 1)] = {
            "train": summarize(train_rows),
            "val": summarize(val_rows),
        }

    write_lines(out / "integrated" / "train.txt", [image_path(r) for r in dev])
    write_lines(out / "integrated" / "test.txt", test_paths)
    for machine in ("1", "2", "3"):
        machine_dev = [r for r in dev if r["machine"] == machine]
        machine_test = [r for r in test if r["machine"] == machine]
        write_lines(out / "machines" / f"machine_{machine}" / "train.txt", [image_path(r) for r in machine_dev])
        write_lines(out / "machines" / f"machine_{machine}" / "test.txt", [image_path(r) for r in machine_test])
        summary.setdefault("machine_details", {})[machine] = {
            "train": summarize(machine_dev),
            "test": summarize(machine_test),
        }

    (out / "fold_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
