#!/usr/bin/env python3
"""Create a deduplicated, session-safe, machine-stratified dataset split.

This script never edits or deletes raw/processed audit files.  Exact duplicate
copies are excluded from the curated dataset and recorded in a manifest.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def parse_timestamp(source_path: str) -> dt.datetime | None:
    match = re.search(r"_(20\d{6})_(\d{6})", Path(source_path).name)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")


def load_rows(log_path: Path) -> list[dict[str, str]]:
    with log_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["timestamp"] = parse_timestamp(row["source_path"]).isoformat() if parse_timestamp(row["source_path"]) else ""
        row["box_count"] = row.get("label_count", "0") or "0"
        row["sample_type"] = "negative_user_confirmed" if int(row["box_count"]) == 0 else "positive_weak_label"
    return rows


def deduplicate(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    seen: dict[str, dict[str, str]] = {}
    unique: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    for row in rows:
        digest = row["sha256"]
        if digest not in seen:
            seen[digest] = row
            row["kept_output_id"] = row["output_id"]
            unique.append(row)
        else:
            duplicate = dict(row)
            duplicate["kept_output_id"] = seen[digest]["output_id"]
            duplicates.append(duplicate)
    return unique, duplicates


def build_sessions(rows: list[dict[str, str]], gap_seconds: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        timestamp = parse_timestamp(row["source_path"])
        date_key = timestamp.strftime("%Y%m%d") if timestamp else f"unknown-{row['output_id']}"
        grouped[(row["machine"], date_key)].append(row)

    sessions: list[dict[str, object]] = []
    for (machine, date_key), group_rows in sorted(grouped.items()):
        group_rows.sort(key=lambda row: (parse_timestamp(row["source_path"]) or dt.datetime.min, row["output_id"]))
        current: list[dict[str, str]] = []
        previous: dt.datetime | None = None
        session_index = 0

        def flush() -> None:
            nonlocal current, session_index
            if not current:
                return
            session_index += 1
            session_id = f"m{machine}_{date_key}_s{session_index:04d}"
            for member in current:
                member["session_id"] = session_id
            timestamps = [parse_timestamp(member["source_path"]) for member in current]
            timestamps = [value for value in timestamps if value is not None]
            sessions.append(
                {
                    "session_id": session_id,
                    "machine": machine,
                    "date": date_key,
                    "start": min(timestamps).isoformat() if timestamps else "",
                    "end": max(timestamps).isoformat() if timestamps else "",
                    "images": len(current),
                    "boxes": sum(int(member["box_count"]) for member in current),
                    "negatives": sum(member["sample_type"] == "negative_user_confirmed" for member in current),
                    "rows": current,
                }
            )
            current = []

        for row in group_rows:
            timestamp = parse_timestamp(row["source_path"])
            if current and previous is not None and timestamp is not None:
                if (timestamp - previous).total_seconds() > gap_seconds:
                    flush()
            current.append(row)
            previous = timestamp
        flush()
    return sessions


def session_features(session: dict[str, object]) -> dict[str, float]:
    return {
        "images": float(session["images"]),
        "boxes": float(session["boxes"]),
        "negatives": float(session["negatives"]),
        "sessions": 1.0,
    }


def assign_splits(sessions: list[dict[str, object]], seed: int) -> None:
    by_machine: dict[str, list[dict[str, object]]] = defaultdict(list)
    for session in sessions:
        by_machine[str(session["machine"])].append(session)

    keys = ("images", "boxes", "negatives")
    global_totals = {
        key: sum(float(session[key]) for session in sessions)
        for key in keys
    }
    machine_totals = {
        machine: {
            key: sum(float(session[key]) for session in machine_sessions)
            for key in keys
        }
        for machine, machine_sessions in by_machine.items()
    }

    best_assignment: dict[str, str] | None = None
    best_score: float | None = None

    # Session sizes and box density vary substantially. A deterministic random
    # search gives a much closer 70/15/15 balance than image-level splitting
    # while still keeping every capture session intact.
    for trial in range(20000):
        rng = random.Random(seed + trial)
        assignment: dict[str, str] = {}
        global_state = {split: defaultdict(float) for split in SPLITS}
        machine_state = {
            machine: {split: defaultdict(float) for split in SPLITS}
            for machine in by_machine
        }

        for machine, machine_sessions in sorted(by_machine.items()):
            ordered = list(machine_sessions)
            rng.shuffle(ordered)
            count = len(ordered)
            train_end = round(count * RATIOS["train"])
            val_end = train_end + round(count * RATIOS["val"])
            for index, session in enumerate(ordered):
                split = "train" if index < train_end else ("val" if index < val_end else "test")
                assignment[str(session["session_id"])] = split
                for key in keys:
                    value = float(session[key])
                    global_state[split][key] += value
                    machine_state[machine][split][key] += value

        score = 0.0
        for split in SPLITS:
            for key in keys:
                target = global_totals[key] * RATIOS[split]
                score += ((global_state[split][key] - target) / max(target, 1.0)) ** 2
        for machine in by_machine:
            for split in SPLITS:
                for key in ("images", "boxes"):
                    target = machine_totals[machine][key] * RATIOS[split]
                    score += 0.25 * ((machine_state[machine][split][key] - target) / max(target, 1.0)) ** 2

        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment

    assert best_assignment is not None
    for session in sessions:
        session["split"] = best_assignment[str(session["session_id"])]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--session-gap-seconds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()

    processed = args.processed.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output}")

    rows = load_rows(processed / "processing_log.csv")
    unique, duplicates = deduplicate(rows)
    sessions = build_sessions(unique, args.session_gap_seconds)
    assign_splits(sessions, args.seed)

    session_lookup = {str(session["session_id"]): str(session["split"]) for session in sessions}
    for row in unique:
        row["split"] = session_lookup[row["session_id"]]

    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in unique:
        split = row["split"]
        source_image = processed / row["clean_image"]
        source_label = processed / row["label"]
        image_destination = output / "images" / split / source_image.name
        label_destination = output / "labels" / split / source_label.name
        shutil.copy2(source_image, image_destination)
        shutil.copy2(source_label, label_destination)
        row["curated_image"] = str(image_destination.relative_to(output))
        row["curated_label"] = str(label_destination.relative_to(output))

    manifest_fields = [
        "output_id", "split", "session_id", "machine", "timestamp", "sample_type",
        "box_count", "sha256", "source_path", "source_relative_path", "clean_image",
        "label", "curated_image", "curated_label", "status", "warnings",
    ]
    write_csv(output / "manifests" / "all_unique.csv", unique, manifest_fields)
    for split in SPLITS:
        write_csv(
            output / "manifests" / f"{split}.csv",
            [row for row in unique if row["split"] == split],
            manifest_fields,
        )
    write_csv(
        output / "manifests" / "negative_user_confirmed.csv",
        [row for row in unique if row["sample_type"] == "negative_user_confirmed"],
        manifest_fields,
    )
    duplicate_fields = [
        "output_id", "kept_output_id", "sha256", "source_path", "source_relative_path",
        "clean_image", "label", "box_count", "status", "warnings",
    ]
    write_csv(output / "manifests" / "duplicates_excluded.csv", duplicates, duplicate_fields)

    session_rows: list[dict[str, object]] = []
    for session in sessions:
        session_rows.append({key: value for key, value in session.items() if key != "rows"})
    write_csv(
        output / "manifests" / "sessions.csv",
        session_rows,
        ["session_id", "split", "machine", "date", "start", "end", "images", "boxes", "negatives"],
    )

    split_summary: dict[str, object] = {}
    for split in SPLITS:
        subset = [row for row in unique if row["split"] == split]
        split_summary[split] = {
            "images": len(subset),
            "boxes": sum(int(row["box_count"]) for row in subset),
            "positives": sum(row["sample_type"] == "positive_weak_label" for row in subset),
            "negatives": sum(row["sample_type"] == "negative_user_confirmed" for row in subset),
            "sessions": sum(session["split"] == split for session in sessions),
            "machines": {
                machine: sum(row["machine"] == machine for row in subset)
                for machine in sorted({row["machine"] for row in unique})
            },
        }

    summary = {
        "training_status": "BLOCKED_PENDING_USER_APPROVAL",
        "source_images": len(rows),
        "unique_images": len(unique),
        "duplicate_copies_excluded": len(duplicates),
        "positive_weak_label_images": sum(row["sample_type"] == "positive_weak_label" for row in unique),
        "negative_user_confirmed_images": sum(row["sample_type"] == "negative_user_confirmed" for row in unique),
        "boxes": sum(int(row["box_count"]) for row in unique),
        "session_gap_seconds": args.session_gap_seconds,
        "sessions": len(sessions),
        "seed": args.seed,
        "ratios": RATIOS,
        "splits": split_summary,
        "notes": [
            "No training or fine-tuning was run.",
            "Empty labels are included as user-confirmed negative samples.",
            "Color-box-derived labels remain weak labels and require visual QA before training approval.",
        ],
    }
    (output / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "dataset.yaml").write_text(
        "# TRAINING BLOCKED: wait for explicit user approval.\n"
        f"path: {output}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n  0: defect\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Curated dataset\n\n"
        "Training is blocked pending explicit user approval.\n\n"
        f"Exact duplicates were excluded. Sessions use detector/machine, capture date, and a {args.session_gap_seconds}-second frame-gap boundary. "
        "All frames in one session remain in the same split. Empty labels are user-confirmed negatives.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
