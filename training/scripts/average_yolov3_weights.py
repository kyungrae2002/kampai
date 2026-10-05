#!/usr/bin/env python3
"""Weighted checkpoint averaging with optional BatchNorm recalibration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

np.int = int  # type: ignore[attr-defined]
_numpy_save = np.save


def _legacy_compatible_numpy_save(file, array, *args, **kwargs):
    if isinstance(array, list):
        array = np.asarray(array, dtype=object)
    return _numpy_save(file, array, *args, **kwargs)


np.save = _legacy_compatible_numpy_save  # type: ignore[assignment]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def clear_legacy_cache(list_path: Path) -> None:
    lines = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return
    image = Path(lines[0])
    parts = list(image.parts)
    if "images" not in parts:
        return
    parts[parts.index("images")] = "labels"
    cache = Path(str(Path(*parts).with_suffix(".txt").parent) + ".npy")
    if cache.exists():
        cache.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs=3, type=Path, required=True)
    parser.add_argument("--weights", nargs=3, type=float, required=True)
    parser.add_argument("--calibration-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--skip-bn-calibration", action="store_true")
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    sys.path.insert(0, str(args.repo))
    from models import Darknet
    from utils.datasets import LoadImagesAndLabels

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    device = torch.device(args.device)
    raw = [torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoints]
    states = [item["model"] for item in raw]
    normalized = np.asarray(args.weights, dtype=float)
    normalized /= normalized.sum()
    keys = list(states[0])
    if any(list(state) != keys for state in states[1:]):
        raise ValueError("Checkpoint keys differ; averaging is unsafe")
    averaged = {}
    for key in keys:
        values = [state[key] for state in states]
        if values[0].dtype.is_floating_point:
            tensor = torch.zeros_like(values[0], dtype=torch.float32)
            for weight, value in zip(normalized, values):
                tensor += float(weight) * value.float()
            averaged[key] = tensor.to(values[0].dtype)
        else:
            averaged[key] = torch.stack(values).max(dim=0).values

    model = Darknet(str(args.cfg), args.img_size)
    model.load_state_dict(averaged, strict=True)
    model.to(device)
    bn_layers = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    calibration_images = 0
    if not args.skip_bn_calibration:
        clear_legacy_cache(args.calibration_list)
        dataset = LoadImagesAndLabels(str(args.calibration_list), args.img_size, args.batch_size, augment=False, rect=False, single_cls=True)
        clear_legacy_cache(args.calibration_list)
        loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False, num_workers=0, pin_memory=False, collate_fn=dataset.collate_fn)
        for bn in bn_layers:
            bn.reset_running_stats()
            bn.momentum = None
        model.train()
        with torch.no_grad():
            for images, _, _, _ in loader:
                model(images.to(device).float() / 255.0)
        calibration_images = len(dataset)
    model.eval()
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method": "image_count_weighted_parameter_average" + ("" if args.skip_bn_calibration else "_then_bn_recalibration"),
        "checkpoints": [str(p.resolve()) for p in args.checkpoints],
        "raw_weights": args.weights,
        "normalized_weights": normalized.tolist(),
        "calibration_images": calibration_images,
        "bn_layers_recalibrated": 0 if args.skip_bn_calibration else len(bn_layers),
    }
    torch.save({"epoch": None, "model": cpu_state, "ensemble": metadata}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
