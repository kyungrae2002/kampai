#!/usr/bin/env python3
"""YOLOv3-SPP trainer adapted for Apple Metal and modern PyTorch."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# Compatibility with the 2020 YOLOv3 data loader.
np.int = int  # type: ignore[attr-defined]
_numpy_save = np.save


def _legacy_compatible_numpy_save(file, array, *args, **kwargs):
    """NumPy 2.x no longer coerces ragged label lists to object arrays."""
    if isinstance(array, list):
        array = np.asarray(array, dtype=object)
    return _numpy_save(file, array, *args, **kwargs)


np.save = _legacy_compatible_numpy_save  # type: ignore[assignment]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


HYP = {
    "giou": 3.54,
    "cls": 37.4 / 80.0,
    "cls_pw": 1.0,
    "obj": 64.3,
    "obj_pw": 1.0,
    "iou_t": 0.20,
    "weight_decay": 5e-4,
    "hsv_h": 0.0138,
    "hsv_s": 0.35,
    "hsv_v": 0.20,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "fl_gamma": 0.0,
}


def choose_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable. Run outside the filesystem sandbox on macOS.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("--device must be mps or cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def legacy_cache_path(list_path: Path) -> Path | None:
    lines = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    image = Path(lines[0])
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return None
    parts[index] = "labels"
    label = Path(*parts).with_suffix(".txt")
    return Path(str(label.parent) + ".npy")


def clear_legacy_cache(list_path: Path) -> None:
    cache = legacy_cache_path(list_path)
    if cache and cache.exists():
        cache.unlink()


def bbox_iou_local(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    # xywh boxes, both shaped Nx4.
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    inter = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(0) * (
        torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)
    ).clamp(0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - inter + eps
    iou = inter / union
    cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)
    return iou - (cw * ch - union) / (cw * ch + eps)


def wh_iou_local(wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
    wh1 = wh1[:, None]
    wh2 = wh2[None]
    inter = torch.minimum(wh1, wh2).prod(2)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + 1e-9)


def build_targets(predictions, targets: torch.Tensor, model):
    nt = targets.shape[0]
    tcls, tbox, indices, anchors_out = [], [], [], []
    device = targets.device
    gain = torch.ones(6, device=device)
    for layer_idx, yolo_idx in enumerate(model.yolo_layers):
        anchors = model.module_list[yolo_idx].anchor_vec.to(device)
        gain[2:] = torch.tensor(
            [predictions[layer_idx].shape[3], predictions[layer_idx].shape[2],
             predictions[layer_idx].shape[3], predictions[layer_idx].shape[2]],
            device=device,
            dtype=gain.dtype,
        )
        na = anchors.shape[0]
        anchor_index = torch.arange(na, device=device).view(na, 1).repeat(1, nt)
        matched = targets * gain
        if nt:
            mask = wh_iou_local(anchors, matched[:, 4:6]) > model.hyp["iou_t"]
            a = anchor_index[mask]
            matched = matched.repeat(na, 1, 1)[mask]
        else:
            a = torch.empty(0, dtype=torch.long, device=device)
            matched = targets
        b, c = matched[:, :2].long().T
        gxy, gwh = matched[:, 2:4], matched[:, 4:6]
        gij = gxy.long()
        gi, gj = gij.T
        indices.append((b, a.long(), gj, gi))
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        anchors_out.append(anchors[a.long()])
        tcls.append(c)
    return tcls, tbox, indices, anchors_out


def compute_loss(predictions, targets: torch.Tensor, model):
    device = predictions[0].device
    lcls = torch.zeros(1, device=device)
    lbox = torch.zeros(1, device=device)
    lobj = torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(predictions, targets, model)
    h = model.hyp
    bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
    bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))
    for idx, prediction in enumerate(predictions):
        b, a, gj, gi = indices[idx]
        tobj = torch.zeros_like(prediction[..., 0])
        if b.numel():
            selected = prediction[b, a, gj, gi]
            pxy = selected[:, :2].sigmoid()
            pwh = selected[:, 2:4].exp().clamp(max=1e3) * anchors[idx]
            giou = bbox_iou_local(torch.cat((pxy, pwh), 1), tbox[idx])
            lbox += (1.0 - giou).mean()
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).to(tobj.dtype)
            if model.nc > 1:
                class_target = torch.zeros_like(selected[:, 5:])
                class_target[torch.arange(b.shape[0], device=device), tcls[idx]] = 1.0
                lcls += bce_cls(selected[:, 5:], class_target)
        lobj += bce_obj(prediction[..., 4], tobj)
    lbox *= h["giou"]
    lobj *= h["obj"]
    lcls *= h["cls"]
    total = lbox + lobj + lcls
    return total, torch.cat((lbox, lobj, lcls, total)).detach()


def cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def validate_loss(model, loader, device) -> tuple[float, list[float]]:
    model.eval()
    sums = torch.zeros(4, dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for images, targets, _, _ in loader:
            images = images.to(device).float() / 255.0
            targets = targets.to(device)
            _, train_output = model(images)
            _, items = compute_loss(train_output, targets, model)
            sums += items.cpu().double()
            count += 1
    means = (sums / max(count, 1)).tolist()
    return means[-1], means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path)
    parser.add_argument("--init-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    sys.path.insert(0, str(args.repo))
    from models import Darknet, load_darknet_weights
    from utils.datasets import LoadImagesAndLabels

    seed_everything(args.seed)
    device = choose_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config["torch_version"] = torch.__version__
    config["device_resolved"] = str(device)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = Darknet(str(args.cfg), args.img_size).to(device)
    model.nc = 1
    model.hyp = HYP.copy()
    model.gr = 1.0
    if args.init_weights.suffix == ".pt":
        checkpoint = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        load_darknet_weights(model, str(args.init_weights))

    clear_legacy_cache(args.train_list)
    train_set = LoadImagesAndLabels(
        str(args.train_list), args.img_size, args.batch_size, augment=True, hyp=HYP, rect=False, single_cls=True
    )
    clear_legacy_cache(args.train_list)
    train_loader = DataLoader(
        train_set, batch_size=min(args.batch_size, len(train_set)), shuffle=True,
        num_workers=args.workers, pin_memory=False, collate_fn=train_set.collate_fn,
    )
    val_loader = None
    if args.val_list:
        clear_legacy_cache(args.val_list)
        val_set = LoadImagesAndLabels(
            str(args.val_list), args.img_size, args.batch_size, augment=False, hyp=HYP, rect=True, single_cls=True
        )
        clear_legacy_cache(args.val_list)
        val_loader = DataLoader(
            val_set, batch_size=min(args.batch_size, len(val_set)), shuffle=False,
            num_workers=args.workers, pin_memory=False, collate_fn=val_set.collate_fn,
        )

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": HYP["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    history = []
    best_metric = math.inf
    best_epoch = -1
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = torch.zeros(4, dtype=torch.float64)
        optimizer.zero_grad(set_to_none=True)
        batches = len(train_loader)
        for batch_idx, (images, targets, _, _) in enumerate(train_loader):
            images = images.to(device).float() / 255.0
            targets = targets.to(device)
            model.gr = min(1.0, (epoch * batches + batch_idx + 1) / max(batches, 1))
            predictions = model(images)
            loss, items = compute_loss(predictions, targets, model)
            (loss / args.accumulate).backward()
            if (batch_idx + 1) % args.accumulate == 0 or batch_idx + 1 == batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += items.cpu().double()
        scheduler.step()
        train_items = (running / max(batches, 1)).tolist()
        if val_loader is not None:
            metric, val_items = validate_loss(model, val_loader, device)
        else:
            metric, val_items = train_items[-1], None
        record = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train": {"box": train_items[0], "object": train_items[1], "class": train_items[2], "total": train_items[3]},
            "val": None if val_items is None else {"box": val_items[0], "object": val_items[1], "class": val_items[2], "total": val_items[3]},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if metric < best_metric:
            best_metric, best_epoch = metric, epoch + 1
            torch.save({"epoch": epoch + 1, "model": cpu_state_dict(model), "config": config}, args.output / "best.pt")
        print(json.dumps(record), flush=True)

    torch.save({"epoch": args.epochs, "model": cpu_state_dict(model), "config": config}, args.output / "last.pt")
    summary = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "epochs": args.epochs,
        "elapsed_seconds": time.time() - started,
        "images": len(train_set),
        "device": str(device),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
