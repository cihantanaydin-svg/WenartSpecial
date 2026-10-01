"""Train the plan segmentation model on synthetic sheets (ADR-M06). UNVERIFIED-ON-GPU.

    python -m archrender.plan.seg_train OUT_DIR --sheets 400 --epochs 30 --device cuda
    python -m archrender.plan.seg_train OUT_DIR --smoke        # 2 sheets, 1 epoch, CPU

Needs PyTorch (the ``gpu`` extra; it is in the pod image, not in the CPU test environment). The
network is a small U-Net in plain PyTorch (no extra dependency). Data come only from the synthetic
generator (``plan.seg_data``; owner answer Q-4). Writes:

- ``OUT_DIR/plan_seg.pt``: the TorchScript model (RGB float tile in [0, 1] → class logits);
- ``OUT_DIR/plan_seg.json``: classes, tile size, the training and held-out seeds, per-class IoU on
  the held-out sheets and the training log.

The weights are the firm's own (licence ``proprietary-firm``). A checkpoint is registered for the
``plan_segmenter`` role and replaces the CV wall body (``raster_seg``) only once ``make eval`` shows
it beats the CV baseline on held-out sheets and meets the clean-raster targets (promotion rule).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from archrender.plan.seg_data import CLASSES, class_frequencies, sheet_sample, tiles

VAL_SEED0 = 100_000  # held-out sheets never overlap the training seeds


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "PyTorch is not installed: train on the pod image (or `uv sync --extra gpu`)."
        ) from e
    return torch


def build_unet(n_classes: int = len(CLASSES), width: int = 32) -> Any:
    torch = _torch()
    nn = torch.nn

    def block(cin: int, cout: int) -> Any:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            w = width
            self.enc = nn.ModuleList(
                [block(3, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w)]
            )
            self.mid = block(8 * w, 16 * w)
            self.up = nn.ModuleList(
                [nn.ConvTranspose2d(c * 2, c, 2, stride=2) for c in (8 * w, 4 * w, 2 * w, w)]
            )
            self.dec = nn.ModuleList([block(c * 2, c) for c in (8 * w, 4 * w, 2 * w, w)])
            self.head = nn.Conv2d(w, n_classes, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x: Any) -> Any:
            skips = []
            for enc in self.enc:
                x = enc(x)
                skips.append(x)
                x = self.pool(x)
            x = self.mid(x)
            for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
                x = dec(torch.cat([up(x), skip], dim=1))
            return self.head(x)

    return UNet()


def _batches(seeds: list[int], tile: int, batch: int, rng: np.random.Generator) -> Any:
    """Tiles of the given sheets in random order, with flips / quarter turns (labels follow)."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for seed in seeds:
        for t in tiles(sheet_sample(seed), tile):
            k = int(rng.integers(4))
            img, lab = np.rot90(t.image, k), np.rot90(t.label, k)
            if rng.random() < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
            xs.append(np.ascontiguousarray(img))
            ys.append(np.ascontiguousarray(lab))
            if len(xs) == batch:
                yield np.stack(xs), np.stack(ys)
                xs, ys = [], []
    if xs:
        yield np.stack(xs), np.stack(ys)


def iou(pred: np.ndarray, label: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k, name in enumerate(CLASSES):
        inter = float(np.sum((pred == k) & (label == k)))
        union = float(np.sum((pred == k) | (label == k)))
        out[name] = round(inter / union, 4) if union else None
    return out


def train(
    out: Path,
    *,
    sheets: int,
    epochs: int,
    val_sheets: int,
    tile: int = 512,
    batch: int = 8,
    width: int = 32,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_seeds = list(range(seed * 10_000, seed * 10_000 + sheets))
    val_seeds = list(range(VAL_SEED0, VAL_SEED0 + val_sheets))
    freq = np.mean(
        [class_frequencies(sheet_sample(s).label) for s in train_seeds[: min(8, sheets)]], axis=0
    )
    weight = torch.tensor(1.0 / np.sqrt(np.maximum(freq, 1e-4)), dtype=torch.float32, device=device)
    model = build_unet(width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight / weight.mean())
    log = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        order = list(rng.permutation(train_seeds))
        losses = []
        for x, y in _batches(order, tile, batch, rng):
            xb = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255).to(device)
            yb = torch.from_numpy(y).long().to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        log.append(
            {
                "epoch": epoch,
                "loss": round(float(np.mean(losses)), 4),
                "s": round(time.time() - t0, 1),
            }
        )
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in _batches(val_seeds, tile, batch, np.random.default_rng(1)):
            xb = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
            labels.append(y)
    metrics = iou(np.concatenate(preds), np.concatenate(labels)) if preds else {}
    out.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, tile, tile, device=device)
    torch.jit.trace(model, example).save(str(out / "plan_seg.pt"))
    info = {
        "classes": list(CLASSES),
        "tile": tile,
        "width": width,
        "train_seeds": [train_seeds[0], train_seeds[-1]] if train_seeds else [],
        "val_seeds": [val_seeds[0], val_seeds[-1]] if val_seeds else [],
        "epochs": epochs,
        "val_iou": metrics,
        "log": log,
        "data": "synthetic only (archrender.plan.seg_data; owner answer Q-4)",
        "license": "proprietary-firm",
    }
    (out / "plan_seg.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--sheets", type=int, default=400)
    ap.add_argument("--val-sheets", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="2 sheets, 1 epoch, small tiles, CPU")
    a = ap.parse_args(argv)
    if a.smoke:
        info = train(
            a.out, sheets=2, epochs=1, val_sheets=1, tile=256, batch=4, width=8, device="cpu"
        )
    else:
        info = train(
            a.out,
            sheets=a.sheets,
            epochs=a.epochs,
            val_sheets=a.val_sheets,
            tile=a.tile,
            batch=a.batch,
            width=a.width,
            device=a.device,
            seed=a.seed,
        )
    print(json.dumps({"val_iou": info["val_iou"], "log": info["log"][-1:]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
