"""Plan segmentation (ADR-M06): synthetic training data, the ``raster_seg`` hook, training smoke."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from archrender.core.schemas.document import Word
from archrender.plan.builder import build_plan
from archrender.plan.metrics import compose, plan_scores, transform_plan
from archrender.plan.raster import raster_extract
from archrender.plan.seg_data import (
    CLASSES,
    DOOR,
    ROOM,
    WALL,
    WINDOW,
    class_frequencies,
    label_map,
    sheet_sample,
    tiles,
)
from archrender.synth.plan import gt_plan, random_spec


def test_labels_follow_the_ground_truth_plan() -> None:
    plan = gt_plan(random_spec(np.random.default_rng(3), variant="manhattan"))
    to_px = [[100.0, 0.0, 50.0], [0.0, -100.0, 1500.0]]  # 1 cm per px, y down (a page)
    lab = label_map(plan, to_px, (1600, 1600))
    k = 0.0001  # m² per px²
    wall_area = sum(
        w.centerline.length() * w.thickness_m.value  # type: ignore[union-attr]
        for w in plan.walls
    )
    opening_area = sum(
        o.width_m.value * next(w for w in plan.walls if w.id == o.host_wall).thickness_m.value
        for o in plan.openings
    )
    walls_px = float((lab == WALL).sum()) * k
    assert walls_px == pytest.approx(wall_area - opening_area, rel=0.08)
    rooms_px = float((lab == ROOM).sum()) * k
    from shapely.geometry import Polygon

    room_area = sum(Polygon([(p.x, p.y) for p in r.polygon]).area for r in plan.rooms)
    assert rooms_px == pytest.approx(room_area, rel=0.03)
    assert (lab == DOOR).any() and (lab == WINDOW).any()


def test_sheet_samples_and_tiles() -> None:
    s = sheet_sample(5, dpi=150.0, quality="noisy")
    assert s.image.shape[:2] == s.label.shape and s.image.dtype == np.uint8
    freq = class_frequencies(s.label)
    assert len(freq) == len(CLASSES) and freq[WALL] > 0.005 and freq[ROOM] > 0.02
    ts = list(tiles(s, 256))
    assert ts and all(t.image.shape == (256, 256, 3) and t.label.any() for t in ts)
    # deterministic per seed
    again = sheet_sample(5, dpi=150.0, quality="noisy")
    assert np.array_equal(again.label, s.label) and np.array_equal(again.image, s.image)


def test_a_learned_wall_mask_drives_the_raster_extractor() -> None:
    """``raster_seg`` with a perfect mask (the ceiling of the learned path): a noisy hatch scan with
    an approval stamp, where the CV body merges two rooms, extracts exactly."""
    import io

    from PIL import Image

    from archrender.synth.raster import scan
    from archrender.synth.sheets import floor_plan_page

    rng = np.random.default_rng(702)
    spec = random_spec(rng, variant="skewed")
    sheet = floor_plan_page(rng, spec=spec, wall_style="hatch")
    data, _, gt = scan(sheet.pdf, sheet.gt, rng, dpi=300.0, quality="noisy")
    rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    words = [
        Word(
            text=w["text"],
            x0=w["bbox"][0],
            y0=w["bbox"][1],
            x1=w["bbox"][2],
            y1=w["bbox"][3],
            angle_deg=w.get("angle_deg", 0.0),
            source="ocr:ground-truth",
        )
        for w in gt["words"]
    ]
    m = gt["scale"] * 0.0254 / 300.0
    mask = label_map(gt_plan(spec), gt["plan_to_page_px"], gray.shape) == WALL
    ex = raster_extract(gray, words, m, wall_mask=mask)
    assert ex.prims.method == "raster_seg"
    res = build_plan(ex.prims, project="p", version="v", source="raster")
    truth = transform_plan(gt_plan(spec), compose(gt["plan_to_page_px"], ex.doc_to_plan))
    s = plan_scores(res.plan, truth)
    assert s["walls"]["f1"] >= 0.99 and s["rooms"]["f1"] == 1.0 and s["openings"]["f1"] >= 0.85, s
    assert res.plan.walls[0].thickness_m.provenance[0].method == "raster_seg"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs PyTorch (gpu extra)")
def test_training_smoke(tmp_path: Path) -> None:
    from archrender.plan.seg_train import main

    assert main([str(tmp_path), "--smoke"]) == 0
    assert (tmp_path / "plan_seg.pt").exists() and (tmp_path / "plan_seg.json").exists()
