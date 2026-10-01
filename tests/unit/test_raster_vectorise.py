"""Raster vectoriser building blocks: skeleton paths and junctions, straight/arc runs, refitted
corners, and the photo rectifier."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from skimage.morphology import skeletonize

from archrender.plan.raster import skeleton_paths, split_runs
from archrender.understand.rectify import find_sheet, rectify

KW = {"eps_px": 1.0, "max_chord_px": 400.0, "min_radius_px": 20.0, "max_radius_px": 5000.0}


def _skel(img: np.ndarray) -> np.ndarray:
    return np.asarray(skeletonize(img > 0), bool)


def test_a_crossing_splits_paths_at_a_shared_junction() -> None:
    img = np.zeros((200, 200), np.uint8)
    cv2.line(img, (20, 100), (180, 100), 255, 3)
    cv2.line(img, (100, 20), (100, 180), 255, 3)
    paths = skeleton_paths(_skel(img))
    assert len(paths) == 4
    ends = [tuple(np.round(p[k]).astype(int)) for p in paths for k in (0, -1)]
    centre = [e for e in ends if abs(e[0] - 100) <= 2 and abs(e[1] - 100) <= 2]
    assert len(centre) == 4  # every arm ends at the junction centroid
    assert len(set(centre)) == 1


def test_an_arc_cut_by_a_circle_is_split_at_both_crossings() -> None:
    img = np.zeros((400, 400), np.uint8)
    cv2.ellipse(img, (100, 300), (200, 200), 0, -90, 0, 255, 2)  # door swing, r = 200 px
    cv2.circle(img, (241, 160), 30, 255, 2)  # a tag bubble drawn across it
    paths = skeleton_paths(_skel(img))
    long_arcs = [p for p in paths if len(p) > 40 and np.hypot(*(p[0] - p[-1])) > 40]
    assert len(long_arcs) >= 2  # the swing on both sides of the bubble


def test_split_runs_finds_a_quantised_arc_and_its_radius() -> None:
    t = np.radians(np.linspace(10, 80, 300))
    pts = np.round(np.column_stack([500 + 260 * np.cos(t), 500 + 260 * np.sin(t)])) + 0.5
    runs = split_runs(pts, closed=False, **KW)
    arcs = [r for r in runs if r.curve]
    assert len(arcs) == 1 and arcs[0].circle is not None
    assert arcs[0].circle[2] == pytest.approx(260, abs=1.5)
    assert arcs[0].sweep_deg == pytest.approx(70, abs=2)


def test_split_runs_keeps_a_bumpy_drawn_line_straight() -> None:
    img = np.zeros((300, 500), np.uint8)
    cv2.line(img, (20, 100), (480, 160), 255, 3)
    for x in (120, 250, 380):  # pixel bumps on its edge (scan noise, junction blobs)
        cv2.circle(img, (x, int(100 + 60 * (x - 20) / 460) + 2), 2, 255, -1)
    paths = skeleton_paths(_skel(img))
    longest = max(paths, key=len)
    runs = split_runs(longest, closed=False, **KW)
    assert len(runs) == 1 and not runs[0].curve
    a, b = runs[0].pts[0], runs[0].pts[-1]
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180  # either direction
    assert angle == pytest.approx(math.degrees(math.atan2(60, 460)), abs=0.3)


def test_corners_are_refitted_through_chamfers_and_bumps() -> None:
    """A rotated rectangle with a chamfered corner and a pixel bump on one side: the refitted
    polygon has four corners within half a pixel of the true ones."""
    img = np.zeros((600, 600), np.uint8)
    rect = ((300.0, 300.0), (400.0, 120.0), 17.0)
    box = cv2.boxPoints(rect)
    cv2.fillPoly(img, [np.round(box).astype(np.int32)], 255)
    corner = np.round(box[0]).astype(int)
    cv2.circle(img, tuple(int(v) for v in corner), 3, 0, -1)  # chamfer
    mid = np.round((box[1] + box[2]) / 2).astype(int)
    cv2.circle(img, tuple(int(v) for v in mid), 2, 255, -1)  # bump
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = contours[0].reshape(-1, 2).astype(np.float64) + 0.5
    runs = split_runs(pts, closed=True, **{**KW, "eps_px": 2.5})
    verts = np.array([r.pts[0] for r in runs if np.hypot(*(r.pts[-1] - r.pts[0])) > 1e-6])
    assert len(verts) == 4
    # contour points are pixel centres: the true edge is half a pixel further out
    for b in box:
        d = np.hypot(*(verts - b).T).min()
        assert d <= 1.5


def _photo(rng: np.random.Generator, crop: bool) -> tuple[np.ndarray, np.ndarray]:
    h, w = 1500, 2000
    img = np.full((h, w, 3), (120, 95, 70), np.uint8)
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    quad = np.array([[420, 260], [1650, 300], [1600, 1180], [380, 1120]], np.float64)
    if crop:
        quad[1] += (450, 0)  # the top-right corner outside the frame
    page = np.full((1000, 1414, 3), 245, np.uint8)
    cv2.putText(page, "PLAN", (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 6, (20, 20, 20), 12)
    hm = cv2.getPerspectiveTransform(
        np.array([[0, 0], [1414, 0], [1414, 1000], [0, 1000]], np.float32), quad.astype(np.float32)
    )
    warped = cv2.warpPerspective(page, hm, (w, h))
    mask = cv2.warpPerspective(np.ones((1000, 1414), np.uint8), hm, (w, h)) > 0
    img[mask] = warped[mask]
    return cv2.GaussianBlur(img, (0, 0), 0.8), quad


@pytest.mark.parametrize("crop", [False, True])
def test_the_sheet_in_a_photo_is_found_to_a_pixel_and_rectified(crop: bool) -> None:
    rgb, quad = _photo(np.random.default_rng(5), crop)
    found = find_sheet(rgb)
    assert found is not None
    assert np.hypot(*(found.corners - quad).T).max() <= 2.0
    assert found.corners_in_frame is (not crop)
    r = rectify(rgb, found)
    assert r.aspect_source == "iso216"
    hh, ww = r.image.shape[:2]
    assert ww / hh == pytest.approx(math.sqrt(2), rel=0.01)


def test_a_scan_filling_the_frame_is_not_a_photo() -> None:
    img = np.full((800, 1131, 3), 240, np.uint8)
    cv2.line(img, (100, 100), (900, 700), (0, 0, 0), 3)
    assert find_sheet(img) is None
