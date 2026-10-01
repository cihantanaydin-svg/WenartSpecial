"""Raster plans (scans, rectified phone photos) → drawing primitives (ARCHITECTURE §S2 step 3,
the CV baseline; a segmentation checkpoint can replace step 3 when one is registered).

1. **Darkness.** The paper level is estimated at low resolution (a max filter wider than any
   wall) and divided out, so uneven light and paper tint do not matter: d ∈ [0, 1], paper ≈ 0.
2. **Ink.** Hysteresis threshold of d (thresholds above the measured noise). Coloured ink (stamps,
   signatures) is not drawing ink; connected components that are text (inside OCR word boxes) and
   isolated specks smaller than any building element are dropped.
3. **Wall body.** Poché walls (solid or grey) are the ink that survives a morphological opening at
   a fraction of the thinnest wall. Outline and hatched walls are the thin enclosed background
   regions (narrower than the thickest wall, without text inside) together with the strokes
   between them.
4. **Windows inside a filled outline.** A straight thin line running along the body at a constant
   distance from both faces is a glazing line (hatch lines cross the body instead): the body is
   cut there, so the builder sees the opening and its window lines.
5. **Primitives.** The body's contours become closed, filled outlines (one drawing object per
   connected body; circular runs are resampled on a common angular grid so concentric faces stay
   parallel); the skeleton of the thin ink becomes open polylines and arcs (door swings, window
   lines, dimension lines).

Everything is measured in page pixels and converted with the page's metres per pixel, so every
threshold that refers to a building element is in metres.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from archrender.core.schemas.document import Word
from archrender.core.schemas.provenance import Method
from archrender.plan.builder import MAX_T
from archrender.plan.prims import circle_fit
from archrender.understand.text import normalise_tag, parse_dimension

MIN_WALL_M = 0.08  # thinnest wall the opening kernel must keep
MIN_ELEMENT_M = 0.2  # isolated ink smaller than this is not a building element
GLAZING_MIN_M = 0.3
MAX_GAP_PX = 4000  # bound on walking along a glazing line
ARC_GRID_DEG = 2.0

# 8-neighbourhood in ring order (for crossing numbers) as (dy, dx)
_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_CROSS = np.array(
    [sum(((c >> k) & 1) != ((c >> ((k + 1) % 8)) & 1) for k in range(8)) // 2 for c in range(256)],
    np.uint8,
)


@dataclass
class RasterVectors:
    """Primitives in page pixels, in the PDF task's ``paths`` layout (``pdf_extract`` reads it)."""

    paths: dict[str, Any]
    stroke_px: float
    notes: list[str] = field(default_factory=list)
    debug: dict[str, NDArray[Any]] = field(default_factory=dict)
    body: NDArray[np.bool_] | None = None  # wall body mask (page px)
    ink: NDArray[np.bool_] | None = None  # drawing ink without text and specks (page px)


def _odd(v: float) -> int:
    return max(3, round(v) // 2 * 2 + 1)


def _disc(d: int) -> NDArray[np.uint8]:
    return np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)), np.uint8)


# ---------------------------------------------------------------------------------------------
# text that is text (page OCR also "reads" hatching and symbols)
# ---------------------------------------------------------------------------------------------
_LETTERS = re.compile(r"[A-Za-zÇĞİÖŞÜÂÎÛçğıöşüâîû]")
_UNITS = re.compile(r"^(m²|m2|m\^2|cm|mm)$", re.I)


def text_words(words: list[Word]) -> list[Word]:
    """Words that are drawing text: a confident reading that is a dimension, a tag, an area or a
    unit, or a word of letters. OCR turns hatching and symbols into words too ('////', 'SSS',
    '2222', tall narrow boxes along strokes); those must not mask walls or fill rooms."""
    hs = sorted(w.y1 - w.y0 for w in words if w.angle_deg in (0.0, 180.0) and w.confidence >= 0.8)
    med = hs[len(hs) // 2] if hs else None
    out = []
    for w in words:
        t = w.text.strip()
        if not t or w.confidence < 0.6:
            continue
        if len(t) > 1 and len(set(t)) == 1:
            continue
        if med is not None and (w.y1 - w.y0) > 2.5 * med and (w.x1 - w.x0) < 3 * med:
            continue
        letters = len(_LETTERS.findall(t))
        if (
            parse_dimension(t) is not None
            or normalise_tag(t) is not None
            or _UNITS.match(t)
            or (letters >= 2 and letters >= 0.7 * len(t))
            or (any(ch.isdigit() for ch in t) and sum(ch.isdigit() for ch in t) >= 0.6 * len(t))
        ):
            out.append(w)
    return out


def word_boxes(words: list[Word], pad: float = 0.0) -> list[NDArray[np.float64]]:
    """The words' boxes as oriented rectangles (4 corners, page px): a rotated word's
    axis-aligned box also covers what lies beside the text; the text's width and height follow
    from the box and the angle."""
    out = []
    for wd in words:
        bw, bh = wd.x1 - wd.x0, wd.y1 - wd.y0
        cx, cy = (wd.x0 + wd.x1) / 2, (wd.y0 + wd.y1) / 2
        a = math.radians(wd.angle_deg % 180.0)
        c, s = abs(math.cos(a)), abs(math.sin(a))
        den = c * c - s * s
        if s < 1e-3 or c < 1e-3 or abs(den) < 0.2:  # axis-aligned, or too near 45° to unfold
            tw, th, ang = bw, bh, 0.0
        else:
            tw = max(1.0, (bw * c - bh * s) / den)
            th = max(1.0, (bh * c - bw * s) / den)
            ang = -wd.angle_deg  # counter-clockwise text → clockwise in image coordinates
        out.append(
            np.asarray(cv2.boxPoints(((cx, cy), (tw + 2 * pad, th + 2 * pad), ang)), np.float64)
        )
    return out


def word_mask(words: list[Word], shape: tuple[int, int], pad: float = 0.0) -> NDArray[np.bool_]:
    out = np.zeros(shape, np.uint8)
    for box in word_boxes(words, pad):
        cv2.fillPoly(out, [np.round(box).astype(np.int32)], 1)
    return out > 0


# ---------------------------------------------------------------------------------------------
# 1–2: darkness and ink
# ---------------------------------------------------------------------------------------------
def darkness(gray: NDArray[np.uint8], bg_kernel_px: float) -> NDArray[np.float32]:
    h, w = gray.shape
    f = 8
    small = cv2.resize(gray, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    k = _odd(bg_kernel_px / f)
    bg = np.asarray(cv2.dilate(small, _disc(k)), np.float32)
    bg = np.asarray(cv2.GaussianBlur(bg, (0, 0), k / 2), np.float32)
    bg = np.asarray(cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR), np.float32)
    d = 1.0 - gray.astype(np.float32) / np.maximum(bg, 1.0)
    return np.asarray(np.clip(d, 0.0, 1.0), np.float32)


def ink_mask(
    d: NDArray[np.float32], rgb: NDArray[np.uint8] | None
) -> tuple[NDArray[np.bool_], float]:
    """Hysteresis threshold; returns (ink, noise σ of d on paper)."""
    ds = cv2.GaussianBlur(d, (0, 0), 0.8)
    paper = ds[ds < 0.1]
    sigma = float(1.4826 * np.median(np.abs(paper - np.median(paper)))) if paper.size else 0.02
    weak = ds > max(0.13, 6 * sigma)
    strong = ds > max(0.32, 12 * sigma)
    if rgb is not None:
        # coloured ink (stamps, signatures) is not drawing ink; where it covers black ink the
        # pixel stays dark in every channel and is kept (the drawing line under the stamp)
        lo = rgb.min(axis=2).astype(np.int16)
        chroma = rgb.max(axis=2).astype(np.int16) - lo
        tinted = ((chroma > 70) & (lo > 50)).astype(np.uint8)
        coloured = np.asarray(cv2.dilate(tinted, _disc(3)) > 0, np.bool_)
        dark = np.asarray(cv2.GaussianBlur(lo.astype(np.float32), (0, 0), 0.8) < 50, np.bool_)
        weak &= ~coloured | dark
        strong &= ~coloured | dark
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    return keep[lab], sigma


def drop_text_and_specks(
    ink: NDArray[np.bool_], words: list[Word], m_per_px: float
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Remove components that lie inside word boxes and isolated specks; returns (ink, text box
    mask)."""
    h, w = ink.shape
    tmask = word_mask(words, (h, w), pad=2.0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    in_text = np.bincount(np.asarray(lab, np.int64)[tmask], minlength=n).astype(np.float64)
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    diag = np.hypot(bw, bh) * m_per_px
    # the sheet frame (with the title block grid attached): ink spanning most of the page
    frame = (bw >= 0.7 * w) & (bh >= 0.7 * h) & (area < 0.05 * bw * bh)
    drop = (in_text >= 0.85 * area) | (diag < MIN_ELEMENT_M) | frame
    drop[0] = True
    return ~drop[lab] & ink, tmask


# ---------------------------------------------------------------------------------------------
# skeleton → ordered paths
# ---------------------------------------------------------------------------------------------
def _crossing(sk: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """(crossing number, neighbour count) per pixel of a 0/1 skeleton."""
    p = np.pad(sk, 1)
    h, w = sk.shape
    code = np.zeros((h, w), np.uint16)
    count = np.zeros((h, w), np.uint8)
    for k, (dy, dx) in enumerate(_RING):
        nb = p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        code |= nb.astype(np.uint16) << k
        count += nb
    return _CROSS[code], count


def skeleton_paths(skel: NDArray[np.bool_], min_px: int = 3) -> list[NDArray[np.float64]]:
    """Ordered pixel paths (x, y at pixel centres) between endpoints and junctions; a path that
    ends at a junction ends at the junction's centroid, so lines meeting there share a point."""
    sk = skel.astype(np.uint8)
    cross, count = _crossing(sk)
    # junctions: clusters of pixels with ≥ 3 neighbours from which ≥ 3 branches leave (crossing
    # numbers alone miss crossings whose centre is a 2-pixel cluster; staircase pixels with 3
    # neighbours have 2 branches)
    cand = ((sk > 0) & ((count >= 3) | (cross >= 3))).astype(np.uint8)
    nc, clab = cv2.connectedComponents(cand, connectivity=8)
    junction = np.zeros(sk.shape, bool)
    if nc > 1:
        ring = (cv2.dilate(cand, np.ones((3, 3), np.uint8)) > 0) & (sk > 0) & (cand == 0)
        nr, rlab = cv2.connectedComponents(ring.astype(np.uint8), connectivity=8)
        owner = cv2.dilate(clab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int64)
        ys, xs = np.nonzero(ring)
        pairs = np.unique(np.stack([owner[ys, xs], rlab[ys, xs]], axis=1), axis=0)
        exits = np.bincount(pairs[:, 0], minlength=nc)
        good = exits >= 3
        good[0] = False
        junction = good[clab] & (cand > 0)
        del nr
    nj, jlab, _, jcent = cv2.connectedComponentsWithStats(junction.astype(np.uint8), connectivity=8)
    rest = (sk > 0) & ~junction
    h, w = sk.shape
    jl = np.pad(jlab, 1)
    ys, xs = np.nonzero(rest)
    if len(xs) == 0:
        return []
    flat = set((ys * w + xs).tolist())
    visited: set[int] = set()
    nbr4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    nbr8 = [*nbr4, (-1, 1), (1, 1), (1, -1), (-1, -1)]

    def neighbours(i: int) -> list[int]:
        y, x = divmod(i, w)
        out = []
        for dy, dx in nbr8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                j = yy * w + xx
                if j in flat:
                    out.append(j)
        return out

    def junction_at(i: int) -> int:
        y, x = divmod(i, w)
        win = jl[y : y + 3, x : x + 3]
        labs = win[win > 0]
        return int(labs[0]) if labs.size else 0

    # endpoints first (open paths), then whatever is left (cycles)
    deg = {i: len(neighbours(i)) for i in flat}
    starts = [i for i, d in deg.items() if d <= 1] + list(flat)
    paths: list[NDArray[np.float64]] = []
    for s in starts:
        if s in visited:
            continue
        seq = [s]
        visited.add(s)
        cur = s
        while True:
            nxt = [j for j in neighbours(cur) if j not in visited]
            if not nxt:
                break
            cy, cx = divmod(cur, w)
            nxt.sort(key=lambda j: abs(j // w - cy) + abs(j % w - cx))  # 4-neighbours first
            cur = nxt[0]
            visited.add(cur)
            seq.append(cur)
        if len(seq) < min_px:
            continue
        pts = [(float(i % w) + 0.5, float(i // w) + 0.5) for i in seq]
        closed_loop = len(seq) > 3 and seq[-1] in neighbours(seq[0])
        for end in (0, -1) if not closed_loop else ():
            jlbl = junction_at(seq[end])
            if jlbl:
                c = (float(jcent[jlbl][0]) + 0.5, float(jcent[jlbl][1]) + 0.5)
                if end == 0:
                    pts.insert(0, c)
                else:
                    pts.append(c)
        if closed_loop:
            pts.append(pts[0])
        paths.append(np.array(pts, np.float64))
    del nj
    return paths


# ---------------------------------------------------------------------------------------------
# runs: straight pieces and circular arcs
# ---------------------------------------------------------------------------------------------
@dataclass
class Run:
    pts: NDArray[np.float64]  # vertices (straight) or the fitted arc's samples
    curve: bool
    circle: tuple[float, float, float] | None = None  # cx, cy, r (px)
    sweep_deg: float = 0.0
    span: tuple[int, int] | None = None  # straight runs: vertex indices of their raw pixels


def _turn(a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64]) -> float:
    u, v = b - a, c - b
    return math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], float(np.dot(u, v))))


def _arc_samples(
    cx: float, cy: float, r: float, a_from: float, a_to: float, ccw: bool
) -> NDArray[np.float64]:
    """Points on the circle from angle a_from to a_to (degrees), through the multiples of
    ARC_GRID_DEG in between, so concentric faces sampled this way have parallel chords."""
    if ccw:
        sweep = (a_to - a_from) % 360
        grid = np.arange(
            math.ceil(a_from / ARC_GRID_DEG), math.floor((a_from + sweep) / ARC_GRID_DEG) + 1
        )
        angs = [a_from, *[float(g * ARC_GRID_DEG) for g in grid], a_from + sweep]
    else:
        sweep = (a_from - a_to) % 360
        grid = np.arange(
            math.floor(a_from / ARC_GRID_DEG), math.ceil((a_from - sweep) / ARC_GRID_DEG) - 1, -1
        )
        angs = [a_from, *[float(g * ARC_GRID_DEG) for g in grid], a_from - sweep]
    out: list[tuple[float, float]] = []
    for a in angs:
        p = (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
        if not out or math.dist(out[-1], p) > 1e-6:
            out.append(p)
    return np.array(out, np.float64)


def _arc_run(span: NDArray[np.float64], min_r: float, max_r: float) -> Run | None:
    """The points as one circular arc (fit rms ≤ 1 px, sweep ≥ 8°), or None."""
    if len(span) < 5:
        return None
    cx, cy, r, rms = circle_fit(span)
    if not (rms <= 1.0 and min_r <= r <= max_r):
        return None
    # visibly curved: a straight line fits clearly worse (a straight face with a chamfered end
    # also fits a large circle within a pixel)
    centred = span - span.mean(axis=0)
    line_rms = math.sqrt(max(float(np.linalg.eigvalsh(centred.T @ centred / len(span))[0]), 0.0))
    if line_rms < max(1.5, 2.5 * rms):
        return None
    ang = np.degrees(np.arctan2(span[:, 1] - cy, span[:, 0] - cx))
    steps = (np.diff(ang) + 180) % 360 - 180
    total = float(steps.sum())
    if abs(total) < 8 or (np.sign(steps) == -np.sign(total)).mean() > 0.3:
        return None
    a0 = float(ang[0])
    a1 = a0 + total
    samples = _arc_samples(cx, cy, r, a0, a1 % 360, total > 0)
    return Run(samples, True, (cx, cy, r), abs(total))


def merge_arcs(runs: list[Run]) -> list[Run]:
    """Arc pieces on one circle (a door swing cut by its tag circle, a curved wall face split at
    junctions) become one arc over the union of their angles."""
    arcs = [r for r in runs if r.curve and r.circle is not None]
    rest = [r for r in runs if not (r.curve and r.circle is not None)]
    groups: list[list[Run]] = []
    for a in sorted(arcs, key=lambda r: -r.sweep_deg):
        assert a.circle is not None
        for g in groups:
            c0 = g[0].circle
            assert c0 is not None
            tol = 0.04 * c0[2] + 3.0
            if (
                math.hypot(a.circle[0] - c0[0], a.circle[1] - c0[1]) <= tol
                and abs(a.circle[2] - c0[2]) <= 0.03 * c0[2] + 2.0
            ):
                g.append(a)
                break
        else:
            groups.append([a])
    out = list(rest)
    for g in groups:
        if len(g) == 1:
            out.append(g[0])
            continue
        pts = np.vstack([r.pts for r in g])
        cx, cy, r, _ = circle_fit(pts)
        ang = np.sort(np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360)
        gaps = np.diff(np.concatenate([ang, [ang[0] + 360]]))
        k = int(np.argmax(gaps))  # the arc spans everything but the largest angular gap
        a_from = float(ang[(k + 1) % len(ang)])
        sweep = 360.0 - float(gaps[k])
        out.append(
            Run(
                _arc_samples(cx, cy, r, a_from, (a_from + sweep) % 360, True),
                True,
                (cx, cy, r),
                sweep,
            )
        )
    return out


def split_runs(
    pts: NDArray[np.float64],
    *,
    closed: bool,
    eps_px: float,
    max_chord_px: float,
    min_radius_px: float,
    max_radius_px: float,
) -> list[Run]:
    """Douglas–Peucker vertices, then maximal runs of ≥ 3 short chords turning gently the same way
    that fit one circle become arcs; the rest stay straight."""
    if len(pts) < 2:
        return []
    raw = pts[:-1] if closed and np.allclose(pts[0], pts[-1]) else pts
    if len(raw) < 2:
        return []
    if not closed and len(raw) >= 8:  # a piece between junctions that is one arc
        whole = _arc_run(raw, min_radius_px, max_radius_px)
        if whole is not None:
            return [whole]
    approx = cv2.approxPolyDP(raw.astype(np.float32).reshape(-1, 1, 2), eps_px, closed)
    idx = [
        int(np.argmin(np.hypot(raw[:, 0] - q[0], raw[:, 1] - q[1]))) for q in approx.reshape(-1, 2)
    ]
    if closed:
        order = sorted(set(idx))
        if len(order) < 3:
            return []
        vv = raw[order]
        corner = [abs(_turn(vv[k - 1], vv[k], vv[(k + 1) % len(vv)])) for k in range(len(vv))]
        k = int(np.argmax(corner))  # start at the sharpest corner, so no arc straddles the start
        idx = order[k:] + order[:k]
    else:
        idx = sorted(set(idx) | {0, len(raw) - 1})
    v = raw[idx]
    nseg = len(v) if closed else len(v) - 1
    if nseg < 1:
        return []

    def seg(i: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return v[i % len(v)], v[(i + 1) % len(v)]

    def raw_span(i0: int, i1: int) -> NDArray[np.float64]:
        a, b = idx[i0 % len(idx)], idx[i1 % len(idx)]
        if b >= a:
            return raw[a : b + 1]
        return np.vstack([raw[a:], raw[: b + 1]])

    # chords of an arc at this tolerance are ≥ 5 px; shorter ones are corner chamfers
    short = [5.0 <= float(np.hypot(*(seg(i)[1] - seg(i)[0]))) <= max_chord_px for i in range(nseg)]
    turns = []
    for i in range(nseg):
        if not closed and i == nseg - 1:
            turns.append(999.0)
            continue
        a, b = seg(i)
        _, c = seg(i + 1)
        turns.append(_turn(a, b, c))
    runs: list[Run] = []
    i = 0
    while i < nseg:
        j = i
        if short[i]:
            sign = 0.0
            while j + 1 < nseg and short[j + 1]:
                t = turns[j]
                if abs(t) > 25 or abs(t) < 0.3 or (sign and math.copysign(1, t) != sign):
                    break
                sign = math.copysign(1, t)
                j += 1
        if j - i + 1 >= 2:
            arc = _arc_run(raw_span(i, j + 1), min_radius_px, max_radius_px)
            if arc is not None and (j - i + 1 >= 3 or arc.sweep_deg >= 12):
                runs.append(arc)
                i = j + 1
                continue
        a, b = seg(i)
        runs.append(Run(np.array([a, b], np.float64), False, span=(i, i + 1)))
        i += 1
    runs = _merge_collinear_runs(runs, raw_span, closed)
    _refit_straight(runs, raw_span, closed)
    if not closed:
        runs = _absorb_end_hooks(runs, raw)
    return runs


def _absorb_end_hooks(runs: list[Run], raw: NDArray[np.float64]) -> list[Run]:
    """A path ending in a junction bends over its last pixels towards the junction's centroid; a
    short end run beside a long straight one is that hook: the straight run is extended to the
    path's end (projected on its line), so a dimension line keeps its full length."""

    def length(r: Run) -> float:
        return float(np.hypot(*(r.pts[-1] - r.pts[0])))

    out = list(runs)
    for end in (0, -1):
        if len(out) < 2:
            break
        hook, main = (out[0], out[1]) if end == 0 else (out[-1], out[-2])
        if (
            hook.curve
            or main.curve
            or length(hook) > 8.0
            or length(main) < 4 * max(length(hook), 1.0)
        ):
            continue
        a, b = main.pts[0], main.pts[-1]
        u = (b - a) / max(length(main), 1e-9)
        tip = raw[0] if end == 0 else raw[-1]
        p = a + u * float((tip - a) @ u)
        if end == 0:
            main.pts = np.array([p, b], np.float64)
            out = out[1:]
        else:
            main.pts = np.array([a, p], np.float64)
            out = out[:-1]
    return out


def _merge_collinear_runs(runs: list[Run], raw_span: Any, closed: bool) -> list[Run]:
    """Consecutive straight runs that are one line with a pixel bump between them (hatch lines
    meeting an outline at a shallow angle, noise) become one run."""

    def mergeable(a: Run, b: Run) -> bool:
        if a.curve or b.curve or a.span is None or b.span is None:
            return False
        p, q, r = a.pts[0], a.pts[-1], b.pts[-1]
        d1, d2 = q - p, r - q
        n1, n2 = float(np.hypot(*d1)), float(np.hypot(*d2))
        if n1 < 1e-9 or n2 < 1e-9:
            return False
        if abs(_turn(p, q, r)) > 4.0:
            return False
        chord = r - p
        length = float(np.hypot(*chord))
        nrm = np.array([-chord[1], chord[0]]) / max(length, 1e-9)
        pts = raw_span(a.span[0], b.span[1])
        return bool(float(np.abs((pts - p) @ nrm).max()) <= 2.0)

    out = list(runs)
    changed = True
    while changed and len(out) > 1:
        changed = False
        n = len(out)
        for i in range(n if closed else n - 1):
            j = (i + 1) % n
            if i == j or not mergeable(out[i], out[j]):
                continue
            a, b = out[i], out[j]
            assert a.span is not None and b.span is not None
            merged = Run(
                np.array([a.pts[0], b.pts[-1]], np.float64), False, span=(a.span[0], b.span[1])
            )
            # j == 0: the merge wrapped around the start of a closed contour
            out = [merged, *out[1:i]] if j == 0 else [*out[:i], merged, *out[j + 1 :]]
            changed = True
            break
    return out


def _refit_straight(runs: list[Run], raw_span: Any, closed: bool) -> None:
    """Straight runs take the orthogonal least-squares line of their pixels (the middle 70 %: the
    ends round off at corners); a vertex between two straight runs becomes their lines'
    intersection. Douglas–Peucker vertices sit on pixel bumps; the fitted lines do not."""
    lines: list[tuple[NDArray[np.float64], NDArray[np.float64]] | None] = []
    for r in runs:
        if r.curve or r.span is None:
            lines.append(None)
            continue
        sp = raw_span(*r.span)
        k = len(sp)
        if k < 6:
            lines.append(None)
            continue
        core = sp[int(0.15 * k) : k - int(0.15 * k)] if k >= 14 else sp
        c = core.mean(axis=0)
        _, evecs = np.linalg.eigh(np.cov((core - c).T))
        u = evecs[:, 1]
        lines.append((c, u))
        r.pts = np.array([c + u * float((q - c) @ u) for q in r.pts], np.float64)
    n = len(runs)
    # a chamfer (a run of a few pixels between two straight runs meeting at a corner) collapses
    # into the corner
    for i in range(n) if closed else range(1, n - 1):
        h, k = (i - 1) % n, (i + 1) % n
        r = runs[i]
        size = float(np.hypot(*(r.pts[-1] - r.pts[0])))
        if r.curve or size > 10.0:
            continue
        lh, lk = lines[h], lines[k]
        if lh is None or lk is None or h == k:
            continue
        if (
            min(
                float(np.hypot(*(runs[h].pts[-1] - runs[h].pts[0]))),
                float(np.hypot(*(runs[k].pts[-1] - runs[k].pts[0]))),
            )
            < 2.5 * size
        ):
            continue
        (ch, uh), (ck, uk) = lh, lk
        cross = float(uh[0] * uk[1] - uh[1] * uk[0])
        if abs(cross) < math.sin(math.radians(30)):
            continue
        t = float(((ck - ch)[0] * uk[1] - (ck - ch)[1] * uk[0]) / cross)
        p = ch + uh * t
        if math.dist(p, (r.pts[0] + r.pts[-1]) / 2) > 8.0:
            continue
        runs[h].pts[-1] = p
        runs[k].pts[0] = p
        r.pts = np.array([p, p], np.float64)
        lines[i] = None
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        li, lj = lines[i], lines[j]
        if li is None or lj is None:
            continue
        (ci, ui), (cj, uj) = li, lj
        cross = float(ui[0] * uj[1] - ui[1] * uj[0])
        if abs(cross) > math.sin(math.radians(8)):
            t = float(((cj - ci)[0] * uj[1] - (cj - ci)[1] * uj[0]) / cross)
            p = ci + ui * t
            if math.dist(p, runs[i].pts[-1]) > 6.0:  # a far intersection: keep the vertex
                continue
        else:
            p = (runs[i].pts[-1] + runs[j].pts[0]) / 2
        runs[i].pts[-1] = p
        runs[j].pts[0] = p


def _join_runs(runs: list[Run], closed: bool) -> tuple[NDArray[np.float64], bool]:
    pts: list[tuple[float, float]] = []
    curve = False
    for r in runs:
        curve |= r.curve
        for q in r.pts:
            t = (float(q[0]), float(q[1]))
            if not pts or math.dist(pts[-1], t) > 1e-6:
                pts.append(t)
    if closed and len(pts) > 1 and math.dist(pts[0], pts[-1]) < 1e-6:
        pts.pop()
    return np.array(pts, np.float64), curve


# ---------------------------------------------------------------------------------------------
# 3–4: wall body and glazing cuts
# ---------------------------------------------------------------------------------------------
def _thin_regions(
    ink: NDArray[np.bool_],
    words: list[Word],
    m_per_px: float,
) -> NDArray[np.bool_]:
    """Enclosed background regions narrower than the thickest wall, with no text inside. Found
    twice: with the strokes thickened by a pixel (a one-pixel break in a noisy outline does not
    open a wall into its room) and as drawn (a wall only a few pixels thick keeps the narrow gap
    between its two face strokes)."""
    h, w = ink.shape
    boxes = word_boxes(words)
    out = np.zeros((h, w), bool)
    ink_u8 = ink.astype(np.uint8)
    for sealed in (True, False):
        mask = cv2.dilate(ink_u8, np.ones((3, 3), np.uint8)) if sealed else ink_u8
        bg = (mask == 0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
        if n <= 1:
            continue
        dist = cv2.distanceTransform(bg, cv2.DIST_L2, 5)
        maxd = np.asarray(ndimage.maximum(dist, lab, index=np.arange(n)), np.float64)
        width_m = (2 * maxd + (2 if sealed else 0)) * m_per_px
        x0, y0 = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
        x1 = x0 + stats[:, cv2.CC_STAT_WIDTH]
        y1 = y0 + stats[:, cv2.CC_STAT_HEIGHT]
        border = (x0 <= 0) | (y0 <= 0) | (x1 >= w) | (y1 >= h)
        cand = (width_m <= MAX_T + 0.05) & ~border
        cand[0] = False
        # wall bodies hold no text; dimension-chain cells, table cells and tag bubbles (also when
        # a door swing cuts the bubble in two) do: a region holding at least half of a word is
        # not a wall (text drawn across walls, like a stamp's, touches many regions a little)
        for box in boxes:
            x0, y0 = max(0, int(box[:, 0].min())), max(0, int(box[:, 1].min()))
            x1, y1 = min(w, int(box[:, 0].max()) + 1), min(h, int(box[:, 1].max()) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            local = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(local, [np.round(box - [x0, y0]).astype(np.int32)], 1)
            labs = lab[y0:y1, x0:x1][local > 0]
            if labs.size == 0:
                continue
            counts = np.bincount(labs, minlength=n)
            counts[0] = 0
            k = int(np.argmax(counts))
            if k and counts[k] >= 0.5 * labs.size:
                cand[k] = False
        out |= np.asarray(cand[lab], bool)
    return out


def _glazing_cuts(
    body: NDArray[np.uint8],
    runs: list[Run],
    thin: NDArray[np.bool_],
    thick: NDArray[np.uint8],
    m_per_px: float,
    stroke_px: float,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64], float]]:
    """Straight thin-stroke runs inside the body at a constant distance from both faces →
    (start, end, half the body width there). A run is extended along its line while the stroke
    continues inside the body (a glazing line drawn in pieces ends at both jambs)."""
    db = cv2.distanceTransform(body, cv2.DIST_L2, 5)
    h, w = body.shape
    ink_near = cv2.dilate(thin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    poche = thick > 0

    limit = int(MAX_T / m_per_px) + 2

    def reach(p: NDArray[np.float64], d: NDArray[np.float64]) -> float:
        for k in range(1, limit):
            x, y = int(p[0] + d[0] * k), int(p[1] + d[1] * k)
            if not (0 <= x < w and 0 <= y < h) or not body[y, x]:
                return float(k)
        return float(limit)

    def extend(p: NDArray[np.float64], d: NDArray[np.float64]) -> NDArray[np.float64]:
        """Walk on while the stroke continues. Ending at poché, the jamb is the poché's edge;
        ending past a thin jamb stroke, step back to that stroke's centre line."""
        q = p.copy()
        moved = False
        for _ in range(int(MAX_GAP_PX)):
            nxt = q + d
            x, y = int(nxt[0]), int(nxt[1])
            if not (0 <= x < w and 0 <= y < h) or db[y, x] < 1.0:
                break
            if poche[y, x]:
                return nxt
            if not ink_near[y, x]:
                break
            q = nxt
            moved = True
        return q - d * (stroke_px / 2 + 1) if moved else q

    out = []
    for r in runs:
        if r.curve:
            continue
        a, b = r.pts[0], r.pts[-1]
        length = float(np.hypot(*(b - a)))
        if length * m_per_px < GLAZING_MIN_M:
            continue
        f = np.linspace(0.1, 0.9, max(8, int(0.8 * length)))
        q = a[None, :] + (b - a)[None, :] * f[:, None]
        vals = db[np.clip(q[:, 1].astype(int), 0, h - 1), np.clip(q[:, 0].astype(int), 0, w - 1)]
        if vals.min() < 1.5:
            continue
        u = (b - a) / length
        nrm = np.array([-u[1], u[0]])
        # the body reaches about equally far on both sides all along the line: a face stroke has
        # it on one side only, a hatch line crossing the wall obliquely on both sides unequally
        qs = q[:: max(1, len(q) // 10)]
        plus = np.array([reach(p, nrm) for p in qs])
        minus = np.array([reach(p, -nrm) for p in qs])
        tot = plus + minus
        if min(plus.min(), minus.min()) < 2.0 or float(np.abs(plus - minus).max()) > max(
            2.5, 0.25 * float(np.median(tot))
        ):
            continue
        half = float(np.median(tot)) / 2
        if not 0.04 <= 2 * half * m_per_px <= MAX_T + 0.05:
            continue
        out.append((extend(a, -u), extend(b, u), half))
    # pieces of one glazing line give overlapping cuts; that is harmless
    return out


@dataclass
class _Layers:
    ink_u8: NDArray[np.uint8]
    thick: NDArray[np.uint8]
    thin: NDArray[np.bool_]
    skel: NDArray[np.bool_]
    stroke: float
    paths: list[NDArray[np.float64]]
    runs: list[Run]
    low_res: bool


def _layers(ink: NDArray[np.bool_], m_per_px: float, kw: dict[str, float]) -> _Layers:
    """Ink → poché (thick) and strokes (thin), the stroke skeleton, its paths and runs."""
    from skimage.morphology import skeletonize

    ink_u8 = ink.astype(np.uint8)
    # a first stroke width from the ridges of all ink (strokes outnumber poché): no stroke may
    # pass as poché, whatever the resolution
    d_ink = cv2.distanceTransform(ink_u8, cv2.DIST_L2, 3)
    ridge = (d_ink >= 1.0) & (d_ink >= cv2.dilate(d_ink, np.ones((3, 3), np.uint8)) - 1e-6)
    stroke0 = float(np.percentile(2 * d_ink[ridge], 30)) if ridge.any() else 2.0
    kd = max(_odd(0.6 * MIN_WALL_M / m_per_px), _odd(stroke0 + 2))
    opened = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, _disc(kd))
    # poché, not stroke junctions: a thick component holds at least a short piece of thin wall
    _, lab, st, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    small = st[:, cv2.CC_STAT_AREA] < 0.25 * MIN_WALL_M / (m_per_px * m_per_px)
    small[0] = True
    opened = (~small[lab] & (opened > 0)).astype(np.uint8)
    # the opening rounds convex corners (wall ends): give back the ink within half the kernel
    thick = (np.asarray(cv2.dilate(opened, _disc(_odd(kd / 2 + 1))), np.uint8) & ink_u8).astype(
        np.uint8
    )
    thin = ink & ~(cv2.dilate(thick, _disc(3)) > 0)
    skel = np.asarray(skeletonize(thin), np.bool_)  # type: ignore[no-untyped-call]
    sw = d_ink[skel]
    stroke = float(2 * np.median(sw)) if sw.size else 2.0
    paths = skeleton_paths(skel)
    runs: list[Run] = []
    for p in paths:
        runs += split_runs(p, closed=False, **kw)
    low_res = 0.10 / m_per_px < 2 * stroke0 + 1  # a 10 cm wall barely wider than two strokes
    return _Layers(ink_u8, thick, np.asarray(thin, np.bool_), skel, stroke, paths, runs, low_res)


def _stamp_rings(runs: list[Run]) -> list[tuple[float, float, float, float]]:
    """Circular stamps (and other drawn rings): a near-full circle (sweep ≥ 300°; a circle
    crossed by walls is found in pieces and merged) → (cx, cy, 0.75 r, r), the band where a
    stamp's inner ring and lettering lie. A wall is never drawn as a lone full circle with a
    band inside it; a stamp's annulus would read as a curved wall."""
    out = []
    for r in merge_arcs(runs):
        if r.curve and r.circle is not None and r.sweep_deg >= 300:
            cx, cy, rad = r.circle
            out.append((cx, cy, 0.75 * rad, rad))
    return out


def _on_ring(r: Run, rings: list[tuple[float, float, float, float]], stroke: float) -> bool:
    for cx, cy, r_in, r_out in rings:
        d = np.hypot(r.pts[:, 0] - cx, r.pts[:, 1] - cy)
        if bool(((d >= r_in - 2 * stroke) & (d <= r_out + 2 * stroke)).all()):
            return True
    return False


def _fill_small_holes(mask: NDArray[np.uint8], max_area_px: float) -> NDArray[np.uint8]:
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return mask
    out = mask.copy()
    small = [
        c
        for c, hh in zip(contours, hier[0], strict=True)
        if hh[3] != -1 and cv2.contourArea(c) <= max_area_px
    ]
    if small:
        cv2.drawContours(out, small, -1, 1, thickness=cv2.FILLED)
    return out


def _body_polygons(body: NDArray[np.uint8], inset_px: float, kw: dict[str, float]) -> list[Polygon]:
    """Body components → polygons (straight runs refitted, arcs resampled), offset inwards by
    ``inset_px`` with mitred corners. Contour points are boundary pixel centres, half a pixel
    inside the mask edge."""
    contours, hier = cv2.findContours(body, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    out: list[Polygon] = []

    def ring(k: int) -> list[tuple[float, float]] | None:
        pts = contours[k].reshape(-1, 2).astype(np.float64) + 0.5
        if len(pts) < 4:
            return None
        r, _ = _join_runs(split_runs(pts, closed=True, **kw), True)
        return [(float(x), float(y)) for x, y in r] if len(r) >= 3 else None

    for i in range(len(contours)):
        if hier[i][3] != -1:
            continue  # holes are taken with their parent
        shell = ring(i)
        if shell is None:
            continue
        holes = []
        j = hier[i][2]
        while j != -1:
            h = ring(j)
            if h is not None:
                holes.append(h)
            j = hier[j][0]
        poly: Any = Polygon(shell, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        off = inset_px - 0.5
        if abs(off) > 1e-6:
            poly = poly.buffer(-off, join_style="mitre", mitre_limit=4.0)
        for g in getattr(poly, "geoms", [poly]):
            if isinstance(g, Polygon) and not g.is_empty and g.area > 4.0:
                out.append(orient(g))
    return out


def _cut(
    body: NDArray[np.uint8], a: NDArray[np.float64], b: NDArray[np.float64], half: float
) -> None:
    u = (b - a) / max(float(np.hypot(*(b - a))), 1e-9)
    n = np.array([-u[1], u[0]])
    quad = np.array([a + n * half, b + n * half, b - n * half, a - n * half]) - 0.5
    cv2.fillPoly(body, [np.round(quad * 16).astype(np.int32)], 0, lineType=cv2.LINE_8, shift=4)


# ---------------------------------------------------------------------------------------------
# 5: vectorise
# ---------------------------------------------------------------------------------------------
def vectorise(
    gray: NDArray[np.uint8],
    words: list[Word],
    m_per_px: float,
    *,
    rgb: NDArray[np.uint8] | None = None,
    debug: bool = False,
    wall_mask: NDArray[np.bool_] | None = None,
) -> RasterVectors:
    """``wall_mask``: a wall body predicted by the plan segmentation model (``raster_seg``,
    faces on its edges); it replaces the morphological wall body below."""
    notes: list[str] = []
    words = text_words(words)
    d = darkness(gray, 2.2 * MAX_T / m_per_px)
    ink, sigma = ink_mask(d, rgb)
    ink, _tmask = drop_text_and_specks(ink, words, m_per_px)
    kw = {
        "eps_px": 1.0,
        "max_chord_px": 2.0 / m_per_px,
        "min_radius_px": 0.4 / m_per_px,
        "max_radius_px": 60.0 / m_per_px,
    }
    lay = _layers(ink, m_per_px, kw)
    if lay.low_res:
        notes.append(
            f"low resolution: {m_per_px * 100:.2f} cm per pixel leaves thin walls narrower than "
            "two strokes; thin poché walls may be missed (scan at a higher DPI)"
        )
    ink_u8, thick, thin, skel, stroke = lay.ink_u8, lay.thick, lay.thin, lay.skel, lay.stroke
    paths, stroke_runs = lay.paths, lay.runs
    # wall body
    fill = _thin_regions(ink, words, m_per_px)
    rings = _stamp_rings(stroke_runs)
    if rings:
        band = np.zeros(fill.shape, np.uint8)
        for cx, cy, r_in, r_out in rings:
            cv2.circle(band, (round(cx), round(cy)), round(r_out + stroke), 1, -1)
            cv2.circle(band, (round(cx), round(cy)), max(1, round(r_in - stroke)), 0, -1)
        # enclosed regions lying in a ring's band are pieces of its annulus (walls crossing the
        # ring reach far beyond it and stay)
        n_f, lab_f, st_f, _ = cv2.connectedComponentsWithStats(
            fill.astype(np.uint8), connectivity=4
        )
        in_band = np.bincount(np.asarray(lab_f, np.int64)[band > 0], minlength=n_f)
        ring_part = in_band >= 0.8 * st_f[:, cv2.CC_STAT_AREA]
        ring_part[0] = False
        # an annulus piece runs along the ring (≥ 20° of it); hatch strips of a wall crossing
        # the ring are short pieces and stay
        for k in np.nonzero(ring_part)[0]:
            ys, xs = np.nonzero(lab_f == k)
            cx, cy = rings[0][0], rings[0][1]
            best = min(rings, key=lambda r: math.hypot(xs.mean() - r[0], ys.mean() - r[1]))
            cx, cy = best[0], best[1]
            ang = np.sort(np.degrees(np.arctan2(ys - cy, xs - cx)) % 360)
            gaps = np.diff(np.concatenate([ang, [ang[0] + 360]]))
            if 360 - float(gaps.max()) < 20:
                ring_part[k] = False
        fill &= ~ring_part[lab_f]
        stroke_runs = [r for r in stroke_runs if not _on_ring(r, rings, stroke)]
        notes.append(f"{len(rings)} drawn ring(s): enclosed bands inside them are not walls")
    # enclosed regions separated only by a stroke (hatch strips, window halves) are one body; it
    # takes in the strokes that bound it (outline, hatch lines) and the small pockets they enclose
    fill_u8 = np.asarray(
        cv2.morphologyEx(fill.astype(np.uint8), cv2.MORPH_CLOSE, _disc(_odd(2 * stroke + 4))),
        np.uint8,
    )
    near = np.asarray(cv2.dilate(fill_u8, _disc(_odd(4 * stroke + 5))), np.uint8)
    outlined = np.asarray(
        cv2.morphologyEx(fill_u8 | (ink_u8 & near), cv2.MORPH_CLOSE, _disc(_odd(2 * stroke + 1))),
        np.uint8,
    )
    outlined = _fill_small_holes(outlined, (0.1 / m_per_px) ** 2)
    # strokes leaving the body (door leaves, swing ends) were taken in up to that reach: spikes
    outlined = np.asarray(
        cv2.morphologyEx(outlined, cv2.MORPH_OPEN, _disc(_odd(stroke + 1))), np.uint8
    )
    body = (outlined | thick).astype(np.uint8)
    # the mask edges lie on the outer edge of the outline strokes; the faces are the strokes'
    # centre lines, half a stroke inside (applied to the polygons, which keeps corners sharp)
    bias_px = stroke / 2
    if wall_mask is not None:
        # the learned wall body: faces on its edges, openings already left out
        body = _fill_small_holes(wall_mask.astype(np.uint8), (0.1 / m_per_px) ** 2)
        bias_px = 0.0
        notes.append("wall body from the plan segmentation model (raster_seg)")
    cuts = (
        _glazing_cuts(body, stroke_runs, thin, thick, m_per_px, stroke) if wall_mask is None else []
    )
    for a, b, half in cuts:
        _cut(body, a, b, half + 1.5)
    if cuts:
        notes.append(f"{len(cuts)} window(s) cut out of filled wall outlines (glazing lines)")
    # primitives
    out_paths: list[dict[str, Any]] = []
    for poly in _body_polygons(body, bias_px, {**kw, "eps_px": 2.5}):
        loops = [list(poly.exterior.coords)[:-1]] + [list(h.coords)[:-1] for h in poly.interiors]
        out_paths.append(
            {
                "sub": [[list(q) for q in r] for r in loops],
                "closed": [True] * len(loops),
                "curve": [False] * len(loops),  # arcs are resampled into chords on a common grid
                "w": 0.0,
                "fill": True,
                "stroke": False,
                "clip": False,
                "source": "body",
            }
        )
    for run in merge_arcs(stroke_runs):
        out_paths.append(
            {
                "sub": [run.pts.tolist()],
                "closed": [False],
                "curve": [run.curve],
                "w": stroke,
                "fill": False,
                "stroke": True,
                "clip": False,
                "source": "stroke",
            }
        )
    dbg: dict[str, NDArray[Any]] = {}
    if debug:
        dbg = {"ink": ink, "thick": thick > 0, "fill": fill, "body": body > 0, "skel": skel}
    notes.append(f"noise σ {sigma:.3f}, stroke {stroke:.1f} px, {len(paths)} stroke paths")
    return RasterVectors(
        {"paths": out_paths, "truncated": False, "frame": "page_px"},
        stroke,
        notes,
        dbg,
        body > 0,
        ink.astype(bool),
    )


# ---------------------------------------------------------------------------------------------
# extraction with a measured scale
# ---------------------------------------------------------------------------------------------
def stroke_segments(
    rv: RasterVectors,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Straight stroke runs (dimension lines among them), page pixels."""
    out = []
    for p in rv.paths["paths"]:
        if p.get("fill") or p["curve"][0]:
            continue
        sub = p["sub"][0]
        out.append(((float(sub[0][0]), float(sub[0][1])), (float(sub[-1][0]), float(sub[-1][1]))))
    return out


@dataclass
class RasterScale:
    m_per_px: float
    result: Any  # plan.scale.ScaleResult
    passes: int
    notes: list[str]


def _body_width_px(body: NDArray[np.bool_] | None) -> float | None:
    """Typical wall body width (px): twice the distance transform along the body's ridges."""
    if body is None or not body.any():
        return None
    dist = cv2.distanceTransform(body.astype(np.uint8), cv2.DIST_L2, 5)
    ridge = (dist >= 1.5) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-6)
    if not ridge.any():
        return None
    return float(2 * np.median(dist[ridge]))


def _door_widths_px(rv: RasterVectors, words: list[Word], prior: float) -> list[float]:
    from archrender.plan.builder import build_plan
    from archrender.plan.extract import pdf_extract

    ex = pdf_extract(rv.paths, words, prior, method="raster_cv")
    plan = build_plan(ex.prims, project="scale", version="probe", source="raster").plan
    return [o.width_m.value / prior for o in plan.openings if o.type == "door"]


def raster_scale(
    gray: NDArray[np.uint8],
    words: list[Word],
    *,
    stated_n: float | None,
    dpi: float | None,
    rgb: NDArray[np.uint8] | None = None,
    reader: Any = None,
    wall_mask: NDArray[np.bool_] | None = None,
) -> tuple[RasterScale, RasterVectors]:
    """Scale of a raster plan: the stated scale (with the scan's DPI) and the dimension strings
    paired with the dimension lines vectorised at the best prior; vectorised again at the fused
    scale when it moved by more than 3 %. Without DPI (rectified photos) the stated scale is
    unusable and only the dimension strings (or, last, the door prior) give the scale."""
    from archrender.plan.scale import (
        dimension_pairs,
        from_dimensions,
        from_door_radii,
        fuse,
        stated,
    )

    notes: list[str] = []
    words = text_words(words)
    ests = []
    if stated_n and dpi:
        ests.append(stated(stated_n, 0.0254 / dpi, f"1:{stated_n:g} at {dpi:g} dpi"))
    prior = ests[0].value if ests else 0.01
    if not ests:
        notes.append("no stated scale with a known DPI: 1 cm/px prior for the first pass")
    rv = vectorise(gray, words, prior, rgb=rgb, wall_mask=wall_mask)
    heights = sorted(w.y1 - w.y0 for w in words if w.angle_deg in (0.0, 180.0))
    # dimension text is the small text of a drawing (2–3 mm)
    text_px = float(np.percentile(heights, 25)) if heights else (2.5 * dpi / 25.4 if dpi else 20.0)
    gap = 6 * dpi / 25.4 if dpi else 3.0 * text_px
    segments = stroke_segments(rv)
    dim_words = list(words)
    if reader is not None and hasattr(reader, "read_line"):
        from archrender.plan.dimread import read_dimension_lines

        extra = read_dimension_lines(
            gray,
            segments,
            words,
            reader,
            text_px=text_px,
            exclude=[rv.body] if rv.body is not None else None,
        )
        if extra:
            notes.append(f"{len(extra)} dimension strings read along their dimension lines")
        dim_words += extra
    dims = from_dimensions(dimension_pairs(dim_words, segments, max_gap_px=gap))
    if dims is not None:
        ests.append(dims)
    if not any(e.rel_sigma <= 0.01 for e in ests):
        # last resort: door swings of a first-pass plan (≈ 0.85 m)
        doors = _door_widths_px(rv, words, prior)
        prior_est = from_door_radii(doors)
        if prior_est is not None:
            ests.append(prior_est)
    # a scale must give walls of a plausible thickness (5–60 cm): a misread dimension does not
    width_px = _body_width_px(rv.body)
    if width_px is not None:
        kept = [e for e in ests if 0.05 <= width_px * e.value <= 0.6]
        for e in ests:
            if e not in kept:
                notes.append(
                    f"scale estimate {e.method} ({e.detail}) rejected: walls would be "
                    f"{width_px * e.value * 100:.1f} cm thick"
                )
        ests = kept
    res = fuse(ests)
    passes = 1
    if res.estimate is None:
        notes.append("no scale estimate: neither a stated scale with DPI nor dimension strings")
        return RasterScale(prior, res, passes, notes), rv
    m = res.estimate.value
    if abs(m - prior) / prior > 0.03:
        rv = vectorise(gray, words, m, rgb=rgb, wall_mask=wall_mask)
        passes = 2
    return RasterScale(m, res, passes, notes), rv


def raster_extract(
    gray: NDArray[np.uint8],
    words: list[Word],
    m_per_px: float,
    *,
    rgb: NDArray[np.uint8] | None = None,
    vectors: RasterVectors | None = None,
    wall_mask: NDArray[np.bool_] | None = None,
) -> Any:
    """Raster page → ``Extracted`` (primitives in plan metres) for the PlanBuilder."""
    from archrender.plan.extract import pdf_extract

    words = text_words(words)
    rv = vectors or vectorise(gray, words, m_per_px, rgb=rgb, wall_mask=wall_mask)
    method: Method = "raster_seg" if wall_mask is not None else "raster_cv"
    ex = pdf_extract(rv.paths, words, m_per_px, method=method)
    ex.notes.extend(rv.notes)
    ex.prims.notes.extend(rv.notes)
    return ex
