"""Hint sources built from synthetic ground truth, for evaluating the assist path (ADR-S19).

``OracleHints`` answers like a good VLM would: the ground-truth elements inside the trigger's tile,
with pixel noise. ``AdversarialHints`` answers with confident nonsense (empty paper, text, hatch,
furniture, off-by-a-wall offsets): none of it may ever enter a plan. Neither is used outside
``make eval`` and the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from archrender.core.schemas.common import ModelRef
from archrender.core.schemas.plan import PlanGraph, Segment
from archrender.plan.assist import Hint, HintKind, Trigger


def _apply(m: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(p, float) @ m[:, :2].T + m[:, 2]


def gt_elements_px(gt: PlanGraph, plan_to_page: list[list[float]]) -> list[Hint]:
    """Ground-truth walls (centre-line ends) and openings (jambs) in page pixels."""
    m = np.array(plan_to_page, float)[:2]
    out: list[Hint] = []
    walls = {w.id: w for w in gt.walls}
    for w in gt.walls:
        if isinstance(w.centerline, Segment):
            a, b = _apply(
                m,
                np.array(
                    [[w.centerline.a.x, w.centerline.a.y], [w.centerline.b.x, w.centerline.b.y]]
                ),
            )
            out.append(Hint("wall", (float(a[0]), float(a[1])), (float(b[0]), float(b[1])), 0.9))
    for o in gt.openings:
        w = walls[o.host_wall]
        if not isinstance(w.centerline, Segment):
            continue
        a = np.array([w.centerline.a.x, w.centerline.a.y])
        b = np.array([w.centerline.b.x, w.centerline.b.y])
        u = (b - a) / np.hypot(*(b - a))
        c, half = o.offset_m.value, o.width_m.value / 2
        j = _apply(m, np.array([a + (c - half) * u, a + (c + half) * u]))
        kind: HintKind = (
            "window" if o.type == "window" else "door" if "door" in o.type else "opening"
        )
        out.append(
            Hint(kind, (float(j[0][0]), float(j[0][1])), (float(j[1][0]), float(j[1][1])), 0.9)
        )
    return out


def _inside(h: Hint, box: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    mx, my = (h.a[0] + h.b[0]) / 2, (h.a[1] + h.b[1]) / 2
    return x0 <= mx <= x1 and y0 <= my <= y1


@dataclass
class OracleHints:
    elements: list[Hint]
    rng: np.random.Generator
    noise_px: float = 6.0
    name: str = "oracle"
    calls: list[Trigger] = field(default_factory=list)

    def model(self) -> ModelRef | None:
        return None

    def hints(
        self, tile: NDArray[np.uint8], origin: tuple[int, int], trigger: Trigger
    ) -> list[Hint]:
        self.calls.append(trigger)
        ox, oy = origin
        box = (float(ox), float(oy), float(ox + tile.shape[1]), float(oy + tile.shape[0]))
        out = []
        for h in self.elements:
            if not _inside(h, box):
                continue
            if trigger.want == "wall" and h.kind != "wall":
                continue
            if trigger.want == "opening" and h.kind == "wall":
                continue
            na, nb = self.rng.normal(0, self.noise_px, 2), self.rng.normal(0, self.noise_px, 2)
            out.append(
                Hint(
                    h.kind, (h.a[0] + na[0], h.a[1] + na[1]), (h.b[0] + nb[0], h.b[1] + nb[1]), 0.8
                )
            )
        return out


@dataclass
class AdversarialHints:
    """Hints wherever the caller wants them (e.g. drawn by hypothesis), whatever the trigger."""

    planted: list[Hint]
    name: str = "adversarial"
    calls: list[Trigger] = field(default_factory=list)

    def model(self) -> ModelRef | None:
        return None

    def hints(
        self, tile: NDArray[np.uint8], origin: tuple[int, int], trigger: Trigger
    ) -> list[Hint]:
        self.calls.append(trigger)
        return list(self.planted)
