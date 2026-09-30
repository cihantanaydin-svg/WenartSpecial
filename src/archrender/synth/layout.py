"""Random apartment layouts (Manhattan) for synthetic sheets.

A rectangular footprint is split recursively into rooms. Every split line gets one door, so the
room graph is connected by construction; habitable rooms on the facade get windows. All lengths
are metres, snapped to 5 cm like real drawings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SNAP = 0.05
EXT_T = 0.25
INT_T = 0.10

ROOM_NAMES_TR = [
    "Salon",
    "Mutfak",
    "Yatak Odası",
    "Yatak Odası",
    "Banyo",
    "Antre",
    "WC",
    "Çalışma Odası",
    "Kiler",
]
ROOM_NAMES_EN = [
    "Living Room",
    "Kitchen",
    "Bedroom",
    "Bedroom",
    "Bathroom",
    "Entrance",
    "WC",
    "Study",
    "Storage",
]


def snap(v: float) -> float:
    return round(round(v / SNAP) * SNAP, 3)


@dataclass
class Room:
    id: str
    name: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


@dataclass
class Wall:
    """Centerline segment (axis-aligned) with thickness."""

    ax: float
    ay: float
    bx: float
    by: float
    thickness: float
    exterior: bool

    @property
    def horizontal(self) -> bool:
        return abs(self.ay - self.by) < 1e-9

    @property
    def length(self) -> float:
        return abs(self.bx - self.ax) + abs(self.by - self.ay)


@dataclass
class Opening:
    tag: str
    kind: str  # door | window
    wall: int  # index into Layout.walls
    t: float  # position of the opening centre along the wall (metres from a)
    width: float
    swing: int = 1  # door swing side (+1/-1)
    sill: float = 0.0
    height: float = 2.1


@dataclass
class Layout:
    width: float
    depth: float
    rooms: list[Room]
    walls: list[Wall]
    openings: list[Opening] = field(default_factory=list)


def _split(
    rng: np.random.Generator,
    rects: list[tuple[float, float, float, float]],
    n: int,
    splits: list[tuple[float, float, float, float]],
) -> None:
    min_dim = 2.4
    while len(rects) < n:
        rects.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
        for i, (x0, y0, x1, y1) in enumerate(rects):
            w, h = x1 - x0, y1 - y0
            vertical = w >= h if abs(w - h) > 0.5 else bool(rng.integers(2))
            span = w if vertical else h
            if span < 2 * min_dim:
                continue
            f = rng.uniform(0.35, 0.65)
            cut = snap((x0 if vertical else y0) + f * span)
            lo, hi = (x0, x1) if vertical else (y0, y1)
            cut = min(max(cut, lo + min_dim), hi - min_dim)
            rects.pop(i)
            if vertical:
                rects += [(x0, y0, cut, y1), (cut, y0, x1, y1)]
                splits.append((cut, y0, cut, y1))
            else:
                rects += [(x0, y0, x1, cut), (x0, cut, x1, y1)]
                splits.append((x0, cut, x1, cut))
            break
        else:
            return


def random_layout(rng: np.random.Generator, *, english: bool = False) -> Layout:
    width = snap(rng.uniform(9.0, 15.0))
    depth = snap(rng.uniform(7.0, 11.5))
    n_rooms = int(rng.integers(5, 9))
    rects: list[tuple[float, float, float, float]] = [(0.0, 0.0, width, depth)]
    splits: list[tuple[float, float, float, float]] = []
    _split(rng, rects, n_rooms, splits)
    rects.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    names = ROOM_NAMES_EN if english else ROOM_NAMES_TR
    order = [0, 1, 2, 3, 5, 7, 8, 6, 4]  # biggest → Salon … smallest → Banyo/WC
    rooms = []
    for i, r in enumerate(rects):
        name = names[order[i]] if i < len(order) else names[2]
        if i == len(rects) - 1:
            name = names[4]  # smallest room is the bathroom
        rooms.append(Room(f"R{i + 1}", name, *r))
    walls = [
        Wall(0, 0, width, 0, EXT_T, True),
        Wall(width, 0, width, depth, EXT_T, True),
        Wall(width, depth, 0, depth, EXT_T, True),
        Wall(0, depth, 0, 0, EXT_T, True),
    ]
    walls += [Wall(ax, ay, bx, by, INT_T, False) for ax, ay, bx, by in splits]
    layout = Layout(width, depth, rooms, walls)
    _add_openings(rng, layout)
    return layout


def _room_at(layout: Layout, x: float, y: float) -> Room | None:
    for r in layout.rooms:
        if r.x0 - 1e-6 <= x <= r.x1 + 1e-6 and r.y0 - 1e-6 <= y <= r.y1 + 1e-6:
            return r
    return None


def _free(layout: Layout, wall: int, t: float, width: float) -> bool:
    """Clear of the wall ends, other openings on the wall and junctions with perpendicular walls."""
    w = layout.walls[wall]
    if t - width / 2 < 0.3 or t + width / 2 > w.length - 0.3:
        return False
    for o in layout.openings:
        if o.wall == wall and abs(o.t - t) < (o.width + width) / 2 + 0.3:
            return False
    px, py = _point(w, t)
    eps = 1e-6
    for j, other in enumerate(layout.walls):
        if j == wall or other.horizontal == w.horizontal:
            continue
        if w.horizontal:
            lo, hi = sorted((other.ay, other.by))
            meets = (
                lo - eps <= w.ay <= hi + eps
                and min(w.ax, w.bx) - eps <= other.ax <= max(w.ax, w.bx) + eps
            )
            if meets and abs(px - other.ax) < width / 2 + 0.2:
                return False
        else:
            lo, hi = sorted((other.ax, other.bx))
            meets = (
                lo - eps <= w.ax <= hi + eps
                and min(w.ay, w.by) - eps <= other.ay <= max(w.ay, w.by) + eps
            )
            if meets and abs(py - other.ay) < width / 2 + 0.2:
                return False
    return True


def _point(w: Wall, t: float) -> tuple[float, float]:
    ux = (w.bx - w.ax) / w.length
    uy = (w.by - w.ay) / w.length
    return w.ax + ux * t, w.ay + uy * t


def _add_openings(rng: np.random.Generator, layout: Layout) -> None:
    doors = windows = 0
    # one door per split wall (connectivity by construction)
    for wi, w in enumerate(layout.walls):
        if w.exterior:
            continue
        width = 0.9 if rng.random() < 0.5 else 0.8
        for _ in range(40):
            t = snap(rng.uniform(0.3 + width / 2, w.length - 0.3 - width / 2))
            if _free(layout, wi, t, width):
                doors += 1
                layout.openings.append(
                    Opening(f"K{doors}", "door", wi, t, width, int(rng.choice([-1, 1])))
                )
                break
    # windows on the facade for rooms with an exterior edge
    for wi, w in enumerate(layout.walls):
        if not w.exterior:
            continue
        pos = 0.0
        while pos < w.length - 1.5:
            width = snap(rng.uniform(1.0, 1.8))
            t = snap(pos + rng.uniform(0.8, 2.5))
            if t + width / 2 < w.length - 0.4 and _free(layout, wi, t, width):
                windows += 1
                layout.openings.append(
                    Opening(f"P{windows}", "window", wi, t, width, sill=0.9, height=1.4)
                )
            pos = t + width / 2 + rng.uniform(1.0, 3.0)
    # the entrance door on the facade (exterior)
    for _ in range(100):
        wi = int(rng.integers(0, 4))
        w = layout.walls[wi]
        t = snap(rng.uniform(1.0, w.length - 1.0))
        if _free(layout, wi, t, 1.0):
            doors += 1
            layout.openings.append(Opening(f"K{doors}", "door", wi, t, 1.0, 1))
            break


def opening_point(layout: Layout, o: Opening) -> tuple[float, float]:
    return _point(layout.walls[o.wall], o.t)
