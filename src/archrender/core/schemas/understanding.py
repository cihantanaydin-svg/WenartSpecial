"""S1 outputs: page classes, title blocks, north arrows, schedules."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from archrender.core.schemas.common import Strict
from archrender.core.schemas.provenance import Fact

PageClass = Literal[
    "floor_plan",
    "ceiling_plan",  # RCP / Tavan Planı
    "section",
    "elevation",
    "detail",
    "site_plan",
    "schedule",  # door/window schedule, finish (room) schedule
    "text_document",  # brief, specification, notes
    "photo",  # site photo or render
    "moodboard",
    "other",
]
PAGE_CLASSES: tuple[PageClass, ...] = (
    "floor_plan",
    "ceiling_plan",
    "section",
    "elevation",
    "detail",
    "site_plan",
    "schedule",
    "text_document",
    "photo",
    "moodboard",
    "other",
)


class Classification(Strict):
    page_id: str
    label: PageClass
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    sources: dict[str, Any] = Field(default_factory=dict)  # per-source evidence (heuristics, vlm)
    needs_review: bool = False


class TitleBlock(Strict):
    page_id: str
    bbox_px: tuple[float, float, float, float]
    fields: dict[str, Fact[str]] = Field(default_factory=dict)  # project, sheet_title, sheet_no, …
    scale: Fact[float] | None = None  # drawing scale denominator (1:50 → 50)
    level: Fact[str] | None = None


class NorthArrow(Strict):
    page_id: str
    bbox_px: tuple[float, float, float, float]
    angle_deg: Fact[float]  # north direction, counter-clockwise from sheet-up


class ScheduleRow(Strict):
    id: str
    tag: str | None = None  # K1, P3, 101 …
    fields: dict[str, str | float | None]
    source_page: str
    row_index: int
    links: list[dict[str, Any]] = Field(default_factory=list)  # [{page_id, bbox_px, text}]


class Schedule(Strict):
    id: str
    kind: Literal["door_window", "finish", "unknown"]
    source_page: str
    header: list[str]
    columns: dict[str, str]  # canonical field → original header
    rows: list[ScheduleRow]
