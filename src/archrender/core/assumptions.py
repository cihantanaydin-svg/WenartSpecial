"""Assumption register (principle 3).

Stage code never uses a bare default for a physical quantity. It calls
``register.use(key, value, reason)``, which records the assumption so it can be shown in the UI and
printed in the QA report. Defaults live in ``DEFAULTS`` with their justification.
"""

from __future__ import annotations

from typing import Any

from archrender.core.schemas.provenance import Assumption

# Documented defaults (metres, degrees, kelvin). Each entry: value, reason.
DEFAULTS: dict[str, tuple[Any, str]] = {
    "ceiling_height_m": (
        2.70,
        "Typical residential clear height in Türkiye; no section/annotation found.",
    ),
    "door_head_m": (2.10, "Standard door head height; no schedule/elevation value found."),
    "window_sill_m": (0.90, "Standard window sill height; no schedule/elevation value found."),
    "window_head_m": (2.10, "Window head aligned with door heads; no value found."),
    "wall_thickness_interior_m": (0.10, "Typical interior partition; no drawing evidence."),
    "wall_thickness_exterior_m": (0.25, "Typical exterior wall; no drawing evidence."),
    "camera_height_m": (1.30, "Architectural eye-level convention."),
    "camera_focal_mm": (24.0, "24 mm full-frame equivalent interior default."),
    "sun_datetime_local": (
        "2026-06-21T15:00:00",
        "Mid-afternoon on the summer solstice; no brief value.",
    ),
    "lighting_cct_k": (4000.0, "Neutral white; brief does not specify colour temperature."),
    "skirting_height_m": (0.08, "Common skirting height; no finish schedule value."),
}


class AssumptionRegister:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self._items: dict[str, Assumption] = {}

    def use(self, key: str, value: Any, reason: str, *, requires_review: bool = False) -> Any:
        self._items[key] = Assumption(
            key=key, value=value, reason=reason, stage=self.stage, requires_review=requires_review
        )
        return value

    def default(self, key: str, *, requires_review: bool = False) -> Any:
        value, reason = DEFAULTS[key]
        return self.use(key, value, reason, requires_review=requires_review)

    def items(self) -> list[Assumption]:
        return list(self._items.values())

    def extend(self, items: list[Assumption]) -> None:
        for a in items:
            self._items[a.key] = a
