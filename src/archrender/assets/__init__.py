"""Bundled assets (fonts for synthetic sheets and reports). See ``fonts/LICENSE-DejaVu.txt``."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

FONTS = {
    "sans": "DejaVuSans.ttf",
    "sans-bold": "DejaVuSans-Bold.ttf",
    "mono": "DejaVuSansMono.ttf",
}


def font_path(name: str = "sans") -> Path:
    """Path of a bundled TTF (DejaVu covers Turkish ç ğ ı İ ö ş ü)."""
    return Path(str(resources.files(__name__).joinpath("fonts", FONTS[name])))
