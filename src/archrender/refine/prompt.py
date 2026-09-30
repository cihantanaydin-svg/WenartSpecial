"""Deterministic prompt compilation from the DesignBrief (templated and logged)."""

from __future__ import annotations

from importlib import resources

from jinja2 import Environment, StrictUndefined

from archrender.core.hashing import sha256_bytes
from archrender.core.schemas.brief import DesignBrief

_TEMPLATE_NAME = "faithful_v1.j2"


def _template_text() -> str:
    return (
        resources.files("archrender.refine")
        .joinpath("templates", _TEMPLATE_NAME)
        .read_text("utf-8")
    )


def compile_prompt(brief: DesignBrief, materials: dict[str, str]) -> tuple[str, str]:
    """Return ``(prompt, template_hash)``. ``materials`` maps surface → human-readable material."""
    text = _template_text()
    env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=False)  # noqa: S701
    prompt = env.from_string(text).render(
        style=brief.style.value,
        cct=int(brief.lighting.cct_k.value),
        mood=brief.lighting.mood,
        surfaces=sorted(materials.items()),
        decor=brief.decor_density,
        avoid=brief.avoid,
    )
    prompt = " ".join(prompt.split())
    return prompt, sha256_bytes(text.encode())[:16]
