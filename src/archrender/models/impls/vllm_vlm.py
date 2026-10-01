"""VLM served by vLLM's OpenAI-compatible server (ADR-S02, ADR-M01).

The server runs as a supervisord program (``vllm-<role>``) from its own environment
(``/opt/venv-vllm``) and is started only once the model's weights are installed. This client:

- waits for the server on ``load`` (model loading takes minutes on first boot),
- sends schema-constrained requests (``response_format: json_schema``, temperature 0),
- puts the server to sleep on ``unload`` so diffusion can use the VRAM (vLLM sleep mode).

UNVERIFIED-ON-GPU: exercised here against a fake OpenAI server only.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import time
from typing import Any

import httpx
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.common import ModelRef
from archrender.core.schemas.understanding import PAGE_CLASSES
from archrender.models.registry import RegistryEntry

VLLM_PORTS = {"vlm": 8101, "judge2": 8102}

CLASS_GUIDE = {
    "floor_plan": "architectural floor plan: walls, doors with swings, room names, dimensions",
    "ceiling_plan": "reflected ceiling plan (tavan planı): room outlines, light fixtures, ceiling heights",
    "section": "building section (kesit): cut floors/slabs, levels, heights",
    "elevation": "elevation / facade (görünüş, cephe)",
    "detail": "construction detail: material layers, hatches, notes, large scale (1:5–1:20)",
    "site_plan": "site plan (vaziyet planı): parcel, building footprint, streets, north arrow",
    "schedule": "table/schedule: door-window list (doğrama listesi), room finish list (mahal listesi)",
    "text_document": "running text: brief, specification, notes",
    "photo": "photograph or rendering of a real space",
    "moodboard": "mood board / material palette: swatches, textures, reference images",
    "other": "anything else: cover sheet, drawing list, charts",
}


def image_data_url(rgb: NDArray[np.uint8], max_px: int = 1536) -> str:
    img = Image.fromarray(rgb)
    img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class VllmVlm:
    def __init__(self, entry: RegistryEntry, *, http: httpx.Client | None = None) -> None:
        self.entry = entry
        port = int(
            os.environ.get(
                f"ARCHRENDER_VLLM_PORT_{entry.role.upper()}", VLLM_PORTS.get(entry.role, 8101)
            )
        )
        self.base = f"http://127.0.0.1:{port}"
        self.http = http or httpx.Client(
            base_url=self.base, timeout=httpx.Timeout(10.0, read=300.0)
        )
        self.ready_timeout_s = float(os.environ.get("ARCHRENDER_VLLM_READY_TIMEOUT_S", "900"))
        self.loaded = False

    def probe(self) -> bool:
        """Is the server up and serving this model right now? (2 s, no waiting)"""
        try:
            r = self.http.get("/v1/models", timeout=2.0)
        except httpx.TransportError:
            return False
        return r.status_code == 200 and any(
            m.get("id") == self.entry.name for m in r.json().get("data", [])
        )

    def load(self) -> None:
        deadline = time.monotonic() + self.ready_timeout_s
        last = ""
        while True:
            try:
                r = self.http.get("/v1/models")
                if r.status_code == 200 and any(
                    m.get("id") == self.entry.name for m in r.json().get("data", [])
                ):
                    break
                last = f"HTTP {r.status_code}"
            except httpx.TransportError as e:
                last = str(e)
            if time.monotonic() > deadline:
                raise ArchRenderError(
                    ErrorCode.MODEL_NOT_DOWNLOADED,
                    f"The vLLM server for {self.entry.name} is not serving at {self.base} ({last}).",
                    "Check `supervisorctl status` and logs/vllm-*.log on the pod; the weights must be "
                    "installed (python -m archrender.ops.models download).",
                )
            time.sleep(2.0)
        # wake it up in case a previous unload put it to sleep
        self.http.post("/wake_up")
        self.loaded = True

    def unload(self) -> None:
        if self.loaded:
            with contextlib.suppress(httpx.TransportError):
                self.http.post("/sleep", params={"level": 1})
        self.loaded = False

    def ref(self) -> ModelRef:
        return self.entry.ref()

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        name: str,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        body = {
            "model": self.entry.name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
        }
        r = self.http.post("/v1/chat/completions", json=body)
        if r.status_code != 200:
            raise ArchRenderError(
                ErrorCode.STAGE_FAILED,
                f"VLM request failed with HTTP {r.status_code}: {r.text[:300]}",
                "See logs/vllm-*.log on the pod.",
                retryable=r.status_code >= 500,
            )
        content = r.json()["choices"][0]["message"]["content"]
        try:
            out = json.loads(content)
        except json.JSONDecodeError as e:
            raise ArchRenderError(
                ErrorCode.STAGE_FAILED,
                "The VLM returned text that is not the requested JSON.",
                "Structured output should prevent this; check the vLLM version (≥ 0.26) and guided decoding backend.",
                retryable=True,
            ) from e
        if not isinstance(out, dict):
            raise ArchRenderError(
                ErrorCode.STAGE_FAILED,
                "The VLM returned JSON that is not an object.",
                "",
                retryable=True,
            )
        return out

    def classify_page(self, preview: NDArray[np.uint8], text_excerpt: str) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "class": {"type": "string", "enum": list(PAGE_CLASSES)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string", "maxLength": 300},
            },
            "required": ["class", "confidence", "evidence"],
            "additionalProperties": False,
        }
        guide = "\n".join(f"- {k}: {v}" for k, v in CLASS_GUIDE.items())
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You classify pages of architectural project documents (Turkish or English). "
                "Answer only with the JSON object requested.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(preview)}},
                    {
                        "type": "text",
                        "text": f"Classes:\n{guide}\n\nText found on the page (may be partial):\n"
                        f"{text_excerpt[:1500]}\n\n"
                        "Which class is this page? Give your confidence and the visual/text evidence.",
                    },
                ],
            },
        ]
        out = self.chat_json(messages, schema, name="page_class")
        if out.get("class") not in PAGE_CLASSES:
            raise ArchRenderError(
                ErrorCode.STAGE_FAILED,
                f"VLM answered an unknown class {out.get('class')!r}.",
                "",
                retryable=True,
            )
        return out

    def locate_elements(self, tile: NDArray[np.uint8], question: str) -> dict[str, Any]:
        """Points on a full-resolution plan tile (ADR-S19): walls as two centre-line ends,
        openings as two jambs, in the tile's pixels. The answer is a hint only; the caller snaps
        it to measured evidence. UNVERIFIED-ON-GPU."""
        h, w = tile.shape[:2]
        if max(h, w) > 1536:
            raise ArchRenderError(
                ErrorCode.STAGE_FAILED,
                f"Tile {w}×{h} px is larger than the VLM input; it would be downscaled.",
                "This is a bug in the caller: tiles are cut at full resolution (≤ 1536 px).",
            )
        point = {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        }
        schema = {
            "type": "object",
            "properties": {
                "elements": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["wall", "door", "window", "opening"],
                            },
                            "points": {
                                "type": "array",
                                "items": point,
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["kind", "points", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["elements"],
            "additionalProperties": False,
        }
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You locate elements on architectural floor plan drawings. Coordinates "
                "are pixels of the given image: x to the right, y down, origin top-left. Answer "
                "only with the JSON object requested.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(tile, max_px=1536)}},
                    {
                        "type": "text",
                        "text": f"Image size: {w}×{h} px.\n{question}\nFor a wall give the two "
                        "end points of its centre line; for a door, window or opening the two "
                        "jambs.",
                    },
                ],
            },
        ]
        out = self.chat_json(messages, schema, name="plan_points", max_tokens=1024)
        for el in out.get("elements", []):
            for x, y in el.get("points", []):
                if not (-0.05 * w <= x <= 1.05 * w and -0.05 * h <= y <= 1.05 * h):
                    raise ArchRenderError(
                        ErrorCode.STAGE_FAILED,
                        f"VLM point ({x}, {y}) lies outside the {w}×{h} tile.",
                        "",
                        retryable=True,
                    )
        return out
