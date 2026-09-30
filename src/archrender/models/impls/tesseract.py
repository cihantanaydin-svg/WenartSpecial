"""Tesseract 5 OCR (Apache-2.0; LSTM models ``tur`` + ``eng`` from the distribution packages).

CPU-only, deterministic and installable everywhere (apt), so it is the verifiable OCR baseline and
the fallback of the VLM OCR. Runs as a separate process with a timeout.

Noisy input is despeckled (3×3 median on grey) before sparse-text OCR: on noisy scans
``--psm 11`` treats every speck as a text candidate and can spend minutes on one tile. Clean input
is left alone, since the median costs small text on clean scans a few dimension strings.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.common import ModelRef
from archrender.models.registry import RegistryEntry
from archrender.models.roles import OcrWord

LANG_CODES = {"tr": "tur", "en": "eng"}
TIMEOUT_S = 120
# pixel-noise σ above which a tile is despeckled; measured on synthetic scans: clean ≤ 1.5,
# "medium" 1.5–6, "noisy" 7–13 (grey levels)
DESPECKLE_SIGMA = 3.0


def noise_sigma(gray: NDArray[np.uint8]) -> float:
    """Robust pixel-noise estimate: MAD of the residual after a 3×3 median (text is sparse)."""
    residual = gray.astype(np.int16) - cv2.medianBlur(gray, 3)
    return 1.4826 * float(np.median(np.abs(residual)))


class TesseractOcr:
    def __init__(self, entry: RegistryEntry) -> None:
        self.entry = entry
        self.binary: str | None = None
        self.version = ""
        self.languages: set[str] = set()

    def load(self) -> None:
        self.binary = shutil.which("tesseract")
        if self.binary is None:
            raise ArchRenderError(
                ErrorCode.MODEL_NOT_DOWNLOADED,
                "Tesseract is not installed.",
                "Install tesseract-ocr, tesseract-ocr-tur and tesseract-ocr-eng (the release image has them).",
            )
        out = subprocess.run(
            [self.binary, "--version"], capture_output=True, text=True, check=False
        )
        self.version = (
            (out.stdout or out.stderr).splitlines()[0].strip()
            if (out.stdout or out.stderr)
            else "?"
        )
        langs = subprocess.run(
            [self.binary, "--list-langs"], capture_output=True, text=True, check=False
        )
        self.languages = {line.strip() for line in langs.stdout.splitlines()[1:] if line.strip()}
        missing = {"tur", "eng"} - self.languages
        if missing:
            raise ArchRenderError(
                ErrorCode.MODEL_NOT_DOWNLOADED,
                f"Tesseract language data missing: {sorted(missing)}.",
                "Install tesseract-ocr-tur and tesseract-ocr-eng.",
            )

    def unload(self) -> None:
        pass

    def ref(self) -> ModelRef:
        r = self.entry.ref()
        return r.model_copy(update={"revision": self.version or r.revision})

    def read_line(self, img: NDArray[np.uint8], whitelist: str) -> str:
        """Read one short line (e.g. a door tag inside a bubble) with a character whitelist."""
        if self.binary is None:
            self.load()
        assert self.binary is not None
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "PNG")
        try:
            proc = self._run(
                [
                    self.binary,
                    "stdin",
                    "stdout",
                    "-l",
                    "eng",
                    "--psm",
                    "7",
                    "--oem",
                    "1",
                    "-c",
                    f"tessedit_char_whitelist={whitelist}",
                ],
                buf.getvalue(),
            )
        except subprocess.TimeoutExpired:
            return ""  # no reading: the caller treats the bubble as unread
        return proc.stdout.decode("utf-8", errors="replace").strip() if proc.returncode == 0 else ""

    @staticmethod
    def _run(cmd: list[str], data: bytes) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            cmd,
            input=data,
            capture_output=True,
            timeout=TIMEOUT_S,
            check=False,
            env={"OMP_THREAD_LIMIT": "1", "PATH": "/usr/bin:/bin"},
        )

    def read(self, img: NDArray[np.uint8], langs: list[str]) -> list[OcrWord]:
        if self.binary is None:
            self.load()
        assert self.binary is not None
        codes = "+".join(
            dict.fromkeys(LANG_CODES.get(lang, lang) for lang in (langs or ["tr", "en"]))
        )
        gray: NDArray[np.uint8] = np.ascontiguousarray(
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img, dtype=np.uint8
        )
        if noise_sigma(gray) > DESPECKLE_SIGMA:
            gray = np.asarray(cv2.medianBlur(gray, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(gray).save(buf, "PNG")
        try:
            proc = self._run(
                [
                    self.binary,
                    "stdin",
                    "stdout",
                    "-l",
                    codes,
                    "--psm",
                    "11",
                    "--oem",
                    "1",
                    "--dpi",
                    "300",
                    "tsv",
                ],
                buf.getvalue(),
            )
        except subprocess.TimeoutExpired as e:
            raise ArchRenderError(
                ErrorCode.STAGE_FAILED,
                f"Tesseract did not finish a {img.shape[1]}×{img.shape[0]} px tile in {TIMEOUT_S} s.",
                "The words on this tile are missing; check the page in the review queue or type "
                "the facts you need.",
                retryable=False,
            ) from e
        if proc.returncode != 0:
            raise ArchRenderError(
                ErrorCode.STAGE_FAILED,
                f"Tesseract failed: {proc.stderr.decode(errors='replace')[-300:]}",
                "Check the page raster; this is not expected for valid images.",
                retryable=True,
            )
        words: list[OcrWord] = []
        rows = csv.DictReader(
            io.StringIO(proc.stdout.decode("utf-8", errors="replace")),
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )
        for row in rows:
            if row.get("level") != "5":
                continue
            text = (row.get("text") or "").strip()
            conf = float(row.get("conf") or -1)
            if not text or conf < 0:
                continue
            x, y, w, h = (float(row[k]) for k in ("left", "top", "width", "height"))
            words.append(OcrWord(text, (x, y, x + w, y + h), conf / 100.0))
        return words
