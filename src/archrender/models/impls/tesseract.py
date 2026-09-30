"""Tesseract 5 OCR (Apache-2.0; LSTM models ``tur`` + ``eng`` from the distribution packages).

CPU-only, deterministic and installable everywhere (apt), so it is the verifiable OCR baseline and
the fallback of the VLM OCR. Runs as a separate process under the sandbox limits.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.common import ModelRef
from archrender.models.registry import RegistryEntry
from archrender.models.roles import OcrWord

LANG_CODES = {"tr": "tur", "en": "eng"}


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
        proc = subprocess.run(
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
            input=buf.getvalue(),
            capture_output=True,
            timeout=60,
            check=False,
            env={"OMP_THREAD_LIMIT": "1", "PATH": "/usr/bin:/bin"},
        )
        return proc.stdout.decode("utf-8", errors="replace").strip() if proc.returncode == 0 else ""

    def read(self, img: NDArray[np.uint8], langs: list[str]) -> list[OcrWord]:
        if self.binary is None:
            self.load()
        assert self.binary is not None
        codes = "+".join(
            dict.fromkeys(LANG_CODES.get(lang, lang) for lang in (langs or ["tr", "en"]))
        )
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "PNG")
        proc = subprocess.run(
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
            input=buf.getvalue(),
            capture_output=True,
            timeout=300,
            check=False,
            env={"OMP_THREAD_LIMIT": "1", "PATH": "/usr/bin:/bin"},
        )
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
