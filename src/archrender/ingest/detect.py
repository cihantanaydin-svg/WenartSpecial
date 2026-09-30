"""File type detection by content (magic bytes), never by extension."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from archrender.core.errors import ArchRenderError, ErrorCode

HEAD = 8192


@dataclass(frozen=True)
class Detected:
    kind: str
    media_type: str


_OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _zip_kind(path: Path) -> Detected:
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist()[:5000])
    except zipfile.BadZipFile as e:
        raise ArchRenderError(
            ErrorCode.INGEST_UNSUPPORTED_TYPE,
            "Corrupt ZIP container.",
            "Re-export or re-zip the file.",
        ) from e
    if "[Content_Types].xml" in names:
        if any(n.startswith("word/") for n in names):
            if "word/vbaProject.bin" in names:
                return Detected("docm", "application/vnd.ms-word.document.macroEnabled.12")
            return Detected(
                "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        if any(n.startswith("xl/") for n in names):
            if "xl/vbaProject.bin" in names:
                return Detected("xlsm", "application/vnd.ms-excel.sheet.macroEnabled.12")
            return Detected(
                "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        if any(n.startswith("ppt/") for n in names):
            if "ppt/vbaProject.bin" in names:
                return Detected(
                    "pptm", "application/vnd.ms-powerpoint.presentation.macroEnabled.12"
                )
            return Detected(
                "pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
    return Detected("zip", "application/zip")


def detect(path: Path) -> Detected:
    with path.open("rb") as fh:
        head = fh.read(HEAD)
    text_head = head.lstrip(b"\xef\xbb\xbf").lstrip()
    if head.startswith(b"%PDF-"):
        return Detected("pdf", "application/pdf")
    if head[:4] == b"\x89PNG":
        return Detected("png", "image/png")
    if head[:3] == b"\xff\xd8\xff":
        return Detected("jpeg", "image/jpeg")
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return Detected("webp", "image/webp")
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return Detected("tiff", "image/tiff")
    if head[4:8] == b"ftyp" and head[8:12] in (
        b"heic",
        b"heix",
        b"mif1",
        b"msf1",
        b"heim",
        b"heis",
        b"hevc",
    ):
        return Detected("heic", "image/heic")
    if head[:2] == b"PK":
        return _zip_kind(path)
    if head[:4] == b"AC10" or head[:6] in (b"AC1.40", b"AC1.50", b"AC2.10"):
        return Detected("dwg", "image/vnd.dwg")
    if head.startswith(b"ISO-10303-21"):
        return Detected("ifc", "application/x-step")
    if head.startswith(b"AutoCAD Binary DXF"):
        return Detected("dxf", "image/vnd.dxf")
    if head.startswith(b"3D Geometry File Format"):
        return Detected("3dm", "model/vnd.3dm")
    if head[:8] == _OLE:
        if (
            b"B\x00a\x00s\x00i\x00c\x00F\x00i\x00l\x00e\x00I\x00n\x00f\x00o" in head
            or b"Revit" in head
        ):
            return Detected("rvt", "application/vnd.autodesk.revit")
        return Detected("ole", "application/x-ole-storage")
    if (
        head.startswith(b"\xff\xfe\xff\x0eSketchUp Model")
        or b"S\x00k\x00e\x00t\x00c\x00h\x00U\x00p\x00" in head[:64]
    ):
        return Detected("skp", "application/vnd.sketchup.skp")
    try:
        text = text_head.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is not None:
        stripped = text.lstrip()
        if (
            stripped.startswith("0\n")
            or stripped.startswith("0\r\n")
            or stripped[:40].replace("\r", "").startswith("  0\nSECTION")
        ) and "SECTION" in text[:200]:
            return Detected("dxf", "image/vnd.dxf")
        low = stripped[:512].lower()
        if low.startswith("<?xml") or low.startswith("<svg"):
            if "<svg" in stripped[:4096].lower():
                return Detected("svg", "image/svg+xml")
            return Detected("xml", "application/xml")
        if stripped.startswith("#") or path.suffix.lower() == ".md":
            return Detected("md", "text/markdown")
        return Detected("txt", "text/plain")
    raise ArchRenderError(
        ErrorCode.INGEST_UNSUPPORTED_TYPE,
        f"Unrecognised file content ({path.name}).",
        "Supported: PDF, DXF, DWG, IFC, SVG, 3DM, JPG/PNG/WebP/TIFF/HEIC, DOCX/XLSX/PPTX, TXT/MD, ZIP.",
    )


SUPPORTED = {
    "pdf",
    "dxf",
    "dwg",
    "ifc",
    "svg",
    "3dm",
    "png",
    "jpeg",
    "webp",
    "tiff",
    "heic",
    "zip",
    "docx",
    "xlsx",
    "xlsm",  # read as data only; the VBA project is ignored, never executed
    "pptx",
    "txt",
    "md",
}


def check_supported(d: Detected, filename: str) -> None:
    if d.kind == "rvt":
        raise ArchRenderError(
            ErrorCode.INGEST_UNSUPPORTED_NATIVE,
            f"{filename} is a native Revit model (RVT).",
            "Export IFC (preferred), DWG or PDF from Revit and upload that instead.",
        )
    if d.kind == "skp":
        raise ArchRenderError(
            ErrorCode.INGEST_UNSUPPORTED_NATIVE,
            f"{filename} is a native SketchUp model (SKP).",
            "Export IFC, DWG or PDF from SketchUp and upload that instead.",
        )
    if d.kind in ("docm", "pptm"):
        plain = "docx" if d.kind == "docm" else "pptx"
        raise ArchRenderError(
            ErrorCode.INGEST_UNSUPPORTED_TYPE,
            f"{filename} is a macro-enabled Office file ({d.kind}).",
            f"Save it as a plain .{plain}; macros are never executed or accepted.",
        )
    if d.kind not in SUPPORTED:
        raise ArchRenderError(
            ErrorCode.INGEST_UNSUPPORTED_TYPE,
            f"{filename}: unsupported content type '{d.kind}'.",
            "Supported: PDF, DXF, DWG, IFC, SVG, 3DM, images, DOCX/XLSX/PPTX, TXT/MD, ZIP.",
        )
