"""S0 intake: detect → sandboxed parse → CAS blobs + ``documents``/``pages`` rows.

Every parser of untrusted content runs in a resource-limited child process
(:mod:`archrender.ingest.sandbox`). Archives are unpacked recursively (depth-limited); embedded
images of Office files and archive entries become child documents. A child that cannot be ingested
is reported in ``skipped`` with its coded error, and never silently dropped.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from archrender.core.cas import CasRef, ProjectStore
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.hashing import sha256_file
from archrender.core.ids import new_id, now_iso
from archrender.core.paths import sanitize_filename
from archrender.core.schemas.document import IntakeResult, PageRef, SkippedEntry, Tile
from archrender.ingest.detect import Detected, check_supported, detect
from archrender.ingest.sandbox import run_task, run_tool, which
from archrender.pipeline.services import Services

LIBREDWG_DEFAULT = "/opt/libredwg/bin/dwg2dxf"  # deploy/libredwg/build.sh
RASTER_KINDS = {"png", "jpeg", "webp", "tiff"}
TEXT_LIMIT = 2_000_000  # characters kept from a plain-text document


def tiles(width: int, height: int, tile_px: int, overlap: float) -> list[Tile]:
    """Cover the raster with ``tile_px`` squares overlapping by ``overlap`` (last row/col flush)."""
    stride = max(1, int(tile_px * (1.0 - overlap)))

    def starts(n: int) -> list[int]:
        if n <= tile_px:
            return [0]
        s = list(range(0, n - tile_px, stride))
        s.append(n - tile_px)
        return s

    return [
        Tile(x=x, y=y, w=min(tile_px, width), h=min(tile_px, height))
        for y in starts(height)
        for x in starts(width)
    ]


@dataclass
class _Run:
    svc: Services
    project_id: str
    store: ProjectStore
    skipped: list[SkippedEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _page_from_reply(
    run: _Run, doc_id: str, out: Path, p: dict[str, Any], extra_meta: dict[str, Any]
) -> PageRef:
    s = run.svc.settings

    def blob(key: str, media: str) -> CasRef | None:
        name = p.get(key)
        if not name:
            return None
        return run.store.put_file(out / name, media, f"{doc_id}_{name}")

    raster = blob("raster", "image/png")
    preview = blob("preview", "image/png")
    meta = {**p.get("meta", {}), **extra_meta}
    if preview is not None:
        meta["preview"] = preview.model_dump()
    w, h = p.get("width_px"), p.get("height_px")
    return PageRef(
        id=f"{doc_id}_p{p['index']}",
        document_id=doc_id,
        index=p["index"],
        kind=p["kind"],
        width_px=w,
        height_px=h,
        dpi=p.get("dpi"),
        width_mm=p.get("width_mm"),
        height_mm=p.get("height_mm"),
        raster=raster,
        words=blob("words", "application/json"),
        content=blob("content", "application/json"),
        tiles=tiles(w, h, s.tile_px, s.tile_overlap) if raster is not None and w and h else [],
        meta=meta,
    )


def _heic_to_png(run: _Run, src: Path, work: Path) -> Path:
    s = run.svc.settings
    candidates = [s.heif_bin] if s.heif_bin else ["heif-dec", "heif-convert"]
    binary = next((b for b in candidates if b and which(b)), None)
    if binary is None:
        raise ArchRenderError(
            ErrorCode.INGEST_TOOL_MISSING,
            "HEIC images need libheif's heif-dec (or heif-convert), which is not installed.",
            "Use the release image (it includes libheif with the decoder plugin only), or convert "
            "the photo to JPEG/PNG before uploading.",
        )
    dest = work / "heic.png"
    proc = run_tool(
        [binary, str(src), str(dest)],
        s,
        work,
        what="HEIC decoding",
        fix_hint="Convert the photo to JPEG/PNG and upload that.",
    )
    if proc.returncode != 0 or not dest.exists():
        raise ArchRenderError(
            ErrorCode.INGEST_CORRUPT,
            f"The HEIC image cannot be decoded ({proc.stderr.strip()[:300]}).",
            "Convert the photo to JPEG/PNG and upload that.",
        )
    return dest


def _dwg_to_dxf(run: _Run, src: Path, work: Path) -> tuple[Path, dict[str, Any]]:
    s = run.svc.settings
    binary = which(s.dwg2dxf_bin) or which(LIBREDWG_DEFAULT)
    if binary is None:
        raise ArchRenderError(
            ErrorCode.INGEST_TOOL_MISSING,
            "DWG files need LibreDWG's dwg2dxf, which is not installed.",
            "Use the release image (it includes LibreDWG), or export DXF or PDF from the CAD "
            "application and upload that.",
        )
    dest = work / "converted.dxf"
    proc = run_tool(
        [binary, "-y", "-o", str(dest), str(src)],
        s,
        work,
        what="DWG conversion (LibreDWG)",
        fix_hint="Export DXF or PDF from the CAD application and upload that.",
    )
    # LibreDWG exit codes are an error bitmask; values below 128 are non-critical.
    if (
        proc.returncode >= 128
        or proc.returncode < 0
        or not dest.exists()
        or dest.stat().st_size == 0
    ):
        raise ArchRenderError(
            ErrorCode.INGEST_DWG_CONVERSION_FAILED,
            f"LibreDWG could not convert the DWG (status {proc.returncode}: "
            f"{proc.stderr.strip()[-300:]}).",
            "Export DXF (R2010 or later) or PDF from the CAD application and upload that.",
        )
    version = run_tool([binary, "--version"], s, work, what="dwg2dxf --version", fix_hint="")
    tool_version = version.stdout.strip().split()[-1] if version.stdout.strip() else "?"
    meta: dict[str, Any] = {"converted_by": f"LibreDWG {tool_version}"}
    if proc.returncode:
        meta["conversion_warnings"] = proc.returncode
        run.warnings.append(f"LibreDWG reported non-critical issues (code {proc.returncode})")
    return dest, meta


def _ifc_header(path: Path) -> dict[str, Any]:
    head = path.open("rb").read(16384).decode("latin-1", errors="replace")
    schema = None
    for token in ("IFC4X3_ADD2", "IFC4X3", "IFC4", "IFC2X3"):
        if token in head:
            schema = token
            break
    return {"schema": schema}


def _parse(
    run: _Run, detected: Detected, src: Path, work: Path, doc_id: str
) -> tuple[list[PageRef], list[dict[str, Any]], dict[str, Any]]:
    """Returns (pages, children, document meta)."""
    s = run.svc.settings
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    kind = detected.kind
    extra: dict[str, Any] = {}
    if kind == "heic":
        src = _heic_to_png(run, src, work)
        extra["converted_from"] = "heic"
        kind = "png"
    if kind == "dwg":
        dxf, conv_meta = _dwg_to_dxf(run, src, work)
        dxf_ref = run.store.put_file(dxf, "image/vnd.dxf", f"{doc_id}_converted.dxf")
        extra.update(conv_meta, dxf=dxf_ref.model_dump())
        src, kind = dxf, "dxf"
    if kind == "pdf":
        req: dict[str, Any] = {
            "dpi": s.pdf_dpi,
            "max_pages": s.pdf_max_pages,
            "max_page_pixels": s.pdf_max_page_pixels,
        }
        reply = run_task("pdf", {"path": str(src), "out": str(out), **req}, s, work)
    elif kind in RASTER_KINDS:
        reply = run_task(
            "image", {"path": str(src), "out": str(out), "max_pixels": s.image_max_pixels}, s, work
        )
    elif kind == "dxf":
        reply = run_task("dxf", {"path": str(src), "out": str(out), "preview_px": 4096}, s, work)
    elif kind in ("docx", "pptx"):
        reply = run_task(kind, {"path": str(src), "out": str(out)}, s, work)
    elif kind in ("xlsx", "xlsm"):
        req = {
            "path": str(src),
            "out": str(out),
            "max_cells": s.office_max_cells,
            "macro": kind == "xlsm",
        }
        reply = run_task("xlsx", req, s, work)
    elif kind == "zip":
        req = {
            "path": str(src),
            "out": str(out),
            "max_total_bytes": s.zip_max_total_bytes,
            "max_ratio": s.zip_max_ratio,
            "max_entries": s.zip_max_entries,
        }
        reply = run_task("unzip", req, s, work)
    elif kind in ("txt", "md"):
        text = src.read_bytes()[: TEXT_LIMIT * 4].decode("utf-8", errors="replace")[:TEXT_LIMIT]
        (out / "p0_text.json").write_text(
            json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8"
        )
        reply = {"pages": [{"index": 0, "kind": "text", "content": "p0_text.json", "meta": {}}]}
    elif kind == "ifc":
        reply = {"pages": [{"index": 0, "kind": "ifc", "meta": _ifc_header(src)}]}
    elif kind == "svg":
        reply = {"pages": [{"index": 0, "kind": "svg", "meta": {}}]}
    elif kind == "3dm":
        reply = {"pages": [{"index": 0, "kind": "model_3dm", "meta": {}}]}
    else:
        raise ArchRenderError(
            ErrorCode.INGEST_UNSUPPORTED_TYPE,
            f"No intake handler for '{kind}'.",
            "Report this as a bug.",
        )
    run.warnings.extend(reply.get("warnings", []))
    run.skipped.extend(SkippedEntry.model_validate(x) for x in reply.get("skipped", []))
    pages = [_page_from_reply(run, doc_id, out, p, extra) for p in reply.get("pages", [])]
    children = [{"path": out / c["path"], "name": c["name"]} for c in reply.get("children", [])]
    return (
        pages,
        children,
        {**reply.get("meta", {}), **{k: v for k, v in extra.items() if k != "dxf"}},
    )


def ingest_file(
    svc: Services,
    project_id: str,
    path: Path,
    filename: str,
    *,
    parent_id: str | None = None,
    depth: int = 0,
    meta: dict[str, Any] | None = None,
    run: _Run | None = None,
) -> IntakeResult:
    """Ingest one file (and, recursively, its children). Raises on failure of *this* file."""
    run = run or _Run(svc, project_id, svc.store(project_id))
    detected = detect(path)
    check_supported(detected, filename)
    sha = sha256_file(path)
    existing = svc.db.one(
        "SELECT id, kind FROM documents WHERE project_id = ? AND sha256 = ?", (project_id, sha)
    )
    if existing is not None:
        n = svc.db.one("SELECT COUNT(*) AS n FROM pages WHERE document_id = ?", (existing["id"],))
        return IntakeResult(
            document_id=existing["id"],
            kind=existing["kind"],
            sha256=sha,
            deduplicated=True,
            pages=int(n["n"]) if n else 0,
        )
    doc_id = new_id("doc")
    work = run.store.scratch_dir("intake")
    try:
        # parse first: a file that fails intake never enters the project store
        pages, children, doc_meta = _parse(run, detected, path, work, doc_id)
        ref = run.store.put_file(path, detected.media_type, filename)
        with svc.db.tx(immediate=True) as c:
            c.execute(
                "INSERT INTO documents(id, project_id, sha256, filename, kind, media_type, size, parent_id,"
                " meta_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    project_id,
                    ref.sha256,
                    filename,
                    detected.kind,
                    detected.media_type,
                    ref.size,
                    parent_id,
                    json.dumps({**(meta or {}), **doc_meta}, ensure_ascii=False),
                    now_iso(),
                ),
            )
            for pg in pages:
                c.execute(
                    "INSERT INTO pages(id, document_id, project_id, idx, kind, page_json) VALUES (?,?,?,?,?,?)",
                    (pg.id, doc_id, project_id, pg.index, pg.kind, pg.model_dump_json()),
                )
        child_ids: list[str] = []
        for child in children:
            name = str(child["name"])
            if detected.kind == "zip" and depth + 1 > svc.settings.zip_max_depth:
                run.skipped.append(
                    SkippedEntry(
                        name=name,
                        code=ErrorCode.INGEST_UNSAFE_ARCHIVE.value,
                        message=f"Archives nested deeper than {svc.settings.zip_max_depth} levels are not unpacked.",
                        fix_hint="Flatten the archive.",
                    )
                )
                continue
            display = sanitize_filename(PurePosixPath(name).name) or "entry"
            try:
                r = ingest_file(
                    svc,
                    project_id,
                    Path(child["path"]),
                    display,
                    parent_id=doc_id,
                    depth=depth + 1,
                    meta={"archive_path": name}
                    if detected.kind == "zip"
                    else {"embedded_in": filename},
                    run=run,
                )
                child_ids.append(r.document_id)
            except ArchRenderError as e:
                run.skipped.append(
                    SkippedEntry(
                        name=name, code=e.code.value, message=e.message, fix_hint=e.fix_hint
                    )
                )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return IntakeResult(
        document_id=doc_id,
        kind=detected.kind,
        sha256=ref.sha256,
        deduplicated=False,
        pages=len(pages),
        children=child_ids,
        skipped=list(run.skipped) if depth == 0 else [],
        warnings=list(run.warnings) if depth == 0 else [],
    )


def page_refs(svc: Services, project_id: str, document_id: str | None = None) -> list[PageRef]:
    sql = "SELECT page_json FROM pages WHERE project_id = ?"
    args: tuple[str, ...] = (project_id,)
    if document_id is not None:
        sql += " AND document_id = ?"
        args = (project_id, document_id)
    rows = svc.db.query(sql + " ORDER BY document_id, idx", args)
    return [PageRef.model_validate_json(r["page_json"]) for r in rows]
