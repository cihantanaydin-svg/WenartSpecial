"""Chunked, resumable uploads (ADR-S08) and the intake job that assembles them."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from archrender.core.errors import ArchRenderError, ErrorCode, not_found
from archrender.core.ids import new_id, now_iso
from archrender.core.paths import check_id, check_sha256, safe_join, sanitize_filename
from archrender.core.schemas.document import IntakeResult
from archrender.core.schemas.jobs import JobKind
from archrender.ingest.intake import ingest_file
from archrender.pipeline.services import Services


def upload_dir(svc: Services, project_id: str, upload_id: str) -> Path:
    return safe_join(
        svc.settings.projects_dir(), check_id(project_id), "uploads", check_id(upload_id)
    )


def create_upload(
    svc: Services, project_id: str, filename: str, size: int, sha256: str, user_id: str
) -> dict[str, Any]:
    check_sha256(sha256)
    if size <= 0:
        raise ArchRenderError(
            ErrorCode.VALIDATION, "Empty files cannot be uploaded.", "Choose a non-empty file."
        )
    if size > svc.settings.max_upload_bytes:
        raise ArchRenderError(
            ErrorCode.INGEST_TOO_LARGE,
            f"File is {size / 1e9:.2f} GB; the limit is {svc.settings.max_upload_bytes / 1e9:.2f} GB.",
            "Split the drawing set or zip it with fewer pages per file.",
        )
    used = svc.db.one(
        "SELECT COALESCE(SUM(size), 0) AS s FROM documents WHERE project_id = ?", (project_id,)
    )
    if used is not None and used["s"] + size > svc.settings.max_project_bytes:
        raise ArchRenderError(
            ErrorCode.INGEST_TOO_LARGE,
            "This upload would exceed the project storage limit.",
            "Remove unused documents or ask an admin to raise ARCHRENDER_MAX_PROJECT_BYTES.",
        )
    uid = new_id("upl")
    chunk = svc.settings.upload_chunk_bytes
    with svc.db.tx(immediate=True) as c:
        c.execute(
            "INSERT INTO uploads(id, project_id, filename, size, sha256, chunk_size, status, created_by, created_at)"
            " VALUES (?,?,?,?,?,?, 'open', ?, ?)",
            (uid, project_id, sanitize_filename(filename), size, sha256, chunk, user_id, now_iso()),
        )
    upload_dir(svc, project_id, uid).mkdir(parents=True, exist_ok=True)
    return {"upload_id": uid, "chunk_size": chunk, "chunks": (size + chunk - 1) // chunk}


def _upload_row(svc: Services, upload_id: str) -> Any:
    row = svc.db.one("SELECT * FROM uploads WHERE id = ?", (upload_id,))
    if row is None:
        raise not_found("Upload", upload_id)
    return row


def put_chunk(
    svc: Services, upload_id: str, index: int, data: bytes, chunk_sha256: str | None
) -> dict[str, Any]:
    row = _upload_row(svc, upload_id)
    if row["status"] != "open":
        raise ArchRenderError(ErrorCode.CONFLICT, "Upload is not open.", "Start a new upload.")
    n_chunks = (row["size"] + row["chunk_size"] - 1) // row["chunk_size"]
    if not 0 <= index < n_chunks:
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            f"Chunk index {index} out of range 0..{n_chunks - 1}.",
            "Check the chunk index.",
        )
    expected = (
        row["chunk_size"]
        if index < n_chunks - 1
        else row["size"] - row["chunk_size"] * (n_chunks - 1)
    )
    if len(data) != expected:
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            f"Chunk {index} has {len(data)} bytes; expected {expected}.",
            "Send exactly chunk_size bytes per chunk (the last chunk holds the remainder).",
        )
    digest = hashlib.sha256(data).hexdigest()
    if chunk_sha256 is not None and digest != chunk_sha256:
        raise ArchRenderError(
            ErrorCode.INGEST_CHECKSUM_MISMATCH,
            f"Chunk {index} checksum mismatch.",
            "The chunk was corrupted in transit; resend it.",
            retryable=True,
        )
    d = upload_dir(svc, row["project_id"], upload_id)
    fd, tmp = tempfile.mkstemp(dir=d)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, d / f"{index:06d}.part")
    with svc.db.tx(immediate=True) as c:
        c.execute(
            "INSERT OR REPLACE INTO upload_chunks(upload_id, idx, sha256, size) VALUES (?,?,?,?)",
            (upload_id, index, digest, len(data)),
        )
    return {"index": index, "sha256": digest}


def upload_status(svc: Services, upload_id: str) -> dict[str, Any]:
    row = _upload_row(svc, upload_id)
    got = [
        r["idx"]
        for r in svc.db.query(
            "SELECT idx FROM upload_chunks WHERE upload_id = ? ORDER BY idx", (upload_id,)
        )
    ]
    n_chunks = (row["size"] + row["chunk_size"] - 1) // row["chunk_size"]
    return {
        "upload_id": upload_id,
        "project_id": row["project_id"],
        "filename": row["filename"],
        "status": row["status"],
        "chunk_size": row["chunk_size"],
        "chunks": n_chunks,
        "received": got,
        "missing": sorted(set(range(n_chunks)) - set(got)),
        "document_id": row["document_id"],
    }


def complete_upload(svc: Services, upload_id: str, user_id: str) -> str:
    st = upload_status(svc, upload_id)
    if st["missing"]:
        raise ArchRenderError(
            ErrorCode.INGEST_INCOMPLETE_UPLOAD,
            f"{len(st['missing'])} chunk(s) missing.",
            "Resume the upload: GET the upload status and send the missing chunks.",
            context={"missing": st["missing"][:50]},
        )
    return svc.queue.enqueue(
        st["project_id"], JobKind.INTAKE, "cpu", {"upload_id": upload_id}, created_by=user_id
    )


def run_intake(svc: Services, upload_id: str) -> dict[str, Any]:
    """Assemble chunks, verify the declared SHA-256, detect the type and register the document."""
    row = _upload_row(svc, upload_id)
    project_id = row["project_id"]
    if row["status"] == "complete" and row["document_id"]:
        doc = svc.db.one(
            "SELECT kind, sha256, (SELECT COUNT(*) FROM pages WHERE document_id = documents.id) AS n"
            " FROM documents WHERE id = ?",
            (row["document_id"],),
        )
        if doc is None:
            raise not_found("Document", row["document_id"])
        return IntakeResult(
            document_id=row["document_id"],
            kind=doc["kind"],
            sha256=doc["sha256"],
            deduplicated=True,
            pages=int(doc["n"]),
        ).model_dump(mode="json")
    d = upload_dir(svc, project_id, upload_id)
    assembled = d / "assembled.bin"
    h = hashlib.sha256()
    with assembled.open("wb") as out:
        for part in sorted(d.glob("*.part")):
            with part.open("rb") as fh:
                while buf := fh.read(1 << 20):
                    h.update(buf)
                    out.write(buf)
    if h.hexdigest() != row["sha256"]:
        raise ArchRenderError(
            ErrorCode.INGEST_CHECKSUM_MISMATCH,
            "Assembled file does not match the declared SHA-256.",
            "Restart the upload; the file changed or chunks were mixed up.",
        )
    try:
        result = ingest_file(
            svc, project_id, assembled, row["filename"], meta={"upload_id": upload_id}
        )
    finally:
        assembled.unlink(missing_ok=True)
    with svc.db.tx(immediate=True) as c:
        c.execute(
            "UPDATE uploads SET status = 'complete', document_id = ? WHERE id = ?",
            (result.document_id, upload_id),
        )
    shutil.rmtree(d, ignore_errors=True)
    return result.model_dump(mode="json")
