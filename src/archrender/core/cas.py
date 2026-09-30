"""Per-project content-addressed storage (ADR-S05).

Blobs live at ``<data_dir>/projects/<project_id>/cas/sha256/<ab>/<hash>``. Writes go to a temporary
file in the same directory tree and are renamed into place, which is atomic on one filesystem, so a
reader never sees a partial blob. Nothing is shared across projects.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, Field

from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.hashing import sha256_bytes, sha256_file
from archrender.core.paths import check_id, check_sha256, safe_join


class CasRef(BaseModel):
    """Reference to an immutable blob in a project's CAS."""

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    media_type: str = "application/octet-stream"
    name: str | None = None


class ProjectStore:
    """Content-addressed blob store scoped to one project."""

    def __init__(self, projects_dir: Path, project_id: str) -> None:
        self.project_id = check_id(project_id, "project id")
        self.root = safe_join(projects_dir, self.project_id)
        self.cas_dir = self.root / "cas" / "sha256"
        self.tmp_dir = self.root / "cas" / "tmp"
        self.cas_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def _blob_path(self, sha: str) -> Path:
        check_sha256(sha)
        return self.cas_dir / sha[:2] / sha

    def path(self, ref: CasRef) -> Path:
        p = self._blob_path(ref.sha256)
        if not p.exists():
            raise ArchRenderError(
                ErrorCode.NOT_FOUND,
                f"Blob {ref.sha256[:12]}… is missing from project {self.project_id}.",
                "Re-run the stage that produced it; the cache entry will be rebuilt.",
            )
        return p

    def exists(self, sha: str) -> bool:
        return self._blob_path(sha).exists()

    def _publish(self, tmp: Path, sha: str) -> Path:
        dest = self._blob_path(sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            tmp.unlink(missing_ok=True)
        else:
            os.chmod(tmp, 0o444)
            os.replace(tmp, dest)
        return dest

    def put_bytes(
        self, data: bytes, media_type: str = "application/octet-stream", name: str | None = None
    ) -> CasRef:
        sha = sha256_bytes(data)
        if not self.exists(sha):
            fd, tmp_name = tempfile.mkstemp(dir=self.tmp_dir)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            self._publish(Path(tmp_name), sha)
        return CasRef(sha256=sha, size=len(data), media_type=media_type, name=name)

    def put_file(
        self,
        src: Path,
        media_type: str = "application/octet-stream",
        name: str | None = None,
        *,
        move: bool = False,
    ) -> CasRef:
        sha = sha256_file(src)
        size = src.stat().st_size
        if not self.exists(sha):
            fd, tmp_name = tempfile.mkstemp(dir=self.tmp_dir)
            os.close(fd)
            tmp = Path(tmp_name)
            if move:
                shutil.move(str(src), tmp)
            else:
                shutil.copyfile(src, tmp)
            self._publish(tmp, sha)
        elif move:
            src.unlink(missing_ok=True)
        return CasRef(sha256=sha, size=size, media_type=media_type, name=name)

    def put_stream(
        self, stream: BinaryIO, media_type: str = "application/octet-stream", name: str | None = None
    ) -> CasRef:
        fd, tmp_name = tempfile.mkstemp(dir=self.tmp_dir)
        with os.fdopen(fd, "wb") as fh:
            shutil.copyfileobj(stream, fh, 1 << 20)
        tmp = Path(tmp_name)
        try:
            return self.put_file(tmp, media_type, name, move=True)
        finally:
            tmp.unlink(missing_ok=True)

    def read_bytes(self, ref: CasRef) -> bytes:
        return self.path(ref).read_bytes()

    def scratch_dir(self, prefix: str) -> Path:
        """A fresh private working directory inside the project (same filesystem as the CAS)."""
        return Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=self.tmp_dir))

    def usage_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.cas_dir.rglob("*") if p.is_file())

    def purge(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
