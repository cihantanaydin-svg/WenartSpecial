"""Runtime settings (environment variables with the ``ARCHRENDER_`` prefix)."""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]

Profile = Literal["cpu_test", "gpu48", "gpu80", "gpu96plus"]
BlenderMode = Literal["binary", "module"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARCHRENDER_", extra="ignore")

    # storage
    data_dir: Path = Field(default=REPO_ROOT / "var" / "workspace")
    db_path: Path = Field(default=REPO_ROOT / "var" / "db" / "archrender.sqlite")
    configs_dir: Path = Field(default=REPO_ROOT / "configs")
    ui_dist: Path = Field(default=REPO_ROOT / "ui" / "dist")

    # hardware / models
    profile: Profile = "cpu_test"

    # Blender runtime: "binary" = official tarball (`blender -b --python`),
    # "module" = a Python interpreter that has the official `bpy` wheel of the same version.
    blender_mode: BlenderMode = "binary"
    blender_bin: Path = Path("/opt/blender/blender")
    blender_python: Path = REPO_ROOT / ".venv-blender" / "bin" / "python"
    blender_timeout_s: int = 3600

    # security
    admin_token: SecretStr | None = None  # one-shot bootstrap token (RunPod secret)
    key_pepper: SecretStr | None = None  # generated and persisted under data_dir when unset
    session_ttl_s: int = 12 * 3600
    cookie_secure: bool = True

    # API behaviour
    request_timeout_s: float = 55.0  # RunPod proxy cuts at 100 s
    sse_heartbeat_s: float = 15.0
    upload_chunk_bytes: int = 16 * 1024 * 1024
    max_upload_bytes: int = 4 * 1024 * 1024 * 1024
    max_project_bytes: int = 50 * 1024 * 1024 * 1024

    # intake limits (S0). Untrusted parsers run in a separate process under these limits.
    zip_max_total_bytes: int = 20 * 1024**3
    zip_max_ratio: float = 100.0  # uncompressed / compressed, per entry and overall
    zip_max_entries: int = 10_000
    zip_max_depth: int = 3
    image_max_pixels: int = 300_000_000
    pdf_max_pages: int = 500
    pdf_dpi: float = 300.0
    pdf_max_page_pixels: int = 160_000_000  # A0 at 300 DPI is 139 MP; larger sheets get lower DPI
    tile_px: int = 1536
    tile_overlap: float = 0.2
    sandbox_memory_mb: int = 6144
    sandbox_cpu_s: int = 900
    sandbox_timeout_s: float = 1200.0
    office_max_cells: int = 1_000_000
    dwg2dxf_bin: str = "dwg2dxf"  # LibreDWG (GPL, separate process; ADR-S16)
    heif_bin: str = (
        ""  # empty → heif-dec, then heif-convert (libheif, decoder plugin only; ADR-S17)
    )

    # workers
    cpu_workers: int = 2
    job_lease_s: float = 120.0
    worker_poll_s: float = 0.5

    def app_root(self) -> Path:
        """Checkout root in development, /opt/archrender in the image (parent of configs/)."""
        return self.configs_dir.parent

    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    def secrets_dir(self) -> Path:
        return self.data_dir / "secrets"

    def pepper(self) -> bytes:
        """Server-side pepper for API-key hashing (persisted on first use, 0600)."""
        if self.key_pepper is not None:
            return self.key_pepper.get_secret_value().encode()
        path = self.secrets_dir() / "key_pepper"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(secrets.token_hex(32))
        return path.read_text().strip().encode()

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.projects_dir(), self.secrets_dir(), self.db_path.parent):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
