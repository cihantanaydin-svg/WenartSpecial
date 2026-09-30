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

    # workers
    cpu_workers: int = 2
    job_lease_s: float = 120.0
    worker_poll_s: float = 0.5

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
