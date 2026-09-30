from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from archrender.core.config import REPO_ROOT, Settings
from archrender.db.database import Database

# CI sets this so missing runtimes (Blender, built UI) fail the run instead of skipping tests.
STRICT = os.environ.get("ARCHRENDER_TEST_STRICT") == "1"


def blender_available(settings: Settings) -> bool:
    if settings.blender_mode == "module":
        return settings.blender_python.exists()
    return settings.blender_bin.exists() or shutil.which("blender") is not None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    mode = os.environ.get("ARCHRENDER_BLENDER_MODE")
    if mode is None:
        mode = "module" if (REPO_ROOT / ".venv-blender" / "bin" / "python").exists() else "binary"
    s = Settings(
        data_dir=tmp_path / "workspace",
        db_path=tmp_path / "db" / "archrender.sqlite",
        profile="cpu_test",
        blender_mode=mode,  # type: ignore[arg-type]
        cookie_secure=False,
        admin_token="bootstrap-token-for-tests",  # type: ignore[arg-type]
        job_lease_s=30.0,
        worker_poll_s=0.05,
    )
    s.ensure_dirs()
    return s


@pytest.fixture
def db(settings: Settings) -> Iterator[Database]:
    d = Database(settings.db_path)
    d.migrate()
    yield d
    d.close()


@pytest.fixture
def project_id(db: Database) -> str:
    from archrender.core.ids import now_iso

    db.execute(
        "INSERT INTO users(id, name, role, created_at) VALUES ('usr_test', 'tester', 'admin', ?)",
        (now_iso(),),
    )
    db.execute(
        "INSERT INTO projects(id, name, created_by, created_at) VALUES ('prj_test', 'Test', 'usr_test', ?)",
        (now_iso(),),
    )
    return "prj_test"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    s = Settings(
        blender_mode="module"
        if (REPO_ROOT / ".venv-blender" / "bin" / "python").exists()
        else "binary"
    )
    if not blender_available(s):
        if STRICT:
            raise pytest.UsageError(
                "ARCHRENDER_TEST_STRICT=1 but no Blender runtime was found (run `make setup-blender`)."
            )
        skip = pytest.mark.skip(reason="no Blender runtime (run `make setup-blender`)")
        for item in items:
            if "blender" in item.keywords:
                item.add_marker(skip)
