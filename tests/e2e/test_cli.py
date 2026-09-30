from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pytest

from archrender.cli.main import main
from tests.e2e.conftest import Live
from tests.helpers import MINIMAL_DXF

pytestmark = [pytest.mark.e2e, pytest.mark.blender]


def test_cli_full_flow(live: Live, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--url", live.url, "--api-key", live.admin_key]
    assert (
        main([*base, "project", "create", "Salon projesi", "--lat", "41.0", "--lon", "29.0"]) == 0
    )
    pid = re.search(r"id: (prj_\w+)", capsys.readouterr().out).group(1)  # type: ignore[union-attr]

    plan = tmp_path / "Zemin Kat Planı.dxf"
    plan.write_bytes(MINIMAL_DXF)
    assert main([*base, "upload", pid, str(plan)]) == 0
    assert "(dxf)" in capsys.readouterr().out

    assert (
        main(
            [
                *base,
                "run",
                pid,
                "--views",
                "1",
                "--width",
                "96",
                "--height",
                "54",
                "--samples",
                "8",
                "--wait",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    run_id = re.search(r"run (run_\w+)", out).group(1)  # type: ignore[union-attr]
    assert "gate D_final pending" in out

    assert main([*base, "gate", "approve", run_id, "D_final", "--notes", "cli test"]) == 0
    assert main([*base, "status", run_id, "--follow"]) == 0
    out = capsys.readouterr().out
    assert "run status: succeeded" in out and "mock models were used" in out

    dest = tmp_path / "bundle.zip"
    assert main([*base, "download", run_id, "-o", str(dest)]) == 0
    with zipfile.ZipFile(dest) as zf:
        assert "qa/qa_report.html" in zf.namelist()


def test_cli_reports_coded_errors(live: Live, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--url", live.url, "--api-key", "ark_000000000000_bad", "project", "list"]) == 1
    err = capsys.readouterr().err
    assert "UNAUTHORIZED" in err and "Bearer" in err
