from __future__ import annotations

import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pytest

from archrender.cli.main import main
from archrender.synth.layout import random_layout
from archrender.synth.sheets import floor_plan_page, schedule_rows
from tests.e2e.conftest import Live
from tests.helpers import plan_dxf

pytestmark = [pytest.mark.e2e, pytest.mark.blender]


def test_cli_full_flow(live: Live, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--url", live.url, "--api-key", live.admin_key]
    assert (
        main([*base, "project", "create", "Salon projesi", "--lat", "41.0", "--lon", "29.0"]) == 0
    )
    pid = re.search(r"id: (prj_\w+)", capsys.readouterr().out).group(1)  # type: ignore[union-attr]

    plan = tmp_path / "Zemin Kat Planı.dxf"
    plan.write_bytes(plan_dxf())
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

    # Gate A from the CLI: list, show, edit (JSON Patch), approve
    assert main([*base, "plan", "list", pid]) == 0
    out = capsys.readouterr().out
    v1 = re.search(r"(plv_\w+)\s+v1\s+draft\s+extraction", out).group(1)  # type: ignore[union-attr]
    assert "dxf:" in out
    assert main([*base, "plan", "show", pid, v1]) == 0
    assert "source dxf" in capsys.readouterr().out
    patch = tmp_path / "patch.json"
    patch.write_text('[{"op": "replace", "path": "/rooms/0/name/value", "value": "Kiler"}]')
    assert main([*base, "plan", "edit", pid, v1, "--patch", str(patch), "--note", "cli"]) == 0
    v2 = re.search(r"(plv_\w+)\s+v2\s+draft\s+edit", capsys.readouterr().out).group(1)  # type: ignore[union-attr]
    assert main([*base, "plan", "approve", pid, v2]) == 0
    assert re.search(r"v2\s+approved", capsys.readouterr().out)


def test_cli_reports_coded_errors(live: Live, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--url", live.url, "--api-key", "ark_000000000000_bad", "project", "list"]) == 1
    err = capsys.readouterr().err
    assert "UNAUTHORIZED" in err and "Bearer" in err


def _xlsx(header: list[str], rows: list[list[str]]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(header)
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_cli_pages_schedules_and_review(
    live: Live, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["--url", live.url, "--api-key", live.admin_key]
    assert main([*base, "project", "create", "S1 projesi"]) == 0
    pid = re.search(r"id: (prj_\w+)", capsys.readouterr().out).group(1)  # type: ignore[union-attr]
    rng = np.random.default_rng(31)
    layout = random_layout(rng)
    plan = tmp_path / "Zemin Kat Planı.pdf"
    plan.write_bytes(floor_plan_page(rng, layout).pdf)
    header, rows = schedule_rows(layout, "door_window", "tr", rng)
    sched = tmp_path / "Doğrama Listesi.xlsx"
    sched.write_bytes(_xlsx(header, rows))
    assert main([*base, "upload", pid, str(plan), str(sched)]) == 0
    capsys.readouterr()

    deadline = time.monotonic() + 300
    while live.svc.db.one(
        "SELECT 1 FROM jobs WHERE project_id = ? AND kind = 'understand'"
        " AND status IN ('queued', 'running')",
        (pid,),
    ):
        assert time.monotonic() < deadline, "S1 did not finish"
        time.sleep(0.5)

    assert main([*base, "pages", pid]) == 0
    out = capsys.readouterr().out
    lines = {ln.rsplit("  ", 1)[-1]: ln for ln in out.strip().splitlines()}
    assert " floor_plan " in lines["Zemin Kat Planı.pdf p1"]
    assert " schedule " in lines["Doğrama Listesi.xlsx p1"]

    assert main([*base, "schedules", pid]) == 0
    n = len(layout.openings)
    assert f"door_window {n} rows, {n} linked to plan tags" in capsys.readouterr().out

    page_id = lines["Zemin Kat Planı.pdf p1"].split()[0]
    assert main([*base, "pages", pid, "--set", page_id, "ceiling_plan", "--note", "RCP"]) == 0
    capsys.readouterr()
    assert main([*base, "--json", "pages", pid]) == 0
    pages = {p["page_id"]: p for p in json.loads(capsys.readouterr().out)}
    assert pages[page_id]["label"] == "ceiling_plan" and pages[page_id]["overridden"]

    assert main([*base, "review", pid, "--status", "all"]) == 0  # listing works with any content
