"""Phase-2 acceptance on the golden projects (docs/PLAN.md): every document of G1 and G2 goes
through intake and S1, each page gets its expected class, and every schedule row is linked to a tag
found on a plan (G1: vector PDF + DXF; G2: a noisy 300-DPI scan, tags read inside bubbles).
Requires the trained classifier in configs/ and Tesseract."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from archrender.core.config import Settings
from archrender.core.schemas.jobs import JobStatus
from archrender.core.schemas.understanding import Schedule
from archrender.pipeline.services import Services
from archrender.pipeline.worker import Worker
from archrender.synth.golden import GoldenProject, g1_daire, g2_loft
from tests.conftest import require_tool
from tests.helpers import upload_bytes


@pytest.mark.parametrize("make", [g1_daire, g2_loft], ids=["G1-daire", "G2-loft"])
def test_golden_project_pages_classified_and_schedules_linked(
    settings: Settings, project_id: str, make: Callable[[], GoldenProject]
) -> None:
    require_tool("tesseract")
    g = make()
    svc = Services.create(settings)
    for d in g.docs:
        upload_bytes(svc, project_id, d.filename, d.data)
    Worker(svc, ["cpu", "gpu"], name="w").run_until_idle()

    jobs = svc.db.query(
        "SELECT kind, status, error_json FROM jobs WHERE project_id = ?", (project_id,)
    )
    assert jobs and all(j["status"] == JobStatus.SUCCEEDED.value for j in jobs), [
        dict(j) for j in jobs
    ]

    rows = svc.db.query(
        "SELECT d.filename, a.label, a.confidence FROM page_analysis a"
        " JOIN documents d ON d.id = a.document_id WHERE a.project_id = ?",
        (project_id,),
    )
    labels: dict[str, set[str]] = {}
    for r in rows:
        labels.setdefault(r["filename"], set()).add(r["label"])
    for d in g.docs:
        assert d.filename in labels, f"{d.filename} produced no analysed page"
        if d.expected_class is not None:
            assert labels[d.filename] == {d.expected_class}, (d.filename, labels[d.filename])

    schedules = [
        Schedule.model_validate_json(r["schedule_json"])
        for r in svc.db.query(
            "SELECT schedule_json FROM schedules WHERE project_id = ?", (project_id,)
        )
    ]
    row_tags = {row.tag for s in schedules for row in s.rows if row.tag}
    linked = {row.tag for s in schedules for row in s.rows if row.tag and row.links}
    assert row_tags == g.schedule_tags  # every schedule was found and read completely
    assert linked == row_tags, sorted(row_tags - linked)  # … and every row is on a plan
    open_links = svc.db.query(
        "SELECT payload_json FROM review_items WHERE project_id = ? AND kind = 'schedule_link'"
        " AND status = 'open'",
        (project_id,),
    )
    assert not open_links, [json.loads(r["payload_json"]) for r in open_links]
