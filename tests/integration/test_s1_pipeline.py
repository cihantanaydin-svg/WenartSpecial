"""S1 end to end: uploads → intake → UNDERSTAND job → page classes, schedules linked to plan tags,
review queue, class overrides through the API. Requires the trained classifier in configs/."""

from __future__ import annotations

import io
from collections.abc import Iterator

import numpy as np
import openpyxl
import pytest
from fastapi.testclient import TestClient

from archrender.api.app import create_app
from archrender.core.config import Settings
from archrender.core.schemas.jobs import JobKind, JobStatus
from archrender.pipeline.services import Services
from archrender.pipeline.worker import Worker
from archrender.synth.layout import random_layout
from archrender.synth.sheets import floor_plan_page, schedule_rows, section_page, text_page
from tests.helpers import upload_bytes


@pytest.fixture
def env(
    settings: Settings, project_id: str
) -> Iterator[tuple[TestClient, Services, dict[str, str]]]:
    svc = Services.create(settings)
    app = create_app(settings, svc)
    with TestClient(app, base_url="http://testserver") as client:
        r = client.post(
            "/api/v1/auth/bootstrap", json={"token": "bootstrap-token-for-tests", "name": "admin"}
        )
        yield client, svc, {"Authorization": f"Bearer {r.json()['api_key']}"}


def _xlsx(header: list[str], rows: list[list[str]]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Doğrama Listesi"
    ws.append(["DOĞRAMA LİSTESİ"])
    ws.append(header)
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_pages_classified_schedule_linked_and_overrides(env, project_id: str) -> None:  # type: ignore[no-untyped-def]
    client, svc, h = env
    rng = np.random.default_rng(12)
    layout = random_layout(rng)
    plan = floor_plan_page(rng, layout)
    header, rows = schedule_rows(layout, "door_window", "tr", rng)
    worker = Worker(svc, ["cpu", "gpu"], name="w")
    upload_bytes(svc, project_id, "Zemin Kat Planı.pdf", plan.pdf)
    upload_bytes(svc, project_id, "Doğrama Listesi.xlsx", _xlsx(header, rows))
    upload_bytes(svc, project_id, "Kesit.pdf", section_page(rng).pdf)
    upload_bytes(svc, project_id, "Tasarım Özeti.pdf", text_page(rng).pdf)
    worker.run_until_idle()
    jobs = svc.db.query(
        "SELECT kind, status, error_json FROM jobs WHERE project_id = ?", (project_id,)
    )
    assert all(j["status"] == JobStatus.SUCCEEDED.value for j in jobs), [dict(j) for j in jobs]
    assert any(j["kind"] == "understand" for j in jobs)

    pages = client.get(f"/api/v1/projects/{project_id}/pages", headers=h).json()
    by_file = {p["filename"]: p for p in pages}
    assert by_file["Zemin Kat Planı.pdf"]["label"] == "floor_plan"
    assert by_file["Zemin Kat Planı.pdf"]["scale"] == plan.gt["scale"]
    assert by_file["Kesit.pdf"]["label"] == "section"
    assert by_file["Tasarım Özeti.pdf"]["label"] == "text_document"
    assert by_file["Doğrama Listesi.xlsx"]["label"] == "schedule"

    schedules = client.get(f"/api/v1/projects/{project_id}/schedules", headers=h).json()
    assert len(schedules) == 1 and schedules[0]["kind"] == "door_window"
    linked = {r["tag"] for r in schedules[0]["rows"] if r["links"]}
    assert linked == {o.tag for o in layout.openings}  # every schedule row found on the plan

    # a user override resolves the page's review item and survives re-analysis
    pid = by_file["Kesit.pdf"]["page_id"]
    r = client.put(
        f"/api/v1/projects/{project_id}/pages/{pid}/class", json={"label": "elevation"}, headers=h
    )
    assert r.status_code == 200, r.text
    again = {
        p["page_id"]: p
        for p in client.get(f"/api/v1/projects/{project_id}/pages", headers=h).json()
    }
    assert (
        again[pid]["label"] == "elevation"
        and again[pid]["overridden"]
        and again[pid]["model_label"] == "section"
    )

    # re-running S1 hits the cache for every page
    job = svc.queue.enqueue(project_id, JobKind.UNDERSTAND, "gpu", {}, created_by="usr_test")
    worker.run_until_idle()
    res = svc.queue.get(job).result
    assert res is not None and all(t["cached"] for t in res["timings"])


def test_review_queue_lists_and_decides(env, project_id: str) -> None:  # type: ignore[no-untyped-def]
    client, svc, h = env
    svc.db.execute(
        "INSERT INTO review_items(id, project_id, kind, subject_id, status, payload_json, created_at)"
        " VALUES ('rev_1', ?, 'page_class', 'doc_x_p0', 'open', '{}', '2026-09-30T00:00:00Z')",
        (project_id,),
    )
    items = client.get(f"/api/v1/projects/{project_id}/review", headers=h).json()
    assert [i["id"] for i in items] == ["rev_1"]
    r = client.post(
        f"/api/v1/projects/{project_id}/review/rev_1", json={"action": "dismiss"}, headers=h
    )
    assert r.json()["status"] == "dismissed"
    r = client.post(
        f"/api/v1/projects/{project_id}/review/rev_1", json={"action": "resolve"}, headers=h
    )
    assert r.status_code == 409
    assert client.get(f"/api/v1/projects/{project_id}/review", headers=h).json() == []
