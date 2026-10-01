from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from archrender.api.app import create_app
from archrender.core.config import Settings
from archrender.pipeline.services import Services
from archrender.pipeline.worker import Worker
from tests.helpers import plan_dxf

PUBLIC = {
    ("GET", "/healthz"),
    ("GET", "/readyz"),
    ("POST", "/api/v1/auth/bootstrap"),
    ("POST", "/api/v1/auth/session"),
}


@pytest.fixture
def app_env(settings: Settings) -> Iterator[tuple[TestClient, Services]]:
    svc = Services.create(settings)
    app = create_app(settings, svc)
    with TestClient(app, base_url="http://testserver") as client:
        yield client, svc


def _admin(client: TestClient) -> dict[str, str]:
    r = client.post(
        "/api/v1/auth/bootstrap", json={"token": "bootstrap-token-for-tests", "name": "admin"}
    )
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['api_key']}"}


def test_every_route_requires_auth(app_env: tuple[TestClient, Services]) -> None:
    client, _ = app_env
    schema = client.app.openapi()  # type: ignore[attr-defined]
    checked = 0
    for path, methods in schema["paths"].items():
        for method in methods:
            m = method.upper()
            if (m, path) in PUBLIC:
                continue
            url = re.sub(r"\{[^}]+\}", "x1", path)
            r = client.request(m, url, json={})
            assert r.status_code == 401, f"{m} {path} → {r.status_code}"
            assert r.json()["error"]["code"] == "UNAUTHORIZED"
            checked += 1
    assert checked >= 20


def test_bootstrap_is_one_shot_and_bad_key_rejected(app_env: tuple[TestClient, Services]) -> None:
    client, _ = app_env
    assert (
        client.post(
            "/api/v1/auth/bootstrap", json={"token": "wrong-token-xx", "name": "x"}
        ).status_code
        == 401
    )
    _admin(client)
    r = client.post(
        "/api/v1/auth/bootstrap", json={"token": "bootstrap-token-for-tests", "name": "again"}
    )
    assert r.status_code == 409
    r = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer ark_000000000000_nope"})
    assert r.status_code == 401


def test_session_cookie_requires_csrf(app_env: tuple[TestClient, Services]) -> None:
    client, _ = app_env
    key = _admin(client)["Authorization"][7:]
    r = client.post("/api/v1/auth/session", json={"api_key": key})
    assert r.status_code == 200
    csrf = r.json()["csrf_token"]
    assert "ark_session" in r.cookies or client.cookies.get("ark_session")
    assert client.get("/api/v1/auth/me").json()["role"] == "admin"
    assert client.post("/api/v1/projects", json={"name": "No CSRF"}).status_code == 403
    r = client.post("/api/v1/projects", json={"name": "With CSRF"}, headers={"X-CSRF-Token": csrf})
    assert r.status_code == 201


def test_roles_and_project_isolation(app_env: tuple[TestClient, Services]) -> None:
    client, _ = app_env
    admin = _admin(client)
    viewer = client.post(
        "/api/v1/users", json={"name": "vi", "role": "viewer"}, headers=admin
    ).json()
    editor = client.post(
        "/api/v1/users", json={"name": "ed", "role": "editor"}, headers=admin
    ).json()
    vh = {"Authorization": f"Bearer {viewer['api_key']}"}
    eh = {"Authorization": f"Bearer {editor['api_key']}"}
    assert client.post("/api/v1/projects", json={"name": "p"}, headers=vh).status_code == 403
    pid = client.post("/api/v1/projects", json={"name": "p"}, headers=eh).json()["id"]
    assert client.get(f"/api/v1/projects/{pid}", headers=vh).status_code == 404  # not a member
    assert (
        client.post(
            f"/api/v1/projects/{pid}/members",
            json={"user_id": viewer["user"]["id"], "role": "viewer"},
            headers=eh,
        ).status_code
        == 201
    )
    assert client.get(f"/api/v1/projects/{pid}", headers=vh).status_code == 200
    assert client.post(f"/api/v1/projects/{pid}/runs", json={}, headers=vh).status_code == 403


def _upload(
    client: TestClient, h: dict[str, str], pid: str, name: str, data: bytes, resume: bool = False
) -> str:
    sha = hashlib.sha256(data).hexdigest()
    up = client.post(
        f"/api/v1/projects/{pid}/uploads",
        json={"filename": name, "size": len(data), "sha256": sha},
        headers=h,
    ).json()
    cs = up["chunk_size"]
    for i in range(up["chunks"]):
        if resume and i == 0:
            continue  # simulate an interrupted upload
        chunk = data[i * cs : (i + 1) * cs]
        r = client.put(
            f"/api/v1/uploads/{up['upload_id']}/chunks/{i}",
            content=chunk,
            headers={**h, "X-Chunk-SHA256": hashlib.sha256(chunk).hexdigest()},
        )
        assert r.status_code == 200, r.text
    if resume:
        st = client.get(f"/api/v1/uploads/{up['upload_id']}", headers=h).json()
        assert st["missing"] == [0]
        r = client.post(f"/api/v1/uploads/{up['upload_id']}/complete", headers=h)
        assert r.status_code == 409 and r.json()["error"]["code"] == "INGEST_INCOMPLETE_UPLOAD"
        chunk = data[:cs]
        client.put(f"/api/v1/uploads/{up['upload_id']}/chunks/0", content=chunk, headers=h)
    r = client.post(f"/api/v1/uploads/{up['upload_id']}/complete", headers=h)
    assert r.status_code == 202, r.text
    return str(r.json()["job_id"])


def test_upload_rejects_revit_and_bad_chunk(app_env: tuple[TestClient, Services]) -> None:
    client, svc = app_env
    h = _admin(client)
    pid = client.post("/api/v1/projects", json={"name": "p"}, headers=h).json()["id"]
    rvt = (
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        + b"\x00" * 64
        + "BasicFileInfo".encode("utf-16-le")
        + b"\x00" * 64
    )
    job = _upload(client, h, pid, "model.rvt", rvt)
    Worker(svc, ["cpu"], name="w").run_until_idle()
    j = client.get(f"/api/v1/jobs/{job}", headers=h).json()
    assert j["status"] == "failed" and j["error"]["code"] == "INGEST_UNSUPPORTED_NATIVE"
    assert "IFC" in j["error"]["fix_hint"]
    data = b"hello world"
    up = client.post(
        f"/api/v1/projects/{pid}/uploads",
        json={"filename": "a.txt", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()},
        headers=h,
    ).json()
    r = client.put(
        f"/api/v1/uploads/{up['upload_id']}/chunks/0",
        content=data,
        headers={**h, "X-Chunk-SHA256": "0" * 64},
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "INGEST_CHECKSUM_MISMATCH"


@pytest.mark.blender
def test_full_run_through_api_with_sse_and_bundle(app_env: tuple[TestClient, Services]) -> None:
    client, svc = app_env
    h = _admin(client)
    pid = client.post(
        "/api/v1/projects",
        json={"name": "Daire 3+1", "latitude": 41.01, "longitude": 28.97},
        headers=h,
    ).json()["id"]
    worker = Worker(svc, ["cpu", "gpu"], name="w")
    intake = _upload(client, h, pid, "Kat Planı.dxf", plan_dxf(), resume=True)
    worker.run_until_idle()
    assert client.get(f"/api/v1/jobs/{intake}", headers=h).json()["status"] == "succeeded"
    docs = client.get(f"/api/v1/projects/{pid}/documents", headers=h).json()
    assert docs[0]["filename"] == "Kat Planı.dxf" and docs[0]["kind"] == "dxf"

    r = client.post(
        f"/api/v1/projects/{pid}/runs",
        json={"views": 1, "width": 96, "height": 54, "samples": 8},
        headers=h,
    )
    assert r.status_code == 202
    run_id, job_id = r.json()["run_id"], r.json()["job_id"]
    worker.run_until_idle()
    run = client.get(f"/api/v1/runs/{run_id}", headers=h).json()
    assert run["status"] == "waiting_gate"
    pending = [g for g in run["gates"] if g["status"] == "pending"]
    assert [g["gate"] for g in pending] == ["D_final"]
    r = client.post(
        f"/api/v1/runs/{run_id}/gates/D_final", json={"approve": True, "notes": "ok"}, headers=h
    )
    assert r.status_code == 200
    worker.run_until_idle()

    with client.stream("GET", f"/api/v1/jobs/{job_id}/events", headers=h) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = b"".join(resp.iter_bytes()).decode()
    assert "event: gate" in body and "event: result" in body and "event: end" in body
    ids = [int(x) for x in re.findall(r"^id: (\d+)$", body, re.M)]
    assert ids == sorted(ids)
    with client.stream(
        "GET", f"/api/v1/jobs/{job_id}/events", headers={**h, "Last-Event-ID": str(ids[-2])}
    ) as resp:
        resumed = b"".join(resp.iter_bytes()).decode()
    assert re.findall(r"^id: (\d+)$", resumed, re.M) == [str(ids[-1])]

    run = client.get(f"/api/v1/runs/{run_id}", headers=h).json()
    assert run["status"] == "succeeded" and run["bundle"]["status"] == "ready"
    view = run["result"]["views"][0]
    img = client.get(f"/api/v1/projects/{pid}/blobs/{view['delivered_jpg']['sha256']}", headers=h)
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    rng = client.get(
        f"/api/v1/projects/{pid}/blobs/{view['delivered']['sha256']}",
        headers={**h, "Range": "bytes=0-7"},
    )
    assert rng.status_code == 206 and rng.content == b"\x89PNG\r\n\x1a\n"
    z = client.get(f"/api/v1/bundles/{run['bundle']['id']}/download", headers=h)
    assert z.status_code == 200
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        assert "renders/view_1.png" in zf.namelist()
    audit = client.get("/api/v1/audit", headers=h).json()
    assert {"run.start", "gate.approve", "bundle.download"} <= {a["action"] for a in audit}


def test_gate_a_plan_versions_through_the_api(app_env: tuple[TestClient, Services]) -> None:
    client, svc = app_env
    h = _admin(client)
    users = {
        role: client.post("/api/v1/users", json={"name": role[:2], "role": role}, headers=h).json()
        for role in ("viewer", "reviewer", "editor")
    }
    hdr = {k: {"Authorization": f"Bearer {u['api_key']}"} for k, u in users.items()}
    pid = client.post("/api/v1/projects", json={"name": "p"}, headers=h).json()["id"]
    for role, u in users.items():
        client.post(
            f"/api/v1/projects/{pid}/members",
            json={"user_id": u["user"]["id"], "role": role},
            headers=h,
        )
    _upload(client, h, pid, "Kat Planı.dxf", plan_dxf())
    Worker(svc, ["cpu", "gpu"], name="w").run_until_idle()

    listing = client.get(f"/api/v1/projects/{pid}/plans", headers=hdr["viewer"]).json()
    assert listing["job"]["status"] == "succeeded" and len(listing["versions"]) == 1
    v1 = listing["versions"][0]
    assert v1["status"] == "draft" and v1["origin"] == "extraction" and v1["blocking"] == 0
    got = client.get(f"/api/v1/projects/{pid}/plans/{v1['id']}", headers=hdr["viewer"]).json()
    assert got["plan"]["version"] == v1["id"] and got["plan"]["source"] == "dxf"

    ops = [{"op": "replace", "path": "/rooms/0/name/value", "value": "Kiler"}]
    url = f"/api/v1/projects/{pid}/plans/{v1['id']}"
    assert client.post(f"{url}/edits", json={"ops": ops}, headers=hdr["viewer"]).status_code == 403
    assert (
        client.post(f"{url}/edits", json={"ops": ops}, headers=hdr["reviewer"]).status_code == 403
    )
    r = client.post(f"{url}/edits", json={"ops": ops, "note": "kiler"}, headers=hdr["editor"])
    assert r.status_code == 201, r.text
    v2 = r.json()
    assert v2["parent_id"] == v1["id"] and v2["plan"]["rooms"][0]["name"]["value"] == "Kiler"
    assert v2["plan"]["rooms"][0]["name"]["provenance"][0]["method"] == "user"
    bad = client.post(
        f"{url}/edits", json={"ops": [{"op": "remove", "path": "/nope"}]}, headers=hdr["editor"]
    )
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "VALIDATION"
    assert client.post(f"{url}/edits", json={"ops": []}, headers=hdr["editor"]).status_code == 422
    r = client.post(f"{url}/resolve", json={"key": "scale/x", "choice": 0}, headers=hdr["editor"])
    assert r.status_code == 404
    assert client.get(f"/api/v1/projects/{pid}/plans/bad.id", headers=h).status_code == 422

    assert v2["suggestions"] == []
    r = client.post(f"{url}/assists/confirm", json={"element_ids": ["W1"]}, headers=hdr["editor"])
    assert r.status_code == 422 and "VLM help" in r.json()["error"]["message"]
    r = client.post(f"{url}/suggestions/pg_x/sg1", json={"action": "reject"}, headers=hdr["editor"])
    assert r.status_code == 404
    v2url = f"/api/v1/projects/{pid}/plans/{v2['id']}"
    assert client.post(f"{v2url}/approve", headers=hdr["viewer"]).status_code == 403
    r = client.post(f"{v2url}/approve", headers=hdr["reviewer"])
    assert r.status_code == 200 and r.json()["status"] == "approved"
    examples = client.get(f"/api/v1/projects/{pid}/training-examples", headers=h).json()
    assert len(examples) == 1 and examples[0]["training_use_allowed"] is False
    assert examples[0]["payload"]["edits"][0]["patch"] == ops
    r = client.post(f"/api/v1/projects/{pid}/plans/extract", headers=hdr["editor"])
    assert r.status_code == 202
    Worker(svc, ["cpu", "gpu"], name="w").run_until_idle()
    listing = client.get(f"/api/v1/projects/{pid}/plans", headers=h).json()
    assert len(listing["versions"]) == 2  # unchanged pages: the extraction is reused
    audit = {a["action"] for a in client.get("/api/v1/audit", headers=h).json()}
    assert {"plan.edit", "plan.approve", "plan.extract"} <= audit


def test_healthz_and_readyz(app_env: tuple[TestClient, Services]) -> None:
    client, _ = app_env
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert set(r.json()) == {"ready"}  # no details without auth


def test_request_timeout_middleware() -> None:
    import asyncio

    from fastapi import FastAPI

    from archrender.api.app import RequestTimeout

    app = FastAPI()

    @app.get("/slow")
    async def slow() -> dict[str, str]:
        await asyncio.sleep(2)
        return {"ok": "no"}

    app.add_middleware(RequestTimeout, timeout=0.2)
    with TestClient(app) as c:
        r = c.get("/slow")
    assert r.status_code == 504 and r.json()["error"]["code"] == "REQUEST_TIMEOUT"
