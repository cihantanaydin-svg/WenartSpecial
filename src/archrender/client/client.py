"""Python client for the ArchRender API (used by the CLI). Resumable uploads, SSE with reconnect."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx

TERMINAL = {"succeeded", "failed", "cancelled"}


class ArchRenderClientError(Exception):
    def __init__(self, status: int, code: str, message: str, fix_hint: str = "") -> None:
        super().__init__(f"[{code}] {message}" + (f"\n  → {fix_hint}" if fix_hint else ""))
        self.status = status
        self.code = code
        self.message = message
        self.fix_hint = fix_hint


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while buf := fh.read(1 << 20):
            h.update(buf)
    return h.hexdigest()


class ArchRenderClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        http: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.http.headers.update(headers)

    # ---- plumbing -----------------------------------------------------------------------------
    def _check(self, r: httpx.Response) -> Any:
        if r.status_code >= 400:
            try:
                err = r.json()["error"]
                raise ArchRenderClientError(
                    r.status_code, err["code"], err["message"], err.get("fix_hint", "")
                )
            except (ValueError, KeyError, TypeError):
                raise ArchRenderClientError(r.status_code, "HTTP", r.text[:500]) from None
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return r.content

    def _get(self, url: str, **kw: Any) -> Any:
        return self._check(self._retry(lambda: self.http.get(url, **kw)))

    def _post(self, url: str, **kw: Any) -> Any:
        return self._check(self._retry(lambda: self.http.post(url, **kw)))

    def _put(self, url: str, **kw: Any) -> Any:
        return self._check(self._retry(lambda: self.http.put(url, **kw)))

    def _unreachable(self, e: httpx.TransportError) -> ArchRenderClientError:
        return ArchRenderClientError(
            0,
            "UNREACHABLE",
            f"Cannot reach {self.http.base_url}: {e}",
            "Check the URL and that the server is up (GET /healthz). A RunPod pod can take several "
            "minutes after start (model download, self-test) before /readyz reports ready.",
        )

    def _retry(self, fn: Callable[[], httpx.Response], attempts: int = 4) -> httpx.Response:
        delay = 1.0
        for i in range(attempts):
            try:
                r = fn()
            except httpx.TransportError as e:
                if i == attempts - 1:
                    raise self._unreachable(e) from e
            else:
                if r.status_code not in (502, 503, 504, 524) or i == attempts - 1:
                    return r
            time.sleep(delay)
            delay *= 2
        raise RuntimeError("unreachable")

    # ---- auth / projects ----------------------------------------------------------------------
    def bootstrap(self, token: str, name: str) -> dict[str, Any]:
        return dict(self._post("/api/v1/auth/bootstrap", json={"token": token, "name": name}))

    def me(self) -> dict[str, Any]:
        return dict(self._get("/api/v1/auth/me"))

    def create_project(
        self, name: str, latitude: float | None = None, longitude: float | None = None
    ) -> dict[str, Any]:
        return dict(
            self._post(
                "/api/v1/projects",
                json={"name": name, "latitude": latitude, "longitude": longitude},
            )
        )

    def list_projects(self) -> list[dict[str, Any]]:
        return list(self._get("/api/v1/projects"))

    def documents(self, project_id: str) -> list[dict[str, Any]]:
        return list(self._get(f"/api/v1/projects/{project_id}/documents"))

    # ---- understanding (S1) -------------------------------------------------------------------
    def pages(self, project_id: str) -> list[dict[str, Any]]:
        return list(self._get(f"/api/v1/projects/{project_id}/pages"))

    def set_page_class(
        self, project_id: str, page_id: str, label: str, note: str | None = None
    ) -> dict[str, Any]:
        return dict(
            self._put(
                f"/api/v1/projects/{project_id}/pages/{page_id}/class",
                json={"label": label, "note": note},
            )
        )

    def understanding_status(self, project_id: str) -> dict[str, Any] | None:
        """The project's latest S1 (page analysis) job, or None before the first upload."""
        job = self._get(f"/api/v1/projects/{project_id}/understand")
        return dict(job) if job else None

    def schedules(self, project_id: str) -> list[dict[str, Any]]:
        return list(self._get(f"/api/v1/projects/{project_id}/schedules"))

    def review_items(self, project_id: str, status: str = "open") -> list[dict[str, Any]]:
        return list(self._get(f"/api/v1/projects/{project_id}/review", params={"status": status}))

    def decide_review(
        self, project_id: str, item_id: str, action: str, note: str | None = None
    ) -> dict[str, Any]:
        return dict(
            self._post(
                f"/api/v1/projects/{project_id}/review/{item_id}",
                json={"action": action, "note": note},
            )
        )

    # ---- uploads ------------------------------------------------------------------------------
    def upload(
        self,
        project_id: str,
        path: Path,
        *,
        upload_id: str | None = None,
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> tuple[str, str]:
        """Upload ``path`` (resumable: pass a previous ``upload_id`` to send only missing chunks).
        Returns ``(upload_id, intake_job_id)``."""
        size = path.stat().st_size
        if upload_id is None:
            up = self._post(
                f"/api/v1/projects/{project_id}/uploads",
                json={"filename": path.name, "size": size, "sha256": _sha256_file(path)},
            )
            upload_id, chunk, missing = up["upload_id"], up["chunk_size"], list(range(up["chunks"]))
        else:
            st = self._get(f"/api/v1/uploads/{upload_id}")
            chunk, missing = st["chunk_size"], st["missing"]
        total = (size + chunk - 1) // chunk
        with path.open("rb") as fh:
            for i in missing:
                fh.seek(i * chunk)
                data = fh.read(chunk)
                digest = hashlib.sha256(data).hexdigest()
                self._check(self._retry(self._chunk_sender(str(upload_id), i, data, digest)))
                if on_chunk:
                    on_chunk(i + 1, total)
        job = self._post(f"/api/v1/uploads/{upload_id}/complete")
        return str(upload_id), str(job["job_id"])

    def _chunk_sender(
        self, upload_id: str, index: int, data: bytes, digest: str
    ) -> Callable[[], httpx.Response]:
        def send() -> httpx.Response:
            return self.http.put(
                f"/api/v1/uploads/{upload_id}/chunks/{index}",
                content=data,
                headers={"X-Chunk-SHA256": digest},
            )

        return send

    # ---- runs / jobs --------------------------------------------------------------------------
    def start_run(self, project_id: str, **config: Any) -> dict[str, Any]:
        return dict(self._post(f"/api/v1/projects/{project_id}/runs", json=config))

    def get_run(self, run_id: str) -> dict[str, Any]:
        return dict(self._get(f"/api/v1/runs/{run_id}"))

    def get_job(self, job_id: str) -> dict[str, Any]:
        return dict(self._get(f"/api/v1/jobs/{job_id}"))

    def decide_gate(
        self,
        run_id: str,
        gate: str,
        approve: bool,
        notes: str | None = None,
        plan_version: str | None = None,
    ) -> dict[str, Any]:
        """``plan_version`` (Gate A only): the plan version to approve; default the latest edit."""
        body: dict[str, Any] = {"approve": approve, "notes": notes}
        if plan_version is not None:
            body["plan_version"] = plan_version
        return dict(self._post(f"/api/v1/runs/{run_id}/gates/{gate}", json=body))

    # ---- plans (S2 / Gate A) ------------------------------------------------------------------
    def plan_versions(self, project_id: str) -> dict[str, Any]:
        """``{"versions": [...newest first], "job": latest PLAN job or None}``."""
        return dict(self._get(f"/api/v1/projects/{project_id}/plans"))

    def plan_version(self, project_id: str, version_id: str) -> dict[str, Any]:
        return dict(self._get(f"/api/v1/projects/{project_id}/plans/{version_id}"))

    def extract_plan(self, project_id: str) -> dict[str, Any]:
        return dict(self._post(f"/api/v1/projects/{project_id}/plans/extract"))

    def edit_plan(
        self, project_id: str, version_id: str, ops: list[dict[str, Any]], note: str = ""
    ) -> dict[str, Any]:
        """Apply an RFC 6902 JSON Patch; returns the new draft version (with its plan)."""
        return dict(
            self._post(
                f"/api/v1/projects/{project_id}/plans/{version_id}/edits",
                json={"ops": ops, "note": note},
            )
        )

    def resolve_plan_conflict(
        self, project_id: str, version_id: str, key: str, choice: int
    ) -> dict[str, Any]:
        return dict(
            self._post(
                f"/api/v1/projects/{project_id}/plans/{version_id}/resolve",
                json={"key": key, "choice": choice},
            )
        )

    def approve_plan(self, project_id: str, version_id: str) -> dict[str, Any]:
        return dict(self._post(f"/api/v1/projects/{project_id}/plans/{version_id}/approve"))

    def training_examples(self, project_id: str) -> list[dict[str, Any]]:
        return list(self._get(f"/api/v1/projects/{project_id}/training-examples"))

    def events(self, job_id: str, last_event_id: int = 0) -> Iterator[dict[str, Any]]:
        """Follow SSE events; reconnects with Last-Event-ID until the server sends 'end'."""
        cursor = last_event_id
        failures = 0
        while True:
            try:
                with self.http.stream(
                    "GET",
                    f"/api/v1/jobs/{job_id}/events",
                    headers={"Last-Event-ID": str(cursor), "Accept": "text/event-stream"},
                    timeout=httpx.Timeout(10.0, read=60.0),
                ) as resp:
                    if resp.status_code >= 400:
                        resp.read()
                        self._check(resp)
                    event: dict[str, Any] = {}
                    for line in resp.iter_lines():
                        if line.startswith("id: "):
                            event["id"] = int(line[4:])
                        elif line.startswith("event: "):
                            event["event"] = line[7:]
                        elif line.startswith("data: "):
                            event["data"] = json.loads(line[6:])
                        elif line == "" and event:
                            if event.get("event") == "end":
                                return
                            cursor = int(event.get("id", cursor))
                            failures = 0
                            yield event
                            event = {}
                # stream closed by the server (max duration): reconnect
            except httpx.TransportError as e:
                failures += 1
                if failures > 5:
                    raise self._unreachable(e) from e
                time.sleep(min(30.0, 2.0**failures))

    def wait(
        self,
        job_id: str,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        *,
        stop_at_gate: bool = True,
    ) -> dict[str, Any]:
        for ev in self.events(job_id):
            if on_event:
                on_event(ev)
            if (
                stop_at_gate
                and ev.get("event") == "status"
                and ev["data"].get("status") == "waiting_gate"
                # the stream replays history: a gate that was already decided must not stop us
                and self.get_job(job_id)["status"] == "waiting_gate"
            ):
                break
        return self.get_job(job_id)

    def download_bundle(self, bundle_id: str, dest: Path) -> Path:
        """Resumable download (HTTP Range) to ``dest``."""
        part = dest.with_suffix(dest.suffix + ".part")
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with self.http.stream(
                "GET",
                f"/api/v1/bundles/{bundle_id}/download",
                headers=headers,
                timeout=httpx.Timeout(10.0, read=120.0),
            ) as r:
                if r.status_code >= 400:
                    r.read()
                    self._check(r)
                mode = "ab" if r.status_code == 206 else "wb"
                with part.open(mode) as fh:
                    for buf in r.iter_bytes():
                        fh.write(buf)
        except httpx.TransportError as e:
            err = self._unreachable(e)
            err.fix_hint = f"Run the same download again to resume from {part} (HTTP Range)."
            raise ArchRenderClientError(err.status, err.code, err.message, err.fix_hint) from e
        part.replace(dest)
        return dest
