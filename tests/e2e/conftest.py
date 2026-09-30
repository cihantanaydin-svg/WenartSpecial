from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import pytest
import uvicorn

from archrender.api.app import create_app
from archrender.core.config import Settings
from archrender.pipeline.services import Services
from archrender.pipeline.worker import Worker


@dataclass
class Live:
    url: str
    admin_key: str
    svc: Services


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def live(settings: Settings) -> Iterator[Live]:
    svc = Services.create(settings)
    app = create_app(settings, svc)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    wsvc = Services.create(settings, migrate=False)
    worker = Worker(wsvc, ["cpu", "gpu"], name="e2e-worker")
    wt = threading.Thread(target=worker.run_forever, daemon=True)
    wt.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{url}/healthz", timeout=1).status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.1)
    key = httpx.post(
        f"{url}/api/v1/auth/bootstrap", json={"token": "bootstrap-token-for-tests", "name": "admin"}
    ).json()["api_key"]
    yield Live(url, key, svc)
    worker.stop()
    server.should_exit = True
    t.join(timeout=10)
    wt.join(timeout=10)
