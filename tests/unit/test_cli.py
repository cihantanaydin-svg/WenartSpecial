"""CLI argument handling and client error reporting (no server needed)."""

from __future__ import annotations

import httpx
import pytest

from archrender.cli.main import build_parser, main
from archrender.client.client import ArchRenderClient, ArchRenderClientError


@pytest.mark.parametrize(
    "argv",
    [
        ["--url", "http://h", "--api-key", "ark_k", "status", "run_1"],
        ["status", "run_1", "--url", "http://h", "--api-key", "ark_k"],
        ["--url", "http://other", "status", "run_1", "--url", "http://h", "--api-key", "ark_k"],
    ],
)
def test_global_options_work_before_or_after_the_subcommand(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    assert (args.url, args.api_key, args.id) == ("http://h", "ark_k", "run_1")


def test_env_defaults_survive_subcommand_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHRENDER_URL", "http://env")
    monkeypatch.setenv("ARCHRENDER_API_KEY", "ark_env")
    args = build_parser().parse_args(["status", "run_1"])
    assert (args.url, args.api_key, args.json) == ("http://env", "ark_env", False)


def test_run_material_option_parses_pairs() -> None:
    args = build_parser().parse_args(
        ["run", "prj_1", "--material", "floor=stone_porcelain_grey", "--material", "wall:W1=x"]
    )
    assert args.material == ["floor=stone_porcelain_grey", "wall:W1=x"]
    assert main(["--url", "http://h", "--api-key", "k", "run", "prj_1", "--material", "floor"]) == 2


def test_unreachable_server_gives_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    monkeypatch.setattr("archrender.client.client.time.sleep", lambda s: None)
    http = httpx.Client(base_url="http://pod.invalid", transport=httpx.MockTransport(refuse))
    client = ArchRenderClient("http://pod.invalid", http=http)
    with pytest.raises(ArchRenderClientError) as e:
        client.me()
    assert e.value.code == "UNREACHABLE" and "/healthz" in e.value.fix_hint


def _sse(events: list[tuple[int, str, dict[str, object]]]) -> str:
    import json

    body = "".join(
        f"id: {i}\nevent: {name}\ndata: {json.dumps(data)}\n\n" for i, name, data in events
    )
    return body + "event: end\ndata: {}\n\n"


def test_wait_does_not_stop_at_an_already_decided_gate() -> None:
    """Regression (CI run 1): after `gate approve`, `status --follow` replays the history, which
    contains the old waiting_gate event; following must continue until the job finishes."""
    history = [
        (1, "status", {"status": "running"}),
        (2, "status", {"status": "waiting_gate", "gate": "D_final"}),
        (3, "status", {"status": "queued"}),
        (4, "status", {"status": "running"}),
        (5, "status", {"status": "succeeded"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200, text=_sse(history), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json={"id": "job_1", "status": "succeeded"})

    http = httpx.Client(base_url="http://h", transport=httpx.MockTransport(handler))
    seen: list[int] = []
    job = ArchRenderClient("http://h", http=http).wait("job_1", lambda e: seen.append(e["id"]))
    assert job["status"] == "succeeded" and seen == [1, 2, 3, 4, 5]


def test_wait_stops_at_a_pending_gate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            events = [(1, "status", {"status": "waiting_gate"}), (2, "heartbeat", {})]
            return httpx.Response(
                200, text=_sse(events), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json={"id": "job_1", "status": "waiting_gate"})

    http = httpx.Client(base_url="http://h", transport=httpx.MockTransport(handler))
    seen: list[int] = []
    job = ArchRenderClient("http://h", http=http).wait("job_1", lambda e: seen.append(e["id"]))
    assert job["status"] == "waiting_gate" and seen == [1]
