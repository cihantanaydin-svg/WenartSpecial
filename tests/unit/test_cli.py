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
