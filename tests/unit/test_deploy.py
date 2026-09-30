"""deploy.py: payloads validate against RunPod's v2 OpenAPI schemas; the up/down flows run against
a fake API (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("deploy", ROOT / "deploy" / "runpod" / "deploy.py")
deploy = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["deploy"] = deploy
spec.loader.exec_module(deploy)  # type: ignore[union-attr]

SUBSET = json.loads((ROOT / "deploy" / "runpod" / "openapi_v2_subset.json").read_text())


def validator(name: str) -> Draft202012Validator:
    schema = {"$ref": f"#/components/schemas/{name}", "components": SUBSET["components"]}
    return Draft202012Validator(schema)


def assert_valid(name: str, payload: dict[str, Any]) -> None:
    errors = sorted(validator(name).iter_errors(payload), key=lambda e: e.path)
    assert not errors, [f"{list(e.path)}: {e.message}" for e in errors]


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    for k in list(__import__("os").environ):
        if k.startswith(("RUNPOD_", "ARCHRENDER_", "GHCR_", "HF_", "DEPLOY_")):
            monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "RUNPOD_API_KEY=rpa_test   # inline comment\n"
        "ARCHRENDER_IMAGE='ghcr.io/acme/archrender@sha256:" + "a" * 64 + "'\n"
        "GHCR_USERNAME=acme-bot\nGHCR_TOKEN=ghp_secret\nHF_TOKEN=hf_secret\n"
        "DEPLOY_DATACENTERS=EU-RO-1,EU-CZ-1\nDEPLOY_ENABLE_SSH=true\n"
    )
    return deploy.Config.load(env, dry_run=False)


def test_env_file_parsing(cfg: Any) -> None:
    assert cfg.api_key == "rpa_test"
    assert cfg.image.endswith("a" * 64)
    assert cfg.admin_token_generated and len(cfg.admin_token) >= 24
    assert cfg.gpu_types[0] == "NVIDIA H100 80GB HBM3"


def test_payloads_match_openapi_schemas(cfg: Any) -> None:
    assert_valid(
        "CreateNetworkVolumeRequest", deploy.volume_payload(cfg, "EU-RO-1", "HIGH_PERFORMANCE")
    )
    assert_valid("CreateRegistryRequest", deploy.registry_payload(cfg))
    assert_valid("CreateTemplateRequest", deploy.template_payload(cfg, "reg_1"))
    upd = {k: v for k, v in deploy.template_payload(cfg, "reg_1").items() if k != "name"}
    assert_valid("UpdateTemplateRequest", upd)
    assert_valid(
        "CreatePodRequest",
        deploy.pod_payload(cfg, "tpl_1", "NVIDIA H100 80GB HBM3", "EU-RO-1", "vol_1"),
    )
    assert_valid(
        "CreateSecretRequest",
        {"name": deploy.SECRET_ADMIN, "value": "x", "description": "ArchRender deploy.py"},
    )
    assert_valid("UpdateSecretRequest", {"value": "y"})
    assert_valid("PodActionRequest", {"action": "stop"})
    # the schema really is strict: an unknown field is rejected
    bad = deploy.pod_payload(cfg, "tpl_1", "g", "d", "v") | {"volumeInGb": 20}
    assert list(validator("CreatePodRequest").iter_errors(bad))


def test_secrets_never_in_template(cfg: Any) -> None:
    text = json.dumps(deploy.template_payload(cfg, "reg"))
    assert "hf_secret" not in text and cfg.admin_token not in text and "ghp_secret" not in text
    assert "{{ RUNPOD_SECRET_archrender_hf_token }}" in text


def test_volume_size_budget(cfg: Any) -> None:
    p = deploy.PROFILES["gpu80"]
    expected = (p["models_gb"] + p["assets_gb"] + p["caches_gb"]) * p["headroom"] + 200
    assert cfg.required_volume_gb() >= expected and cfg.required_volume_gb() % 10 == 0


def test_placement_candidates_respect_preference_and_volume_support(cfg: Any) -> None:
    dcs = [
        {"id": "EU-RO-1", "networkVolumeTypes": ["STANDARD"]},
        {"id": "EU-CZ-1", "networkVolumeTypes": ["STANDARD", "HIGH_PERFORMANCE"]},
        {"id": "EU-FR-1", "networkVolumeTypes": []},
    ]
    gpus = [
        {
            "id": "NVIDIA H100 80GB HBM3",
            "dataCenters": [
                {"id": "EU-RO-1", "availability": "NONE"},
                {"id": "EU-CZ-1", "availability": "LOW"},
            ],
        },
        {
            "id": "NVIDIA A100-SXM4-80GB",
            "dataCenters": [
                {"id": "EU-RO-1", "availability": "HIGH"},
                {"id": "EU-FR-1", "availability": "HIGH"},
            ],
        },
    ]
    assert deploy.placement_candidates(cfg, dcs, gpus, None) == [
        ("NVIDIA A100-SXM4-80GB", "EU-RO-1"),
        ("NVIDIA H100 80GB HBM3", "EU-CZ-1"),
    ]
    assert deploy.placement_candidates(cfg, dcs, gpus, "EU-CZ-1") == [
        ("NVIDIA H100 80GB HBM3", "EU-CZ-1")
    ]


class FakeResponse:
    def __init__(
        self, status: int, body: Any = None, headers: dict[str, str] | None = None
    ) -> None:
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.content = b"" if body is None else json.dumps(body).encode()
        self.text = self.content.decode()

    def json(self) -> Any:
        return self._body


class FakeRunPod:
    """Minimal in-memory v2 API: records calls, validates POST/PATCH bodies against the schemas."""

    SCHEMAS: ClassVar[dict[tuple[str, str], str]] = {
        ("POST", "/network-volumes"): "CreateNetworkVolumeRequest",
        ("POST", "/account/secrets"): "CreateSecretRequest",
        ("POST", "/registries"): "CreateRegistryRequest",
        ("POST", "/templates"): "CreateTemplateRequest",
        ("POST", "/pods"): "CreatePodRequest",
    }

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, str, Any]] = []
        self.volumes: list[dict[str, Any]] = []
        self.secrets = [
            {"id": "sec_old", "name": deploy.SECRET_ADMIN, "createdAt": "2026-01-01T00:00:00Z"}
        ]
        self.pods: list[dict[str, Any]] = []
        self.pod_attempts = 0

    def request(
        self, method: str, url: str, json: Any = None, params: Any = None, timeout: float = 0
    ) -> FakeResponse:
        path = url.replace(deploy.API, "")
        self.calls.append((method, path, json))
        schema = self.SCHEMAS.get((method, path))
        if schema:
            assert_valid(schema, json)
        page = {"pagination": {"nextCursor": None, "hasNextPage": False}}
        if (method, path) == ("GET", "/catalog/datacenters"):
            return FakeResponse(
                200,
                {
                    "dataCenters": [
                        {"id": "EU-RO-1", "networkVolumeTypes": ["STANDARD"], "gpuAvailability": []}
                    ]
                },
            )
        if (method, path) == ("GET", "/catalog/gpus"):
            assert params["minCudaVersion"] == "13.0" and params["cloud"] == "SECURE"
            return FakeResponse(
                200,
                {
                    "gpus": [
                        {
                            "id": "NVIDIA H100 80GB HBM3",
                            "price": {"secure": 2.99, "community": 0},
                            "dataCenters": [{"id": "EU-RO-1", "availability": "LOW"}],
                        },
                        {
                            "id": "NVIDIA A100-SXM4-80GB",
                            "price": {"secure": 1.89, "community": 0},
                            "dataCenters": [{"id": "EU-RO-1", "availability": "HIGH"}],
                        },
                    ]
                },
            )
        if (method, path) == ("GET", "/network-volumes"):
            return FakeResponse(200, {"networkVolumes": self.volumes})
        if (method, path) == ("POST", "/network-volumes"):
            vol = {"id": "vol_1", **json}
            self.volumes.append(vol)
            return FakeResponse(201, vol)
        if (method, path) == ("POST", "/account/secrets"):
            if any(s["name"] == json["name"] for s in self.secrets):
                return FakeResponse(409, {"title": "Conflict", "status": 409, "detail": "exists"})
            self.secrets.append({"id": f"sec_{len(self.secrets)}", "name": json["name"]})
            return FakeResponse(201, self.secrets[-1])
        if (method, path) == ("GET", "/account/secrets"):
            return FakeResponse(200, {"secrets": self.secrets})
        if method == "PATCH" and path.startswith("/account/secrets/"):
            assert_valid("UpdateSecretRequest", json)
            return FakeResponse(200, {"id": path.rsplit("/", 1)[-1]})
        if (method, path) == ("GET", "/registries"):
            return FakeResponse(200, {"registries": []})
        if (method, path) == ("POST", "/registries"):
            return FakeResponse(201, {"id": "reg_1", "name": json["name"]})
        if (method, path) == ("GET", "/templates"):
            return FakeResponse(200, {"templates": [], **page})
        if (method, path) == ("POST", "/templates"):
            return FakeResponse(201, {"id": "tpl_1", **json})
        if (method, path) == ("GET", "/pods"):
            return FakeResponse(200, {"pods": self.pods, **page})
        if (method, path) == ("POST", "/pods"):
            self.pod_attempts += 1
            if json["gpu"]["id"] == "NVIDIA H100 80GB HBM3":
                return FakeResponse(
                    400, {"title": "Bad Request", "status": 400, "detail": "could not be placed"}
                )
            pod = {
                "id": "pod_abc",
                "name": json["name"],
                "status": "PROVISIONING",
                "gpu": {"id": json["gpu"]["id"]},
                "dataCenterId": "EU-RO-1",
            }
            self.pods.append(pod)
            return FakeResponse(201, pod)
        if method == "GET" and path.startswith("/pods/"):
            self.pods[0]["status"] = "RUNNING"
            return FakeResponse(200, self.pods[0])
        if method == "DELETE" and path.startswith("/pods/"):
            self.pods.clear()
            return FakeResponse(204)
        if method == "POST" and path.endswith("/action"):
            assert_valid("PodActionRequest", json)
            return FakeResponse(200, {})
        if method == "DELETE" and path.startswith("/network-volumes/"):
            self.volumes.clear()
            return FakeResponse(204)
        raise AssertionError(f"unexpected call {method} {path}")


def test_up_flow_places_pod_after_capacity_miss(
    cfg: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeRunPod()
    api = deploy.RunPodAPI("rpa_test", session=fake, sleep=lambda s: None)  # type: ignore[arg-type]
    assert deploy.cmd_up(cfg, api, wait=False) == 0
    posted = [c for c in fake.calls if c[0] in ("POST", "PATCH")]
    assert ("POST", "/network-volumes") in [(m, p) for m, p, _ in posted]
    assert any(
        m == "PATCH" and p == "/account/secrets/sec_old" for m, p, _ in posted
    )  # 409 → update
    assert fake.pod_attempts == 2  # H100 had no capacity → A100
    out = capsys.readouterr().out
    assert "https://pod_abc-8000.proxy.runpod.net" in out and "archrender bootstrap" in out
    assert fake.headers["Authorization"] == "Bearer rpa_test"


def test_up_refuses_when_pod_exists_and_down_terminates(cfg: Any) -> None:
    fake = FakeRunPod()
    fake.pods.append({"id": "pod_x", "name": cfg.pod_name, "status": "RUNNING"})
    api = deploy.RunPodAPI("k", session=fake, sleep=lambda s: None)  # type: ignore[arg-type]
    assert deploy.cmd_up(cfg, api, wait=False) == 1
    assert deploy.cmd_down(cfg, api, stop=False, purge_volume=False, yes=True) == 0
    assert ("DELETE", "/pods/pod_x", None) in fake.calls


def test_down_stop_and_purge(cfg: Any) -> None:
    fake = FakeRunPod()
    fake.pods.append({"id": "pod_x", "name": cfg.pod_name, "status": "RUNNING"})
    fake.volumes.append(
        {"id": "vol_9", "name": cfg.volume_name, "size": 500, "dataCenter": "EU-RO-1"}
    )
    api = deploy.RunPodAPI("k", session=fake, sleep=lambda s: None)  # type: ignore[arg-type]
    assert deploy.cmd_down(cfg, api, stop=True, purge_volume=True, yes=True) == 0
    assert ("POST", "/pods/pod_x/action", {"action": "stop"}) in fake.calls
    assert ("DELETE", "/network-volumes/vol_9", None) in fake.calls


def test_dry_run_prints_masked_payloads(cfg: Any, capsys: pytest.CaptureFixture[str]) -> None:
    assert deploy.cmd_plan(cfg) == 0
    out = capsys.readouterr().out
    assert "ghp_secret" not in out and "hf_secret" not in out and cfg.admin_token not in out
    assert "runpodctl pod create" in out and "Manual console checklist" in out


def test_down_dry_run_describes_teardown_without_api_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    env = tmp_path / ".env"
    env.write_text("ARCHRENDER_IMAGE=registry.example.com/archrender:1\n")
    monkeypatch.setattr(deploy, "RunPodAPI", None)  # any API use would raise
    assert deploy.main(["down", "--dry-run", "--purge-volume", "--env-file", str(env)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["pod"]["action"] == "terminate" and plan["pod"]["call"] == "DELETE /pods/<pod-id>"
    assert plan["network_volume"]["call"] == "DELETE /network-volumes/<volume-id>"
    assert deploy.main(["down", "--dry-run", "--stop", "--env-file", str(env)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["pod"]["action"] == "stop" and plan["network_volume"]["action"].startswith("keep")
