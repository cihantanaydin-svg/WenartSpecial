#!/usr/bin/env python3
"""One-command RunPod deployment for ArchRender (RunPod REST API v2; Python 3.9+ and requests only).

    python deploy/runpod/deploy.py up      [--env-file deploy/runpod/.env] [--dry-run]
    python deploy/runpod/deploy.py status
    python deploy/runpod/deploy.py down    [--stop] [--purge-volume --yes]
    python deploy/runpod/deploy.py plan    # payloads + runpodctl equivalents + console checklist

REST v1 (rest.runpod.io/v1) retires 2026-11-15, so everything here uses v2
(https://api.runpod.io/v2; field names from RunPod's OpenAPI spec, see docs/research/RUNPOD.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

API = "https://api.runpod.io/v2"
HERE = Path(__file__).resolve().parent
PROFILES: dict[str, Any] = json.loads((HERE / "profiles.json").read_text(encoding="utf-8"))
# RunPod secret *names* (values live only in RunPod secrets)
SECRET_HF = "archrender_hf_token"  # noqa: S105
SECRET_ADMIN = "archrender_admin_token"  # noqa: S105
VOLUME_PRICE_GB_MONTH = 0.07  # STANDARD, < 1 TB (RunPod docs, 2026-09); HIGH_PERFORMANCE costs more
APP_PORT = 8000


class DeployError(Exception):
    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------
def read_env_file(path: Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if path is None or not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        elif val.startswith("#"):
            val = ""
        elif " #" in val:
            val = val.split(" #", 1)[0].strip()  # inline comment
        values[key.strip()] = val
    return values


def _bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def _csv(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass
class Config:
    api_key: str
    image: str
    image_digest: str
    profile: str
    gpu_types: list[str]
    datacenters: list[str]
    compliance: list[str]
    volume_name: str
    volume_gb: int
    volume_type: str
    project_allowance_gb: int
    container_disk_gb: int
    pod_name: str
    template_name: str
    cloud: str
    min_cuda: str
    enable_ssh: bool
    idle_stop_minutes: int
    ready_timeout_min: int
    ghcr_username: str
    ghcr_token: str
    hf_token: str
    admin_token: str
    admin_token_generated: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, env_file: Path | None, *, dry_run: bool) -> Config:
        env = {
            **read_env_file(env_file),
            **{
                k: v
                for k, v in os.environ.items()
                if k.startswith(("RUNPOD_", "ARCHRENDER_", "GHCR_", "HF_", "DEPLOY_"))
            },
        }

        def get(key: str, default: str = "") -> str:
            return env.get(key, default).strip()

        profile = get("ARCHRENDER_PROFILE", "gpu80")
        if profile not in PROFILES or profile.startswith("_"):
            raise DeployError(
                f"Unknown ARCHRENDER_PROFILE {profile!r}.", "Use gpu48, gpu80 or gpu96plus."
            )
        admin = get("ARCHRENDER_ADMIN_TOKEN")
        generated = False
        if not admin:
            admin = secrets.token_urlsafe(24)
            generated = True
        cfg = cls(
            api_key=get("RUNPOD_API_KEY"),
            image=get("ARCHRENDER_IMAGE"),
            image_digest=get("ARCHRENDER_IMAGE_DIGEST"),
            profile=profile,
            gpu_types=_csv(get("DEPLOY_GPU_TYPES")) or list(PROFILES[profile]["gpu_types"]),
            datacenters=_csv(get("DEPLOY_DATACENTERS", "EU-RO-1,EU-CZ-1")),
            compliance=_csv(get("DEPLOY_COMPLIANCE")),
            volume_name=get("DEPLOY_VOLUME_NAME", "archrender-data"),
            volume_gb=int(get("DEPLOY_VOLUME_GB", "0") or 0),
            volume_type=get("DEPLOY_VOLUME_TYPE", "auto").upper(),
            project_allowance_gb=int(get("DEPLOY_PROJECT_ALLOWANCE_GB", "200")),
            container_disk_gb=int(get("DEPLOY_CONTAINER_DISK_GB", "100")),
            pod_name=get("DEPLOY_POD_NAME", "archrender"),
            template_name=get("DEPLOY_TEMPLATE_NAME", f"archrender-{profile}"),
            cloud=get("DEPLOY_CLOUD", "SECURE").upper(),
            min_cuda=get("DEPLOY_MIN_CUDA_VERSION", "13.0"),
            enable_ssh=_bool(get("DEPLOY_ENABLE_SSH", "false")),
            idle_stop_minutes=int(get("ARCHRENDER_IDLE_STOP_MINUTES", "0") or 0),
            ready_timeout_min=int(get("DEPLOY_READY_TIMEOUT_MIN", "60")),
            ghcr_username=get("GHCR_USERNAME"),
            ghcr_token=get("GHCR_TOKEN"),
            hf_token=get("HF_TOKEN"),
            admin_token=admin,
            admin_token_generated=generated,
        )
        cfg.validate(dry_run=dry_run)
        return cfg

    def validate(self, *, dry_run: bool) -> None:
        problems = []
        if not self.api_key and not dry_run:
            problems.append("RUNPOD_API_KEY is required (RunPod console → Settings → API Keys).")
        if not self.image:
            problems.append(
                "ARCHRENDER_IMAGE is required, e.g. ghcr.io/<org>/archrender@sha256:<digest>."
            )
        if self.image.startswith("ghcr.io/") and not (self.ghcr_username and self.ghcr_token):
            problems.append(
                "GHCR_USERNAME and GHCR_TOKEN (classic PAT with read:packages) are required for a private GHCR image."
            )
        if self.cloud not in ("SECURE", "COMMUNITY"):
            problems.append("DEPLOY_CLOUD must be SECURE or COMMUNITY.")
        if self.cloud == "COMMUNITY":
            problems.append(
                "Network volumes are Secure Cloud only, and client data is confidential: use SECURE."
            )
        if self.volume_type not in ("AUTO", "STANDARD", "HIGH_PERFORMANCE"):
            problems.append("DEPLOY_VOLUME_TYPE must be auto, STANDARD or HIGH_PERFORMANCE.")
        if not self.gpu_types:
            problems.append("No GPU types for the profile.")
        if problems:
            raise DeployError(
                "Invalid deployment configuration:\n  - " + "\n  - ".join(problems),
                "Fix deploy/runpod/.env (see .env.example).",
            )

    def required_volume_gb(self) -> int:
        if self.volume_gb:
            return self.volume_gb
        p = PROFILES[self.profile]
        need = (p["models_gb"] + p["assets_gb"] + p["caches_gb"]) * p[
            "headroom"
        ] + self.project_allowance_gb
        return int(min(4096, math.ceil(need / 10.0) * 10))


# ---------------------------------------------------------------------------------------------
# payloads (pure functions; validated against RunPod's OpenAPI schemas in tests)
# ---------------------------------------------------------------------------------------------
def secret_ref(name: str) -> str:
    return f"{{{{ RUNPOD_SECRET_{name} }}}}"


def registry_name(cfg: Config) -> str:
    digest = hashlib.sha256(f"{cfg.ghcr_username}:{cfg.ghcr_token}".encode()).hexdigest()[:8]
    return f"archrender-ghcr-{digest}"


def pod_env(cfg: Config) -> dict[str, str]:
    env = {
        "ARCHRENDER_PROFILE": cfg.profile,
        "ARCHRENDER_ADMIN_TOKEN": secret_ref(SECRET_ADMIN),
        "ARCHRENDER_COOKIE_SECURE": "true",
        "ARCHRENDER_IDLE_STOP_MINUTES": str(cfg.idle_stop_minutes),
    }
    if cfg.hf_token:
        env["HF_TOKEN"] = secret_ref(SECRET_HF)
    if cfg.image_digest:
        env["ARCHRENDER_IMAGE_DIGEST"] = cfg.image_digest
    env.update(cfg.extra_env)
    return env


def volume_payload(cfg: Config, datacenter: str, volume_type: str) -> dict[str, Any]:
    return {
        "name": cfg.volume_name,
        "size": cfg.required_volume_gb(),
        "dataCenter": datacenter,
        "type": volume_type,
    }


def registry_payload(cfg: Config) -> dict[str, Any]:
    return {"name": registry_name(cfg), "username": cfg.ghcr_username, "password": cfg.ghcr_token}


def template_payload(cfg: Config, registry_id: str | None) -> dict[str, Any]:
    ports = [f"{APP_PORT}/http"] + (["22/tcp"] if cfg.enable_ssh else [])
    body: dict[str, Any] = {
        "name": cfg.template_name,
        "image": cfg.image,
        "category": "NVIDIA",
        "disk": cfg.container_disk_gb,
        "ports": ports,
        "env": pod_env(cfg),
        "startJupyter": False,
        "startSsh": cfg.enable_ssh,
        "public": False,
        "serverless": False,
    }
    if registry_id:
        body["registry"] = registry_id
    return body


def pod_payload(
    cfg: Config, template_id: str, gpu_id: str, datacenter: str, volume_id: str
) -> dict[str, Any]:
    return {
        "name": cfg.pod_name,
        "templateId": template_id,
        "cloud": cfg.cloud,
        "gpu": {
            "id": gpu_id,
            "count": 1,
            "minCudaVersion": cfg.min_cuda,
            "minRamPerGpu": int(PROFILES[cfg.profile]["min_system_ram_gb"]),
        },
        "dataCenterIds": [datacenter],
        "mounts": {"network": [{"volumeId": volume_id, "path": "/workspace"}]},
        "startJupyter": False,
        "startSsh": cfg.enable_ssh,
    }


def mask(payload: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(payload))
    for key in ("password", "value"):
        if key in out:
            out[key] = "***"
    return out


# ---------------------------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------------------------
class RunPodAPI:
    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.s = session or requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {api_key}", "Accept": "application/json"})
        self.sleep = sleep

    def call(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict[str, Any] | None = None,
        ok: tuple[int, ...] = (200, 201, 204),
    ) -> requests.Response:
        for attempt in range(6):
            r = self.s.request(method, API + path, json=json_body, params=params, timeout=60)
            if r.status_code == 429:
                self.sleep(float(r.headers.get("Retry-After", "5")))
                continue
            if r.status_code >= 500 and attempt < 3:
                self.sleep(2.0 * (2**attempt))
                continue
            if r.status_code not in ok:
                return r
            return r
        return r

    @staticmethod
    def problem(r: requests.Response) -> str:
        try:
            p = r.json()
            detail = p.get("detail") or p.get("title") or r.text
            errs = p.get("errors") or []
            return f"{r.status_code} {detail}" + (f" ({'; '.join(map(str, errs))})" if errs else "")
        except ValueError:
            return f"{r.status_code} {r.text[:300]}"

    def expect(self, r: requests.Response, what: str) -> Any:
        if r.status_code not in (200, 201, 204):
            raise DeployError(f"{what} failed: {self.problem(r)}", _hint_for(r.status_code))
        return r.json() if r.content else {}

    def list_all(
        self, path: str, key: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor = None
        while True:
            q = dict(params or {})
            if cursor:
                q["cursor"] = cursor
            data = self.expect(self.call("GET", path, params=q), f"GET {path}")
            items.extend(data.get(key, []))
            page = data.get("pagination") or {}
            if not page.get("hasNextPage"):
                return items
            cursor = page.get("nextCursor")


def _hint_for(status: int) -> str:
    return {
        401: "Check RUNPOD_API_KEY.",
        402: "Insufficient RunPod balance (at least one hour of credits is required to deploy).",
        403: "The API key lacks permission for this resource.",
        413: "Request larger than 100 KB: reduce template env size.",
        422: "Payload rejected by the API schema: this is a deploy.py bug, please report it with the output.",
    }.get(status, "")


# ---------------------------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


def find_volume(api: RunPodAPI, cfg: Config) -> dict[str, Any] | None:
    vols = [
        v
        for v in api.list_all("/network-volumes", "networkVolumes")
        if v.get("name") == cfg.volume_name
    ]
    if len(vols) > 1:
        raise DeployError(
            f"Several network volumes are named {cfg.volume_name!r}.",
            "Rename or delete the extras in the console.",
        )
    return vols[0] if vols else None


def datacenter_catalog(api: RunPodAPI, cfg: Config) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"include": "GPU_AVAILABILITY"}
    if cfg.compliance:
        params["compliance"] = ",".join(cfg.compliance)
    return api.list_all("/catalog/datacenters", "dataCenters", params)


def gpu_catalog(api: RunPodAPI, cfg: Config) -> list[dict[str, Any]]:
    params = {
        "include": "AVAILABILITY",
        "product": "POD",
        "cloud": cfg.cloud,
        "minCudaVersion": cfg.min_cuda,
    }
    return api.list_all("/catalog/gpus", "gpus", params)


def placement_candidates(
    cfg: Config, dcs: list[dict[str, Any]], gpus: list[dict[str, Any]], fixed_dc: str | None
) -> list[tuple[str, str]]:
    """Ordered (gpu_type, datacenter) pairs: preferred DCs first, then profile GPU order; only DCs
    with network-volume support and GPUs whose availability there is not NONE."""
    dc_ok = {d["id"]: d for d in dcs if d.get("networkVolumeTypes")}
    order = [fixed_dc] if fixed_dc else [d for d in cfg.datacenters if d in dc_ok]
    avail: dict[tuple[str, str], str] = {}
    for g in gpus:
        for d in g.get("dataCenters") or []:
            avail[(g["id"], d["id"])] = d.get("availability", "NONE")
    for d in dcs:
        for g in d.get("gpuAvailability") or []:
            avail.setdefault((g["id"], d["id"]), g.get("availability", "NONE"))
    out = []
    for dc in order:
        for gpu in cfg.gpu_types:
            if avail.get((gpu, dc), "NONE") != "NONE":
                out.append((gpu, dc))
    return out


def ensure_volume(
    api: RunPodAPI, cfg: Config, dcs: list[dict[str, Any]], datacenter: str
) -> dict[str, Any]:
    offered = next((d.get("networkVolumeTypes") or [] for d in dcs if d["id"] == datacenter), [])
    vtype = cfg.volume_type
    if vtype == "AUTO":
        vtype = "HIGH_PERFORMANCE" if "HIGH_PERFORMANCE" in offered else "STANDARD"
    payload = volume_payload(cfg, datacenter, vtype)
    log(
        f"creating network volume {payload['name']} ({payload['size']} GB, {vtype}) in {datacenter}"
    )
    return dict(
        api.expect(api.call("POST", "/network-volumes", json_body=payload), "create network volume")
    )


def ensure_secret(api: RunPodAPI, name: str, value: str) -> None:
    r = api.call(
        "POST",
        "/account/secrets",
        json_body={"name": name, "value": value, "description": "ArchRender deploy.py"},
    )
    if r.status_code in (200, 201):
        log(f"secret {name} created")
        return
    if r.status_code == 409:
        found = api.list_all("/account/secrets", "secrets", {"name": name})
        sid = next((s["id"] for s in found if s["name"] == name), None)
        if sid is None:
            raise DeployError(
                f"Secret {name} exists but cannot be listed.", "Check the API key permissions."
            )
        api.expect(
            api.call("PATCH", f"/account/secrets/{sid}", json_body={"value": value}),
            f"update secret {name}",
        )
        log(f"secret {name} updated")
        return
    raise DeployError(f"create secret {name} failed: {api.problem(r)}", _hint_for(r.status_code))


def ensure_registry(api: RunPodAPI, cfg: Config) -> str | None:
    if not (cfg.ghcr_username and cfg.ghcr_token):
        return None
    name = registry_name(cfg)
    for reg in api.list_all("/registries", "registries"):
        if reg.get("name") == name:
            return str(reg["id"])
    created = api.expect(
        api.call("POST", "/registries", json_body=registry_payload(cfg)),
        "create registry credential",
    )
    log(f"registry credential {name} created")
    return str(created["id"])


def ensure_template(api: RunPodAPI, cfg: Config, registry_id: str | None) -> str:
    body = template_payload(cfg, registry_id)
    for t in api.list_all("/templates", "templates"):
        if t.get("name") == cfg.template_name:
            api.expect(
                api.call(
                    "PATCH",
                    f"/templates/{t['id']}",
                    json_body={k: v for k, v in body.items() if k != "name"},
                ),
                "update template",
            )
            log(f"template {cfg.template_name} updated")
            return str(t["id"])
    created = api.expect(api.call("POST", "/templates", json_body=body), "create template")
    log(f"template {cfg.template_name} created")
    return str(created["id"])


def find_pod(api: RunPodAPI, cfg: Config) -> dict[str, Any] | None:
    pods = [
        p
        for p in api.list_all("/pods", "pods")
        if p.get("name") == cfg.pod_name and p.get("status") != "TERMINATED"
    ]
    return pods[0] if pods else None


def create_pod(
    api: RunPodAPI, cfg: Config, template_id: str, volume_id: str, candidates: list[tuple[str, str]]
) -> dict[str, Any]:
    """Placement loop (v2 creates one GPU type per call; retry semantics from the OpenAPI spec)."""
    last = ""
    for gpu, dc in candidates:
        body = pod_payload(cfg, template_id, gpu, dc, volume_id)
        log(f"trying {gpu} in {dc}")
        for attempt in range(4):
            r = api.call("POST", "/pods", json_body=body)
            if r.status_code in (200, 201):
                return dict(r.json())
            last = api.problem(r)
            if r.status_code >= 500 and attempt < 3:
                continue
            break
        if r.status_code == 402:
            raise DeployError(f"Pod creation refused: {last}", _hint_for(402))
        if r.status_code == 422:
            raise DeployError(f"Pod payload rejected: {last}", _hint_for(422))
        log(f"  no placement: {last}")
    raise DeployError(
        f"No capacity for {cfg.gpu_types} in {sorted({dc for _, dc in candidates}) or cfg.datacenters} (last: {last}).",
        "Retry later, add GPU types (DEPLOY_GPU_TYPES), or pick another profile. A network volume pins its datacenter.",
    )


def pod_logs(api: RunPodAPI, pod_id: str) -> str:
    r = api.s.get(
        f"{API}/pods/{pod_id}/logs",
        params={"source": "system", "tail": 200},
        timeout=30,
        stream=True,
    )
    lines: list[str] = []
    try:
        for i, line in enumerate(r.iter_lines(decode_unicode=True)):
            if line and line.startswith("data:"):
                lines.append(line[5:].strip())
            if i > 2000:
                break
    finally:
        r.close()
    return "\n".join(lines[-60:])


def wait_running(
    api: RunPodAPI, pod_id: str, timeout_s: float, sleep: Callable[[float], None] = time.sleep
) -> dict[str, Any]:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        pod = api.expect(api.call("GET", f"/pods/{pod_id}"), "get pod")
        status = pod.get("status", "")
        if status != last:
            log(f"pod {pod_id}: {status}")
            last = status
        if status == "RUNNING":
            return dict(pod)
        if status in ("EXITED", "ERROR", "TERMINATED"):
            raise DeployError(
                f"Pod entered {status}.\n{pod_logs(api, pod_id)}",
                "Image pull/auth errors: check GHCR credentials; CUDA_VERSION_MISMATCH: the host driver is too old.",
            )
        sleep(10)
    raise DeployError(
        "Pod did not reach RUNNING in time.", "Check the RunPod console for the pod's system logs."
    )


def proxy_url(pod_id: str) -> str:
    return f"https://{pod_id}-{APP_PORT}.proxy.runpod.net"


def wait_ready(url: str, timeout_s: float, sleep: Callable[[float], None] = time.sleep) -> None:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{url}/readyz", timeout=20)
            code = r.status_code
        except requests.RequestException as e:
            code = type(e).__name__
        if code == 200:
            return
        if code != last:
            log(
                f"/readyz → {code} (first boot downloads models and warms kernels; this can take a while)"
            )
            last = code
        sleep(15)
    raise DeployError(
        f"{url}/readyz did not become ready.",
        "Open the pod logs (RunPod console or /workspace/logs) to see which check fails.",
    )


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------
def dry_run_plan(cfg: Config) -> dict[str, Any]:
    dc = cfg.datacenters[0] if cfg.datacenters else "<datacenter>"
    return {
        "volume": volume_payload(
            cfg, dc, "HIGH_PERFORMANCE" if cfg.volume_type == "AUTO" else cfg.volume_type
        ),
        "secrets": [mask({"name": SECRET_ADMIN, "value": cfg.admin_token})]
        + ([mask({"name": SECRET_HF, "value": cfg.hf_token})] if cfg.hf_token else []),
        "registry": mask(registry_payload(cfg)) if cfg.ghcr_username else None,
        "template": template_payload(cfg, "<registry-id>" if cfg.ghcr_username else None),
        "pod": pod_payload(cfg, "<template-id>", cfg.gpu_types[0], dc, "<volume-id>"),
        "placement_order": [(g, d) for d in cfg.datacenters for g in cfg.gpu_types],
    }


def runpodctl_equivalents(cfg: Config) -> list[str]:
    dc = cfg.datacenters[0] if cfg.datacenters else "<DC>"
    env = json.dumps(pod_env(cfg))
    cmds = [
        "# runpodctl (v2.14 still calls REST v1/GraphQL; best-effort equivalents)",
        f"runpodctl network-volume create --name {cfg.volume_name} --size {cfg.required_volume_gb()} "
        f"--data-center-id {dc}",
    ]
    if cfg.ghcr_username:
        cmds.append(
            f'echo "$GHCR_TOKEN" | runpodctl registry create --name {registry_name(cfg)} '
            f"--username {cfg.ghcr_username} --password-stdin"
        )
    cmds += [
        f"runpodctl template create --name {cfg.template_name} --image {cfg.image} "
        f"--container-disk-in-gb {cfg.container_disk_gb} "
        f"--ports '{APP_PORT}/http' --env '{env}'"
        + (" --registry-auth-id <REGISTRY_ID>" if cfg.ghcr_username else ""),
        f"runpodctl pod create --name {cfg.pod_name} --template-id <TEMPLATE_ID> "
        f"--gpu-id '{cfg.gpu_types[0]}' --gpu-count 1 "
        f"--cloud-type {cfg.cloud} --data-center-ids {dc} --network-volume-id <VOLUME_ID> "
        f"--min-cuda-version {cfg.min_cuda}",
        "# secrets are created in the console (Settings → Secrets): "
        f"{SECRET_ADMIN}" + (f", {SECRET_HF}" if cfg.hf_token else ""),
    ]
    return cmds


CHECKLIST = """Manual console checklist (equivalent to `deploy.py up`):
  1. Settings → Secrets: create archrender_admin_token (and archrender_hf_token with your HF read token).
  2. Hugging Face: with the HF_TOKEN account, accept the gated model terms (e.g. facebook/sam3).
  3. Settings → Container Registry Auth: add ghcr.io credentials (username + classic PAT with read:packages).
  4. Storage → Network Volumes: create {volume} ({size} GB) in {dc} (Secure Cloud).
  5. Templates → New: image {image}, container disk {disk} GB, HTTP port 8000, env as printed above,
     registry auth from step 3, Jupyter off.
  6. Pods → Deploy: Secure Cloud, GPU {gpu}, the template, the network volume at /workspace,
     CUDA version filter ≥ {cuda}.
  7. Wait until https://<POD_ID>-8000.proxy.runpod.net/readyz returns 200, then run
     `archrender bootstrap --url https://<POD_ID>-8000.proxy.runpod.net --token <admin token>`."""


def cmd_plan(cfg: Config) -> int:
    print(json.dumps(dry_run_plan(cfg), indent=2))
    print("\n".join(runpodctl_equivalents(cfg)))
    print(
        CHECKLIST.format(
            volume=cfg.volume_name,
            size=cfg.required_volume_gb(),
            dc=cfg.datacenters[0] if cfg.datacenters else "<DC>",
            image=cfg.image,
            disk=cfg.container_disk_gb,
            gpu=cfg.gpu_types[0],
            cuda=cfg.min_cuda,
        )
    )
    return 0


def cmd_up(cfg: Config, api: RunPodAPI, *, wait: bool = True) -> int:
    existing = find_pod(api, cfg)
    if existing is not None:
        log(
            f"pod {cfg.pod_name} already exists ({existing['id']}, {existing.get('status')}); "
            "use `down` first or `status`."
        )
        return 1
    dcs = datacenter_catalog(api, cfg)
    gpus = gpu_catalog(api, cfg)
    price = {g["id"]: (g.get("price") or {}).get("secure") for g in gpus}
    volume = find_volume(api, cfg)
    fixed_dc = volume["dataCenter"] if volume else None
    if fixed_dc and cfg.datacenters and fixed_dc not in cfg.datacenters:
        log(
            f"note: existing volume {cfg.volume_name} lives in {fixed_dc} (outside DEPLOY_DATACENTERS); "
            "the pod must run there."
        )
    candidates = placement_candidates(cfg, dcs, gpus, fixed_dc)
    if not candidates:
        raise DeployError(
            f"No datacenter in {[fixed_dc] if fixed_dc else cfg.datacenters} currently lists "
            f"{cfg.gpu_types} with network-volume support.",
            "Retry later, widen DEPLOY_DATACENTERS/DEPLOY_GPU_TYPES (mind data residency), or use another profile.",
        )
    if volume is None:
        volume = ensure_volume(api, cfg, dcs, candidates[0][1])
        candidates = [c for c in candidates if c[1] == volume["dataCenter"]]
    elif volume.get("size", 0) < cfg.required_volume_gb():
        log(f"growing volume {cfg.volume_name} {volume['size']} → {cfg.required_volume_gb()} GB")
        api.expect(
            api.call(
                "PATCH",
                f"/network-volumes/{volume['id']}",
                json_body={"size": cfg.required_volume_gb()},
            ),
            "resize volume",
        )
    ensure_secret(api, SECRET_ADMIN, cfg.admin_token)
    if cfg.hf_token:
        ensure_secret(api, SECRET_HF, cfg.hf_token)
    else:
        log("warning: HF_TOKEN not set; gated models (SAM 3) cannot be downloaded")
    registry_id = ensure_registry(api, cfg)
    template_id = ensure_template(api, cfg, registry_id)
    pod = create_pod(api, cfg, template_id, volume["id"], candidates)
    pod_id = pod["id"]
    gpu = (pod.get("gpu") or {}).get("id") or candidates[0][0]
    hourly = price.get(gpu)
    log(
        f"pod {pod_id} created on {gpu} in {pod.get('dataCenterId')}"
        + (f" (~${hourly:.2f}/h GPU list price)" if hourly else "")
    )
    log(
        f"network volume {volume['name']}: {volume['size']} GB "
        f"(~${volume['size'] * VOLUME_PRICE_GB_MONTH:.0f}/month at STANDARD rates)"
    )
    if cfg.admin_token_generated:
        log(
            f"generated admin bootstrap token (store it now; it is also in the RunPod secret "
            f"{SECRET_ADMIN}): {cfg.admin_token}"
        )
    url = proxy_url(pod_id)
    if wait:
        wait_running(api, pod_id, 20 * 60)
        wait_ready(url, cfg.ready_timeout_min * 60)
    print(f"\nArchRender is {'ready' if wait else 'starting'} at {url}")
    print(
        f"First login:\n  archrender bootstrap --url {url} --token <admin token>\n"
        f"  then open {url} and paste the printed API key."
    )
    return 0


def cmd_status(cfg: Config, api: RunPodAPI) -> int:
    pod = find_pod(api, cfg)
    if pod is None:
        print(f"no pod named {cfg.pod_name}")
        return 1
    url = proxy_url(pod["id"])
    try:
        ready = requests.get(f"{url}/readyz", timeout=20).status_code == 200
    except requests.RequestException:
        ready = False
    print(
        json.dumps(
            {
                "id": pod["id"],
                "status": pod.get("status"),
                "dataCenter": pod.get("dataCenterId"),
                "gpu": pod.get("gpu"),
                "costPerHr": pod.get("cost"),
                "url": url,
                "ready": ready,
            },
            indent=2,
        )
    )
    return 0


def cmd_down(cfg: Config, api: RunPodAPI, *, stop: bool, purge_volume: bool, yes: bool) -> int:
    pod = find_pod(api, cfg)
    if pod is not None:
        if stop:
            api.expect(
                api.call("POST", f"/pods/{pod['id']}/action", json_body={"action": "stop"}),
                "stop pod",
            )
            log(
                f"pod {pod['id']} stopped (compute released; /workspace on the network volume is kept)"
            )
        else:
            api.expect(api.call("DELETE", f"/pods/{pod['id']}"), "terminate pod")
            log(f"pod {pod['id']} terminated (the network volume is kept; `up` reuses it)")
    else:
        log(f"no pod named {cfg.pod_name}")
    if purge_volume:
        vol = find_volume(api, cfg)
        if vol is None:
            log("no network volume to delete")
            return 0
        if not yes:
            answer = input(
                f"Delete network volume {vol['name']} ({vol['size']} GB) with ALL projects and models? "
                "Type the name to confirm: "
            )
            if answer.strip() != vol["name"]:
                log("aborted")
                return 1
        api.expect(api.call("DELETE", f"/network-volumes/{vol['id']}"), "delete network volume")
        log(f"network volume {vol['name']} deleted")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ArchRender RunPod deployment (REST API v2)")
    ap.add_argument("command", choices=["up", "down", "status", "plan"])
    ap.add_argument("--env-file", default=str(HERE / ".env"))
    ap.add_argument("--dry-run", action="store_true", help="print every payload; make no API calls")
    ap.add_argument("--no-wait", action="store_true", help="return after pod creation")
    ap.add_argument("--stop", action="store_true", help="down: stop instead of terminate")
    ap.add_argument(
        "--purge-volume", action="store_true", help="down: also delete the network volume"
    )
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    try:
        cfg = Config.load(Path(args.env_file), dry_run=args.dry_run or args.command == "plan")
        if args.dry_run or args.command == "plan":
            return cmd_plan(cfg)
        api = RunPodAPI(cfg.api_key)
        if args.command == "up":
            return cmd_up(cfg, api, wait=not args.no_wait)
        if args.command == "status":
            return cmd_status(cfg, api)
        return cmd_down(cfg, api, stop=args.stop, purge_volume=args.purge_volume, yes=args.yes)
    except DeployError as e:
        print(f"error: {e}", file=sys.stderr)
        if e.hint:
            print(f"  → {e.hint}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
