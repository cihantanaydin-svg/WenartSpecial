"""Ops: model pin/download against a fake Hub (licence gate, SHA-256, gated errors), supervisor
config, licence-audit expression logic, deploy manifest freshness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from archrender.core.config import REPO_ROOT, Settings
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.models.license_gate import LicenseGate
from archrender.models.profiles import HardwareProfile
from archrender.models.registry import Registry
from archrender.ops import deploy_manifest, license_audit, supervisor
from archrender.ops.hub import HubAccessError, HubFile, HubModelInfo
from archrender.ops.models import download, pin

CFG = REPO_ROOT / "configs"


class FakeHub:
    def __init__(
        self,
        files: dict[str, bytes],
        license_id: str | None = "apache-2.0",
        fail: int | None = None,
    ) -> None:
        self.files = files
        self.license = license_id
        self.fail = fail
        self.snapshots: list[str] = []

    def model_info(self, repo: str, revision: str | None) -> HubModelInfo:
        if self.fail:
            raise HubAccessError(repo, self.fail, "denied")
        return HubModelInfo(
            repo=repo,
            revision=revision or "rev" + "0" * 37,
            license=self.license,
            gated=False,
            files=[
                HubFile(p, len(b), hashlib.sha256(b).hexdigest()) for p, b in self.files.items()
            ],
        )

    def snapshot(
        self, repo: str, revision: str, dest: Path, allow_patterns: list[str] | None
    ) -> Path:
        self.snapshots.append(repo)
        for p, b in self.files.items():
            (dest / p).parent.mkdir(parents=True, exist_ok=True)
            (dest / p).write_bytes(b)
        return dest

    def read_text(self, repo: str, revision: str, path: str) -> str | None:
        return "Apache License 2.0" if path == "LICENSE" else None


def _implemented_registry(repo_files: dict[str, str]) -> Registry:
    """Registry where the depth model counts as implemented (Phase-1 builds have none)."""
    reg = Registry.load(CFG)
    entries = []
    for e in reg.entries():
        if e.name == "da3-mono-large":
            from archrender.models.registry import ModelFile

            e = e.model_copy(
                update={
                    "impl": "x:Y",
                    "revision": "rev" + "1" * 37,
                    "files": [ModelFile(path=p, sha256=s) for p, s in repo_files.items()],
                }
            )
        entries.append(e)
    return Registry(entries)


def _profile() -> HardwareProfile:
    return HardwareProfile.load(CFG, "gpu80")


def test_download_verifies_hashes_and_writes_marker(tmp_path: Path) -> None:
    data = {"model.safetensors": b"weights" * 100, "config.json": b"{}"}
    reg = _implemented_registry({k: hashlib.sha256(v).hexdigest() for k, v in data.items()})
    s = Settings(data_dir=tmp_path)
    rep = download(s, reg, LicenseGate.load(CFG), _profile(), FakeHub(data))
    assert rep.installed == ["da3-mono-large"]
    assert "not implemented" in rep.skipped["qwen-image-edit-2511"]
    marker = json.loads((tmp_path / "models" / "installed" / "da3-mono-large.ok").read_text())
    assert marker["revision"].startswith("rev1")
    blob = tmp_path / "models" / "blobs" / "sha256"
    assert len(list(blob.rglob("*"))) >= 2  # content-addressed
    rep2 = download(s, reg, LicenseGate.load(CFG), _profile(), FakeHub(data))
    assert rep2.already == ["da3-mono-large"]


def test_download_rejects_tampered_file(tmp_path: Path) -> None:
    reg = _implemented_registry({"model.safetensors": "0" * 64})
    with pytest.raises(ArchRenderError) as e:
        download(
            Settings(data_dir=tmp_path),
            reg,
            LicenseGate.load(CFG),
            _profile(),
            FakeHub({"model.safetensors": b"x"}),
        )
    assert e.value.code == ErrorCode.MODEL_CHECKSUM_MISMATCH


def test_download_refuses_hub_licence_mismatch(tmp_path: Path) -> None:
    reg = _implemented_registry({})
    with pytest.raises(ArchRenderError) as e:
        download(
            Settings(data_dir=tmp_path),
            reg,
            LicenseGate.load(CFG),
            _profile(),
            FakeHub({}, license_id="cc-by-nc-4.0"),
        )
    assert e.value.code == ErrorCode.MODEL_LICENSE_MISMATCH


def test_gated_access_error_names_the_hf_page(tmp_path: Path) -> None:
    reg = _implemented_registry({})
    with pytest.raises(ArchRenderError) as e:
        download(
            Settings(data_dir=tmp_path),
            reg,
            LicenseGate.load(CFG),
            _profile(),
            FakeHub({}, fail=403),
        )
    assert e.value.code == ErrorCode.MODEL_GATED_ACCESS
    assert "https://huggingface.co/depth-anything/DA3MONO-LARGE" in e.value.fix_hint


def test_unpinned_models_are_refused(tmp_path: Path) -> None:
    reg = _implemented_registry({})
    reg = Registry(
        [
            e.model_copy(update={"revision": None}) if e.name == "da3-mono-large" else e
            for e in reg.entries()
        ]
    )
    with pytest.raises(ArchRenderError) as e:
        download(Settings(data_dir=tmp_path), reg, LicenseGate.load(CFG), _profile(), FakeHub({}))
    assert "not pinned" in e.value.message


def test_pin_writes_lock_and_licence_snapshots(tmp_path: Path) -> None:
    cfg = tmp_path / "configs"
    cfg.mkdir()
    for f in ("models.yaml", "licenses.yaml", "deployment.yaml"):
        (cfg / f).write_text((CFG / f).read_text())
    lock = pin(
        Registry.load(cfg),
        LicenseGate.load(cfg),
        FakeHub({"w.safetensors": b"abc"}, license_id=None),
        cfg,
    )
    assert (
        "qwen-image-edit-2511" in lock["models"] and "flux2-dev" not in lock["models"]
    )  # blocked → not pinned
    data = yaml.safe_load((cfg / "models.lock.yaml").read_text())
    assert data["models"]["sam3"]["files"][0]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert list((cfg / "licenses" / "snapshots").glob("sam3@*.txt"))


def test_supervisor_config_runs_only_implemented_services() -> None:
    text = supervisor.render(
        _profile(),
        Registry.load(CFG),
        Path("/workspace/logs"),
        venv="/opt/venv",
        litestream="/usr/local/bin/litestream",
    )
    assert (
        "[program:api]" in text
        and "[program:worker-gpu]" in text
        and "[program:litestream]" in text
    )
    assert "vllm" not in text  # no vLLM-served role is implemented in this build
    assert "chmod=0700" in text


@pytest.mark.parametrize(
    ("expr", "ok"),
    [
        ("MIT", True),
        ("Apache-2.0 OR GPL-3.0", True),
        ("MIT AND BSD-3-Clause", True),
        ("MIT AND AGPL-3.0", False),
        ("PolyForm-Noncommercial-1.0.0", False),
        ("GPL-3.0-or-later", False),
        ("Apache-2.0 WITH LLVM-exception", True),
    ],
)
def test_licence_expressions(expr: str, ok: bool) -> None:
    cfg = yaml.safe_load((CFG / "licenses.yaml").read_text())["packages"]
    assert license_audit.allowed_expression(expr, set(cfg["allowed"]), set(cfg["blocked"])) is ok


def test_spdx_from_classifiers() -> None:
    assert (
        license_audit.spdx_of({"classifiers": ["License :: OSI Approved :: MIT License"]}) == "MIT"
    )
    assert license_audit.spdx_of({"license": "Apache 2.0"}) == "Apache-2.0"
    assert license_audit.spdx_of({"expression": "BSD-3-Clause"}) == "BSD-3-Clause"


def test_deploy_manifest_is_fresh() -> None:
    assert deploy_manifest.main(["--check"]) == 0
