"""Model pinning (``pin``) and download (``download``): license-gated, SHA-256 verified, resumable.

python -m archrender.ops.models pin                         # writes configs/models.lock.yaml
python -m archrender.ops.models download --profile gpu80    # on the pod, at boot
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from archrender.core.config import Settings, get_settings
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.hashing import sha256_file
from archrender.core.ids import now_iso
from archrender.models.license_gate import LicenseGate
from archrender.models.profiles import HardwareProfile
from archrender.models.registry import Registry, RegistryEntry
from archrender.ops.hub import Hub, HubAccessError, HubModelInfo

LICENSE_FILES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENCE")
Scope = Literal["implemented", "all"]


def _norm(lic: str | None) -> str:
    return (lic or "").strip().lower()


def _selected(info: HubModelInfo, patterns: list[str]) -> list[Any]:
    if not patterns:
        return list(info.files)
    return [f for f in info.files if any(fnmatch.fnmatch(f.path, p) for p in patterns)]


def _hub_error(entry: RegistryEntry, e: HubAccessError) -> ArchRenderError:
    url = f"https://huggingface.co/{entry.repo}"
    if e.status in (401, 403):
        return ArchRenderError(
            ErrorCode.MODEL_GATED_ACCESS,
            f"Access to {entry.repo} was refused ({e.status}).",
            f"Log in to Hugging Face with the account of HF_TOKEN, open {url} and accept the access "
            "terms, then restart the pod.",
            context={"repo": entry.repo, "url": url},
        )
    return ArchRenderError(
        ErrorCode.MODEL_NOT_DOWNLOADED,
        f"Hub request for {entry.repo} failed ({e.status}): {e}",
        "Check network access to huggingface.co and the repo id/revision in configs/models.yaml.",
        retryable=True,
    )


def check_hub_license(entry: RegistryEntry, info: HubModelInfo) -> None:
    """The Hub's license tag must match the registry (a silent relicense blocks the download)."""
    if info.license is None:
        return  # no tag on the Hub; the registry evidence URL + pinned LICENSE snapshot decide
    if _norm(info.license) != _norm(entry.license.id) and _norm(info.license) not in {
        "other",
        "unknown",
    }:
        raise ArchRenderError(
            ErrorCode.MODEL_LICENSE_MISMATCH,
            f"{entry.repo}: Hub license '{info.license}' != registry '{entry.license.id}'.",
            "Re-verify the licence (docs/MODEL_SELECTION.md), update the registry evidence, re-pin.",
            context={"hub": info.license, "registry": entry.license.id},
        )


def pin(registry: Registry, gate: LicenseGate, hub: Hub, configs_dir: Path) -> dict[str, Any]:
    lock: dict[str, Any] = {"generated_at": now_iso(), "models": {}}
    snap_dir = configs_dir / "licenses" / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for entry in registry.entries():
        if entry.mock or not entry.repo or not gate.allowed(entry):
            continue
        try:
            info = hub.model_info(entry.repo, None)
        except HubAccessError as e:
            raise _hub_error(entry, e) from e
        check_hub_license(entry, info)
        files = [
            {"path": f.path, "sha256": f.sha256, "size": f.size}
            for f in _selected(info, entry.allow_patterns)
            if f.sha256
        ]
        lock["models"][entry.name] = {"repo": entry.repo, "revision": info.revision, "files": files}
        for name in LICENSE_FILES:
            text = hub.read_text(entry.repo, info.revision, name)
            if text:
                (snap_dir / f"{entry.name}@{info.revision[:12]}.txt").write_text(
                    text, encoding="utf-8"
                )
                break
    (configs_dir / "models.lock.yaml").write_text(
        yaml.safe_dump(lock, sort_keys=True), encoding="utf-8"
    )
    return lock


@dataclass
class DownloadReport:
    installed: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    already: list[str] = field(default_factory=list)


def _dedupe(root: Path, path: Path, sha: str) -> None:
    """Content-address weights across repos: identical files share one inode (hard links)."""
    blob = root / "blobs" / "sha256" / sha[:2] / sha
    blob.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not blob.exists():
            os.link(path, blob)
            return
        if blob.stat().st_ino == path.stat().st_ino:
            return
        tmp = path.with_suffix(path.suffix + ".lnk")
        os.link(blob, tmp)
        os.replace(tmp, path)
    except OSError:
        return  # filesystem without hard links: keep the plain copy


def download(
    settings: Settings,
    registry: Registry,
    gate: LicenseGate,
    profile: HardwareProfile,
    hub: Hub,
    *,
    scope: Scope = "implemented",
    allow_unpinned: bool = False,
) -> DownloadReport:
    root = settings.data_dir / "models"
    installed_dir = root / "installed"
    installed_dir.mkdir(parents=True, exist_ok=True)
    report = DownloadReport()
    for role, name in profile.roles.items():
        entry = registry.get(name)
        if entry.mock:
            report.skipped[name] = "mock"
            continue
        if scope == "implemented" and entry.impl is None:
            report.skipped[name] = f"role {role}: runtime not implemented in this build"
            continue
        gate.check(entry)
        if entry.runtime.startswith("system:"):
            report.skipped[name] = "system package (installed in the image, no weights to download)"
            continue
        if not entry.repo:
            raise ArchRenderError(
                ErrorCode.VALIDATION, f"{name} has no repo.", "Fix configs/models.yaml."
            )
        if entry.revision is None and not allow_unpinned:
            raise ArchRenderError(
                ErrorCode.MODEL_NOT_DOWNLOADED,
                f"{name} is not pinned (no revision in configs/models.lock.yaml).",
                "Run `python -m archrender.ops.models pin` with Hub access and commit the lock file.",
            )
        marker = installed_dir / f"{name}.ok"
        if marker.exists():
            data = json.loads(marker.read_text())
            if data.get("revision") == entry.revision:
                report.already.append(name)
                continue
        try:
            info = hub.model_info(entry.repo, entry.revision)
            check_hub_license(entry, info)
            dest = root / "snapshots" / name / info.revision
            hub.snapshot(entry.repo, info.revision, dest, entry.allow_patterns or None)
        except HubAccessError as e:
            raise _hub_error(entry, e) from e
        verified = []
        for f in entry.files:
            local = dest / f.path
            if not local.exists():
                raise ArchRenderError(
                    ErrorCode.MODEL_CHECKSUM_MISMATCH,
                    f"{name}: pinned file {f.path} missing after download.",
                    "Delete the snapshot directory and retry; check allow_patterns.",
                )
            digest = sha256_file(local)
            if f.sha256 and digest != f.sha256:
                raise ArchRenderError(
                    ErrorCode.MODEL_CHECKSUM_MISMATCH,
                    f"{name}: {f.path} SHA-256 mismatch.",
                    "Delete the snapshot directory and retry; if it persists the upstream file changed: re-pin.",
                    context={"expected": f.sha256, "actual": digest},
                )
            _dedupe(root, local, digest)
            verified.append({"path": f.path, "sha256": digest})
        marker.write_text(
            json.dumps(
                {
                    "name": name,
                    "repo": entry.repo,
                    "revision": info.revision,
                    "license": entry.license.id,
                    "files": verified,
                    "installed_at": now_iso(),
                },
                indent=1,
            )
        )
        report.installed.append(name)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="archrender-models")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "pin", help="resolve revisions + file SHA-256 and write configs/models.lock.yaml"
    )
    d = sub.add_parser("download", help="download the active profile's models")
    d.add_argument("--profile", default=None)
    d.add_argument(
        "--scope",
        choices=["implemented", "all"],
        default=os.environ.get("ARCHRENDER_DOWNLOAD_SCOPE", "implemented"),
    )
    d.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args(argv)
    settings = get_settings()
    cfg = settings.configs_dir
    registry, gate = Registry.load(cfg), LicenseGate.load(cfg)
    try:
        if args.cmd == "pin":
            from archrender.ops.hub import HfHub

            lock = pin(registry, gate, HfHub(os.environ.get("HF_TOKEN")), cfg)
            print(f"pinned {len(lock['models'])} models → {cfg / 'models.lock.yaml'}")
            return 0
        profile = HardwareProfile.load(cfg, args.profile or settings.profile)
        needs_hub = any(
            not registry.get(n).mock and (args.scope == "all" or registry.get(n).impl is not None)
            for n in profile.roles.values()
        )
        if not needs_hub:
            print(
                f"profile {profile.name}: nothing to download (all roles mocked or not yet implemented)"
            )
            return 0
        from archrender.ops.hub import HfHub

        rep = download(
            settings,
            registry,
            gate,
            profile,
            HfHub(os.environ.get("HF_TOKEN")),
            scope=args.scope,
            allow_unpinned=args.allow_unpinned,
        )
        print(
            json.dumps(
                {"installed": rep.installed, "already": rep.already, "skipped": rep.skipped},
                indent=1,
            )
        )
        return 0
    except ArchRenderError as e:
        print(f"error [{e.code}]: {e.message}\n  → {e.fix_hint}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
