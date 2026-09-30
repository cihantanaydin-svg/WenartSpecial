"""Hugging Face Hub access behind a small interface (real client on the pod, fakes in tests)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class HubFile:
    path: str
    size: int | None
    sha256: str | None  # LFS sha256 (None for small git files)


@dataclass(frozen=True)
class HubModelInfo:
    repo: str
    revision: str
    license: str | None
    gated: bool
    files: list[HubFile] = field(default_factory=list)


class HubAccessError(Exception):
    def __init__(self, repo: str, status: int, message: str) -> None:
        super().__init__(message)
        self.repo = repo
        self.status = status


class Hub(Protocol):
    def model_info(self, repo: str, revision: str | None) -> HubModelInfo: ...
    def snapshot(
        self, repo: str, revision: str, dest: Path, allow_patterns: list[str] | None
    ) -> Path: ...
    def read_text(self, repo: str, revision: str, path: str) -> str | None: ...


class HfHub:
    """Real implementation (huggingface_hub; installed with the ``gpu`` extra in the image)."""

    def __init__(self, token: str | None) -> None:
        from huggingface_hub import HfApi

        self.token = token
        self.api = HfApi(token=token)

    def model_info(self, repo: str, revision: str | None) -> HubModelInfo:
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

        try:
            info = self.api.model_info(repo, revision=revision, files_metadata=True)
        except GatedRepoError as e:
            raise HubAccessError(repo, 403, str(e)) from e
        except RepositoryNotFoundError as e:
            raise HubAccessError(repo, 404, str(e)) from e
        except HfHubHTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            raise HubAccessError(repo, status, str(e)) from e
        card = getattr(info, "card_data", None)
        lic = getattr(card, "license", None) if card is not None else None
        files = [
            HubFile(
                path=s.rfilename,
                size=getattr(s, "size", None),
                sha256=(s.lfs.sha256 if getattr(s, "lfs", None) is not None else None),
            )
            for s in (info.siblings or [])
        ]
        return HubModelInfo(
            repo=repo,
            revision=str(info.sha),
            license=str(lic) if lic else None,
            gated=bool(getattr(info, "gated", False)),
            files=files,
        )

    def snapshot(
        self, repo: str, revision: str, dest: Path, allow_patterns: list[str] | None
    ) -> Path:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

        try:
            path = snapshot_download(
                repo_id=repo,
                revision=revision,
                local_dir=str(dest),
                allow_patterns=allow_patterns or None,
                token=self.token,
            )
        except GatedRepoError as e:
            raise HubAccessError(repo, 403, str(e)) from e
        except HfHubHTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            raise HubAccessError(repo, status, str(e)) from e
        return Path(path)

    def read_text(self, repo: str, revision: str, path: str) -> str | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        try:
            local = hf_hub_download(
                repo_id=repo, filename=path, revision=revision, token=self.token
            )
        except EntryNotFoundError:
            return None
        return Path(local).read_text(encoding="utf-8", errors="replace")
