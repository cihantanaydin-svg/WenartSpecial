from __future__ import annotations

from pathlib import Path

import pytest

from archrender.core.cas import ProjectStore
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.hashing import canonical_json, sha256_json
from archrender.core.logging import redact
from archrender.core.paths import check_id, safe_join, sanitize_filename


def test_canonical_json_is_order_and_float_noise_independent() -> None:
    a = {"b": 1, "a": [0.1 + 0.2, "x"]}
    b = {"a": [0.3, "x"], "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert sha256_json(a) == sha256_json(b)


def test_safe_join_rejects_escape(tmp_path: Path) -> None:
    assert safe_join(tmp_path, "a", "b") == (tmp_path / "a" / "b").resolve()
    with pytest.raises(ArchRenderError) as e:
        safe_join(tmp_path, "..", "etc")
    assert e.value.code == ErrorCode.VALIDATION
    with pytest.raises(ArchRenderError):
        safe_join(tmp_path, "/etc/passwd")


@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".hidden", "x" * 65, "a b"])
def test_check_id_rejects(bad: str) -> None:
    with pytest.raises(ArchRenderError):
        check_id(bad)


def test_sanitize_filename_keeps_turkish_letters() -> None:
    assert sanitize_filename("../../Kat Planı ölçek 1-50.pdf") == "Kat Planı ölçek 1-50.pdf"
    assert sanitize_filename("..\\evil\\..\\name.dxf") == "name.dxf"
    assert sanitize_filename("...") == "file"


def test_cas_roundtrip_dedupe_and_isolation(tmp_path: Path) -> None:
    s1 = ProjectStore(tmp_path, "prj_a")
    s2 = ProjectStore(tmp_path, "prj_b")
    r1 = s1.put_bytes(b"hello", "text/plain", "h.txt")
    r1b = s1.put_bytes(b"hello")
    assert r1.sha256 == r1b.sha256
    assert s1.read_bytes(r1) == b"hello"
    assert not s2.exists(r1.sha256)  # no cross-project sharing
    with pytest.raises(ArchRenderError):
        s2.path(r1)
    src = tmp_path / "f.bin"
    src.write_bytes(b"x" * 10_000)
    r2 = s1.put_file(src, move=True)
    assert not src.exists()
    assert s1.path(r2).stat().st_size == 10_000
    assert not list(s1.tmp_dir.iterdir())  # no leftovers


def test_redaction() -> None:
    line = "key ark_abc123_SECRETsecret hf_abcdefghijklmnop Bearer xyz.token ghp_abcdefghijk1234"
    out = redact(line)
    assert (
        "SECRET" not in out and "hf_abc" not in out and "xyz.token" not in out and "ghp_" not in out
    )
