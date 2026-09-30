"""VLM client against a fake OpenAI-compatible server, the pre-calibration combiner, and the
supervisor starting vLLM only when weights are installed. (The real vLLM path is UNVERIFIED-ON-GPU.)"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from archrender.core.config import REPO_ROOT
from archrender.core.errors import ArchRenderError
from archrender.core.schemas.understanding import PAGE_CLASSES, Classification
from archrender.models.impls.vllm_vlm import VllmVlm
from archrender.models.profiles import HardwareProfile
from archrender.models.registry import Registry
from archrender.ops import supervisor
from archrender.understand.classify import combine_with_vlm

CFG = REPO_ROOT / "configs"


def _entry():  # type: ignore[no-untyped-def]
    return Registry.load(CFG).get("qwen3.6-27b-fp8")


def test_classify_page_sends_schema_constrained_request_and_parses_the_answer() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3.6-27b-fp8"}]})
        if request.url.path == "/wake_up":
            return httpx.Response(200)
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            seen["body"] = body
            answer = {
                "class": "section",
                "confidence": 0.93,
                "evidence": "cut slabs and level markers",
            }
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(answer)}}]}
            )
        return httpx.Response(404)

    vlm = VllmVlm(
        _entry(), http=httpx.Client(base_url="http://vllm", transport=httpx.MockTransport(handler))
    )
    vlm.load()
    out = vlm.classify_page(np.zeros((40, 60, 3), np.uint8), "A-A KESİTİ ±0,00 +3,00")
    assert out["class"] == "section"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["temperature"] == 0 and body["model"] == "qwen3.6-27b-fp8"
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["class"]["enum"] == list(PAGE_CLASSES)
    content = body["messages"][1]["content"]
    assert (
        content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        and "KESİTİ" in content[1]["text"]
    )


def test_unreachable_server_is_reported_with_a_fix() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    vlm = VllmVlm(
        _entry(), http=httpx.Client(base_url="http://vllm", transport=httpx.MockTransport(refuse))
    )
    vlm.ready_timeout_s = 0.0
    assert vlm.probe() is False
    with pytest.raises(ArchRenderError) as e:
        vlm.load()
    assert "supervisorctl" in e.value.fix_hint


def test_invalid_json_from_the_model_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    vlm = VllmVlm(
        _entry(), http=httpx.Client(base_url="http://vllm", transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ArchRenderError) as e:
        vlm.chat_json([], {"type": "object"}, name="x")
    assert e.value.retryable


def _cls(label: str, conf: float) -> Classification:
    rest = (1 - conf) / (len(PAGE_CLASSES) - 1)
    probs = {c: (conf if c == label else rest) for c in PAGE_CLASSES}
    return Classification(
        page_id="p", label=label, confidence=conf, probabilities=probs, sources={}
    )  # type: ignore[arg-type]


def test_combiner_agreement_raises_confidence_and_disagreement_goes_to_review() -> None:
    classes = list(PAGE_CLASSES)
    agree = combine_with_vlm(
        _cls("floor_plan", 0.7), {"class": "floor_plan", "confidence": 0.95}, classes
    )
    assert (
        agree.label == "floor_plan" and agree.confidence > 0.7 and not agree.needs_review
    )  # raised, above review
    clash = combine_with_vlm(
        _cls("floor_plan", 0.8), {"class": "ceiling_plan", "confidence": 0.9}, classes
    )
    assert clash.needs_review and clash.sources["vlm"]["class"] == "ceiling_plan"


def test_supervisor_starts_vllm_only_when_weights_are_installed(tmp_path: Path) -> None:
    prof = HardwareProfile.load(CFG, "gpu80")
    reg = Registry.load(CFG)
    text = supervisor.render(
        prof, reg, tmp_path / "logs", venv="/opt/venv", litestream=None, data_dir=tmp_path
    )
    assert "[program:vllm" not in text  # weights not installed → role degrades, no crash loop
    snap = tmp_path / "models" / "snapshots" / "qwen3.6-27b-fp8" / "abc123"
    snap.mkdir(parents=True)
    (tmp_path / "models" / "installed").mkdir(parents=True)
    (tmp_path / "models" / "installed" / "qwen3.6-27b-fp8.ok").write_text(
        json.dumps({"revision": "abc123"})
    )
    text = supervisor.render(
        prof, reg, tmp_path / "logs", venv="/opt/venv", litestream=None, data_dir=tmp_path
    )
    assert f"vllm serve {snap}" in text and "--port 8101" in text and "VLLM_SERVER_DEV_MODE" in text
    assert "--gpu-memory-utilization 0.5" in text
