"""Page classifier mechanics on toy data: fit, temperature bound, save/load, and the training-range
guard that sends unfamiliar pages to review instead of trusting an extrapolated answer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from archrender.core.errors import ArchRenderError
from archrender.core.schemas.understanding import PAGE_CLASSES, Classification
from archrender.understand.classify import REVIEW_THRESHOLD, PageClassifier, combine_with_vlm, fit
from archrender.understand.features import FEATURE_NAMES

CLASSES = ["floor_plan", "section"]


def _toy(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = rng.normal(0.0, 1.0, (n, len(FEATURE_NAMES)))
    x[:, 0] += np.where(y == 1, 1.5, -1.5)  # one informative feature, overlapping classes
    return x, y


def test_fit_separates_classes_and_never_sharpens_calibration(tmp_path: Path) -> None:
    x, y = _toy(400, 1)
    xc, yc = _toy(200, 2)
    model = fit(x, y, CLASSES, x_cal=xc, y_cal=yc)
    assert model.temperature >= 1.0
    pred = model.predict_proba(xc).argmax(axis=1)
    assert np.mean(pred == yc) > 0.85
    model.save(tmp_path / "m.json")
    again = PageClassifier.load(tmp_path / "m.json")
    np.testing.assert_allclose(again.predict_proba(xc), model.predict_proba(xc), atol=1e-6)


def test_pages_outside_the_training_range_go_to_review_with_the_features_named() -> None:
    x, y = _toy(400, 3)
    model = fit(x, y, CLASSES)
    inside = model.classify("p1", x[0], sources={})
    assert "outside_training_range" not in inside.sources

    far = x[0].copy()
    far[5] = x[:, 5].max() + 50.0  # nothing like it was seen in training
    far[0] = 40.0  # would give an extreme logit if not clipped
    got = model.classify("p2", far, sources={})
    assert got.needs_review
    assert set(got.sources["outside_training_range"]) == {FEATURE_NAMES[0], FEATURE_NAMES[5]}
    clipped = far.copy()
    clipped[0], clipped[5] = x[:, 0].max(), x[:, 5].max()
    np.testing.assert_allclose(model.logits(far), model.logits(clipped))


def test_model_without_training_range_is_refused(tmp_path: Path) -> None:
    import json

    x, y = _toy(100, 4)
    fit(x, y, CLASSES).save(tmp_path / "m.json")
    data = json.loads((tmp_path / "m.json").read_text())
    del data["lo"]
    (tmp_path / "m.json").write_text(json.dumps(data))
    with pytest.raises(ArchRenderError) as e:
        PageClassifier.load(tmp_path / "m.json")
    assert "Retrain" in e.value.fix_hint


def _unfamiliar(label: str, conf: float) -> Classification:
    rest = (1 - conf) / (len(PAGE_CLASSES) - 1)
    return Classification(
        page_id="p",
        label=label,  # type: ignore[arg-type]
        confidence=conf,
        probabilities={c: (conf if c == label else rest) for c in PAGE_CLASSES},
        sources={"outside_training_range": ["log_paths"]},
        needs_review=True,
    )


def test_an_unfamiliar_page_leaves_review_only_when_the_vlm_confirms_it() -> None:
    classes = list(PAGE_CLASSES)
    confirmed = combine_with_vlm(
        _unfamiliar("floor_plan", 0.97), {"class": "floor_plan", "confidence": 0.95}, classes
    )
    assert not confirmed.needs_review and confirmed.confidence >= REVIEW_THRESHOLD
    unsure = combine_with_vlm(
        _unfamiliar("floor_plan", 0.97), {"class": "floor_plan", "confidence": 0.6}, classes
    )
    assert unsure.needs_review
    other = combine_with_vlm(
        _unfamiliar("floor_plan", 0.97), {"class": "moodboard", "confidence": 0.5}, classes
    )
    assert other.needs_review and other.sources["outside_training_range"] == ["log_paths"]
