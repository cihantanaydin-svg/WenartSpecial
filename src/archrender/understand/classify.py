"""Page classification: calibrated multinomial logistic regression over page features.

The model (weights, feature standardisation, temperature, class list, training provenance) is a
small JSON file under ``configs/classifier/``. It is trained by :mod:`archrender.understand.train`
on a synthetic corpus. With a VLM serving, its answer is combined with this model's distribution
(``combine_with_vlm``). Pages below the review threshold go to the review queue, and so does every
page with a feature outside the range seen in training: a linear model extrapolates to near-certain
answers there (a one-line DXF was "moodboard, 100 %"). Such features are clipped to the range.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.understanding import PAGE_CLASSES, Classification
from archrender.understand.features import FEATURE_NAMES

REVIEW_THRESHOLD = 0.8


def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / e.sum(axis=-1, keepdims=True), dtype=np.float64)


@dataclass
class PageClassifier:
    classes: list[str]
    features: list[str]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    weights: NDArray[np.float64]  # (n_features + 1, n_classes), last row = bias
    temperature: float
    info: dict[str, Any]
    lo: NDArray[np.float64]  # per-feature range seen in training (raw units)
    hi: NDArray[np.float64]

    @classmethod
    def load(cls, path: Path) -> PageClassifier:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise ArchRenderError(
                ErrorCode.MODEL_NOT_DOWNLOADED,
                f"Page classifier model {path.name} is missing.",
                "Train it: python -m archrender.understand.train (see configs/classifier/README.md).",
            ) from e
        if data["features"] != FEATURE_NAMES or "lo" not in data:
            raise ArchRenderError(
                ErrorCode.MODEL_NOT_DOWNLOADED,
                f"Page classifier {path.name} was trained on a different feature layout.",
                "Retrain it with python -m archrender.understand.train.",
            )
        return cls(
            classes=data["classes"],
            features=data["features"],
            mean=np.array(data["mean"]),
            std=np.array(data["std"]),
            weights=np.array(data["weights"]),
            temperature=float(data["temperature"]),
            info=data.get("info", {}),
            lo=np.array(data["lo"]),
            hi=np.array(data["hi"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "classes": self.classes,
                    "features": self.features,
                    "mean": self.mean.round(8).tolist(),
                    "std": self.std.round(8).tolist(),
                    "weights": self.weights.round(8).tolist(),
                    "temperature": round(self.temperature, 6),
                    "lo": self.lo.round(8).tolist(),
                    "hi": self.hi.round(8).tolist(),
                    "info": self.info,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )

    def out_of_range(self, x: NDArray[np.float64]) -> list[str]:
        """Features of one page outside the training range (5 % of the range as tolerance)."""
        tol = 0.05 * (self.hi - self.lo) + 1e-9
        v = np.atleast_2d(x)[0]
        bad = (v < self.lo - tol) | (v > self.hi + tol)
        return [f for f, b in zip(self.features, bad, strict=True) if b]

    def logits(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = (np.clip(np.atleast_2d(x), self.lo, self.hi) - self.mean) / self.std
        z = np.hstack([z, np.ones((z.shape[0], 1))])
        return np.asarray(z @ self.weights, dtype=np.float64)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return softmax(self.logits(x) / self.temperature)

    def classify(
        self, page_id: str, x: NDArray[np.float64], sources: dict[str, Any]
    ) -> Classification:
        p = self.predict_proba(x)[0]
        i = int(p.argmax())
        label = self.classes[i]
        unfamiliar = self.out_of_range(x)
        if unfamiliar:
            sources = {**sources, "outside_training_range": unfamiliar}
        if label not in PAGE_CLASSES:
            raise ArchRenderError(
                ErrorCode.INTERNAL, f"Unknown class {label!r} in the model.", "Retrain the model."
            )
        return Classification(
            page_id=page_id,
            label=label,
            confidence=round(float(p[i]), 4),
            probabilities={c: round(float(v), 4) for c, v in zip(self.classes, p, strict=True)},
            sources=sources,
            needs_review=bool(p[i] < REVIEW_THRESHOLD or unfamiliar),
        )


def fit(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    classes: list[str],
    *,
    l2: float = 1e-2,
    x_cal: NDArray[np.float64] | None = None,
    y_cal: NDArray[np.int64] | None = None,
    info: dict[str, Any] | None = None,
) -> PageClassifier:
    """L2-regularised multinomial logistic regression (L-BFGS), then temperature scaling.

    The temperature may soften the regularised fit (T ≥ 1) but never sharpen it: the calibration
    corpus is synthetic and nearly separable, so an unbounded fit drives T to its lower limit and
    makes every real page look certain, which would empty the review queue."""
    from scipy.optimize import minimize, minimize_scalar

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    z = np.hstack([(x - mean) / std, np.ones((len(x), 1))])
    n, d = z.shape
    k = len(classes)
    onehot = np.eye(k)[y]

    def loss(wflat: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        w = wflat.reshape(d, k)
        p = softmax(z @ w)
        nll = -np.sum(onehot * np.log(p + 1e-12)) / n
        reg = 0.5 * l2 * np.sum(w[:-1] ** 2)
        grad = z.T @ (p - onehot) / n
        grad[:-1] += l2 * w[:-1]
        return float(nll + reg), grad.ravel()

    res = minimize(loss, np.zeros(d * k), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    w = res.x.reshape(d, k)
    seen = x if x_cal is None else np.vstack([x, x_cal])
    model = PageClassifier(
        classes,
        list(FEATURE_NAMES),
        mean,
        std,
        w,
        1.0,
        dict(info or {}),
        lo=seen.min(axis=0),
        hi=seen.max(axis=0),
    )
    if x_cal is not None and y_cal is not None and len(x_cal):
        logits = model.logits(x_cal)

        def nll_t(t: float) -> float:
            p = softmax(logits / t)
            return float(-np.mean(np.log(p[np.arange(len(y_cal)), y_cal] + 1e-12)))

        model.temperature = float(minimize_scalar(nll_t, bounds=(1.0, 20.0), method="bounded").x)
    model.info.update({"converged": bool(res.success), "train_nll": round(float(res.fun), 5)})
    return model


def combine_with_vlm(
    cls: Classification, answer: dict[str, Any], classes: list[str]
) -> Classification:
    """Pre-calibration combination of the heuristic model and the VLM (UNVERIFIED-ON-GPU).

    Equal-weight geometric mean of the two distributions; the VLM's distribution puts its stated
    confidence on its class and spreads the rest. Disagreement between two confident sources is
    always sent to review, and so is a page outside the heuristic model's training range unless the
    VLM confidently gives the same class. To be replaced by a combiner calibrated on VLM answers for real pages,
    which needs the pod (not built yet; PROGRESS.md lists it under UNVERIFIED-ON-GPU).
    """
    k = len(classes)
    c = min(max(float(answer.get("confidence", 0.0)), 0.0), 1.0)
    v = str(answer.get("class"))
    q = np.array([c if name == v else (1.0 - c) / (k - 1) for name in classes])
    p = np.array([cls.probabilities.get(name, 0.0) for name in classes])
    comb = np.sqrt(np.clip(p, 1e-9, 1) * np.clip(q, 1e-9, 1))
    comb /= comb.sum()
    i = int(comb.argmax())
    disagree = cls.label != v and cls.confidence >= 0.6 and c >= 0.6
    # heuristics on an unfamiliar page count only when the VLM confirms them
    unconfirmed = bool(cls.sources.get("outside_training_range")) and (
        v != cls.label or c < REVIEW_THRESHOLD
    )
    return Classification(
        page_id=cls.page_id,
        label=classes[i],
        confidence=round(float(comb[i]), 4),
        probabilities={n: round(float(x), 4) for n, x in zip(classes, comb, strict=True)},
        sources={
            **cls.sources,
            "vlm": {"class": v, "confidence": c, "evidence": str(answer.get("evidence", ""))[:300]},
            "combiner": "geometric-mean (pre-calibration)",
        },
        needs_review=bool(disagree or unconfirmed or comb[i] < REVIEW_THRESHOLD),
    )
