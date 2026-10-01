"""Scale reconciliation: any disagreement > 1.5 % between precise estimators is always flagged
(Phase-3 acceptance, property test), agreement never is; dimension RANSAC resolves units."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from archrender.plan.scale import (
    CONFLICT_REL,
    ScaleEstimate,
    from_dimensions,
    from_door_radii,
    fuse,
    stated,
)


@given(
    base=st.floats(min_value=1e-4, max_value=10.0),
    rel=st.floats(min_value=CONFLICT_REL * 1.0001, max_value=0.5),
    sigma=st.floats(min_value=0.0001, max_value=0.01),
)
@settings(max_examples=300)
def test_disagreement_above_threshold_is_always_flagged(
    base: float, rel: float, sigma: float
) -> None:
    a = ScaleEstimate(base, 0.0005, "stated")
    b = ScaleEstimate(base * (1 + rel), sigma, "dimensions", 5)
    res = fuse([a, b])
    assert res.conflicted
    best = a if a.rel_sigma <= b.rel_sigma else b
    assert res.estimate is best  # the best-documented estimator is proposed, the user decides


@given(
    base=st.floats(min_value=1e-4, max_value=10.0),
    rel=st.floats(min_value=0.0, max_value=CONFLICT_REL * 0.999),
)
@settings(max_examples=300)
def test_agreement_within_threshold_is_never_flagged(base: float, rel: float) -> None:
    res = fuse(
        [
            ScaleEstimate(base, 0.0005, "stated"),
            ScaleEstimate(base * (1 + rel), 0.004, "dimensions", 4),
        ]
    )
    assert not res.conflicted
    assert res.estimate is not None
    assert (
        min(base, base * (1 + rel)) * 0.999999
        <= res.estimate.value
        <= max(base, base * (1 + rel)) * 1.000001
    )


def test_priors_never_raise_a_conflict_and_are_used_only_alone() -> None:
    prior = from_door_radii([100.0, 110.0, 95.0])
    assert prior is not None and prior.method == "door_prior"
    res = fuse([stated(50, 0.0254 / 300), prior])
    assert not res.conflicted and res.estimate is not None and res.estimate.method == "stated"
    alone = fuse([prior])
    assert alone.estimate is prior


def test_dimension_ransac_picks_the_consistent_unit_reading() -> None:
    # 1 px = 0.004 m; "350" is cm (3.50 m), "4,20" is m; one outlier string
    pairs = [(875.0, (3.5, 0.35)), (1050.0, (4.2,)), (500.0, (2.0, 0.2)), (300.0, (9.99,))]
    est = from_dimensions(pairs)
    assert est is not None and est.n == 3
    assert abs(est.value - 0.004) / 0.004 < 1e-6


def test_a_single_weak_estimate_is_a_blocking_question_for_gate_a() -> None:
    from archrender.plan.scale import ScaleEstimate, ScaleResult
    from archrender.plan.stage import _scale_conflict

    door = ScaleEstimate(0.0091, 0.12, "door_prior", 6, "median door swing")
    [c] = _scale_conflict("pg1", ScaleResult(door, [door]))
    assert c.key == "scale/pg1" and len(c.candidates) == 1 and c.severity == "error"
    assert "weak" in c.rule
    good = ScaleEstimate(0.0095, 0.0005, "stated", 1, "1:75 at 200 dpi")
    assert _scale_conflict("pg1", ScaleResult(good, [good])) == []
