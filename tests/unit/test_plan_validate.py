"""Plan validators: every injected defect is detected (Phase-3 acceptance: 100 %), and the clean
synthetic plans raise nothing.

Defects: an open room (a wall deleted), an opening outside its host wall, overlapping rooms, an
area label more than 3 % off, an unreachable room (its doors removed).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pytest

from archrender.core.schemas.plan import PlanGraph
from archrender.plan.defects import DEFECTS, Injected, area_label_off, detected, found
from archrender.plan.validate import validate_plan
from archrender.synth.plan import Variant, gt_plan, random_spec

VARIANTS: list[Variant] = ["manhattan", "rotated", "skewed", "arc", "skewed_arc"]
CASES = list(itertools.product(range(6), VARIANTS))


def _plan(seed: int, variant: Variant) -> PlanGraph:
    return gt_plan(random_spec(np.random.default_rng(seed), variant=variant))


def _codes(plan: PlanGraph) -> dict[str, set[str]]:
    return found(plan)


@pytest.mark.parametrize(("seed", "variant"), CASES)
def test_clean_plans_raise_no_issue(seed: int, variant: Variant) -> None:
    assert validate_plan(_plan(seed, variant)) == []


@pytest.mark.parametrize("defect", DEFECTS, ids=lambda d: d.__name__)
@pytest.mark.parametrize(("seed", "variant"), CASES)
def test_every_injected_defect_is_detected(
    seed: int, variant: Variant, defect: Callable[[PlanGraph], Injected]
) -> None:
    inj = defect(_plan(seed, variant))
    assert inj.element_ids, inj.defect
    assert detected(inj), (inj.defect, inj.code, inj.element_ids, _codes(inj.plan))


@pytest.mark.parametrize(("seed", "variant"), CASES)
def test_area_labels_within_3_percent_are_not_flagged(seed: int, variant: Variant) -> None:
    inj = area_label_off(_plan(seed, variant), factor=0.02)
    assert "ROOM_AREA_MISMATCH" not in _codes(inj.plan)
