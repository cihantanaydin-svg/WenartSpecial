"""S8+S9 retry policy per view (ARCHITECTURE §S9 Policy).

geometry fail → lower strength (×0.7, at most twice) → new seed … up to ``max_retries``
→ hard structural composite (re-QA'd) → fallback to the pure Cycles render.
A geometry-failing image is never delivered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from archrender.core.schemas.qa import CandidateQA, CheckResult, ViewOutcome
from archrender.core.schemas.scene import SceneSpec
from archrender.models.profiles import RefineProfile
from archrender.models.roles import Img, Refiner
from archrender.qa.runner import QAContext
from archrender.refine.orchestrator import refine_image
from archrender.refine.strength import hard_structural_composite, strength_map
from archrender.render.passes import RenderPasses

StoreImage = Callable[[Img, str], str]  # (image, label) → sha256 of the stored PNG


@dataclass
class ViewPolicyResult:
    outcome: ViewOutcome
    delivered: Img
    log: list[str] = field(default_factory=list)


def _rank_key(c: CandidateQA) -> tuple[float, int]:
    edge = next((x.delta for x in c.checks if x.name == "structural_edge_f" and x.delta is not None), 0.0)
    return (edge, c.seed)


def run_view_policy(
    *,
    view_id: str,
    camera_id: str,
    base: Img,
    base_sha: str,
    passes: RenderPasses,
    spec: SceneSpec,
    refiner: Refiner,
    qa: QAContext,
    profile: RefineProfile,
    prompt: str,
    seed: int,
    store: StoreImage,
    cancelled: Callable[[], bool] = lambda: False,
    decor_enabled: bool = False,
) -> ViewPolicyResult:
    base_m = qa.measure_base(base)
    log: list[str] = []
    attempts: list[CandidateQA] = []
    images: dict[str, Img] = {}
    scale = 1.0
    reductions = 0
    seeds = [seed + i for i in range(profile.best_of_n)]
    retry = 0
    while True:
        for s in seeds:
            if cancelled():
                raise InterruptedError("cancelled")
            smap = strength_map(passes, spec, profile, scale=scale, decor_enabled=decor_enabled)
            img = refine_image(refiner, base, smap, prompt, s, profile)
            checks = qa.evaluate(img, base_m)
            sha = store(img, f"{view_id}_a{retry}_s{s}")
            cand = CandidateQA(
                view_id=view_id,
                candidate_id=f"{view_id}_a{retry}_s{s}",
                attempt=retry,
                seed=s,
                params={"strength_scale": round(scale, 4)},
                checks=checks,
                image_sha256=sha,
            )
            attempts.append(cand)
            images[cand.candidate_id] = img
        current = attempts[-len(seeds) :]
        passing = [c for c in current if c.passed]
        if passing:
            best = sorted(passing, key=_rank_key)[0]
            log.append(f"attempt {retry}: {len(passing)}/{len(current)} candidates passed; chose {best.candidate_id}")
            return _result(view_id, camera_id, "refined", best.image_sha256, base_sha, attempts,
                           f"Refined candidate {best.candidate_id} passed all QA checks.", images[best.candidate_id], log)
        failed_names = sorted({x.name for c in current for x in c.checks if not x.passed})
        log.append(f"attempt {retry}: all candidates failed ({', '.join(failed_names)})")
        if retry >= profile.max_retries:
            break
        retry += 1
        geometry_failed = all(not c.geometry_passed for c in current)
        if geometry_failed and reductions < 2:
            scale *= 0.7
            reductions += 1
            log.append(f"retry {retry}: geometry failure → strength scale {scale:.3f}")
        else:
            log.append(f"retry {retry}: new seed")
        seeds = [seed + 1000 * retry]

    # escalation rung: hard structural composite of the best geometry candidate
    best_fail = sorted(attempts, key=_rank_key)[0]
    composite = hard_structural_composite(base, images[best_fail.candidate_id], passes, spec)
    checks = qa.evaluate(composite, base_m)
    sha = store(composite, f"{view_id}_hard_composite")
    hc = CandidateQA(
        view_id=view_id,
        candidate_id=f"{view_id}_hard_composite",
        attempt=retry + 1,
        seed=best_fail.seed,
        params={"hard_composite_of": best_fail.candidate_id},
        checks=checks,
        image_sha256=sha,
    )
    attempts.append(hc)
    if hc.passed:
        log.append("hard structural composite passed QA")
        return _result(view_id, camera_id, "hard_composite", sha, base_sha, attempts,
                       "Refinement drifted; delivered the hard structural composite (Cycles structure).",
                       composite, log)
    log.append("hard structural composite failed; falling back to the Cycles render")
    return _result(view_id, camera_id, "fallback_base", base_sha, base_sha, attempts,
                   "All refinement attempts failed QA; delivered the unrefined Cycles render.", base, log)


def _result(
    view_id: str, camera_id: str, status: str, delivered_sha: str, base_sha: str,
    attempts: list[CandidateQA], reason: str, img: Img, log: list[str],
) -> ViewPolicyResult:
    uses_mocks = any(c.uses_mocks for c in attempts)
    outcome = ViewOutcome(
        view_id=view_id,
        camera_id=camera_id,
        status=status,  # type: ignore[arg-type]
        delivered_sha256=delivered_sha,
        base_sha256=base_sha,
        attempts=attempts,
        reason=reason,
        uses_mocks=uses_mocks,
    )
    return ViewPolicyResult(outcome=outcome, delivered=np.asarray(img, dtype=np.float32), log=log)


def delivered_checks(outcome: ViewOutcome) -> list[CheckResult]:
    for a in outcome.attempts:
        if a.image_sha256 == outcome.delivered_sha256:
            return a.checks
    return []
