"""Plan extraction (S2) from vector sources against exact synthetic ground truth.

Documents go through the real sandboxed intake (S0), then the extractors read the primitives the
intake stored. Acceptance (docs/PLAN.md, Phase 3): vector sources wall F1 ≥ 0.98, opening F1 ≥
0.95, scale error ≤ 1 %. The ground truth is mapped into the extractor's frame through the
document transforms, so geometry and scale are scored separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from archrender.core.config import Settings
from archrender.core.schemas.document import PageRef, Word
from archrender.ingest.intake import ingest_file, page_refs
from archrender.pipeline.services import Services
from archrender.plan.builder import build_plan
from archrender.plan.extract import dxf_extract, pdf_extract
from archrender.plan.ifc import plan_from_ifc
from archrender.plan.metrics import compose, plan_scores, transform_plan
from archrender.plan.scale import dimension_pairs, from_dimensions, fuse, open_segments, stated
from archrender.synth.dxf import plan_dxf, random_plan_dxf
from archrender.synth.ifc import IfcStorey, plan_ifc
from archrender.synth.plan import Variant, gt_plan, random_spec
from archrender.synth.sheets import WALL_STYLES, floor_plan_page
from archrender.understand.titleblock import extract_title_block

VARIANTS: list[Variant] = ["manhattan", "rotated", "skewed", "arc", "skewed_arc"]


@pytest.fixture
def svc(settings: Settings, project_id: str) -> Services:
    return Services.create(settings)


def _ingest(svc: Services, tmp: Path, name: str, data: bytes) -> PageRef:
    p = tmp / name
    p.write_bytes(data)
    res = ingest_file(svc, "prj_test", p, name)
    return next(pg for pg in page_refs(svc, "prj_test") if pg.document_id == res.document_id)


def _json(svc: Services, ref: object) -> object:
    return json.loads(svc.store("prj_test").read_bytes(ref))  # type: ignore[arg-type]


def _check(scores: dict[str, dict[str, float]]) -> None:
    assert scores["walls"]["f1"] >= 0.98, scores
    assert scores["openings"]["f1"] >= 0.95, scores
    assert scores["rooms"]["f1"] >= 0.95, scores


@pytest.mark.parametrize("seed", range(5))
def test_dxf_plans_are_extracted_to_the_vector_targets(
    svc: Services, tmp_path: Path, seed: int
) -> None:
    rng = np.random.default_rng(100 + seed)
    spec = random_spec(rng, variant=VARIANTS[seed % len(VARIANTS)])
    data, gt = random_plan_dxf(spec, rng)
    page = _ingest(svc, tmp_path, f"plan{seed}.dxf", data)
    assert page.vectors is not None
    ex = dxf_extract(_json(svc, page.vectors), _json(svc, page.content), page.document_id)  # type: ignore[arg-type]
    if not gt["unitless"]:  # the unit is exact from $INSUNITS
        assert ex.unit.value == pytest.approx(1 / {"m": 1, "cm": 100, "mm": 1000}[gt["unit"]])
    res = build_plan(ex.prims, project="p", version="v")
    truth = transform_plan(gt_plan(spec), compose(gt["plan_to_doc"], ex.doc_to_plan))
    _check(plan_scores(res.plan, truth))


def test_unitless_dxf_takes_its_unit_from_the_door_swings(svc: Services, tmp_path: Path) -> None:
    spec = random_spec(np.random.default_rng(7), variant="manhattan")
    data, _ = plan_dxf(spec, unit="mm", unitless=True, layers="zero")
    page = _ingest(svc, tmp_path, "unitless.dxf", data)
    ex = dxf_extract(_json(svc, page.vectors), _json(svc, page.content), page.document_id)  # type: ignore[arg-type]
    assert ex.unit.value == pytest.approx(0.001)
    assert ex.unit.provenance[0].method == "derived"


@pytest.mark.parametrize("seed", range(5))
def test_vector_pdf_plans_are_extracted_with_a_measured_scale(
    svc: Services, tmp_path: Path, seed: int
) -> None:
    rng = np.random.default_rng(200 + seed)
    spec = random_spec(rng, variant=VARIANTS[seed % len(VARIANTS)])
    sheet = floor_plan_page(
        rng, spec=spec, wall_style=WALL_STYLES[seed % len(WALL_STYLES)], stamp=seed == 1
    )
    page = _ingest(svc, tmp_path, f"plan{seed}.pdf", sheet.pdf)
    words = [Word.model_validate(w) for w in _json(svc, page.words)]  # type: ignore[attr-defined]
    paths = _json(svc, page.vectors)
    assert page.dpi and page.width_px and page.height_px
    tb = extract_title_block("p", words, page.width_px, page.height_px, "pdf_text")
    assert tb is not None and tb.scale is not None
    ests = [stated(tb.scale.value, 0.0254 / page.dpi)]
    dims = from_dimensions(
        dimension_pairs(words, open_segments(paths), max_gap_px=6 * page.dpi / 25.4)
    )  # type: ignore[arg-type]
    assert dims is not None and dims.n >= 3  # the dimension chains corroborate the stated scale
    ests.append(dims)
    fused = fuse(ests)
    assert not fused.conflicted and fused.estimate is not None
    true_m_per_px = sheet.gt["scale"] * 0.0254 / page.dpi
    assert abs(fused.estimate.value - true_m_per_px) / true_m_per_px <= 0.01
    ex = pdf_extract(paths, words, fused.estimate.value)  # type: ignore[arg-type]
    res = build_plan(ex.prims, project="p", version="v", source="pdf_vector")
    truth = transform_plan(gt_plan(spec), compose(sheet.gt["plan_to_page_px"], ex.doc_to_plan))
    scores = plan_scores(res.plan, truth)
    _check(scores)
    assert scores["rooms"]["name_accuracy"] == 1.0


@pytest.mark.parametrize("variant", ["manhattan", "skewed", "arc"])
def test_ifc_models_are_read_exactly(svc: Services, tmp_path: Path, variant: Variant) -> None:
    spec = random_spec(np.random.default_rng(31), variant=variant)
    page = _ingest(svc, tmp_path, f"{variant}.ifc", plan_ifc([IfcStorey(spec, "Zemin Kat", 0.0)]))
    plan = plan_from_ifc(_json(svc, page.vectors), project="p", version="v")  # type: ignore[arg-type]
    scores = plan_scores(plan, gt_plan(spec))
    assert scores["walls"]["f1"] == 1.0 and scores["openings"]["f1"] == 1.0
    assert scores["rooms"]["f1"] == 1.0 and scores["rooms"]["name_accuracy"] == 1.0
    tags = {o.tag for o in plan.openings}
    assert tags == {o.tag for o in spec.openings}
