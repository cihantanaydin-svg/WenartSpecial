"""S2 in the pipeline: the project's plan (ARCHITECTURE §S2).

Sources, by precedence for each level: an IFC model (authoritative geometry) > a DXF > a vector
PDF > a raster (scan, or a phone photo rectified in S1). Pages count as plan sources when S1 (or
the user) labelled them ``floor_plan``; an IFC model counts as a whole. Every source goes through
its extractor and the shared PlanBuilder; the scale of a PDF or raster page is measured (stated
scale × page resolution, dimension strings, door prior) and fused, a disagreement becoming a
Gate A conflict. Several levels (one plan page each, by the title block's level) are stacked with
a default storey height (an assumption). The result is validated and stored as a draft
PlanVersion for Gate A (``archrender.plan.versions``).
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image

from archrender.core.cas import CasRef
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.common import ModelRef, Severity, Strict
from archrender.core.schemas.document import PageRef, Word
from archrender.core.schemas.plan import DocTransform, Level, PlanGraph, ValidationIssue
from archrender.core.schemas.provenance import Assumption, Conflict, fact
from archrender.ingest.intake import page_refs
from archrender.pipeline.engine import StageContext, StageDef, StageEngine
from archrender.pipeline.services import Services
from archrender.plan import versions
from archrender.plan.builder import build_plan
from archrender.plan.extract import dxf_extract, pdf_extract
from archrender.plan.ifc import plan_from_ifc
from archrender.plan.raster import raster_extract, raster_scale
from archrender.plan.scale import (
    ScaleResult,
    dimension_pairs,
    from_dimensions,
    fuse,
    open_segments,
    stated,
)
from archrender.plan.validate import blocking, validate_plan
from archrender.understand.stage import S1Out, _effective_label, run_understanding
from archrender.understand.text import fold

SourceKind = Literal["ifc", "dxf", "pdf_vector", "raster"]
RANK: dict[str, int] = {"ifc": 0, "dxf": 1, "pdf_vector": 2, "raster": 3}
DEFAULT_LEVEL = "Zemin Kat"
STOREY_HEIGHT_M = 3.0


class S2Page(Strict):
    page: PageRef
    label: str
    analysis: S1Out | None = None


class S2In(Strict):
    project_id: str
    pages: list[S2Page]


class S2Out(Strict):
    plan: PlanGraph
    plan_json: CasRef
    issues: list[ValidationIssue]
    extraction: dict[str, Any]  # sources, scale estimates, notes, assist triggers


@dataclass
class PageResult:
    page_id: str
    source: SourceKind
    level_name: str
    plan: PlanGraph
    transform: DocTransform
    scale: dict[str, Any] | None = None
    conflicts: list[Conflict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unexplained_gaps: int = 0


def _json(store: Any, ref: CasRef | None) -> Any:
    return json.loads(store.read_bytes(ref)) if ref is not None else None


def _source_kind(page: PageRef, content: Any) -> SourceKind | None:
    if page.kind == "ifc":
        return "ifc"
    if page.kind == "dxf":
        return "dxf"
    if page.kind == "pdf_page":
        stats = (content or {}).get("objects", {})
        drawn = int(stats.get("path", 0)) if isinstance(stats, dict) else 0
        image_frac = float((content or {}).get("image_area_fraction", 0.0))
        # a scanned PDF page is one picture: read it as a raster
        return (
            "pdf_vector"
            if page.vectors is not None and drawn >= 30 and image_frac < 0.5
            else ("raster" if page.raster is not None else None)
        )
    if page.kind == "image":
        return "raster"
    return None


def _scale_summary(res: ScaleResult, chosen: float) -> dict[str, Any]:
    return {
        "m_per_unit": chosen,
        "method": res.estimate.method if res.estimate else None,
        "estimates": [
            {
                "method": e.method,
                "value": e.value,
                "rel_sigma": e.rel_sigma,
                "n": e.n,
                "detail": e.detail,
            }
            for e in res.estimates
        ],
        "conflicts": [{"a": a.method, "b": b.method, "rel": rel} for a, b, rel in res.conflicts],
    }


def _scale_conflict(page_id: str, res: ScaleResult) -> list[Conflict]:
    if not res.conflicted or res.estimate is None:
        return []
    cands = [
        {"method": e.method, "m_per_unit": e.value, "rel_sigma": e.rel_sigma, "detail": e.detail}
        for e in res.estimates
    ]
    proposed = next(i for i, e in enumerate(res.estimates) if e is res.estimate)
    return [
        Conflict(
            key=f"scale/{page_id}",
            candidates=cands,
            proposed=proposed,
            rule="scale estimators disagree by more than 1.5 %: the best-documented one is "
            "proposed, the user confirms at Gate A",
            severity=Severity.ERROR,
        )
    ]


class Extractor:
    """Per-page extraction (pure functions of the stored page data and the S1 analysis)."""

    def __init__(self, svc: Services, store: Any, project_id: str) -> None:
        self.svc = svc
        self.store = store
        self.project_id = project_id

    def page(self, p: S2Page, kind: SourceKind) -> PageResult:
        page = p.page
        level = DEFAULT_LEVEL
        a = p.analysis
        if a and a.title_block and a.title_block.level:
            level = a.title_block.level.value
        if kind == "ifc":
            return self._ifc(page)
        if kind == "dxf":
            return self._dxf(page, level)
        if kind == "pdf_vector":
            return self._pdf(page, a, level)
        return self._raster(page, a, level)

    def _ifc(self, page: PageRef) -> PageResult:
        model = _json(self.store, page.vectors)
        plan = plan_from_ifc(
            model, project=self.project_id, version="extraction", source_doc=page.document_id
        )
        t = DocTransform(
            doc_id=page.document_id, page=page.index, matrix=((1, 0, 0), (0, 1, 0)), method="ifc"
        )
        return PageResult(page.id, "ifc", plan.levels[0].name, plan, t)

    def _dxf(self, page: PageRef, level: str) -> PageResult:
        ex = dxf_extract(
            _json(self.store, page.vectors), _json(self.store, page.content), page.document_id
        )
        res = build_plan(
            ex.prims,
            project=self.project_id,
            version="extraction",
            level_name=level,
            source_doc=page.document_id,
            source="dxf",
        )
        t = DocTransform(
            doc_id=page.document_id, page=page.index, matrix=_m(ex.doc_to_plan), method="dxf_units"
        )
        return PageResult(
            page.id,
            "dxf",
            level,
            res.plan,
            t,
            {"m_per_unit": ex.unit.value, "method": ex.unit.provenance[0].method},
            [],
            ex.notes + res.notes,
            res.unexplained_gaps,
        )

    def _words(self, page: PageRef, a: S1Out | None) -> list[Word]:
        raw = _json(self.store, page.words) if page.words is not None else None
        if not raw and a is not None and a.words is not None:
            raw = _json(self.store, a.words)
        return [Word.model_validate(w) for w in raw or []]

    def _pdf(self, page: PageRef, a: S1Out | None, level: str) -> PageResult:
        words = self._words(page, a)
        paths = _json(self.store, page.vectors)
        dpi = page.dpi or 300.0
        ests = []
        if a is not None and a.scale is not None:
            ests.append(stated(a.scale.value, 0.0254 / dpi))
        dims = from_dimensions(
            dimension_pairs(words, open_segments(paths), max_gap_px=6 * dpi / 25.4)
        )
        if dims is not None:
            ests.append(dims)
        res = fuse(ests)
        if res.estimate is None:
            raise ArchRenderError(
                ErrorCode.PLAN_SCALE_CONFLICT,
                f"Page {page.id}: no scale (no stated scale and no dimension strings).",
                "Confirm the scale at Gate A (calibrate with a known length), or upload a DXF/IFC.",
            )
        m = res.estimate.value
        ex = pdf_extract(paths, words, m)
        b = build_plan(
            ex.prims,
            project=self.project_id,
            version="extraction",
            level_name=level,
            source_doc=page.document_id,
            source="pdf_vector",
        )
        t = DocTransform(
            doc_id=page.document_id,
            page=page.index,
            matrix=_m(ex.doc_to_plan),
            method=res.estimate.method,
        )
        return PageResult(
            page.id,
            "pdf_vector",
            level,
            b.plan,
            t,
            _scale_summary(res, m),
            _scale_conflict(page.id, res),
            b.notes,
            b.unexplained_gaps,
        )

    def _raster(self, page: PageRef, a: S1Out | None, level: str) -> PageResult:
        ref = a.rectified if a is not None and a.rectified is not None else page.raster
        if ref is None:
            raise ArchRenderError(
                ErrorCode.PLAN_NO_PLAN_FOUND,
                f"Page {page.id} has no raster.",
                "Re-upload the page.",
            )
        rgb = np.asarray(Image.open(io.BytesIO(self.store.read_bytes(ref))).convert("RGB"))
        gray = np.asarray(Image.fromarray(rgb).convert("L"))
        colour = bool(
            np.abs(rgb[..., 0].astype(np.int16) - rgb[..., 2].astype(np.int16)).mean() > 2
        )
        words = self._words(page, a)
        photo = a is not None and a.rectified is not None
        dpi = None if photo else page.dpi
        ocr, _ = self.svc.models.get_with_fallback("ocr", "S2")
        sc, rv = raster_scale(
            gray,
            words,
            stated_n=a.scale.value if a and a.scale and not photo else None,
            dpi=dpi,
            rgb=rgb if colour else None,
            reader=ocr,
        )
        if sc.result.estimate is None:
            raise ArchRenderError(
                ErrorCode.PLAN_SCALE_CONFLICT,
                f"Page {page.id}: no scale could be measured (no stated scale with a known DPI, no "
                "readable dimension strings, no door swings).",
                "Calibrate the scale at Gate A with a known length, or upload the vector original.",
            )
        ex = raster_extract(gray, words, sc.m_per_px, rgb=rgb if colour else None, vectors=rv)
        b = build_plan(
            ex.prims,
            project=self.project_id,
            version="extraction",
            level_name=level,
            source_doc=page.document_id,
            source="raster",
        )
        t = DocTransform(
            doc_id=page.document_id,
            page=page.index,
            matrix=_m(ex.doc_to_plan),
            method=sc.result.estimate.method,
        )
        notes = sc.notes + ex.notes + b.notes
        if photo:
            notes.insert(
                0, "phone photo: rectified sheet (S1); scale from dimensions or door swings"
            )
        return PageResult(
            page.id,
            "raster",
            level,
            b.plan,
            t,
            _scale_summary(sc.result, sc.m_per_px),
            _scale_conflict(page.id, sc.result),
            notes,
            b.unexplained_gaps,
        )


def _m(a: list[list[float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(a[0][0]), float(a[0][1]), float(a[0][2])),
        (float(a[1][0]), float(a[1][1]), float(a[1][2])),
    )


def _rename(plan: PlanGraph, level_id: str, prefix: str) -> PlanGraph:
    """The single-level plan of one page as level ``level_id`` with prefixed element ids."""
    d = plan.model_dump(mode="json")
    wid = {w["id"]: prefix + w["id"] for w in d["walls"]}
    for w in d["walls"]:
        w["id"], w["level"] = wid[w["id"]], level_id
    for o in d["openings"]:
        o["id"], o["host_wall"] = prefix + o["id"], wid[o["host_wall"]]
    for r in d["rooms"]:
        r["id"], r["level"] = prefix + r["id"], level_id
    for c in d["columns"]:
        c["id"], c["level"] = prefix + c["id"], level_id
    d["levels"] = [{**d["levels"][0], "id": level_id}]
    return PlanGraph.model_validate(d)


_ORDINALS = {
    **dict.fromkeys(("birinci", "first"), 1),
    **dict.fromkeys(("ikinci", "second"), 2),
    **dict.fromkeys(("ucuncu", "third"), 3),
    **dict.fromkeys(("dorduncu", "fourth"), 4),
    **dict.fromkeys(("besinci", "fifth"), 5),
    **dict.fromkeys(("altinci", "sixth"), 6),
}
ROOF = 1000.0


def level_rank(name: str) -> float | None:
    """Storey order from a level name (Turkish or English): basement −1, ground 0, mezzanine 0.5,
    "1. Kat" / "First Floor" / "Level 1" 1, roof last; None when the name says nothing."""
    n = fold(name)
    words = n.split()
    num = re.search(r"\b(\d{1,2})(st|nd|rd|th)?\b", n)
    k = int(num.group(1)) if num else next((v for w, v in _ORDINALS.items() if w in words), None)
    if "bodrum" in words or "basement" in words or "sub" in words:
        return -float(k or 1)
    if "asma" in words or "mezzanine" in words:
        return 0.5
    if "cati" in words or "roof" in words or "teras" in words:
        return ROOF
    if "zemin" in words or "ground" in words or "giris" in words:
        return 0.0
    if k is not None and (
        {"kat", "kati", "floor", "level", "storey", "story"} & set(words)
        or (num is not None and num.group(2))
    ):
        return float(k)
    return None


def _stack(names: list[str]) -> tuple[list[str], list[float], bool]:
    """Levels in storey order with elevations (rank × storey height; unknown names above the
    known ones, in sheet order) and whether any order had to be assumed."""
    ranks = {nm: level_rank(nm) for nm in names}
    known = sorted((r, i, nm) for i, nm in enumerate(names) if (r := ranks[nm]) is not None)
    order = [nm for _, _, nm in known] + [nm for nm in names if ranks[nm] is None]
    top = max((r for r, _, _ in known if r != ROOF), default=-1.0)
    elev: list[float] = []
    for nm in order:
        r = ranks[nm]
        if r is None or r == ROOF:
            top = max(top, (elev[-1] / STOREY_HEIGHT_M) if elev else -1.0) + 1.0
            r = top
        elev.append(round(r * STOREY_HEIGHT_M, 3))
    return order, elev, any(r is None for r in ranks.values())


def assemble(
    project_id: str, results: list[PageResult], north: tuple[float, str] | None
) -> PlanGraph:
    """One plan from the per-page plans: the IFC model whole, or one 2D page per level."""
    ifc = [r for r in results if r.source == "ifc"]
    if ifc:
        plan = ifc[0].plan
        if north is not None:
            plan.north_angle_deg = fact(north[0], "derived", 0.8, note=north[1])
        return plan
    by_level: dict[str, PageResult] = {}
    for r in sorted(results, key=lambda r: (RANK[r.source], -len(r.plan.walls))):
        by_level.setdefault(r.level_name, r)
    if len(by_level) == 1:
        plan = next(iter(by_level.values())).plan
    else:
        names, elevations, guessed = _stack(list(by_level))
        parts = [_rename(by_level[nm].plan, f"L{i}", f"L{i}-") for i, nm in enumerate(names)]
        plan = parts[0].model_copy(deep=True)
        plan.levels = [
            Level(
                id=f"L{i}",
                name=nm,
                elevation_m=elevations[i],
                floor_to_floor_m=fact(
                    STOREY_HEIGHT_M, "default", 0.5, note="default/storey_height_m"
                ),
            )
            for i, nm in enumerate(names)
        ]
        for p in parts[1:]:
            plan.walls += p.walls
            plan.openings += p.openings
            plan.rooms += p.rooms
            plan.columns += p.columns
        plan.assumptions.append(
            Assumption(
                key="default/storey_height_m",
                value=STOREY_HEIGHT_M,
                reason="levels from separate plan sheets; no section or IFC gives their heights",
                stage="S2",
            )
        )
        if guessed:
            plan.assumptions.append(
                Assumption(
                    key="default/level_order",
                    value=names,
                    reason="some level names do not say which storey they are; those are "
                    "stacked above the others in sheet order",
                    stage="S2",
                    requires_review=True,
                )
            )
    plan.doc_transforms = [
        r.transform for r in results if r in by_level.values() or r.source == "ifc"
    ]
    plan.conflicts = [c for r in by_level.values() for c in r.conflicts]
    if north is not None:
        plan.north_angle_deg = fact(north[0], "derived", 0.8, note=north[1])
    plan.version = "extraction"
    plan.project = project_id
    return PlanGraph.model_validate(plan.model_dump())


def _north(pages: list[S2Page], chosen: set[str]) -> tuple[float, str] | None:
    """North from S1's north arrow on a chosen 2D source page (sheet up = plan +y there)."""
    for p in pages:
        if p.page.id in chosen and p.analysis is not None and p.analysis.north is not None:
            return (
                round(p.analysis.north.angle_deg.value % 360.0, 2),
                f"north arrow on page {p.page.id}",
            )
    return None


def build_s2(svc: Services) -> StageDef[S2In, S2Out]:
    def ocr_ref() -> list[ModelRef]:
        entry = svc.models.entry_for("ocr")
        while entry.impl is None and entry.fallback:
            entry = svc.registry.get(entry.fallback)
        return [entry.ref()]

    def run(inp: S2In, ctx: StageContext) -> S2Out:
        ex = Extractor(svc, ctx.store, inp.project_id)
        cands: list[tuple[S2Page, SourceKind]] = []
        for p in inp.pages:
            content = _json(ctx.store, p.page.content)
            kind = _source_kind(p.page, content)
            if kind is None:
                continue
            if kind != "ifc" and p.label != "floor_plan":
                continue
            cands.append((p, kind))
        if not cands:
            raise ArchRenderError(
                ErrorCode.PLAN_NO_PLAN_FOUND,
                "No floor plan was found among the project's pages.",
                "Upload a floor plan (IFC, DXF/DWG, PDF or a scan), or relabel a page as "
                "floor_plan on the Pages screen.",
            )
        best = min(RANK[k] for _, k in cands)
        skipped = []
        if best < RANK["raster"]:  # scans of sheets that are also there as vectors/models
            skipped = [p.page.id for p, k in cands if k == "raster"]
            cands = [(p, k) for p, k in cands if k != "raster"]
        results: list[PageResult] = []
        failures: list[dict[str, str]] = []
        for i, (p, kind) in enumerate(sorted(cands, key=lambda c: RANK[c[1]])):
            ctx.progress(0.05 + 0.8 * i / len(cands), f"S2 {kind} {p.page.id}")
            try:
                results.append(ex.page(p, kind))
            except ArchRenderError as e:
                failures.append({"page": p.page.id, "code": e.code.value, "message": e.message})
        if not results:
            f = failures[0]
            raise ArchRenderError(
                ErrorCode(f["code"]),
                f["message"],
                "Fix the page at Gate A (scale calibration) or upload a better source.",
            )
        chosen = {r.page_id for r in results}
        plan = assemble(inp.project_id, results, _north(inp.pages, chosen))
        issues = validate_plan(plan)
        plan = plan.model_copy(update={"issues": issues})
        ref = ctx.store.put_bytes(
            plan.model_dump_json(indent=1).encode(), "application/json", "plan.json"
        )
        extraction = {
            "sources": [
                {
                    "page": r.page_id,
                    "source": r.source,
                    "level": r.level_name,
                    "walls": len(r.plan.walls),
                    "openings": len(r.plan.openings),
                    "rooms": len(r.plan.rooms),
                    "scale": r.scale,
                    "unexplained_gaps": r.unexplained_gaps,
                    "notes": r.notes,
                }
                for r in results
            ],
            "failed": failures,
            "skipped_rasters": skipped,
            "candidates": [{"page": p.page.id, "source": k} for p, k in cands],
        }
        return S2Out(plan=plan, plan_json=ref, issues=issues, extraction=extraction)

    return StageDef("S2_plan", "2", S2Out, run, models=ocr_ref)


def s2_input(svc: Services, project_id: str) -> S2In:
    pages = []
    for page in page_refs(svc, project_id):
        row = svc.db.one("SELECT analysis_json FROM page_analysis WHERE page_id = ?", (page.id,))
        analysis = S1Out.model_validate_json(row["analysis_json"]) if row else None
        label = (
            _effective_label(svc, page.id, analysis)
            if analysis
            else ("other" if page.kind != "ifc" else "model")
        )
        pages.append(S2Page(page=page, label=label, analysis=analysis))
    return S2In(project_id=project_id, pages=pages)


def has_plan_source(svc: Services, project_id: str) -> bool:
    """Whether the project has a page S2 would read (an IFC model or a floor plan page)."""
    return any(
        p.page.kind == "ifc" or p.label == "floor_plan" for p in s2_input(svc, project_id).pages
    )


@dataclass
class CurrentPlan:
    """The plan version a run uses, with the extraction it descends from."""

    version_id: str
    number: int
    status: str
    approved_by: str | None
    extraction_version: str
    plan: PlanGraph
    plan_json: CasRef
    issues: list[ValidationIssue]
    extraction: dict[str, Any]

    @property
    def blocking(self) -> list[ValidationIssue]:
        return blocking(self.issues)


def version_plan(
    svc: Services, project_id: str, version_id: str, extraction: dict[str, Any] | None = None
) -> CurrentPlan:
    row = versions.get_row(svc, project_id, version_id)
    plan = versions.load(svc, project_id, version_id)
    return CurrentPlan(
        version_id=row["id"],
        number=int(row["number"]),
        status=row["status"],
        approved_by=row["approved_by"],
        extraction_version=row["root_id"],
        plan=plan,
        plan_json=CasRef.model_validate_json(row["plan_ref_json"]),
        issues=plan.issues,
        extraction=extraction if extraction is not None else json.loads(row["extraction_json"]),
    )


def current_plan(
    svc: Services, ctx: StageContext, engine: StageEngine | None = None
) -> CurrentPlan:
    """S2 for the project's current pages (a cache hit when they did not change), stored as a
    draft version, and the version to use: the approved one of that extraction if a reviewer
    approved one at Gate A, otherwise its latest edit (or the extraction itself)."""
    if any(
        svc.db.one("SELECT 1 FROM page_analysis WHERE page_id = ?", (p.id,)) is None
        for p in page_refs(svc, ctx.project_id)
    ):
        run_understanding(svc, ctx)  # a run started before S1 finished: S1 first (cached pages hit)
    out = (engine or StageEngine(svc.db)).run(build_s2(svc), s2_input(svc, ctx.project_id), ctx)
    root = versions.record_extraction(svc, ctx.project_id, out.plan, out.extraction)
    row = versions.working(svc, ctx.project_id, root)
    return version_plan(svc, ctx.project_id, row["id"], out.extraction)


def run_plan_job(svc: Services, ctx: StageContext) -> dict[str, Any]:
    """The PLAN job (after S1): extract and store the draft for Gate A."""
    cur = current_plan(svc, ctx)
    return {
        "plan_version": cur.version_id,
        "number": cur.number,
        "status": cur.status,
        "extraction_version": cur.extraction_version,
        "source": cur.plan.source,
        "walls": len(cur.plan.walls),
        "openings": len(cur.plan.openings),
        "rooms": len(cur.plan.rooms),
        "issues": len(cur.issues),
        "blocking": len(cur.blocking),
        "conflicts": sum(1 for c in cur.plan.conflicts if c.resolution is None),
        "sources": [
            {k: s[k] for k in ("page", "source", "level")} for s in cur.extraction["sources"]
        ],
        "failed": cur.extraction["failed"],
    }
