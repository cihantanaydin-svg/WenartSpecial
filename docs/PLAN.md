# ArchRender: Delivery Plan

Status: Phase 0 draft, awaiting approval · 2026-09-30

Every phase ends with: tests + `make eval` run, `docs/PROGRESS.md` updated (done / how verified /
UNVERIFIED-ON-GPU / next), `CLAUDE.md` refreshed, and conventional commits pushed. The deploy
skeleton (Dockerfile, entrypoint, `deploy.py --dry-run`) stays green from Phase 1 onward.

<!-- BUDGETS -->

## Phases

### Phase 0: Research & design (this document set) → **STOP for approval**
Deliverables: `ARCHITECTURE.md`, `MODEL_SELECTION.md` (one ADR per model role with license
evidence), `DECISIONS.md` (system ADRs), `RISKS.md`, `PLAN.md`, `CLAUDE.md`, `PROGRESS.md`.
Acceptance: owner approval, and answers to the open questions (end of this file) or acceptance of
their defaults.

### Phase 1: Walking skeleton (end-to-end, thin, real plumbing)
Scope: repo scaffold (uv, ruff, mypy strict on `core`, pre-commit, pytest, Hypothesis, GitHub
Actions CPU job). Core schemas + JSON Schema export. Per-project CAS. SQLite job queue with leases +
events + SSE. Auth (API keys, roles, bootstrap token). Chunked resumable uploads. Model registry
schema + LicenseGate + ModelManager (mock backends). Pipeline engine with cache keys, gates and
fan-out. **Mock plan** (a fixed rectangle room with a door and a window, generated from a DXF fixture
via a minimal ezdxf path). SceneCompiler for a box room (manifold3d). Blender build + Cycles CPU
render at tiny resolution with passes. Mock refine (deterministic image op). QA metric framework with
the relative-to-base metrics. Bundle + download. CLI and Python client. Minimal React UI (login,
project, upload, run, SSE progress, gallery, download). Dockerfile + entrypoint + supervisor config.
`download_models.py` (license-gated, SHA256, resumable) against a registry containing only
mocks/tiny models. `deploy.py up --dry-run` / `down --dry-run`. License audit script v0. Litestream
config.

Acceptance:
- `make test` green on CPU. `make e2e` runs upload → mock plan → Blender box room → mock refine →
  QA → bundle through **API, CLI and UI (Playwright)**.
- Re-running with no changes → 100% cache hits. Changing the floor material → only S5+ re-run.
- Route-auth test passes (every route authenticated). No request > 55 s.
- `docker build` succeeds if a daemon can be started here; otherwise the build is verified in CI.
  `deploy.py up --dry-run` prints valid payloads.
- UNVERIFIED-ON-GPU: OptiX/CUDA device selection, vLLM process.

### Phase 2: Ingest & understanding (S0, S1)
Scope: magic-byte detection, safe unzip, EXIF/sRGB/HEIC, PDF split (vector + ≥300 DPI raster),
DWG→DXF converter wrapper (licensed tool, separate process), Office parsing, sandboxed parser
runner. Page classification (heuristics + VLM + calibration), tiled OCR with coordinate mapping,
title block, north arrow, schedule parsing (PDF tables, XLSX, DOCX) and linking.
Acceptance:
- Fuzz/limit tests: zip bomb, path traversal, symlink, oversized image, malformed PDF all rejected
  with coded errors.
- Classification accuracy on the synthetic + golden set ≥ 0.95 macro-F1 (with the real VLM on GPU;
  mocks on CPU verify plumbing only). Low-confidence items land in the review queue.
- OCR: dimension strings on synthetic sheets read with ≥ 0.98 exact-match at 300 DPI (GPU run).
- Schedules linked to plan tags on all golden projects.

### Phase 3: Plan extraction + Gate A (S2, S3) + synthetic plan generator
Scope: synthetic plan generator (raster with noise/skew/stamps/hatching/fonts/languages/phone
distortion; vector PDF; DXF; exact GT). Extractors for IFC, DXF, vector PDF and raster (CV baseline).
Shared PlanBuilder, scale estimators + reconciliation, dimension parser, registration, heights,
validators. Gate A SVG editor. Training-example capture. Plan-segmentation training script
(runs on the pod; trains on synthetic + firm archive; NC datasets forbidden).
Acceptance (`make eval` plan table):
- Vector sources (DXF/PDF): wall F1 ≥ 0.98, opening F1 ≥ 0.95, scale error ≤ 1%.
- Clean raster: wall F1 ≥ 0.92, opening F1 ≥ 0.88. Noisy raster: measured, routed to Gate A.
- Scale disagreement > 1.5% always flagged (property test).
- Validators: 100% detection of injected plan defects (open room, opening outside host, overlapping
  rooms, area-label mismatch > 3%, unreachable room).
- Gate A edits round-trip into an immutable PlanVersion; corrections stored as training examples.

### Phase 4: Scene, cameras, base render (S5 geometry/lighting, S6, S7)
Scope: full SceneCompiler (walls with openings, floors, ceilings, skirting, parametric doors and
windows, section cuts with poché), real-world UVs, lighting (pvlib sun, sky/HDRI, portals, blackbody
fixtures, auto-exposure), Blender build/export (GLB, .blend, manifest). Camera proposal + scoring +
cutaway + section-perspective + 360°. Cycles with OptiX/CUDA selection, all passes, camera JSON,
structural masks. GLB viewer in the UI.
Acceptance:
- Property tests: random valid plans → every mesh watertight; openings inside hosts; no
  self-intersections.
- Pass consistency: rendered depth at probe pixels equals the analytic distance (±1 mm at tiny res);
  camera JSON reprojects known 3D corners to line-art pixels within 1 px.
- Verticals in base renders ≤ 0.1° (two-point perspective).
- The sun direction test: the shadow of a known box falls where pvlib predicts (angle ±1°).
- GPU: 3840×2160 base render timings per profile recorded (UNVERIFIED-ON-GPU until smoke test).

### Phase 5: Brief, assets, layout + Gates B/C (S4, S5 furniture, S6 UI)
Scope: DesignBrief extraction with provenance and contradiction detection. Palette k-means in
CIELAB. Paint-code resolution (flagged approximations). Asset library schema + import validator +
CC0 seed (Poly Haven, ambientCG). Embeddings + VLM re-rank material matching with derive-and-flag.
Material previews. Furniture: furniture-plan ground truth path; zoning rules + constraint DSL + solver
with clearances. Gate B board, Gate C cameras + layout UI.
Acceptance:
- Brief extraction on golden projects: every must-have from the text brief captured (checked
  against hand-written expected briefs); contradictions detected on seeded conflict fixtures.
- Layout solver: 0 violations of hard constraints (collision, door swing, 0.9 m walkway, window
  access) across 500 random rooms; solver returns "no valid layout" with reasons when infeasible.
- Asset import validator rejects mis-scaled / wrong-pivot / unlicensed fixtures.

### Phase 6: Refinement + QA + fault injection (S8, S9)
Scope: Refiner implementations (faithful: per-pixel strength map, depth/edge conditioning, style refs,
tiled 4K, seam checks). Concept generator. Best-of-N, IQA ranking, cross-view harmonisation. All
QA metrics, VLM judge + second-family tie-breaker, retry policy, fallback, DeliverableGuard.
Fault-injection harness (scene-level + image-level corruptions) and eval report.
Acceptance:
- Fault injection: **≥ 95% detection, ≤ 5% false alarms** on clean images, confusion matrix per
  corruption type (GPU run on the pod; the CPU run with mocks verifies plumbing only and is labelled
  as such).
- No geometry-failing image can reach a bundle (guard test with forced failures).
- Quantized variants enabled only after a measured A/B on the QA metrics.

### Phase 7: Iteration + deliverables (S10)
Scope: Gate D gallery (QA badges, compare slider, "why flagged"). Bundle (16-bit PNG, JPEG, EXR,
panorama, GLB, .blend, manifests). QA report HTML/PDF (plan overlay, assumptions, conflicts,
provenance, models, seeds). Change requests (NL → typed JSON-Patch diff → confirm → deterministic
apply → minimal re-run). Version history with diffs.
Acceptance:
- "warmer light", "terrazzo floor", "sofa against the window wall" produce the expected typed diffs
  on golden projects, and only the affected stages re-run (asserted via cache-hit accounting).
- The report contains every assumption and conflict in the run (cross-checked against the DB).

### Phase 8: Deployment hardening + final report
Scope: GHCR build on tag with digest recording. Full entrypoint (GPU/driver/CUDA/disk/RAM checks,
hot staging, self-test, `/readyz`). `deploy.py up/down` complete with datacenter selection, volume
sizing, template, pod, polling, `/readyz` wait. `runpodctl` equivalents + console checklist.
Bootstrap fallback script. Idle watchdog. `scripts/smoke_test.py`. DEPLOY.md (prerequisites,
one-command deploy, cost notes, troubleshooting, upgrade, backup). License audit final +
THIRD_PARTY_LICENSES.md. Final report.
Acceptance: `deploy.py up --dry-run` verified; smoke test ready (and run on a pod if credentials are
provided); license audit clean; every Definition-of-Done box ticked or its deviation documented.

## Golden projects (made from scratch, no copyrighted plans)
1. **G1 "Flat"**: 2-bedroom apartment, Manhattan walls. Inputs: CAD-style vector PDF (1:50) + DXF +
   finish schedule XLSX + text brief + 4 mood-board images (CC0 or generated from our own renders).
2. **G2 "Loft"**: open plan with a non-Manhattan wall, an arc wall, a double-height living area
   (flagged), and a kitchen island. Inputs: raster scan (300 DPI with noise + stamps) + phone photo of
   the printed plan + RCP + door/window schedule PDF + DOCX brief in English + second language.
3. **G3 "Office"**: small office suite, imperial units (feet-inches dimensions), IFC export + furniture
   plan, site photo for window backplates.

## Evaluation table (`make eval`)
Columns: profile · plan (wall F1, opening F1, scale err) per source type · render QA (pass rate,
retries, fallbacks) · fault injection (detection %, false-alarm %, per-type) · timings per stage ·
VRAM peaks. CI runs the CPU subset; the pod smoke test runs the GPU subset.

## Open questions for the owner (defaults apply if unanswered)
<!-- QUESTIONS -->
