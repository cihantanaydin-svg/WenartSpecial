# ArchRender: Progress

| Phase | Status | How verified | UNVERIFIED-ON-GPU | Next |
|---|---|---|---|---|
| 0 Research & design | **Done, approved by owner 2026-09-30** (answers recorded in PLAN.md) | Five parallel research passes (models ×3, RunPod, Blender/stack/licences). Evidence from LICENSE files on GitHub, RunPod's docs/OpenAPI source repo, diffusers source + wheels, the vLLM recipes repo, PyPI and Docker Hub metadata, and web-search snippets of HF cards (marked). | Everything GPU-related: VRAM/time budgets in PLAN.md are estimates | Phase 1 walking skeleton |
| 1 Walking skeleton | **Done 2026-09-30** | `make lint typecheck test e2e eval`, strict mode (no skips); CPU image built, booted and smoke-tested; deploy payloads validated against RunPod's OpenAPI v2 schemas; workflows pass actionlint. Details below | OptiX/CUDA device selection, GPU image build (CUDA base + Blender tarball + torch cu130), vLLM, live RunPod API. List below | Phase 2 ingest & understanding |
| 2 Ingest & understanding | **Done 2026-09-30 (CPU part)** | strict `make test e2e eval` (274 tests), golden projects G1/G2 through the pipeline, mutation fuzzing of intake, CPU image rebuilt and smoke-tested with S1 inside. Details below | VLM classification + combiner, PaddleOCR-VL, the ≥ 0.95 F1 / ≥ 0.98 OCR acceptance runs, VLM A/B, vLLM serving/sleep. List below | Phase 3 plan extraction + Gate A |
| 3 Plan extraction + Gate A | **Done 2026-10-01 (CPU part)** | strict `make test e2e` (637 tests), `make eval` plan table per source with real OCR, golden G1–G3 through S0–S2, Playwright Gate A editor. Details below | The VLM assist with the real VLM, plan-segmentation training and inference, the Phase-2 VLM items. List below | Phase 4 scene, cameras, base render |
| 4 Scene, cameras, base render | Not started | — | — | — |
| 5 Brief, assets, layout + Gates B/C | Not started | — | — | — |
| 6 Refinement + QA + fault injection | Not started | — | — | — |
| 7 Iteration + deliverables | Not started | — | — | — |
| 8 Deployment hardening + final report | Not started | — | — | — |

## Phase 0 deliverables
- `docs/ARCHITECTURE.md`: system design, stages S0–S10, data model, QA metrics, API, security,
  deployment.
- `docs/DECISIONS.md`: system ADRs S01–S19 (S19 = on-demand VLM coordinate assist, per owner comment).
- `docs/MODEL_SELECTION.md`: model ADRs M01–M14 with licence evidence, plus the tools/libraries
  licence baseline.
- `docs/RISKS.md`: risk register (quality, licensing, infra, delivery).
- `docs/PLAN.md`: phases with acceptance tests, VRAM/disk/time budgets per profile, open questions.
- `docs/research/RUNPOD.md`: RunPod API v2 facts with sources.

## Limitations of the Phase 0 evidence
- `huggingface.co`, `docs.runpod.io`, `rest.runpod.io`, `api.runpod.io` and `download.blender.org`
  are blocked by this dev container's egress policy.
- Model revisions/SHA256 and the live Hub licence tags are therefore pinned by
  `python -m archrender.ops.models pin` on a machine with Hub access, and re-checked at download.
- The Blender 5.2.2 tarball SHA256 is taken from a third-party mirror. CI compares it with the
  official `.sha256` before first use.
- No GPU here. (Phase 1 found that a Docker daemon can be started in this container; Docker Hub and
  GitHub release downloads work.)

## Phase 1: walking skeleton (done 2026-09-30)

Upload → intake → mock plan → brief defaults → scene (manifold3d) → cameras → **real Blender 5.2.2
Cycles render with passes** → mock refine → relative QA → Gate D → bundle, through the API, the CLI
and the UI, plus the deploy skeleton.

### How it was verified
| Check | Result |
|---|---|
| `make lint` (ruff check + format), `make typecheck` (mypy strict on core/pipeline/models), `scripts/check_no_stubs.py` | clean |
| `make test` with `ARCHRENDER_TEST_STRICT=1` (a missing Blender runtime or UI build fails instead of skipping) | **83 passed**, real Blender via the official `bpy` 5.2.2 wheel (hash-locked) |
| `make e2e` (live uvicorn + worker; API, CLI, Playwright/Chromium UI) | **3 passed** |
| README quickstart | executed as written: dev server + worker, `bootstrap`, `project create`, `upload`, `run --wait`, `gate approve`, `download` |
| Render correctness | Cycles Z pass is planar depth and matches our camera model; reprojected room corners lie on the deterministic line art within 2 px |
| QA sensitivity on the real render | the unmodified render passes every deterministic check; a 2.5 % shift and a window painted over with wall colour both fail the edge-F check (`test_blender_render.py`). Systematic fault injection is Phase 6 |
| Security | every route requires auth (route-enumeration test), one-shot bootstrap, cookie + CSRF, roles and project isolation, 55 s request timeout, RVT/SKP rejected with coded errors, chunk SHA-256 checked |
| Cache | identical re-run: 8/8 stages cached. Floor material change (`--material floor=stone_porcelain_grey`): only S5 scene, S7 render and S8/S9 re-run; S2 plan, S4 brief and S6 cameras stay cached |
| Container | `make docker-build-local` (Ubuntu 24.04 + bpy wheel) builds; `make docker-smoke` boots it, waits for `/readyz` (self-test passed), runs `scripts/smoke_test.py` end to end: **passed**. The licence audit also runs inside the image. Litestream restore after replacing the container was verified |
| `deploy.py` | payloads validated against RunPod's OpenAPI v2 request schemas; up/down/status flows run against a fake API (placement loop, 429/5xx retries, secrets never in templates); `up --dry-run` and `down --dry-run` print plans with secrets masked |
| Model tooling | `ops.models pin/download` against a fake Hub: SHA-256 verification, Hub-licence cross-check, gated-access error naming the HF page, unpinned models refused |
| Licences | `make license-audit`: clean; `THIRD_PARTY_LICENSES.md` generated (bpy is GPL and runs only as a separate process) |
| CI / release workflows | `.github/workflows/ci.yml` and `release.yml` pass actionlint 1.7.12; actions pinned by commit SHA. The first GitHub run happens on this push |

### `make eval` (CPU subset, cpu_test profile, 320×180 @ 16 spp, 2 views)
| metric | value |
|---|---|
| plan (wall F1, opening F1, scale err) | not measured: the Phase-1 plan is a mock; extractors + metrics arrive in Phase 3 |
| render QA (mock estimators: plumbing only) | pass 100% (2/2 refined), 2.0 attempts per view, 0 hard composites, 0 fallbacks, 16/16 checks (4 computed with mocks) |
| cache | identical re-run 8/8 cached; floor-material change re-ran S5, S7, S8/S9 only |
| fault injection | not measured: the harness arrives in Phase 6 |
| timings, first run (s) | S2 0.01 · S4 0.00 · S5 1.66 · S6 0.03 · S7 7.42 (2 views) · S8/S9 0.51 |
| VRAM peaks | not recorded: no GPU model stages in this build |

### UNVERIFIED-ON-GPU after Phase 1
- Cycles OptiX → CUDA → CPU device selection, and CUDA/OptiX kernel-cache persistence on the volume.
  Only the CPU path has run.
- The release image: CUDA 13.0 base, the Blender 5.2.2 **tarball** ("binary" mode) and the torch
  cu130 extra. The release workflow builds it, checks the tarball SHA-256 against
  download.blender.org, and smoke-tests it on a CPU runner. It has not run yet (download.blender.org
  is blocked here).
- `preflight` GPU/driver (≥ 580)/RAM/disk checks against a real `nvidia-smi`.
- vLLM (no vLLM-served role exists yet; `INSTALL_VLLM=0`).
- Live RunPod API calls by `deploy.py up/down/status` (only schemas and a fake API so far).
- ModelManager VRAM budgeting with real weights (every role is still a mock).
- All VRAM/time budgets in PLAN.md remain estimates.

### Deviations from the Phase-1 plan (recorded, not hidden)
- **Mock plan does not read the DXF.** PLAN.md said the mock room would be "generated from a DXF
  fixture via a minimal ezdxf path". The mock is a fixed 5 × 4 m room keyed by the uploaded
  documents' hashes, with every fact marked `method="mock"`. A throw-away bounding-box reader would be
  replaced in Phase 3 by the real DXF extractor and its ground-truth metrics, so it would add no
  verification value.
- **Hard composite** uses a feathered alpha blend, not Poisson blending (ADR-S07 amendment).
- **Model pins:** `configs/models.lock.yaml` does not exist yet, because huggingface.co is blocked
  here. `python -m archrender.ops.models pin` writes it on a machine with Hub access; `download`
  refuses unpinned models, and Phase-1 builds download nothing (every role is a mock).

### Deferred, with the phase that owns them
- LibreDWG in the image and the hash-locked vLLM environment: Phase 2 (DWG conversion, VLM roles).
- Sun position from pvlib (fixed SW sun is an assumption today): Phase 4.
- `bootstrap_on_pytorch_template.sh` fallback and the idle watchdog: Phase 8.
- UI for Gate B edits (the API and CLI accept material overrides now): Phase 5.

## Phase 2: ingest & understanding, S0/S1 (done 2026-09-30, CPU part)

Every upload is detected by magic bytes and parsed in a resource-limited child process (ADR-S20)
before it enters the store. S1 then analyses every page in a cached stage (UNDERSTAND job, queued
after each intake):
- words from the PDF text layer, DXF text entities or tiled Turkish/English OCR;
- title block and scale, north arrow, door/window/room tags (text layer, and bubbles on scans);
- page class with a confidence;
- schedules (XLSX/DOCX tables, PDF and scanned tables), linked to the tags on plan pages.

Low-confidence or unfamiliar pages, unlinked schedule rows and incomplete OCR go to the review
queue. The UI's Pages panel follows the analysis job and shows each page's class, confidence, scale,
north angle and tag count, with a menu to correct the class. The CLI has `pages`, `schedules` and
`review`.

### Acceptance (docs/PLAN.md, Phase 2)
| item | result |
|---|---|
| Fuzz/limit tests: zip bomb, path traversal, symlink, oversized image, malformed PDF rejected with coded errors | **met.** Targeted cases in `test_intake.py`; mutation fuzzing in `test_intake_fuzz.py` (8 formats × 6 truncated/bit-flipped/spliced files in CI). A one-off run of 320 mutants: all ingested or rejected with an `INGEST_*` code and a fix hint, 0 crashes |
| Classification ≥ 0.95 macro-F1 (with the real VLM) | heuristic model alone on a held-out synthetic set (88 pages): macro-F1 **1.000**. This set is synthetic, drawn from the same generator as training, so it says little about real documents; those go to review when unfamiliar (below). With the VLM: **UNVERIFIED-ON-GPU** |
| Low-confidence items land in the review queue | **met** (`test_s1_pipeline.py`, `test_classify.py`); held-out review rate 4.5 %, 0 misclassified pages outside review |
| OCR: dimension strings ≥ 0.98 exact at 300 DPI (GPU run) | not met by the CPU fallback: Tesseract reads ≥ 0.95 on a clean 300-DPI scan (test) and 75 % over mixed scans (200/300 DPI, clean to noisy). PaddleOCR-VL run: **UNVERIFIED-ON-GPU** |
| Schedules linked to plan tags on all golden projects | **met**: G1 18/18 rows, G2 11/11 rows (tags read inside bubbles on a noisy 300-DPI scan); every G1/G2 page gets its expected class (`test_golden_s1.py`, `make eval`) |
| VLM A/B (Qwen3.6 vs Qwen3.8) | **UNVERIFIED-ON-GPU**: client, prompt, JSON schema and combiner exist and are tested against a fake OpenAI server; the A/B needs the pod |

### How it was verified
| Check | Result |
|---|---|
| `make lint typecheck`, no-stubs check, schema / deploy-manifest / licence freshness | clean |
| `pytest` with `ARCHRENDER_TEST_STRICT=1` (Tesseract tur+eng, libheif, LibreDWG and Blender must be present) | **274 passed**, incl. e2e (API, CLI, Playwright UI with the Pages panel) |
| Golden projects through upload → intake → S1 | G1 "Daire" (vector PDF plan, the same plan as DXF in cm, XLSX finish schedule, door/window schedule PDF, DOCX brief, mood board, 2 photos) and G2 "Loft" (noisy scanned plan, ceiling plan, schedule PDF, TR/EN brief): all classes right, all schedule rows linked |
| Container | CPU image rebuilt; `make docker-smoke` now also uploads a raster sheet and waits for S1 in the container (OCR + classifier): **passed** |
| Uploads without an extension | every format ingests from a `<upload>.bin` staging path (found: every uploaded XLSX had failed; fixed) |

### `make eval` S1 section (held-out synthetic corpus, seed 3, 8 per class = 88 pages)
| metric | value |
|---|---|
| sources | 43 vector PDF · 37 scans (200/300 DPI, clean/medium/noisy) · 8 photos |
| page classification (heuristic model, Tesseract text) | macro-F1 1.000 · review rate 4.5 % (2 pages outside the training range) · errors not sent to review 0 |
| title-block fields (vector + scans) | 325/344 (94 %) |
| scale from title block | 48/48 |
| north arrow (vector) | 12 measured, 0 missing, mean \|err\| 0.51°, max 0.65° |
| OCR on scans (Tesseract 5.3.4) | dimension strings 15/20 (75 %) · title-block text 318/332 (96 %) · room names 53/67 exact (79 %), Turkish character accuracy 0.82 |
| bubble tags on scans | 31/37 (84 %) over all scans, 0 false. Per quality on 6 plans: 75/77 (97 %) at 300 DPI clean, medium and noisy; 68, 66 and 47 of 77 at 200 DPI |
| schedule ↔ plan tags | 100/100 rows on 8 synthetic projects; golden G1 18/18, G2 11/11 |
| timings | Tesseract ~12–15 s per A3 page at 300 DPI (both reading directions) on this 4-core container; golden G1 20 s, G2 42 s (S0 + S1) |

`make eval` uses 3 pages per class by default (33 pages) to keep CI time down; the table above is
`evaluate(seed=3, per_class=8)`.

### UNVERIFIED-ON-GPU after Phase 2
- VLM page classification (Qwen3.6-27B-FP8 via vLLM, `models/impls/vllm_vlm.py`): request/response
  schema, probe, sleep/wake and the geometric-mean combiner are tested against a fake server only.
  A combiner calibrated on VLM answers for real pages is not built.
- vLLM serving on the pod: `deploy/vllm/requirements.lock` (vLLM 0.30.0, torch 2.13, CUDA 13) is
  licence-pre-audited from PyPI metadata; the supervisor starts vLLM only when the weights are
  installed. Not run.
- PaddleOCR-VL-1.6 (primary OCR) is not implemented; the role falls back to Tesseract with a
  recorded degradation.
- The acceptance numbers that need the GPU: classification ≥ 0.95 macro-F1 with the VLM, OCR
  ≥ 0.98 on dimension strings, the Qwen3.6 vs Qwen3.8 A/B.
- Model pins: `configs/models.lock.yaml` still needs `python -m archrender.ops.models pin` on a
  machine with Hub access (huggingface.co is blocked here).

### Deviations from the Phase-2 plan (recorded, not hidden)
- RapidOCR → **Tesseract 5** as the CPU OCR (ModelScope/HF unreachable here; DECISIONS.md).
- Docling → python-docx/openpyxl/python-pptx for Office files (ADR-S21).
- Classifier confidence: temperature bounded at T ≥ 1, and pages outside the training range go to
  review (the synthetic calibration set is separable; DECISIONS.md, Phase-2 amendments).
- G2's non-Manhattan/arc walls, the phone photo of a printed plan, and G3 (IFC office) come with the
  Phase-3 plan generator and extractors.

### Found and fixed on the way
- Tesseract's sparse-text mode spent > 120 s on the speckle of one noisy scan tile: noisy tiles
  (σ > 3 grey levels) are now despeckled, calls time out after 120 s, and an unreadable tile is
  reported on the page and in the review queue instead of failing it.
- Bubble tags on noisy scans: 0/77 → 75/77 at 300 DPI (local ink threshold, both ring boundaries,
  lettering isolated from door swings, voting with a digit look-alike fallback).
- ZIP/OOXML type detection read the ZIP directory in the worker; it now runs in the sandbox (found
  by the fuzz test).
- Uploaded workbooks always failed (openpyxl requires an `.xlsx` filename; uploads are staged as
  `.bin`).

## Phase 3: plan extraction (S2) and Gate A (S3) (done 2026-10-01, CPU part)

After S1, a PLAN job extracts the project's plan from its best sources: per level an IFC model, else
a DXF, else a vector PDF, else a scan or phone photo (rectified in S1). Every source goes through its
extractor into the shared PlanBuilder (walls from face pairs, arc walls, junctions, openings, rooms
as holes of the wall body). The scale of PDF and raster pages is measured (stated scale × page
resolution, dimension strings read along their lines, door swings) and fused; disagreements are
Gate A conflicts. Several sheets become levels in storey order (ADR-S23). The RCP gives ceiling
heights after registration onto the floor plan, door/window tags and schedules give opening sizes
(ADR-S25). On scans, the VLM assist measures what the CV baseline missed, only where a trigger fires
and only from drawing evidence (ADR-S19/S24). The result is a draft plan version.

Gate A (ADR-S22): an SVG editor over the source page (UI `#/projects/<id>/plan`), the API
(`/api/v1/projects/{id}/plans…`) and the CLI (`archrender plan …`) edit the plan as JSON Patches
into new versions, resolve conflicts, calibrate the scale from two points, confirm VLM-assisted
elements, decide suggestions and approve. Runs render the approved version (or the latest edit)
and pin it; corrections are kept as training examples that are never used without the owner.

### Acceptance (docs/PLAN.md, Phase 3; `make eval` plan table, 6 sheets per source, seed 11)
| item | result |
|---|---|
| Vector DXF / PDF: wall F1 ≥ 0.98, opening F1 ≥ 0.95, scale error ≤ 1 % | **met.** DXF walls 1.000, openings 0.983, scale 0.00 %; vector PDF 1.000 / 1.000 / 0.00 % (also every test seed in `test_plan_extract.py` and golden G1) |
| Clean raster (300 DPI, real Tesseract OCR): wall F1 ≥ 0.92, opening F1 ≥ 0.88 | **met**: walls 1.000, openings 0.969, rooms 1.000, scale 0.00 % (6 sheets, all four wall styles) |
| Noisy raster: measured, routed to Gate A | noisy 200-DPI scans: walls 0.996, openings 0.726, rooms 0.861; phone photos: walls 0.991, openings 0.913, rooms 0.867, scale ≤ 1.7 %. Every raster plan needs a person at Gate A (mandatory gate) |
| Scale disagreement > 1.5 % always flagged | **met** (property test `test_plan_scale.py`); a flagged page is a blocking `PLAN_SCALE_CONFLICT` until the user picks an estimate or calibrates |
| Validators detect 100 % of injected defects | **met**: 150/150 (open room, opening outside its wall, overlapping rooms, area label off > 3 %, unreachable room; 30 plans of 5 variants), 0 issues on the clean plans |
| Gate A edits round-trip into immutable versions; corrections stored as training examples | **met** (`test_plan_versions.py`, `test_api.py`, `test_pipeline.py`, Playwright `test_ui_plan_editor_gate_a`): edits → new drafts with `user` provenance, the source version unchanged, approval refused on blocking issues, run pinned to the approved version, `training_examples` rows with `training_use_allowed = 0` |
| Assist fires only on its triggers | **met**: no trigger → no call (asserted); triggers on 9/12 eval scans |
| Hints without evidence are never inserted | **met**: property test with adversarial hints anywhere on the sheet (`test_plan_assist.py`), and targeted ones on paper, text, beside walls, in solid wall |
| Accepted assists match the ground truth in ≥ 95 % | **met on the measured set**: 2/2 accepted assists on ground-truth elements (ground-truth hints with 6 px noise standing in for the VLM; rooms F1 0.930 → 0.977); in the tests the stamp case (a junction lost under an approval stamp) is fixed (rooms 0.667 → 1.000). The VLM's own pointing: **UNVERIFIED-ON-GPU** |
| Assist and acceptance rates in `make eval` | reported: 29 calls, 32 hints, 2 accepted (6 %: most hints point at walls the extractor already has) |
| Golden projects through the pipeline | G1 (DXF wins over the vector PDF; tags + schedule sizes applied, no conflicts), G2 (noisy scan with stamp + phone photo + RCP: walls 1.00, rooms 1.00, scale 0.0002 %, RCP registered at 3 mm RMS, double-height living room found, schedule widths override drawn widths as conflicts), G3 (IFC office, 2 storeys; the imperial PDF reads 1/8" = 1'-0" and feet-inch dimensions to < 1 %) — `test_golden_s2.py` |

### How it was verified
| Check | Result |
|---|---|
| `make lint typecheck`, schema freshness | clean |
| `pytest` with `ARCHRENDER_TEST_STRICT=1`, incl. e2e (API, CLI, Playwright: the Gate A editor renames a room, adds a wall, sees the blocking issues, deletes it, saves, approves) | **637 passed**, 1 skipped (the PyTorch training smoke: no PyTorch in the CPU environment) |
| Pipeline with real extracted plans | uploads now use a synthetic floor plan instead of the one-line DXF in the pipeline, API, CLI, UI and container smoke tests; Gate A approval of an edited version re-pins the run (`test_pipeline.py`) |
| Scene from extracted plans | extracted plans of all variants (Manhattan, rotated, skewed, arc) compile into watertight meshes (the junction partition was fixed for this, ADR-S04 amendment) |

### `make eval` plan table (`plan.evaluate`, seed 11, 6 sheets per source; real intake and S1)
| metric | value |
|---|---|
| DXF (6) | walls 1.000 · openings 0.983 · rooms 1.000 · scale error 0.00 % · **meets targets** |
| vector PDF (6) | 1.000 · 1.000 · 1.000 · 0.00 % · **meets targets** |
| clean scans, 300 DPI (6, Tesseract) | walls 1.000 · openings 0.969 · rooms 1.000 · 0.00 % · **meets targets** (420 s, OCR dominated) |
| noisy scans, 200 DPI (6) | walls 0.996 · openings 0.726 · rooms 0.861 · 0.00 % |
| phone photos (3) | walls 0.991 · openings 0.913 · rooms 0.867 · scale ≤ 1.69 % |
| validators | 150/150 injected defects detected |
| assist (ground-truth hints on the 12 scans) | triggered on 9 · 29 calls · 32 hints · 2 accepted, both on ground-truth elements · rooms F1 0.930 → 0.977 |

`make eval` runs 3 sheets per source by default (CI time); the numbers above are
`evaluate(per_source=6)`. A smaller run (2 per source) found one noisy scan whose stated scale OCR
missed and whose dimension strings did not agree: its scale came from the door swings alone, 4.3 %
off, with nothing telling the reviewer. A scale that rests on one weak estimate (> 2 %) is now a
blocking Gate A question (confirm or calibrate).

### UNVERIFIED-ON-GPU after Phase 3
- The VLM assist with the real VLM: the `locate_elements` request (schema, full-resolution tile,
  out-of-tile answers refused) is tested against a fake server; how well Qwen points at walls and
  openings, and so the acceptance rate on real misses, is not measured.
- Plan segmentation: the training run (`plan.seg_train`, U-Net on synthetic tiles) and the
  TorchScript inference (`models/impls/plan_seg.py`). No profile maps `plan_segmenter` until a
  checkpoint passes the promotion rule; the hook itself is tested with a ground-truth mask.
- The VLM-dependent S1 items listed under Phase 2 still apply (page classes and OCR feed S2).

### Deviations from the Phase-3 plan (recorded, not hidden)
- Plan segmentation: plain-PyTorch U-Net instead of segmentation-models-pytorch + DINOv2 (ADR-S26).
- Gate A editor: no split/merge-wall tool and no swing editor yet (walls are drawn, moved, deleted;
  openings get type, width and offset); scale bars are not used as a scale estimator; heights from
  sections are not read (RCP labels and defaults only).
- The assist runs on raster pages only (ADR-S24); vector sources go to Gate A as drawn.

### Found and fixed on the way
- Room labels on scans: word heights assumed vertical text for any non-zero angle, so a 1° skew
  joined "m²" to names and lost every area label; lines are now grouped in the text's frame.
- Several levels were stacked in source order, not storey order (level names now parsed).
- The scene partition rejected T/X junctions of extracted plans ("wall footprint split").
- The validators found a wrong scale choice by themselves: choosing the 2 %-off estimate of a
  scale conflict makes every area label disagree by 4 %, and approval is refused.

### Known limitations (measured, open)
- Noisy 200-DPI scans: openings F1 0.73 (gaps measured 3–17 cm wide on noisy poché; schedule
  widths override them as conflicts when a schedule exists), rooms merge where a wall junction is
  lost (the assist repairs those it is pointed at).
- A greyscale stamp across a wall can still break it in the CV body; the validators flag it and the
  assist or the editor fixes it.
- RCP registration needs ≥ 70 % of the RCP's walls to match within 5 cm; a heavily simplified RCP
  is reported and its heights are not used.
