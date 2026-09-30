# ArchRender: Delivery Plan

Status: Phase 0 draft, awaiting approval · 2026-09-30

Every phase ends with: tests + `make eval` run, `docs/PROGRESS.md` updated (done / how verified /
UNVERIFIED-ON-GPU / next), `CLAUDE.md` refreshed, and conventional commits pushed. The deploy
skeleton (Dockerfile, entrypoint, `deploy.py --dry-run`) stays green from Phase 1 onward.

## Budgets per hardware profile
All numbers are **estimates from published measurements and model sizes, UNVERIFIED-ON-GPU**. The
pod smoke test replaces them with measured peaks, and the profile YAMLs are updated accordingly.
The GPU worker time-multiplexes model groups (ARCHITECTURE §5.3), so the budget is per *phase*,
not the sum of all models.

### VRAM (GB)

| Phase (resident set) | gpu48 (L40S, RTX 6000 Ada, A6000) | gpu80 (H100, A100 80GB), **default** | gpu96plus (RTX PRO 6000 96, H100 NVL 94, H200 141, B200 180) |
|---|---|---|---|
| **Understand / judge** | VLM primary Qwen3.6-27B-FP8, 16k ctx (~34) + OCR-VL + layout (~3) + SigLIP 2 (~2) + DA3 (~2) + SAM 3 (~4) ≈ **45**. Judge-2 (Gemma-4-31B QAT, ~22) runs only after the primary sleeps | Primary, 32k ctx (~38) + small models (~11) ≈ **49**; + judge-2 QAT on demand (~22) ≈ **71** | Primary (~38) + judge-2 (QAT ~22 on 94/96 GB; FP8-dynamic ~38 on ≥ 141 GB) + small ≈ **71–87** |
| **Render (Cycles 4K)** | VLMs asleep; Cycles ≤ **12** | ≤ **12** | ≤ **12** (co-resident with VLMs on ≥ 141 GB) |
| **Refine** | Qwen-Image-Edit-2511 BF16 with block-level group offload (~20–26 peak, slow). The FP8 transformer (~20.5 + FP8 text encoder ~8.5 ≈ 35) is enabled **only after the Phase-6 A/B** | BF16 fully resident. Measured peak for the same 20B stack is 61 GiB at 1 MP (vLLM-Omni, H200) → budget **64**; HAT ~2; headroom ~14 | BF16 **64**. On ≥ 141 GB the VLM stays awake (no swap). NVFP4 on Blackwell only after an A/B |
| Global refine resolution / tile | 1.0 MP / 1024 px | 1.6 MP (1664×928) / 1328 px | 1.6 MP / 1328 px |
| Best-of-N (finals) | 2 | 3 | 4 |
| System RAM (vLLM sleep level 1 + offloaded encoders) | ≥ 96 GB; vLLM sleep **level 2** (reload from disk) when group-offloading | ≥ 96 GB (`minRamPerGpu` 96) | ≥ 128 GB |
| Notes | Ampere A6000: FP8 is weight-only (Marlin), so no speed-up | A100 runs Cycles via PTX JIT (cached) and FP8 VLM via Marlin | H100 NVL (94 GB) uses the 94 GB budget |

`cpu_test`: every role mocked; Blender CPU at ≤ 320×180 and 16 spp; no GPU and no model downloads.

### Disk (network volume, GB)

| Item | gpu48 / gpu80 | gpu96plus |
|---|---|---|
| Qwen-Image-Edit-2511 (BF16 repo) + Lightning LoRA | 59 | 59 |
| Qwen-Image-2512 + InstantX ControlNet-Union (text encoder/VAE deduped by SHA256 if identical) | 45–62 | 45–62 |
| FLUX.2 [klein] 4B (fast fallback) | 16 | 16 |
| Qwen3.6-27B-FP8 (primary VLM) | 31 | 31 |
| Gemma-4-31B-it judge-2 (QAT W4A16 / FP8-dynamic) | 20 | 33 |
| OCR (PaddleOCR-VL-1.6, PP-DocLayoutV3, GLM-OCR, RapidOCR, Docling Heron) | 6 | 6 |
| Vision tools (SigLIP 2, SAM 3, SAM 2.1 + Grounding DINO, DA3 ×2, MoGe-2, HAT, Real-ESRGAN, LAION aesthetic + CLIP L/14, own plan segmenter) | 17 | 17 |
| **Model subtotal** | **≈ 194–211** | **≈ 207–224** |
| Optional challengers (Qwen3.8-27B-FP8 +31, FireRed-Image-Edit +41) | +72 | +72 |
| CC0 asset seed (≈150 PBR materials, ≈200 furniture models, ≈30 HDRIs) | 25 | 25 |
| Kernel/compile caches (CUDA, OptiX, torch, vLLM) | 15 | 15 |
| DB replica + snapshots | 2 | 2 |
| ×1.3 headroom on the above | ≈ 306–328 | ≈ 323–345 |
| Project allowance (configurable; ≈ 2–5 GB per section run incl. EXR passes and bundles) | 200 | 200 |
| **Network volume size** | **≈ 550** | **≈ 550** |

- **Cost:** a STANDARD network volume is $0.07/GB/month below 1 TB (RunPod docs), so ~550 GB ≈
  **$39/month**. HIGH_PERFORMANCE is priced at a premium (UNVERIFIED). GPU hourly prices are fetched
  live by `deploy.py` from `/v2/catalog/gpus` and printed before creation.
- **Container disk (template `disk`):** 100 GB (image ≈ 35–40 GB + scratch). 160 GB if hot staging
  of the diffusion weights is enabled after measurement.
- **Load time:** at the documented 200–400 MB/s, a 41 GB transformer loads in ≈ 100–200 s from a
  STANDARD volume. This is why weights stay resident or in pinned CPU RAM between phases, and why
  HIGH_PERFORMANCE volumes are preferred.

### Time per section (3 views, 3840×2160, gpu80), estimate
- Cycles 4K base, 3 views: ≈ 2–5 min each.
- Refine global pass: ≈ 20–30 s per candidate at 30–40 steps with true-CFG (≈ 0.34 s/step without
  CFG on H200).
- Tiled 4K pass: 8 tiles, ≈ 1.5–2 min per candidate.
- QA: ≈ 30–60 s per candidate.
- Best-of-3 per view, plus retries.

**≈ 25–45 min per section.** Measured in the smoke test, and reported by `make eval` per profile.


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
`download_models.py` (license-gated, SHA256, resumable, Hub-license cross-check) and
`pin_models.py` (writes `models.lock.yaml` + LICENSE snapshots), exercised against a registry that
contains only mocks/tiny models. `deploy.py up --dry-run` / `down --dry-run`. License audit script v0. Litestream
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
- **VLM A/B** (GPU): Qwen3.6-27B-FP8 vs Qwen3.8-27B-FP8 on classification, overlay verification and
  judge accuracy. The winner becomes primary by config (ADR-M01).

### Phase 3: Plan extraction + Gate A (S2, S3) + synthetic plan generator
Scope: synthetic plan generator (raster with noise/skew/stamps/hatching/fonts/languages/phone
distortion; vector PDF; DXF; exact GT). Extractors for IFC, DXF, vector PDF and raster (CV baseline).
Shared PlanBuilder, scale estimators + reconciliation, dimension parser, registration, heights,
validators. On-demand VLM coordinate assist (hint → snap to evidence → `vlm_assisted` provenance → Gate A confirmation). Gate A SVG editor with suggestion review. Training-example capture. Plan-segmentation training script
(runs on the pod; trains on synthetic + firm archive; NC datasets forbidden).
Acceptance (`make eval` plan table):
- Vector sources (DXF/PDF): wall F1 ≥ 0.98, opening F1 ≥ 0.95, scale error ≤ 1%.
- Clean raster: wall F1 ≥ 0.92, opening F1 ≥ 0.88. Noisy raster: measured, routed to Gate A.
- Scale disagreement > 1.5% always flagged (property test).
- Validators: 100% detection of injected plan defects (open room, opening outside host, overlapping
  rooms, area-label mismatch > 3%, unreachable room).
- Gate A edits round-trip into an immutable PlanVersion; corrections stored as training examples.
- Assist path: fires only on its triggers (asserted); hints with no evidence are never inserted
  (property test with adversarial hints); accepted assists match synthetic GT within the normal
  tolerance in ≥ 95% of cases; assist rate and acceptance rate reported in `make eval`.

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
- Quantized variants enabled only after a measured A/B on the QA metrics. The A/Bs cover: FP8
  transformer vs BF16 (gpu48 enablement), Lightning 8-step vs full steps for finals, and
  FireRed-Image-Edit vs Qwen-Image-Edit-2511.
- The per-pixel-strength wrapper for `QwenImageEditPlusPipeline` and the tiled refiner are covered by
  CPU tests with a tiny random-weight transformer (shape/blend/seam logic) and by fault injection on
  GPU.

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
Answer these when approving. Each has a default that applies if unanswered.

| # | Question | Why it matters | Default if unanswered |
|---|---|---|---|
| Q-1 | **Which countries** do the firm and its clients operate in? Any data-residency requirement (e.g. EU-only processing)? | License territorial clauses; datacenter choice | Placeholder jurisdictions `[EU, UK, US, CH, TR, KR]`. Any territorial/cap/NC clause blocks the model. Datacenters restricted to EU/EEA with network-volume support (EU-RO-1, EU-CZ-1, EUR-IS-1, EUR-NO-1, …), resolved live |
| Q-2 | **Second document language(s)?** | OCR model choice (GLM-OCR fallback lacks Arabic and Turkish), dimension/label dictionaries, CAD layer-name synonyms | English only in the dictionaries. The primary OCR covers 109 languages |
| Q-3 | **Accept the SAM License** for SAM 3 (trade-controls/sanctions screening of clients, indemnity to Meta, Meta may change terms unilaterally, no patent grant)? | Best text-prompted segmentation for QA opening checks | **Not accepted.** Grounding DINO + SAM 2.1 (Apache) are used instead, and QA thresholds are unaffected (relative metrics) |
| Q-4 | May the firm's **CAD archive** be used to train the plan-recognition model (client contracts)? May we use **CC BY 4.0 datasets** (ResPlan, Modified Swiss Dwellings) with attribution? | Raster-plan accuracy beyond the CV baseline | Synthetic data only, and Gate A stays mandatory for raster plans |
| Q-5 | Is the firm an **ODA member** (ODA File Converter)? | DWG conversion quality | No: LibreDWG, with a request for DXF/PDF export when conversion fails |
| Q-6 | **Budget:** OK with gpu80 (H100/A100 class, hourly price shown at deploy) + a ~550 GB network volume (~$39/month)? Should the idle watchdog be on by default? | Cost | gpu80; watchdog **off** (per brief) |
| Q-7 | Can you share 2–3 **anonymised real projects** for acceptance testing (in addition to the synthetic golden projects)? | Realism of acceptance | Synthetic golden projects only |
| Q-8 | **Credentials for Phase 8:** RunPod API key, Hugging Face read token (with gated terms accepted if Q-3 is yes), GitHub classic PAT with `read:packages` for the pod to pull from GHCR. | One-command deploy | Not needed before Phase 8. CI pushes to GHCR with `GITHUB_TOKEN` |

