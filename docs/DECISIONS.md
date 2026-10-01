# ArchRender: System Architecture Decision Records

Format: Context → Decision → Alternatives → Consequences. Model-role ADRs live in
[MODEL_SELECTION.md](MODEL_SELECTION.md). Status is *Proposed* until Phase 0 approval.

---

## ADR-S01: Single pod, supervised processes, one GPU worker as VRAM arbiter
**Context.** One GPU runs a VLM server (vLLM), diffusion, depth/segmentation and Blender Cycles.
Several independent GPU users on one card cause OOMs and non-reproducible timing.
**Decision.** Exactly one `worker-gpu` process owns all GPU work. It loads diffusion/vision models
in-process (diffusers/transformers), controls the vLLM servers through sleep/wake, and spawns
Blender with a declared VRAM reservation. CPU work runs in a separate `worker-cpu` pool. Everything
runs under `supervisord` (restart policies, log capture, XML-RPC over a 0700 unix socket for
on-demand vLLM judge-2).
**Alternatives.** One process per model with an RPC mesh (more isolation but no global VRAM view).
Kubernetes-style sidecars (not available on a pod). s6-overlay (fine, but supervisord's on-demand
start/stop via XML-RPC is simpler to drive from Python).
**Consequences.** Simple global scheduling (model-affinity batching). GPU throughput is limited to
one job at a time, which is acceptable for a single-GPU firm tool.

## ADR-S02: Three isolated Python environments
**Context.** vLLM 0.30.0 (2026-09-22) pins `torch==2.13.0` and `fastapi<0.137`. The app wants
torch 2.14.0 (2026-09-02; default PyPI wheel cu130) for diffusers 0.40.0, and FastAPI 0.142. Blender
5.2 LTS bundles CPython 3.13.13 with NumPy 2.3.4. (PyPI metadata + Blender `versions.cmake`, checked
2026-09-30.)
**Decision.** Three environments, communicating by HTTP (vLLM) and by files + CLI (Blender):
- `/opt/venv`: app + diffusers + torch 2.14 cu130, Python 3.12, `uv.lock`.
- `/opt/venv-vllm`: vLLM 0.30.x with its pinned torch, hash-locked requirements.
- Blender's Python: our `archrender_blender` scripts only.
**Consequences.** Two torch copies in the image (+~5 GB). No dependency deadlocks, and vLLM can be
upgraded independently. The license audit scans all three environments.

## ADR-S03: Job/state store: SQLite (WAL) on local disk + Litestream replication to the volume
**Context.** The store holds the durable job queue and metadata (projects, versions, gates, audit,
API keys). There is one node and one GPU worker, with a low write rate (< 100 writes/s).
`/workspace` is a RunPod **network volume**, and there is evidence that it is **MooseFS over FUSE**:
`findmnt` shows `mfs#<dc>.runpod.net:9421`, fstype `fuse` (github.com/bschilder/genomeOS/issues/253).
SQLite's docs say "WAL does not work over a network filesystem". RunPod's docs warn that concurrent
writes "may cause data corruption" (`storage/network-volumes.mdx`). The container disk is wiped on
restart.
**Decision.**
- SQLite in WAL mode at `/var/lib/archrender/db` (container disk: local, correct POSIX locks, fast).
- **Litestream** (Apache-2.0, pinned ≥ 0.5.2, which fixed the 0.5.0 restore bugs) continuously
  replicates to a *file* replica at `/workspace/db/replica`. The replica is written only by that one
  process, as immutable LTX files.
- Plus an hourly `VACUUM INTO` snapshot to `/workspace/db/snapshots/` (keep 48) as a second recovery
  path.
- At boot: `litestream restore -if-replica-exists`, falling back to the latest snapshot.
- SIGTERM from the supervisor triggers a final checkpoint.
- Job leasing uses `UPDATE … RETURNING` inside `BEGIN IMMEDIATE`.
**Alternatives.**
- SQLite directly on the network volume: unsafe, per the evidence above.
- Redis/Valkey: still needs a relational store; Redis ≥ 8 is RSALv2/SSPLv1/AGPLv3, and Valkey 9 is
  BSD-3.
- Postgres: robust but heavier, and its data dir on MooseFS has the same caveats.
- Replicating to RunPod's S3-compatible API: possible later, if volumes are shared across pods.
**Consequences.**
- On a hard crash, only state after the last replicated WAL frame (≈1 s) can be lost.
- Stages are idempotent with CAS outputs, so work re-runs from cache.
- User actions are acknowledged only after commit.
- The smoke test includes a kill-and-restore check.

## ADR-S04: Geometry compiled in the app; Blender only materializes, lights, renders, exports
**Context.** Watertight, correct geometry must be testable in CI without Blender. Blender's Python
API scripts are GPL-covered if distributed.
**Decision.** The app (shapely + manifold3d + numpy) compiles all architectural meshes and writes
them into the SceneSpec package. `archrender_blender/build.py` (GPL-3.0-or-later, separate package)
imports meshes, binds materials, lights, cameras, renders passes and exports GLB/.blend. The
interface is `scene.json` (JSON Schema validated on both sides) + mesh files + CLI.
**Consequences.** Property tests on geometry run in milliseconds. Blender scripts stay thin. The
GPL boundary is arm's length, and the image is used internally.

## ADR-S05: Per-project content-addressed storage + stage cache keys
**Decision.** Artifacts are stored under `projects/<id>/cas/sha256/…`, written to a temp file, then
renamed into place (atomic on the same filesystem). A stage cache key is the SHA256 over the stage
id/version, canonical inputs (artifacts by hash), declared config subset, model refs + weight
hashes, and seeds. A stage row is published only after its outputs exist.
**Consequences.** Edits re-run only affected stages. Purge = delete directory + rows. No cross-project
leakage. Some duplication across projects (acceptable).

## ADR-S06: QA geometry metrics are measured relative to the unrefined base render
**Context.** Monocular depth, edge detectors and segmenters have their own error on synthetic-looking
renders. Absolute thresholds would blame the refiner for estimator error.
**Decision.** Every geometry metric is computed on both base and refined images against the same
ground truth, and the *delta* is thresholded (ΔF ≤ 0.05, ΔAbsRel ≤ 0.02, per-opening IoU within
0.05, no new missing/extra openings). Verticals are absolute (±0.5°) because the base is exact by
construction.
**Consequences.** Robust to estimator bias. Needs the base render always (we have it). Fault
injection validates sensitivity.

## ADR-S07: Structural protection by per-pixel strength maps; hard composite only as an escalation rung
**Context.** Pasting base-render structure over a refined image risks lighting seams. Unmasked
refinement risks geometry drift. `QwenImageEditPlusPipeline` (diffusers 0.40.0) has **no `strength`
and no `mask` argument**, and its depth/edge "ControlNet" is in-context only (verified in the
diffusers source), so protection has to be our own code.
**Decision.**
1. **Default: per-pixel strength.** Start from the Cycles render's latents noised to σ₀ with a
   truncated schedule. At every step, re-anchor latents to the render latents noised to the current
   σ according to a per-pixel strength map (structural low, furniture moderate, decor-allowed
   higher), in `callback_on_step_end`. This is differential-diffusion style. Depth and edge maps go
   in as in-context images.
2. **Escalation rung (after the strength-reduction retries fail geometry QA):** a hard composite
   that keeps the Cycles pixels inside the structural masks, with a feathered blend at the mask
   borders (see the amendment below). The result is re-QA'd like any candidate.
3. **Final rung:** the pure Cycles render.
**Consequences.** Coherent lighting and bounded geometric freedom. Requires pipeline-level code per
model family (the wrapper is covered by mocks on CPU and by fault injection on GPU).
**Amendment (Phase 1, 2026-09-30): feathered alpha blend instead of Poisson blending.** Phase 0
proposed gradient-domain (Poisson) blending at the borders. Poisson blending solves for the colours
of the *whole* masked region so that they match the surrounding refined image. That re-colours the
Cycles pixels this rung exists to keep, and QA's relative colour checks would then compare against
pixels that are no longer the render. The implementation
(`refine/strength.py:hard_structural_composite`) uses a Gaussian-feathered alpha of the structural
mask (σ = image width / 800, at least 1 px): structural pixels are kept exactly except within a band
of about 2σ either side of the mask border, where render and candidate are mixed. Seam visibility in
that band is measured by the Phase-6 fault-injection harness; Poisson blending is reconsidered only
if seams are visible there.

## ADR-S08: Async API: jobs + SSE + polling; custom chunked upload protocol
**Context.** The RunPod proxy cuts requests at 100 s. Uploads can be GBs. Browser EventSource cannot
set auth headers.
**Decision.** 202 + job id for anything long. SSE with 15 s heartbeat comments, event ids and
`Last-Event-ID` resume, plus a plain polling endpoint. The server caps every request at 55 s. Uploads
use a small custom resumable protocol (create → PUT chunks with per-chunk SHA256 → list received →
complete). UI auth via an httpOnly session cookie (so EventSource works) + CSRF token. CLI via
Bearer key.
**Alternatives.** tus protocol (Python server implementations are thin/unmaintained). WebSockets
(proxy behaviour and reconnection are harder than SSE).
**Consequences.** Small protocol surface, fully testable. The CLI and UI share the same client
logic.

## ADR-S09: Line-art ground truth derived from render buffers + projected CAD edges
**Context.** Freestyle/Line Art is slow and parameter-sensitive. QA needs exact structural edges.
**Decision.** Line art = discontinuities in depth/normal/object-index buffers restricted to
structural objects ∪ projected 3D architectural edges (from the compiled geometry and camera
matrices) with depth-test occlusion.
**Consequences.** Exact, fast, deterministic, and unit-testable (reprojection within 1 px).

## ADR-S10: Jurisdiction = Türkiye; EU datacenters closest to Türkiye (owner answer Q-1)
**Context.** Owner answer (2026-09-30): the firm and its clients operate in **Türkiye**. Licenses
have territorial clauses. For example, the Tencent licences exclude EU/UK/South Korea: Türkiye is
not excluded, but the MAU caps still block them. RunPod has no datacenter in Türkiye.
**Decision.**
- `configs/deployment.yaml: jurisdictions: [TR]`.
- The strict rule stays: **any** territorial exclusion, MAU/revenue cap or non-commercial clause
  disables an entry, whatever the jurisdiction list says.
- Datacenter preference: network-volume-capable EU/EEA DCs nearest to Türkiye, in order: `EU-RO-1`
  (Romania), `EU-CZ-1` (Czechia), then the others returned live by the catalog (EUR-NO-1, EUR-IS-*…).
  This order is overridable in `.env`.
- Meta-licensed models (SAM 3, accepted per Q-3) require that clients are not targets of trade
  controls, so the firm screens clients.
**Consequences.**
- Client documents leave Türkiye for an EU datacenter. Under **KVKK (Law No. 6698), Art. 9**,
  cross-border transfers of personal data need a legal basis/safeguards (e.g. standard contracts
  notified to the KVKK Authority). This is a **legal action item for the firm**, not something code
  can satisfy.
- The system minimises personal data: EXIF GPS and owner metadata are stripped at intake, and
  purge/retention endpoints exist. DEPLOY.md records the item.

## ADR-S11: PDF handling without AGPL
**Decision.** pypdfium2 (Apache-2.0 / BSD-3; PDFium BSD-3) for vector paths, text objects and
rasterisation; pdfplumber (MIT, on pdfminer.six MIT) for table-ish text layout as a secondary.
PyMuPDF is excluded (AGPL) unless `licenses.pymupdf_commercial` is configured. No poppler-utils
binaries (GPL) in the runtime path.

## ADR-S12: UI served same-origin with relative URLs and hash routing
**Decision.** Vite `base: './'`, all fetches relative (`api/v1/...`), `HashRouter`. This works
behind `https://{POD_ID}-8000.proxy.runpod.net` or any prefix without server rewrites.

## ADR-S13: Reports as HTML → PDF with WeasyPrint
**Decision.** Jinja2 HTML report (also delivered as HTML) → PDF via WeasyPrint 70 (BSD-3; system
Pango libs, LGPL, dynamically linked).
**Alternatives.** ReportLab (BSD, programmatic layout, more code); headless Chromium (large).

## ADR-S14: RunPod deployment through REST API v2, with an explicit placement loop
**Context.** REST v1 (`rest.runpod.io/v1`) "is deprecated and will be retired on November 15,
2026", and GraphQL is retiring in early 2027. REST v2 (`https://api.runpod.io/v2`) went GA on
2026-08-18. A v2 pod create "places one specific GPU type … does not fall back". v2 can create
secrets (`/v2/account/secrets`) and exposes catalogs with price, stock, CUDA versions and datacenter
compliance. Evidence: `github.com/runpod/docs` @07ba10e, see [research/RUNPOD.md](research/RUNPOD.md).
**Decision.** `deploy.py` (stdlib + requests) uses v2 only:
1. Read the GPU and datacenter catalogs (filtered by profile GPU list, `minCudaVersion`,
   residency region/compliance, network-volume support).
2. Create or reuse the network volume in the chosen DC (`HIGH_PERFORMANCE` type where offered).
3. Upsert secrets (`archrender_hf_token`, `archrender_admin_token`) and registry credentials.
4. Upsert the template (no mounts; `startJupyter: false`).
5. Loop create-pod over (GPU type, DC) candidates, following the spec's retry semantics.
6. Poll `status`; stream `/v2/pods/{id}/logs?source=system` on failure (image-pull and CUDA-mismatch
   diagnostics).
7. Wait for `/readyz` via the proxy URL.

`down` **terminates** the pod and keeps the volume, because a stopped pod may come back with
"Zero GPUs". `runpodctl` equivalents are emitted as a courtesy (runpodctl v2.14 still calls v1 and
GraphQL).
**Consequences.** Robust against capacity gaps. No console clicks except accepting gated-model terms
on Hugging Face.

## ADR-S15: Blender 5.2 LTS, Cycles GPU with OptiX → CUDA fallback, persistent kernel caches
**Context.**
- **Versions:** Blender 5.2 LTS (5.2.2, tagged 2026-09-14) is supported until July 2028; 4.5 LTS
  until July 2027.
- **CUDA kernels:** 5.2 ships CUDA cubins for `sm_50…sm_75, sm_86, sm_120` + `compute_75` PTX (the
  `CYCLES_CUDA_BINARIES_ARCH` default on the 5.2 branch). So A100 (sm_80), H100/H200 (sm_90) and B200
  (sm_100) **JIT-compile PTX on first use**, while L40S/RTX 6000 Ada use the sm_86 cubin.
- **OptiX:** needs driver ≥ 575 (5.2 manual *(snippet)*). `libnvoptix.so` is mounted only with the
  `graphics` driver capability (`libnvidia-container src/nvc_info.c`).
- **API changes in 5.x:** `Scene.node_tree` → `compositing_node_group`; `get_devices()` deprecated
  → `refresh_devices()`.
**Decision.**
- Blender 5.2.2 LTS tarball in `/opt/blender`, SHA256-verified. The official `.sha256` is fetched in
  CI and compared to the pinned `ARG BLENDER_SHA256`.
- `ENV NVIDIA_DRIVER_CAPABILITIES=all`.
- A device probe at boot: OPTIX → CUDA fallback, logged to the manifest.
- `CUDA_CACHE_PATH`/`OPTIX_CACHE_PATH` on `/workspace/cache` with a 4 GiB max, so the one-time
  JIT is paid once per volume. The boot self-test warms the cache with a tiny render.
- Cycles only: no EEVEE, so no EGL dependency for rendering.
- A golden-image test guards version drift.

## ADR-S16: DWG conversion via LibreDWG CLI; ODA only with membership
**Context.** ODA File Converter is free, but "if you are not an ODA member … you can use them for
non-commercial applications only" (opendesign.com FAQ *(snippet)*). LibreDWG is GPL-3.0-or-later.
**Decision.**
- Default: `dwg2dxf` (LibreDWG 0.14) run as a separate sandboxed process on files, which is
  arm's-length use with no linking.
- The ODA converter is enabled only if `licenses.oda_member: true` is configured.
- Conversion failures, or unsupported DWG versions, yield `INGEST_DWG_CONVERSION_FAILED` with the
  hint "export DXF (or PDF/IFC) from your CAD tool".

## ADR-S17: HEIC decode via libheif CLI (decoder plugin only)
**Context.** pillow-heif wheels bundle x265 (GPL-2) + libheif/libde265 (LGPL-3)
(`LICENSES_bundled.txt`).
**Decision.** Convert HEIC → PNG with Ubuntu's `heif-dec` (libheif, LGPL) in the sandboxed parser
subprocess, installing only the libde265 decoder plugin (no x265 encoder). pillow-heif is not a
dependency.

## ADR-S18: CUDA 13.0 runtime base; minimum host CUDA 13.0
**Context.**
- torch 2.14 wheels: cu126 (no Blackwell), cu130, cu132. The cu130 wheel covers sm_75–sm_120.
- vLLM 0.30.0 defaults to cu130 (a cu129 variant exists).
- Minimum drivers: CUDA 13.0 ≥ 580.65.06; 12.9 ≥ 575.51.03; 12.8 ≥ 570.26.
**Decision.**
- `FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04@sha256:76f46f3e…` (digest from Docker Hub
  2026-09-30; re-confirmed with `docker buildx imagetools inspect` in CI).
- Pods are created with `gpu.minCudaVersion: "13.0"`.
- The entrypoint checks `nvidia-smi` driver ≥ 580 and exits with a clear message otherwise.
**Consequences.** One image for Ampere → Blackwell. The host pool is limited to drivers ≥ 580, and
deploy.py reports when no capacity matches.

## ADR-S19: VLM coordinate assistance: on demand, hint-then-snap, never direct
**Context.** The original brief said the VLM "never supplies coordinates". The owner has since asked
(2026-09-30 review comment on ARCHITECTURE.md) to **get help from the VLM for coordinates, but only
when needed**. Deterministic extractors fail in predictable places: noisy rasters, phone photos,
unusual symbols, missing title blocks or north arrows. Modern VLMs (Qwen3.6/3.8) can ground
points and boxes, but their coordinates are imprecise, and on downscaled sheets they can be
hallucinated.
**Decision.** Add an **assist path** (`plan/assist.py`) with five rules:
1. **Triggered, not default.** It runs only on named triggers: no candidates where ink exists, a
   validator failure, low extractor confidence, a missing title block/north arrow/scale bar, or
   undetected sheet corners. Each call is budgeted per sheet and logged.
2. **Full-resolution tiles only.** The VLM receives a tile, not a downscaled sheet. Its answer is
   schema-constrained points or boxes in tile pixels, mapped back to page coordinates.
3. **Hint, then snap.** The VLM output only defines a search window. Geometry is rebuilt by
   deterministic fitting from actual evidence: raster strokes, vector segments, arcs, text boxes.
   Acceptance requires evidence coverage ≥ 80% and a snap residual ≤ 1.5× the normal tolerance.
4. **Visible and reviewable.** Accepted results carry `Provenance.method = "vlm_assisted"` with the
   raw hint and evidence score, and Gate A cannot auto-pass until the user confirms them. Hints
   without evidence are shown only as dashed suggestions and are never inserted automatically.
5. **Never for scale or dimensions.** Metric scale always comes from the scale estimators.
**Alternatives.** (a) Keep "never coordinates": more manual work at Gate A for noisy inputs.
(b) Accept VLM coordinates directly: violates "measured, never hallucinated".
**Consequences.** Fewer manual edits at Gate A on hard inputs, while every coordinate in an approved
PlanGraph is still backed by measured evidence or a human decision. Suggestion accept/reject
decisions become training data. The eval reports assist rate, acceptance rate and precision of
accepted assists against the synthetic ground truth.

## ADR-S20: Untrusted parsers run in a resource-limited child process (Phase 2)
**Context.** Uploads are adversarial by default: zip bombs, malformed PDFs that crash PDFium,
decompression bombs in images, pathological DXF. The worker also runs heartbeat threads, so
`preexec_fn` resource limits are unsafe.
**Decision.** Every parser of uploaded content (pypdfium2, Pillow, ezdxf, python-docx, openpyxl,
python-pptx, zipfile) and every external tool (LibreDWG `dwg2dxf`, libheif) runs in a fresh process
launched through `archrender.ingest.limits`, which sets RLIMIT_AS/CPU/FSIZE/NOFILE/CORE and then
`exec`s the target. Requests and replies are JSON on stdin/stdout; limits, timeouts, crashes and
parser exceptions become coded errors (`INGEST_LIMIT_EXCEEDED`, `INGEST_CORRUPT`, …) with fix hints.
A file enters the project store only after it parses. Type detection in the worker reads magic
bytes only; the directory of a ZIP container (to tell DOCX/XLSX/PPTX from an archive) is listed in
the sandbox too (found by the intake fuzz test: a corrupted DOCX raised inside the worker).
**Consequences.** ~100 ms process start-up per file; a hostile file cannot exhaust the worker or
the host. No network namespace isolation (not available in the pod); the parsers make no network
calls.

## ADR-S21: Office files through python-docx/openpyxl/python-pptx, not Docling (Phase 2)
**Context.** ADR-M03 routed Office documents through Docling. For DOCX/XLSX/PPTX, Docling's own
backends are these same libraries; the rest of Docling (layout models, torch) adds weight and model
downloads without improving OOXML parsing.
**Decision.** Parse OOXML directly (paragraphs, tables, sheets, slides, embedded images). Macros are
never executed: `.xlsm` is read as data with the VBA project ignored; `.docm`/`.pptm` are rejected
with a hint to save plain files. openpyxl uses defusedxml (XML bomb protection).
**Consequences.** Smaller image, no model downloads for Office files. Docling stays out of the
dependency set (it can return for PDF layout if PP-DocLayoutV3 disappoints; it would then be an
explicit ADR change).

## ADR-S22: Plan versions, Gate A and the plan a run renders (Phase 3)
**Context.** Gate A must let a person correct the extracted plan without losing what was extracted,
runs are re-entrant (a parked run restarts from the top), and the owner wants corrections kept
(but not used for training without their consent, answer Q-4).
**Decision.**
- S2 stores each new extraction as a **draft plan version** (content-hashed: unchanged pages reuse
  their version). Every edit is an RFC 6902 JSON Patch that makes a new draft; edited facts get
  `user` provenance. Versions never change; only their status moves (draft → approved →
  superseded).
- Approval is refused while the plan has blocking issues (errors: open rooms, unresolved scale
  conflicts, unconfirmed VLM-assisted elements …). Approving makes the previous approval superseded
  and stores the corrections since the extraction as a `training_examples` row with
  `training_use_allowed = 0`.
- **The approved version wins**: a run uses the approved version of the current extraction if
  there is one (later drafts take effect once approved), otherwise its latest edit. A run **pins**
  the version it starts with (`runs.plan_version`), so a resumed run never picks up a plan approved
  in the meantime; approving Gate A approves a version (the one named, else the latest edit of the
  parked one) and re-pins the run to it. A later run with an approved version records Gate A as
  approved ("approved before this run") instead of asking again.
- Resolving a conflict is an edit: a scale choice rescales the page's level (lengths, not heights or
  stated area labels); a schedule/drawing width choice sets the opening's width. Two-point
  calibration (a known length) rescales the plan with the user's measurement as provenance.
**Consequences.** Every coordinate a run renders is traceable to an extraction plus named edits.
Edits after approval need a new approval; the editor shows this. Plan rows are per project and
cascade with it.

## ADR-S23: S2 sources, levels and the PLAN job (Phase 3)
**Decision.**
- Plan pages are those S1 labels `floor_plan` (or the user relabels), plus IFC models. Per level,
  precedence IFC > DXF > vector PDF > raster; raster pages are not extracted at all when a vector or
  model source exists (scans of sheets we also have as vectors). A PDF page counts as vector when it
  has ≥ 30 drawn paths and images cover < 50 % of it.
- Several 2D sources become levels in storey order parsed from their names (Turkish and English:
  bodrum/basement < zemin/ground < asma/mezzanine < "1. Kat"/"First Floor"/"Level 1" < çatı/roof),
  elevation = rank × a default 3.0 m storey height (an assumption). Names that say nothing go above
  the others in sheet order, flagged for review.
- A PLAN job follows S1 whenever the project has a plan source; a run that starts before S1 has
  finished runs S1 first (cached pages are hits). Extraction failures of one page are recorded and
  the others used; a project with no usable page fails with `PLAN_NO_PLAN_FOUND` and a fix hint.
- Assumptions that need a person (level order, an RCP that does not register) are kept on the plan
  and shown as `PLAN_ASSUMPTION_REVIEW` warnings (validators recompute issues per version).
- A page whose scale rests on one weak estimate (relative uncertainty > 2 %: door swings, a single
  dimension string) gets a one-candidate scale conflict: blocking until the user confirms it or
  calibrates (a calibration settles the page's scale questions). Found by the eval: a noisy scan
  whose stated scale OCR missed was 4.3 % off with no warning.

## ADR-S24: The VLM assist as built (Phase 3; refines ADR-S19)
**Decision.**
- Scope: raster pages (scans, rectified photos), where the CV baseline misses things. Vector and
  model sources are exact; their problems are drawing errors for Gate A.
- Triggers (`plan/assist.py`): validator failures (open room, area-label mismatch → ask for walls;
  unreachable room → ask for openings), wall body joined to the walls that no extracted wall
  explains, walls with confidence < 0.5. Overlapping windows are merged; ≤ 20 calls per sheet; a
  failed call is logged and the extraction continues.
- Snapping: a wall is measured from cross-sections of the wall body (or two face strokes) along
  the hint, robust line fit (junction/crossing outliers rejected), thickness from the median width.
  New walls need evidence over ≥ 80 % of the hinted span and a fit residual ≤ 1.5 strokes. A hint
  along a wall already in the plan counts only where it **extends** it: over continuous wall body
  (a junction lost where a stamp touches the wall), or across a gap with a door/window symbol to the
  next wall (a short piece beside a door was lost). Openings are measured as the gap in the host's
  body with jambs on both faces; the symbol in the gap (swing arc about either jamb, glazing lines)
  decides the type.
- Accepted walls are added as filled rectangles of their drawn pieces to the primitives and the plan
  is rebuilt, so junctions, rooms and the openings in them come from the normal builder; walls that
  are new or changed get `vlm_assisted` provenance (confidence 0.6, with the hint, coverage and
  residual). Accepted openings are hosted directly. `PLAN_ASSIST_UNCONFIRMED` (error) blocks
  approval and auto-pass until a person confirms them. Unsnapped hints are stored as suggestions;
  accepting one inserts the user's own element (rooms re-derived), rejecting it is recorded; both
  decisions are training examples (`assist_decision`, not for training without the owner).
- The stage cache key includes whether the VLM serves, so a page extracted without it is redone
  once it does.
**Measured** (`make eval`, ground-truth hints standing in for the VLM; property test with
adversarial hints): see PROGRESS.md. The VLM's own pointing accuracy is UNVERIFIED-ON-GPU.

## ADR-S25: Annotations from the other sheets: RCP heights and schedules (Phase 3)
**Decision.**
- The reflected ceiling plan is extracted like a floor plan and registered onto its level's plan:
  similarity transform by ICP on wall centre-line samples (target sampled every 1 cm so the residual
  is a distance to the lines), started from 0/90/180/270° and from the dominant-direction offset.
  It is used only if ≥ 70 % of its walls match with RMS ≤ 20 mm; otherwise the heights are not used
  and the user is told. Height labels ("+2,70", "h=2.80", "TH 3.00"; a bare number is not one) set
  the room's ceiling height (`pdf_text`/`ocr` provenance); ≥ 1.6 × storey → double height; walls
  reach the highest ceiling of the rooms they bound.
- Door/window tags are read from the extraction's own texts (DXF, PDF and OCR alike) and given to
  the nearest compatible opening (≤ 1.5 m, nearest pairs first). A schedule row for the tag supplies
  height and sill (replacing defaults). Its width is the specification: when it differs from the
  drawn width by > 3 cm, the schedule width is used and the disagreement is a Gate A conflict
  (warning, resolvable either way).

## ADR-S26: Plan segmentation v1 (Phase 3; amends ADR-M06)
**Context.** ADR-M06 named segmentation-models-pytorch with a DINOv2/ConvNeXt encoder. The training
data are synthetic only (owner answer Q-4) and no GPU is available in the build environment.
**Decision.** v1 is a small U-Net in plain PyTorch (`plan/seg_train.py`; no new dependency), trained
on tiles of scanned synthetic sheets labelled from the exact ground truth (`plan/seg_data.py`:
background, wall, door, window, room). The `raster_seg` hook is built: a predicted wall mask
replaces the vectoriser's morphological wall body; everything downstream is unchanged. The
`plan_segmenter` role (TorchScript impl) is mapped in no profile until a checkpoint passes the
promotion rule (beats the CV baseline on held-out sheets and meets the clean-raster targets).
**Measured ceiling.** With the ground-truth wall mask, noisy 300-DPI scans that the CV body gets
wrong (stamps, hatch, outline walls) extract with walls 1.00, rooms 1.00, openings 0.90–1.00.
**Consequences.** A stronger encoder can replace the U-Net later behind the same hook; the
training run on the pod is UNVERIFIED-ON-GPU.

## Phase-2 amendments to earlier ADRs
- **ADR-M03 (OCR): Tesseract 5 replaces RapidOCR as the CPU fallback.** RapidOCR's ONNX models are
  hosted on ModelScope/Hugging Face, which this build environment cannot reach, so it could not be
  verified here. Tesseract 5.3.4 (Apache-2.0) with the `tur` and `eng` LSTM data (Apache-2.0,
  Ubuntu packages) runs everywhere, handles Turkish (ç ğ ı İ ö ş ü), and is measured in CI.
  Door/window tags inside bubbles are read with a dedicated single-line pass. PaddleOCR-VL-1.6
  stays the primary (GPU, not yet implemented: the role falls back to Tesseract with a recorded
  degradation). Tiles are despeckled (3×3 median) before Tesseract's sparse-text mode, which
  otherwise spent minutes on the paper noise of a single 200-DPI scan tile (measured: > 120 s vs
  0.4 s; accuracy unchanged on the synthetic benchmark). Each call has a 120 s timeout; a tile that
  still fails is recorded on the page and sent to review (`page_ocr`), and the other tiles are kept.
- **Bubble tags on scans:** ink is thresholded against the local paper level (not a fixed grey),
  both ring boundaries are ellipse candidates (a door swing touching the ring destroys the outer
  one), and the lettering is isolated by connected components after an opening that detaches thin
  swing/wall strokes. A tag counts only when ≥ 2 of 3 magnifications read it and none reads a
  different tag; only if that fails are digit look-alikes after a door/window prefix repaired
  (`PS` → `P5`) and the vote repeated. Measured on 6 plans per quality: 75/77 (97 %) at 300 DPI for
  clean, medium and noisy scans; at 200 DPI 68, 66 and 47 of 77; 0 false tags throughout.
- **Classifier temperature is bounded below by 1.** Temperature scaling on the (synthetic, nearly
  separable) calibration corpus drove T to its lower limit (0.05), i.e. near-certain confidence on
  every page and an empty review queue on real documents. T may soften the L2-regularised fit but
  never sharpen it; the calibrated combiner trained on real pages (pod) replaces this.
- **Pages outside the classifier's training range go to review.** The model stores each feature's
  training range; features outside it (5 % tolerance) are clipped before scoring and the page is
  flagged for review with the features named (a one-line DXF had come out as "moodboard, 100 %").
  With the VLM serving, such a page leaves review only if the VLM confidently gives the same class.
- **Uploads are staged without their extension** (`<upload id>.bin`); parsers must not depend on
  the filename. openpyxl did (every uploaded workbook failed); workbooks are now opened from a file
  object, and a test ingests every format from a staging path.
- **ADR-S16 (DWG): LibreDWG 0.14 built from the pinned, checksum-verified source tarball**
  (`deploy/libredwg/build.sh`, read-only build) because Ubuntu 24.04 has no LibreDWG package. Its
  R2000 output writes handle 0 for some ENDBLK records; intake renumbers them and reports it.
- **ADR-S17 (HEIC): Ubuntu 24.04's libheif 1.17 ships `heif-convert`** (`heif-dec` from 1.18);
  intake uses whichever exists. Decoder plugin libde265 only.
- **ADR-M01 (VLM) classification combiner:** until the calibrated combiner with VLM features is
  trained on the pod (`archrender.understand.train`, UNVERIFIED-ON-GPU), heuristics and VLM are
  combined by an equal-weight geometric mean; confident disagreement always goes to review.
- **Readiness and fallbacks:** roles whose model (and fallbacks) have no runtime in the current
  build run on their mock with a recorded degradation; `/readyz` lists them as degraded instead of
  blocking. Readiness only considers the roles the implemented stages call.

## Phase-3 amendments to earlier ADRs
- **ADR-S04 (scene geometry): wall footprints may have several parts, and junctions go to the main
  walls.** Extracted plans have T and X junctions in any wall order; the Phase-1 partition (first
  wall owns the junction; a split footprint was an error) rejected them. Walls now own their
  footprint in priority order (exterior, thicker, longer), and a wall crossed by another keeps
  every part. Extracted plans of all variants compile watertight.
- **Room labels on rotated scans.** Words are joined into lines in the text's own frame (baseline
  clusters, then left to right), and a word's letter height comes from its box and angle; a 1°
  scan rotation had turned every word "vertical" (height = box width) and split "Yatak Odası".
