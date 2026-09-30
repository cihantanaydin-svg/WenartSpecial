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
   that keeps the Cycles pixels inside the structural masks, with gradient-domain (Poisson) blending
   at the mask borders. The result is re-QA'd like any candidate.
3. **Final rung:** the pure Cycles render.
**Consequences.** Coherent lighting and bounded geometric freedom. Requires pipeline-level code per
model family (the wrapper is covered by mocks on CPU and by fault injection on GPU).

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

## ADR-S10: Default jurisdiction set until the owner specifies countries
**Context.** Fill-in §0 "countries" is blank, and licenses have territorial clauses (e.g. Hunyuan
excludes EU/UK/South Korea).
**Decision.** `configs/deployment.yaml: jurisdictions: [EU, UK, US, CH, TR, KR]` as a conservative
placeholder, and **any** territorial exclusion, MAU/revenue cap or non-commercial clause disables the
entry regardless of the list, until the owner confirms. Data residency: EU datacenters preferred in
the placeholder.
**Consequences.** Some models are disabled that might be legal for the firm. Revisited after answer
Q-1 in PLAN.md. Meta-licensed models (SAM 3) additionally require that clients are not targets of
trade controls; the firm screens clients before enabling them.

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
