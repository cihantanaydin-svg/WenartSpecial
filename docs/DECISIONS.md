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
**Context.** vLLM wheels pin an exact torch/CUDA combination. diffusers support for new image models
often needs a newer torch. Blender bundles its own CPython.
**Decision.** `/opt/venv` (app + diffusers, `uv.lock`), `/opt/venv-vllm` (vLLM, hash-locked
requirements), and Blender's Python (our `archrender_blender` scripts only). They communicate by
HTTP (vLLM) and by files + CLI (Blender).
**Consequences.** Two torch copies in the image (+~5 GB). No dependency deadlocks, and vLLM can be
upgraded independently. The license audit scans all three environments.

## ADR-S03: Job/state store: SQLite (WAL) on local disk + Litestream replication to the volume
**Context.** Durable job queue + metadata (projects, versions, gates, audit, API keys). One node, one
GPU worker, low write rate (< 100 writes/s). `/workspace` is a RunPod **network volume**. SQLite
over network filesystems risks broken locking and WAL shared-memory corruption; SQLite's own
documentation warns against network filesystems. The container disk is wiped on restart.
**Decision.** SQLite in WAL mode at `/var/lib/archrender/db` (container disk: local, correct POSIX
locks, fast) + **Litestream** (Apache-2.0, pinned ≥ 0.5.2, which fixed the 0.5.0 restore bugs)
continuously replicating to a *file* replica at `/workspace/db/replica`. At boot, `litestream restore
-if-replica-exists`. A graceful shutdown (SIGTERM from supervisor) takes a final checkpoint. Job
leasing uses `UPDATE … RETURNING` inside `BEGIN IMMEDIATE`.
**Alternatives.** (a) SQLite directly on the network volume: unsafe unless the filesystem's locking
is proven, and we cannot prove it from here. (b) Redis/Valkey with AOF on the volume: still needs a
relational store for everything else; adds a server; Redis ≥ 7.4 license changes (Valkey is
BSD-licensed). (c) Postgres: robust but heavier, and its data dir on a network FS has similar caveats.
**Consequences.** On a hard crash, state after the last replicated WAL frame (seconds) can be
lost. Stages are idempotent with CAS outputs, so work re-runs from cache. User actions are
acknowledged only after commit, and Litestream's sync interval is set to 1 s. Evidence on the volume
filesystem type is recorded in MODEL_SELECTION.md §Tools. The smoke test includes a kill-and-restore
check.

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

## ADR-S07: Structural protection by per-pixel strength maps, not hard compositing
**Context.** Pasting base-render structure over a refined image creates lighting seams. Unmasked
refinement risks geometry drift.
**Decision.** Implement differential-diffusion-style latent blending: at each denoising step,
latents in each region are re-anchored to the noised base latent according to a per-pixel strength
map (structural low, furniture moderate, decor-allowed higher). Implemented as our own pipeline
wrapper over the model's transformer/scheduler, with the model's native depth/edge conditioning on
top.
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
Q-1 in PLAN.md.

## ADR-S11: PDF handling without AGPL
**Decision.** pypdfium2 (Apache-2.0 / BSD-3; PDFium BSD-3) for vector paths, text objects and
rasterisation; pdfplumber (MIT, on pdfminer.six MIT) for table-ish text layout as a secondary.
PyMuPDF is excluded (AGPL) unless `licenses.pymupdf_commercial` is configured. No poppler-utils
binaries (GPL) in the runtime path.

## ADR-S12: UI served same-origin with relative URLs and hash routing
**Decision.** Vite `base: './'`, all fetches relative (`api/v1/...`), `HashRouter`. This works
behind `https://{POD_ID}-8000.proxy.runpod.net` or any prefix without server rewrites.

## ADR-S13: Reports as HTML → PDF with WeasyPrint
**Decision.** Jinja2 HTML report (also delivered as HTML) → PDF via WeasyPrint (BSD-3; system Pango
libs, LGPL, dynamically linked).
**Alternatives.** ReportLab (BSD, programmatic layout, more code); headless Chromium (large).
