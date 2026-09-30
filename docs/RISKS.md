# ArchRender: Risk Register

Status: Phase 0 draft · 2026-09-30. Likelihood (L) and impact (I) are rated H/M/L. Each risk has
an owner mechanism and an early-warning signal. Re-review at the end of every phase.

## A. Product quality (the "never ship a flawed image" promise)

| ID | Risk | L | I | Mitigation | Early signal / test |
|---|---|---|---|---|---|
| Q1 | Generative refinement subtly changes geometry (moved window, bent wall) and QA misses it | M | H | Low per-pixel strength on structural masks; depth+edge conditioning. QA metrics are **relative to the base render** (removes estimator bias). Geometry is decided by deterministic metrics, never by the VLM. `DeliverableGuard` in S10 re-checks every delivered hash. Fallback to the pure Cycles render. | Fault-injection confusion matrix (target ≥ 95% detection, ≤ 5% false alarms) in every `make eval` |
| Q2 | Fault-injection targets not met for small corruptions (window moved 0.3 m, slight warp) | M | M | Improve detectors (opening matching, per-opening IoU, edge metrics at native res crops), never the thresholds or the fixtures. Report per-corruption detection honestly with analysis. | Phase 6 eval table |
| Q3 | VLM judge hallucinations or inconsistency | H | M | Temperature 0, schema-constrained binary questions, evidence (bbox + reason) required, second-family tie-breaker on critical checks, deterministic overrides. Measure judge accuracy on labelled fault sets. | Judge agreement rate; fault-injection results per check |
| Q4 | Raster plan extraction below target without a trained checkpoint | H | M | Gate A is **mandatory** for raster plans. CV baseline + VLM overlay verification. Synthetic generator + firm CAD archive → train on pod. Never use NC datasets. | `make eval` raster F1 vs targets (≥ 0.92 walls / ≥ 0.88 openings clean) |
| Q5 | Scale mis-estimation (wrong sheet scale, mixed units, "NTS" sheets) | M | H | ≥ 2 independent estimators, a 1.5% disagreement flag, 2-click calibration at Gate A, dimension-string RANSAC, unit inference by magnitude clustering | Scale error on synthetic set ≤ 1% |
| Q6 | Cross-view inconsistency (floor looks different per view) | M | M | Materials live in the 3D scene; low strength; per-material ΔE harmonisation; VLM pairwise checks | Cross-view ΔE in eval |
| Q7 | Asset library too thin or stylistically off for the brief (CC0 furniture is limited) | H | M | Firm-owned asset import pipeline with validator; derive-and-flag materials; optional image-to-3D (flagged generated); Gate B/C show exactly what will be used | Brief-adherence failures attributed to "no asset match" |
| Q8 | Refiner "improves" materials away from the spec (e.g. oak → walnut) | M | M | Per-material ΔE vs base render; structural strength low; prompt compiled from brief; deterministic re-fix first | Brief-adherence metrics |
| Q9 | Lighting realism: exposure/white balance inconsistent across views | M | L | Auto-exposure prepass per view with a shared section-level exposure bias; AgX; CCT from brief; recorded as assumptions | Technical QA (clipping, cast) |

## B. Licensing and legal

| ID | Risk | L | I | Mitigation | Early signal / test |
|---|---|---|---|---|---|
| L1 | A model or its dependency is not commercial-OK for our jurisdictions (split licenses in a family, NC text encoders or VAEs, territorial exclusions, MAU/revenue caps, API-only newer versions) | M | H | Registry with license evidence URL + date; `LicenseGate` at download, load and CI; at download, `download_models.py` compares the Hub's license metadata with the registry and **blocks on mismatch** | CI license audit; download-time mismatch error |
| L2 | **Jurisdictions not yet specified** (fill-in §0 blank) | H | M | Until specified, default to the most restrictive set: any territorial exclusion, MAU/revenue cap or NC clause → disabled | Open question Q-1 in PLAN.md |
| L3 | AGPL/GPL contamination in a network service (PyMuPDF, Ultralytics, MinerU, poppler tools, LibreDWG) | M | H | Dependency audit with an allowlist. GPL tools run only as separate processes via CLI with files (arms-length), and are optional. AGPL is excluded unless a commercial license is configured. | `scripts/license_audit.py` in CI |
| L4 | Blender GPL: our `bpy` scripts are GPL-covered if distributed | L | L | Keep `archrender_blender/` as a separately licensed GPL-3.0-or-later package. The app talks to it via JSON files + CLI only. The image is used internally, not distributed to clients. | Audit lists the boundary |
| L5 | Gated model terms (e.g. SAM License) impose obligations (attribution, acceptable use) | M | M | Record obligations in the registry; `THIRD_PARTY_LICENSES.md` includes notices; fallback to an Apache model if terms are rejected | Registry `obligations` field |
| L6 | CC0 asset provenance (mislabelled uploads) | L | M | Seed only from curated CC0 sources with source URLs; firm assets require an explicit license field; the import validator rejects unknown licenses | Asset audit |
| L7 | Paint-code (RAL/NCS/Pantone) conversion tables have their own licenses | M | L | Use only public-domain or openly licensed conversion data; results flagged as approximations; unsupported codes routed to the user | License audit covers data files |
| L8 | Client confidentiality: data leaves the pod | L | H | No third-party AI APIs; HF offline after boot; telemetry off; Secure Cloud; model servers on localhost; audit log; purge endpoint | Boot egress self-test; route-auth test |

## C. Infrastructure (RunPod, GPU, containers)

| ID | Risk | L | I | Mitigation | Early signal / test |
|---|---|---|---|---|---|
| I1 | RunPod proxy cuts requests at 100 s (HTTP 524) | H | M | Everything long is a job; 55 s handler cap; SSE heartbeats every 15 s + `Last-Event-ID` resume; 16 MiB upload chunks; bundles prebuilt and served with Range | API timing tests; smoke test through the proxy URL |
| I2 | Proxy URLs are public | H | H | Auth on every route (test enumerates routes); rate-limit auth failures; high-entropy keys; one-shot bootstrap token from a RunPod secret | Route-auth test |
| I3 | OptiX unavailable in the container (driver libs not injected) | M | M | `NVIDIA_DRIVER_CAPABILITIES=all`; Blender device probe at boot; automatic CUDA fallback, logged in the manifest | `/readyz` self-test output |
| I4 | Host driver too old for the image's CUDA (Blackwell needs CUDA 12.8+) | M | H | Pod creation filters by allowed CUDA versions; entrypoint checks driver ≥ required and exits with a clear message | Entrypoint check |
| I5 | Network-volume throughput makes model loads slow | M | M | Measure load times per model at boot; stage hot models on container disk (sized in the template) when faster; keep models resident where VRAM allows | Smoke-test load timings |
| I6 | SQLite locking/corruption on a network filesystem | M | H | DB on local container disk + Litestream continuous replication to the volume; restore at boot (ADR-S03) | Kill-and-restart test in the smoke test |
| I7 | Container disk wiped on restart → lost installs (fallback path) | H | M | Image-based deploy by default; the bootstrap fallback installs under `/workspace/opt` | Restart test in DEPLOY.md checklist |
| I8 | CUDA OOM (co-residency, 4K tiles, best-of-N) | M | M | ModelManager budgets, LRU, text-encoder offload, vLLM sleep, recorded OOM degradation ladder | Simulated OOM tests; VRAM peaks in eval |
| I9 | System RAM insufficient for sleeping VLM weights + offloaded encoders | M | M | Profiles declare a minimum RAM; entrypoint checks; ladder falls back to vLLM sleep level 2 (discard weights) at the cost of reload time | Boot check |
| I10 | Chosen GPU type is unavailable in datacenters that satisfy data residency | M | M | `deploy.py` queries availability, tries the allowed DC list in order, and prints alternatives (other GPU types in the same profile) | `deploy.py up --dry-run` |
| I11 | Forgotten running pod → cost | M | M | `deploy.py down`; optional idle watchdog; cost estimate printed at deploy | — |
| I12 | Gated HF model access not accepted → 403 at boot | H | L | `download_models.py` fails fast naming the exact HF page to accept; the profile may fall back to the non-gated alternative if configured | Boot log |
| I13 | vLLM / diffusers / torch version conflicts | H | M | Separate venvs (ADR-S02); pinned locks; the image build runs import smoke tests | CI image job |
| I14 | Runtime support lags for new models (e.g. an edit model's conditioning feature not in diffusers) | M | M | Pin the diffusers version that supports the needed classes; a headless ComfyUI backend is allowed only if a capability is missing; fallback model in the registry | Phase-6 spike |

## D. Development environment and delivery

| ID | Risk | L | I | Mitigation | Early signal / test |
|---|---|---|---|---|---|
| D1 | This dev environment has **no GPU and no Docker daemon**, and its network policy **blocks huggingface.co, docs.runpod.io, rest.runpod.io, download.blender.org** | H | M | Everything is testable on CPU with mocks. GPU-only paths are marked **UNVERIFIED-ON-GPU** until the pod smoke test covers them. License evidence is gathered from GitHub LICENSE files + search snippets of model cards, and re-checked at download time on the pod. RunPod field names come from RunPod's open-source docs/CLI repos, and `deploy.py` validates them against the live API with a first read-only call. | PROGRESS.md status column |
| D2 | Scope is very large; risk of shallow implementations | H | H | Walking skeleton first, then deepen stage by stage with acceptance tests per phase; "no stubs/TODOs" check in CI (`grep` gate + mock-only-in-tests rule) | Per-phase acceptance |
| D3 | Blender version drift between dev (CPU tests) and pod | M | L | Same Blender LTS tarball version everywhere; version asserted by build scripts | Blender tests print the version |
| D4 | Golden projects not representative of real firm documents | M | M | Build three golden projects spanning vector PDF, DXF and raster/phone-photo inputs; ask the firm for 2–3 anonymised real projects for acceptance (optional) | Open question Q-4 |
| D5 | Untrusted input files exploit parsers (zip bombs, malformed PDFs/DXF/IFC) | M | H | Magic-byte detection, limits, parsers in subprocesses with RLIMITs + timeouts, no macros, `defusedxml`, fuzz tests on parsers | Fuzz tests in CI (short budget) |
| D6 | Cost/time per section exceeds expectations (4K + best-of-N + QA + retries) | M | M | Profile-level budgets; measure in smoke test; affinity batching; cache; Lightning/distilled variants only after a QA A/B | Timings in `make eval` |
