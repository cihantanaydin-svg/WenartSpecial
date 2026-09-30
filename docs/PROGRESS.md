# ArchRender: Progress

| Phase | Status | How verified | UNVERIFIED-ON-GPU | Next |
|---|---|---|---|---|
| 0 Research & design | **Done, approved by owner 2026-09-30** (answers recorded in PLAN.md) | Five parallel research passes (models ×3, RunPod, Blender/stack/licences). Evidence from LICENSE files on GitHub, RunPod's docs/OpenAPI source repo, diffusers source + wheels, the vLLM recipes repo, PyPI and Docker Hub metadata, and web-search snippets of HF cards (marked). | Everything GPU-related: VRAM/time budgets in PLAN.md are estimates | Phase 1 walking skeleton |
| 1 Walking skeleton | **Done 2026-09-30** | `make lint typecheck test e2e eval`, strict mode (no skips); CPU image built, booted and smoke-tested; deploy payloads validated against RunPod's OpenAPI v2 schemas; workflows pass actionlint. Details below | OptiX/CUDA device selection, GPU image build (CUDA base + Blender tarball + torch cu130), vLLM, live RunPod API. List below | Phase 2 ingest & understanding |
| 2 Ingest & understanding | Not started | — | — | — |
| 3 Plan extraction + Gate A | Not started | — | — | — |
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
