# CLAUDE.md: working notes for ArchRender

## Resume here
1. Read `docs/PROGRESS.md` (phase status, what is verified, what is UNVERIFIED-ON-GPU, next steps).
2. Read `docs/PLAN.md` for the current phase's scope and acceptance tests.
3. Design references: `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` (system ADRs),
   `docs/MODEL_SELECTION.md` (model ADRs with license evidence), `docs/RISKS.md`.

## Current state
Phase 0 approved 2026-09-30 (owner answers in docs/PLAN.md: Türkiye, Turkish, SAM licence accepted,
no archive/datasets for training, no ODA, gpu80). Phase 1 (walking skeleton) done 2026-09-30; see
PROGRESS.md for what is verified, the UNVERIFIED-ON-GPU list, deviations and deferred items.
Phase 2 (ingest & understanding, S0/S1) done on CPU 2026-09-30. Phase 3 (plan extraction, S2,
and Gate A, S3) done on CPU 2026-10-01: see PROGRESS.md for the measured plan table, misses, the
UNVERIFIED-ON-GPU list and deviations. Next: Phase 4 (scene, cameras, base render).

## Commands
- `make setup` (uv sync) · `make setup-blender` (official bpy 5.2.2 wheel, hash-locked, into
  `.venv-blender`; the dev/CI Blender runtime) · `make ui` (npm ci + build)
- `make lint typecheck test` · `make e2e` (live server + worker, CLI, Playwright UI) · `make eval`
- CI runs tests with `ARCHRENDER_TEST_STRICT=1`: a missing Blender runtime or UI build fails the run
  instead of skipping tests. Use it locally before claiming "green".
- `make schemas` (JSON Schema into `schemas/`; CI checks freshness) · `make license-audit` (writes
  `THIRD_PARTY_LICENSES.md`; CI checks freshness)
- `make dev-server` + `make dev-worker` for local runs; `archrender --help` for the CLI.
- `make docker-build-local DOCKER_EXTRA_CA=/root/.ccr/ca-bundle.crt` then `make docker-smoke`
  (the CA secret is needed only behind this container's TLS proxy; it never lands in a layer).
- `make deploy-dry-run` · `python deploy/runpod/deploy.py up|down|status|plan [--dry-run]`
- `python -m archrender.ops.deploy_manifest` regenerates `deploy/runpod/profiles.json` after
  profile/model changes (CI checks it).
- Releases: push a `v*` tag → `.github/workflows/release.yml` builds, smoke-tests, audits and
  publishes the GPU image to GHCR; the digest is in the release notes and `image.lock.json`.
- `python -m archrender.synth.corpus OUT --per-class N` writes a labelled synthetic corpus;
  `make train-classifier` retrains `configs/classifier/page_classifier_v1.json` (~20 min, OCR);
  `python -m archrender.understand.evaluate` runs the held-out S1 evaluation (also in `make eval`).
- `make eval` prints the S2 plan table (`archrender.plan.evaluate`; `--plan-per-source N`, 0 skips):
  per source type, real intake + S1 OCR, ground truth from the synthetic generator.
- `python -m archrender.plan.seg_train OUT --smoke|--sheets N` trains the plan segmentation U-Net
  (needs PyTorch: pod image); `archrender plan list|show|edit|resolve|confirm|accept|reject|approve`
  is Gate A from the CLI; the UI editor is `#/projects/<id>/plan`.
- `python -m archrender.ops.license_audit --lock deploy/vllm/requirements.lock` pre-audits the vLLM
  environment from PyPI metadata (CI does this); the release audits it again inside the image.
- System tools used in tests (Tesseract tur+eng, libheif, LibreDWG): CI installs them (LibreDWG
  via `deploy/libredwg/build.sh`, cached); strict mode fails if one is missing.

## Conventions (apply from Phase 1)
- Python 3.12 (`>=3.12,<3.14`), `uv` with a committed `uv.lock`; package `archrender` under
  `src/archrender/`. The Blender venv is Python 3.13 (bpy 5.2 requirement).
- pydantic v2 schemas in `archrender.core.schemas` are the single source of truth; export JSON Schema.
- Never `import bpy` in the app. Blender runs as `blender -b --factory-startup --python
  archrender_blender/build.py -- scene.json out/`. `archrender_blender/` is GPL-3.0-or-later.
- Every default goes through the assumption register; every extracted value is a `Fact` with
  `Provenance`.
- VLMs verify and label by default. Coordinate help from the VLM happens **only when needed** (named
  triggers) and only as hints snapped to measured evidence (`method="vlm_assisted"`, Gate A
  confirmation). It is never used for scale (ADR-S19).
- Every model role has primary / fallback / mock; tests run on CPU with mocks only.
- The license gate is enforced at download, load and in CI; never bypass it.
- Never weaken a threshold, special-case a fixture, or fake a result to get green; report misses.
- GPU-only code paths are labelled UNVERIFIED-ON-GPU in PROGRESS.md until the pod smoke test covers them.
- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `build:`, `ci:`, `refactor:`, `chore:`).
- Develop on branch `claude/ecstatic-ramanujan-0461me` unless told otherwise.

- Paths inside the image differ from the checkout: use `Settings.configs_dir` / `Settings.app_root()`,
  never `REPO_ROOT`, in code that runs in the container.
- Run-level user choices (e.g. `RunConfig.materials`) are applied *after* the cached stage they edit,
  so they invalidate only downstream stages.
- Untrusted input is parsed only through `ingest.sandbox.run_task`/`run_tool` (resource-limited
  child process, ADR-S20). A new external tool also needs: licences.yaml `subprocess_tools`,
  the Dockerfile, CI, and a strict-mode test (`tests.conftest.require_tool`).
- Changing an enum or schema that is exported (e.g. `ErrorCode`, `JobKind`) → run `make schemas`;
  CI checks freshness. Changing models/profiles → `python -m archrender.ops.deploy_manifest`.
- Model roles resolve through `ModelManager.get_with_fallback(role, stage)`: profile model →
  fallbacks → the role's mock, each step recorded as a degradation (never silent). Readiness only
  checks `ops.readiness.PIPELINE_ROLES`; add a role there when a stage starts using it.
- Classifier features are versioned by `understand.features.FEATURE_NAMES`; changing them requires
  retraining (loading refuses a mismatched model). So does changing what feeds them (OCR
  preprocessing, text/visual extraction): retrain and commit the new JSON with the change.
- Golden projects (`synth.golden`: G1 vector PDF + DXF + XLSX, G2 loft scan + phone photo + RCP,
  G3 IFC office + imperial PDF) are the Phase-2/3 acceptance (`test_golden_s1.py`,
  `test_golden_s2.py`); each plan document carries its plan → document transform for scoring.
  Evaluate OCR/CV changes on scans of several seeds and qualities (clean/medium/noisy, 200 and 300
  DPI), not on one page (`make eval` plan table, `tests/integration/test_plan_raster.py`).
- Plans: S2 writes draft plan versions; edits are JSON Patches making new versions (never mutate a
  version); runs pin `runs.plan_version`; approval needs no blocking issue (ADR-S22). Anything that
  changes how a plan is derived from the same pages must bump the S2 stage version
  (`plan/stage.py`), or cached extractions stay stale.
- VLM coordinates only through `plan/assist.py` (triggers → hint → snap to evidence → `vlm_assisted`
  + `PLAN_ASSIST_UNCONFIRMED`); never insert a hinted element without measured evidence
  (ADR-S19/S24). Ground-truth hint sources (`synth.hints`) are for eval/tests only.
- Validator defects for tests and eval live in `plan.defects` (one injector per defect).

## Environment notes (this cloud dev container)
- No GPU. A Docker daemon can be started (`nohup dockerd >/tmp/dockerd.log 2>&1 &`); Docker Hub and
  GitHub release downloads work.
- The egress policy blocks huggingface.co, docs.runpod.io, rest.runpod.io, api.runpod.io and
  download.blender.org. PyPI, npm, GitHub and raw.githubusercontent.com work. Web search works.
- Chromium for Playwright: `/opt/pw-browsers/chromium` (tests fall back to Playwright's own).
- npm behind the proxy: `NODE_EXTRA_CA_CERTS=/root/.ccr/ca-bundle.crt`.
