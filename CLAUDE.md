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
Next: Phase 2 (ingest & understanding).

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

## Environment notes (this cloud dev container)
- No GPU. A Docker daemon can be started (`nohup dockerd >/tmp/dockerd.log 2>&1 &`); Docker Hub and
  GitHub release downloads work.
- The egress policy blocks huggingface.co, docs.runpod.io, rest.runpod.io, api.runpod.io and
  download.blender.org. PyPI, npm, GitHub and raw.githubusercontent.com work. Web search works.
- Chromium for Playwright: `/opt/pw-browsers/chromium` (tests fall back to Playwright's own).
- npm behind the proxy: `NODE_EXTRA_CA_CERTS=/root/.ccr/ca-bundle.crt`.
