# CLAUDE.md: working notes for ArchRender

## Resume here
1. Read `docs/PROGRESS.md` (phase status, what is verified, what is UNVERIFIED-ON-GPU, next steps).
2. Read `docs/PLAN.md` for the current phase's scope and acceptance tests.
3. Design references: `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` (system ADRs),
   `docs/MODEL_SELECTION.md` (model ADRs with license evidence), `docs/RISKS.md`.

## Current state
Phase 0 (research + design docs) is complete and **awaiting approval**. No code yet. Do not start
Phase 1 until the owner approves the Phase 0 docs.

## Conventions (apply from Phase 1)
- Python 3.11+, `uv` with a committed `uv.lock`; package `archrender` under `src/archrender/`.
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

## Environment notes (this cloud dev container)
- No GPU, no Docker daemon.
- The egress policy blocks huggingface.co, docs.runpod.io, rest.runpod.io and download.blender.org.
  PyPI and raw.githubusercontent.com work. Web search works.
