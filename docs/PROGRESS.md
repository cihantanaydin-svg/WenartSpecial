# ArchRender: Progress

| Phase | Status | How verified | UNVERIFIED-ON-GPU | Next |
|---|---|---|---|---|
| 0 Research & design | **Done, awaiting owner approval** | Five parallel research passes (models ×3, RunPod, Blender/stack/licences). Evidence from LICENSE files on GitHub, RunPod's docs/OpenAPI source repo, diffusers source + wheels, the vLLM recipes repo, PyPI and Docker Hub metadata, and web-search snippets of HF cards (marked). | Everything GPU-related: VRAM/time budgets in PLAN.md are estimates | Owner approval + answers to Q-1…Q-8 (PLAN.md), then Phase 1 walking skeleton |
| 1 Walking skeleton | Not started | — | — | — |
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
  `scripts/pin_models.py` on a machine with Hub access (Phase 1/2), and re-checked at download.
- The Blender 5.2.2 tarball SHA256 is taken from a third-party mirror. CI compares it with the
  official `.sha256` before first use.
- No GPU and no Docker daemon here (the Docker CLI is present, the daemon is not running).
