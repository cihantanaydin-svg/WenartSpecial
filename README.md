# ArchRender

Self-hosted system that ingests an architecture project's documents (plans, CAD/BIM, schedules,
photos, mood boards, briefs) and produces **dimensionally faithful, photoreal renders of a selected
section**. All AI runs on open-weight models on a single RunPod GPU pod, and deployment is one
command.

Status: **Phase 2 (ingest & understanding) complete on CPU.** Uploads (PDF, DXF/DWG, images incl.
HEIC, DOCX/XLSX/PPTX, ZIP) are parsed in a sandbox; every page is classified with a calibrated
confidence, read (text layer, DXF text or Turkish/English OCR), and title block, scale, north arrow,
door/window/room tags and schedules are extracted, with schedule rows linked to the plan tags.
Uncertain results go to a review queue. The render path from Phase 1 still runs end to end: plan
(mock) → scene → real Blender 5.2 Cycles render → refine (mock) → relative QA → review gates →
bundle, through the API, the CLI and the web UI. Real models arrive phase by phase; see
[docs/PROGRESS.md](docs/PROGRESS.md). Anything produced with mock models is labelled as such in the
UI, the QA report and the run manifest.

## Quickstart (local, CPU)
```bash
make setup setup-blender ui        # uv env, Blender 5.2.2 (bpy wheel), React UI
# system tools (apt): tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng libheif-examples
#   libheif-plugin-libde265; LibreDWG: sudo deploy/libredwg/build.sh
make test                          # unit + integration tests (real Blender renders)

export ARCHRENDER_ADMIN_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')
make dev-server &                  # API + UI on http://127.0.0.1:8000
make dev-worker &                  # background worker (CPU + GPU queues)

export ARCHRENDER_URL=http://127.0.0.1:8000
uv run archrender bootstrap --token "$ARCHRENDER_ADMIN_TOKEN"   # one-shot: prints the admin API key
export ARCHRENDER_API_KEY=ark_…                                 # the key printed above
uv run archrender project create "Daire 3+1"                    # prints the project id
uv run archrender upload <project id> "Kat Planı.dxf" "Doğrama Listesi.xlsx"
uv run archrender pages <project id>              # page classes, confidence, scale (S1)
uv run archrender schedules <project id>          # schedule rows and their plan tags
uv run archrender review <project id>             # items waiting for a person
uv run archrender run <project id> --views 2 --width 640 --height 360 --wait
uv run archrender gate approve <run id> D_final   # runs with mock models always stop at Gate D
uv run archrender download <run id>               # bundle: renders, GLB, QA report, manifests
```
Or open http://127.0.0.1:8000, log in with the API key and do the same in the browser. `make e2e`
runs this flow automatically through the API, the CLI and a headless browser.

## Deploy on RunPod
```bash
cp deploy/runpod/.env.example deploy/runpod/.env   # fill in the RunPod key, image digest, tokens
python deploy/runpod/deploy.py up --dry-run        # every payload, secrets masked, no API calls
python deploy/runpod/deploy.py up                  # volume, secrets, template, pod, wait for /readyz
python deploy/runpod/deploy.py down                # terminate the pod, keep the volume
```
The image comes from the release workflow (tag `v*`); its digest is in the release notes.

Start with:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how it works
- [docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md): which models, and why they are licence-clean
- [docs/DECISIONS.md](docs/DECISIONS.md): system design decisions
- [docs/PLAN.md](docs/PLAN.md): phases, acceptance tests, budgets, **open questions**
- [docs/RISKS.md](docs/RISKS.md): risk register
- [docs/PROGRESS.md](docs/PROGRESS.md): status
