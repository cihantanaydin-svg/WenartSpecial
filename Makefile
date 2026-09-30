SHELL := /bin/bash
comma := ,
UV ?= uv
PY := .venv/bin/python
BLENDER_VENV := .venv-blender
BLENDER_VERSION := 5.2.2

.PHONY: help setup setup-blender ui lint typecheck test test-fast e2e eval schemas license-audit \
        dev-server dev-worker docker-build docker-build-local docker-smoke deploy-dry-run clean

help:
	@echo "setup           install app + dev deps (uv), Playwright uses /opt/pw-browsers"
	@echo "setup-blender   create $(BLENDER_VENV) with the official bpy $(BLENDER_VERSION) wheel (dev/CI Blender runtime)"
	@echo "ui              build the React UI into ui/dist"
	@echo "lint/typecheck  ruff + mypy"
	@echo "test            full CPU test suite (Blender tests run when a Blender runtime exists)"
	@echo "e2e             end-to-end through API, CLI and UI (live server + worker)"
	@echo "eval            print the evaluation table (CPU subset)"
	@echo "schemas         export JSON Schema for all pydantic schemas"
	@echo "license-audit   check dependency/model/asset licences, write THIRD_PARTY_LICENSES.md"
	@echo "docker-build-local / docker-smoke  CPU image (bpy wheel) and a boot + end-to-end smoke test"
	@echo "deploy-dry-run  print every RunPod API payload without calling the API"

setup:
	$(UV) sync

setup-blender:
	$(UV) venv -q -p 3.13 $(BLENDER_VENV)
	$(UV) pip install -q -p $(BLENDER_VENV)/bin/python --require-hashes -r deploy/blender/requirements.lock
	$(BLENDER_VENV)/bin/python -c "import bpy; print('bpy', bpy.app.version_string)"

ui:
	cd ui && npm ci && npm run build

lint:
	$(UV) run ruff check src tests scripts deploy
	$(UV) run ruff format --check src tests scripts deploy

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest -m "not e2e"

test-fast:
	$(UV) run pytest -m "not e2e and not blender" -x -q

e2e: ui
	$(UV) run pytest -m e2e

eval:
	$(UV) run python scripts/eval.py

schemas:
	$(UV) run python scripts/export_schemas.py

license-audit:
	$(UV) run python -m archrender.ops.license_audit --prod-only --subprocess-python $(BLENDER_VENV)/bin/python

dev-server:
	ARCHRENDER_COOKIE_SECURE=false ARCHRENDER_BLENDER_MODE=module $(UV) run uvicorn archrender.api.app:create_app --factory --port 8000

dev-worker:
	ARCHRENDER_BLENDER_MODE=module $(UV) run python -m archrender.pipeline.worker --queue all

docker-build:
	docker build -f deploy/Dockerfile -t archrender:dev .

# CPU-only image with the bpy wheel (no CUDA base, no torch). DOCKER_EXTRA_CA=/path/ca.pem adds a
# corporate/proxy CA for network steps only (BuildKit secret; not stored in the image).
DOCKER_EXTRA_CA ?=
UBUNTU_IMAGE := ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3
docker-build-local:
	docker build -f deploy/Dockerfile --build-arg BLENDER_SOURCE=wheel --build-arg APP_EXTRAS= \
	  --build-arg BASE_IMAGE=$(UBUNTU_IMAGE) --build-arg GIT_COMMIT=$$(git rev-parse --short HEAD) \
	  $(if $(DOCKER_EXTRA_CA),--secret id=extra_ca$(comma)src=$(DOCKER_EXTRA_CA),) -t archrender:local .

docker-smoke:
	scripts/container_smoke.sh archrender:local

# Uses deploy/runpod/.env when present; otherwise the example with placeholder credentials
# (secret values are always masked in the output).
ifneq ($(wildcard deploy/runpod/.env),)
DRY_RUN_ENV := --env-file deploy/runpod/.env
else
DRY_RUN_ENV := --env-file deploy/runpod/.env.example
DRY_RUN_VARS := GHCR_USERNAME=example GHCR_TOKEN=placeholder HF_TOKEN=placeholder
endif
deploy-dry-run:
	$(DRY_RUN_VARS) $(UV) run python deploy/runpod/deploy.py up --dry-run $(DRY_RUN_ENV)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache var/ ui/dist
