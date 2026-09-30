SHELL := /bin/bash
UV ?= uv
PY := .venv/bin/python
BLENDER_VENV := .venv-blender
BLENDER_VERSION := 5.2.2

.PHONY: help setup setup-blender ui lint typecheck test test-fast e2e eval schemas license-audit \
        dev-server dev-worker docker-build docker-build-local deploy-dry-run clean

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
	@echo "deploy-dry-run  print every RunPod API payload without calling the API"

setup:
	$(UV) sync

setup-blender:
	$(UV) venv -q -p 3.13 $(BLENDER_VENV)
	$(UV) pip install -q -p $(BLENDER_VENV)/bin/python bpy==$(BLENDER_VERSION)
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
	$(UV) run python scripts/license_audit.py

dev-server:
	ARCHRENDER_COOKIE_SECURE=false ARCHRENDER_BLENDER_MODE=module $(UV) run uvicorn archrender.api.app:create_app --factory --port 8000

dev-worker:
	ARCHRENDER_BLENDER_MODE=module $(UV) run python -m archrender.pipeline.worker --queue all

docker-build:
	docker build -f deploy/Dockerfile -t archrender:dev .

docker-build-local:
	docker build -f deploy/Dockerfile --build-arg BLENDER_SOURCE=wheel --build-arg BASE_IMAGE=ubuntu:24.04 -t archrender:local .

deploy-dry-run:
	$(PY) deploy/runpod/deploy.py up --dry-run --env-file deploy/runpod/.env.example

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache var/ ui/dist
