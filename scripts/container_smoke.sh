#!/usr/bin/env bash
# Boot an ArchRender image on this machine with the cpu_test profile, wait for /readyz, then run
# scripts/smoke_test.py against it (upload → run → bundle). Used by CI, the release workflow and
# `make docker-smoke`. Exits non-zero with the container log on any failure.
#
#   scripts/container_smoke.sh archrender:local
set -Eeuo pipefail
IMAGE=${1:?usage: scripts/container_smoke.sh IMAGE}
PORT=${SMOKE_PORT:-18000}
READY_TIMEOUT_S=${SMOKE_READY_TIMEOUT_S:-600}
NAME="archrender-smoke-$$"
TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')
WS=$(mktemp -d)

cleanup() {
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "---- container log (last 120 lines) ----" >&2
    docker logs "$NAME" 2>&1 | tail -n 120 >&2 || true
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  exit $status
}
trap cleanup EXIT

docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:8000" \
  -e ARCHRENDER_PROFILE=cpu_test -e ARCHRENDER_ADMIN_TOKEN="$TOKEN" -e ARCHRENDER_COOKIE_SECURE=false \
  -v "$WS:/workspace" "$IMAGE" >/dev/null

deadline=$((SECONDS + READY_TIMEOUT_S))
code=000
while (( SECONDS < deadline )); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/readyz" || true)
  [[ "$code" == 200 ]] && break
  if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]]; then
    echo "container exited before becoming ready" >&2
    exit 1
  fi
  sleep 3
done
if [[ "$code" != 200 ]]; then
  echo "not ready after ${READY_TIMEOUT_S}s (last /readyz status $code)" >&2
  exit 1
fi
echo "ready after ${SECONDS}s"

uv run python scripts/smoke_test.py --url "http://127.0.0.1:${PORT}" --bootstrap-token "$TOKEN" \
  --width 320 --height 180 --views 2 --samples 16
