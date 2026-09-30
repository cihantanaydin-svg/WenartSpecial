#!/usr/bin/env bash
# ArchRender container entrypoint (idempotent; safe on every restart).
#   preflight → /workspace tree → DB restore → model download → migrations → supervisor
#   → self-test → /readyz flips when the self-test marker exists.
set -Eeuo pipefail

log() { printf '[entrypoint %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
PY=/opt/venv/bin/python
export ARCHRENDER_PROFILE="${ARCHRENDER_PROFILE:-gpu80}"
export ARCHRENDER_DATA_DIR="${ARCHRENDER_DATA_DIR:-/workspace}"
export ARCHRENDER_DB_PATH="${ARCHRENDER_DB_PATH:-/var/lib/archrender/db/archrender.sqlite}"
export HF_HOME="${HF_HOME:-/workspace/cache/hf}"
export ARCHRENDER_BLENDER_MODE="${ARCHRENDER_BLENDER_MODE:-$(cat /etc/archrender-blender-mode 2>/dev/null || echo binary)}"
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 TRANSFORMERS_NO_ADVISORY_WARNINGS=1

log "ArchRender ${ARCHRENDER_GIT_COMMIT:-dev} profile=${ARCHRENDER_PROFILE}"
"$PY" -m archrender.ops.preflight

mkdir -p "$ARCHRENDER_DATA_DIR"/{models,assets,projects,db/snapshots,cache/cuda,cache/optix,logs} "$(dirname "$ARCHRENDER_DB_PATH")"

# Database: local disk (correct locking), restored from the volume replica (ADR-S03).
if [[ ! -s "$ARCHRENDER_DB_PATH" ]]; then
  if command -v litestream >/dev/null && [[ -d "$ARCHRENDER_DATA_DIR/db/replica" ]]; then
    log "restoring database from litestream replica"
    litestream restore -config /opt/archrender/deploy/litestream.yml -if-replica-exists "$ARCHRENDER_DB_PATH" \
      || log "litestream restore failed; trying the latest snapshot"
  fi
  if [[ ! -s "$ARCHRENDER_DB_PATH" ]]; then
    latest=$(ls -1 "$ARCHRENDER_DATA_DIR"/db/snapshots/archrender-*.sqlite 2>/dev/null | tail -n1 || true)
    if [[ -n "${latest:-}" ]]; then
      log "restoring database from snapshot $latest"
      cp "$latest" "$ARCHRENDER_DB_PATH"
    fi
  fi
fi

# Weights baked into the image (optional build arg) are linked instead of downloaded.
if [[ -d /opt/baked/models ]]; then
  cp -rsn /opt/baked/models/. "$ARCHRENDER_DATA_DIR/models/" 2>/dev/null || true
fi

log "downloading models for profile ${ARCHRENDER_PROFILE} (licence-gated, SHA-256 verified, resumable)"
if ! "$PY" -m archrender.ops.models download --profile "$ARCHRENDER_PROFILE"; then
  # keep booting: roles whose model is missing degrade (and /readyz lists them) instead of a restart loop
  log "MODEL DOWNLOAD INCOMPLETE: see 'MODEL NOT INSTALLED' lines above; affected roles degrade"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # no Hub access at runtime after the boot download

"$PY" -m archrender.ops.preflight --migrate >/dev/null
mkdir -p /etc/archrender
"$PY" -m archrender.ops.supervisor --out /etc/archrender/supervisord.conf

supervisord -c /etc/archrender/supervisord.conf &
SUP=$!
shutdown() {
  log "stopping (final DB snapshot)"
  "$PY" -m archrender.ops.snapshot || true
  kill -TERM "$SUP" 2>/dev/null || true
  wait "$SUP" || true
  exit 0
}
trap shutdown TERM INT

log "running self-test (Blender device probe + tiny render; warms the kernel cache)"
if "$PY" -m archrender.ops.selftest; then
  log "self-test passed; /readyz will report ready"
else
  log "SELF-TEST FAILED: the API stays up for diagnosis but /readyz reports not ready (see above)"
fi

( while sleep 3600; do "$PY" -m archrender.ops.snapshot >/dev/null 2>&1 || true; done ) &

wait "$SUP"
