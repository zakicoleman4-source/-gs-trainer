#!/usr/bin/env bash
# gs_pipeline container entrypoint.
#
# Ensures the data directories exist (compose mounts them but if a user runs
# the image directly with `docker run` we still want a clean tree), then
# hands off to supervisord which runs both the Streamlit UI and the trainer
# watcher.

set -euo pipefail

for d in /data/inbox /data/outbox /data/logs /data/work /data/config; do
    mkdir -p "$d"
done

for d in /data/inbox /data/outbox /data/logs /data/work; do
    if ! touch "$d/.write_test" 2>/dev/null; then
        echo "[entrypoint] ERROR: $d is not writable — check volume mount permissions" >&2
        exit 1
    fi
    rm -f "$d/.write_test"
done

# Surface the actual GPU we'll train on in the container logs (helps debug
# nvidia-container-toolkit / driver mismatches).
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
else
    echo "[entrypoint] WARNING: nvidia-smi not present; training will fail." >&2
fi

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/gs_pipeline.conf
