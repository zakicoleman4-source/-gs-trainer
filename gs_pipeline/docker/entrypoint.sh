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

# Single source of truth for the upload limit. The UI reads MAX_UPLOAD_GB to
# render the "Max N GB" hint; streamlit enforces STREAMLIT_SERVER_MAX_UPLOAD_SIZE
# (in MB). Derive the latter from the former here so the displayed and enforced
# limits can never drift. `set -u` is active, so default MAX_UPLOAD_GB if unset.
MAX_UPLOAD_GB="${MAX_UPLOAD_GB:-8}"
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE="$(( MAX_UPLOAD_GB * 1024 ))"
echo "[entrypoint] upload limit: ${MAX_UPLOAD_GB} GB (${STREAMLIT_SERVER_MAX_UPLOAD_SIZE} MB)"

# Surface the actual GPU we'll train on in the container logs (helps debug
# nvidia-container-toolkit / driver mismatches).
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
else
    echo "[entrypoint] WARNING: nvidia-smi not present; training will fail." >&2
fi

# The apt `supervisor` runs under the native python3.10 (#!/usr/bin/python3),
# whose system python3-setuptools still provides pkg_resources. No version
# gymnastics needed now that the whole image is python3.10.
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/gs_pipeline.conf
