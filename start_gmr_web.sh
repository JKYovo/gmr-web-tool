#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${GMR_WEB_HOST:-127.0.0.1}"
PORT="${GMR_WEB_PORT:-7870}"
ENV_NAME="${GMR_CONDA_ENV:-gvhmr}"
PID_FILE="${ROOT_DIR}/runtime/gmr_web.pid"
LOG_FILE="${ROOT_DIR}/runtime/gmr_web.log"

mkdir -p "${ROOT_DIR}/runtime"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "GMR Web is already running."
  echo "URL: http://${HOST}:${PORT}/ui"
  exit 0
fi

cd "${ROOT_DIR}"
setsid env PYTHONUNBUFFERED=1 conda run -n "${ENV_NAME}" python -m gmr_web.server --host "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "GMR Web is starting."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "URL: http://${HOST}:${PORT}/ui"
