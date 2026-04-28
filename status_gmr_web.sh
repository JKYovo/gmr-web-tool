#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${ROOT_DIR}/runtime/gmr_web.pid"
LOG_FILE="${ROOT_DIR}/runtime/gmr_web.log"
PORT="${GMR_WEB_PORT:-7870}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "GMR Web is running."
  echo "PID: $(cat "${PID_FILE}")"
  echo "URL: http://127.0.0.1:${PORT}/ui"
else
  echo "GMR Web is not running."
fi

if [[ -f "${LOG_FILE}" ]]; then
  echo "Log: ${LOG_FILE}"
fi

