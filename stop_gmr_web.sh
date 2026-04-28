#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${ROOT_DIR}/runtime/gmr_web.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "GMR Web is not running."
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" 2>/dev/null; then
  kill -TERM "-${PID}" 2>/dev/null || kill -TERM "${PID}" 2>/dev/null || true
  sleep 1
fi

rm -f "${PID_FILE}"
echo "GMR Web stopped."

