#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="0.0.0.0"
PORT="${GMR_WEB_PORT:-7870}"
LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [[ -z "${LAN_IP}" ]]; then
  LAN_IP="127.0.0.1"
fi

GMR_WEB_HOST="${HOST}" GMR_WEB_PORT="${PORT}" bash "${ROOT_DIR}/start_gmr_web.sh"
echo "LAN URL: http://${LAN_IP}:${PORT}/ui"

