#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${ROOT_DIR}/xeno_canto_api_key.env" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/xeno_canto_api_key.env"
fi

cd "${ROOT_DIR}"

# make fetch-xeno-canto-recording-json
make extract-xeno-canto-recording-ids
make fetch-xeno-canto-audio
