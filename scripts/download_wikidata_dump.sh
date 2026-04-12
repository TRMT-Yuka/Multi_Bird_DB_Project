#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DUMP_DIR="${DUMP_DIR:-$PROJECT_ROOT/data/raw/wikidata/dumps}"
DUMP_URL="${DUMP_URL:-https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2}"
CHECKSUM_URL="${CHECKSUM_URL:-${DUMP_URL}.md5}"
CONFIRM_LARGE_DOWNLOAD="${CONFIRM_LARGE_DOWNLOAD:-0}"

echo "Warning: this download may take several hours and requires a large amount of storage." >&2
echo "Expected file: latest-all.json.bz2 (about 94 GiB as of 2026-04)." >&2
echo "Download destination: $DUMP_DIR" >&2
echo "Make sure you have enough free space and time before continuing." >&2

if [ -t 0 ] && [ "$CONFIRM_LARGE_DOWNLOAD" != "1" ]; then
  read -r -p "Continue downloading the Wikidata dump? [y/N] " reply
  case "$reply" in
    [yY]|[yY][eE][sS]) ;;
    *)
      echo "Cancelled." >&2
      exit 1
      ;;
  esac
elif [ "$CONFIRM_LARGE_DOWNLOAD" != "1" ]; then
  echo "Set CONFIRM_LARGE_DOWNLOAD=1 to run this script non-interactively." >&2
  exit 1
fi

mkdir -p "$DUMP_DIR"
cd "$DUMP_DIR"

curl --fail --silent --show-error --location --remote-name --continue-at - "$DUMP_URL"

if curl --fail --silent --show-error --location --output latest-all.json.bz2.md5 "$CHECKSUM_URL"; then
  md5sum -c latest-all.json.bz2.md5
fi
