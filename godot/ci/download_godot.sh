#!/usr/bin/env bash
# Download a pinned Godot 4.3-stable Linux x86_64 binary and verify SHA-512
# against the official release SUMS file (also pinned below).
set -euo pipefail

GODOT_VERSION="4.3-stable"
ZIP_NAME="Godot_v${GODOT_VERSION}_linux.x86_64.zip"
# Official SHA-512 from:
# https://github.com/godotengine/godot/releases/download/4.3-stable/SHA512-SUMS.txt
EXPECTED_SHA512="fd52bb4ba8acc30ca5accd1c566d470ad7282f891ccc0995dfafabcf92bcf76280ce182bf9d80ebd885f3ed2165d01e1fc3f2928436b15498dfbd98656c2a45a"
DOWNLOAD_URL="https://github.com/godotengine/godot/releases/download/${GODOT_VERSION}/${ZIP_NAME}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CACHE_DIR="${GODOT_CACHE_DIR:-$ROOT/.godot-ci}"
BIN="$CACHE_DIR/Godot_v${GODOT_VERSION}_linux.x86_64"

mkdir -p "$CACHE_DIR"
if [[ -x "$BIN" ]]; then
  echo "Using cached Godot at $BIN" >&2
  echo "$BIN"
  exit 0
fi

TMP_ZIP="$CACHE_DIR/${ZIP_NAME}"
echo "Downloading $DOWNLOAD_URL" >&2
curl -fsSL -o "$TMP_ZIP" "$DOWNLOAD_URL"
echo "${EXPECTED_SHA512}  ${TMP_ZIP}" | sha512sum -c - >&2
unzip -o -d "$CACHE_DIR" "$TMP_ZIP" >&2
chmod +x "$BIN"
rm -f "$TMP_ZIP"
echo "Godot ready: $BIN" >&2
echo "$BIN"
