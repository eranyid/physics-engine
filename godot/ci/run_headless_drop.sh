#!/usr/bin/env bash
# Build pe-godot, stage the cdylib into godot/example/bin/, run the headless drop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

GODOT="${GODOT_BIN:-}"
if [[ -z "$GODOT" ]]; then
  GODOT="$("$ROOT/godot/ci/download_godot.sh")"
fi

cargo build -p pe-godot
mkdir -p "$ROOT/godot/example/bin"
cp -f "$ROOT/target/debug/libpe_godot.so" "$ROOT/godot/example/bin/libpe_godot.so"
test -s "$ROOT/godot/example/bin/libpe_godot.so"
export LD_LIBRARY_PATH="$ROOT/godot/example/bin:${LD_LIBRARY_PATH:-}"

# Fresh checkouts have no .godot/; --import registers the GDExtension once.
echo "Importing Godot project (registers GDExtension)..." >&2
IMPORT=$("$GODOT" --headless --path "$ROOT/godot/example" --import 2>&1 || true)
echo "$IMPORT" >&2
if ! echo "$IMPORT" | grep -q "Initialize godot-rust"; then
  echo "::error::GDExtension did not initialize during --import (missing 'Initialize godot-rust')" >&2
  ls -la "$ROOT/godot/example/bin" >&2 || true
  cat "$ROOT/godot/example/pe_godot.gdextension" >&2 || true
  exit 3
fi

echo "Running ci_drop_test.tscn..." >&2
set +e
OUT=$("$GODOT" --headless --path "$ROOT/godot/example" res://ci_drop_test.tscn 2>&1)
STATUS=$?
set -e
echo "$OUT"
if ! echo "$OUT" | grep -q "GODOT_CI_OK"; then
  echo "::error::Godot drop test did not print GODOT_CI_OK (exit=$STATUS)" >&2
  exit 1
fi
test "$STATUS" -eq 0
