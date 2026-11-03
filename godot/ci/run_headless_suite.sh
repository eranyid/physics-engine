#!/usr/bin/env bash
# Build pe-godot, stage the cdylib, run all headless Godot CI scenes.
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

echo "Importing Godot project (registers GDExtension)..." >&2
IMPORT=$("$GODOT" --headless --path "$ROOT/godot/example" --import 2>&1 || true)
echo "$IMPORT" >&2
if ! echo "$IMPORT" | grep -q "Initialize godot-rust"; then
  echo "::error::GDExtension did not initialize during --import (missing 'Initialize godot-rust')" >&2
  ls -la "$ROOT/godot/example/bin" >&2 || true
  cat "$ROOT/godot/example/pe_godot.gdextension" >&2 || true
  exit 3
fi

run_scene() {
  local scene="$1"
  local ok_token="$2"
  echo "Running $scene..." >&2
  set +e
  OUT=$("$GODOT" --headless --path "$ROOT/godot/example" "$scene" 2>&1)
  STATUS=$?
  set -e
  echo "$OUT"
  if ! echo "$OUT" | grep -q "$ok_token"; then
    echo "::error::Godot scene $scene did not print $ok_token (exit=$STATUS)" >&2
    exit 1
  fi
  test "$STATUS" -eq 0
}

run_scene "res://ci_drop_test.tscn" "GODOT_CI_OK"
run_scene "res://ci_shapes_test.tscn" "GODOT_CI_SHAPES_OK"
run_scene "res://ci_joint_test.tscn" "GODOT_CI_JOINT_OK"
run_scene "res://ci_raycast_test.tscn" "GODOT_CI_RAYCAST_OK"
run_scene "res://ci_contact_signal_test.tscn" "GODOT_CI_CONTACT_OK"

echo "All Godot headless CI scenes passed." >&2
