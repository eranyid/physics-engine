#!/usr/bin/env bash
# Build and run the C smoke test against libpe_ffi.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
cargo build -p pe-ffi --release
LIBDIR="$ROOT/target/release"
INCDIR="$ROOT/crates/pe-ffi/include"
OUT="$ROOT/target/pe_ffi_c_test"
cc -O2 -Wall -Wextra -I"$INCDIR" \
  "$ROOT/crates/pe-ffi/c_tests/test_pe_ffi.c" \
  -L"$LIBDIR" -lpe_ffi -lm -ldl -lpthread \
  -o "$OUT"
# Prefer rpath / LD_LIBRARY_PATH for the cdylib if linked dynamically.
export LD_LIBRARY_PATH="$LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$OUT"
# Header drift check when cbindgen is available.
if command -v cbindgen >/dev/null 2>&1; then
  TMP=$(mktemp)
  cbindgen --config "$ROOT/crates/pe-ffi/cbindgen.toml" --crate pe-ffi -o "$TMP"
  if ! diff -u "$INCDIR/pe_ffi.h" "$TMP"; then
    echo "::error::pe_ffi.h drifted from cbindgen output — regenerate and commit"
    rm -f "$TMP"
    exit 1
  fi
  rm -f "$TMP"
  echo "pe_ffi.h matches cbindgen"
fi
