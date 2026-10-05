#!/usr/bin/env bash
# Rebuild the accepted library on the Linux host that will perform inference.
set -euo pipefail
GOLOCO_OPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'The recorded native build and inference environment are Linux; use the intended Linux host.' >&2
  exit 2
fi
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-1.99.0}"
export RUSTFLAGS='-C target-cpu=native'
export CARGO_BUILD_JOBS=1
cargo build --release --locked --offline --manifest-path "$GOLOCO_OPT_DIR/rust/Cargo.toml"
printf '%s\n' "$GOLOCO_OPT_DIR/rust/target/release/libgoloco_forest.so"
