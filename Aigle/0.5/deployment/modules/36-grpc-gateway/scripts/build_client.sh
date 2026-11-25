#!/usr/bin/env bash
# Builds the raptor-grpc-client wheel: generates the stubs from proto/ into
# client/src/ (gitignored) and runs `pip wheel` on client/. Does not publish.
#   usage: scripts/build_client.sh [output-dir]      (default: client/dist)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/client/src"
OUT_DIR="${1:-$ROOT_DIR/client/dist}"

rm -rf "$SRC_DIR"
mkdir -p "$SRC_DIR" "$OUT_DIR"

python -m grpc_tools.protoc \
  -I "$ROOT_DIR/proto" \
  --python_out="$SRC_DIR" \
  --grpc_python_out="$SRC_DIR" \
  "$ROOT_DIR/proto/raptor/v1/raptor.proto"

# raptor/ stays an implicit namespace package; only raptor/v1 is a regular package.
touch "$SRC_DIR/raptor/v1/__init__.py"

python -m pip wheel --no-deps -w "$OUT_DIR" "$ROOT_DIR/client"
echo "Built raptor-grpc-client into $OUT_DIR"
