#!/usr/bin/env bash
# Regenerates Python gRPC stubs from proto/raptor/v1/raptor.proto into generated/.
# Used locally, in CI (validation only), and inside the Docker build.
# generated/ is gitignored — never commit its contents.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/generated}"

mkdir -p "$OUT_DIR/raptor/v1"
touch "$OUT_DIR/__init__.py" "$OUT_DIR/raptor/__init__.py" "$OUT_DIR/raptor/v1/__init__.py"

python -m grpc_tools.protoc \
  -I "$ROOT_DIR/proto" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$ROOT_DIR/proto/raptor/v1/raptor.proto"

echo "Generated stubs in $OUT_DIR"
