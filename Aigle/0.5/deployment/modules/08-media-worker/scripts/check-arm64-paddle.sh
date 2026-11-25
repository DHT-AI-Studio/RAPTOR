#!/usr/bin/env bash
# Runs before build.py's docker build step for module 08. No-op on amd64, and
# a no-op on arm64 once a paddlepaddle_gpu*.whl is staged in arm64-wheels/
# (see build-paddlepaddle-gpu-arm64.sh). Only prompts when about to silently
# fall back to CPU-only PaddlePaddle on a fresh arm64 host -- the thing this
# is for is making sure nobody forgets that step, not blocking anyone who
# already did it or who is fine with CPU for now.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[ "$(uname -m)" = "aarch64" ] || exit 0
compgen -G "$MODULE_DIR/arm64-wheels/paddlepaddle_gpu*.whl" > /dev/null 2>&1 && exit 0

echo "No paddlepaddle-gpu wheel staged in arm64-wheels/ for this arm64 host."
echo "Without it, PaddleOCR (modules 10/11/12) runs on CPU only -- still works, just slower."
# No TTY (CI/non-interactive): read fails immediately -- default to proceeding,
# same as the Dockerfile's own automatic fallback, never block an unattended run.
read -r -p "Build 08 with the CPU fallback now? [Y/n] " answer || answer="y"
case "${answer:-y}" in
  [nN]*)
    echo "Aborting -- run scripts/build-paddlepaddle-gpu-arm64.sh, copy the wheel into" \
         "arm64-wheels/, then re-run build.py 08."
    exit 1
    ;;
  *) exit 0 ;;
esac
