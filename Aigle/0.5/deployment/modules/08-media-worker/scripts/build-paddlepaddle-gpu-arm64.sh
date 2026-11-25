#!/usr/bin/env bash
# Builds a paddlepaddle-gpu wheel for aarch64 + NVIDIA Grace-Blackwell (GX10/
# DGX Spark, GB10, sm_121) from source. There is no official arm64 GPU build
# (PaddlePaddle's own docs list arm64 as GPU-unsupported) -- this follows an
# unofficial, community-reproduced recipe from
# https://github.com/PaddlePaddle/PaddleOCR/discussions/17328 (ximekon,
# 2026-04-06; cuDNN/NCCL/build-dir corrections from juliema1205, 2026-04-21).
# No PaddlePaddle maintainer endorsed this -- re-check that discussion if a
# step below starts failing after a PaddlePaddle version bump.
#
# Must run ON the real GX10/DGX Spark hardware (needs a live GPU + working
# nvcc toolchain for this exact arch; not buildable under QEMU emulation).
# Not invoked by `docker build` -- copy the resulting wheel into
# ../arm64-wheels/ yourself, which 08's Dockerfile picks up automatically on
# the next arm64 image build (falls back to CPU-only paddlepaddle if absent).
set -euo pipefail

PADDLE_VERSION="v3.3.0" # matches x86_64's pinned paddlepaddle-gpu==3.3.0
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/arm64-wheels"

echo "== Step 1: sanity checks =="
[ "$(uname -m)" = "aarch64" ] || { echo "not aarch64, aborting"; exit 1; }
nvcc --version
nvidia-smi --query-gpu=name,compute_cap --format=csv
python3 --version

echo "== Step 2: apt dependencies (build tools + cuDNN/NCCL for CUDA 13) =="
sudo apt-get update
sudo apt-get install -y \
    git cmake ninja-build \
    python3-dev python3-pip python3-venv \
    libopenblas-dev liblapack-dev \
    gfortran patchelf swig \
    wget curl unzip \
    libssl-dev zlib1g-dev \
    libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
    libnccl2 libnccl-dev

echo "== Step 3: CUDA 13.0+ toolkit present? =="
dpkg -l | grep cuda-toolkit || sudo apt-get install -y cuda-toolkit-13-0

echo "== Step 4-5: build venv =="
python3 -m venv "$HOME/paddle_build_env"
# shellcheck disable=SC1091
source "$HOME/paddle_build_env/bin/activate"
pip install --upgrade pip
pip install numpy protobuf cython wheel setuptools
# Paddle v3.3.0's own tools/gen_pybind11_stub.py calls pybind11-stubgen's 2.x
# run() signature (a `sub_dir` kwarg), but python/requirements.txt doesn't
# pin it -- an unpinned install resolves to 3.0.0+, which dropped that
# kwarg, and the wheel-packaging step fails at the very end of the build
# with "TypeError: run() got an unexpected keyword argument 'sub_dir'".
# Real upstream gap in this Paddle release, not an ARM64/CUDA issue.
pip install "pybind11-stubgen==2.5.5"

echo "== Step 6: clone PaddlePaddle =="
cd "$HOME"
[ -d Paddle ] || git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
git checkout "$PADDLE_VERSION"
git submodule update --init --recursive

echo "== Step 7: install packaging deps =="
pip install -r "$HOME/Paddle/python/requirements.txt"

echo "== Step 8: configure (the Eigen/ARM-NEON workaround is required) =="
mkdir -p "$HOME/Paddle/build"
cd "$HOME/Paddle/build"
cmake .. \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_GPU=ON \
    -DWITH_TESTING=OFF \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="12.1" \
    -DWITH_ARM=ON \
    -DWITH_AVX=OFF \
    -DWITH_MKL=OFF \
    -DWITH_MKLDNN=OFF \
    -DWITH_TENSORRT=OFF \
    -DCMAKE_CUDA_FLAGS="-U__ARM_NEON -DEIGEN_DONT_VECTORIZE=1" \
    -DPYTHON_EXECUTABLE="$(which python3)" \
    2>&1 | tee cmake_output.log

echo "== Step 9: build (~40min on a DGX Spark per the source thread) =="
ninja -j"$(nproc)" 2>&1 | tee build_output.log

echo "== Step 10: collect the wheel =="
mkdir -p "$OUT_DIR"
WHEEL="$(find "$HOME/Paddle/build/python/dist" -name '*.whl' | head -n1)"
[ -n "$WHEEL" ] || { echo "no wheel produced, check build_output.log"; exit 1; }
cp "$WHEEL" "$OUT_DIR/"
echo "copied $(basename "$WHEEL") to $OUT_DIR"
echo "next: rebuild 08's image on this host (arm64) to pick it up"
