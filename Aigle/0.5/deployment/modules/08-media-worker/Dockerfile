# CUDA 13.1 (upgraded from 12.8): onnxruntime-gpu wheels with working sm_120
# kernels need Python >=3.11, and the CUDA12.8 PyPI wheel has no sm_120
# kernels at all (officially only PTX for it, which JIT-fails on Blackwell --
# see microsoft/onnxruntime#26177). Ubuntu 24.04 ships Python 3.12.
FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

# Set by buildx/BuildKit to the native arch being built for (amd64/arm64) --
# used below to branch the GX10/ARM64 fixes (torch/paddle/onnxruntime index,
# decord install) without needing a second Dockerfile.
ARG TARGETARCH

# Remove the base image's CUDA forward-compatibility libraries (libcuda from driver 590).
# When the host driver is older than the image's CUDA, the NVIDIA Container Toolkit puts
# them ahead of the host driver at container start. Forward compatibility only works on
# datacenter GPUs: on a GeForce host (e.g. RTX 4090, driver 580 / CUDA 13.0) every CUDA call
# then fails with error 804 -- torch.cuda.is_available() is False and onnxruntime falls back
# to CPU. Removing only the ld.so.conf entry is not enough, since the toolkit re-adds the
# directory on every start. Without them the host driver's libcuda is used, and CUDA 13.x
# minor-version compatibility runs 13.1 code on a 13.0 driver. Hosts need driver >= 580.
RUN rm -rf /usr/local/cuda-*/compat /usr/local/cuda/compat \
 && rm -f /etc/ld.so.conf.d/*compat* \
 && ldconfig

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System packages. libcudnn8 dropped -- not carried by the CUDA13 apt repo;
# see the cuDNN symlink step below for how ctranslate2 still gets its
# cuDNN8-named libs.
RUN apt-get update -qq && \
    apt-get install -y \
      python3 python3-pip python3-dev python3-venv \
      libgl1 libglib2.0-0 libsm6 libxext6 \
      ffmpeg libsndfile1 \
      poppler-utils tesseract-ocr \
      curl wget patchelf git build-essential && \
    rm -rf /var/lib/apt/lists/*

# venv, not a bare python3 install -- Ubuntu 24.04 marks the system Python as
# externally managed (PEP 668) and refuses top-level pip installs outright.
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# PyTorch GPU (cu128) — required for RTX 50-series (sm_120 / Blackwell).
# arm64 (GX10/SBSA): download.pytorch.org never published an aarch64 wheel,
# and (checked directly) pypi.nvidia.com doesn't host torch/torchaudio/
# torchvision at all. 2.7.1's aarch64 PyPI wheel is CPU-only (its
# nvidia-cuda-*/triton deps are gated to platform_machine=="x86_64"); 2.11+
# drops that gating and pulls in real CUDA 13 deps on arm64 too -- but 2.11.0
# itself pins nvidia-cudnn-cu13==9.19.0.56, which has no aarch64 wheel (only
# an x86_64/win one), so that exact version fails to resolve on arm64.
# torch==2.14.1 is the newest release whose full aarch64 dependency chain
# (nvidia-cudnn/cusparselt/nccl/nvshmem-cu13, triton) was confirmed to
# actually have aarch64 wheels by checking each one's PyPI files directly,
# not assumed from torch's own metadata alone. torchvision==0.29.2
# (requires torch>=2.14.0) and torchaudio==2.11.1 (requires torch>=2.11) are
# both compatible and both confirmed to publish an aarch64 wheel. Plain
# PyPI, no special index needed on arm64.
RUN set -eux; \
    if [ "$TARGETARCH" = "arm64" ]; then \
      pip install --no-cache-dir \
        torch==2.14.1 \
        torchaudio==2.11.1 \
        torchvision==0.29.2; \
    else \
      pip install --no-cache-dir \
        torch==2.7.1 \
        torchaudio==2.7.1 \
        torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128; \
    fi

# arm64 only: torchaudio 2.11.1 above removed torchaudio.AudioMetaData/info/
# load/list_audio_backends entirely, which whisperx==3.3.1's pinned
# pyannote.audio==3.3.2 still calls -- see torchaudio_backend_shim.py for the
# full story and the soundfile-based shim itself. Appended directly onto
# torchaudio's own installed __init__.py, not shipped as a sitecustomize.py
# -- tried that first, but this base image's own
# /usr/lib/python3.12/sitecustomize.py sits earlier on sys.path and shadows
# anything placed in site-packages (confirmed live: `import sitecustomize`
# resolved to the system one, ours never ran). Appending onto torchaudio's
# own __init__.py instead runs on every `import torchaudio`, from anywhere,
# regardless of sys.path order. amd64 keeps the real torchaudio file
# untouched.
COPY 08-media-worker/torchaudio_backend_shim.py /tmp/torchaudio_backend_shim.py
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      cat /tmp/torchaudio_backend_shim.py >> /venv/lib/python3.12/site-packages/torchaudio/__init__.py; \
    fi

# PaddlePaddle GPU (cu129) — required for sm_120 (Blackwell / RTX 50-series).
# paddlepaddle-gpu==3.3.0 is the first official release shipping sm_120 kernels;
# CUDA 12.9 is the recommended wheel per PaddlePaddle docs (cu130 also works).
# Wheel is ~3.5 GB, needs extended socket timeout.
# arm64 (GX10/SBSA): PaddlePaddle's own docs list arm64 as GPU-unsupported,
# and no official PyPI aarch64 GPU wheel exists. If
# scripts/build-paddlepaddle-gpu-arm64.sh's output wheel has been copied into
# arm64-wheels/ (built on real GX10 hardware -- see that script for
# provenance, it's an unofficial community recipe, not an NVIDIA/PaddlePaddle
# build), install it; otherwise fall back to the official CPU-only build.
COPY 08-media-worker/arm64-wheels/ /tmp/arm64-wheels/
RUN set -eux; \
    if [ "$TARGETARCH" = "arm64" ]; then \
      WHEEL="$(find /tmp/arm64-wheels -maxdepth 1 -iname 'paddlepaddle_gpu*.whl' | head -n1)"; \
      if [ -n "$WHEEL" ]; then \
        pip install --no-cache-dir "$WHEEL" && echo "$WHEEL" > /tmp/arm64-wheels/.selected; \
      else \
        pip install --no-cache-dir --timeout 600 paddlepaddle==3.2.2; \
      fi; \
    else \
      pip install --no-cache-dir --timeout 600 paddlepaddle-gpu==3.3.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ \
        --extra-index-url https://pypi.org/simple; \
    fi

# PaddleOCR (separate step to avoid topological sort resolver bug -- installed
# before requirements.txt, not after, so requirements.txt's own pins are what
# wins for any package both of them touch; previously this ran the other way
# around and paddleocr[all]'s resolver could silently override a requirements.txt
# pin, since the two were never resolved together in the first place)
RUN pip install --no-cache-dir "paddleocr[all]"

# paddleocr[all] pulls in its own (CPU-only) paddlepaddle as a transitive
# dependency, silently overriding the from-source GPU wheel installed above
# on arm64 -- a real gotcha reported independently by two people building
# this exact recipe (see scripts/build-paddlepaddle-gpu-arm64.sh's source
# discussion). Re-force it back if it was used.
RUN if [ -f /tmp/arm64-wheels/.selected ]; then \
      pip install --no-cache-dir --force-reinstall --no-deps "$(cat /tmp/arm64-wheels/.selected)"; \
    fi

# EasyOCR (baked in at build time to avoid runtime downloads). Explicitly
# installed here rather than assumed to come transitively from paddleocr[all]'s
# extras -- it turns out that resolution is order-sensitive (confirmed on a
# real build: with requirements.txt now installed after paddleocr[all]
# instead of before, pip's resolver stopped pulling easyocr in at all,
# `import easyocr` failed with ModuleNotFoundError). Installing it directly
# removes the dependency on resolver behavior entirely.
RUN pip install --no-cache-dir easyocr
RUN python3 -c "import easyocr; easyocr.Reader(['ch_tra', 'en'], gpu=False)"

# PaddleOCR's official models (baked in at build time to avoid runtime
# downloads). paddleocr[all] above does NOT bundle these -- PaddleX (its
# backend) auto-downloads them lazily on first real use, which hung
# indefinitely here: its "checking connectivity to the model hosters" step
# tries several candidate hosts (HuggingFace Hub, ModelScope, AIStudio, BOS)
# in turn before ever reaching the actual file, and never completed -- not a
# reachability problem with the file host itself (confirmed live: a direct
# `curl` to its real BOS URL from the same host returns 200 in under a
# second). PaddleX's own _get_model_local_path() checks
# ~/.paddlex/official_models/<name>/ for an existing directory *before* it
# ever runs that hoster-probe code, so placing the files there at build time
# (bypassing PaddleX's download path entirely, same as EasyOCR's own
# pre-fetch above) sidesteps the hang rather than needing to root-cause it.
# PP-LCNet_x1_0_textline_ori is PaddleOCR's text-line orientation classifier,
# used internally on every OCR call regardless of OCR_DET_MODEL/OCR_REC_MODEL
# (module 11's video_ocr_frame env vars) -- always fetched, not optional.
RUN set -eux; \
    BASE=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0; \
    mkdir -p /root/.paddlex/official_models; \
    for name in PP-LCNet_x1_0_textline_ori PP-OCRv5_mobile_det PP-OCRv5_mobile_rec; do \
      curl -fL --retry 3 -o "/tmp/${name}_infer.tar" "${BASE}/${name}_infer.tar"; \
      tar -xf "/tmp/${name}_infer.tar" -C /root/.paddlex/official_models/; \
      mv "/root/.paddlex/official_models/${name}_infer" "/root/.paddlex/official_models/${name}"; \
      rm "/tmp/${name}_infer.tar"; \
    done

# PANNS metadata (baked in at build time)
RUN mkdir -p /root/panns_data && \
    curl -L -o /root/panns_data/class_labels_indices.csv \
      https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv

# Remaining dependencies -- kept as the LAST pip step (not right after
# paddlepaddle-gpu, where it used to sit) so that a requirements.txt-only
# change (e.g. adding a package) only invalidates the Docker layer cache from
# here on: this install + patchelf + the cuDNN symlinks below + the final
# COPY of service code. paddleocr[all]/easyocr/panns above don't depend on
# anything in requirements.txt, so they stay cached across a plain dependency
# bump instead of re-running their (slow, network-bound) model downloads.
COPY 08-media-worker/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# decord (module 10/11 video frame extraction) — amd64 only. The project is
# unmaintained and never published an aarch64 wheel. Both call sites already
# use it in CPU-decode mode (ctx=cpu(0)), so arm64 uses an OpenCV-based
# drop-in instead (video_reader_compat.py in that service's own directory) --
# not a GPU/perf regression, since there was no GPU decode to lose.
RUN if [ "$TARGETARCH" != "arm64" ]; then \
      pip install --no-cache-dir decord==0.6.0; \
    fi

# onnxruntime-gpu + insightface (Module 11 face inference on the same GPU
# nodes). insightface's onnx dep pulls ml_dtypes, which can resolve to a
# numpy>=2.0-only release depending on the day this image is built --
# install-then-downgrade-numpy doesn't fix that (ml_dtypes' compiled
# extension is already ABI-bound to NumPy 2.0's C-API). Pin numpy as a
# constraint during install instead so pip's resolver backtracks to a
# NumPy-1.x-compatible ml_dtypes/onnx from the start.
# arm64 (GX10/SBSA): PyPI never shipped an aarch64+CUDA onnxruntime-gpu wheel
# until 1.31.1 (2026-10-26) -- confirmed by extracting that wheel and finding
# a real libonnxruntime_providers_cuda.so inside, not a CPU-only build. Pin
# arm64 to 1.31.1 specifically.
# amd64: 1.27.0+ switched onnxruntime-gpu's CUDA EP to require CUDA 13's
# runtime (its own "cuda"/"cudnn" extras moved from nvidia-cuda-runtime-cu12
# to a cu13 build); a host whose driver tops out at CUDA 12.x (confirmed on
# one of our GPU hosts: driver 550.144.03, CUDA Driver Version 12.4) fails
# cudaSetDevice with "CUDA driver version is insufficient for CUDA runtime
# version" and silently falls back to CPU. Pin to 1.26.0, the last version
# whose CUDA EP still targets CUDA 12, and install it WITH its cuda/cudnn
# extras (not --no-deps) so it brings its own self-contained CUDA 12 runtime
# the same way torch's cu128 wheel above does -- this makes it independent
# of this image's own `FROM nvidia/cuda:13.1.1-...` base and of the host's
# system CUDA install entirely, same reasoning as the cuDNN symlink step
# below for ctranslate2/torch.
RUN echo "numpy==1.26.4" > /tmp/numpy-constraint.txt && \
    pip install --no-cache-dir -c /tmp/numpy-constraint.txt insightface==2.0 && \
    rm -f /tmp/numpy-constraint.txt && \
    pip uninstall -y onnxruntime && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu==1.31.1; \
    else \
      pip install --no-cache-dir --force-reinstall "onnxruntime-gpu[cuda,cudnn]==1.26.0"; \
    fi

# patchelf ctranslate2 (removes execstack warning from whisperx) -- must come
# after requirements.txt: ctranslate2 is a transitive dependency of whisperx.
RUN CT2_LIB=$(find / -name "libctranslate2*.so*" 2>/dev/null | head -n 1) && \
    [ -n "$CT2_LIB" ] && patchelf --clear-execstack "$CT2_LIB" || true

# cuDNN v8-named symlinks (ctranslate2 4.4.x / pyannote.audio expect
# libcudnn_ops_infer.so.8 etc.), pointed at the pip nvidia-cudnn-cu12/cu13
# package's own lib dir -- NOT the base image's system cuDNN. torch/paddle
# preload their pip cuDNN build's sublibraries by absolute path (bypassing
# ldconfig); ctranslate2 has no RPATH and resolves libcudnn_ops_infer.so.8
# purely via ldconfig/LD_LIBRARY_PATH. Symlinking into system cuDNN instead
# mixes two different cuDNN 9.x builds in one process and fails at inference
# time with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH (import succeeds; the
# failure only shows up on a real forward pass, e.g. WhisperX transcription).
RUN CUDNN_LIB=/venv/lib/python3.12/site-packages/nvidia/cudnn/lib && \
    [ -d "$CUDNN_LIB" ] || { echo "ERROR: pip nvidia-cudnn-cu12/cu13 lib dir not found" && exit 1; } && \
    cd "$CUDNN_LIB" && \
    ln -sf libcudnn_ops.so.9 libcudnn_ops_infer.so.8 && \
    ln -sf libcudnn_cnn.so.9 libcudnn_cnn_infer.so.8 && \
    ln -sf libcudnn_adv.so.9 libcudnn_adv_infer.so.8 && \
    ln -sf libcudnn.so.9     libcudnn.so.8
# onnxruntime-gpu's CUDA EP .so is in the same boat as ctranslate2 above (no
# RPATH, pure ldconfig/LD_LIBRARY_PATH lookup) -- it needs nvidia-cublas-cu12
# specifically, which onnxruntime-gpu's own "cuda" extra does NOT declare
# (confirmed: pip metadata lists cuda-nvrtc/cuda-runtime/cufft/curand, no
# cublas) even though the compiled .so links against libcublasLt.so.12.
# torch's cu128 install already pulls nvidia-cublas-cu12 as its own
# dependency (same reasoning as cudnn above), so the package is present --
# it's just missing from the library search path. Add it, plus the other
# "cuda" extra components, so nothing else surfaces one-at-a-time later.
# amd64 only: arm64's torch build pulls a cu13-era cublas, not cu12, so this
# directory never exists there and the check must not run on that arch (it
# did, briefly, and broke every arm64 build with no amd64-specific bug of
# its own -- confirmed live on a GX10 host).
RUN if [ "$TARGETARCH" != "arm64" ]; then \
      CUBLAS_LIB=/venv/lib/python3.12/site-packages/nvidia/cublas/lib && \
      [ -d "$CUBLAS_LIB" ] || { echo "ERROR: pip nvidia-cublas-cu12 lib dir not found" && exit 1; }; \
    fi
ENV LD_LIBRARY_PATH="/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:\
/venv/lib/python3.12/site-packages/nvidia/cublas/lib:\
/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:\
/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:\
/venv/lib/python3.12/site-packages/nvidia/cufft/lib:\
/venv/lib/python3.12/site-packages/nvidia/curand/lib:${LD_LIBRARY_PATH}"

# Copy all service code (build context = deployment/modules/)
COPY 09-audio-processing/worker/services/ /app/services/
COPY 10-image-processing/worker/services/ /app/services/
COPY 11-video-processing/worker/services/ /app/services/
COPY 12-document-processing/worker/services/ /app/services/
COPY 08-media-worker/main.py /app/main.py
# CM-5 / VIE01-408: shared by all four modules' moderation code to report a flagged
# category to Module 23's review store. On /app (PYTHONPATH below), not under
# /app/services/<service>/ -- importable regardless of which service_dir main.py prepends.
COPY common/moderation_flag_reporter.py /app/moderation_flag_reporter.py

ENV PYTHONPATH=/app

CMD ["python3", "main.py"]
