# Breeze TTS 2 server for Module 34's optional `breeze-tts-2` TTS backend (compose profile "breeze").
#
# LICENCE: the inference code (github.com/breezeblue-ai/breeze-tts) is Apache-2.0; the Breeze TTS 2
# weights, derivative models and self-hosted outputs are for research and non-commercial use only
# (BreezeBlue Research and Non-Commercial License). Module 34 uses this server only with
# RT_TTS_BREEZE_ENABLED=true.
#
# It needs torch 2.9 / transformers 4.57 / numpy 2, which do not fit the media-worker base, hence a
# separate image. The weights are mounted read-only at /models/breeze-tts-2 (no download).
# The upstream API loads the model with eager attention, so flash-attn (a long source build) is
# not installed.
#
# Base is nvidia/cuda, not pytorch/pytorch -- pytorch/pytorch's own Docker Hub images are
# amd64-only (no arm64 manifest at all, confirmed via `docker manifest inspect`), which fails
# outright on GX10. nvidia/cuda is multi-arch and already proven on both amd64/arm64 by module 08.
FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

# Set by buildx/BuildKit to the native arch being built for (amd64/arm64) -- used below to pick
# a torch CUDA build with kernels for GX10's GB10 GPU (Grace-Blackwell, sm_121).
ARG TARGETARCH

# Same forward-compat-library removal as module 08's Dockerfile: when the host driver is older
# than this image's CUDA 13.1, the NVIDIA Container Toolkit puts the image's own (datacenter-only)
# compat libs ahead of the host driver's, breaking CUDA on GeForce-class hosts (error 804). The
# toolkit re-adds the directory on every start, so removing the ld.so.conf entry alone isn't
# enough. Hosts need driver >= 580.
RUN rm -rf /usr/local/cuda-*/compat /usr/local/cuda/compat \
 && rm -f /etc/ld.so.conf.d/*compat* \
 && ldconfig

ARG BREEZE_TTS_REPO=https://github.com/breezeblue-ai/breeze-tts.git
ARG BREEZE_TTS_COMMIT=6eba901d1845353802652c708a4740fdfc5baa75

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git libsndfile1 \
      python3 python3-pip python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

# venv, not a bare python3 install -- Ubuntu 24.04 marks the system Python as externally
# managed (PEP 668) and refuses top-level pip installs outright.
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /opt/breeze-infer
# Pinned commit; test tooling is left out. torch/torchaudio no longer come from the base image
# (that was pytorch/pytorch's job) -- installed explicitly below, by architecture.
RUN git init -q . && git remote add origin "${BREEZE_TTS_REPO}" \
    && git fetch -q --depth 1 origin "${BREEZE_TTS_COMMIT}" && git checkout -q FETCH_HEAD \
    && grep -v -E "^(torch|torchaudio|pytest|ruff)([=<> ]|$)" requirements.txt > /tmp/requirements.txt

# torch==2.9.1 / torchaudio==2.9.1, pinned by breeze-tts's own requirements.txt (not ours to
# change -- a third-party repo pinned at a fixed commit).
#
# arm64 (GX10, Grace-Blackwell GB10, compute capability sm_121): cu128 ships no kernels at all
# for this GPU (confirmed on 23-guardrail-service -- cu126/cu128 only cover sm_80/sm_90).
# cu130's torch build does carry sm_120/sm_120a cubins (checked directly: `strings` on the
# extracted libtorch_cuda.so) -- the same arch coverage as the cu130 build module 08's arm64
# branch already uses and confirmed live on GX10 (torch.cuda.is_available()==True, no
# kernel-coverage warning); sm_120's embedded PTX JIT-compiles forward onto sm_121 (same major
# Blackwell generation, NVIDIA's documented minor-version forward compatibility). Not yet
# live-verified on GX10 itself for this exact image -- do that before trusting it in production.
#
# torchaudio has no CUDA-tagged aarch64 wheel on ANY download.pytorch.org index (checked both
# cu128 and cu130 directly) -- harmless here: reading qwen-tts's own source, the only thing it
# imports from torchaudio is `torchaudio.compliance.kaldi` (fbank feature extraction), a
# CPU-only DSP function, never a GPU op.
RUN set -eux; \
    if [ "$TARGETARCH" = "arm64" ]; then \
      pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.9.1 torchaudio==2.9.1; \
    else \
      pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1 torchaudio==2.9.1; \
    fi \
    && pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 7860
CMD ["python", "-m", "breeze_infer.api", "/models/breeze-tts-2", "--host", "0.0.0.0", "--port", "7860"]
