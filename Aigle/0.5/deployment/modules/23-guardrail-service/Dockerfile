FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set by buildx/BuildKit to the native arch being built for (amd64/arm64) --
# used below to branch the GX10/ARM64 torch CUDA-build selection.
ARG TARGETARCH

WORKDIR /app

# Prevent interactive package configuration during Docker image builds.
ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04's own python3.11 is 3.11.0rc1, which lacks
# sys.get_int_max_str_digits (torch >= 2.11 needs it at import); use the
# deadsnakes PPA's final 3.11 release instead.
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
    && apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    python3.11 \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSfL --retry 5 --retry-all-errors --retry-delay 2 https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py && rm /tmp/get-pip.py

# Build context is deployment/modules/ (see docker-compose.yml), so every
# path below is relative to that directory.
COPY 23-guardrail-service/requirements.txt .
# PyPI's torch 2.14 wheels are built for CUDA 13 (driver >= 580); install the
# same versions from PyTorch's CUDA 12.6 index so driver >= 525 (e.g. 550) works.
# requirements.txt's torch/torchvision pins are then already satisfied.
#
# arm64 (GX10/DGX Spark, Grace-Blackwell GB10, compute capability sm_121) needs
# a newer CUDA build than cu126 -- confirmed live: torch==2.14.1+cu126 installs
# and reports torch.cuda.is_available()==True on GX10 (the GPU is detected),
# but its own warning at import says this build's kernels only cover
# sm_80/sm_90, not sm_121 -- any real CUDA op (incl. the weight
# dequantization/conversion step on model load) then fails. cu130 is the
# minimum PyTorch itself names as covering this GPU; matches what 08's arm64
# branch already resolves to via plain PyPI, confirmed working live
# (torch.cuda.is_available()==True, NVIDIA GB10, no kernel-coverage warning).
RUN set -eux; \
    if [ "$TARGETARCH" = "arm64" ]; then \
      python3.11 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.14.1 torchvision==0.29.1; \
    else \
      python3.11 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.14.1 torchvision==0.29.1; \
    fi \
    && python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY 23-guardrail-service/app/ ./app/
# Shared CF-4 audit-event library (VIE01-234), importable as `raptor_audit`.
COPY common/raptor_audit/ ./raptor_audit/

CMD ["python3.11", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8026"]
