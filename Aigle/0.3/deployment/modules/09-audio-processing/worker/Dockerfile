# Audio Processing Worker
# Aigle Platform v0.2
# GPU required for:
#   audio_recognizer   (whisperx ASR)
#   audio_diarization  (whisperx + pyannote speaker diarization)
#   audio_classifier   (panns_inference audio tagging)

# CUDA 12.8 is the minimum runtime that ships sm_120 (Blackwell, RTX 50-series) kernels.
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

# System dependencies (ffmpeg required for audio decoding)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsndfile1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# PyTorch built against CUDA 12.8 — required for RTX 50-series (sm_120).
# Must be installed BEFORE requirements.txt so the cu128 wheels win over any
# transitive PyPI torch resolution.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1 torchaudio==2.7.1

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Dispatcher entrypoint + all service code
COPY main.py /app/
COPY services/ /app/services/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
