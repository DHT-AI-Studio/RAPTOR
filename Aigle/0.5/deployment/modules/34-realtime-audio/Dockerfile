# Module 34 — real-time audio (WS /audio/stream on 8034, gRPC AudioService on 8044). Built on the
# shared media worker image (module 08) for its torch/torchaudio/PyAV/faster-whisper stack;
# silero-VAD runs on CPU, STT (faster-whisper) and TTS (Kokoro) on the GPU reserved in compose.
# Compose supplies BASE_IMAGE from RAPTOR_VERSION; the default below is only
# for a standalone build run outside compose. RT_BASE_IMAGE can select an
# older media-worker tag.
ARG BASE_IMAGE=raptor/media-worker:0.5
FROM ${BASE_IMAGE}

ARG SILERO_VAD_VERSION=6.2.3
WORKDIR /opt/realtime-audio

COPY requirements.txt requirements-tts.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps "silero-vad==${SILERO_VAD_VERSION}" && \
    pip install --no-cache-dir -r requirements-tts.txt

# RA-6: Python stubs for proto/raptor/audio/v1/audio.proto (generated/ is never committed)
COPY proto/ ./proto/
COPY scripts/gen_proto.py ./scripts/
RUN python scripts/gen_proto.py /opt/realtime-audio/generated

# Kokoro TTS weights (hexgrad/Kokoro-82M-v1.1-zh, ~0.7GB, public repo -- no token needed), baked
# into the image at build time instead of requiring a host-provisioned RT_TTS_KOKORO_DIR mount.
# kokoro.py's load() already forces HF_HUB_OFFLINE=1 (no runtime network); this keeps that promise
# true from a cold host too, instead of depending on someone manually staging the snapshot first.
COPY scripts/prefetch_kokoro.py ./scripts/
RUN python scripts/prefetch_kokoro.py --root /models/tts/kokoro

COPY app/ ./app/

ENV PYTHONPATH=/opt/realtime-audio:/opt/realtime-audio/generated
# 8034 HTTP + WS /audio/stream; 8044 gRPC AudioService (RT_GRPC_PORT)
EXPOSE 8034 8044
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8034"]
