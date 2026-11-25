# MTR-7 / VIE01-306: GPU-only thumbnail extraction worker.
# python:3.10-slim + apt's ffmpeg package (not the heavier nvidia/cuda base
# image module 11's PaddleOCR/InternVL workers use) -- confirmed live
# against this project's own GPU node that `ffmpeg -hwaccels` lists `cuda`
# and a real -hwaccel cuda decode succeeds, as long as
# NVIDIA_DRIVER_CAPABILITIES=video,compute,utility is set (see this
# service's docker-compose.yml block). This worker only runs ffmpeg/
# ffprobe -- no PyTorch/PaddleOCR, so the full CUDA runtime image's extra
# weight buys nothing here.
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8034

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8034"]
