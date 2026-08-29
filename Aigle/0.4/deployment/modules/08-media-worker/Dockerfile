# CUDA 12.8 is the minimum runtime that ships sm_120 (Blackwell, RTX 50-series) kernels.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System packages
# libcudnn8: cuDNN 8 runtime needed by ctranslate2 4.4.x (compiled for cuDNN 8, not 9)
RUN apt-get update -qq && \
    apt-get install -y \
      python3.10 python3-pip python3.10-dev \
      libgl1 libglib2.0-0 libsm6 libxext6 \
      ffmpeg libsndfile1 \
      poppler-utils tesseract-ocr \
      curl wget patchelf git \
      libcudnn8 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# PyTorch GPU (cu128) — required for RTX 50-series (sm_120 / Blackwell).
RUN pip install --no-cache-dir \
      torch==2.7.1 \
      torchaudio==2.7.1 \
      torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cu128

# PaddlePaddle GPU (cu129) — required for sm_120 (Blackwell / RTX 50-series).
# paddlepaddle-gpu==3.3.0 is the first official release shipping sm_120 kernels;
# CUDA 12.9 is the recommended wheel per PaddlePaddle docs (cu130 also works).
# Wheel is ~3.5 GB, needs extended socket timeout.
RUN pip install --no-cache-dir --timeout 600 paddlepaddle-gpu==3.3.0 \
      -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ \
      --extra-index-url https://pypi.org/simple

# PaddleOCR (separate step to avoid topological sort resolver bug -- installed
# before requirements.txt, not after, so requirements.txt's own pins are what
# wins for any package both of them touch; previously this ran the other way
# around and paddleocr[all]'s resolver could silently override a requirements.txt
# pin, since the two were never resolved together in the first place)
RUN pip install --no-cache-dir "paddleocr[all]"

# EasyOCR (baked in at build time to avoid runtime downloads). Explicitly
# installed here rather than assumed to come transitively from paddleocr[all]'s
# extras -- it turns out that resolution is order-sensitive (confirmed on a
# real build: with requirements.txt now installed after paddleocr[all]
# instead of before, pip's resolver stopped pulling easyocr in at all,
# `import easyocr` failed with ModuleNotFoundError). Installing it directly
# removes the dependency on resolver behavior entirely.
RUN pip install --no-cache-dir easyocr
RUN python3 -c "import easyocr; easyocr.Reader(['ch_tra', 'en'], gpu=False)"

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

# patchelf ctranslate2 (removes execstack warning from whisperx) -- must come
# after requirements.txt: ctranslate2 is a transitive dependency of whisperx.
RUN CT2_LIB=$(find / -name "libctranslate2*.so*" 2>/dev/null | head -n 1) && \
    [ -n "$CT2_LIB" ] && patchelf --clear-execstack "$CT2_LIB" || true

# cuDNN v8 symlinks (pyannote.audio expects libcudnn_ops_infer.so.8 etc.)
# nvidia/cuda:12.8.0-cudnn-runtime installs cuDNN 9 into /usr/lib/x86_64-linux-gnu/,
# so find by filename rather than by the old nvidia/cudnn/lib directory path.
RUN CUDNN_LIB=$(find /usr/lib -name "libcudnn_ops.so.9" 2>/dev/null | head -n 1 | xargs -r dirname) && \
    if [ -n "$CUDNN_LIB" ]; then \
      cd "$CUDNN_LIB" && \
      [ -f libcudnn_ops_infer.so.8 ] || ln -sf libcudnn_ops.so.9  libcudnn_ops_infer.so.8 ; \
      [ -f libcudnn_cnn_infer.so.8 ] || ln -sf libcudnn_cnn.so.9  libcudnn_cnn_infer.so.8 ; \
      [ -f libcudnn_adv_infer.so.8 ] || ln -sf libcudnn_adv.so.9  libcudnn_adv_infer.so.8 ; \
      [ -f libcudnn.so.8 ]           || ln -sf libcudnn.so.9       libcudnn.so.8 ; \
    else echo "ERROR: cuDNN 9 libs not found, build aborted" && exit 1 ; fi

# Copy all service code (build context = deployment/modules/)
COPY 09-audio-processing/worker/services/ /app/services/
COPY 10-image-processing/worker/services/ /app/services/
COPY 11-video-processing/worker/services/ /app/services/
COPY 12-document-processing/worker/services/ /app/services/
COPY 08-media-worker/main.py /app/main.py

ENV PYTHONPATH=/app

CMD ["python", "main.py"]
