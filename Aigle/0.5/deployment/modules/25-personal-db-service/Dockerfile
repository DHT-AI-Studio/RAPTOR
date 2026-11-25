FROM python:3.11-slim

WORKDIR /service

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# sentence-transformers pulls in torch with no pin of its own -- PyPI's
# default wheel bundles a CUDA runtime newer than this deployment's host
# driver supports, so the bge-m3 embedder / bge-reranker silently run on
# CPU. Pin the same torch+cu128 build as 08-media-worker/17-hybrid-search
# instead -- confirmed working with real GPU memory usage on this host
# (09/10/11/12, which share module 08's image, via ./deploy.sh --status).
RUN pip install --no-cache-dir "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
