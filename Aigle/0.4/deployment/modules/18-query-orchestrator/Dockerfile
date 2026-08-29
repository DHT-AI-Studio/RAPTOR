FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# faiss-cpu's AVX512 auto-detection is unreliable on this deployment's CPU
# (virtualized, no real AVX512) -- it still tries to load the AVX512-compiled
# module and SIGILLs the first time it actually runs. Deleting the file makes
# faiss's own loader hit ImportError and cleanly fall back to AVX2 instead,
# which this CPU does support. Verified: faiss still works correctly with
# just this file removed (IndexFlatIP add/search both still pass).
RUN find /usr/local/lib -name '_swigfaiss_avx512*.so' -delete

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
