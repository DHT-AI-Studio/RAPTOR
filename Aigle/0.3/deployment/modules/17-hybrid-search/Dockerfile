FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# cu128 wheels are required for RTX 50-series (sm_120 / Blackwell).
# pip>=24.1 is required: python:3.10-slim ships an older pip that fails to
# install PyTorch's cu128 wheels due to PEP-625 name-normalization strictness
# (e.g. rejects "typing_extensions" wheels because the parsed name is
# "typing-extensions"). pip 24.1 made this check lenient again.
RUN pip install --no-cache-dir --upgrade "pip>=24.1" && \
    pip install --no-cache-dir "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

COPY ./app app/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]