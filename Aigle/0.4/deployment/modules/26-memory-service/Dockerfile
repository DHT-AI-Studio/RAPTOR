FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10026 memsvc \
    && useradd --uid 10026 --gid memsvc --no-create-home --shell /usr/sbin/nologin memsvc

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
RUN chown -R memsvc:memsvc /app

USER memsvc

EXPOSE 8026

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8026"]
