FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# StreamVideoSegment's transcode formats (MP4_H264/MPEG_TS) shell out to
# FFmpeg -- RAW_FILE_RANGE does not need it (see media_stream_service.py).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY 36-grpc-gateway/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# proto/ + scripts/ copied first so `generated/` can be produced at build time —
# it is gitignored and never committed, so the image must regenerate it itself.
COPY 36-grpc-gateway/proto/ proto/
COPY 36-grpc-gateway/scripts/ scripts/
RUN bash scripts/gen_proto.sh

COPY 36-grpc-gateway/config.py 36-grpc-gateway/server.py ./
COPY 36-grpc-gateway/interceptors/ interceptors/
COPY 36-grpc-gateway/services/ services/

# Build context is deployment/modules/ so we can reuse module 13's shared
# service classes instead of duplicating them. video_search_service.py is the
# one exception: module 13 no longer ships it, so it lives in vendor/.
#
# video_search_service.py is needed for SearchByPerson/StreamSearchByPerson;
# personal_db_service.py + search_service.py (URL resolution) + storage_service.py
# are its own dependencies. core/config.py is copied only because
# storage_service.py/personal_db_service.py do `from app.core.config import
# Settings` at module scope as a type hint — its Settings class is never
# instantiated here (it has required fields this gateway doesn't set); we
# pass our own grpc-gateway Settings object instead, which is duck-type
# compatible (same attribute names).
COPY 13-api-services/app/__init__.py app/__init__.py
COPY 13-api-services/app/core/__init__.py app/core/__init__.py
COPY 13-api-services/app/core/config.py app/core/config.py
COPY 13-api-services/app/services/__init__.py app/services/__init__.py
COPY 13-api-services/app/services/search_service.py app/services/search_service.py
COPY 13-api-services/app/services/storage_service.py app/services/storage_service.py
COPY 13-api-services/app/services/personal_db_service.py app/services/personal_db_service.py
COPY 36-grpc-gateway/vendor/app/services/video_search_service.py app/services/video_search_service.py
# kafka_service.py backs SubmitJob — same Kafka-producer class REST's
# /processing/process-file uses, so a job submitted over gRPC lands in the
# exact same topics/orchestrator Redis keys as one submitted over REST.
COPY 13-api-services/app/services/kafka_service.py app/services/kafka_service.py

EXPOSE 8825 9825

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import grpc; from grpc_health.v1 import health_pb2, health_pb2_grpc; \
    channel = grpc.insecure_channel('localhost:8825'); \
    stub = health_pb2_grpc.HealthStub(channel); \
    resp = stub.Check(health_pb2.HealthCheckRequest(service=''), timeout=5); \
    exit(0 if resp.status == health_pb2.HealthCheckResponse.SERVING else 1)"

CMD ["python", "server.py"]
