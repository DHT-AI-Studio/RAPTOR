FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10033 privacysvc \
    && useradd --uid 10033 --gid privacysvc --no-create-home --shell /usr/sbin/nologin privacysvc

# Build context is deployment/modules/ (see docker-compose.yml), so every
# path below is relative to that directory.
COPY 33-privacy-compliance-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 33-privacy-compliance-service/app/ ./app/
# Shared CF-4 audit-event library (VIE01-234), importable as `raptor_audit`.
COPY common/raptor_audit/ ./raptor_audit/
RUN chown -R privacysvc:privacysvc /app

USER privacysvc

EXPOSE 8033

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8033"]
