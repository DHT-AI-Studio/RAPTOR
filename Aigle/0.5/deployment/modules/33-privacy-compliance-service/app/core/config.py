"""Privacy Compliance Service configuration (pydantic-settings, env prefix GD_)."""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GD_", case_sensitive=False)

    # Shared secret module 13 (and nothing else) presents as X-Internal-Token
    # when calling this service, and this service in turn presents to modules
    # 04/25/26's own /internal/gdpr/user/{user_id} endpoints. Not a JWT --
    # these are service-to-service calls with no end-user session behind them.
    internal_token: str

    # DPO notification on PARTIAL_FAILURE
    dpo_email: str = ""
    # GD-9: Article 30 RoPA controller name (GD_CONTROLLER_NAME env var)
    controller_name: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_from: str = ""
    smtp_user: str = ""
    smtp_password: str = ""

    # Per-store fan-out timing
    erasure_timeout_seconds: float = 300.0
    erasure_max_attempts: int = 3
    erasure_retry_delay_seconds: float = 60.0

    # Downstream services this orchestrator calls directly. Container-
    # internal ports (8000/8000/8026) -- NOT each service's host-mapped
    # port (8025 for module 25 etc.), which only matters from outside Docker.
    module_04_url: str = "http://raptor-asset-management:8000"
    module_25_url: str = "http://raptor-personal-db-service:8000"
    module_26_url: str = "http://raptor-memory-service:8026"

    # PostgreSQL (Module 03) -- this codebase uses one database per service
    # (asset_management, benchmark, personal_db, guardrails, ...), not a
    # single shared one. gdpr_erasure_jobs lives in its own `privacy_compliance`
    # database, matching that convention.
    postgres_host: str = "raptor-postgres"
    postgres_port: int = 5432
    postgres_user: str = "raptor"
    postgres_password: str
    postgres_db: str = "privacy_compliance"

    # Module 03's `benchmark` database, `benchmark_runs.branch_id` column
    # (== submitter's JWT sub -- see 001_init.sql) -- the only real,
    # per-user-scoped table this ticket's "Module 03" deletion bullet maps
    # to. Its other named tables (violations, assets, consent_history) don't
    # exist in this codebase (see MTR/GD-1 implementation notes) -- omitted
    # entirely rather than faked.
    benchmark_postgres_db: str = "benchmark"

    # Redis (Module 02) -- direct SCAN+DEL for the user's cache keys, and the
    # gdpr:erased_users set Module 25's Kafka consumer checks.
    redis_host: str = "raptor-redis-standalone"
    redis_port: int = 6379
    redis_password: str = ""
    erased_users_ttl_seconds: int = 2592000  # 30 days

    # Kafka (Module 05) -- tombstone publish
    kafka_bootstrap_servers: str = "kafka-broker1:9092,kafka-broker2:9092,kafka-broker3:9092"
    kafka_erasure_topic: str = "gdpr-erasure-requests"

    # Keycloak (Module 06) -- disable-then-hard-delete via the Admin API;
    # also GET .../users/{id} for the export's identity/keycloak_profile.json.
    keycloak_url: str = "http://raptor-keycloak:8080"
    keycloak_realm: str = "dhtsolution"
    keycloak_admin_username: str
    keycloak_admin_password: str

    # GD-2 / VIE01-247: SeaweedFS (S3-compatible), same endpoint/credentials
    # Module 04 already uses -- but a DIFFERENT, dedicated bucket, not
    # Module 04's own `lakefs` bucket. That bucket's object layout is
    # managed by LakeFS's own commit/branch/GC machinery; the export ZIP is
    # an ephemeral, unversioned artifact that has no business inside it.
    # Created on first use if missing, same as Module 04's _ensure_bucket.
    s3_endpoint: str = "http://seaweedfs-s3:8333"
    # Presigned URLs are generated against s3_endpoint (only reachable inside
    # the raptor Docker network) then rewritten onto this public host/scheme
    # before being returned -- same swap Module 04's own
    # generate_presigned_url does, for the same reason: whoever downloads the
    # export isn't necessarily on the Docker network.
    s3_public_url: str = ""
    s3_bucket: str = "raptor-gdpr-exports"
    aws_access_key: str = ""
    aws_secret_key: str = ""

    # 72h matches the ZIP's own presigned-URL validity (the ticket's stated
    # TTL) -- past this, both the S3 object and the URL are meaningless.
    export_ttl_hours: float = 72.0
    export_max_zip_bytes: int = 500 * 1024 * 1024  # 500 MB, ticket's split threshold
    # How often the background sweep checks for and deletes expired export
    # ZIPs still sitting in SeaweedFS past their expires_at.
    export_cleanup_interval_seconds: float = 3600.0

    # GD-3 / VIE01-248: retention engine. guardrail_violations lives in its
    # own `guardrails` database (module 23), same Postgres instance/
    # credentials as everything else here -- same convention as
    # benchmark_postgres_db above.
    guardrails_postgres_db: str = "guardrails"
    # Ticket's own spec: "Background scheduler runs daily at 02:00 UTC".
    # Configurable rather than hardcoded so a stuck run can be rescheduled
    # without a code change, but this default is the ticket's literal ask.
    retention_hour_utc: int = 2
    retention_minute_utc: int = 0

    # Applied to the processing-request topics' retention.ms via the Kafka
    # Admin API on startup, idempotently (describe current value first, only
    # ALTER if different). Empty/0 disables this -- no topic-config changes
    # are attempted. No existing AdminClient usage anywhere else in this
    # codebase to follow; this is a new pattern.
    kafka_retention_ms: int = 0

    # GD-4 / VIE01-249: DSAR. Ticket's own spec: synchronous for normal
    # accounts, an async job for accounts with >10k indexed items, and a
    # response within 10 seconds either way. The sync budget is kept under
    # 10s (not equal to it) so Module 13's own proxy hop -- whose
    # GATEWAY request_timeout also defaults to 10s -- still fits inside it;
    # a sync assembly that overruns this is handed off to a job rather
    # than left to time out at the gateway.
    dsar_async_threshold: int = 10000
    dsar_sync_timeout_seconds: float = 8.0
    # A completed DSAR job's stored report (it carries the user's email) is
    # cleared past this, by the same hourly sweep that expires export ZIPs
    # -- gdpr_erasure_jobs rows themselves are a 3-year legal hold (GD-3),
    # which a copy of the report has no reason to share.
    dsar_result_ttl_hours: float = 72.0
    # The DSAR's `rights` block points the data subject at Module 13's
    # public routes, mounted at /api/{RAPTOR_API_VERSION}/gdpr/... -- the
    # same version every other route in this deployment uses (see GD-3's
    # note in Module 13's gdpr.py on the ticket set's literal "0.5").
    api_version: str = "0.5"

    # GD-5 / VIE01-250: consent management.
    # Stamped on each new user as gdpr_consent_version. Showing the notice
    # to users whose version is older is the Keycloak REVIEW_CONSENT
    # required action's job (see 06-authentication/README.md), not this
    # service's -- bump this and that action's configured version together.
    consent_policy_version: str = "1.0"
    # Withdrawal cascades. Module 22 owns benchmark_schemas/benchmark_runs.
    # Module 34 (realtime-audio) deletes its WAV recordings at DELETE
    # /internal/gdpr/user/{user_id} (VIE01-302; docker-compose.yml points here
    # at raptor-realtime-audio:8034). With no URL that deletion (on withdrawal
    # and on GD-1 erasure) is recorded as `skipped` and the export omits
    # voice/recordings.json; the Redis flag Module 34 checks at session start
    # is set either way. GD-10 / VIE01-416: on by default, as for Module 35.
    module_22_url: str = "http://raptor-benchmark-service:8000"
    module_34_url: str = "http://raptor-realtime-audio:8034"
    # GD-1 / VIE01-246: Module 35 (inference) deletes the user's voice-clone
    # speaker embeddings on erasure. Set it to "" on a deployment without
    # Module 35 -- module_35 is then recorded as `skipped`, not failed.
    module_35_url: str = "http://raptor-inference-service:8035"

    # V05-10 / VIE01-364: CF-4 audit events (raptor_audit) this service
    # writes to its own audit_events table -- one `gdpr.erasure` event per
    # finished erasure or consent-withdrawal job. Kept as long as
    # gdpr_erasure_jobs (3-year legal record) by default, since each event
    # is the durable proof that job ran. The daily retention run deletes
    # every audit_events row past its own retention_days.
    audit_retention_days: int = Field(1095, ge=1)

    # CM-7 / VIE01-410: event rows per section of the compliance report (the
    # counts always cover the whole period). Default and upper bound for the
    # route's max_rows.
    compliance_report_max_rows: int = Field(50000, ge=1)


settings = Settings()
