"""grpc.aio server bootstrap for the Raptor gRPC gateway.

Binds RaptorService + the standard grpc.health.v1.Health service on the same
port, wires the full interceptor chain, and exposes Prometheus metrics on a
separate HTTP port (gRPC itself isn't HTTP, so metrics need their own port).

SearchByPerson/StreamSearchByPerson, DownloadAsset, UploadAsset, the
SubmitJob/GetJob/WatchJob job-lifecycle RPCs, StreamVideoSegment and
JobSession are real, backed by shared services copied in from module 13's
REST app (see the Dockerfile). AnalyzeStream's chunk-forwarding-to-Kafka
half is real too, but it always ends in UNIMPLEMENTED once the client's
send side closes -- see media_stream_service.py's module docstring for why.
Every other RPC still returns UNIMPLEMENTED; later feature phases build on
this same interceptor chain without reworking it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import grpc
import httpx
from aiokafka import AIOKafkaProducer
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from prometheus_client import start_http_server
from redis.asyncio import Redis

sys.path.insert(0, str(Path(__file__).parent / "generated"))
from app.services.kafka_service import KafkaService
from app.services.personal_db_service import PersonalDBService
from app.services.search_service import HybridSearchService
from app.services.storage_service import StorageService
from app.services.video_search_service import VideoSearchService
from config import get_settings
from interceptors.auth import AuthenticationInterceptor
from interceptors.authz import AuthorizationInterceptor
from interceptors.context import request_id_var, user_var
from interceptors.errors import (
    ExceptionMappingInterceptor,
    UnavailableError,
    UnimplementedError,
)
from interceptors.guardrail import GuardrailInterceptor
from interceptors.identity import IdentityInterceptor
from interceptors.logging import LoggingInterceptor
from interceptors.metrics import MetricsInterceptor
from interceptors.request_id import RequestIdInterceptor
from interceptors.stream_limit import StreamLimitInterceptor
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from services import asset_service as asset_handlers
from services import job_service as job_handlers
from services import media_stream_service as media_stream_handlers
from services import search_service as search_handlers
from services.media_probe import DurationProbe, make_duration_probe

SERVICE_NAME = "raptor.v1.RaptorService"

_logger = logging.getLogger("grpc_gateway.server")


class _ServingHealthServicer(health.aio.HealthServicer):
    """grpc.health.v1 for a grpc.aio server, with the given services already SERVING.

    The synchronous `health.HealthServicer` must not be used here: its Watch calls
    `context.add_callback`, which the aio context does not have, so every Watch failed with
    INTERNAL (Check happened to work). The aio servicer's own `set()` is a coroutine and
    build_server() is synchronous, so the initial statuses are written to the same dict the
    servicer reads (pinned grpcio-health-checking; tests/unit/test_health.py would fail if
    that attribute ever moved).
    """

    def __init__(self, services):
        super().__init__()
        for service in services:
            self._server_status[service] = health_pb2.HealthCheckResponse.SERVING


class RaptorServiceServicer(raptor_pb2_grpc.RaptorServiceServicer):
    """Ping, Search, SearchBm25, SearchVector, their Stream versions, GraphRagSearch, TKGSearch, SearchByPerson, StreamSearchByPerson, DownloadAsset, UploadAsset,
    SubmitJob, GetJob, WatchJob, StreamVideoSegment (asset branch only --
    its video_id branch is UNIMPLEMENTED, see media_stream_service.py) and
    JobSession (subscribe direction only -- cancel/control are intent-only,
    see job_service.py) are real; every other RPC returns UNIMPLEMENTED in
    this phase."""

    def __init__(
        self,
        video_search_service: VideoSearchService | None,
        storage_service: StorageService,
        http_client: httpx.AsyncClient,
        redis_client,
        kafka_producer: AIOKafkaProducer | None = None,
        duration_probe: DurationProbe | None = None,
        personal_db: PersonalDBService | None = None,
        analyze_stream_enabled: bool | None = None,
    ):
        self._duration_probe = duration_probe
        self._personal_db = personal_db
        self._video_search_service = video_search_service
        self._storage_service = storage_service
        self._http_client = http_client
        self._redis_client = redis_client
        self._kafka_producer = kafka_producer
        settings = get_settings()
        self._upload_idle_timeout_s = settings.upload_idle_timeout_s
        self._kafka_asset_finalized_topic = settings.kafka_asset_finalized_topic
        self._watch_job_poll_interval_s = settings.watch_job_poll_interval_s
        self._ffmpeg_binary = settings.ffmpeg_binary
        self._ffmpeg_terminate_timeout_s = settings.ffmpeg_terminate_timeout_s
        self._analyze_idle_timeout_s = settings.analyze_idle_timeout_s
        self._analyze_stream_enabled = (
            settings.analyze_stream_enabled if analyze_stream_enabled is None else analyze_stream_enabled
        )
        self._kafka_analysis_requests_topic = settings.kafka_analysis_requests_topic

    async def Ping(self, request, context):
        return raptor_pb2.PingResponse(message=request.message, request_id=request_id_var.get())

    async def Search(self, request, context):
        return await search_handlers.search(self._personal_db, request, user_var.get())

    async def SearchBm25(self, request, context):
        return await search_handlers.search_bm25(self._personal_db, request, user_var.get())

    async def SearchVector(self, request, context):
        return await search_handlers.search_vector(self._personal_db, request, user_var.get())

    async def StreamSearch(self, request, context):
        async for hit in self._stream(search_handlers.stream_search(self._personal_db, request, user_var.get())):
            yield hit

    async def StreamSearchBm25(self, request, context):
        async for hit in self._stream(search_handlers.stream_search_bm25(self._personal_db, request, user_var.get())):
            yield hit

    async def StreamSearchVector(self, request, context):
        async for hit in self._stream(search_handlers.stream_search_vector(self._personal_db, request, user_var.get())):
            yield hit

    async def GraphRagSearch(self, request, context):
        return await search_handlers.graphrag_search(self._personal_db, request, user_var.get())

    async def TKGSearch(self, request, context):
        return await search_handlers.tkg_search(self._personal_db, request, user_var.get())

    @staticmethod
    async def _stream(agen):
        """Passes an inner async generator through and always closes it (see DownloadAsset for
        why an explicit aclose() matters when the outer generator is cancelled)."""
        try:
            async for item in agen:
                yield item
        finally:
            await agen.aclose()

    async def SearchByPerson(self, request, context):
        return await search_handlers.search_by_person(self._video_search_service, request, user_var.get())

    async def StreamSearchByPerson(self, request, context):
        agen = search_handlers.stream_search_by_person(self._video_search_service, request, user_var.get())
        try:
            async for hit in agen:
                yield hit
        finally:
            # Explicit close, not just letting `async for` unwind on
            # GeneratorExit: Python does not auto-close an inner async
            # generator when the outer one wrapping it is cancelled/closed
            # (see DownloadAsset below, where this matters for real —
            # here it just guarantees the same cleanup contract).
            await agen.aclose()

    async def GetAsset(self, request, context):
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "GetAsset not implemented in this phase.")

    async def UploadAsset(self, request_iterator, context):
        return await asset_handlers.upload_asset(
            self._storage_service,
            self._redis_client,
            self._kafka_producer,
            self._kafka_asset_finalized_topic,
            request_iterator,
            user_var.get(),
            self._upload_idle_timeout_s,
            context,
        )

    async def Chat(self, request, context):
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Chat not implemented in this phase.")
        return
        yield  # pragma: no cover - keeps this an async generator matching the streaming RPC shape

    async def GetJobStatus(self, request, context):
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "GetJobStatus not implemented in this phase.")

    async def StreamVideo(self, request, context):
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "StreamVideo not implemented in this phase.")
        return
        yield  # pragma: no cover - keeps this an async generator matching the streaming RPC shape

    async def DownloadAsset(self, request, context):
        agen = asset_handlers.download_asset(
            self._storage_service, self._http_client, request, user_var.get(), self._duration_probe
        )
        try:
            async for chunk in agen:
                yield chunk
        finally:
            # Explicit close, not just letting `async for` unwind on
            # GeneratorExit: Python does not auto-close an inner async
            # generator when the outer one wrapping it is cancelled/closed,
            # so without this, cancellation would leave the storage read
            # loop's `finally` (queue teardown, reader_task.cancel(), the
            # `async with http_client.stream(...)` exit) to run whenever the
            # inner generator is eventually garbage-collected instead of
            # synchronously on cancellation -- exactly the "releases all
            # open handles" guarantee this RPC promises.
            await agen.aclose()

    def _kafka_service(self) -> KafkaService:
        if self._kafka_producer is None:
            raise UnavailableError("Kafka is not configured on this gateway; job submission is unavailable.")
        return KafkaService(self._kafka_producer, get_settings(), self._redis_client)

    async def SubmitJob(self, request, context):
        return await job_handlers.submit_job(
            self._kafka_service(), self._storage_service, self._redis_client, request, user_var.get()
        )

    async def GetJob(self, request, context):
        return await job_handlers.get_job(self._redis_client, request, user=user_var.get())

    async def WatchJob(self, request, context):
        agen = job_handlers.watch_job(
            self._redis_client, request, self._watch_job_poll_interval_s, user=user_var.get()
        )
        try:
            async for event in agen:
                yield event
        finally:
            await agen.aclose()

    async def StreamVideoSegment(self, request, context):
        agen = media_stream_handlers.stream_video_segment(
            self._storage_service,
            self._http_client,
            self._ffmpeg_binary,
            self._ffmpeg_terminate_timeout_s,
            request,
            user_var.get(),
        )
        try:
            async for chunk in agen:
                yield chunk
        finally:
            # See DownloadAsset above for why this explicit aclose() matters
            # on cancellation -- here it's also what guarantees the FFmpeg
            # subprocess actually gets SIGTERM'd synchronously rather than
            # whenever the generator happens to be garbage-collected.
            await agen.aclose()

    async def JobSession(self, request_iterator, context):
        agen = job_handlers.job_session(
            self._redis_client, request_iterator, self._watch_job_poll_interval_s, user=user_var.get()
        )
        try:
            async for response in agen:
                yield response
        finally:
            await agen.aclose()

    async def AnalyzeStream(self, request_iterator, context):
        if not self._analyze_stream_enabled:
            # Refused before reading a byte: nothing is forwarded to Kafka, so a client is not
            # left believing its media is being analysed when no service reads the topic.
            raise UnimplementedError(
                "AnalyzeStream is disabled on this gateway: no service consumes the media it would "
                "forward, so there is nothing to analyse it. See docs/_internal/OPEN_QUESTION_analyze_stream_results.md."
            )
        agen = media_stream_handlers.analyze_stream(
            self._kafka_producer,
            self._kafka_analysis_requests_topic,
            self._redis_client,
            request_iterator,
            user_var.get(),
            self._analyze_idle_timeout_s,
        )
        try:
            async for event in agen:
                yield event
        finally:
            await agen.aclose()


def _build_services(http_client: httpx.AsyncClient) -> tuple[VideoSearchService, StorageService]:
    settings = get_settings()
    personal_db = PersonalDBService(client=http_client, settings=settings)
    storage_service = StorageService(http_client, settings)
    url_resolver = HybridSearchService(
        hybrid_search_url=settings.hybrid_search_url,
        http_client=http_client,
        storage_service=storage_service,
    )
    video_search_service = VideoSearchService(personal_db=personal_db, url_resolver=url_resolver)
    return video_search_service, storage_service


def _build_redis_client(settings) -> Redis:
    return Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        decode_responses=True,
    )


def build_server(
    video_search_service: VideoSearchService | None = None,
    storage_service: StorageService | None = None,
    http_client: httpx.AsyncClient | None = None,
    redis_client: Redis | None = None,
    kafka_producer: AIOKafkaProducer | None = None,
    max_streams_per_user: int | None = None,
    identity_http_client: httpx.AsyncClient | None = None,
    identity_dev_fallback: bool | None = None,
    authz_http_client: httpx.AsyncClient | None = None,
    duration_probe: DurationProbe | None = None,
    personal_db: PersonalDBService | None = None,
    guardrail_interceptor: grpc.aio.ServerInterceptor | None = None,
    analyze_stream_enabled: bool | None = None,
) -> grpc.aio.Server:
    """Constructs the grpc.aio server with the full interceptor chain and health service.

    All service dependencies default to lazily-constructed instances so
    existing tests that only exercise Ping/exception-mapping don't need to
    supply any of them; serve() always constructs and passes them explicitly
    so their connections can be cleaned up on shutdown.

    identity_http_client is separate from http_client: tests that need
    IdentityInterceptor to fetch JWKS from a mocked transport (see
    tests/unit/conftest.py) must not also redirect the shared http_client
    that StorageService/VideoSearchService use for real downstream calls.
    Defaults to http_client, matching serve()'s real deployment behaviour
    (one shared connection pool).
    """
    if http_client is None:
        http_client = httpx.AsyncClient()
    if video_search_service is None or storage_service is None:
        built_vss, built_storage = _build_services(http_client)
        if video_search_service is None:
            video_search_service = built_vss
        if storage_service is None:
            storage_service = built_storage
    if personal_db is None:
        # Search/SearchBm25/SearchVector: stateless, shares the connection pool above.
        personal_db = PersonalDBService(client=http_client, settings=get_settings())
    if redis_client is None:
        redis_client = _build_redis_client(get_settings())
    # kafka_producer is NOT lazily constructed here (unlike the clients
    # above) -- AIOKafkaProducer needs an awaited .start() to connect, and
    # build_server() is sync; serve() constructs and starts it explicitly.
    # Staying None means UploadAsset's Kafka publish step no-ops (logged),
    # which is what existing tests that don't care about Kafka rely on.

    interceptors = [
        AuthenticationInterceptor(),
        IdentityInterceptor(http_client=identity_http_client or http_client, dev_fallback=identity_dev_fallback),
        AuthorizationInterceptor(http_client=authz_http_client or http_client),
        RequestIdInterceptor(),
        LoggingInterceptor(),
        MetricsInterceptor(),
        StreamLimitInterceptor(redis_client, max_streams_per_user or get_settings().max_streams_per_user),
    ]
    if guardrail_interceptor is None:
        guardrail_url = _screening_guardrail_url(get_settings())
        if guardrail_url:
            guardrail_interceptor = GuardrailInterceptor(
                guardrail_url, get_settings().guardrail_timeout_ms / 1000, http_client=http_client
            )
    if guardrail_interceptor is not None:
        interceptors.append(guardrail_interceptor)
    interceptors.append(ExceptionMappingInterceptor())
    server = grpc.aio.server(interceptors=interceptors)

    raptor_pb2_grpc.add_RaptorServiceServicer_to_server(
        RaptorServiceServicer(
            video_search_service, storage_service, http_client, redis_client, kafka_producer, duration_probe,
            personal_db, analyze_stream_enabled,
        ),
        server,
    )

    health_pb2_grpc.add_HealthServicer_to_server(_ServingHealthServicer(("", SERVICE_NAME)), server)

    reflection.enable_server_reflection(
        (
            SERVICE_NAME,
            health_pb2.DESCRIPTOR.services_by_name["Health"].full_name,
            reflection.SERVICE_NAME,
        ),
        server,
    )

    return server


def _build_server_credentials(settings) -> grpc.ServerCredentials:
    """Builds TLS (optionally mTLS) server credentials from the cert/key
    paths in settings. Only called when settings.grpc_secure is true."""
    if not settings.grpc_tls_cert_path or not settings.grpc_tls_key_path:
        raise RuntimeError(
            "grpc_secure=true requires both grpc_tls_cert_path and grpc_tls_key_path to be set."
        )
    with open(settings.grpc_tls_key_path, "rb") as f:
        private_key = f.read()
    with open(settings.grpc_tls_cert_path, "rb") as f:
        certificate_chain = f.read()

    if settings.grpc_mtls_ca_cert_path:
        with open(settings.grpc_mtls_ca_cert_path, "rb") as f:
            root_certificates = f.read()
        return grpc.ssl_server_credentials(
            [(private_key, certificate_chain)],
            root_certificates=root_certificates,
            require_client_auth=True,
        )
    return grpc.ssl_server_credentials([(private_key, certificate_chain)])


async def _start_kafka_producer(settings) -> AIOKafkaProducer | None:
    """Starts the Kafka producer, or returns None when Kafka is not configured or unreachable
    (UploadAsset events are then skipped, SubmitJob answers UNAVAILABLE).

    A producer whose start() failed or timed out still owns a client and sockets; it is stopped
    before being dropped, otherwise aiokafka logs `Unclosed AIOKafkaProducer` at exit."""
    if not settings.kafka_bootstrap_servers:
        return None
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )
    try:
        await asyncio.wait_for(producer.start(), timeout=10.0)
    except Exception:
        _logger.warning("Kafka producer failed to start -- UploadAsset events will be skipped.", exc_info=True)
        try:
            await producer.stop()
        except Exception:
            _logger.debug("Stopping the Kafka producer that failed to start also failed.", exc_info=True)
        return None
    return producer


def _screening_guardrail_url(settings) -> str:
    """Module 23's URL when GuardrailInterceptor should run, else "". Screening needs both
    the switch and the URL: GUARDRAIL_URL is shared by every module (root .env), so its
    presence alone must not turn screening on."""
    if not settings.guardrail_enabled:
        return ""
    if not settings.guardrail_url:
        _logger.error("Guardrail screening is enabled but no guardrail URL is set -- RPCs are NOT screened.")
        return ""
    return settings.guardrail_url


_GUARDRAIL_COVERAGE_KEY = "guardrail:integration:grpc_interceptor"


async def _register_guardrail_coverage(settings, redis_client) -> None:
    """GB-22 coverage matrix: grpc_interceptor is screened exactly when GuardrailInterceptor
    runs. The key has no TTL, so when screening is off it is deleted -- a "1" left by an
    earlier start would otherwise keep reporting screened. Fail-open."""
    try:
        if _screening_guardrail_url(settings):
            await redis_client.set(_GUARDRAIL_COVERAGE_KEY, "1")
            _logger.info("[GB-22] guardrail integration registered: grpc_interceptor")
        else:
            await redis_client.delete(_GUARDRAIL_COVERAGE_KEY)
    except Exception as exc:  # noqa: BLE001 — fail-open
        _logger.warning("[GB-22] guardrail startup registration failed (screened=false): %s", exc)


async def serve() -> None:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    start_http_server(settings.metrics_port)
    _logger.info("Prometheus metrics listening on :%d/metrics", settings.metrics_port)

    http_client = httpx.AsyncClient()
    video_search_service, storage_service = _build_services(http_client)
    redis_client = _build_redis_client(settings)

    kafka_producer = await _start_kafka_producer(settings)

    server = build_server(
        video_search_service=video_search_service,
        storage_service=storage_service,
        http_client=http_client,
        redis_client=redis_client,
        kafka_producer=kafka_producer,
        duration_probe=make_duration_probe(settings),
    )
    bind_address = f"0.0.0.0:{settings.grpc_port}"
    if settings.grpc_secure:
        credentials = _build_server_credentials(settings)
        server.add_secure_port(bind_address, credentials)
        _logger.info(
            "Raptor gRPC gateway listening on %s (TLS%s)",
            bind_address, ", client cert required" if settings.grpc_mtls_ca_cert_path else "",
        )
    else:
        server.add_insecure_port(bind_address)
        _logger.info("Raptor gRPC gateway listening on %s (plaintext)", bind_address)
    await _register_guardrail_coverage(settings, redis_client)
    await server.start()
    try:
        await server.wait_for_termination()
    finally:
        await http_client.aclose()
        await redis_client.aclose()
        if kafka_producer is not None:
            await kafka_producer.stop()


if __name__ == "__main__":
    asyncio.run(serve())
