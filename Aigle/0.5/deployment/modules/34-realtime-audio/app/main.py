"""Module 34 -- real-time audio: WebSocket audio streaming with per-session VAD.

RA-1: GET /health, POST /audio/session/token (one-time WS ticket), WS /audio/stream.
RA-2: silero-VAD utterance segmentation inside each stream (app/vad.py).
RA-3 / RTA-X2: faster-whisper final and rolling-window transcripts (app/stt.py).
RA-4: pluggable TTS (app/tts_backends/, app/tts.py), GET /tts/models; RA-5: Module 15 voice turns.
RTA-X3 / X4 / X5: numbered frames and RESUME, bounded queues and GET /metrics, barge-in
(app/stream.py, app/outbox.py, app/metrics.py).
RTA-X6: Module 23 screening, Module 25 indexing, Module 26 archiving (app/guardrail.py,
app/integrations.py).  RTA-X7 / RA-9: consent-aware recording (app/recording.py) and
DELETE /internal/gdpr/user/{user_id} for Module 33.  GD-10: GET /internal/gdpr/user/{user_id}/recordings
(recording metadata for Module 33's export).
RA-6: gRPC AudioService.AudioChat on RT_GRPC_PORT (app/grpc_audio.py), the same session pipeline.
INF-10 / INF-11: POST /audio/transcribe, POST /audio/synthesize -- one-shot STT/TTS for Module
35's own /inference/voice/stt and /inference/voice/tts, internal-token gated (VIE01-316/317).
"""
from __future__ import annotations

import io
import logging
import os
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
import numpy as np
import soundfile as sf
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
)
from redis.asyncio import Redis

from app.auth import KeyProvider, get_current_user
from app.blobstore import BlobError, SeaweedFS
from app.config import Settings, get_settings
from app.errors import install_error_handlers
from app.guardrail import Guardrail
from app.integrations import KafkaPublisher, MemoryArchive, TranscriptIndexer
from app.llm_bridge import LlmBridge
from app.metrics import registry, render
from app.recording import ConsentChecker, Recorder, RecordingService
from app.recordings_db import RecordingsDb
from app.session import SessionStore, TicketStore
from app.stream import SESSION_ROLES, StreamHandler, UtteranceSink, WavDumpSink
from app.stt import (
    SAMPLE_RATE,
    SttBackend,
    SttOptions,
    SttService,
    faster_whisper_loader,
    wants_traditional,
    whisper_language,
)
from app.tts import TtsService
from app.tts_backends import TtsBackend, load_backend
from app.vad import ProbFn, VadModelPool, silero_loader, silero_version

MODULE_ID = 34
logger = logging.getLogger(__name__)


def build_services(
    app: FastAPI,
    settings: Settings,
    redis: Redis,
    vad_loader: Callable[[], ProbFn],
    sink: UtteranceSink | None = None,
    stt_loader: Callable[[str], SttBackend] | None = None,
    options: SttOptions | None = None,
    tts_loader: Callable[[str, Settings], TtsBackend] | None = None,
    http: httpx.AsyncClient | None = None,
    kafka=None,
    recordings_db=None,
    blobs=None,
) -> None:
    """``stt_loader`` maps a model name to a loaded backend, ``tts_loader`` a backend name to a
    loaded TTS backend (None disables either); ``http`` is the client for Modules 15/23/26/33;
    ``kafka`` publishes to Module 25 and the recording consumer; ``recordings_db`` / ``blobs``
    hold recordings (None: built from settings)."""
    tickets = TicketStore(redis, settings.ticket_ttl_s)
    sessions = SessionStore(redis, settings.session_ttl_s, settings.max_sessions, settings.heartbeat_s,
                            settings.max_sessions_per_user)
    pool = VadModelPool(vad_loader, settings.vad_pool_size)
    if sink is None:
        dump_dir = os.environ.get("RT_UTTERANCE_DUMP_DIR")
        sink = WavDumpSink(dump_dir) if dump_dir else UtteranceSink()
    app.state.settings = settings
    app.state.tickets = tickets
    app.state.sessions = sessions
    app.state.vad_pool = pool
    stt = SttService(stt_loader, options or stt_options(settings)) if stt_loader is not None else None
    tts = TtsService(settings, tts_loader) if tts_loader is not None else None
    llm = (LlmBridge(settings.chat_service_url, settings.llm_timeout_s, settings.chat_stream, http,
                     first_token_timeout_s=settings.llm_first_token_timeout_s)
           if settings.llm_enabled and http is not None else None)
    app.state.stt = stt
    app.state.tts = tts
    app.state.key_provider = KeyProvider(settings)
    if kafka is None and (settings.index_transcripts or settings.recording_enabled):
        kafka = KafkaPublisher(settings.kafka_bootstrap)
    guardrail = (Guardrail(settings.guardrail_url, http, settings.guardrail_timeout_s, settings.guardrail_fail_open)
                 if settings.guardrail_enabled and http is not None else None)
    indexer = TranscriptIndexer(kafka, settings.personal_index_topic) if settings.index_transcripts and kafka else None
    memory = MemoryArchive(settings.memory_url, http) if settings.memory_enabled and http is not None else None
    if recordings_db is None and settings.postgres_dsn:
        recordings_db = RecordingsDb(settings.postgres_dsn)
    if blobs is None and http is not None:
        blobs = SeaweedFS(settings.seaweedfs_master, http, settings.recording_collection)
    consent = factory = None
    if settings.recording_enabled and http is not None and blobs is not None and kafka is not None:
        consent = ConsentChecker(settings, redis, http)

        def factory(session_id: str, user_id: str) -> Recorder:
            return Recorder(blobs, redis, kafka, settings.recording_topic, session_id, user_id,
                            settings.recording_chunk_s)
    app.state.kafka = kafka
    app.state.recordings_db = recordings_db
    app.state.recordings = (RecordingService(blobs, redis, recordings_db, settings.recording_retention_days)
                            if blobs is not None else None)
    app.state.stream_handler = StreamHandler(settings, tickets, sessions, pool, sink, stt, tts, llm,
                                             guardrail=guardrail, indexer=indexer, memory=memory,
                                             consent=consent, recorder_factory=factory)
    app.state.metrics = registry(app.state.stream_handler)


def stt_options(settings: Settings) -> SttOptions:
    return SttOptions(
        model=settings.stt_model, model_dir=settings.stt_model_dir,
        provisional_model=settings.stt_provisional_model, device=settings.stt_device,
        compute_type=settings.stt_compute_type, cpu_model=settings.stt_cpu_model,
        beam_size=settings.stt_beam_size, max_concurrent=settings.stt_max_concurrent,
        zh_convert=settings.stt_zh_convert,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    redis = Redis.from_url(settings.redis_url, socket_timeout=5, socket_connect_timeout=5)
    options = stt_options(settings)
    stt_loader = faster_whisper_loader(options) if settings.stt_enabled else None
    http = httpx.AsyncClient()
    build_services(app, settings, redis, silero_loader, stt_loader=stt_loader,
                   tts_loader=load_backend if settings.tts_enabled else None, http=http, options=options)
    # RA-2 / RA-3 / RA-4: load silero-VAD, the STT model(s) and the TTS backend at startup;
    # an unknown RT_TTS_MODEL or an unavailable backend stops the service here. One at a time,
    # not asyncio.gather: each loader's first call is each library's first import of torch/
    # transformers in this process, off the main thread (asyncio.to_thread) -- running them
    # concurrently raced on torch/transformers' own lazy module-init state and failed
    # non-deterministically on GX10 (confirmed live: "cannot import name 'AlbertModel' from
    # transformers", and separately "Cannot copy out of meta tensor", on different restarts of
    # the same unchanged image). A few extra seconds once at startup is a trivial cost next to a
    # service that crash-loops under Docker's restart policy until it happens to not race.
    await app.state.vad_pool.start()
    for service in (app.state.stt, app.state.tts):
        if service is not None:
            await service.start()
    grpc_server = None
    app.state.grpc_port = None
    if settings.grpc_enabled:
        # RA-6: AudioService on RT_GRPC_PORT, sharing this process's sessions and models
        from app.grpc_audio import start_server

        grpc_server, app.state.grpc_port = await start_server(app.state.stream_handler, app.state.key_provider,
                                                              settings)
    try:
        yield
    finally:
        if grpc_server is not None:
            await grpc_server.stop(grace=5)
        if app.state.kafka is not None:
            await app.state.kafka.stop()
        if app.state.recordings_db is not None:
            await app.state.recordings_db.close()
        await http.aclose()
        await redis.aclose()


def create_app(use_lifespan: bool = True) -> FastAPI:
    app = FastAPI(
        title="Raptor Module 34 — Real-time Audio",
        version="0.5.0",
        lifespan=lifespan if use_lifespan else None,
    )
    install_error_handlers(app)

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, Any]:
        return {"status": "ok", "module": MODULE_ID}

    @app.get("/audio/status", tags=["audio"], summary="Active sessions and VAD model info")
    async def status(request: Request) -> dict[str, Any]:
        settings: Settings = request.app.state.settings
        return {
            "active_sessions": await request.app.state.sessions.active_count(),
            "max_sessions": settings.max_sessions,
            "vad": {"model": "silero-vad", "version": silero_version()},
            "stt": request.app.state.stt.info() if request.app.state.stt else {"enabled": False},
            "tts": request.app.state.tts.info() if request.app.state.tts else {"enabled": False},
        }

    @app.get("/tts/models", tags=["audio"], summary="Installed TTS backends: loaded / available / unavailable")
    async def tts_models(request: Request) -> dict[str, Any]:
        tts: TtsService | None = request.app.state.tts
        if tts is None:
            return {"enabled": False, "models": []}
        return {"enabled": True, "models": await tts.models()}

    def require_internal_token(request: Request,
                               x_internal_token: str | None = Header(None, alias="X-Internal-Token")) -> None:
        expected = request.app.state.settings.gd_internal_token
        if not expected or not x_internal_token or not secrets.compare_digest(x_internal_token, expected):
            raise HTTPException(status_code=401, detail="invalid_internal_token")

    async def _acquire_batch_slot(request: Request, user_id: str) -> str:
        """Shares SessionStore's per-user slot pool with live WS sessions (VIE01-316/317): a
        batch call and a live session both tie up this user's share of Module 34's resources,
        so they draw from one quota instead of two independently-maintained ones."""
        sessions: SessionStore = request.app.state.sessions
        session_id, status = await sessions.acquire_slot(user_id)
        if status != "ok":
            raise HTTPException(429, status)
        return session_id

    @app.post("/audio/transcribe", tags=["audio"], dependencies=[Depends(require_internal_token)],
              summary="Module 35 only: one-shot speech-to-text on a finished clip (INF-10)")
    async def transcribe_once(
        request: Request, audio: Annotated[UploadFile, File()], language: str | None = Form(None),
        x_user_id: str = Header(..., alias="X-User-Id"),
    ) -> dict[str, Any]:
        """16 kHz mono PCM16 WAV in, transcript out -- no session, no rolling decode, no VAD.
        The only caller is Module 35's ``POST /inference/voice/stt`` (INF-10); it already
        normalizes the upload to this exact format (its own ``app/services/voice/audio.py``),
        so this endpoint carries no format-transcoding path of its own. Reuses the same
        ``SttService`` (and its admission-controlled GPU slots) that live sessions use --
        a batch call here can add latency to a concurrent live session, see README."""
        sessions: SessionStore = request.app.state.sessions
        slot_id = await _acquire_batch_slot(request, x_user_id)
        try:
            stt: SttService | None = request.app.state.stt
            if stt is None or not stt.ready:
                raise HTTPException(503, "stt_unavailable")
            data = await audio.read()
            try:
                samples, sample_rate = sf.read(io.BytesIO(data), dtype="int16")
            except Exception as exc:
                raise HTTPException(400, f"invalid_audio: {exc}") from exc
            if getattr(samples, "ndim", 1) > 1:
                samples = samples[:, 0]
            if sample_rate != SAMPLE_RATE:
                raise HTTPException(400, f"audio must be {SAMPLE_RATE} Hz mono (got {sample_rate} Hz)")
            try:
                lang = whisper_language(language)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            try:
                transcript = await stt.transcribe(samples, lang)
            except Exception as exc:
                raise HTTPException(502, f"stt_failed: {exc}") from exc
            text = (stt.convert(transcript.text) if wants_traditional(language, transcript.language)
                   else transcript.text)
            return {
                "text": text, "language": transcript.language,
                "language_probability": transcript.language_probability, "confidence": transcript.confidence,
            }
        finally:
            await sessions.release_slot(slot_id, x_user_id)

    @app.post("/audio/synthesize", tags=["audio"], dependencies=[Depends(require_internal_token)],
              summary="Module 35 only: one-shot text-to-speech (INF-11)",
              responses={200: {"content": {"audio/wav": {}}, "description": "Synthesized WAV audio"}})
    async def synthesize_once(
        request: Request, text: str = Form(...), language: str | None = Form(None),
        x_user_id: str = Header(..., alias="X-User-Id"),
    ) -> Response:
        """The only caller is Module 35's ``POST /inference/voice/tts`` (INF-11); no session, no
        streaming frames out -- the whole clip is generated, then returned as one WAV body.
        Reuses the same ``TtsService`` (and its admission-controlled synthesis slots) that live
        sessions use -- see the same latency note as ``/audio/transcribe`` above."""
        sessions: SessionStore = request.app.state.sessions
        slot_id = await _acquire_batch_slot(request, x_user_id)
        try:
            tts: TtsService | None = request.app.state.tts
            if tts is None:
                raise HTTPException(503, "tts_unavailable")
            try:
                pieces = [piece async for piece in tts.stream(text, language)]
            except Exception as exc:
                raise HTTPException(502, f"tts_failed: {exc}") from exc
            if not pieces:
                raise HTTPException(502, "tts_produced_no_audio")
            buffer = io.BytesIO()
            sf.write(buffer, np.concatenate(pieces), tts.sample_rate, format="WAV", subtype="PCM_16")
            return Response(buffer.getvalue(), media_type="audio/wav")
        finally:
            await sessions.release_slot(slot_id, x_user_id)

    @app.get("/metrics", tags=["system"], summary="Prometheus metrics: queue depths, GPU slots, backpressure")
    async def metrics(request: Request) -> Response:
        return Response(render(request.app.state.metrics), media_type="text/plain; version=0.0.4")

    @app.post("/audio/session/token", tags=["audio"], summary="Issue a one-time WebSocket ticket (30 s)")
    async def issue_ticket(request: Request, claims: Annotated[dict[str, Any], Depends(get_current_user)]):
        tickets: TicketStore = request.app.state.tickets
        realm_roles = (claims.get("realm_access") or {}).get("roles") or []
        token = await tickets.issue(str(claims["sub"]), [r for r in realm_roles if r in SESSION_ROLES])
        return {"token": token, "expires_in": tickets.ttl_s}

    @app.delete("/internal/gdpr/user/{user_id}", tags=["GDPR (internal)"], dependencies=[Depends(require_internal_token)],
                summary="Module 33 only: erase the user's voice recordings (GD-1 / consent withdrawal)")
    async def gdpr_erase(request: Request, user_id: str) -> dict[str, Any]:
        handler = request.app.state.stream_handler
        for conn in list(handler.live.values()):           # live sessions stop recording at once
            if conn.subject == user_id and conn.recorder is not None:   # RTA-X10: the recorded subject
                await conn.recorder.erase()
        service: RecordingService | None = request.app.state.recordings
        if service is None:
            return {"user_id": user_id, "erased_sessions": 0, "bytes_freed": 0}
        try:
            result = await service.erase_user(user_id)
        except (BlobError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=f"recording store unavailable: {exc}") from exc
        except Exception as exc:  # asyncpg errors: the store is down, Module 33 retries
            if type(exc).__module__.startswith("asyncpg"):
                raise HTTPException(status_code=503, detail="recording database unavailable") from exc
            raise
        logger.info("GDPR erasure of user %s: %s", user_id, result)
        return {"user_id": user_id, **result}

    @app.get("/internal/gdpr/user/{user_id}/recordings", tags=["GDPR (internal)"],
             dependencies=[Depends(require_internal_token)],
             summary="Module 33 only: the user's recording metadata for the GDPR export (GD-2 / GD-10)")
    async def gdpr_recordings(request: Request, user_id: str) -> dict[str, Any]:
        service: RecordingService | None = request.app.state.recordings
        if service is None:
            return {"user_id": user_id, "recordings": [], "pending_sessions": 0}
        try:
            result = await service.manifest(user_id)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=f"recording store unavailable: {exc}") from exc
        except Exception as exc:  # asyncpg errors: the store is down, Module 33 records module_34 as failed
            if type(exc).__module__.startswith("asyncpg"):
                raise HTTPException(status_code=503, detail="recording database unavailable") from exc
            raise
        return {"user_id": user_id, **result}

    @app.websocket("/audio/stream")
    async def audio_stream(websocket: WebSocket, token: str | None = Query(None),
                           session_id: str | None = Query(None)):
        # session_id: reconnecting to resume that session (RTA-X3); the first frame is RESUME.
        await websocket.app.state.stream_handler.handle(websocket, token, session_id)

    return app


app = create_app()
