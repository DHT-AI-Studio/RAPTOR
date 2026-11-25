"""Consent-aware session recording, finalization, erasure and retention."""
from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from typing import Any, Protocol

import httpx
import numpy as np

from app.blobstore import BlobError, SeaweedFS
from app.recordings_db import NoDatabase, Recording
from app.session import _dec

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VOICE_DISABLED_KEY = "gdpr:voice_recording_disabled:{user_id}"
ERASED_USERS_KEY = "gdpr:erased_users"
PENDING_KEY = "rt:rec:pending"
PENDING_TTL_S = 7 * 24 * 3600


def meta_key(session_id: str) -> str:
    return f"rt:rec:{session_id}"


def chunks_key(session_id: str) -> str:
    return f"rt:rec:{session_id}:chunks"


def user_key(user_id: str) -> str:
    return f"rt:rec:user:{user_id}"


class Publisher(Protocol):
    async def send(self, topic: str, value: dict[str, Any], key: str | None = None) -> bool: ...


class Recordings(Protocol):
    async def insert(self, user_id: str, session_id: str, fid: str, duration_s: int, size_bytes: int,
                     retention_days: int) -> bool: ...

    async def live_for_user(self, user_id: str) -> list[Recording]: ...

    async def expired(self, limit: int = 100) -> list[Recording]: ...

    async def mark_erased(self, ids: list[str]) -> None: ...


# ---------------------------------------------------------------- consent


class ConsentChecker:
    def __init__(self, settings, redis, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.redis = redis
        self.client = client

    async def check(self, user_id: str) -> tuple[bool, str | None]:
        """(record?, why not)."""
        if not self.settings.recording_enabled:
            return False, "disabled_by_config"
        if not user_id:
            return False, "no_user"
        try:
            if await self.redis.exists(VOICE_DISABLED_KEY.format(user_id=user_id)):
                return False, "consent_withdrawn"
        except Exception as exc:  # noqa: BLE001 -- cannot tell: fail closed
            logger.warning("consent flag for %s unreadable: %s", user_id, exc)
            return False, "consent_unavailable"
        try:
            resp = await self.client.get(f"{self.settings.module33_url}/consent", timeout=5.0, headers={
                "X-User-ID": user_id, "X-Internal-Token": self.settings.gd_internal_token})
            if resp.status_code == 404:
                return False, "no_consent"                      # not a Keycloak user: never consented
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("purposes"), list):
                raise ValueError("consent response must contain a purposes list")
            if any(not isinstance(p, dict) or not isinstance(p.get("id"), str)
                   or not isinstance(p.get("granted"), bool) for p in payload["purposes"]):
                raise ValueError("invalid consent purpose")
            purposes = {p["id"]: p["granted"] for p in payload["purposes"]}
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Module 33 consent check for %s failed: %s", user_id, exc)
            return False, "module33_unavailable"
        return (True, None) if purposes.get("voice_recording") is True else (False, "no_consent")


# ---------------------------------------------------------------- one session


def wav_bytes(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class Recorder:
    """The user's audio of one session, uploaded in chunks as it arrives."""

    def __init__(self, blobs: SeaweedFS, redis, kafka: Publisher, topic: str, session_id: str, user_id: str,
                 chunk_s: float = 10.0) -> None:
        self.blobs = blobs
        self.redis = redis
        self.kafka = kafka
        self.topic = topic
        self.session_id = session_id
        self.user_id = user_id
        self.chunk_samples = max(1, int(chunk_s * SAMPLE_RATE))
        self._buf: list[np.ndarray] = []
        self._buffered = 0
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self.erased = False
        self.finished = False
        self.samples = 0
        self.failed_chunks = 0

    async def start(self) -> None:
        pipe = self.redis.pipeline()
        pipe.hset(meta_key(self.session_id), mapping={"user_id": self.user_id, "session_id": self.session_id,
                                                      "sample_rate": SAMPLE_RATE, "started_at": time.time(),
                                                      "state": "recording", "samples": 0})
        pipe.sadd(PENDING_KEY, self.session_id)
        pipe.sadd(user_key(self.user_id), self.session_id)
        for key in (meta_key(self.session_id), user_key(self.user_id)):
            pipe.expire(key, PENDING_TTL_S)
        await pipe.execute()

    def add(self, pcm: np.ndarray) -> None:
        if self.erased or self.finished or not len(pcm):
            return
        self._buf.append(pcm.astype(np.int16, copy=True))
        self._buffered += len(pcm)
        if self._buffered >= self.chunk_samples:
            task = asyncio.create_task(self._flush())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _flush(self) -> None:
        async with self._lock:
            if not self._buf or self.erased:
                return
            data = np.concatenate(self._buf).astype("<i2").tobytes()
            self._buf, self._buffered = [], 0
            try:
                fid = await self.blobs.put(data, f"{self.session_id}.pcm")
            except BlobError as exc:
                self.failed_chunks += 1
                logger.error("session %s: recording chunk lost: %s", self.session_id, exc)
                return
            # Keep uploaded chunks discoverable even if erasure raced the upload.
            self.samples += len(data) // 2
            pipe = self.redis.pipeline()
            pipe.rpush(chunks_key(self.session_id), fid)
            pipe.hset(meta_key(self.session_id), mapping={"samples": self.samples, "updated_at": time.time()})
            pipe.expire(chunks_key(self.session_id), PENDING_TTL_S)
            await pipe.execute()

    async def finish(self, reason: str) -> bool:
        """Upload the rest and ask for the WAV; False when there is nothing to finalize."""
        if self.finished or self.erased:
            return False
        self.finished = True
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._flush()
        await self.redis.hset(meta_key(self.session_id), mapping={"state": "ended", "end_reason": reason})
        sent = await self.kafka.send(self.topic, {
            "event": "session_ended", "session_id": self.session_id, "user_id": self.user_id,
            "wav_chunks_key": chunks_key(self.session_id), "sample_rate": SAMPLE_RATE,
            "samples": self.samples, "reason": reason,
        }, key=self.session_id)
        if not sent:
            logger.warning("session %s: finalize request not published; the consumer's sweep will pick it up",
                           self.session_id)
        return True

    async def erase(self) -> None:
        """Consent withdrawn / user erased while the session runs: stop and drop what was recorded."""
        self.erased = True
        self._buf, self._buffered = [], 0
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


async def _quiet_delete(blobs: SeaweedFS, fid: str) -> int:
    try:
        return await blobs.delete(fid)
    except BlobError as exc:
        logger.warning("blob %s not deleted: %s", fid, exc)
        return 0


# ---------------------------------------------------------------- finalize, erase, retention


class RecordingService:
    def __init__(self, blobs: SeaweedFS, redis, db: Recordings | None, retention_days: int = 90) -> None:
        self.blobs = blobs
        self.redis = redis
        self.db = db
        self.retention_days = retention_days

    async def _discard_pending(self, session_id: str, user_id: str | None = None) -> int:
        fids = [_dec(f) for f in await self.redis.lrange(chunks_key(session_id), 0, -1)]
        freed = 0
        for fid in fids:
            freed += await self.blobs.delete(fid)
            await self.redis.lrem(chunks_key(session_id), 0, fid)
        pipe = self.redis.pipeline()
        pipe.delete(chunks_key(session_id), meta_key(session_id))
        pipe.srem(PENDING_KEY, session_id)
        if user_id:
            pipe.srem(user_key(user_id), session_id)
        await pipe.execute()
        return freed

    async def _user_erased(self, user_id: str) -> bool:
        return bool(await self.redis.exists(VOICE_DISABLED_KEY.format(user_id=user_id))
                    or await self.redis.sismember(ERASED_USERS_KEY, user_id))

    async def finalize(self, session_id: str) -> str:
        """Join a session's chunks into its WAV.  Returns what happened (stored / empty / erased / gone)."""
        meta = {_dec(k): _dec(v)
                for k, v in (await self.redis.hgetall(meta_key(session_id))).items()}
        if not meta:
            await self.redis.srem(PENDING_KEY, session_id)
            return "gone"                                  # already finalized (or erased)
        user_id = meta.get("user_id", "")
        if await self._user_erased(user_id):
            await self._discard_pending(session_id, user_id)
            return "erased"
        fids = [_dec(f) for f in await self.redis.lrange(chunks_key(session_id), 0, -1)]
        slots = asyncio.Semaphore(8)

        async def fetch(fid):
            async with slots:
                return await self.blobs.get(fid)

        pcm = b"".join(await asyncio.gather(*(fetch(fid) for fid in fids)))
        if not pcm:
            await self._discard_pending(session_id, user_id)
            return "empty"
        if self.db is None:
            raise RuntimeError("no rt_audio_recordings database configured (RT_POSTGRES_DSN / POSTGRES_*)")
        data = wav_bytes(pcm, int(meta.get("sample_rate") or SAMPLE_RATE))
        fid = await self.blobs.put(data, f"{session_id}.wav", "audio/wav")
        duration = round(len(pcm) / 2 / SAMPLE_RATE)
        if not await self.db.insert(user_id, session_id, fid, duration, len(data), self.retention_days):
            await _quiet_delete(self.blobs, fid)            # redelivered: the first one is kept
        await self._discard_pending(session_id, user_id)
        logger.info("session %s: recording stored (%d s, %d chunks)", session_id, duration, len(fids))
        return "stored"

    async def erase_user(self, user_id: str) -> dict[str, int]:
        erased, freed = 0, 0
        for raw in await self.redis.smembers(user_key(user_id)):
            session_id = _dec(raw)
            freed += await self._discard_pending(session_id, user_id)
            erased += 1
        await self.redis.delete(user_key(user_id))
        if self.db is not None:
            try:
                rows = await self.db.live_for_user(user_id)
            except NoDatabase:
                rows = []                                  # recording never set up: nothing stored
            for rec in rows:
                freed += await self.blobs.delete(rec.fid)
                await self.db.mark_erased([rec.id])
                erased += 1
        return {"erased_sessions": erased, "bytes_freed": freed}

    async def manifest(self, user_id: str) -> dict[str, Any]:
        """What Module 33's export lists under voice/recordings.json: metadata of each stored
        recording (never the audio or its SeaweedFS fid) and how many sessions are still recording."""
        pending = await self.redis.scard(user_key(user_id))
        rows: list = []
        if self.db is not None:
            try:
                rows = await self.db.live_for_user(user_id)
            except NoDatabase:
                rows = []                                  # recording never set up: nothing stored
        recordings = [{
            "session_id": rec.session_id,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
            "expires_at": rec.expires_at.isoformat() if rec.expires_at else None,
            "duration_s": rec.duration_s,
            "size_bytes": rec.size_bytes,
        } for rec in rows]
        return {"recordings": recordings, "pending_sessions": int(pending)}

    async def enforce_retention(self) -> int:
        if self.db is None:
            return 0
        rows = await self.db.expired()
        for rec in rows:
            await self.blobs.delete(rec.fid)
            await self.db.mark_erased([rec.id])
        return len(rows)

    async def orphans(self, active: set[str], idle_s: float = 300.0) -> list[str]:
        """Pending recordings whose session is gone without a session_ended message."""
        out = []
        now = time.time()
        for raw in await self.redis.smembers(PENDING_KEY):
            session_id = _dec(raw)
            if session_id in active:
                continue
            meta = await self.redis.hgetall(meta_key(session_id))
            stamp = meta.get(b"updated_at") or meta.get("updated_at") or meta.get(b"started_at") or meta.get("started_at")
            if not meta or now - float(stamp or 0) >= idle_s:
                out.append(session_id)
        return out
