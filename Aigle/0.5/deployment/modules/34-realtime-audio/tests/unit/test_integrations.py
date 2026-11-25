"""RTA-X6 / RTA-X7 / RA-9: Module 23 screening, Module 25 indexing, Module 26 archiving, consent-aware
recording, finalizing, erasure and retention."""
from __future__ import annotations

import asyncio
import hashlib
import io
import time
import wave
from dataclasses import replace

import fakeredis
import httpx
import numpy as np
import pytest
from app.blobstore import BlobError, SeaweedFS
from app.config import Settings
from app.integrations import MemoryArchive, compute_event_id, transcript_event
from app.recording import (
    PENDING_KEY,
    ConsentChecker,
    Recorder,
    RecordingService,
    chunks_key,
    meta_key,
    user_key,
)
from fakeredis import aioredis

from .conftest import (
    FakeBlobs,
    FakeChat,
    FakeKafka,
    FakeRecordingsDb,
    FakeTts,
    FakeWhisper,
    issue_ticket,
    json_handler,
    make_settings,
    silence,
    speech,
)
from .test_session_control import Session
from .test_stream import hgetall

REFUSAL = Settings().guardrail_block_text


def guard(decisions: dict[str, str] | None = None, default: str = "allow", status: int = 200):
    """Module 23 stand-in: ``decisions`` maps a substring of the content to a decision."""

    def decide(_request, body):
        if status != 200:
            return status, {"detail": {"type": "guardrail_unavailable"}}
        decision = next((d for k, d in (decisions or {}).items() if k in body["content"]), default)
        out = {"decision": decision, "reason": f"test {decision}", "request_id": body["request_id"], "latency_ms": 1.0,
               "redacted_content": body["content"].replace("0912345678", "[PHONE]") if decision == "redact" else None}
        return 200, out

    return json_handler(decide)


def run_turn(make_client, chat=None, stt=None, guardrail=None, kafka=None, settings=None, **services):
    chat = chat if chat is not None else FakeChat(answer="好的。|我幫您查一下。")
    routes = {"/guardrail/": guardrail} if guardrail is not None else {}
    routes.update(services)
    client = make_client(stt=stt or FakeWhisper(), tts=FakeTts(), chat=chat, services=routes, kafka=kafka,
                         settings=settings or make_settings(stt_streaming=False, guardrail_enabled=guardrail is not None))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'user-7')}") as ws:
        s = Session(ws)
        ready = s.start()
        s.send_pcm(np.concatenate([speech(800), silence(500)]))
        ws.send_json({"type": "END"})
        s.drain()
    return s, chat, ready, client


def spoken(s: Session) -> list[str]:
    return [e["text"] for e in s.events("RESPONSE_TEXT")]


# ---------------------------------------------------------------- RTA-X6 guardrail


def test_allowed_transcript_and_answer_are_screened_with_direction(make_client):
    g = guard()
    s, chat, ready, _ = run_turn(make_client, guardrail=g)
    assert spoken(s) == ["好的。", "我幫您查一下。"] and len(chat.requests) == 1
    calls = [c["body"] for c in g.calls]
    assert [c["direction"] for c in calls] == ["input", "output", "output"]
    assert calls[0]["content"] == "你好嗎" and calls[1]["content"] == "好的。"
    assert {c["module"] for c in calls} == {"34"} and {c["user_id"] for c in calls} == {"user-7"}
    assert {c["session_id"] for c in calls} == {ready["session_id"]}
    assert len({c["request_id"] for c in calls}) == 1                  # one request id for the whole turn
    assert "x-guardrail-warning" not in chat.requests[0]["headers"]


def test_blocked_transcript_is_refused_without_asking_module_15(make_client):
    kafka = FakeKafka()
    s, chat, _, client = run_turn(make_client, guardrail=guard({"你好": "block"}), kafka=kafka,
                                  settings=make_settings(stt_streaming=False, guardrail_enabled=True,
                                                         index_transcripts=True))
    block = s.events("GUARDRAIL_BLOCK")
    assert len(block) == 1 and block[0]["direction"] == "input" and block[0]["utterance"] == 1
    assert chat.requests == [] and spoken(s) == [REFUSAL]
    assert kafka.topic("personal-index-requests") == []                # a blocked transcript is not indexed
    assert "rt_guardrail_blocks_total 1.0" in client.get("/metrics").text


def test_warned_transcript_goes_on_with_the_warning_header(make_client):
    s, chat, _, _ = run_turn(make_client, guardrail=guard({"你好": "warn"}))
    assert chat.requests[0]["headers"]["x-guardrail-warning"] == "true" and spoken(s)


def test_redacted_transcript_reaches_module_15_and_the_index_redacted(make_client):
    kafka = FakeKafka()
    stt = FakeWhisper(words=[("我的電話是", 0, 300), ("0912345678", 300, 600)])
    _, chat, _, _ = run_turn(make_client, stt=stt, guardrail=guard({"0912": "redact"}), kafka=kafka,
                             settings=make_settings(stt_streaming=False, guardrail_enabled=True, index_transcripts=True))
    assert chat.requests[0]["body"]["message"] == "我的電話是[PHONE]"
    assert kafka.topic("personal-index-requests")[0]["payload"]["transcript"] == "我的電話是[PHONE]"


def test_blocked_answer_sentence_is_replaced_and_the_rest_dropped(make_client):
    chat = FakeChat(answer="第一句。|不能說的第二句。|第三句。")
    s, _, _, _ = run_turn(make_client, chat=chat, guardrail=guard({"不能說": "block"}))
    assert spoken(s) == ["第一句。", REFUSAL]
    block = s.events("GUARDRAIL_BLOCK")
    assert len(block) == 1 and block[0]["direction"] == "output"
    assert s.events("RESPONSE_END")[0]["sentences"] == 2


def test_tts_failure_warns_response_failed_and_counts_as_a_model_error(make_client):
    client = make_client(stt=FakeWhisper(), tts=FakeTts(fail=True), chat=FakeChat(answer="好的。"),
                         settings=make_settings(stt_streaming=False))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'user-7')}") as ws:
        s = Session(ws)
        s.start()
        s.send_pcm(np.concatenate([speech(800), silence(500)]))
        ws.send_json({"type": "END"})
        s.drain()
    warnings = s.events("WARNING")
    assert [w["code"] for w in warnings] == ["response_failed"]
    assert "rt_response_failed_total 1.0" in client.get("/metrics").text


@pytest.mark.parametrize(("fail_open", "answered"), [(True, True), (False, False)])
def test_guardrail_unavailable(make_client, fail_open, answered):
    s, chat, _, _ = run_turn(make_client, guardrail=guard(status=503),
                             settings=make_settings(stt_streaming=False, guardrail_enabled=True,
                                                    guardrail_fail_open=fail_open))
    assert bool(chat.requests) is answered
    assert spoken(s) == (["好的。", "我幫您查一下。"] if answered else [REFUSAL])


def test_output_screening_can_be_turned_off(make_client):
    g = guard()
    run_turn(make_client, guardrail=g, settings=make_settings(stt_streaming=False, guardrail_enabled=True,
                                                              guardrail_output=False))
    assert [c["body"]["direction"] for c in g.calls] == ["input"]


# ---------------------------------------------------------------- RTA-X6 Module 25 / Module 26


def test_transcript_event_matches_the_personal_index_contract():
    ev = transcript_event("u1", "s1", 3, "你好嗎", language="zh", start_ms=1200, end_ms=2400)
    assert ev["schema_version"] == "personal-index-v1" and ev["source_module"] == "34"
    raw = "realtime-audio/s1\x1fs1:3\x1fu1"                        # Module 25's separator
    assert ev["event_id"] == compute_event_id("realtime-audio/s1", "s1:3", "u1") == hashlib.sha256(raw.encode()).hexdigest()
    payload = ev["payload"]
    assert (payload["branch_id"], payload["user_id"], payload["session_id"], payload["transcript"]) == ("u1", "u1", "s1", "你好嗎")
    assert payload["timestamp"]
    chunk = payload["parameters"]["chunks"][0]
    assert chunk["id"] == hashlib.sha256(b"s1:3").hexdigest()
    p = chunk["payload"]
    assert (p["type"], p["text"], p["start_time"], p["end_time"], p["speaker"]) == ("audios", "你好嗎", 1.2, 2.4, "user")
    assert payload["parameters"]["asset_path"] == p["asset_path"] == "realtime-audio/s1"


def test_final_transcripts_are_indexed_even_without_a_voice_reply(make_client):
    kafka = FakeKafka()
    client = make_client(stt=FakeWhisper(), kafka=kafka, settings=make_settings(stt_streaming=False, index_transcripts=True))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'user-9')}") as ws:
        s = Session(ws)
        sid = s.start()["session_id"]
        s.send_pcm(np.concatenate([speech(800), silence(500), speech(800), silence(500)]))
        ws.send_json({"type": "END"})
        s.drain()
    events = kafka.topic("personal-index-requests")
    assert [e["payload"]["parameters"]["version_id"] for e in events] == [f"{sid}:1", f"{sid}:2"]
    assert {k for t, _, k in kafka.sent} == {"user-9"}


def test_module_15_turns_are_not_archived_twice(make_client):
    memory = json_handler(lambda _r, _b: (200, {"frame_id": "f", "turn_index": 0, "session_id": "s"}))
    s, _, _, _ = run_turn(make_client, **{"/memory/": memory})
    assert spoken(s) and memory.calls == []                            # Module 15 archives its own turns


def test_memory_archive_posts_the_turn():
    calls = json_handler(lambda _r, _b: (200, {"frame_id": "f", "turn_index": 0, "session_id": "s1"}))
    archive = MemoryArchive("http://mem:8026", httpx.AsyncClient(transport=httpx.MockTransport(calls)))
    assert asyncio.run(archive.append_turn("u1", "s1", "你好嗎", "很好。")) is True
    call = calls.calls[0]
    assert call["url"] == "http://mem:8026/memory/sessions/s1/turns" and call["headers"]["x-user-id"] == "u1"
    assert (call["body"]["user_message"], call["body"]["assistant_response"]) == ("你好嗎", "很好。")


# ---------------------------------------------------------------- RTA-X7 consent


def consent(granted: bool | None = True, status: int = 200):
    def answer(_request, _body):
        if status != 200:
            return status, {"detail": "nope"}
        purposes = [] if granted is None else [{"id": "voice_recording", "granted": granted, "description": "x"}]
        return 200, {"purposes": [{"id": "memory", "granted": True, "description": "x"}, *purposes]}

    return json_handler(answer)


@pytest.mark.parametrize(("enabled", "flag", "handler", "expected"), [
    (False, False, consent(True), (False, "disabled_by_config")),
    (True, True, consent(True), (False, "consent_withdrawn")),
    (True, False, consent(True), (True, None)),
    (True, False, consent(False), (False, "no_consent")),
    (True, False, consent(None), (False, "no_consent")),
    (True, False, consent(status=404), (False, "no_consent")),
    (True, False, consent(status=503), (False, "module33_unavailable")),
])
def test_consent_check(enabled, flag, handler, expected):
    redis = aioredis.FakeRedis(server=fakeredis.FakeServer())

    async def scenario():
        if flag:
            await redis.set("gdpr:voice_recording_disabled:u1", "1")
        checker = ConsentChecker(replace(make_settings(), recording_enabled=enabled), redis,
                                 httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        return await checker.check("u1")

    assert asyncio.run(scenario()) == expected
    if expected[0] or expected[1] in ("no_consent", "module33_unavailable"):
        call = handler.calls[-1]
        assert call["headers"]["x-user-id"] == "u1" and call["headers"]["x-internal-token"] == "test-internal-token"


@pytest.mark.parametrize("payload", [None, [], {"purposes": None}, {"purposes": {}},
                                     {"purposes": [None]}, {"purposes": [{"id": [], "granted": True}]},
                                     {"purposes": [{"id": "voice_recording", "granted": "true"}]}])
def test_malformed_consent_fails_closed(payload):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda _r: httpx.Response(200, json=payload))) as client:
            checker = ConsentChecker(make_settings(recording_enabled=True), aioredis.FakeRedis(), client)
            assert await checker.check("u1") == (False, "module33_unavailable")

    asyncio.run(scenario())


def recording_client(make_client, granted=True, **settings):
    kafka, blobs, db = FakeKafka(), FakeBlobs(), FakeRecordingsDb()
    client = make_client(stt=FakeWhisper(), kafka=kafka, blobs=blobs, recordings_db=db,
                         services={"/consent": consent(granted)},
                         settings=make_settings(stt_streaming=False, recording_enabled=True, recording_chunk_s=0.5,
                                                **settings))
    return client, kafka, blobs, db


def test_consented_session_is_recorded_in_chunks_and_finalize_is_requested(make_client, sync_redis):
    client, kafka, blobs, _ = recording_client(make_client)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'user-r')}") as ws:
        s = Session(ws)
        ready = s.start()
        sid = ready["session_id"]
        s.send_pcm(np.concatenate([speech(800), silence(1400)]))
        ws.send_json({"type": "END"})
        s.drain()
    assert ready["recording"] is True
    record = hgetall(sync_redis, sid)
    assert record["recording_enabled"] == "1" and record.get("recording_skipped_reason", "") == ""
    fids = [f.decode() for f in sync_redis.lrange(chunks_key(sid), 0, -1)]
    assert len(fids) >= 4 and sum(len(blobs.blobs[f]) for f in fids) == 2200 * 32     # all of it, 0.5 s chunks
    msg = kafka.topic("audio-recording-finalize")
    assert msg == [{"event": "session_ended", "session_id": sid, "user_id": "user-r", "wav_chunks_key": chunks_key(sid),
                    "sample_rate": 16000, "samples": 2200 * 16, "reason": "end"}]
    assert sync_redis.hget(meta_key(sid), "state") == b"ended" and sync_redis.sismember(PENDING_KEY, sid)


@pytest.mark.parametrize(("granted", "reason"), [(False, "no_consent"), (None, "no_consent")])
def test_session_without_consent_is_not_recorded(make_client, sync_redis, granted, reason):
    client, kafka, blobs, _ = recording_client(make_client, granted=granted)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'user-n')}") as ws:
        s = Session(ws)
        ready = s.start()
        s.send_pcm(np.concatenate([speech(800), silence(1400)]))
        ws.send_json({"type": "END"})
        s.drain()
    record = hgetall(sync_redis, ready["session_id"])
    assert ready["recording"] is False and record["recording_enabled"] == "0"
    assert record["recording_skipped_reason"] == reason
    assert blobs.blobs == {} and kafka.topic("audio-recording-finalize") == []


def test_recording_switched_off_for_everyone(make_client, sync_redis):
    client = make_client(stt=FakeWhisper(), kafka=FakeKafka(), blobs=FakeBlobs(),
                         services={"/consent": consent(True)}, settings=make_settings(recording_enabled=False))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        ready = s.start()
        ws.send_json({"type": "END"})
        s.drain()
    assert ready["recording"] is False
    assert hgetall(sync_redis, ready["session_id"])["recording_skipped_reason"] == "disabled_by_config"


# ---------------------------------------------------------------- finalize, erasure, retention


async def _record(redis, blobs, kafka, session_id="s1", user_id="u1", seconds=1.2):
    rec = Recorder(blobs, redis, kafka, "audio-recording-finalize", session_id, user_id, chunk_s=0.5)
    await rec.start()
    pcm = (np.sin(np.arange(int(16000 * seconds)) / 10) * 8000).astype(np.int16)
    for i in range(0, len(pcm), 1600):
        rec.add(pcm[i:i + 1600])
        await asyncio.sleep(0)
    await rec.finish("end")
    return pcm


def test_finalize_joins_the_chunks_into_one_wav():
    redis, blobs, kafka, db = aioredis.FakeRedis(server=fakeredis.FakeServer()), FakeBlobs(), FakeKafka(), FakeRecordingsDb()

    async def scenario():
        pcm = await _record(redis, blobs, kafka)
        chunk_fids = [f.decode() for f in await redis.lrange(chunks_key("s1"), 0, -1)]
        svc = RecordingService(blobs, redis, db, retention_days=90)
        first, again = await svc.finalize("s1"), await svc.finalize("s1")
        left = {k for k in (meta_key("s1"), chunks_key("s1")) if await redis.exists(k)}
        return pcm, chunk_fids, first, again, left, await redis.sismember(PENDING_KEY, "s1")

    pcm, chunk_fids, first, again, left, pending = asyncio.run(scenario())
    assert (first, again, left, pending) == ("stored", "gone", set(), 0)
    # how the audio was cut into chunks depends on task timing; that all of it is there does not
    assert len(chunk_fids) >= 2 and not any(f in blobs.blobs for f in chunk_fids)       # chunks deleted
    row = db.rows[0]
    assert (row["user_id"], row["session_id"], row["duration_s"]) == ("u1", "s1", 1)
    assert (row["expires_at"] - row["created_at"]).days == 90
    with wave.open(io.BytesIO(blobs.blobs[row["fid"]])) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert np.array_equal(np.frombuffer(w.readframes(w.getnframes()), dtype="<i2"), pcm)


def test_finalize_of_an_erased_user_stores_nothing():
    redis, blobs, kafka, db = aioredis.FakeRedis(server=fakeredis.FakeServer()), FakeBlobs(), FakeKafka(), FakeRecordingsDb()

    async def scenario():
        await _record(redis, blobs, kafka)
        await redis.set("gdpr:voice_recording_disabled:u1", "1")
        return await RecordingService(blobs, redis, db).finalize("s1")

    assert asyncio.run(scenario()) == "erased" and db.rows == [] and blobs.blobs == {}


def test_erase_user_removes_pending_and_stored_recordings():
    redis, blobs, kafka, db = aioredis.FakeRedis(server=fakeredis.FakeServer()), FakeBlobs(), FakeKafka(), FakeRecordingsDb()

    async def scenario():
        svc = RecordingService(blobs, redis, db)
        await _record(redis, blobs, kafka, "s1")
        await svc.finalize("s1")                                       # stored
        await _record(redis, blobs, kafka, "s2")                       # still pending
        await _record(redis, blobs, kafka, "s3", user_id="someone-else")
        result = await svc.erase_user("u1")
        return result, await redis.exists(user_key("u1")), await redis.exists(chunks_key("s3"))

    result, user_set, other = asyncio.run(scenario())
    assert result["erased_sessions"] == 2 and result["bytes_freed"] > 2 * 1.2 * 32000
    assert user_set == 0 and other == 1                                # only that user's data
    assert db.rows[0]["erased_at"] is not None
    assert list(blobs.blobs) and all(not v.startswith(b"RIFF") for v in blobs.blobs.values())


def test_retention_deletes_expired_recordings():
    from datetime import datetime, timedelta, timezone

    redis, blobs, kafka, db = aioredis.FakeRedis(server=fakeredis.FakeServer()), FakeBlobs(), FakeKafka(), FakeRecordingsDb()

    async def scenario():
        svc = RecordingService(blobs, redis, db, retention_days=90)
        await _record(redis, blobs, kafka, "old")
        await svc.finalize("old")
        await _record(redis, blobs, kafka, "new")
        await svc.finalize("new")
        db.rows[0]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        return await svc.enforce_retention()

    assert asyncio.run(scenario()) == 1
    assert db.rows[0]["erased_at"] is not None and db.rows[1]["erased_at"] is None and len(blobs.blobs) == 1


@pytest.mark.parametrize("pending", [False, True])
def test_failed_gdpr_delete_returns_503_and_can_be_retried(make_client, pending):
    class UnavailableBlobs(FakeBlobs):
        unavailable = False

        async def delete(self, fid):
            if self.unavailable:
                raise BlobError("SeaweedFS offline")
            return await super().delete(fid)

    blobs, db = UnavailableBlobs(), FakeRecordingsDb()
    client = make_client(blobs=blobs, recordings_db=db)
    redis = client.app.state.recordings.redis

    async def seed():
        await _record(redis, blobs, FakeKafka())
        if not pending:
            await client.app.state.recordings.finalize("s1")

    client.portal.call(seed)
    blobs.unavailable = True
    url = "/internal/gdpr/user/u1"
    headers = {"X-Internal-Token": "test-internal-token"}
    assert client.delete(url, headers=headers).status_code == 503
    assert blobs.blobs
    if pending:
        assert client.portal.call(redis.sismember, PENDING_KEY, "s1")
        assert client.portal.call(redis.llen, chunks_key("s1")) > 0
    else:
        assert db.rows[0]["erased_at"] is None
    blobs.unavailable = False
    response = client.delete(url, headers=headers)
    assert response.status_code == 200
    assert response.json()["erased_sessions"] == 1
    assert blobs.blobs == {}
    assert client.delete(url, headers=headers).json()["bytes_freed"] == 0


def test_retention_keeps_failed_rows_for_retry():
    class UnavailableBlobs(FakeBlobs):
        unavailable = True

        async def delete(self, fid):
            if self.unavailable and fid == "bad":
                raise BlobError("SeaweedFS offline")
            return await super().delete(fid)

    async def scenario():
        blobs, db = UnavailableBlobs(), FakeRecordingsDb()
        for fid in ("good", "bad"):
            blobs.blobs[fid] = b"wav"
            await db.insert("u1", fid, fid, 1, 3, -1)
        service = RecordingService(blobs, aioredis.FakeRedis(), db)
        with pytest.raises(BlobError):
            await service.enforce_retention()
        assert db.rows[0]["erased_at"] is not None
        assert db.rows[1]["erased_at"] is None
        assert list(blobs.blobs) == ["bad"]
        blobs.unavailable = False
        assert await service.enforce_retention() == 1
        assert blobs.blobs == {}

    asyncio.run(scenario())


@pytest.mark.parametrize("status,payload", [(503, {}), (200, {"locations": []})])
def test_seaweedfs_lookup_failure_does_not_confirm_erasure(status, payload):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda _r: httpx.Response(status, json=payload))) as client:
            with pytest.raises(BlobError):
                await SeaweedFS("http://master:9333", client).delete("7,abc")

    asyncio.run(scenario())


def test_orphaned_recordings_are_found_once_their_session_is_gone():
    redis, blobs, kafka = aioredis.FakeRedis(server=fakeredis.FakeServer()), FakeBlobs(), FakeKafka()

    async def scenario():
        svc = RecordingService(blobs, redis, None)
        for sid in ("live", "crashed", "fresh"):
            rec = Recorder(blobs, redis, kafka, "t", sid, "u1", chunk_s=0.1)
            await rec.start()
        old = time.time() - 600
        for sid in ("live", "crashed"):
            await redis.hset(meta_key(sid), "updated_at", old)
        return sorted(await svc.orphans(active={"live"}, idle_s=300))

    assert asyncio.run(scenario()) == ["crashed"]


def test_consumer_sweep_finalizes_recordings_of_crashed_sessions():
    from app.recording_consumer import sweep
    from app.session import ACTIVE_KEY

    redis, blobs, kafka, db = aioredis.FakeRedis(server=fakeredis.FakeServer()), FakeBlobs(), FakeKafka(), FakeRecordingsDb()

    async def scenario():
        svc = RecordingService(blobs, redis, db)
        for sid in ("crashed", "alive"):
            rec = Recorder(blobs, redis, kafka, "t", sid, "u1", chunk_s=0.1)
            await rec.start()
            rec.add(np.ones(3200, dtype=np.int16))
            await asyncio.gather(*rec._tasks)
            await redis.hset(meta_key(sid), "updated_at", time.time() - 600)
        now = time.time()
        await redis.zadd(ACTIVE_KEY, {"crashed": now - 120, "alive": now})   # stale vs fresh heartbeat
        await sweep(svc, redis, retention_every=1, count=[0], idle_s=300, heartbeat_s=10)
        return [r["session_id"] for r in db.rows], await redis.sismember(PENDING_KEY, "alive")

    assert asyncio.run(scenario()) == (["crashed"], 1)


def test_gdpr_endpoint(make_client):
    client, kafka, blobs, _ = recording_client(make_client)
    url = "/internal/gdpr/user/user-g"
    assert client.delete(url).status_code == 401
    assert client.delete(url, headers={"X-Internal-Token": "wrong"}).status_code == 401
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'user-g')}") as ws:
        s = Session(ws)
        s.start()
        s.send_pcm(np.concatenate([speech(800), silence(700)]))
        s.until("UTTERANCE")
        resp = client.delete(url, headers={"X-Internal-Token": "test-internal-token"})   # mid-session
        s.send_pcm(silence(1000))
        ws.send_json({"type": "END"})
        s.drain()
    assert resp.status_code == 200 and resp.json()["user_id"] == "user-g"
    assert resp.json()["erased_sessions"] == 1 and resp.json()["bytes_freed"] > 0
    assert blobs.blobs == {} and kafka.topic("audio-recording-finalize") == []        # stopped, nothing left
    assert client.delete(url, headers={"X-Internal-Token": "test-internal-token"}).json()["erased_sessions"] == 0


def test_gdpr_endpoint_without_recordings_configured(make_client):
    client = make_client(settings=make_settings(recording_enabled=False))
    resp = client.delete("/internal/gdpr/user/u1", headers={"X-Internal-Token": "test-internal-token"})
    assert resp.status_code == 200 and resp.json() == {"user_id": "u1", "erased_sessions": 0, "bytes_freed": 0}


def test_gdpr_endpoint_when_the_recordings_database_was_never_created(make_client):
    from app.recordings_db import NoDatabase

    class Missing(FakeRecordingsDb):
        async def live_for_user(self, user_id):
            raise NoDatabase('database "realtime_audio" does not exist')

    client = make_client(blobs=FakeBlobs(), recordings_db=Missing(), services={"/consent": consent(True)},
                         settings=make_settings(recording_enabled=False))
    resp = client.delete("/internal/gdpr/user/u1", headers={"X-Internal-Token": "test-internal-token"})
    assert resp.status_code == 200 and resp.json()["erased_sessions"] == 0


def test_gdpr_recordings_lists_metadata_never_audio(make_client):
    """GD-10: GET /internal/gdpr/user/{id}/recordings is Module 33's voice/recordings.json."""
    blobs, db = FakeBlobs(), FakeRecordingsDb()
    client = make_client(blobs=blobs, recordings_db=db)
    redis = client.app.state.recordings.redis

    async def seed():
        await _record(redis, blobs, FakeKafka(), session_id="s1", user_id="u1")
        await client.app.state.recordings.finalize("s1")             # stored
        await _record(redis, blobs, FakeKafka(), session_id="s2", user_id="u1")   # still recording
        await _record(redis, blobs, FakeKafka(), session_id="s3", user_id="other")
        await client.app.state.recordings.finalize("s3")

    client.portal.call(seed)
    url = "/internal/gdpr/user/u1/recordings"
    assert client.get(url).status_code == 401
    assert client.get(url, headers={"X-Internal-Token": "wrong"}).status_code == 401
    body = client.get(url, headers={"X-Internal-Token": "test-internal-token"}).json()
    assert body["user_id"] == "u1" and body["pending_sessions"] == 1
    [rec] = body["recordings"]
    assert set(rec) == {"session_id", "created_at", "expires_at", "duration_s", "size_bytes"}   # no fid, no audio
    assert rec["session_id"] == "s1" and rec["size_bytes"] > 0 and rec["expires_at"] > rec["created_at"]

    assert client.delete("/internal/gdpr/user/u1", headers={"X-Internal-Token": "test-internal-token"}).status_code == 200
    after = client.get(url, headers={"X-Internal-Token": "test-internal-token"}).json()
    assert after["recordings"] == [] and after["pending_sessions"] == 0      # erased rows are not listed


def test_gdpr_recordings_without_recordings_configured(make_client):
    client = make_client(settings=make_settings(recording_enabled=False))
    resp = client.get("/internal/gdpr/user/u1/recordings", headers={"X-Internal-Token": "test-internal-token"})
    assert resp.status_code == 200 and resp.json() == {"user_id": "u1", "recordings": [], "pending_sessions": 0}


def test_gdpr_recordings_when_the_recordings_database_was_never_created(make_client):
    from app.recordings_db import NoDatabase

    class Missing(FakeRecordingsDb):
        async def live_for_user(self, user_id):
            raise NoDatabase('database "realtime_audio" does not exist')

    client = make_client(blobs=FakeBlobs(), recordings_db=Missing(), services={"/consent": consent(True)},
                         settings=make_settings(recording_enabled=False))
    resp = client.get("/internal/gdpr/user/u1/recordings", headers={"X-Internal-Token": "test-internal-token"})
    assert resp.status_code == 200 and resp.json()["recordings"] == []


def test_seaweedfs_client():
    calls: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.url.path == "/dir/assign":
            return httpx.Response(200, json={"fid": "7,01ab", "url": "vol1:8081"})
        if request.url.path == "/dir/lookup":
            return httpx.Response(200, json={"locations": [{"url": "vol1:8081"}]})
        if request.method == "POST":
            return httpx.Response(201, json={"size": 4})
        if request.method == "GET":
            return httpx.Response(200, content=b"data")
        return httpx.Response(202, json={"size": 4})

    blobs = SeaweedFS("http://master:9333", httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def scenario():
        fid = await blobs.put(b"data", "a.wav", "audio/wav")
        return fid, await blobs.get(fid), await blobs.delete(fid)

    assert asyncio.run(scenario()) == ("7,01ab", b"data", 4)
    assert calls[0] == ("GET", "http://master:9333/dir/assign") and calls[1] == ("POST", "http://vol1:8081/7,01ab")
    assert ("GET", "http://master:9333/dir/lookup?volumeId=7") in calls and calls[-1] == ("DELETE", "http://vol1:8081/7,01ab")


def test_settings_from_env(monkeypatch):
    for name, value in {"RT_GUARDRAIL_ENABLED": "false", "RT_GUARDRAIL_FAIL_OPEN": "false", "RT_RECORDING_ENABLED": "true",
                        "RT_RECORDING_RETENTION_DAYS": "30", "POSTGRES_HOST": "pg", "POSTGRES_PASSWORD": "p@ss",
                        "KAFKA_BOOTSTRAP_SERVERS": "k1:9092", "GD_INTERNAL_TOKEN": "tok"}.items():
        monkeypatch.setenv(name, value)
    s = Settings.from_env()
    assert (s.guardrail_enabled, s.guardrail_fail_open, s.recording_enabled, s.recording_retention_days) == (False, False, True, 30)
    assert s.postgres_dsn == "postgresql://raptor:p%40ss@pg:5432/realtime_audio"
    assert (s.kafka_bootstrap, s.gd_internal_token) == ("k1:9092", "tok")
    d = Settings()
    assert (d.guardrail_enabled, d.guardrail_output, d.index_transcripts, d.memory_enabled, d.recording_enabled) == \
        (True, True, True, True, False)
