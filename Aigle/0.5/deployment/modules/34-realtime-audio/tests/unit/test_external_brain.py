"""RTA-X9 / RTA-X10: external-brain sessions (SPEAK / SPEAK_CANCEL), subject delegation and
per-session side effects."""
from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest
from app.config import Settings

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
from .test_integrations import guard
from .test_session_control import Session
from .test_stream import hgetall

BRAIN = ["rt-external-brain"]
CS = ["rt-external-brain", "rt-delegate"]                    # a call-centre service account
EXTERNAL = {"codec": "pcm", "sample_rate": 16000, "language": "zh-TW", "brain": "external"}


def external_client(make_client, chat=None, guardrail=None, kafka=None, services=None, **settings):
    routes = dict(services or {})
    if guardrail is not None:
        routes["/guardrail/"] = guardrail
    return make_client(stt=FakeWhisper(), tts=FakeTts(), chat=chat or FakeChat(), services=routes, kafka=kafka,
                       settings=make_settings(stt_streaming=False, guardrail_enabled=guardrail is not None, **settings))


def say_question(s: Session) -> dict:
    s.send_pcm(np.concatenate([speech(800), silence(500)]))
    return s.until("TRANSCRIPT_FINAL")


def answers(s: Session) -> list[tuple[str, list[str]]]:
    """(request_id, its RESPONSE_TEXTs) per answer, in order."""
    out: dict[str, list[str]] = {}
    for f in s.frames:
        if isinstance(f, dict) and f["type"] == "RESPONSE_TEXT":
            out.setdefault(f["request_id"], []).append(f["text"])
    return list(out.items())


# ---------------------------------------------------------------- RTA-X9


def test_external_brain_needs_its_role(make_client):
    client = external_client(make_client)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'alice')}") as ws:
        ws.send_json(EXTERNAL)
        s = Session(ws)
        assert s.drain() == 4403
    assert s.events("ERROR")[-1]["code"] == "BRAIN_MODE_FORBIDDEN"


def test_external_brain_speaks_what_the_client_says_and_never_asks_module_15(make_client):
    chat = FakeChat()
    client = external_client(make_client, chat=chat)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'cs', BRAIN)}") as ws:
        s = Session(ws)
        ready = s.start(EXTERNAL)
        final = say_question(s)
        ws.send_json({"type": "SPEAK", "utterance_id": "a1", "text": "您好，", "final": False})
        ws.send_json({"type": "SPEAK", "utterance_id": "a1", "text": "我幫您查一下。", "final": True})
        end = s.until("RESPONSE_END", request_id="a1")
        ws.send_json({"type": "END"})
        assert s.drain() == 1000
    assert ready["config"]["brain"] == "external" and final["text"] == "你好嗎"
    assert chat.requests == []                                         # Module 15 never asked
    assert answers(s) == [("a1", ["您好，", "我幫您查一下。"])]
    kinds = s.kinds()
    start, stop = kinds.index("RESPONSE_START"), kinds.index("RESPONSE_END")
    between = kinds[start:stop]
    assert between.count("RESPONSE_START") == 1 and "AUDIO" in between        # one seamless answer
    speech_ms = 50 * (len("您好，") + len("我幫您查一下。"))
    assert end["duration_ms"] == speech_ms + 200 and end["sentences"] == 2   # padded once, at the end
    assert ready["side_effects"] == {"record": False, "index_transcripts": False, "store_memory": False}


def test_answers_play_in_the_order_they_arrive(make_client):
    client = external_client(make_client)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'cs', BRAIN)}") as ws:
        s = Session(ws)
        s.start(EXTERNAL)
        say_question(s)
        ws.send_json({"type": "SPEAK", "utterance_id": "first", "text": "第一段。", "final": False})
        ws.send_json({"type": "SPEAK", "utterance_id": "second", "text": "第二段。", "final": True})
        ws.send_json({"type": "SPEAK", "utterance_id": "first", "text": "第一段結束。", "final": True})
        s.until("RESPONSE_END", request_id="second")
        ws.send_json({"type": "SPEAK", "utterance_id": "first", "text": "太晚了。", "final": True})
        warning = s.until("WARNING")
        ws.send_json({"type": "END"})
        s.drain()
    assert answers(s) == [("first", ["第一段。", "第一段結束。"]), ("second", ["第二段。"])]
    assert warning["code"] == "speak_after_final"


def test_speak_cancel_stops_one_answer(make_client):
    long = "這是一段很長的回答，" * 6
    client = external_client(make_client)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'cs', BRAIN)}") as ws:
        s = Session(ws)
        s.start(EXTERNAL)
        say_question(s)
        ws.send_json({"type": "SPEAK", "utterance_id": "long", "text": long, "final": False})
        ws.send_json({"type": "SPEAK", "utterance_id": "queued", "text": "不會念到。", "final": True})
        ws.send_json({"type": "SPEAK", "utterance_id": "after", "text": "接著念這句。", "final": True})
        s.until("RESPONSE_START", request_id="long")
        ws.send_json({"type": "SPEAK_CANCEL", "utterance_id": "queued"})
        ws.send_json({"type": "SPEAK_CANCEL", "utterance_id": "long"})
        cancelled = s.until("RESPONSE_END", request_id="long")
        s.until("RESPONSE_END", request_id="after")
        ws.send_json({"type": "END"})
        s.drain()
    assert cancelled.get("cancelled") is True
    assert [rid for rid, _ in answers(s)] == ["long", "after"]                  # "queued" never spoken


def test_barge_in_cancels_every_queued_speak(make_client):
    client = external_client(make_client, output_lead_ms=300)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'cs', BRAIN)}") as ws:
        s = Session(ws)
        s.start(EXTERNAL)
        say_question(s)
        # u1's text is long enough that its paced playback (~50ms/char) comfortably outlasts the
        # sleep below even on a slow/loaded CI runner -- a short u1 flaked once in CI (u2 already
        # playing by the time the barge audio landed) while always passing locally.
        ws.send_json({"type": "SPEAK", "utterance_id": "u1", "text": "今天天氣很好，適合出去走走，" * 10, "final": True})
        ws.send_json({"type": "SPEAK", "utterance_id": "u2", "text": "這句排在後面。", "final": True})
        s.until("RESPONSE_START", request_id="u1")
        time.sleep(0.8)
        s.send_pcm(np.concatenate([speech(600), silence(500)]))
        barge = s.until("BARGE_IN")
        s.until("TRANSCRIPT_FINAL", utterance=2)
        ws.send_json({"type": "END"})
        s.drain()
    assert barge["request_id"] == "u1"
    assert [rid for rid, _ in answers(s)] == ["u1"]                            # u2 dropped with it


def test_barge_in_cannot_start_the_next_speak_while_waiting_for_cancellation(make_client, monkeypatch):
    original_wait = asyncio.wait

    async def delayed_wait(*args, **kwargs):
        result = await original_wait(*args, **kwargs)
        if asyncio.current_task().get_name() == "review-barge":
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        return result

    monkeypatch.setattr(asyncio, "wait", delayed_wait)
    client = external_client(make_client)
    started = []
    ready = asyncio.Event()

    async def controlled_speak(speech):
        started.append(speech.id)
        ready.set()
        await asyncio.Event().wait()

    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'cs', BRAIN)}") as ws:
        s = Session(ws)
        sid = s.start(EXTERNAL)["session_id"]
        conn = client.app.state.stream_handler.live[sid]
        monkeypatch.setattr(conn, "_speak", controlled_speak)
        for uid in ("active", "queued"):
            ws.send_json({"type": "SPEAK", "utterance_id": uid, "text": "回答。", "final": True})
        async def queued():
            await ready.wait()
            while "queued" not in conn._speeches:
                await asyncio.sleep(0)

        client.portal.call(queued)

        async def interrupt():
            asyncio.current_task().set_name("review-barge")
            await conn._barge_in(conn._segmenter)

        client.portal.call(interrupt)
        ws.send_json({"type": "END"})
        assert s.drain() == 1000
    assert started == ["active"]


def test_completed_speak_ids_remain_deduplicated_after_300_answers(make_client):
    class TinyTts(FakeTts):
        sample_rate = 16000

        def stream(self, text, voice, language, speed):
            yield np.zeros(16, dtype=np.float32)

    # 301 real round trips can cross the default ping_interval_s on a loaded CI runner; Session
    # never answers PING on its own, so disable it here rather than let the test race it.
    client = make_client(tts=TinyTts(), settings=make_settings(output_pace=False, tts_silence_pad_ms=0,
                                                                ping_interval_s=0))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'cs', BRAIN)}") as ws:
        s = Session(ws)
        s.start(EXTERNAL)
        for i in range(301):
            uid = f"answer-{i}"
            ws.send_json({"type": "SPEAK", "utterance_id": uid, "text": "回答。", "final": True})
            s.until("RESPONSE_END", request_id=uid)
        ws.send_json({"type": "SPEAK", "utterance_id": "answer-0", "text": "重複。", "final": True})
        assert s.until("WARNING")["code"] == "speak_after_final"
        ws.send_json({"type": "END"})
        assert s.drain() == 1000
    assert s.events("ENDED")[-1]["responses"] == 301


def test_guardrail_refusal_is_counted_as_a_response(make_client):
    chat = FakeChat()
    client = external_client(make_client, chat=chat, guardrail=guard(default="block"))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        s.start()
        say_question(s)
        s.until("RESPONSE_END")
        ws.send_json({"type": "END"})
        assert s.drain() == 1000
    assert chat.requests == []
    assert s.events("ENDED")[-1]["responses"] == 1


def test_invalid_external_guardrail_configuration_stops_startup(monkeypatch):
    for value in ("outptu", "none", " "):
        with pytest.raises(ValueError, match="RT_EXTERNAL_GUARDRAIL"):
            Settings(external_guardrail=value)
        monkeypatch.setenv("RT_EXTERNAL_GUARDRAIL", value)
        with pytest.raises(ValueError, match="RT_EXTERNAL_GUARDRAIL"):
            Settings.from_env()
    for value in ("off", "input", "output", "both"):
        monkeypatch.setenv("RT_EXTERNAL_GUARDRAIL", value.upper())
        assert Settings.from_env().external_guardrail == value


def test_external_guardrail_screens_speak_text_by_default(make_client):
    g = guard({"不能說": "block"})
    client = external_client(make_client, guardrail=g)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'cs', BRAIN)}") as ws:
        s = Session(ws)
        s.start(EXTERNAL)
        say_question(s)
        ws.send_json({"type": "SPEAK", "utterance_id": "x", "text": "可以說。不能說的事。", "final": True})
        s.until("RESPONSE_END", request_id="x")
        ws.send_json({"type": "END"})
        s.drain()
    assert [c["body"]["direction"] for c in g.calls] == ["output", "output"]   # RT_EXTERNAL_GUARDRAIL=output
    assert answers(s) == [("x", ["可以說。", "抱歉，這個問題我無法協助處理。"])]
    assert s.events("GUARDRAIL_BLOCK")[0]["direction"] == "output"


@pytest.mark.parametrize(("mode", "directions"), [("off", []), ("input", ["input"]), ("both", ["input", "output"])])
def test_external_guardrail_modes(make_client, mode, directions):
    g = guard()
    client = external_client(make_client, guardrail=g, external_guardrail=mode)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'cs', BRAIN)}") as ws:
        s = Session(ws)
        s.start(EXTERNAL)
        say_question(s)
        ws.send_json({"type": "SPEAK", "utterance_id": "x", "text": "好的。", "final": True})
        s.until("RESPONSE_END", request_id="x")
        ws.send_json({"type": "END"})
        s.drain()
    assert [c["body"]["direction"] for c in g.calls] == directions


def test_speak_is_ignored_in_module15_sessions(make_client):
    client = external_client(make_client)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'u1')}") as ws:
        s = Session(ws)
        s.start()
        ws.send_json({"type": "SPEAK", "utterance_id": "x", "text": "好的。", "final": True})
        assert s.until("WARNING")["code"] == "unexpected_frame"
        ws.send_json({"type": "END"})
        s.drain()


# ---------------------------------------------------------------- RTA-X10


def recording_setup(make_client, granted_users):
    kafka, blobs, db = FakeKafka(), FakeBlobs(), FakeRecordingsDb()

    def answer(request, _body):
        uid = request.headers.get("x-user-id")
        return 200, {"purposes": [{"id": "voice_recording", "granted": uid in granted_users, "description": "x"}]}

    m33 = json_handler(answer)
    memory = json_handler(lambda _r, _b: (200, {"frame_id": "f", "turn_index": 0, "session_id": "s"}))
    client = make_client(stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat(), kafka=kafka, blobs=blobs, recordings_db=db,
                         services={"/consent": m33, "/memory/": memory},
                         settings=make_settings(stt_streaming=False, recording_enabled=True, recording_chunk_s=0.5,
                                                index_transcripts=True))
    return client, kafka, blobs, db, m33, memory


def delegated_call(client, subject, **flags):
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'cs-service', CS)}") as ws:
        s = Session(ws)
        ready = s.start({**EXTERNAL, "subject_id": subject, **flags})
        say_question(s)
        ws.send_json({"type": "SPEAK", "utterance_id": "a", "text": "好的。", "final": True})
        s.until("RESPONSE_END", request_id="a")
        ws.send_json({"type": "END"})
        s.drain()
    return s, ready


def test_subject_is_used_for_consent_recordings_index_and_memory(make_client, sync_redis):
    client, kafka, _, _, m33, memory = recording_setup(make_client, {"cs-cust-1"})
    _, ready = delegated_call(client, "cs-cust-1", record=True, index_transcripts=True, store_memory=True)
    sid = ready["session_id"]
    assert ready["subject_id"] == "cs-cust-1" and ready["recording"] is True
    assert m33.calls[-1]["headers"]["x-user-id"] == "cs-cust-1"
    assert kafka.topic("audio-recording-finalize")[0]["user_id"] == "cs-cust-1"
    assert kafka.topic("personal-index-requests")[0]["payload"]["branch_id"] == "cs-cust-1"
    call = memory.calls[0]
    assert call["headers"]["x-user-id"] == "cs-cust-1" and call["body"]["user_message"] == "你好嗎"
    assert call["body"]["assistant_response"] == "好的。" and call["url"].endswith(f"/memory/sessions/{sid}/turns")
    record = hgetall(sync_redis, sid)
    assert (record["actor_sub"], record["subject_id"], record["user_id"]) == ("cs-service", "cs-cust-1", "cs-service")


def test_external_sessions_have_no_side_effects_unless_asked(make_client):
    client, kafka, blobs, _, m33, memory = recording_setup(make_client, {"cs-cust-1"})
    _, ready = delegated_call(client, "cs-cust-1")
    assert ready["recording"] is False and ready["side_effects"] == {"record": False, "index_transcripts": False,
                                                                       "store_memory": False}
    assert kafka.sent == [] and blobs.blobs == {} and memory.calls == [] and m33.calls == []


def test_subject_without_the_delegate_role_is_ignored(make_client):
    client, kafka, *_ = recording_setup(make_client, set())
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'mallory')}") as ws:
        s = Session(ws)
        ready = s.start({"codec": "pcm", "sample_rate": 16000, "subject_id": "someone-else"})
        warning = s.until("WARNING")
        say_question(s)
        ws.send_json({"type": "END"})
        s.drain()
    assert warning["code"] == "subject_ignored" and ready["subject_id"] is None
    assert kafka.topic("personal-index-requests")[0]["payload"]["branch_id"] == "mallory"   # their own data


def test_service_account_without_subject_never_indexes_into_its_own_branch(make_client):
    client, kafka, *_ = recording_setup(make_client, set())
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'cs-service', CS)}") as ws:
        s = Session(ws)
        ready = s.start({"codec": "pcm", "sample_rate": 16000})                # module15 brain, no subject
        say_question(s)
        ws.send_json({"type": "END"})
        s.drain()
    assert ready["side_effects"]["index_transcripts"] is False and kafka.topic("personal-index-requests") == []


def test_two_subjects_of_one_service_account_are_kept_apart(make_client, sync_redis):
    from app.recording import chunks_key

    client, kafka, _, _, m33, _ = recording_setup(make_client, {"cs-cust-1", "cs-cust-2"})
    for subject in ("cs-cust-1", "cs-cust-2"):
        delegated_call(client, subject, record=True)
    assert sorted(c["headers"]["x-user-id"] for c in m33.calls) == ["cs-cust-1", "cs-cust-2"]   # own consent each
    sessions = {m["user_id"]: m["session_id"] for m in kafka.topic("audio-recording-finalize")}
    assert set(sessions) == {"cs-cust-1", "cs-cust-2"}                         # own recording each
    resp = client.delete("/internal/gdpr/user/cs-cust-1", headers={"X-Internal-Token": "test-internal-token"})
    assert resp.json()["erased_sessions"] == 1
    assert not sync_redis.exists(chunks_key(sessions["cs-cust-1"]))
    assert sync_redis.llen(chunks_key(sessions["cs-cust-2"])) > 0                # the other one untouched


def test_config_validation(make_client):
    client = external_client(make_client)
    for bad in ({"brain": "gpt"}, {"subject_id": "has space"}, {"record": "yes"}):
        with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
            ws.send_json({"codec": "pcm", "sample_rate": 16000, **bad})
            s = Session(ws)
            assert s.drain() == 4400 and s.events("ERROR")[0]["code"] == "invalid_config", bad


def test_ticket_keeps_only_the_roles_module_34_uses(make_client, sync_redis):
    client = make_client()
    token = issue_ticket(client, "cs", ["rt-delegate", "offline_access", "rt-external-brain"])
    stored = json.loads(sync_redis.get(f"rt:ws_ticket:{token}"))
    assert stored["user_id"] == "cs" and stored["roles"] == ["rt-delegate", "rt-external-brain"]
