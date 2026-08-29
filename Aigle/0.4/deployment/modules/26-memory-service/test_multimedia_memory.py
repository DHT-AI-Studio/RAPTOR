"""
Unit tests for MultimediaMemoryService (MV-4).
Run from the 26-memory-service/ directory:

"""
import os
import sys

os.environ.setdefault("MEM_REDIS_HOST", "localhost")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import pytest
import pytest_asyncio

from services.multimedia_memory import (
    AudioIndexRequest,
    ImageIndexRequest,
    MediaSearchRequest,
    MultimediaMemoryService,
    VideoIndexRequest,
)


@pytest_asyncio.fixture
async def svc(tmp_path):
    yield MultimediaMemoryService(storage_root=str(tmp_path))


# ── index_video ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_index_video_returns_response(svc):
    resp = await svc.index_video("u1", VideoIndexRequest(
        asset_path="videos/cooling.mp4",
        version_id="v1",
        start_sec=120.0,
        end_sec=145.0,
        transcription="冷卻系統透過循環冷卻液降低引擎溫度",
    ))
    assert resp.frame_id is not None
    assert resp.asset_path == "videos/cooling.mp4"
    assert resp.media_type == "video"


@pytest.mark.asyncio
async def test_index_video_same_asset_multiple_moments(svc):
    user_id = "u_multi"
    for i, (start, end, text) in enumerate([
        (0.0, 30.0, "引擎概述"),
        (120.0, 145.0, "冷卻系統說明"),
        (200.0, 220.0, "水泵運作原理"),
    ]):
        resp = await svc.index_video(user_id, VideoIndexRequest(
            asset_path="videos/engine.mp4",
            version_id="v1",
            start_sec=start,
            end_sec=end,
            transcription=text,
        ))
        assert resp.frame_id == str(i)


# ── index_audio ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_index_audio_with_transcription(svc):
    resp = await svc.index_audio("u2", AudioIndexRequest(
        asset_path="audio/meeting.wav",
        version_id="v1",
        start_sec=0.0,
        end_sec=60.0,
    ), transcription="會議記錄第一段內容")
    assert resp.media_type == "audio"
    assert resp.asset_path == "audio/meeting.wav"


# ── index_image ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_index_image_returns_response(svc):
    resp = await svc.index_image("u3", ImageIndexRequest(
        asset_path="images/diagram.png",
        version_id="v1",
        description="引擎冷卻系統示意圖",
        ocr_text="冷卻液 水泵 散熱器",
    ))
    assert resp.frame_id is not None
    assert resp.media_type == "image"


# ── search ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_3_video_2_image_returns_correct_assets(svc):
    user_id = "u_search"

    # 3 video moments
    await svc.index_video(user_id, VideoIndexRequest(
        asset_path="videos/cooling_system.mp4",
        version_id="v1", start_sec=120.0, end_sec=145.0,
        transcription="coolant circulates through the engine cooling system",
    ))
    await svc.index_video(user_id, VideoIndexRequest(
        asset_path="videos/cooling_system.mp4",
        version_id="v1", start_sec=200.0, end_sec=220.0,
        transcription="the water pump pushes coolant from the radiator to the engine",
    ))
    await svc.index_video(user_id, VideoIndexRequest(
        asset_path="videos/engine_overview.mp4",
        version_id="v1", start_sec=0.0, end_sec=30.0,
        transcription="engine overview covering all major components",
    ))

    # 2 images
    await svc.index_image(user_id, ImageIndexRequest(
        asset_path="images/engine_diagram.png",
        version_id="v1",
        description="engine cooling system diagram",
        ocr_text="coolant pump radiator",
    ))
    await svc.index_image(user_id, ImageIndexRequest(
        asset_path="images/ui_screenshot.png",
        version_id="v1",
        description="system dashboard screenshot",
        ocr_text="CPU temperature 85C",
    ))

    hits = await svc.search(user_id, MediaSearchRequest(query="cooling system", top_k=5))
    assert len(hits) >= 1
    asset_paths = {h.asset_path for h in hits}
    assert "videos/cooling_system.mp4" in asset_paths


@pytest.mark.asyncio
async def test_search_empty_returns_empty(svc):
    hits = await svc.search("u_nobody", MediaSearchRequest(query="anything"))
    assert hits == []


@pytest.mark.asyncio
async def test_search_media_type_filter_video_only(svc):
    user_id = "u_filter"
    await svc.index_video(user_id, VideoIndexRequest(
        asset_path="videos/test.mp4",
        version_id="v1", start_sec=0.0, end_sec=10.0,
        transcription="test video content",
    ))
    await svc.index_image(user_id, ImageIndexRequest(
        asset_path="images/test.png",
        version_id="v1",
        description="test image content",
    ))

    hits = await svc.search(user_id, MediaSearchRequest(query="test", media_type="video"))
    assert all(h.media_type == "video" for h in hits)


@pytest.mark.asyncio
async def test_search_media_type_filter_image_only(svc):
    user_id = "u_filter_img"
    await svc.index_video(user_id, VideoIndexRequest(
        asset_path="videos/sample.mp4",
        version_id="v1", start_sec=0.0, end_sec=10.0,
        transcription="sample video",
    ))
    await svc.index_image(user_id, ImageIndexRequest(
        asset_path="images/sample.png",
        version_id="v1",
        description="sample image",
    ))

    hits = await svc.search(user_id, MediaSearchRequest(query="sample", media_type="image"))
    assert all(h.media_type == "image" for h in hits)


@pytest.mark.asyncio
async def test_search_hit_has_required_metadata(svc):
    user_id = "u_meta"
    await svc.index_video(user_id, VideoIndexRequest(
        asset_path="videos/meta_test.mp4",
        version_id="abc123",
        start_sec=10.0, end_sec=25.0,
        transcription="metadata test content",
        session_id="sess_x",
    ))

    hits = await svc.search(user_id, MediaSearchRequest(query="metadata test"))
    assert len(hits) >= 1
    h = hits[0]
    assert h.asset_path == "videos/meta_test.mp4"
    assert h.version_id == "abc123"
    assert h.start_sec == 10.0
    assert h.end_sec == 25.0
    assert h.session_id == "sess_x"
    assert h.media_type == "video"
    assert 0.0 <= h.score <= 1.0


# ── list_all ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_all_empty_returns_empty(svc):
    assert await svc.list_all("u_none") == []


@pytest.mark.asyncio
async def test_list_all_returns_all_types_across_shards(svc):
    user_id = "u_list_all"
    await svc.index_video(user_id, VideoIndexRequest(
        asset_path="videos/a.mp4", version_id="v1",
        start_sec=0.0, end_sec=5.0, transcription="video text",
    ))
    await svc.index_audio(user_id, AudioIndexRequest(
        asset_path="audio/a.mp3", version_id="v1",
        start_sec=0.0, end_sec=5.0, transcription="audio text",
    ), transcription="audio text")
    await svc.index_image(user_id, ImageIndexRequest(
        asset_path="images/a.png", version_id="v1", description="image text",
    ))

    frames = await svc.list_all(user_id)
    assert len(frames) == 3
    assert {f["media_type"] for f in frames} == {"video", "audio", "image"}
    assert all("text" in f for f in frames)


@pytest.mark.asyncio
async def test_list_all_sorted_by_recency(svc):
    user_id = "u_list_all_sorted"
    await svc.index_video(user_id, VideoIndexRequest(
        asset_path="videos/old.mp4", version_id="v1",
        start_sec=0.0, end_sec=5.0, transcription="older",
    ))
    await svc.index_video(user_id, VideoIndexRequest(
        asset_path="videos/new.mp4", version_id="v1",
        start_sec=0.0, end_sec=5.0, transcription="newer",
    ))

    frames = await svc.list_all(user_id)
    assert len(frames) == 2
    assert frames[0]["timestamp"] >= frames[1]["timestamp"]
