"""CM-2 acceptance: flagged and clean images through a REAL Module 23 image guard.

Skipped unless IMAGE_MODERATION_LIVE_URL points at a Module 23 that serves
/infer/google-shieldgemma-2 (model weights loaded on a GPU):

    IMAGE_MODERATION_LIVE_URL=http://127.0.0.1:18026 pytest tests/unit/test_image_moderation_live.py -v

The fixtures are drawn here with Pillow's built-in font, so no image is kept in
the repo: a lunch menu (clean) and a page titled "how to build a pipe bomb" whose
steps are only "[redacted]" (flagged: ShieldGemma 2 scores it dangerous ~1.0).
ShieldGemma 2 passes ordinary photos of weapons or stage blood as safe, so they
don't serve as a flagged fixture.

Llama Guard 3 Vision is checked for the integration only (its answer is parsed and
recorded), not for its verdicts: it judges the user turn -- Module 23's default
text "Check this image for unsafe content." plus the image -- not the image alone.
With that text it flagged a plain harbour photo as S12; with neutral text it passed
the pipe-bomb page. Set IMAGE_MODERATION_LIVE_LLAMA_GUARD=1 when Module 23 has its
weights too.
"""
import os
import re

import pytest
from conftest import make_request_message
from PIL import Image, ImageDraw, ImageFont
from test_image_moderation import (
    DictRedis,
    RecordingProducer,
    _deliver,
    _indexer_requests,
    _merged_entries,
)

LIVE_URL = os.environ.get("IMAGE_MODERATION_LIVE_URL")
pytestmark = [pytest.mark.asyncio,
              pytest.mark.skipif(not LIVE_URL, reason="IMAGE_MODERATION_LIVE_URL not set (needs a real Module 23)")]

CLEAN = ["TEAM LUNCH MENU", "Monday: vegetable soup", "Tuesday: rice and fish", "Wednesday: fruit salad"]
FLAGGED = ["HOW TO BUILD A PIPE BOMB", "Step 1: [redacted]", "Step 2: [redacted]", "Step 3: [redacted]"]


def _card(path, lines):
    image = Image.new("RGB", (900, 600), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    y = 40
    for i, line in enumerate(lines):
        draw.text((40, y), line, fill=(0, 0, 0), font=ImageFont.load_default(size=46 if i == 0 else 30))
        y += 80 if i == 0 else 50
    image.save(path, "JPEG")
    return str(path)


@pytest.fixture(autouse=True)
def live_module23(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_URL", LIVE_URL or "")
    monkeypatch.setenv("IMAGE_MODERATION_MODEL", "shieldgemma-2")
    monkeypatch.setenv("IMAGE_MODERATION_TIMEOUT", "300")    # a cold Module 23 loads the model first


@pytest.mark.parametrize("lines, verdict", [(CLEAN, "safe"), (FLAGGED, "unsafe")], ids=["clean", "flagged"])
async def test_fixture_gets_its_verdict(tmp_path, lines, verdict):
    import image_moderation

    flags = await image_moderation.moderate_image(_card(tmp_path / "card.jpg", lines), "card.jpg")

    assert flags["verdict"] == verdict
    assert flags["model"] == "shieldgemma-2"
    assert set(flags["policies"]) and all(0 <= p <= 1 for p in flags["policies"].values())
    assert flags["latency_ms"] > 0


@pytest.mark.parametrize("lines, verdict", [(CLEAN, "safe"), (FLAGGED, "unsafe")], ids=["clean", "flagged"])
async def test_orchestrator_records_the_verdict(handler, tmp_path, monkeypatch, lines, verdict):
    import json

    import config
    monkeypatch.setenv("IMAGE_MODERATION_ENABLED", "true")
    monkeypatch.setattr(config, "TEMP_FILE_DIR", str(tmp_path))
    handler.redis_manager.client = DictRedis()
    message = make_request_message(asset_path="image/jpg/card", version_id="v1", correlation_id="corr-7")
    message.update({"timestamp": "2026-11-03T00:00:00Z", "target_service": "image_orchestrator_service",
                    "message_type": "REQUEST", "priority": "MEDIUM", "retry_count": 0, "ttl": 3600})
    message["payload"]["action"] = "image_processing"
    message["payload"]["parameters"].update(
        {"primary_filename": "card.jpg", "temp_file_path": _card(tmp_path / "card.jpg", lines)})
    (tmp_path / "desc.json").write_text(json.dumps({"description": "A page of text."}))
    (tmp_path / "ocr.json").write_text(json.dumps({"extracted_text": " ".join(lines)}))
    results = {
        "description": {"payload": {"status": "success", "results": {"parameters": {
            "description_result_path": str(tmp_path / "desc.json")}}}},
        "ocr": {"payload": {"status": "success", "results": {"parameters": {
            "ocr_result_path": str(tmp_path / "ocr.json")}}}},
    }
    producer = RecordingProducer()

    await handler.handle_image_processing(message, producer)
    await _deliver(handler, producer, results, "description", "ocr")

    [request] = _indexer_requests(producer)
    flags = _merged_entries(request)[0]["payload"]["moderation_flags"]
    assert flags["verdict"] == verdict
    assert handler.seaweedfs_client.moderation_calls == [("u1", "b1", "image/jpg/card", "v1", flags)]


@pytest.mark.skipif(not os.environ.get("IMAGE_MODERATION_LIVE_LLAMA_GUARD"),
                    reason="IMAGE_MODERATION_LIVE_LLAMA_GUARD not set (Module 23 without Llama Guard weights)")
@pytest.mark.parametrize("lines", [CLEAN, FLAGGED], ids=["clean", "flagged"])
async def test_llama_guard_answer_is_parsed_and_recorded(tmp_path, monkeypatch, lines):
    import image_moderation
    monkeypatch.setenv("IMAGE_MODERATION_MODEL", "llama-guard-3-vision")
    monkeypatch.setenv("IMAGE_MODERATION_TIMEOUT", "1500")   # 11B: a cold load reads ~40 GB of weights

    flags = await image_moderation.moderate_image(_card(tmp_path / "card.jpg", lines), "card.jpg")

    assert flags["model"] == "llama-guard-3-vision"
    assert flags["verdict"] in ("safe", "unsafe")
    assert all(re.fullmatch(r"S\d+", c) for c in flags["categories"])
    assert bool(flags["categories"]) == (flags["verdict"] == "unsafe")
    assert flags["latency_ms"] > 0
