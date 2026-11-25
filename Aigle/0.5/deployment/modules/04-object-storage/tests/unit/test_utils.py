"""Pure function tests for asset_management/utils.py -- no AssetManager,
no fakes, nothing to construct.

Run from the 04-object-storage/ directory:
    python -m pytest tests/unit/test_utils.py -v
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from asset_management.models import MediaType
from asset_management.utils import (
    calculate_md5_bytes,
    detect_file_type,
    guess_media_type,
    mime_from_filename,
    parse_absolute_datetime,
    parse_date_or_ttl,
    parse_ttl,
)
from fastapi import HTTPException

# ── calculate_md5_bytes ──────────────────────────────────────────────────

def test_calculate_md5_bytes_known_value():
    assert calculate_md5_bytes(b"hello") == "5d41402abc4b2a76b9719d911017c592"


def test_calculate_md5_bytes_deterministic():
    data = b"mtr-1 test content"
    assert calculate_md5_bytes(data) == calculate_md5_bytes(data)


# ── mime_from_filename ───────────────────────────────────────────────────

@pytest.mark.parametrize("filename,expected", [
    ("video.mp4", "video/mp4"),
    ("IMAGE.PNG", "image/png"),
    ("doc.pdf", "application/pdf"),
    ("no_extension", None),
    ("unknown.xyz", None),
])
def test_mime_from_filename(filename, expected):
    assert mime_from_filename(filename) == expected


# ── guess_media_type ─────────────────────────────────────────────────────

@pytest.mark.parametrize("mime_type,expected", [
    ("video/mp4", MediaType.VIDEO),
    ("image/png", MediaType.IMAGE),
    ("audio/mpeg", MediaType.AUDIO),
    ("text/plain", MediaType.DOCUMENT),
    ("application/pdf", MediaType.DOCUMENT),
    ("application/zip", MediaType.OTHER),
    (None, MediaType.OTHER),
])
def test_guess_media_type(mime_type, expected):
    assert guess_media_type(mime_type) == expected


# ── detect_file_type ─────────────────────────────────────────────────────

def test_detect_file_type_plain_text():
    result = detect_file_type(b"hello world", "test.txt")
    assert result["mime_type"] == "text/plain"
    assert result["base_path"] == "document/txt"


# ── parse_absolute_datetime ──────────────────────────────────────────────

def test_parse_absolute_datetime_date_only():
    dt = parse_absolute_datetime("2026-01-01", "Asia/Taipei")
    assert dt == datetime(2026, 1, 1, tzinfo=ZoneInfo("Asia/Taipei"))


def test_parse_absolute_datetime_slash_format():
    dt = parse_absolute_datetime("2026/1/1", "Asia/Taipei")
    assert dt == datetime(2026, 1, 1, tzinfo=ZoneInfo("Asia/Taipei"))


def test_parse_absolute_datetime_iso_with_time():
    dt = parse_absolute_datetime("2026-01-01T10:30:00", "Asia/Taipei")
    assert dt.hour == 10 and dt.minute == 30


def test_parse_absolute_datetime_invalid_raises_400():
    with pytest.raises(HTTPException) as exc_info:
        parse_absolute_datetime("not-a-date", "Asia/Taipei")
    assert exc_info.value.status_code == 400


# ── parse_ttl ─────────────────────────────────────────────────────────────

BASE = datetime(2026, 1, 1, tzinfo=ZoneInfo("Asia/Taipei"))


@pytest.mark.parametrize("value,expected_days", [
    (5, 5),
    ("10", 10),
    ("2w", 14),
    ("1m", 30),
    ("1y", 365),
])
def test_parse_ttl(value, expected_days):
    result = parse_ttl(value, BASE)
    assert (result - BASE).days == expected_days


def test_parse_ttl_invalid_raises_400():
    with pytest.raises(HTTPException) as exc_info:
        parse_ttl("not-a-ttl", BASE)
    assert exc_info.value.status_code == 400


# ── parse_date_or_ttl ─────────────────────────────────────────────────────

def test_parse_date_or_ttl_none_returns_base_date():
    assert parse_date_or_ttl(None, BASE, "Asia/Taipei") == BASE


def test_parse_date_or_ttl_ttl_path():
    result = parse_date_or_ttl("30d", BASE, "Asia/Taipei")
    assert (result - BASE).days == 30


def test_parse_date_or_ttl_absolute_path():
    result = parse_date_or_ttl("2026-06-01", BASE, "Asia/Taipei")
    assert result == datetime(2026, 6, 1, tzinfo=ZoneInfo("Asia/Taipei"))


def test_parse_date_or_ttl_empty_string_raises_400():
    with pytest.raises(HTTPException) as exc_info:
        parse_date_or_ttl("", BASE, "Asia/Taipei")
    assert exc_info.value.status_code == 400
