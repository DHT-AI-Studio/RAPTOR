"""GB-17 / VIE01-245 operational startup checks (app/main.py lifespan).

* An unreachable guard model logs ERROR "Guard model {model} unreachable at
  {url} — guardrail will fail open", and startup carries on.
* GR_DEFAULT_ENABLED=true with no active policy logs WARNING "Guardrail enabled
  but no active policy configured — run seed_default_policy.py".
* TIMEZONE is wired to os.environ["TZ"], and log timestamps follow it.

The lifespan is run for real with everything outside the process (Postgres,
Redis, Ollama) replaced by fakes, so a test sees exactly what startup logs.
"""
from __future__ import annotations

import logging
import os
import time

import httpx
import pytest
from app import main
from app.core.config import Settings
from fastapi import FastAPI

OLLAMA = "http://fake-ollama:11434"


@pytest.fixture(autouse=True)
def _restore_tz(monkeypatch):
    """_apply_timezone() sets os.environ["TZ"]; give it back afterwards."""
    if "TZ" in os.environ:
        monkeypatch.setenv("TZ", os.environ["TZ"])
    else:
        monkeypatch.delenv("TZ", raising=False)
    yield
    monkeypatch.undo()
    time.tzset()


def _ollama(monkeypatch, models=None, error: Exception | None = None):
    """Fake Ollama /api/tags: lists `models`, or raises `error`."""
    real = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        if error is not None:
            raise error
        return httpx.Response(200, json={"models": [{"name": m} for m in (models or [])]})

    monkeypatch.setattr(main.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def _settings(**overrides) -> Settings:
    s = Settings(ollama_url=OLLAMA, guard_model="llama-guard3:8b", guard_models="",
                 proxy_model="gemma4:12b", gr_default_enabled=True, timezone="Asia/Taipei")
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


@pytest.fixture
def startup(monkeypatch):
    """Run the real lifespan with fake Postgres/Redis; returns a runner taking
    the settings and the active-policy row (None = no active policy)."""

    class _Redis:
        async def set(self, *a, **kw):
            return True

    async def _noop(*a, **kw):
        return None

    async def _redis():
        return _Redis()

    for name in ("init_db", "close_db"):
        monkeypatch.setattr(main.db_conn, name, _noop)
    monkeypatch.setattr(main.db_conn, "get_pool", _noop)
    for name in ("init_redis", "close_redis"):
        monkeypatch.setattr(main.redis_conn, name, _noop)
    monkeypatch.setattr(main.redis_conn, "get_redis", _redis)
    monkeypatch.setattr(main.state, "init_default_state", _noop)
    for name in ("close_checker_client", "close_forward_client", "close_guard_client"):
        monkeypatch.setattr(main, name, _noop)
    monkeypatch.setattr(main, "shutdown_model_manager", lambda: None)
    # _configure_logging() replaces the root handlers, which would drop caplog's.
    monkeypatch.setattr(main, "_configure_logging", lambda level: None)

    async def run(settings: Settings, active_row=None, fallback_policy=None) -> bool:
        """fallback_policy: what activating the default Standard Policy gives;
        None = that fails (database error)."""
        async def _active(pool):
            return active_row

        async def _fallback():
            if fallback_policy is None:
                raise RuntimeError("database unavailable")
            return fallback_policy

        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main.policy_repo, "get_active_policy", _active)
        monkeypatch.setattr(main.policy_store, "ensure_default_policy_active", _fallback)
        ready = False
        async with main.lifespan(FastAPI()):
            ready = True
        return ready

    return run


def _messages(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelno == level]


def _policy():
    from tests.unit.conftest import make_policy
    return make_policy("[]")


# ── guard model reachability ──────────────────────────────────────────────

async def test_unreachable_ollama_logs_error_for_each_guard_model_and_starts(monkeypatch, startup, caplog):
    _ollama(monkeypatch, error=httpx.ConnectError("connection refused"))
    settings = _settings(guard_model_2="granite3-guardian:8b")

    with caplog.at_level(logging.INFO):
        ready = await startup(settings, fallback_policy=_policy())

    assert ready, "startup must log and continue, not fail"
    errors = _messages(caplog, logging.ERROR)
    for model in ("llama-guard3:8b", "granite3-guardian:8b"):
        expected = f"Guard model {model} unreachable at {OLLAMA} — guardrail will fail open"
        assert any(e.startswith(expected) for e in errors), errors
    assert any("connection refused" in e for e in errors)


async def test_guard_model_not_on_ollama_logs_error(monkeypatch, startup, caplog):
    _ollama(monkeypatch, models=["gemma4:12b"])

    with caplog.at_level(logging.INFO):
        assert await startup(_settings(), fallback_policy=_policy())

    errors = _messages(caplog, logging.ERROR)
    assert errors == [(f"Guard model llama-guard3:8b unreachable at {OLLAMA} — guardrail will fail open"
                       " (model not on Ollama)")]


async def test_reachable_guard_model_logs_no_error(monkeypatch, startup, caplog):
    _ollama(monkeypatch, models=["llama-guard3:8b", "gemma4:12b"])

    with caplog.at_level(logging.INFO):
        assert await startup(_settings(), fallback_policy=_policy())

    assert _messages(caplog, logging.ERROR) == []
    assert any("guard model 'llama-guard3:8b' verified" in m for m in _messages(caplog, logging.INFO))


async def test_guard_models_list_overrides_guard_model(monkeypatch, startup, caplog):
    """GUARD_MODELS wins over GUARD_MODEL/2/3 (Settings.active_models)."""
    _ollama(monkeypatch, models=["a:1"])

    with caplog.at_level(logging.INFO):
        assert await startup(_settings(guard_models="a:1,b:2"), fallback_policy=_policy())

    errors = _messages(caplog, logging.ERROR)
    assert len(errors) == 1 and errors[0].startswith("Guard model b:2 unreachable")


async def test_check_guard_models_returns_reachability(monkeypatch):
    log = logging.getLogger("test")
    _ollama(monkeypatch, models=["llama-guard3:8b"])
    assert await main._check_guard_models(_settings(), log) is True
    _ollama(monkeypatch, models=[])
    assert await main._check_guard_models(_settings(), log) is False
    _ollama(monkeypatch, error=httpx.ReadTimeout(""))
    assert await main._check_guard_models(_settings(), log) is False


async def test_empty_timeout_message_still_gives_a_reason(monkeypatch, caplog):
    _ollama(monkeypatch, error=httpx.ReadTimeout(""))
    with caplog.at_level(logging.ERROR):
        await main._check_guard_models(_settings(), logging.getLogger("test"))
    assert _messages(caplog, logging.ERROR)[0].endswith("(ReadTimeout)")


# ── active policy ─────────────────────────────────────────────────────────

NO_POLICY = "Guardrail enabled but no active policy configured — run seed_default_policy.py"


async def test_enabled_without_active_policy_logs_warning(monkeypatch, startup, caplog):
    _ollama(monkeypatch, models=["llama-guard3:8b"])

    with caplog.at_level(logging.INFO):
        ready = await startup(_settings(gr_default_enabled=True), active_row=None, fallback_policy=None)

    assert ready
    assert NO_POLICY in _messages(caplog, logging.WARNING)


async def test_disabled_without_active_policy_logs_no_warning(monkeypatch, startup, caplog):
    _ollama(monkeypatch, models=["llama-guard3:8b"])

    with caplog.at_level(logging.INFO):
        assert await startup(_settings(gr_default_enabled=False), active_row=None, fallback_policy=None)

    assert NO_POLICY not in _messages(caplog, logging.WARNING)


async def test_default_policy_fallback_counts_as_configured(monkeypatch, startup, caplog):
    """No active row, but the built-in default Standard Policy got activated."""
    _ollama(monkeypatch, models=["llama-guard3:8b"])

    with caplog.at_level(logging.INFO):
        assert await startup(_settings(), active_row=None, fallback_policy=_policy())

    assert NO_POLICY not in _messages(caplog, logging.WARNING)


async def test_no_policy_warning_is_logged_once(monkeypatch, startup, caplog):
    _ollama(monkeypatch, models=["llama-guard3:8b"])

    with caplog.at_level(logging.INFO):
        assert await startup(_settings(), active_row=None, fallback_policy=None)

    assert _messages(caplog, logging.WARNING).count(NO_POLICY) == 1


async def test_existing_active_policy_logs_no_warning(monkeypatch, startup, caplog):
    import datetime as dt
    from uuid import uuid4
    row = {"id": uuid4(), "name": "p", "version": "1.0", "raw_content": "[]",
           "content_llama_guard": None, "content_granite_guardian": None,
           "content_gpt_oss_safeguard": None, "is_active": True,
           "created_at": dt.datetime.now(dt.timezone.utc)}
    _ollama(monkeypatch, models=["llama-guard3:8b"])

    with caplog.at_level(logging.INFO):
        assert await startup(_settings(), active_row=row)

    assert NO_POLICY not in _messages(caplog, logging.WARNING)


# ── TIMEZONE ──────────────────────────────────────────────────────────────

def test_timezone_setting_reads_TIMEZONE_then_TZ(monkeypatch):
    monkeypatch.delenv("TIMEZONE", raising=False)
    monkeypatch.setenv("TZ", "Europe/Paris")
    assert Settings().timezone == "Europe/Paris"
    monkeypatch.setenv("TIMEZONE", "America/New_York")
    assert Settings().timezone == "America/New_York"
    monkeypatch.delenv("TIMEZONE")
    monkeypatch.delenv("TZ")
    assert Settings().timezone == "Asia/Taipei"


def test_apply_timezone_sets_os_environ_tz():
    assert main._apply_timezone("America/New_York") is None
    assert os.environ["TZ"] == "America/New_York"
    assert time.strftime("%Z", time.localtime(0)) in ("EST", "EDT")


def test_invalid_timezone_falls_back_to_default():
    error = main._apply_timezone("Mars/Olympus_Mons")
    assert error is not None and "Mars/Olympus_Mons" in error
    assert os.environ["TZ"] == "Asia/Taipei"


async def test_lifespan_applies_timezone_before_logging(monkeypatch, startup):
    _ollama(monkeypatch, models=["llama-guard3:8b"])
    assert await startup(_settings(timezone="Europe/London"), fallback_policy=_policy())
    assert os.environ["TZ"] == "Europe/London"


def test_log_timestamps_follow_the_configured_zone():
    saved = logging.root.handlers[:], logging.root.level
    try:
        main._configure_logging("INFO")
        fmt = logging.root.handlers[0].formatter
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "m", None, None)
        record.created = 0.0

        main._apply_timezone("UTC")
        assert fmt.formatTime(record, "%Y-%m-%d %H:%M:%S") == "1970-01-01 00:00:00"
        main._apply_timezone("Asia/Taipei")
        assert fmt.formatTime(record, "%Y-%m-%d %H:%M:%S") == "1970-01-01 08:00:00"
    finally:
        logging.root.handlers, level = saved
        logging.root.setLevel(level)
