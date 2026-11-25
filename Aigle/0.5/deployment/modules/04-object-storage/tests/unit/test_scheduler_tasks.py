"""endpoints.py's scheduler task wrappers (run_auto_archive/run_auto_destroy).

They must swallow any exception AssetManager.auto_archive()/auto_destroy()
raises and only log it, never let it propagate -- an unhandled exception
here would escape into apscheduler's job runner, and depending on its
error-handling config could silently deregister the job entirely: the
next day's auto-archive/destroy just never runs again, with no
user-facing symptom until archived/destroyed assets quietly pile up.

Not covered here (and not worth unit-testing): whether apscheduler itself
actually fires these jobs at the configured cron time -- that's the
scheduler library's own tested responsibility, not ours.

Run from the 04-object-storage/ directory:
    python -m pytest tests/unit/test_scheduler_tasks.py -v
"""
import pytest
from asset_management import endpoints


@pytest.mark.asyncio
async def test_run_auto_archive_survives_manager_failure(monkeypatch):
    async def _raise(*args, **kwargs):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(endpoints.manager, "auto_archive", _raise)
    await endpoints.run_auto_archive()  # must not raise


@pytest.mark.asyncio
async def test_run_auto_destroy_survives_manager_failure(monkeypatch):
    async def _raise(*args, **kwargs):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(endpoints.manager, "auto_destroy", _raise)
    await endpoints.run_auto_destroy()  # must not raise


@pytest.mark.asyncio
async def test_run_auto_archive_calls_manager_auto_archive(monkeypatch):
    calls = []

    async def _fake(*args, **kwargs):
        calls.append(1)
        return []

    monkeypatch.setattr(endpoints.manager, "auto_archive", _fake)
    await endpoints.run_auto_archive()
    assert calls == [1]


@pytest.mark.asyncio
async def test_run_auto_destroy_calls_manager_auto_destroy(monkeypatch):
    calls = []

    async def _fake(*args, **kwargs):
        calls.append(1)
        return []

    monkeypatch.setattr(endpoints.manager, "auto_destroy", _fake)
    await endpoints.run_auto_destroy()
    assert calls == [1]
