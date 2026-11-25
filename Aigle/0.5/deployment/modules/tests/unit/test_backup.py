"""Check build.py's VIE01-369 write-freeze-coordinated --backup command,
without invoking Docker or a real module 13/gateway."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def builder(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[2] / "build.py"
    path = tmp_path / "build.py"
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text("RAPTOR_VERSION=unit-test\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("raptor_test_builder_backup", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_backup_module_01_clears_the_gateway_url_for_its_own_call(builder, monkeypatch):
    """01's own nfs-freeze-archive.sh has its own built-in maintenance-mode toggle (for its
    independent, cron-scheduled use) -- backup_module() must clear NFS_BACKUP_GATEWAY_URL for
    just this one exec so 01's script takes its own no-op branch instead of double-toggling
    the freeze backup_all() is already holding."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        class Result:
            returncode = 0
            stderr = stdout = ""

        return Result()

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    assert builder.backup_module(builder.MODULE_BY_ID["01"], builder.Logger(None)) is True
    assert calls == [[
        "docker", "exec", "-e", "NFS_BACKUP_GATEWAY_URL=",
        "raptor-nfs-backup-unit-test", "/scripts/nfs-freeze-archive.sh",
    ]]


def test_backup_module_triggers_the_right_sidecar(builder, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        class Result:
            returncode = 0
            stderr = stdout = ""

        return Result()

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    assert builder.backup_module(builder.MODULE_BY_ID["24"], builder.Logger(None)) is True
    assert calls == [["docker", "exec", "raptor-arcadedb-backup-unit-test", "backup"]]


def test_backup_module_reports_a_clear_error_for_an_unconfigured_module(builder):
    logged = []
    log = builder.Logger(None)
    log.error = lambda msg: logged.append(msg)
    assert builder.backup_module(builder.MODULE_BY_ID["09"], log) is False
    assert logged and "no backup mechanism" in logged[0]


def test_backup_module_surfaces_a_failed_trigger(builder, monkeypatch):
    def fake_run(cmd, **kwargs):
        class Result:
            returncode = 1
            stderr = "boom"
            stdout = ""

        return Result()

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    assert builder.backup_module(builder.MODULE_BY_ID["02"], builder.Logger(None)) is False


@pytest.mark.parametrize("module_ids, expect_freeze", [
    (["02"], False), (["06"], False), (["02", "06"], False),
    (["03"], True), (["24"], True), (["01"], True), (["02", "24"], True),
])
def test_backup_all_holds_the_write_freeze_only_for_triangle_members(builder, monkeypatch, module_ids, expect_freeze):
    maintenance_calls = []

    def fake_maintenance(path, _log, **kwargs):
        maintenance_calls.append(path)
        return True

    monkeypatch.setattr(builder, "_maintenance_request", fake_maintenance)
    monkeypatch.setattr(builder, "backup_module", lambda _m, _log: True)

    modules = [builder.MODULE_BY_ID[i] for i in module_ids]
    assert builder.backup_all(modules, builder.Logger(None)) is True
    if expect_freeze:
        assert maintenance_calls == ["/admin/maintenance/enable", "/admin/maintenance/disable"]
    else:
        assert maintenance_calls == []


def test_backup_all_still_restores_writes_after_a_failed_trigger(builder, monkeypatch):
    """The write-freeze teardown (disable maintenance) must run even when a backup trigger
    fails partway through -- it's in a finally block for exactly this reason."""
    maintenance_calls = []
    monkeypatch.setattr(builder, "_maintenance_request", lambda path, _log, **kw: maintenance_calls.append(path) or True)
    monkeypatch.setattr(builder, "backup_module", lambda _m, _log: False)   # every trigger "fails"

    ok = builder.backup_all([builder.MODULE_BY_ID["03"], builder.MODULE_BY_ID["24"]], builder.Logger(None))
    assert ok is False
    assert maintenance_calls == ["/admin/maintenance/enable", "/admin/maintenance/disable"]


def test_maintenance_request_is_a_noop_when_gateway_url_is_unset(builder):
    """No NFS_BACKUP_GATEWAY_URL in the env file this fixture writes -- must warn and return
    False without ever attempting a network call."""
    assert builder._maintenance_request("/admin/maintenance/enable", builder.Logger(None)) is False
