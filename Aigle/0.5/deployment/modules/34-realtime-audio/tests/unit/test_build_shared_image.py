"""Check the real build script and subprocess output without invoking Docker."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def builder(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[3] / "build.py"
    path = tmp_path / "build.py"
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text("RAPTOR_VERSION=unit-test\n", encoding="utf-8")
    for name in ("09-audio-processing", "34-realtime-audio"):
        (tmp_path / name).mkdir()
    spec = importlib.util.spec_from_file_location("raptor_test_builder", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "check_deps", lambda *args, **kwargs: True)
    return module


@pytest.mark.parametrize("module_id, has_own_build_step", [("34", True), ("09", False)])
def test_build_shared_base_and_only_required_child_image(builder, monkeypatch, module_id, has_own_build_step):
    # Doesn't assert an exact command count: module 08's own step list (currently chmod +x its
    # arm64 check script, run that script, then the actual docker build) is free to grow or
    # shrink without rotting this test -- it previously hardcoded a count that went stale the
    # moment 08 gained its arm64 steps (PR #423), silently breaking on main for days.
    commands = []

    def run_step(step, _logger):
        commands.append(step.cmd)
        return True

    monkeypatch.setattr(builder, "run_step", run_step)
    built = set()
    assert builder.run_module(
        builder.MODULE_BY_ID[module_id], builder.Logger(None), build_only=True, built_set=built
    )
    assert built == {"08"}
    docker_build = [i for i, c in enumerate(commands) if c[:4] == ["docker", "build", "-t", "raptor/media-worker:unit-test"]]
    assert len(docker_build) == 1, "expected exactly one shared-base-image build"
    compose_build = [i for i, c in enumerate(commands) if c[0:2] == ["docker", "compose"] and c[-1] == "build"]
    if has_own_build_step:
        assert len(compose_build) == 1, "expected exactly one of this module's own compose build"
        assert "34-realtime-audio/docker-compose.yml" in " ".join(commands[compose_build[0]])
        assert compose_build[0] > docker_build[0], "module's own build must run after the shared base image"
    else:
        assert not compose_build, "module 09 has no build of its own -- it only uses 08's shared image"
    assert all("up" not in command for command in commands)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_run_step_handles_invalid_utf8_and_preserves_exit_code(builder, tmp_path, exit_code):
    script = (
        "import sys\n"
        "sys.stdout.buffer.write('下載進度'.encode('utf-8') + b'\\xff\\xe2\\x96\\n')\n"
        "sys.stdout.buffer.flush()\n"
        "sys.stderr.buffer.write(b'stderr: final message\\n')\n"
        "sys.stderr.buffer.flush()\n"
        f"sys.exit({exit_code})\n"
    )
    log_path = tmp_path / "build.log"
    logger = builder.Logger(log_path)
    try:
        ok = builder.run_step(builder.Step("subprocess output", [sys.executable, "-c", script]), logger)
    finally:
        logger.close()
    output = log_path.read_text(encoding="utf-8")
    assert ok is (exit_code == 0)
    assert "下載進度" in output
    assert "stderr: final message" in output
    assert "Unexpected error" not in output
    if exit_code:
        assert f"Step failed (exit {exit_code})" in output
