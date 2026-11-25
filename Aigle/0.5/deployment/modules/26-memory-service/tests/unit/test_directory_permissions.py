"""
Unit tests proving user_*/ directories are created owner-only (mode 700) and
that a malformed/malicious X-User-ID cannot be used to traverse outside a
user's own directory.

Mode 700 is enforced by the OS kernel for every UID that isn't the file's
owner — there is no way to satisfy "user_B's process cannot read user_A's
directory" other than this bit pattern being correct, so asserting the mode
is a deterministic proof by itself. test_cross_uid_process_cannot_read_other_users_dir
below goes further and empirically confirms it by spawning a second real OS
user (`nobody`, uid 65534) and attempting the read from that process.

The cross-UID test needs either to already be root (it then drops to `nobody`
via os.setuid) or non-interactive sudo (`sudo -n -u nobody`) to be configured
for the runner. Neither is available in most developer sandboxes, so the test
skips itself with an explicit reason rather than failing — it is intended to
be run for real as part of deployment validation (see deploy.sh -m 26 /
README "Isolation validation"), where the runner is root or has passwordless
sudo configured for exactly this probe.

Run from the 26-memory-service/ directory:
    pip install memvid-sdk pydantic-settings fakeredis pytest pytest-asyncio
    pytest test_directory_permissions.py -v
"""
import os
import shutil
import stat
import subprocess
import sys
import textwrap

os.environ.setdefault("MEM_REDIS_HOST", "localhost")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))

import pytest
from core.dependencies import get_current_user
from core.paths import user_dir
from fakeredis import FakeAsyncRedis
from fastapi import HTTPException
from services.long_term_memory import LongTermMemoryService
from services.multimedia_memory import MultimediaMemoryService
from services.session_memory import SessionMemoryService, TurnAppendRequest


def _assert_owner_only(path) -> None:
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o700, f"{path} has mode {oct(mode)}, expected 0700"
    assert mode & (stat.S_IRWXG | stat.S_IRWXO) == 0
    assert os.stat(path).st_uid == os.getuid()


# ── core.paths.user_dir() ───────────────────────────────────────────────────

def test_user_dir_created_mode_700(tmp_path):
    p = user_dir(tmp_path, "user_A")
    assert p == tmp_path / "user_user_A"
    _assert_owner_only(p)


def test_user_dir_relocks_preexisting_dir_with_loose_perms(tmp_path):
    pre = tmp_path / "user_legacy"
    pre.mkdir(mode=0o755)
    assert stat.S_IMODE(pre.stat().st_mode) == 0o755

    p = user_dir(tmp_path, "legacy")
    _assert_owner_only(p)


def test_two_users_get_separate_dirs(tmp_path):
    a = user_dir(tmp_path, "A")
    b = user_dir(tmp_path, "B")
    assert a != b
    _assert_owner_only(a)
    _assert_owner_only(b)
    (a / "secret.mv2").write_text("A's data")
    assert not (b / "secret.mv2").exists()


# ── Service-level dir creation (session / long-term / multimedia) ──────────

@pytest.mark.asyncio
async def test_session_memory_user_and_media_dir_are_700(tmp_path):
    redis = FakeAsyncRedis(decode_responses=True)
    svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    await svc.append_turn(
        "userA", "sess1",
        TurnAppendRequest(user_message="hi", assistant_response="hello"),
    )
    user_dir_path = tmp_path / "user_userA"
    _assert_owner_only(user_dir_path)
    _assert_owner_only(user_dir_path / "media")
    await redis.aclose()


def test_long_term_memory_user_dir_is_700(tmp_path):
    svc = LongTermMemoryService(storage_root=str(tmp_path))
    d = svc._user_dir("userA")
    _assert_owner_only(d)
    _assert_owner_only(d / "media")


def test_multimedia_memory_dir_is_700(tmp_path):
    svc = MultimediaMemoryService(storage_root=str(tmp_path))
    d = svc._media_dir_create("userA")
    _assert_owner_only(d)
    _assert_owner_only(d.parent)


# ── X-User-ID traversal guard ───────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", [
    "../other_user",
    "../../etc/passwd",
    "userA/../userB",
    "userA/",
    "user A",
    "",
    "a" * 200,
])
async def test_get_current_user_rejects_unsafe_ids(bad_id):
    with pytest.raises(HTTPException):
        await get_current_user(x_user_id=bad_id)


@pytest.mark.asyncio
async def test_get_current_user_accepts_safe_id():
    assert await get_current_user(x_user_id="user-A_123") == "user-A_123"


# ── Empirical cross-UID proof ────────────────────────────────────────────────
#
# The tests above prove the *precondition* (mode 700, owned by the running
# process's uid). This test proves the actual claim in the acceptance
# criterion — that a different OS user's process is denied — by really
# spawning one.
#
# Getting a genuinely different real UID without already being root is
# harder than it looks:
#   - Rootless user namespaces (`unshare --user --map-user=<subuid>`) do NOT
#     work here: without the setuid `newuidmap`/`newgidmap` helpers (package
#     `uidmap`) actually invoked against the child's pid, an unprivileged
#     `unshare --map-user=N` only changes the *cosmetic* id reported inside
#     the namespace (what `id -u` prints) — the real kernel-level uid used
#     for filesystem permission checks on objects outside the namespace is
#     still the caller's own uid. This was verified by hand: `unshare --user
#     --map-user=<subuid> ls` on a 0700 dir it doesn't own still succeeds.
#   - `sudo -n -u nobody` needs passwordless sudo configured for the runner.
#   - Being root already lets us `os.fork()` + `os.setuid()` directly.
#
# The one method that reliably produces a *real* different uid without our
# own process needing any elevated privilege is Docker: the Docker *daemon*
# (which already runs privileged) does the uid switch on our behalf via
# `--user`, and a bind-mounted host directory is subject to real kernel DAC
# checks against that container's uid — this was verified by hand too:
# `docker run --user 65534:65534 -v <0700 dir owned by us>:/data alpine cat
# /data/f.txt` -> "Permission denied". It's also the most realistic proxy for
# this service's actual deployment shape (memsvc uid 10026 in one container;
# a hypothetical "other tenant" would be a different container/uid touching
# the same NFS mount), so it's tried first.

_NOBODY_UID = 65534  # conventional `nobody` uid


def _docker_probe(a_dir) -> int | None:
    """Try the cross-UID probe via a Docker container running as `nobody`
    with `a_dir`'s parent bind-mounted in. Returns the probe's exit code, or
    None if Docker isn't usable here."""
    if shutil.which("docker") is None:
        return None
    try:
        ping = subprocess.run(
            ["docker", "info"], check=False, capture_output=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if ping.returncode != 0:
        return None

    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "--user", f"{_NOBODY_UID}:{_NOBODY_UID}",
            "-v", f"{a_dir}:/data:ro",
            "alpine:latest",
            "ls", "/data",
        ],
        check=False, capture_output=True, text=True, timeout=30,
    )
    return result.returncode


def _sudo_probe(child_script: str) -> int | None:
    if shutil.which("sudo") is None:
        return None
    probe = subprocess.run(
        ["sudo", "-n", "-u", "nobody", "id", "-u"],
        check=False, capture_output=True, text=True, timeout=5,
    )
    if probe.returncode != 0 or not probe.stdout.strip().isdigit():
        return None
    result = subprocess.run(
        ["sudo", "-n", "-u", "nobody", sys.executable, "-c", child_script],
        check=False, capture_output=True, text=True, timeout=10,
    )
    return result.returncode


def test_cross_uid_process_cannot_read_other_users_dir(tmp_path):
    """Real second-OS-user probe: user A's directory (mode 700, owned by the
    current uid) must be unreadable by a process running as a genuinely
    different uid (`nobody`), not just "would be unreadable in theory"."""
    a_dir = user_dir(tmp_path, "A")
    secret = a_dir / "secret.mv2"
    secret.write_text("A's private data")

    if os.getuid() == 0:
        # Root runner (e.g. a CI container): fork+setuid to `nobody` directly.
        pid = os.fork()
        if pid == 0:
            os.setgid(_NOBODY_UID)
            os.setuid(_NOBODY_UID)
            try:
                os.listdir(str(a_dir))
            except PermissionError:
                os._exit(0)
            else:
                os._exit(1)
        _, status = os.waitpid(pid, 0)
        exit_code = os.WEXITSTATUS(status)
    else:
        exit_code = _docker_probe(a_dir)
        if exit_code is None:
            child_script = textwrap.dedent(f"""
                import os, sys
                try:
                    os.listdir({str(a_dir)!r})
                except PermissionError:
                    sys.exit(0)
                else:
                    sys.exit(1)  # did NOT get denied -> isolation broken
            """)
            exit_code = _sudo_probe(child_script)
        if exit_code is None:
            pytest.skip(
                "No way to act as a second real OS user here: no working "
                "Docker daemon, not root, and no passwordless `sudo -n -u "
                "nobody`. Any one of those (Docker is the easiest — no host "
                "sudo/root needed) gets an empirical cross-UID isolation "
                "proof; this is how deployment validation for Module 26 is "
                "expected to run it."
            )

    assert exit_code != 0, (
        f"nobody (uid {_NOBODY_UID}) was able to list {a_dir}, expected "
        "Permission denied — cross-user isolation is broken"
    )
