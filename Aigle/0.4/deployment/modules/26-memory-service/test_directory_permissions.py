"""
Unit tests proving user_*/ directories are created owner-only (mode 700) and
that a malformed/malicious X-User-ID cannot be used to traverse outside a
user's own directory.

Mode 700 is enforced by the OS kernel for every UID that isn't the file's
owner — there is no way to satisfy "user_B's process cannot read user_A's
directory" other than this bit pattern being correct, so asserting the mode
(rather than literally spawning a second OS user) is the deterministic proof.

For an empirical, real-second-user confirmation, run manually (needs sudo):

    python -c "
import os, sys
sys.path.insert(0, 'app')
os.environ.setdefault('MEM_REDIS_HOST', 'localhost')
from core.paths import user_dir
from pathlib import Path
p = user_dir(Path('/tmp/mv_perm_demo'), 'A')
print(p)
"
    sudo -u nobody ls /tmp/mv_perm_demo/user_A/
    # Expected: "Permission denied"

Run from the 26-memory-service/ directory:
    pip install memvid-sdk pydantic-settings fakeredis pytest pytest-asyncio
    pytest test_directory_permissions.py -v
"""
import os
import stat
import sys

os.environ.setdefault("MEM_REDIS_HOST", "localhost")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis
from fastapi import HTTPException

from core.dependencies import get_current_user
from core.paths import user_dir
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
