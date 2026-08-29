from __future__ import annotations

from pathlib import Path

# Owner-only: NFS-backed user directories must not be readable/traversable by
# other users sharing the same mount (see Dockerfile's non-root memsvc user).
_USER_DIR_MODE = 0o700


def user_dir(root: Path, user_id: str) -> Path:
    """Return `{root}/user_{user_id}`, creating it (mode 700) if missing.

    Also re-asserts the mode on an already-existing directory so a directory
    created before this check existed — or with a permissive umask — gets
    locked down on next access.
    """
    path = root / f"user_{user_id}"
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(_USER_DIR_MODE)
    return path
