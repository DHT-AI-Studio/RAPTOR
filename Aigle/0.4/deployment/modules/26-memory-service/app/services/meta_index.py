"""Sidecar JSON index for memvid frame metadata.

memvid-sdk timeline/search hits expose only frame_id + text snippets.
We keep a parallel <store>.meta.json that maps frame_id → {frame_type,
session_id, timestamp, text} so callers can reconstruct full records.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

# Per-path lock so concurrent threads don't corrupt the same index file.
_locks: dict[str, threading.Lock] = {}
_lock_map_lock = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    with _lock_map_lock:
        if path not in _locks:
            _locks[path] = threading.Lock()
        return _locks[path]


def _index_path(mv2_path: str) -> Path:
    return Path(mv2_path).with_suffix(".meta.json")


def _atomic_write(p: Path, data: dict) -> None:
    """Write JSON atomically via a temp file in the same directory, then rename."""
    dir_ = p.parent
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data))
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load(mv2_path: str) -> dict[str, Any]:
    p = _index_path(mv2_path)
    if not p.exists():
        return {}
    with _lock_for(str(p)):
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}


def save(mv2_path: str, index: dict[str, Any]) -> None:
    p = _index_path(mv2_path)
    with _lock_for(str(p)):
        _atomic_write(p, index)


def put(mv2_path: str, frame_id: str | int, meta: dict[str, Any]) -> None:
    """Add or update one entry in the index."""
    p = _index_path(mv2_path)
    with _lock_for(str(p)):
        index: dict = {}
        if p.exists():
            try:
                index = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                index = {}
        index[str(frame_id)] = meta
        _atomic_write(p, index)


def get(mv2_path: str, frame_id: str | int) -> dict[str, Any]:
    index = load(mv2_path)
    return index.get(str(frame_id), {})


def all_entries(mv2_path: str) -> list[dict[str, Any]]:
    """Return all metadata entries as a list."""
    index = load(mv2_path)
    return list(index.values())


def remove_entry(mv2_path: str, frame_id: str | int) -> None:
    """Remove a single frame entry from the sidecar (call after mv.remove())."""
    p = _index_path(mv2_path)
    with _lock_for(str(p)):
        if not p.exists():
            return
        try:
            index = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return
        index.pop(str(frame_id), None)
        _atomic_write(p, index)


def delete(mv2_path: str) -> None:
    """Remove sidecar file (call when deleting the .mv2)."""
    p = _index_path(mv2_path)
    if p.exists():
        p.unlink()
