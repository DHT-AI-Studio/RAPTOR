"""Startup reaper for orphaned GPU training workers.

When the API worker process is replaced (uvicorn --reload) or the container is
restarted while a training job's subprocesses still exist, the training
subprocess and its DataLoader/DDP multiprocessing children can be orphaned and
keep holding GPU memory long after the job is gone. This sweep runs once at
startup and — ONLY when no job is currently 'running' — kills leftover *tagged*
training processes, reclaiming the VRAM the next job needs.

Identifying training processes safely
-------------------------------------
Pattern-matching on cmdline is NOT safe here: the uvicorn ``--reload`` worker is
itself launched as ``multiprocessing.spawn ... --multiprocessing-fork`` and holds
a CUDA context — indistinguishable by cmdline/GPU from a torch DataLoader child.
Killing by that heuristic can take down the live API worker.

Instead, the training launcher stamps every training process with the
environment variable ``RAPTOR_TRAINING_CHILD=1`` (``training_worker`` sets it; the
subprocess's torch spawn children inherit it). The reaper keys on that tag read
from ``/proc/<pid>/environ`` — the API worker never carries it, so it can never
be mistaken for an orphan.

PID note: this scans ``/proc`` inside the container, so it uses container-
namespace PIDs (unlike NVML, which reports host PIDs ``os.kill`` cannot address
from inside the container).
"""
from __future__ import annotations

import logging
import os
import signal
import time
from typing import List

logger = logging.getLogger(__name__)

# Environment tag stamped on every training subprocess (see training_worker).
TRAINING_TAG = "RAPTOR_TRAINING_CHILD"


def _environ(pid: str) -> dict:
    """Parse /proc/<pid>/environ into a dict (empty on any error)."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as fh:
            raw = fh.read()
    except OSError:
        return {}
    env = {}
    for entry in raw.split(b"\x00"):
        if b"=" in entry:
            k, _, v = entry.partition(b"=")
            env[k.decode(errors="replace")] = v.decode(errors="replace")
    return env


def _iter_pids():
    for name in os.listdir("/proc"):
        if name.isdigit():
            yield name


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _signal(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except OSError as exc:  # e.g. EPERM — log and move on
        logger.warning("orphan-reaper: could not signal pid %d: %s", pid, exc)


def reap_orphaned_gpu_workers(has_running_jobs: bool, grace_seconds: float = 3.0) -> int:
    """Kill leftover tagged training processes. Returns the count killed.

    No-op (returns 0) when a job is running — an active training run's children
    must never be touched.
    """
    if has_running_jobs:
        logger.info("orphan-reaper: a job is running — skipping GPU cleanup")
        return 0

    me = os.getpid()
    victims: List[int] = []
    for pid in _iter_pids():
        if int(pid) == me:
            continue
        # ONLY processes explicitly stamped as training children are eligible —
        # this can never match the uvicorn API worker.
        if _environ(pid).get(TRAINING_TAG) != "1":
            continue
        victims.append(int(pid))

    if not victims:
        logger.info("orphan-reaper: no orphaned training processes found")
        return 0

    logger.warning("orphan-reaper: reclaiming %d orphaned training process(es): %s",
                   len(victims), victims)
    for pid in victims:
        _signal(pid, signal.SIGTERM)
    time.sleep(grace_seconds)
    for pid in victims:
        if _alive(pid):
            logger.warning("orphan-reaper: pid %d survived SIGTERM — sending SIGKILL", pid)
            _signal(pid, signal.SIGKILL)
    return len(victims)
