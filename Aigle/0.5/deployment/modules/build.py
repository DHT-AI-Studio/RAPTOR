#!/usr/bin/env python3
"""
Raptor — Platform Build Script
Manages startup of all platform modules in dependency order.

Usage:
    python build.py                   # build all modules interactively
    python build.py -a                # build all modules interactively
    python build.py -m 02             # build module 02 only
    python build.py -m 02,03          # build modules 02 and 03
    python build.py -l build.log      # build all, save log to file
    python build.py -m 04 -l out.log  # build module 04, log to file
    python build.py --list            # list all modules and exit
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent  # deployment/modules/
UNIFIED_ENV: Path | None = None  # set by --env-file CLI arg


# VIE01-369: populated by init_env_cmd() as it collects answers to unfilled
# placeholders, so a later placeholder suggestion in the SAME --init-env run
# (e.g. AGE_PUBLIC_KEYS's generator needing AGE_PRIVATE_KEY_PATH's just-chosen
# value) can see it via _read_env_var() before anything's written to disk.
_pending_env_overrides: dict[str, str] = {}


def _read_env_var(key: str, default: str) -> str:
    """Read a single variable from the active .env file (respects UNIFIED_ENV).

    Defined up here (not just before its first use) because MODULES below
    needs it at import time, before main()'s argument parsing has had a
    chance to set UNIFIED_ENV from --env-file -- so a tag built from this
    always reads SCRIPT_DIR/.env regardless of an explicit --env-file
    override. Accepted: --env-file is rarely used, and the common case
    (SCRIPT_DIR/.env) always resolves correctly either way.
    """
    if key in _pending_env_overrides:
        return _pending_env_overrides[key]
    candidates = [UNIFIED_ENV, SCRIPT_DIR / ".env"]
    for env_file in candidates:
        if env_file is None:
            continue
        p = Path(env_file)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                val = v.split("#")[0].strip()
                if val:
                    return val
    return default


_ENV_VAR_REF = re.compile(r"^\$\{([A-Za-z_]\w*)\}$")


def _read_env_var_resolved(key: str, default: str = "") -> str:
    """Like _read_env_var(), but also follows one level of a bare ${OTHER_VAR}
    reference (e.g. HUGGINGFACE_TOKEN=${HF_TOKEN}) -- the indirection this
    repo's .env uses so the same real secret backs both names. Needed because
    docker compose's own ${VAR} substitution only ever reaches a service's
    *runtime* `environment:`/`env_file:`; a build-time secret sourced via
    `secrets.<name>.environment:` is resolved by buildx/BuildKit reading its
    own process environment directly, bypassing that substitution entirely --
    confirmed to silently return empty on some Compose/BuildKit versions, not
    on others, so this can't be left to compose. Callers needing that real
    value as an actual process env var must inject it themselves (see the
    module-35 voice-profile step below).
    """
    val = _read_env_var(key, "")
    m = _ENV_VAR_REF.match(val)
    if m:
        val = _read_env_var(m.group(1), "")
    return val or default


_TRUE_VALUES = ("1", "true", "yes")
_FALSE_VALUES = ("0", "false", "no", "")


def _apply_internal_port_mode() -> bool:
    """CF-3 developer switch: EXPOSE_INTERNAL_PORTS=true restores the pre-CF-3
    port publishing for the internal services (23, 25, 26, 33).

    Their compose files bind `${INTERNAL_BIND_IP:-127.0.0.1}:${PORT_X}:N`, and
    compose interpolation has no conditionals, so the boolean is translated
    here: when true, INTERNAL_BIND_IP=0.0.0.0 is put into this process's
    environment. The shell environment takes precedence over `--env-file` in
    docker compose, and every compose call / install script build.py starts
    inherits os.environ (run_step copies it), so all of them see it.

    An EXPOSE_INTERNAL_PORTS already in the shell environment wins over the
    .env file. Unrecognised values warn and fall back to false (fail closed).
    Returns True when the original (all-interfaces) mode was applied.
    """
    raw = os.environ.get("EXPOSE_INTERNAL_PORTS")
    if raw is None:
        raw = _read_env_var("EXPOSE_INTERNAL_PORTS", "false")
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        os.environ["INTERNAL_BIND_IP"] = "0.0.0.0"  # nosec B104 -- explicit developer opt-in (CF-3 original mode)
        print(f"{C.YELLOW}EXPOSE_INTERNAL_PORTS=true: internal services (23/25/26/33) publish on ALL "
              f"interfaces (INTERNAL_BIND_IP=0.0.0.0). Development only -- not on an untrusted network.{C.RESET}")
        return True
    if value not in _FALSE_VALUES:
        print(f"{C.YELLOW}EXPOSE_INTERNAL_PORTS={raw!r} is not a boolean; treating it as false "
              f"(internal ports stay on INTERNAL_BIND_IP).{C.RESET}")
    return False


def _apply_keycloak_tls_mode() -> bool:
    """06-authentication's TLS entry (CF-3a): KEYCLOAK_TLS_ENABLED=true activates its "tls" Compose
    profile (raptor-tls-proxy, plus Keycloak's own hardened settings) -- see that module's README.md,
    "TLS for Keycloak and the gateway".

    A Compose profile is itself just a value in COMPOSE_PROFILES, so -- same translation-into-this-
    process's-own-environment mechanism as EXPOSE_INTERNAL_PORTS/INTERNAL_BIND_IP above, for the same
    reason (compose interpolation has no conditionals) -- true appends "tls" to whatever
    COMPOSE_PROFILES already holds (e.g. 02's own "cluster", so the two combine rather than one
    replacing the other) and puts the result back into this process's environment.

    A KEYCLOAK_TLS_ENABLED already in the shell environment wins over the .env file. Unrecognised
    values warn and fall back to false (fail closed). Returns True when the TLS entry was activated.
    """
    raw = os.environ.get("KEYCLOAK_TLS_ENABLED")
    if raw is None:
        raw = _read_env_var("KEYCLOAK_TLS_ENABLED", "false")
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        existing = os.environ.get("COMPOSE_PROFILES")
        if existing is None:
            existing = _read_env_var("COMPOSE_PROFILES", "")
        profiles = [p.strip() for p in existing.split(",") if p.strip()]
        if "tls" not in profiles:
            profiles.append("tls")
        os.environ["COMPOSE_PROFILES"] = ",".join(profiles)
        print(f"{C.YELLOW}KEYCLOAK_TLS_ENABLED=true: 06-authentication's TLS entry is active "
              f"(COMPOSE_PROFILES={os.environ['COMPOSE_PROFILES']!r}).{C.RESET}")
        return True
    if value not in _FALSE_VALUES:
        print(f"{C.YELLOW}KEYCLOAK_TLS_ENABLED={raw!r} is not a boolean; treating it as false "
              f"(TLS entry stays off).{C.RESET}")
    return False

# ──────────────────────────────────────────────────────────────────────────────
# TERMINAL COLOURS
# ──────────────────────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"

def strip_ansi(text: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)

# ──────────────────────────────────────────────────────────────────────────────
# LOGGER
# ──────────────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, log_path: Path | None = None):
        self._log_path = log_path
        self._fh = open(log_path, "w", encoding="utf-8") if log_path else None  # noqa: SIM115 -- closed in close(), lifetime spans the Logger instance
        self._lock = threading.Lock()

    def _write(self, line: str, *, colour: bool = True):
        with self._lock:
            print(line if colour else strip_ansi(line))
            if self._fh:
                self._fh.write(strip_ansi(line) + "\n")
                self._fh.flush()

    def info(self, msg: str):
        self._write(f"{C.WHITE}{msg}{C.RESET}")

    def step(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._write(f"{C.DIM}[{ts}]{C.RESET} {C.CYAN}{msg}{C.RESET}")

    def ok(self, msg: str):
        self._write(f"{C.GREEN}  ✔  {msg}{C.RESET}")

    def warning(self, msg: str):
        self._write(f"{C.YELLOW}  ⚠  {msg}{C.RESET}")

    def error(self, msg: str):
        self._write(f"{C.RED}  ✖  {msg}{C.RESET}")

    def header(self, msg: str):
        bar = "─" * 60
        self._write(f"\n{C.BOLD}{C.BLUE}{bar}{C.RESET}")
        self._write(f"{C.BOLD}{C.BLUE}  {msg}{C.RESET}")
        self._write(f"{C.BOLD}{C.BLUE}{bar}{C.RESET}")

    def raw(self, line: str):
        self._write(f"{C.DIM}{line}{C.RESET}", colour=True)

    def close(self):
        if self._fh:
            self._fh.close()

# ──────────────────────────────────────────────────────────────────────────────
# STEP
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Step:
    label: str
    cmd: list[str]
    cwd: Path | None = None
    env_extra: dict = field(default_factory=dict)
    optional: bool = False
    sudo: bool = False
    shell: bool = False
    interactive: bool = False  # if True, inherit terminal stdin/stdout (for scripts with prompts)

# ──────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK helpers
# ──────────────────────────────────────────────────────────────────────────────
def _container_running(name: str) -> bool:
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--filter", f"name=^{name}$",
             "--format", "{{.Status}}"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out) and ("Up" in out or "healthy" in out)
    except (subprocess.CalledProcessError, OSError):
        return False

def _gpu_mem_by_container_id() -> dict:
    """{full 64-char container id: total GPU memory used, in MiB}, summed
    across however many of that container's processes are on the GPU.

    nvidia-smi's own --query-compute-apps has no idea what a "container" is,
    only PIDs, so each PID is mapped to a container the same way `docker
    stats`/cAdvisor do it: a process's cgroup path contains its container's
    full id (cgroup v2 unified hierarchy, e.g. ".../docker-<64 hex>.scope";
    confirmed against a real container on this host). A PID that isn't a
    container's (e.g. something run directly on the host) has no match and
    is silently skipped. Returns {} if nvidia-smi isn't present/usable at
    all -- never raises.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, OSError):
        return {}
    totals: dict = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        pid, mem = parts
        try:
            cgroup = Path(f"/proc/{pid}/cgroup").read_text()
        except OSError:
            continue
        m = re.search(r"[0-9a-f]{64}", cgroup)
        if not m:
            continue
        totals[m.group(0)] = totals.get(m.group(0), 0) + int(mem)
    return totals

def _gpu_devices() -> list[dict]:
    """[{index, uuid, name, total_mib, used_mib, free_mib, mem_is_host_approx?}, ...],
    one per physical GPU nvidia-smi reports (device-wide totals, not
    per-container -- see _gpu_mem_by_container_and_device() for that).
    `uuid` is what nvidia-smi's own --query-compute-apps identifies a
    process's GPU by (it has no concept of "index" there), so it's the
    join key back to a specific device index -- None if this had to fall
    back to the uuid-less query below. `mem_is_host_approx` is only present
    (and True) when total_mib/used_mib/free_mib came from /proc/meminfo
    instead of nvidia-smi -- see the single-device branch below.

    Falls back to a query without `uuid` if the combined query comes back
    empty -- confirmed needed on a real GB10/DGX Spark (unified-memory
    Grace-Blackwell) host where --query-gpu=...,uuid,... returned nothing
    even though nvidia-smi itself works fine there (--query-compute-apps
    succeeds on the same host). total_mib/used_mib/free_mib are each None
    rather than dropping the whole device if nvidia-smi reports a field as
    something non-numeric (e.g. "[N/A]") -- another unified-memory quirk
    already seen elsewhere in this codebase (gpu_free_bytes()); on a
    single-device host this then falls back further to /proc/meminfo
    (_host_memory_mib()) -- confirmed on that same real host that
    nvidia-smi can't report a VRAM figure at all here, but /proc/meminfo
    and nvitop's own GPU-memory gauge agree exactly, since this chip has
    no separate VRAM pool to query in the first place. Returns [] if
    nvidia-smi isn't present/usable at all -- never raises.
    """
    def _to_int(s: str) -> int | None:
        try:
            return int(s)
        except ValueError:
            return None

    for fields, has_uuid in (
        ("index,uuid,name,memory.total,memory.used,memory.free", True),
        ("index,name,memory.total,memory.used,memory.free", False),
    ):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
            ).decode()
        except (subprocess.CalledProcessError, OSError):
            continue
        devices = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if has_uuid:
                if len(parts) != 6:
                    continue
                index, uuid, name, total, used, free = parts
            else:
                if len(parts) != 5:
                    continue
                index, name, total, used, free = parts
                uuid = None
            devices.append({
                "index": index, "uuid": uuid, "name": name,
                "total_mib": _to_int(total), "used_mib": _to_int(used), "free_mib": _to_int(free),
            })
        if devices:
            # Confirmed live on a real GB10/DGX Spark host: nvidia-smi's
            # memory.total/used/free fields come back non-numeric even in
            # the uuid-less fallback query above, even though the chip is
            # very much in active use (--query-compute-apps's own numbers
            # are correct). On a single-GPU unified-memory SoC there's no
            # separate VRAM pool to query in the first place -- /proc/meminfo
            # IS the GPU's memory, confirmed by nvitop showing the exact
            # same figure in both its "GPU memory" and host-memory gauges
            # on that chip. Approximate from there rather than show "N/A"
            # for a number that's actually knowable, but only when there's
            # exactly one device -- with more than one GPU, host RAM isn't
            # attributable to any specific card, so leave those as "N/A".
            if len(devices) == 1 and devices[0]["total_mib"] is None:
                host_mem = _host_memory_mib()
                if host_mem:
                    devices[0]["total_mib"], devices[0]["used_mib"], devices[0]["free_mib"] = host_mem
                    devices[0]["mem_is_host_approx"] = True
            return devices
    return []

def _host_memory_mib() -> tuple[int, int, int] | None:
    """(total_mib, used_mib, free_mib) from /proc/meminfo -- see
    _gpu_devices()'s own comment for why this is a legitimate stand-in for
    "GPU memory" specifically on a unified-memory SoC. Returns None if
    /proc/meminfo isn't readable (e.g. not on Linux) or missing either
    field.
    """
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    kib: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":") in ("MemTotal", "MemAvailable"):
            try:
                kib[parts[0].rstrip(":")] = int(parts[1])
            except ValueError:
                return None
    if "MemTotal" not in kib or "MemAvailable" not in kib:
        return None
    total_mib = kib["MemTotal"] // 1024
    free_mib = kib["MemAvailable"] // 1024
    return total_mib, total_mib - free_mib, free_mib

def _gpu_mem_by_container_and_device() -> dict:
    """{full 64-char container id: {gpu_uuid: used GPU memory, in MiB}} --
    like _gpu_mem_by_container_id() above, but keeping each process's own
    GPU apart instead of summing across every device, so a caller can say
    *which* physical GPU a container is actually using (join gpu_uuid
    against _gpu_devices()'s own uuid field for the human-readable index).
    Returns {} if nvidia-smi isn't present/usable at all -- never raises.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, OSError):
        return {}
    totals: dict = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        gpu_uuid, pid, mem = parts
        try:
            cgroup = Path(f"/proc/{pid}/cgroup").read_text()
        except OSError:
            continue
        m = re.search(r"[0-9a-f]{64}", cgroup)
        if not m:
            continue
        per_device = totals.setdefault(m.group(0), {})
        try:
            per_device[gpu_uuid] = per_device.get(gpu_uuid, 0) + int(mem)
        except ValueError:
            continue
    return totals

_DOCKER_MEM_UNIT_TO_MIB = {
    "b": 1 / 1024 / 1024, "kib": 1 / 1024, "kb": 1 / 1024,
    "mib": 1, "mb": 1, "gib": 1024, "gb": 1024, "tib": 1024 * 1024, "tb": 1024 * 1024,
}

def _cgroup_mem_mib() -> dict[str, int]:
    """{full 64-char container id: memory used, in MiB}, read directly
    from each container's own cgroup v2 memory.current -- an instantaneous
    gauge, unlike `docker stats --no-stream`'s MemUsage figure, which
    (confirmed live: ~2.45s for ~100 containers on a real host, vs. ~0.01s
    for this whole-host walk) pays a fixed internal sampling delay for its
    CPU%% rate calculation even when only memory is wanted. Matches the
    same cgroup v2 path shape (".../docker-<64 hex>.scope",
    systemd cgroup driver) _gpu_mem_by_container_id() already confirmed
    against a real container on this host. Returns {} if this host's
    cgroup layout doesn't match (cgroup v1, a different driver, rootless
    Docker, or no access to the host's own /sys/fs/cgroup) -- the caller
    falls back to `docker stats` in that case; never raises.
    """
    mib: dict[str, int] = {}
    try:
        for scope_dir in Path("/sys/fs/cgroup").rglob("docker-*.scope"):
            m = re.match(r"docker-([0-9a-f]{64})\.scope$", scope_dir.name)
            if not m:
                continue
            try:
                raw = (scope_dir / "memory.current").read_text().strip()
                mib[m.group(1)] = int(raw) // (1024 * 1024)
            except (OSError, ValueError):
                continue
    except OSError:
        return {}
    return mib

def _container_mem_mib() -> dict[str, int]:
    """{container id (full or short): memory used, in MiB} -- tries the
    fast cgroup path first (_cgroup_mem_mib()), falling back to the slow
    `docker stats` subprocess (_docker_stats_mem_mib()) only when that
    finds nothing. Kept as its own function (rather than folded into
    _docker_stats_mem_mib() itself) so existing callers/tests that
    exercise _docker_stats_mem_mib() directly keep testing its own parsing
    logic in isolation, unaffected by this preference order.
    """
    cgroup_mem = _cgroup_mem_mib()
    return cgroup_mem if cgroup_mem else _docker_stats_mem_mib()

def _docker_stats_mem_mib() -> dict[str, int]:
    """{short container id: memory used, in MiB}, from `docker stats`'s own
    cgroup-based MemUsage figure -- unlike nvidia-smi's --query-compute-apps
    (which only sees whatever a CUDA context explicitly registered), this is
    every container's actual resident memory, GPU-using or not. On a
    unified-memory SoC (see _gpu_devices()'s mem_is_host_approx) this is the
    number that actually answers "how much of the one shared memory pool is
    Raptor using", since the host RAM and the "VRAM" are the same physical
    bytes there. Returns {} if docker isn't usable at all -- never raises.
    This is the slow fallback -- see _container_mem_mib().
    """
    try:
        out = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format", "{{.ID}}\t{{.MemUsage}}"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, OSError):
        return {}
    mib: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        cid, mem_usage = parts
        used = mem_usage.split("/")[0].strip()
        m = re.match(r"^([\d.]+)\s*([A-Za-z]+)$", used)
        if not m:
            continue
        value, unit = m.groups()
        multiplier = _DOCKER_MEM_UNIT_TO_MIB.get(unit.lower())
        if multiplier is None:
            continue
        try:
            mib[cid] = int(float(value) * multiplier)
        except ValueError:
            continue
    return mib

def _raptor_host_mem_mib(
    modules: list[Module],
    stats_mem: dict[str, int] | None = None,
    rows_by_id: dict[str, list[tuple[str, str, str, str]]] | None = None,
) -> int:
    """Total host RAM (MiB) actually resident for every container across
    `modules` -- not just GPU modules, since on a unified-memory host every
    Raptor container draws from the same shared pool the "VRAM" figure
    comes from. Pass `stats_mem` (from _container_mem_mib()) to avoid a
    second fetch when the caller already has it. Pass
    `rows_by_id` ({module id: container_rows() result}) when the caller has
    already fetched it too -- each call is its own `docker ps` subprocess
    (see Module.container_rows()'s own docstring for why it can't be one
    shared query across modules), so a caller that also needs these rows
    for its own purposes (e.g. print_mem_usage()'s VRAM breakdown) should
    fetch once and pass both here and there, not trigger a second full
    pass over every module.
    """
    if stats_mem is None:
        stats_mem = _container_mem_mib()
    if not stats_mem:
        return 0
    total = 0
    for m in modules:
        if m.health_images:
            continue
        rows = rows_by_id.get(m.id, []) if rows_by_id is not None else m.container_rows()
        for _service, _state, _status_text, cid in rows:
            mem = next((v for k, v in stats_mem.items() if k == cid or k.startswith(cid) or cid.startswith(k)), None)
            if mem:
                total += mem
    return total

def _image_exists(name: str) -> bool:
    try:
        subprocess.check_output(
            ["docker", "image", "inspect", name],
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False

def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def _network_exists(name: str) -> bool:
    try:
        out = subprocess.check_output(
            ["docker", "network", "ls", "--filter", f"name=^{name}$",
             "--format", "{{.Name}}"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out == name
    except (subprocess.CalledProcessError, OSError):
        return False

# ──────────────────────────────────────────────────────────────────────────────
# RESTORE (VIE01-369 Phase 1)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RestoreSpec:
    """How --restore loads a module's backup output back into its
    datastore. "postgres"/"postgres_multi"/"redis"/"arcadedb" all restore
    into a container's own internal state (stop module -> start just the
    target container -> docker cp/exec into it); "host_path" (01) is
    different in kind, not just another entry in _RESTORE_KINDS -- there's
    no container to exec into, the backup is a plain host directory tree,
    so restore_module() branches to _restore_host_path() for it before
    any of the container-based machinery below runs. The fetch/decrypt/
    extract steps are shared by both shapes.
    """
    kind: str                     # "postgres" | "postgres_multi" | "redis" | "arcadedb" | "host_path"
    backup_prefix: str             # matches the sidecar's BACKUP_FILENAME/BACKUP_PRUNING_PREFIX
    dump_path_in_archive: str = "" # container-based kinds: path inside the extracted tar, e.g. "backup/keycloak-db/keycloak-db.dump"
    target_service: str = ""       # container-based kinds: compose service name holding the datastore, e.g. "keycloak-db"
    host_dest_env_var: str = ""    # "host_path" kind only: env var naming the real host directory to restore onto, e.g. "NFS_BASE_PATH"


# ──────────────────────────────────────────────────────────────────────────────
# MODULE
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Module:
    id: str
    name: str
    description: str
    steps: list[Step]
    deps: list[str]
    health_containers: list[str] = field(default_factory=list)
    health_images: list[str] = field(default_factory=list)
    gpu: bool = False
    nfs_init_script: str | None = None  # path relative to module dir, called with --clean on --delete
    delete_warn: str | None = None      # warning message printed on --delete
    restore: RestoreSpec | None = None  # VIE01-369 Phase 1 -- see restore_module()
    # Kept in the tree for rollback (see README's Deprecated Modules section),
    # not part of the live pipeline -- excluded from a plain --all/no-flag run
    # (build, start, stop, delete alike), but still reachable via `-m <id>`
    # for anyone who explicitly needs to touch one.
    deprecated: bool = False
    extends_shared_image: bool = False  # own image built FROM module 08's

    @property
    def dir(self) -> Path:
        return SCRIPT_DIR / self.name

    def is_healthy(self) -> bool:
        if self.health_images:
            return all(_image_exists(img) for img in self.health_images)
        if not self.health_containers:
            return True
        return any(_container_running(c) for c in self.health_containers)

    def container_rows(self) -> list[tuple[str, str, str, str]]:
        """(service, state, status_text, container_id) for every container
        this module's compose file actually created (running or stopped) --
        e.g. ("redis-standalone", "running", "Up 10 minutes", "a1b2c3...").
        container_id is the short (12-char) id `docker ps` gives, a prefix of
        the full id _gpu_mem_by_container_id() keys on.

        Scoped via the config_files label rather than `docker compose ps`,
        which (confirmed live) ignores -f and returns the whole stack's
        containers when COMPOSE_PROJECT_NAME is shared across modules -- see
        the --logs implementation below for the same fix.
        """
        compose_file = self.dir / "docker-compose.yml"
        if not compose_file.exists():
            return []
        result = subprocess.run(
            ["docker", "ps", "-a",
             "--filter", f"label=com.docker.compose.project.config_files={compose_file}",
             "--format", '{{.Label "com.docker.compose.service"}}\t{{.State}}\t{{.Status}}\t{{.ID}}'],
            check=False, capture_output=True, text=True,
        )
        rows = []
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) == 4:
                rows.append((parts[0], parts[1], parts[2], parts[3]))
        return rows

    def container_breakdown(self, rows: list[tuple[str, str, str, str]] | None = None) -> tuple[str, list[str]]:
        """A fuller check than is_healthy(): 'running'/'degraded'/'stopped'
        plus which services are the problem, by looking at *every* container
        this module's compose file actually created -- not just the one or
        two named anchors is_healthy() checks with `any()`. That's cheap and
        fine for the build-dependency check it's used for, but it's exactly
        why a module can show "running" while half its containers are
        crash-looping: the anchor happens to still be up.

        Pass `rows` (from a container_rows() you already fetched) to avoid
        a second `docker ps` call for the same module.
        """
        if self.health_images:
            return ("running" if self.is_healthy() else "stopped"), []
        if rows is None:
            rows = self.container_rows()
        if not rows:
            return "stopped", []
        problems = []
        for service, state, status_text, _cid in rows:
            # A one-shot init container (e.g. redis-cluster-init,
            # kafka-init-topics) is supposed to exit 0 and stay stopped --
            # that's success, not a problem.
            if state == "running" or (state == "exited" and "Exited (0)" in status_text):
                continue
            problems.append(f"{service} ({status_text or state})")
        return ("degraded" if problems else "running"), problems

# ──────────────────────────────────────────────────────────────────────────────
# STOP MODULE
# ──────────────────────────────────────────────────────────────────────────────
def stop_module(module: Module, log: Logger, remove_volumes: bool = False) -> bool:
    log.header(f"{'Delete' if remove_volumes else 'Stop'} [{module.id}]  {module.name}")
    compose_file = module.dir / "docker-compose.yml"
    if not compose_file.exists():
        log.warning("  No docker-compose.yml found — skipping")
        return True

    # --profile "*" (also used by the --status menu below): `down` only ever
    # sees services active under the current profile selection, so a
    # profile-gated service (02's "cluster", 14's "gpu", 35's text/voice)
    # started under a profile this call doesn't pass would otherwise be
    # silently skipped and left running.
    cmd = ["docker", "compose"] + _env_file_args(module.dir) + ["--profile", "*", "-f", str(compose_file), "down"]
    if remove_volumes:
        cmd.append("-v")
    log.step(f"  → docker compose down{' -v' if remove_volumes else ''}")
    result = subprocess.run(cmd, check=False, cwd=str(module.dir))
    if result.returncode != 0:
        log.error(f"Module [{module.id}] {module.name} — {'delete' if remove_volumes else 'stop'} FAILED")
        return False

    if remove_volumes and module.delete_warn:
        log.warning(module.delete_warn)

    if remove_volumes and module.nfs_init_script:
        script = module.dir / module.nfs_init_script
        log.step(f"  → Cleaning NFS directories ({module.nfs_init_script} --clean)")
        result = subprocess.run(["sudo", "bash", str(script), "--clean"], check=False, cwd=str(module.dir))
        if result.returncode != 0:
            log.warning(f"  NFS cleanup returned {result.returncode} — continuing")

    log.ok(f"Module [{module.id}] {module.name} — {'deleted' if remove_volumes else 'stopped'}")
    return True

# ──────────────────────────────────────────────────────────────────────────────
# RESTORE (VIE01-369 Phase 1)
# ──────────────────────────────────────────────────────────────────────────────
_AGE_MAGIC = b"age-encryption.org/v1"


def _is_age_encrypted(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(len(_AGE_MAGIC)) == _AGE_MAGIC


def _find_latest_backup(dest_dir: Path, prefix: str) -> Path | None:
    """Filenames embed an ISO-ish timestamp (BACKUP_FILENAME's
    %Y-%m-%dT%H-%M-%S), so lexicographic sort is chronological sort.
    `*.tar.gz` alone misses age-encrypted backups -- confirmed live, the
    tool appends `.age` on top (`06-keycloak-db-....tar.gz.age`), it
    doesn't replace the extension."""
    candidates = sorted(dest_dir.glob(f"{prefix}*.tar.gz*"))
    return candidates[-1] if candidates else None


def _wait_for_healthy(container: str, log: Logger, timeout: float = 120.0) -> bool:
    """`docker compose up -d <service>` only blocks on a healthcheck when
    something else depends on that service -- confirmed live: starting
    keycloak-db alone (restore's case, nothing else up yet to depend on it)
    returned immediately, and pg_restore then raced Postgres's own startup
    and failed on a socket that didn't exist yet. Poll explicitly instead of
    assuming `up -d` already waited."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", container],
            check=False, capture_output=True, text=True,
        )
        status = result.stdout.strip()
        if status == "healthy":
            return True
        if status == "unhealthy":
            log.error(f"  {container} reported unhealthy")
            return False
        time.sleep(2)
    log.error(f"  Timed out waiting for {container} to become healthy")
    return False


def _restore_postgres(container: str, dump_file: Path, log: Logger) -> bool:
    """Reuses the target container's own already-set POSTGRES_USER/PASSWORD/
    DB env vars -- same trick the archive-pre backup hook uses, no separate
    restore credentials needed."""
    log.step(f"  → Copying dump into {container}...")
    if subprocess.run(["docker", "cp", str(dump_file), f"{container}:/tmp/restore.dump"], check=False).returncode != 0:
        log.error("  docker cp failed")
        return False
    log.step("  → Running pg_restore --clean...")
    restore_sh = (
        "PGPASSWORD=$POSTGRES_PASSWORD pg_restore -U $POSTGRES_USER -d $POSTGRES_DB "
        "--clean --if-exists --no-owner --no-privileges /tmp/restore.dump"
    )
    ok = subprocess.run(["docker", "exec", container, "sh", "-c", restore_sh], check=False).returncode == 0
    subprocess.run(["docker", "exec", container, "rm", "-f", "/tmp/restore.dump"], check=False)  # nosec B108 -- scoped to this container's own namespace, not a shared host /tmp; removed immediately after use
    if not ok:
        log.error("  pg_restore failed")
        return False
    log.ok("  pg_restore completed")
    return True


def _restore_postgres_multi(container: str, dump_dir: Path, log: Logger) -> bool:
    """03 shares one Postgres instance across several logical databases --
    dump_dir holds one <dbname>.dump file per database (archive-pre's own
    loop names them), each restored with its own -d <dbname> instead of the
    container's single POSTGRES_DB. Reuses the container's own already-set
    POSTGRES_USER/PASSWORD, same as _restore_postgres()."""
    ok = True
    for dump_file in sorted(dump_dir.glob("*.dump")):
        db = dump_file.stem
        log.step(f"  → Restoring {db}...")
        if subprocess.run(["docker", "cp", str(dump_file), f"{container}:/tmp/restore.dump"], check=False).returncode != 0:
            log.error(f"  docker cp failed for {db}")
            ok = False
            continue
        restore_sh = (
            f"PGPASSWORD=$POSTGRES_PASSWORD pg_restore -U $POSTGRES_USER -d {db} "
            "--clean --if-exists --no-owner --no-privileges /tmp/restore.dump"
        )
        if subprocess.run(["docker", "exec", container, "sh", "-c", restore_sh], check=False).returncode != 0:
            log.error(f"  pg_restore failed for {db}")
            ok = False
        subprocess.run(["docker", "exec", container, "rm", "-f", "/tmp/restore.dump"], check=False)  # nosec B108 -- scoped to this container's own namespace, not a shared host /tmp; removed immediately after use
    if ok:
        log.ok("  pg_restore completed for every database")
    return ok


def _restore_redis(container: str, dump_file: Path, log: Logger) -> bool:
    """Confirmed live: with --appendonly yes, Redis 7 ignores a plain
    /data/dump.rdb swap on restart -- the AOF base file (named by the AOF
    manifest, e.g. appendonly.aof.1.base.rdb) is what actually gets loaded.
    Reads the manifest the fresh container already wrote on its own,
    overwrites that base file with the restored RDB, then restarts.
    Assumes the target volume was freshly (re)created by this restore's own
    earlier "start target_service" step (base seq 1) -- not an
    already-long-running instance with multiple AOF rewrites behind it."""
    result = subprocess.run(
        ["docker", "exec", container, "cat", "/data/appendonlydir/appendonly.aof.manifest"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        log.error(f"  Could not read AOF manifest from {container}")
        return False
    base_name = next((p[1] for line in result.stdout.splitlines() if len(p := line.split()) >= 6 and p[5] == "b"), None)
    if base_name is None:
        log.error("  No base-file entry found in the AOF manifest")
        return False
    log.step(f"  → Stopping {container} to swap the AOF base file...")
    if subprocess.run(["docker", "stop", container], check=False).returncode != 0:
        log.error(f"  Failed to stop {container}")
        return False
    ok = subprocess.run(["docker", "cp", str(dump_file), f"{container}:/data/appendonlydir/{base_name}"], check=False).returncode == 0
    if ok:
        incr_name = next((p[1] for line in result.stdout.splitlines() if len(p := line.split()) >= 6 and p[5] == "i"), None)
        if incr_name:
            empty_file = Path(tempfile.mkstemp()[1])
            subprocess.run(["docker", "cp", str(empty_file), f"{container}:/data/appendonlydir/{incr_name}"], check=False)
            empty_file.unlink(missing_ok=True)
    if subprocess.run(["docker", "start", container], check=False).returncode != 0:
        log.error(f"  Failed to restart {container}")
        return False
    if not ok:
        log.error("  docker cp failed")
        return False
    if not _wait_for_healthy(container, log):
        return False
    log.ok("  Redis AOF base file restored")
    return True


def _restore_arcadedb(container: str, dump_dir: Path, log: Logger) -> bool:
    """One ArcadeDB server here hosts one database per user -- BACKUP
    DATABASE is per-database (confirmed live: no server-wide equivalent),
    so dump_dir holds one subdirectory per database, each with its own
    timestamped backup zip (ArcadeDB's own default naming). restore.sh
    needs the server not running (confirmed live) -- restores in place via
    a one-off container sharing the same image and data volume, matching
    the official docs' command-line restore path, rather than touching
    arcadedb.server.restoreImportAllowLocalUrls (a security setting) to
    make the HTTP-API restore path accept a local file:// URL instead."""
    image = subprocess.run(["docker", "inspect", container, "--format", "{{.Config.Image}}"],
                            check=False, capture_output=True, text=True).stdout.strip()
    data_volume = subprocess.run(
        ["docker", "inspect", container, "--format",
         '{{range .Mounts}}{{if eq .Destination "/home/arcadedb/databases"}}{{.Name}}{{end}}{{end}}'],
        check=False, capture_output=True, text=True,
    ).stdout.strip()
    if not image or not data_volume:
        log.error(f"  Could not resolve {container}'s image/data volume")
        return False

    log.step(f"  → Stopping {container} (restore.sh needs the server not running)...")
    if subprocess.run(["docker", "stop", container], check=False).returncode != 0:
        log.error(f"  Failed to stop {container}")
        return False

    ok = True
    for db_dir in sorted(p for p in dump_dir.iterdir() if p.is_dir()):
        zips = sorted(db_dir.glob("*.zip"))
        if not zips:
            continue
        db_name = db_dir.name
        log.step(f"  → Restoring database '{db_name}'...")
        result = subprocess.run(
            ["docker", "run", "--rm",
             "-v", f"{data_volume}:/home/arcadedb/databases",
             "-v", f"{zips[-1]}:/tmp/restore.zip:ro",
             "--entrypoint", "/home/arcadedb/bin/restore.sh",
             image,
             "-f", "file:///tmp/restore.zip", "-d", f"/home/arcadedb/databases/{db_name}", "-o"],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error(f"  Restoring '{db_name}' failed: {result.stderr.strip()[-500:]}")
            ok = False

    if subprocess.run(["docker", "start", container], check=False).returncode != 0:
        log.error(f"  Failed to restart {container}")
        return False
    if not ok:
        return False
    if not _wait_for_healthy(container, log):
        return False
    log.ok("  ArcadeDB databases restored")
    return True


_RESTORE_KINDS = {"postgres": _restore_postgres, "postgres_multi": _restore_postgres_multi,
                  "redis": _restore_redis, "arcadedb": _restore_arcadedb}


def _restore_host_path(module: Module, spec: RestoreSpec, extract_dir: Path, log: Logger) -> bool:
    """01's backup is a plain host directory tree (seaweedfs/, memvid/
    storage), not a container's internal state -- no datastore to exec
    into, so this stops the whole module, rsyncs each top-level entry the
    archive actually has back onto the real host path (HOST_DEST_ENV_VAR,
    e.g. NFS_BASE_PATH), then starts the module again.

    Deliberately scoped to module 01 alone, same as its own backup --
    doesn't stop 04's/26's containers, which also read/write this same
    physical tree live. A real restore should stop those first too (same
    category of gap as the backup side's own write-freeze not covering
    everything that can write under NFS_BASE_PATH -- see
    docs/_internal/backup-restore-v0.5.md's "01" section), not handled here yet.
    """
    if shutil.which("rsync") is None:
        log.error("  'rsync' not found on this host -- required to restore onto NFS_BASE_PATH")
        return False
    dest_root_str = _read_env_var(spec.host_dest_env_var, "")
    if not dest_root_str:
        # Path("") == Path(".") -- checking the raw string before wrapping
        # it in Path() matters: the other way around, an unset var would
        # have silently resolved to the current working directory instead
        # of erroring (confirmed live -- caught by this function's own
        # test, not by inspection).
        log.error(f"  {spec.host_dest_env_var} is not set")
        return False
    dest_root = Path(dest_root_str)

    entries = [p for p in extract_dir.iterdir() if p.is_dir()]
    if not entries:
        log.error(f"  Extracted archive under {extract_dir} has nothing to restore")
        return False

    log.step(f"  → Stopping module [{module.id}] before restore...")
    if not stop_module(module, log, remove_volumes=False):
        log.error("  Failed to stop module before restore -- aborting")
        return False

    # Real incident, 2026-11-03: a mid-loop rsync failure (e.g. "Operation
    # not permitted" setting mtimes on files some earlier container run
    # left root-owned) used to `return False` right here, leaving the
    # module stopped -- a failed restore turned into an outage with no
    # recovery. restore_ok tracks whether the copy itself succeeded, but
    # the module always gets a restart attempt now, same reasoning as
    # scripts/nfs-freeze-archive.sh's own `trap ... EXIT`.
    restore_ok = True
    for entry in entries:
        dest = dest_root / entry.name
        dest.mkdir(parents=True, exist_ok=True)
        log.step(f"  → Restoring {entry.name} -> {dest}...")
        result = subprocess.run(
            # --no-times alongside --no-owner/--no-group, same reason: this
            # runs as a plain host user, but these trees can hold files a
            # container already wrote under its own internal UID (confirmed
            # live -- memvid/storage owned by UID 10026, not this host user).
            # Setting mtime (-t) or permission bits (-p) on a file you don't
            # own requires being root -- world-writable permission bits
            # alone aren't enough -- so -a's implied -t/-p fail with
            # "Operation not permitted" on any such file, aborting the whole
            # rsync (exit 23) even though the actual content copy -- the
            # part that matters -- already succeeded. The chmod -R 777
            # right after this call is what actually has to make the
            # restored tree readable/writable by whatever UID the
            # consuming container expects; these preserved attributes never
            # were load-bearing for that.
            ["rsync", "-a", "--no-owner", "--no-group", "--no-times", "--no-perms", "--delete", f"{entry}/", f"{dest}/"],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error(f"  rsync failed for {entry.name}: {result.stderr.strip()}")
            restore_ok = False
            break
        # init-nfs.sh's own convention -- confirmed live, skipping this makes
        # SeaweedFS's master refuse to start on the restored data at all
        # ("please verify /data/masterN is writable ... permission denied"):
        # --no-owner/--no-group means restored files keep whatever UID ran
        # this rsync, not whatever UID the container that reads them expects.
        subprocess.run(["chmod", "-R", "777", str(dest)], check=False)

    log.step(f"  → Starting module [{module.id}] again...")
    if not run_module(module, log):
        log.error("  Module failed to start after restore")
        return False
    if not restore_ok:
        log.error(f"  Module [{module.id}] {module.name} restarted, but the data restore itself failed above -- its data may be incomplete")
        return False
    log.ok(f"Module [{module.id}] {module.name} — restored")
    return True


def restore_module(module: Module, key_path: Path | None, restore_file: Path | None, log: Logger) -> bool:
    """Fetch (local path only so far -- SSH/WebDAV have no restore-side
    fetch built yet), decrypt (if age-encrypted), extract, and apply the
    latest (or --restore-file) backup, then start the module normally. The
    steps here are shared across every module; only RestoreSpec and the
    per-kind apply function (_RESTORE_KINDS) are module-specific.
    """
    log.header(f"Restore [{module.id}]  {module.name}")
    spec = module.restore
    if spec is None:
        log.error(f"  Module [{module.id}] {module.name} has no backup/restore configured yet")
        return False
    if spec.kind != "host_path":
        apply_fn = _RESTORE_KINDS.get(spec.kind)
        if apply_fn is None:
            log.error(f"  Restore kind '{spec.kind}' not implemented yet")
            return False

    dest_dir = Path(_read_env_var("BACKUP_LOCAL_ARCHIVE_PATH", str(module.dir / "backup-archive-default")))
    archive = restore_file or _find_latest_backup(dest_dir, spec.backup_prefix)
    if archive is None:
        log.error(f"  No backup found under {dest_dir} matching {spec.backup_prefix}*.tar.gz")
        return False
    if not archive.exists():
        log.error(f"  Backup file not found: {archive}")
        return False
    log.step(f"  → Using backup: {archive}")

    work_dir = Path(tempfile.mkdtemp(prefix=f"raptor-restore-{module.id}-"))
    try:
        tar_path = archive
        if _is_age_encrypted(archive):
            if key_path is None:
                log.error("  Backup is age-encrypted -- pass --restore-key <path>")
                return False
            if shutil.which("age") is None:
                log.error("  'age' CLI not found on this host -- install it (e.g. apt install age) to decrypt")
                return False
            tar_path = work_dir / "decrypted.tar.gz"
            log.step("  → Decrypting with age...")
            result = subprocess.run(
                ["age", "-d", "-i", str(key_path), "-o", str(tar_path), str(archive)],
                check=False, capture_output=True, text=True,
            )
            if result.returncode != 0:
                log.error(f"  age decryption failed: {result.stderr.strip()}")
                return False
        else:
            log.step("  → Backup is not age-encrypted, using as-is")

        log.step("  → Extracting archive...")
        extract_dir = work_dir / "extracted"
        extract_dir.mkdir()
        result = subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(extract_dir)], check=False, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"  tar extraction failed: {result.stderr.strip()}")
            return False

        if spec.kind == "host_path":
            return _restore_host_path(module, spec, extract_dir, log)

        dump_file = extract_dir / spec.dump_path_in_archive
        if not dump_file.exists():
            log.error(f"  Expected dump file not found inside archive: {spec.dump_path_in_archive}")
            return False

        log.step(f"  → Stopping module [{module.id}] before restore...")
        if not stop_module(module, log, remove_volumes=False):
            log.error("  Failed to stop module before restore -- aborting")
            return False

        compose_file = module.dir / "docker-compose.yml"
        log.step(f"  → Starting {spec.target_service} only...")
        up_cmd = ["docker", "compose"] + _env_file_args(module.dir) + ["-f", str(compose_file), "up", "-d", spec.target_service]
        if subprocess.run(up_cmd, check=False, cwd=str(module.dir)).returncode != 0:
            log.error(f"  Failed to start {spec.target_service}")
            return False

        # Assumes this repo's universal container_name convention,
        # raptor-<service>-${RAPTOR_VERSION} (confirmed across every
        # module's docker-compose.yml seen so far) -- not resolved via
        # container_rows() because the container was *just* created above
        # and the compose label query has no reason to lag. 03's service key
        # is itself already "raptor-postgres" (confirmed live: prepending
        # unconditionally produced "raptor-raptor-postgres", a container
        # that doesn't exist) -- don't double it.
        base = spec.target_service if spec.target_service.startswith("raptor-") else f"raptor-{spec.target_service}"
        container = _versioned([base])[0]
        log.step(f"  → Waiting for {container} to be healthy...")
        if not _wait_for_healthy(container, log):
            return False

        if not apply_fn(container, dump_file, log):
            return False

        log.step(f"  → Starting the rest of module [{module.id}]...")
        if not run_module(module, log):
            log.error("  Module failed to start after restore")
            return False

        log.ok(f"Module [{module.id}] {module.name} — restored from {archive.name}")
        return True
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

# ──────────────────────────────────────────────────────────────────────────────
# BACKUP (VIE01-369 Phase 1 -- the write-freeze-coordinated trigger --restore
# above was a separate, already-built concern)
# ──────────────────────────────────────────────────────────────────────────────

# module id -> its offen/docker-volume-backup sidecar's own compose service
# name (minus the raptor-/-${RAPTOR_VERSION} wrapping every container_name
# in this repo uses). 01 is handled separately below -- its mechanism is a
# plain script inside its own nfs-backup sidecar, not this tool at all.
_BACKUP_SIDECARS = {
    "02": "redis-standalone-backup",
    "03": "postgres-backup",
    "06": "keycloak-db-backup",
    "24": "arcadedb-backup",
}


def _maintenance_request(path: str, log: Logger, *, ttl_seconds: int | None = None) -> bool:
    """POST to module 13's X-Internal-Token-gated /admin/maintenance/* --
    the exact same endpoint 01-nfs-server/backup/nfs-freeze-archive.sh
    already calls on its own, reusing its env vars (NFS_BACKUP_GATEWAY_URL/
    _INTERNAL_TOKEN) rather than inventing a second pair for the same
    concept, per the VIE01-369 plan's own note that a future --backup
    "can call the same two endpoints ... instead of building a second
    mechanism." stdlib-only (urllib), matching this script's no-extra-deps
    convention. Returns False (and logs, doesn't raise) on any failure or
    if the gateway URL isn't configured -- the caller decides whether that's
    fatal."""
    import urllib.error
    import urllib.request

    base_url = _read_env_var("NFS_BACKUP_GATEWAY_URL", "")
    if not base_url:
        log.warning("  NFS_BACKUP_GATEWAY_URL is unset -- cannot toggle the write-freeze; "
                    "a coordinated backup would be taken from a live, unfrozen stack")
        return False
    token = _read_env_var("NFS_BACKUP_GATEWAY_INTERNAL_TOKEN", "")
    url = base_url.rstrip("/") + path
    if ttl_seconds is not None:
        url += f"?ttl_seconds={ttl_seconds}"
    req = urllib.request.Request(url, method="POST", headers={"X-Internal-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 -- internal gateway URL from our own .env, not user input
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError) as exc:
        log.error(f"  Maintenance-mode request to {url} failed: {exc}")
        return False


def _trigger_sidecar_backup(container: str, log: Logger) -> bool:
    """offen/docker-volume-backup's own on-demand trigger -- `docker exec
    <sidecar> backup` -- confirmed by its docs; there is no cross-container
    coordination built into the tool itself, which is exactly why this
    sequencing has to live here."""
    log.step(f"  → Triggering backup: {container}")
    result = subprocess.run(["docker", "exec", container, "backup"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"  Backup trigger failed for {container}: {(result.stderr or result.stdout).strip()}")
        return False
    return True


def backup_module(module: Module, log: Logger) -> bool:
    """Trigger one module's own backup mechanism, outside any write-freeze
    coordination -- used when the caller asked for a single module that
    isn't part of the 03/04/24 triangle (02 or 06 alone). Triangle members
    should go through backup_all() instead, which wraps this same call in
    the shared write-freeze window."""
    version = _read_env_var("RAPTOR_VERSION", "")
    if module.id == "01":
        container = f"raptor-nfs-backup-{version}"
        result = subprocess.run(
            ["docker", "exec", "-e", "NFS_BACKUP_GATEWAY_URL=", container, "/scripts/nfs-freeze-archive.sh"],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error(f"  01's freeze-archive failed: {(result.stderr or result.stdout).strip()}")
            return False
        return True
    sidecar = _BACKUP_SIDECARS.get(module.id)
    if sidecar is None:
        log.error(f"  Module [{module.id}] {module.name} has no backup mechanism configured")
        return False
    return _trigger_sidecar_backup(f"raptor-{sidecar}-{version}", log)


def backup_all(modules: list[Module], log: Logger) -> bool:
    """Coordinates every requested module's own backup mechanism inside one
    write-freeze window when any of them is a 03/04/24-triangle member (01
    stands in for 04's slice) -- "an atomic snapshot of 01 only fixes the
    torn-copy-during-backup problem" if 03's asset_management dump and 24's
    BACKUP DATABASE are taken at different, uncoordinated moments (VIE01-369
    plan). 02/06 alone need no write-freeze at all; still exercised under it
    here for simplicity when bundled with a triangle member -- harmless,
    since both already have their own live-safe backup primitive (BGSAVE /
    pg_dump) and don't need the freeze to be consistent.

    Write-freeze is just module 13's /admin/maintenance toggle -- no
    container is ever stopped. That's enough: 04's and 26's own services
    each enforce the exact same Redis flag themselves
    (asset_management/maintenance_mode.py, app/core/maintenance_mode.py),
    specifically because 09-12's/11's Kafka workers and modules 21/33 call
    them directly, bypassing module 13's gateway -- confirmed by reading
    both files (VIE01-369, PR #414). So there's no second write path left
    for a container-stop to plug; adding one here would only reintroduce
    the "docker stop/start turns a backup into an outage" failure mode
    nfs-freeze-archive.sh's own history already ruled out (see its
    comments) -- for a gap that isn't actually open.
    Comes up before any backup triggers and back down in a finally block,
    so a crashed run still restores writes -- on top of the maintenance
    endpoint's own server-side TTL safety net.

    01's own nfs-freeze-archive.sh has its *own* built-in maintenance-mode
    toggle (for its independent, cron-scheduled use) -- invoked here with
    NFS_BACKUP_GATEWAY_URL cleared for just this one `docker exec` so it
    takes the no-op "not freezing writes" branch its own script already has,
    deferring to this command's own enable/disable instead of double-toggling.
    """
    needs_freeze = any(m.id in ("01", "03", "24") for m in modules)
    log.header("Coordinated backup" + (" -- write-freeze window" if needs_freeze else ""))
    froze_gateway = _maintenance_request("/admin/maintenance/enable", log,
                                         ttl_seconds=int(_read_env_var("NFS_BACKUP_GATEWAY_FREEZE_TTL_SECONDS", "600"))) \
        if needs_freeze else False
    ok = True
    try:
        for module in modules:
            if not backup_module(module, log):
                ok = False
    finally:
        if froze_gateway:
            _maintenance_request("/admin/maintenance/disable", log)
    (log.ok if ok else log.error)("  Coordinated backup " + ("finished" if ok else "finished with at least one failure -- see above"))
    return ok

# ──────────────────────────────────────────────────────────────────────────────
# PORT CHECKER
# ──────────────────────────────────────────────────────────────────────────────

def _parse_host_ports(env_file: Path) -> list[tuple[str, int]]:
    """Extract host-published port variables from an env file.

    Matches by the `PORT_*` naming convention (see docs/BUILD.md 3.1: `PORT_*` is
    always a host-published port; `*_PORT` is always a fixed, container-
    internal one) rather than by physical position in the file — a variable
    only needs the right name to be picked up here, regardless of which
    section it's defined under. This is deliberate: port vars that live
    outside a "HOST PORT SETTINGS" block (e.g. because someone added a new
    module and left its port in that module's own section) still get
    conflict-checked instead of being silently skipped.
    """
    result: list[tuple[str, int]] = []
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, val = stripped.partition("=")
            key = key.strip()
            if not key.startswith("PORT_"):
                continue
            val = val.split("#")[0].strip()   # strip inline comments
            try:
                port = int(val)
                if 1 <= port <= 65535:
                    result.append((key, port))
            except ValueError:
                pass
    return result


def _container_ports_from_compose(compose_file: Path) -> dict:
    """Extract {host_var_name: container_literal_port} from a module's
    docker-compose.yml `ports:` mappings. Only returns entries where the
    container side is a plain literal — same-variable pairs (e.g. Module
    02's Redis Cluster, where host and container are intentionally coupled)
    are skipped since there's no independent "real container port" to check
    a comment against. A leading bind address ("IP:host:container") is
    ignored."""
    result: dict = {}
    if not compose_file.exists():
        return result
    in_ports = False
    with open(compose_file, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if re.match(r'^ports:\s*$', stripped):
                in_ports = True
                continue
            if in_ports and stripped.startswith('-'):
                entry = stripped.lstrip('-').strip()
                entry = entry.split('#')[0].strip()
                entry = entry.strip('"')
                depth = 0
                parts = ['']
                for ch in entry:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                    if ch == ':' and depth == 0:
                        parts.append('')
                    else:
                        parts[-1] += ch
                # "IP:host:container" (e.g. CF-3's "${INTERNAL_BIND_IP:-127.0.0.1}:${PORT_X}:N",
                # Keycloak's "127.0.0.1:8080:8080") -- the leading bind address is dropped.
                if len(parts) == 3:
                    parts = parts[1:]
                if len(parts) == 2:
                    host, container = parts
                    hm = re.match(r'\$\{([A-Z_0-9]+)', host)
                    if hm and not container.strip().startswith('${'):
                        result[hm.group(1)] = container.strip()
                continue
            if in_ports and stripped and not stripped.startswith('-'):
                in_ports = False
    return result


def verify_container_port_comments(env_file: Path, log: Logger) -> bool:
    """Cross-check every `PORT_X=N  # → container M` line in the env file
    against the real, fixed container-side port in each module's
    docker-compose.yml. These comments are hand-written and nothing else
    keeps them honest — this is what catches them drifting (e.g. a module's
    container port changes but the root .env.example comment is never
    updated to match)."""
    var_to_container: dict = {}
    for module in MODULES:
        var_to_container.update(_container_ports_from_compose(module.dir / "docker-compose.yml"))

    problems: list[str] = []
    with open(env_file, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            m = re.match(r'^(PORT_[A-Z_0-9]+)=(\d+)[ \t]*(#.*)?$', line.rstrip())
            if not m:
                continue
            var, _hostval, comment = m.group(1), m.group(2), (m.group(3) or "").strip()
            real = var_to_container.get(var)
            if real is None:
                continue  # not a plain-literal container port (e.g. Redis Cluster) — nothing to check
            expected_prefix = f"# → container {real}"
            if not comment.startswith(expected_prefix):
                problems.append(
                    f"{env_file.name}:{lineno}  {var}=  real container port is {real}, "
                    f"comment says {comment or '(none)'!r} — expected to start with {expected_prefix!r}"
                )

    if problems:
        log.warning(f"  Container-port comment drift ({len(problems)}):")
        for p in problems:
            log.warning(f"    {p}")
        return False
    log.ok("  All PORT_* container-port comments match reality.")
    return True


def _port_in_use(port: int) -> bool:
    """Return True if the port is already bound on the host (tries IPv4 then IPv6)."""
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                s.bind(("", port))
        except OSError:
            return True
    return False


def _port_user(port: int) -> str:
    """Best-effort: return the process/container that is holding the port."""
    try:
        out = subprocess.check_output(
            ["ss", "-tlnp", f"sport = :{port}"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        for line in out.splitlines()[1:]:
            if line.strip():
                return line.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"  (ss unavailable, trying lsof: {e})", file=sys.stderr)
    try:
        out = subprocess.check_output(
            ["lsof", "-i", f":{port}", "-sTCP:LISTEN", "-n", "-P"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        for line in out.splitlines():
            if str(port) in line:
                return line.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"  (lsof unavailable, port user unknown: {e})", file=sys.stderr)
    return "unknown"


def _find_free_ports(near: int, taken: set[int], count: int = 3) -> list[int]:
    """Return up to `count` free ports by searching forward from near+1."""
    result: list[int] = []
    p = near + 1
    while len(result) < count and p <= 65535:
        if p not in taken and not _port_in_use(p):
            result.append(p)
        p += 1
    return result


def check_ports_cmd(env_file: Path, log: Logger):
    """Check HOST PORT SETTINGS for duplicates / in-use conflicts and auto-fix them."""
    from collections import defaultdict

    log.header(f"Port Conflict Check  —  {env_file.name}")

    ports = _parse_host_ports(env_file)
    if not ports:
        log.warning(f"No PORT_* variables detected in {env_file}.")
        return

    log.info(f"  Checking {len(ports)} port variables...\n")

    # ── Identify problems ─────────────────────────────────────────────────────
    # Duplicates: same port assigned to multiple variables (keep first, fix rest)
    port_to_vars: dict[int, list[str]] = defaultdict(list)
    for var, port in ports:
        port_to_vars[port].append(var)
    dup_vars: set[str] = {
        var
        for vars_ in port_to_vars.values() if len(vars_) > 1
        for var in vars_[1:]   # skip first occurrence
    }

    # In-use: port already bound on the host
    in_use_ports: set[int] = {port for _, port in ports if _port_in_use(port)}

    # ── Plan fixes: duplicates + in-use (both are real conflicts) ────────────
    taken: set[int] = {port for _, port in ports}
    fixes: dict[str, tuple[int, int, str]] = {}   # var → (old, new, reason)

    for var, port in ports:
        reason = ""
        if var in dup_vars:
            reason = "duplicate"
        elif port in in_use_ports:
            reason = "in use"
        if reason and var not in fixes:
            alts = _find_free_ports(port, taken, count=1)
            if alts:
                new_port = alts[0]
                taken.add(new_port)
                fixes[var] = (port, new_port, reason)

    # ── Display table ─────────────────────────────────────────────────────────
    col_w = max(len(v) for v, _ in ports)
    print(f"{C.BOLD}{C.DIM}  {'Variable':<{col_w}}  {'Port':>5}  Status{C.RESET}")
    print(f"  {C.DIM}{'─' * (col_w + 42)}{C.RESET}")

    for var, port in ports:
        if var in fixes:
            old, new, reason = fixes[var]
            print(
                f"  {C.CYAN}{var:<{col_w}}{C.RESET}  {C.BOLD}{old:>5}{C.RESET}  "
                f"{C.RED}✖ {reason.upper()}{C.RESET}  → {C.GREEN}{new}{C.RESET}"
            )
        else:
            print(
                f"  {C.CYAN}{var:<{col_w}}{C.RESET}  {C.BOLD}{port:>5}{C.RESET}  "
                f"{C.GREEN}✔ ok{C.RESET}"
            )

    print()

    # ── Apply fixes to file ───────────────────────────────────────────────────
    if fixes:
        content = env_file.read_text(encoding="utf-8")
        for var, (old, new, _) in fixes.items():
            content = re.sub(
                rf'^({re.escape(var)}\s*=\s*){old}(\s*(?:#.*)?)$',
                rf'\g<1>{new}\2',
                content, flags=re.MULTILINE,
            )
        env_file.write_text(content, encoding="utf-8")

        log.ok(f"  Auto-fixed {len(fixes)} port(s) in {env_file.name}:")
        for var, (old, new, reason) in fixes.items():
            log.ok(f"    {var}: {old} → {new}  ({reason})")
    else:
        log.ok(f"  All {len(ports)} ports are free and unique — no conflicts.")

    print()
    verify_container_port_comments(env_file, log)


def _detect_local_ip() -> str | None:
    """Best-effort outbound-facing local IP, for suggesting HOST_IP-style
    placeholders. Opens no real connection -- UDP connect() just resolves
    which local interface/route would be used."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None


def _generate_fernet_key() -> str:
    """Same scheme as cryptography.fernet.Fernet.generate_key() (32 random
    bytes, base64-urlsafe) without depending on that package."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()


def _generate_age_keypair() -> str | None:
    """Suggested value for an unfilled AGE_PUBLIC_KEYS placeholder --
    generates a real age keypair instead of making the operator run
    `age-keygen` by hand. Where the private key goes is AGE_PRIVATE_KEY_PATH
    (its own placeholder, suggested via _PATH_SUGGESTIONS below) -- read
    through _read_env_var() so it sees a value the operator just answered
    earlier in this same --init-env run, even before anything's written to
    disk (see _pending_env_overrides). Idempotent: if a key already exists
    at that path (e.g. --init-env was re-run after .env got reset), reuses
    it and returns its existing public key, rather than generating a new
    one and silently orphaning every backup already encrypted with the old
    one."""
    if shutil.which("age-keygen") is None:
        return None
    key_path = Path(_read_env_var("AGE_PRIVATE_KEY_PATH", _PATH_SUGGESTIONS["AGE_PRIVATE_KEY_PATH"])).expanduser()
    if key_path.exists():
        result = subprocess.run(["age-keygen", "-y", str(key_path)], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None

    # The whole point of asymmetric age encryption is that whoever can read
    # the encrypted backups still can't decrypt them -- refuse to put the
    # private key where that's not true.
    backup_dir = _read_env_var("BACKUP_LOCAL_ARCHIVE_PATH", "")
    if backup_dir and Path(backup_dir).expanduser().resolve() in (key_path.parent.resolve(), key_path.resolve()):
        print(f"{C.RED}  Refusing to write the age private key under BACKUP_LOCAL_ARCHIVE_PATH ({backup_dir}) -- "
              f"that's the one place it must never be{C.RESET}")
        return None

    key_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["age-keygen", "-o", str(key_path)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    os.chmod(key_path, 0o600)
    pub_result = subprocess.run(["age-keygen", "-y", str(key_path)], capture_output=True, text=True, check=False)
    if pub_result.returncode != 0 or not pub_result.stdout.strip():
        return None
    print(f"{C.YELLOW}  Generated a new age keypair -- private key saved to {key_path} (chmod 600).{C.RESET}")
    print(f"{C.YELLOW}  Move this file off this host and keep it offline -- it's the only thing that can decrypt any backup.{C.RESET}")
    return pub_result.stdout.strip()


def _generate_opensearch_password() -> str:
    # OpenSearch's security plugin requires upper, lower, digit and special chars
    return secrets.token_urlsafe(18) + "Aa1!"


# Credentials issued by an external service or an admin; never guessed locally.
_USER_ONLY_KEYS = {"HF_TOKEN", "MCP_KEYCLOAK_USERNAME", "MCP_KEYCLOAK_PASSWORD"}

_PATH_SUGGESTIONS = {
    "HF_CACHE_PATH": os.path.expanduser("~/.cache/huggingface"),
    "PANNS_CACHE_PATH": os.path.expanduser("~/.cache/panns_data"),
    "NFS_BASE_PATH": str(SCRIPT_DIR.parent.parent / "raptor-nfs"),
    # VIE01-369: deliberately NOT under BACKUP_LOCAL_ARCHIVE_PATH (or
    # anywhere in this repo) -- see _generate_age_keypair()'s own check.
    # Just a suggestion like every other path here: the operator can accept
    # it or type wherever they'll actually keep it offline (a mounted
    # drive, a synced vault directory, etc.).
    "AGE_PRIVATE_KEY_PATH": os.path.expanduser("~/.raptor-backup/age-private-key.txt"),
}

# Keys needing a specific generation scheme, not a generic random token.
_SPECIAL_PLACEHOLDER_GENERATORS = {
    "MCP_SECRET_ENCRYPTION_KEY": _generate_fernet_key,
    "OPENSEARCH_PASSWORD": _generate_opensearch_password,
    "AGE_PUBLIC_KEYS": _generate_age_keypair,
}


def _suggest_placeholder_value(key: str, value: str) -> str | None:
    """Best-effort full replacement for a value containing <placeholder>, or
    None if there's no sensible default to offer (caller just prompts)."""
    if key in _USER_ONLY_KEYS:
        return None
    if key in _PATH_SUGGESTIONS:
        return _PATH_SUGGESTIONS[key]
    if key in _SPECIAL_PLACEHOLDER_GENERATORS:
        return _SPECIAL_PLACEHOLDER_GENERATORS[key]()
    tokens = [t for t in re.split(r"[_<> ]+", f"{key} {value}".lower()) if t]
    if any(t in ("password", "secret", "token") or "key" in t for t in tokens):
        return re.sub(r"<[^<>]*>", lambda _: secrets.token_urlsafe(24), value)
    if "ip" in tokens:
        ip = _detect_local_ip()
        return re.sub(r"<[^<>]*>", lambda _: ip, value) if ip else None
    return None


def init_env_cmd(log: Logger):
    """Bootstrap missing .env files from .env.example (root + each module),
    then interactively fill placeholders in the root .env only -- module .env
    files are copied as-is since the root .env overrides them at deploy time.
    Skips a .env that already exists. Ctrl-C/EOF stops cleanly and keeps
    anything already answered.
    """
    log.header("Environment Setup")

    created: list[Path] = []
    root_example = SCRIPT_DIR / ".env.example"
    root_env = SCRIPT_DIR / ".env"
    if root_example.exists() and not root_env.exists():
        root_env.write_text(root_example.read_text(encoding="utf-8"), encoding="utf-8")
        created.append(root_env)
    for module_dir in sorted(p for p in SCRIPT_DIR.iterdir() if p.is_dir()):
        example = module_dir / ".env.example"
        env = module_dir / ".env"
        if example.exists() and not env.exists():
            env.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
            created.append(env)

    if created:
        log.ok(f"  Created {len(created)} .env file(s) from .env.example:")
        for p in created:
            log.ok(f"    {p.relative_to(SCRIPT_DIR)}")
    else:
        log.ok("  All .env files already exist -- nothing to bootstrap.")
    print()

    total_filled = 0
    total_found = 0
    interrupted = False
    for env_file in ([root_env] if root_env.exists() else []):
        problems: list[tuple] = []
        for lineno, line in enumerate(env_file.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, rest = stripped.partition("=")
            value = rest.split("#", 1)[0].strip()
            if re.search(r"<[^<>]*>", value):
                problems.append((key.strip(), lineno, value))
        if not problems:
            continue
        total_found += len(problems)

        log.header(f"{env_file.relative_to(SCRIPT_DIR)}  —  {len(problems)} unfilled placeholder(s)")
        fixes: list[tuple] = []
        width = max(len(key) for key, _, _ in problems)
        for key, lineno, current in problems:
            suggestion = _suggest_placeholder_value(key, current)
            if suggestion:
                prompt = f"  {C.CYAN}{key:<{width}}{C.RESET}  {C.GREEN}{suggestion}{C.RESET}  [Enter=accept] "
            else:
                prompt = f"  {C.CYAN}{key:<{width}}{C.RESET}  [type value] "
            try:
                answer = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                log.warning("  Interrupted — stopping here, anything already answered is still saved.")
                interrupted = True
                break
            if not answer:
                if suggestion is None:
                    log.warning(f"    skipped {key} (no value given)")
                    continue
                answer = suggestion
            fixes.append((key, current, answer))
            # So a later placeholder's generator in this same run (e.g.
            # AGE_PUBLIC_KEYS reading AGE_PRIVATE_KEY_PATH) sees this answer
            # via _read_env_var() -- nothing's written to disk until the
            # whole loop below finishes.
            _pending_env_overrides[key] = answer

        if fixes:
            content = env_file.read_text(encoding="utf-8")
            for key, old, new in fixes:
                content = re.sub(
                    rf'^({re.escape(key)}\s*=\s*){re.escape(old)}(\s*(?:#.*)?)$',
                    lambda m, new=new: m.group(1) + new + m.group(2),
                    content, count=1, flags=re.MULTILINE,
                )
            env_file.write_text(content, encoding="utf-8")
            total_filled += len(fixes)
            log.ok(f"  Filled {len(fixes)} value(s) in {env_file.name}.")
        print()

    if total_found == 0:
        log.ok("No unfilled placeholders found.")
    elif interrupted:
        log.warning(f"Stopped early — filled {total_filled} of {total_found} found placeholder(s).")
    else:
        log.ok(f"Done — filled {total_filled} placeholder(s) total.")


def check_placeholders_cmd(env_file: Path, log: Logger):
    """Scan every KEY=VALUE line in env_file for a `<...>` placeholder
    (e.g. `<your_postgres_password>`) that was never filled in — the .env.example
    template marks every value a deployer must supply this way, so anything
    still shaped like that in a real .env means the deploy is misconfigured.

    Also catches a double-scheme URL (`http://https://...`) — the shape a
    formula-built URL variable takes when it's fed a value that's already a
    full URL instead of the bare host/port piece it expected (confirmed to
    happen in practice: VIE01-421, GATEWAY_KEYCLOAK_PUBLIC_URL under 06's TLS
    entry before KEYCLOAK_PUBLIC_URL replaced it) — and reminds about the two
    TLS-only variables (06-authentication's README, "TLS for Keycloak and the
    gateway") whenever KEYCLOAK_TLS_ENABLED signals that entry is actually in use."""
    log.header(f"Placeholder Check  —  {env_file.name}")

    values: dict[str, str] = {}
    lines: dict[str, tuple[int, str]] = {}
    with open(env_file, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, rest = stripped.partition("=")
            key = key.strip()
            value = rest.split("#", 1)[0].strip()
            values[key] = value
            lines[key] = (lineno, value)

    placeholder_problems = [(k, ln, v) for k, (ln, v) in lines.items() if re.search(r"<[^<>]*>", v)]
    double_scheme_problems = [(k, ln, v) for k, (ln, v) in lines.items()
                               if len(re.findall(r"[a-zA-Z][a-zA-Z0-9+.-]*://", v)) > 1]

    if not placeholder_problems and not double_scheme_problems:
        log.ok(f"  No unfilled <placeholder> values in {env_file.name}.")
    if placeholder_problems:
        log.warning(f"  {len(placeholder_problems)} unfilled placeholder(s):\n")
        col_w = max(len(k) for k, _, _ in placeholder_problems)
        for key, lineno, value in placeholder_problems:
            print(f"  {C.CYAN}{key:<{col_w}}{C.RESET}  {C.DIM}line {lineno:<5}{C.RESET}  {C.RED}{value}{C.RESET}")
        print()
    if double_scheme_problems:
        log.warning(f"  {len(double_scheme_problems)} value(s) with more than one URL scheme "
                    f"(a formula fed a value that's already a full URL):\n")
        col_w = max(len(k) for k, _, _ in double_scheme_problems)
        for key, lineno, value in double_scheme_problems:
            print(f"  {C.CYAN}{key:<{col_w}}{C.RESET}  {C.DIM}line {lineno:<5}{C.RESET}  {C.RED}{value}{C.RESET}")
        print()

    # KEYCLOAK_TLS_ENABLED=true is what actually turns 06-authentication's TLS entry on (translated into
    # COMPOSE_PROFILES by _apply_keycloak_tls_mode(), not something to read back from values{} here --
    # this file's own raw text is all check_placeholders_cmd() has) -- that's the moment
    # KEYCLOAK_PUBLIC_URL/GATEWAY_PUBLIC_URL also need the https address, easy to forget since they're
    # two separate modules/sections.
    if values.get("KEYCLOAK_TLS_ENABLED", "").strip().lower() in _TRUE_VALUES:
        kc_url, gw_url = values.get("KEYCLOAK_PUBLIC_URL", ""), values.get("GATEWAY_PUBLIC_URL", "")
        if not kc_url.startswith("https://") or not gw_url.startswith("https://"):
            log.warning(
                "  KEYCLOAK_TLS_ENABLED=true, but these don't look like the TLS entry's https "
                "address yet (06-authentication/README.md, \"TLS for Keycloak and the gateway\"):\n"
            )
            print(f"  {C.CYAN}KEYCLOAK_PUBLIC_URL{C.RESET}  {C.RED}{kc_url or '(empty)'}{C.RESET}  "
                  f"{C.DIM}expected https://<host>:<PORT_KEYCLOAK_TLS>{C.RESET}")
            print(f"  {C.CYAN}GATEWAY_PUBLIC_URL{C.RESET}   {C.RED}{gw_url or '(empty)'}{C.RESET}  "
                  f"{C.DIM}expected https://<host>:<PORT_GATEWAY_TLS>{C.RESET}")
            print()


# ──────────────────────────────────────────────────────────────────────────────
# MODULE REGISTRY helpers
# ──────────────────────────────────────────────────────────────────────────────
def _env_file_args(module_dir: Path) -> list:
    if UNIFIED_ENV is not None:
        return ["--env-file", str(UNIFIED_ENV)]
    unified = SCRIPT_DIR / ".env"
    if unified.exists():
        return ["--env-file", str(unified)]
    local = module_dir / ".env"
    if local.exists():
        return ["--env-file", str(local)]
    return []

def _compose_up(compose_file: str, cwd: Path | None = None) -> Step:
    path = SCRIPT_DIR / compose_file
    module_dir = cwd or path.parent
    return Step(
        label=f"docker compose up -d  ({compose_file})",
        cmd=["docker", "compose"] + _env_file_args(module_dir) + ["-f", str(path), "up", "-d"],
        cwd=module_dir,
        sudo=False,
    )

def _make_executable(script: str) -> Step:
    return Step(
        label=f"chmod +x {script}",
        cmd=["chmod", "+x", str(SCRIPT_DIR / script)],
        sudo=True,
        optional=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# MODULE REGISTRY
# ──────────────────────────────────────────────────────────────────────────────
# Module 08 has no docker-compose.yml of its own (built directly via `docker
# build`), so its tag can't pick up RAPTOR_VERSION through compose's own
# ${VAR} substitution the way every other module's image: line does -- built
# here instead, once, for module 08's Module() entry below.
_MEDIA_WORKER_IMAGE = f"raptor/media-worker:{_read_env_var('RAPTOR_VERSION', '')}"

# Every compose file's container_name is now suffixed `-${RAPTOR_VERSION}`
# (so two versions can run on the same host); is_healthy()'s `docker ps`
# lookup has to match that same real, daemon-wide name, not the bare one.
_RAPTOR_VERSION = _read_env_var("RAPTOR_VERSION", "")

def _versioned(names: list[str]) -> list[str]:
    return [f"{n}-{_RAPTOR_VERSION}" for n in names]

MODULES: list[Module] = [

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="01",
        name="01-nfs-server",
        description="NFS Server — shared storage backing SeaweedFS volumes",
        deps=[],
        health_containers=_versioned(["raptor-nfs-server", "raptor-nfs-backup"]),
        nfs_init_script="init-nfs.sh",
        steps=[
            _make_executable("01-nfs-server/init-nfs.sh"),
            Step(
                label="Initialize NFS storage directories",
                cmd=[str(SCRIPT_DIR / "01-nfs-server/init-nfs.sh")]
                    + _env_file_args(SCRIPT_DIR / "01-nfs-server"),
                cwd=SCRIPT_DIR / "01-nfs-server",
                sudo=True,
                interactive=True,
            ),
            _compose_up("01-nfs-server/docker-compose.yml"),
        ],
        restore=RestoreSpec(kind="host_path", backup_prefix="01-nfs-", host_dest_env_var="NFS_BASE_PATH"),
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="02",
        name="02-redis-cluster",
        description="Redis standalone (default; used by all modules) — 6-node cluster is opt-in via COMPOSE_PROFILES=cluster",
        deps=[],
        health_containers=_versioned(["raptor-redis1", "raptor-redis-standalone"]),
        steps=[
            _compose_up("02-redis-cluster/docker-compose.yml"),
        ],
        restore=RestoreSpec(
            kind="redis",
            backup_prefix="02-redis-standalone-",
            dump_path_in_archive="backup/redis-standalone/redis-standalone.rdb",
            target_service="redis-standalone",
        ),
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="03",
        name="03-database",
        description="PostgreSQL (shared) + Qdrant vector DB",
        deps=[],
        health_containers=_versioned(["raptor-postgres", "raptor-qdrant"]),
        steps=[
            _compose_up("03-database/docker-compose.yml"),
        ],
        restore=RestoreSpec(kind="postgres_multi", backup_prefix="03-postgres-",
                            dump_path_in_archive="backup/postgres", target_service="raptor-postgres"),
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="04",
        name="04-object-storage",
        description="SeaweedFS distributed object storage + LakeFS + Asset Management API",
        deps=["01", "03"],
        health_containers=_versioned(["raptor-seaweedfs-master1", "raptor-seaweedfs-s3"]),
        delete_warn=(
            "LakeFS repository/commit metadata lives in PostgreSQL (module 03). "
            "--delete only wipes SeaweedFS volumes and NFS directories. "
            "To fully reset LakeFS, also delete module 03, or manually drop the LakeFS tables in PostgreSQL."
        ),
        steps=[
            _compose_up("04-object-storage/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="05",
        name="05-kafka-cluster",
        description="Kafka KRaft cluster (3 controllers + 3 brokers) + AKHQ UI",
        deps=["02"],
        health_containers=_versioned(["raptor-kafka-broker1", "raptor-kafka-broker2", "raptor-kafka-broker3"]),
        steps=[
            _compose_up("05-kafka-cluster/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="06",
        name="06-authentication",
        description="Keycloak identity provider with embedded PostgreSQL",
        deps=[],
        health_containers=_versioned(["raptor-keycloak"]),
        steps=[
            _compose_up("06-authentication/docker-compose.yml"),
        ],
        restore=RestoreSpec(
            kind="postgres",
            backup_prefix="06-keycloak-db-",
            dump_path_in_archive="backup/keycloak-db/keycloak-db.dump",
            target_service="keycloak-db",
        ),
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="07",
        name="07-ai-ml-services",
        description="MLflow tracking server + AI Lifecycle API",
        deps=["01", "03", "04"],
        health_containers=_versioned(["raptor-mlflow", "raptor-ai-lifecycle-api"]),
        gpu=True,
        steps=[
            _compose_up("07-ai-ml-services/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    # Shared base image for all 4 media processing modules (09–12).
    # Must be built before 09-audio-processing, 10-image-processing,
    # 11-video-processing, or 12-document-processing.
    Module(
        id="08",
        name="08-media-worker",
        description="Shared GPU worker image (torch 2.7.1 cu128 sm_120 / Blackwell + PaddlePaddle + WhisperX + docling) — used by modules 09 10 11 12",
        deps=[],
        health_images=[_MEDIA_WORKER_IMAGE],
        gpu=True,
        steps=[
            _make_executable("08-media-worker/scripts/check-arm64-paddle.sh"),
            Step(
                label="Check arm64 PaddlePaddle GPU wheel (no-op on amd64, or if one's already staged)",
                cmd=[str(SCRIPT_DIR / "08-media-worker/scripts/check-arm64-paddle.sh")],
                cwd=SCRIPT_DIR / "08-media-worker",
                interactive=True,
            ),
            Step(
                label=f"Build {_MEDIA_WORKER_IMAGE}  (build context = deployment/modules/)",
                cmd=[
                    "docker", "build",
                    "-t", _MEDIA_WORKER_IMAGE,
                    "-f", str(SCRIPT_DIR / "08-media-worker/Dockerfile"),
                    str(SCRIPT_DIR),
                ],
                cwd=SCRIPT_DIR,
            ),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="09",
        name="09-audio-processing",
        description="WhisperX STT / diarization / PANNs / audio summary workers",
        deps=["02", "03", "04", "05", "08"],
        health_containers=_versioned(["raptor-audio-recognizer"]),
        gpu=True,
        steps=[
            _compose_up("09-audio-processing/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="10",
        name="10-image-processing",
        description="InternVL image analysis + save-to-hybrid-search workers (GPU)",
        deps=["02", "03", "04", "05", "08"],
        health_containers=_versioned(["raptor-image-processing"]),
        gpu=True,
        steps=[
            _compose_up("10-image-processing/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="11",
        name="11-video-processing",
        description="Video chunking / frame description / OCR / summary workers (GPU)",
        deps=["02", "03", "04", "05", "08"],
        health_containers=_versioned(["raptor-video-frame-description"]),
        gpu=True,
        steps=[
            _compose_up("11-video-processing/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="12",
        name="12-document-processing",
        description="PDF/Office document analysis + LLM summary workers",
        deps=["02", "03", "04", "05", "08"],
        health_containers=_versioned(["raptor-document-analysis"]),
        gpu=True,
        steps=[
            _compose_up("12-document-processing/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="13",
        name="13-api-services",
        description="API Gateway (8012) + Asset Management + Search APIs",
        deps=["02", "03", "04", "06"],
        health_containers=_versioned(["raptor-api-gateway"]),
        steps=[
            _compose_up("13-api-services/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="14",
        name="14-monitoring",
        description="Prometheus + Grafana + Alertmanager + Node Exporter + Loki",
        deps=["03"],
        health_containers=_versioned(["raptor-prometheus", "raptor-grafana"]),
        steps=[
            _compose_up("14-monitoring/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="15",
        name="15-chat-service",
        description="Chat service — LangGraph RAG pipeline with hybrid search and Redis memory",
        # deps=["02", "17"],  # 17 retired, search goes through Module 25 now (never actually called 17's URL)
        deps=["02"],
        health_containers=_versioned(["raptor-chat-service"]),
        steps=[
            _compose_up("15-chat-service/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="16",
        name="16-training-service",
        description="GPU training orchestration via FastAPI (host 163)",
        deps=["01", "02", "07"],
        health_containers=_versioned(["raptor-training-service"]),
        gpu=True,
        steps=[
            _compose_up("16-training-service/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="17",
        name="17-hybrid-search",
        description="OpenSearch cluster (2 nodes) + Dashboards + Hybrid Search API",
        deps=[],
        health_containers=_versioned(["raptor-opensearch-node1", "raptor-hybridsearch-api"]),
        gpu=True,
        deprecated=True,
        steps=[
            Step(
                label="Check vm.max_map_count >= 262144 (required by OpenSearch)",
                cmd=[
                    "bash", "-c",
                    (
                        "val=$(sysctl -n vm.max_map_count 2>/dev/null"
                        " || cat /proc/sys/vm/max_map_count 2>/dev/null"
                        " || echo 0); "
                        'if [ "$val" -lt 262144 ]; then '
                        '  echo "ERROR: vm.max_map_count=$val — OpenSearch 需要 >= 262144"; '
                        '  echo "臨時修正 : sudo sysctl -w vm.max_map_count=262144"; '
                        '  echo "永久生效 : echo vm.max_map_count=262144 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p"; '
                        "  exit 1; "
                        "fi; "
                        'echo "vm.max_map_count=$val — OK"'
                    ),
                ],
            ),
            _compose_up("17-hybrid-search/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="18",
        name="18-query-orchestrator",
        description="Query Orchestrator — intent classification (FAISS+BM25), signal extraction, multi-backend RAG routing",
        # deps=["03", "04", "13", "17"],  # 17 retired, no reference to it anywhere in this module's own source
        deps=["03", "04", "13"],
        health_containers=_versioned(["raptor-query-orchestrator"]),
        gpu=True,   # query-orchestrator itself reserves an nvidia GPU device (docker-compose.yml)
        steps=[
            _compose_up("18-query-orchestrator/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="19",
        name="19-graph-database",
        description="Neo4j graph database",
        deps=[],
        health_containers=_versioned(["raptor-neo4j"]),
        deprecated=True,
        steps=[
            _compose_up("19-graph-database/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="20",
        name="20-graph-service",
        description="Graph Service — LLM-powered knowledge graph query and reasoning",
        deps=["02", "19"],
        health_containers=_versioned(["raptor-graph-service"]),
        deprecated=True,
        steps=[
            _compose_up("20-graph-service/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="21",
        name="21-agent-protocol",
        description="Agent Protocol — A2A discovery and orchestration",
        deps=["03", "13"],
        health_containers=_versioned(["raptor-agent-protocol"]),
        steps=[
            _compose_up("21-agent-protocol/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="22",
        name="22-benchmark-service",
        description="Benchmark Service — user-defined marking schemas, LLM-as-judge scoring, run history",
        deps=["02", "03", "07", "13"],
        health_containers=_versioned(["raptor-benchmark-service"]),
        steps=[
            _compose_up("22-benchmark-service/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="23",
        name="23-guardrail-service",
        description="Guardrail Service — content moderation (Llama Guard 3 / Granite Guardian / GPT-OSS-Safeguard), policy engine, audit logging",
        deps=["02", "03"],
        health_containers=_versioned(["raptor-guardrail-service"]),
        steps=[
            _compose_up("23-guardrail-service/docker-compose.yml"),
        ],
        gpu=True,   # guardrail_service itself reserves an nvidia GPU device (docker-compose.yml)
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="24",
        name="24-arcadedb-server",
        description="ArcadeDB Server — multi-model engine (graph + vector + BM25), one database per user",
        deps=[],
        health_containers=_versioned(["raptor-arcadedb"]),
        steps=[
            _compose_up("24-arcadedb-server/docker-compose.yml"),
        ],
        restore=RestoreSpec(
            kind="arcadedb",
            backup_prefix="24-arcadedb-",
            dump_path_in_archive="backup/arcadedb",  # a directory (one subdir per user database), not a single file
            target_service="arcadedb",
        ),
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="25",
        name="25-personal-db-service",
        description="Personal DB Service — per-user database lifecycle, hybrid/graph/temporal search, Kafka index consumer",
        deps=["02", "03", "05", "07", "13", "24"],
        health_containers=_versioned(["raptor-personal-db-service"]),
        gpu=True,   # bge-m3 embedder + bge-reranker-v2-m3 load on GPU (docker-compose.yml)
        steps=[
            _compose_up("25-personal-db-service/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="26",
        name="26-memory-service",
        description="Memory Service — persistent MemVID session/long-term memory",
        deps=["01", "02", "07", "13"],
        health_containers=_versioned(["raptor-memory-service"]),
        gpu=True,   # memory-service itself reserves an nvidia GPU device (docker-compose.yml)
        steps=[
            _compose_up("26-memory-service/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="27",
        name="27-mcp-server",
        description="MCP Server — exposes Raptor APIs as MCP 1.1 tools/resources/prompts (Streamable HTTP + stdio)",
        deps=["02", "06", "13"],
        health_containers=_versioned(["raptor-mcp-server"]),
        steps=[
            _compose_up("27-mcp-server/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="29",
        name="29-doc-processing-agent",
        description="DocAgent — 7-task document AI service (classify/extract/summarize/compare/qa/parse_table/search)",
        deps=["02", "03", "07", "12"],
        health_containers=_versioned(["raptor-doc-processing-agent"]),
        steps=[
            _compose_up("29-doc-processing-agent/docker-compose.yml"),
        ],
    ),


    # ──────────────────────────────────────────────────────────────────
    Module(
        id="33",
        name="33-privacy-compliance-service",
        description="Privacy Compliance Service — cross-store GDPR/CCPA erasure orchestrator (GD-1)",
        deps=["02", "03", "04", "06", "25", "26"],
        health_containers=_versioned(["raptor-privacy-compliance-service"]),
        steps=[
            _compose_up("33-privacy-compliance-service/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="34",
        name="34-realtime-audio",
        description="Realtime audio WebSocket, tickets, VAD and streaming STT",
        deps=["02", "06", "08"],
        health_containers=_versioned(["raptor-realtime-audio", "raptor-realtime-audio-recording"]),
        gpu=True,   # RA-3: faster-whisper STT reserves REALTIME_AUDIO_GPU (docker-compose.yml)
        extends_shared_image=True,
        steps=[_compose_up("34-realtime-audio/docker-compose.yml")],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="35",
        name="35-inference-service",
        description="LLM Inference Service — translate / summarize / complete / transform APIs and per-task default models on top of module 07",
        deps=["02", "07", "08"],  # 08: Dockerfile.voice's base image, needed once VOICE_ENABLED defaults true
        health_containers=_versioned(["raptor-inference-service"]),
        gpu=True,  # VIE01-376: only the --profile voice service reserves one; see docker-compose.yml
        extends_shared_image=True,  # Dockerfile.voice is FROM module 08's image, same as module 34
        steps=[
            # VIE01-376: GPU reservation sits only on the voice profile (see
            # docker-compose.yml's `services`) -- which --profile to pass is
            # decided below, by module id, same pattern as module 14's
            # ENABLE_GPU_MONITORING -> --profile gpu.
            _compose_up("35-inference-service/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="36",
        name="36-grpc-gateway",
        description="gRPC Gateway — RaptorService on :8825 (health, auth/metrics interceptors, search, assets, jobs)",
        deps=["02", "04", "05", "25"],
        health_containers=_versioned(["raptor-grpc-gateway"]),
        steps=[
            _compose_up("36-grpc-gateway/docker-compose.yml"),
        ],
    ),
]

# Build lookup maps
MODULE_BY_ID:   dict[str, Module] = {m.id: m for m in MODULES}
MODULE_BY_NAME: dict[str, Module] = {m.name: m for m in MODULES}

# ──────────────────────────────────────────────────────────────────────────────
# STEP RUNNER
# ──────────────────────────────────────────────────────────────────────────────
def run_step(step: Step, log: Logger) -> bool:
    log.step(f"  → {step.label}")

    env = os.environ.copy()
    env.update(step.env_extra)

    def _run_once(cmd: list[str], interactive: bool, stream_live: bool) -> tuple[int, list[str]]:
        """captured is [] whenever output already went straight to the
        terminal (interactive, or stream_live printing each line as it
        arrives) -- only non-empty when stream_live=False, for a caller
        that needs to decide whether showing it is actually useful."""
        if interactive:
            # Inherit terminal stdin/stdout/stderr so the script can prompt the user
            # step.shell is never True in any Step this file currently defines
            # (every cmd is this repo's own fixed argv list); kept for a future
            # step needing real shell features (pipes/globs), with every
            # element individually shlex.quote()'d first either way.
            proc = subprocess.run(
                cmd if not step.shell else " ".join(shlex.quote(c) for c in cmd),
                check=False,
                cwd=str(step.cwd) if step.cwd else None,
                env=env,
                shell=step.shell,  # nosec B602
            )
            return proc.returncode, []
        proc = subprocess.Popen(
            cmd if not step.shell else " ".join(shlex.quote(c) for c in cmd),
            cwd=str(step.cwd) if step.cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",  # malformed progress/log bytes must not abort the build
            shell=step.shell,  # nosec B602 -- see interactive branch above
        )
        assert proc.stdout is not None
        captured: list[str] = []
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if stream_live:
                # The normal case for every docker-compose build/up step --
                # real-time progress (e.g. a multi-minute image build) has
                # to print as it happens, not get buffered until the whole
                # step finishes (a real regression this fixes: that made
                # every module's build look hung, no output at all until
                # the entire step was already done).
                log.raw("     " + line)
            else:
                captured.append(line)
        proc.wait()
        return proc.returncode, captured

    try:
        if step.sudo:
            # sudo=True steps (currently just 01's init-nfs.sh and its own
            # chmod +x) only actually need root once something else already
            # wrote files under the target path with a different UID --
            # confirmed live: on a clean host neither needs it at all,
            # verified both ways. But on a host where that's permanently
            # true (e.g. SeaweedFS continuously running against
            # NFS_BASE_PATH, as on every real deployment), this first,
            # unprivileged attempt will fail every single time and sudo
            # will always be needed -- a real incident this surfaced:
            # a page of "Operation not permitted" lines on every single
            # `-m 01` run, alarming on a real host even though harmless and
            # not new (previously silent only because the step always ran
            # as root from the start). This one attempt is buffered, not
            # streamed, and only shown on success -- the sudo retry's own
            # (interactive, so visibly live) run is what actually matters
            # if it fails too.
            rc, captured = _run_once(step.cmd, interactive=False, stream_live=False)
            if rc == 0:
                for line in captured:
                    log.raw("     " + line)
            else:
                log.warning("     Plain attempt failed (needs root here) -- retrying with sudo...")
                rc, _ = _run_once(["sudo"] + step.cmd, interactive=step.interactive, stream_live=True)
        else:
            rc, _ = _run_once(step.cmd, interactive=step.interactive, stream_live=True)

        if rc != 0:
            if step.optional:
                log.warning(f"     (optional step exited {rc} — continuing)")
                return True
            log.error(f"     Step failed (exit {rc})")
            return False
        return True
    except FileNotFoundError as exc:
        if step.optional:
            log.warning(f"     (optional step not found: {exc} — continuing)")
            return True
        log.error(f"     Command not found: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001 -- runs arbitrary build-step subprocess commands; last-resort, already logged
        if step.optional:
            log.warning(f"     (optional step error: {exc} — continuing)")
            return True
        log.error(f"     Unexpected error: {exc}")
        return False

# ──────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECKER
# ──────────────────────────────────────────────────────────────────────────────
def check_deps(module: Module, log: Logger, strict: bool = False) -> bool:
    all_ok = True
    for dep_id in module.deps:
        dep = MODULE_BY_ID.get(dep_id)
        if dep is None:
            log.warning(f"  Unknown dependency module id '{dep_id}' — skipping check")
            continue
        if dep.is_healthy():
            log.ok(f"  Dep [{dep_id}] {dep.name} — running")
        else:
            msg = f"  Dep [{dep_id}] {dep.name} — NOT running"
            if strict:
                log.error(msg)
                all_ok = False
            else:
                log.warning(msg + " (will be started earlier in this run)")
    return all_ok

# ──────────────────────────────────────────────────────────────────────────────
# ENSURE DOCKER NETWORK
# ──────────────────────────────────────────────────────────────────────────────
def ensure_opensearch_metrics_secret(log: Logger):
    """Regenerate 14-monitoring/secrets/opensearch_password from OPENSEARCH_PASSWORD
    in .env, for Prometheus's opensearch scrape job (basic_auth.password_file).

    Prometheus's own config format doesn't support ${VAR} expansion, so the
    password can't be injected the way docker-compose does it for every other
    service -- the only alternative to writing it in plaintext into
    prometheus.yml (and thus into git) is a file mounted into the container,
    regenerated here from the single real source of truth (.env) each run so
    it always matches whatever OPENSEARCH_PASSWORD currently is.
    """
    secret_dir = SCRIPT_DIR / "14-monitoring" / "secrets"
    secret_file = secret_dir / "opensearch_password"
    password = _read_env_var("OPENSEARCH_PASSWORD", "")
    if not password or password.startswith("<"):
        log.warning("  OPENSEARCH_PASSWORD not set in .env — opensearch scrape job will fail auth")
        return
    secret_dir.mkdir(parents=True, exist_ok=True)
    secret_file.write_text(password, encoding="utf-8")
    secret_file.chmod(0o600)
    log.ok(f"  Wrote {secret_file.relative_to(SCRIPT_DIR)} from OPENSEARCH_PASSWORD")


def ensure_alertmanager_webhook(log: Logger):
    """Regenerate 14-monitoring/config/alertmanager/webhook_url from
    ALERT_WEBHOOK_URL in .env, so nobody has to hand-edit a file to get real
    alert notifications (and nobody accidentally commits a real webhook URL
    into a git-tracked file). Falls back to the safe, non-routable placeholder
    when ALERT_WEBHOOK_URL isn't set -- Alertmanager starts fine either way,
    it just logs a failed-delivery error instead of silently dropping alerts.
    """
    webhook_file = SCRIPT_DIR / "14-monitoring" / "config" / "alertmanager" / "webhook_url"
    url = _read_env_var("ALERT_WEBHOOK_URL", "")
    if not url:
        if not webhook_file.exists():
            webhook_file.parent.mkdir(parents=True, exist_ok=True)
            webhook_file.write_text("http://raptor-alert-webhook.invalid/replace-me", encoding="utf-8")
        log.warning("  ALERT_WEBHOOK_URL not set in .env — alerts will fire but nobody gets notified")
        return
    webhook_file.parent.mkdir(parents=True, exist_ok=True)
    webhook_file.write_text(url, encoding="utf-8")
    log.ok(f"  Wrote {webhook_file.relative_to(SCRIPT_DIR)} from ALERT_WEBHOOK_URL")


def ensure_raptor_network(log: Logger):
    network_name = _read_env_var("DOCKER_NETWORK", "raptor")
    log.step(f"Checking Docker network '{network_name}'...")
    if _network_exists(network_name):
        log.ok(f"Network '{network_name}' already exists")
        return
    # No explicit --subnet: nothing in any module's compose file assumes a
    # static IP in 172.30.0.0/16 (checked), and a second raptor-version
    # network on the same host (DOCKER_NETWORK set per-version) would
    # collide with that hardcoded range every time. Let Docker's own
    # default address pool pick a free one instead.
    log.step(f"Creating Docker network '{network_name}'...")
    result = subprocess.run(
        ["docker", "network", "create", network_name],
        check=False, capture_output=True, text=True,
    )
    if result.returncode == 0:
        log.ok(f"Network '{network_name}' created")
    else:
        log.warning(f"Could not create network: {result.stderr.strip()}")

# ──────────────────────────────────────────────────────────────────────────────
# MODULE RUNNER
# ──────────────────────────────────────────────────────────────────────────────
def run_module(
    module: Module,
    log: Logger,
    check_deps_strict: bool = False,
    build: bool = False,
    build_only: bool = False,
    built_set: set | None = None,
) -> bool:
    log.header(f"Module [{module.id}]  {module.name}")
    log.info(f"  {module.description}")

    if module.id == "14":
        log.step("Regenerating OpenSearch metrics secret from .env...")
        ensure_opensearch_metrics_secret(log)
        log.step("Regenerating Alertmanager webhook_url from .env...")
        ensure_alertmanager_webhook(log)

    if module.deps:
        log.step("Checking dependencies...")
        if not check_deps(module, log, strict=check_deps_strict):
            log.error(f"  Module [{module.id}] aborted — unmet dependencies")
            return False

    if not module.dir.exists():
        log.warning(f"  Module directory not found: {module.dir}  — skipping")
        return False

    success = True
    for step in module.steps:
        if (build or build_only) and "up" in step.cmd and "compose" in step.cmd:
            if "08" in module.deps and (built_set is None or "08" not in built_set):
                # Modules 09-12 share the raptor/media-worker image built by module 08.
                # --build/--build-only should rebuild that image, not re-run
                # docker compose --build. Modules with extends_shared_image (34)
                # also have their own image FROM it, rebuilt below after 08's.
                m08 = MODULE_BY_ID.get("08")
                if m08:
                    log.step("  → rebuilding shared image via module 08 first...")
                    for build_step in m08.steps:
                        if not run_step(build_step, log):
                            log.error("  Shared image (module 08) build FAILED")
                            return False
                if built_set is not None:
                    built_set.add("08")
            if "08" in module.deps and not module.extends_shared_image:
                if build_only:
                    continue  # image is built — --build-only never starts a container
                # compose up without --build — image was just rebuilt above
            elif build_only:
                # step.cmd always ends in ["-f", <path>, "up", "-d"] — see _compose_up()
                # shell=step.shell below just copies the existing Step's own
                # flag into the rebuilt one -- not a new subprocess call, and
                # not something bandit's generic "any function with shell="
                # check (B604) can tell apart from one.
                step = Step(
                    label=step.label.replace("up -d", "build"),
                    cmd=step.cmd[:-2] + ["build"],
                    cwd=step.cwd,
                    env_extra=step.env_extra,
                    optional=step.optional,
                    sudo=step.sudo,
                    shell=step.shell,  # nosec B604
                    interactive=step.interactive,
                )
            else:
                step = Step(
                    label=step.label + " --build",
                    cmd=step.cmd + ["--build"],
                    cwd=step.cwd,
                    env_extra=step.env_extra,
                    optional=step.optional,
                    sudo=step.sudo,
                    shell=step.shell,  # nosec B604 -- see rebuild branch above
                    interactive=step.interactive,
                )
        if (module.id == "14" and "compose" in step.cmd and "up" in step.cmd
                and _read_env_var("ENABLE_GPU_MONITORING", "false").strip().lower() in ("1", "true", "yes")):
            # dcgm-exporter is profiles:[gpu] in docker-compose.yml (opt-in --
            # it needs the NVIDIA DCGM stack on the host, so it's not started
            # by a plain `up -d`). Add the profile here instead of making
            # every operator remember `--profile gpu` by hand.
            idx = step.cmd.index("compose") + 1
            step = Step(
                label=step.label + " --profile gpu",
                cmd=step.cmd[:idx] + ["--profile", "gpu"] + step.cmd[idx:],
                cwd=step.cwd,
                env_extra=step.env_extra,
                optional=step.optional,
                sudo=step.sudo,
                shell=step.shell,  # nosec B604 -- see rebuild branch above
                interactive=step.interactive,
            )
        if module.id == "35" and "compose" in step.cmd and ("up" in step.cmd or "build" in step.cmd):
            # VIE01-376: docker-compose.yml's two services (text/voice) are
            # both profile-gated -- neither starts (nor builds) without an
            # explicit --profile -- so this one is mandatory, not opt-in like
            # gpu above. Must also catch "build" (not just "up"): --build-only
            # rewrites this step's cmd to end in "build" instead, above.
            profile = "voice" if _read_env_var("VOICE_ENABLED", "false").strip().lower() in ("1", "true", "yes") else "text"
            env_extra = dict(step.env_extra)
            if profile == "voice":
                # Dockerfile.voice's build secret (hf_token, environment:
                # HUGGINGFACE_TOKEN) is resolved by buildx reading its own
                # process environment, not compose's ${VAR} substitution --
                # see _read_env_var_resolved()'s docstring. Inject the real
                # value as an actual env var here so the build works the same
                # regardless of which Compose/BuildKit version is on PATH.
                token = _read_env_var_resolved("HUGGINGFACE_TOKEN")
                if token and not token.startswith("<"):
                    env_extra["HUGGINGFACE_TOKEN"] = token
            idx = step.cmd.index("compose") + 1
            step = Step(
                label=step.label + f" --profile {profile}",
                cmd=step.cmd[:idx] + ["--profile", profile] + step.cmd[idx:],
                cwd=step.cwd,
                env_extra=env_extra,
                optional=step.optional,
                sudo=step.sudo,
                shell=step.shell,  # nosec B604 -- see rebuild branch above
                interactive=step.interactive,
            )
        ok = run_step(step, log)
        if not ok:
            log.error(f"  Module [{module.id}] failed at step: {step.label}")
            success = False
            break

    if success:
        if module.id == "08" and built_set is not None:
            built_set.add("08")
        log.ok(f"Module [{module.id}] {module.name} — {'built' if build_only else 'started'}")
    else:
        log.error(f"Module [{module.id}] {module.name} — FAILED")

    return success

# ──────────────────────────────────────────────────────────────────────────────
# PRINT MODULE LIST
# ──────────────────────────────────────────────────────────────────────────────
def _print_module_rows(modules: list):
    hdr = f"  {'ID':>2}  {'Module Name':<30}  {'Dependencies':<20}  Description"
    print(f"{C.BOLD}{C.DIM}{hdr}{C.RESET}")
    print(f"  {C.DIM}{'─'*72}{C.RESET}")
    for m in modules:
        deps = ",".join(m.deps) if m.deps else "—"
        print(f"  {C.CYAN}{m.id:>2}{C.RESET}  {m.name:<30}  {C.DIM}{deps:<20}{C.RESET}  {m.description}")


def print_module_list(modules: list[Module] | None = None):
    target = modules or MODULES
    cpu = [m for m in target if not m.gpu]
    gpu = [m for m in target if m.gpu]

    print(f"\n{C.BOLD}{C.BLUE}Raptor {_RAPTOR_VERSION or '?'} — Module Registry{C.RESET}")

    print(f"\n{C.BOLD}{C.GREEN}── CPU-only ({len(cpu)} modules) {'─'*40}{C.RESET}")
    _print_module_rows(cpu)

    print(f"\n{C.BOLD}{C.YELLOW}── GPU required ({len(gpu)} modules) {'─'*37}{C.RESET}")
    _print_module_rows(gpu)
    print()

# ──────────────────────────────────────────────────────────────────────────────
# STATUS
# ──────────────────────────────────────────────────────────────────────────────
def print_status(modules: list[Module] | None = None):
    # The running/degraded/stopped classification always uses the thorough
    # per-container check (catches the "anchor's up but a sibling container
    # isn't" case even in the full overview). Only when -m narrowed to
    # specific module(s) do we also print every container's own state --
    # for the full 29-module overview that would be a wall of text.
    detailed = modules is not None
    target = modules or MODULES

    rows_by_id = {m.id: m.container_rows() for m in target if not m.health_images} if detailed else {}
    breakdowns = {
        m.id: m.container_breakdown(rows_by_id.get(m.id)) for m in target
    }
    def state_of(m: Module) -> str:
        return breakdowns[m.id][0]

    running = [m for m in target if state_of(m) == "running"]
    degraded = [m for m in target if state_of(m) == "degraded"]
    stopped = [m for m in target if state_of(m) == "stopped"]

    print(f"\n{C.BOLD}{C.BLUE}Raptor {_RAPTOR_VERSION or '?'} — Module Status{C.RESET}\n")
    print(f"  {'ID':>2}  {'Module Name':<30}  Status")
    print(f"  {C.DIM}{'─'*50}{C.RESET}")
    for m in target:
        if m.health_images:
            if m.is_healthy():
                status = f"{C.GREEN}● built{C.RESET}    {C.DIM}[image]{C.RESET}"
            else:
                status = f"{C.YELLOW}○ not built{C.RESET} {C.DIM}[image]{C.RESET}"
        else:
            state = state_of(m)
            if state == "running":
                status = f"{C.GREEN}● running{C.RESET}"
            elif state == "degraded":
                problems = breakdowns[m.id][1]
                status = f"{C.YELLOW}◐ degraded{C.RESET}  {C.DIM}{', '.join(problems)}{C.RESET}"
            else:
                status = f"{C.DIM}○ stopped{C.RESET}"
        print(f"  {C.CYAN}{m.id:>2}{C.RESET}  {m.name:<30}  {status}")

        if detailed and not m.health_images:
            rows = rows_by_id[m.id]
            gpu_mem = _gpu_mem_by_container_id() if m.gpu and rows else {}
            for service, state, status_text, cid in rows:
                ok = state == "running" or (state == "exited" and "Exited (0)" in status_text)
                dot = f"{C.GREEN}●{C.RESET}" if ok else f"{C.RED}✗{C.RESET}"
                mem = next((v for k, v in gpu_mem.items() if k.startswith(cid)), None)
                gpu_suffix = f"  {C.MAGENTA}[GPU {mem} MiB]{C.RESET}" if mem else ""
                print(f"        {dot} {service:<28} {C.DIM}{status_text or state}{C.RESET}{gpu_suffix}")
            if not rows:
                print(f"        {C.DIM}(no containers){C.RESET}")

    extra = f"  /  Degraded: {C.YELLOW}{len(degraded)}{C.RESET}" if degraded else ""
    print(f"\n  Running: {C.GREEN}{len(running)}{C.RESET}{extra}  /  Stopped: {C.DIM}{len(stopped)}{C.RESET}\n")

# ──────────────────────────────────────────────────────────────────────────────
# MEMORY USAGE (VRAM + host RAM) -- --mem-usage
# ──────────────────────────────────────────────────────────────────────────────
def _bulk_container_rows(modules: list[Module]) -> dict[str, list[tuple[str, str, str, str]]]:
    """{module id: container_rows() result} for every module in `modules`,
    via a single `docker ps -a` call instead of each module's own
    container_rows() running its own `docker ps` (confirmed live: that's
    what made --mem-usage take several seconds on a host with ~30 modules
    -- one Docker-daemon round trip per module instead of one total).

    Groups every container by its own
    com.docker.compose.project.config_files label -- the exact same label
    container_rows() filters on per module -- instead of filtering
    per-module on the Docker side.
    """
    fmt = ('{{.Label "com.docker.compose.project.config_files"}}\t'
           '{{.Label "com.docker.compose.service"}}\t{{.State}}\t{{.Status}}\t{{.ID}}')
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", fmt],
            check=False, capture_output=True, text=True,
        )
        lines = result.stdout.splitlines()
    except OSError:
        lines = []
    by_compose_file: dict[str, list[tuple[str, str, str, str]]] = {}
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        config_files, service, state, status_text, cid = parts
        by_compose_file.setdefault(config_files, []).append((service, state, status_text, cid))
    # No per-module "does its compose file even exist" short-circuit here
    # (unlike container_rows()) -- a module with no matching containers in
    # the bulk listing above already resolves to [] via .get(), and a
    # module with no compose file simply never has anything to match.
    return {m.id: by_compose_file.get(str(m.dir / "docker-compose.yml"), []) for m in modules}

def _gpu_module_usage(
    modules: list[Module],
    devices: list[dict] | None = None,
    rows_by_id: dict[str, list[tuple[str, str, str, str]]] | None = None,
    per_container: dict | None = None,
) -> list[dict]:
    """[{"module": m, "containers": [(service, state, status_text, mib|None, per_device), ...],
    "module_mib": int}, ...] for every GPU module in `modules`.
    `per_device` is {gpu index string: mib} for however many physical GPUs
    that container's processes are actually using right now -- {} if it
    isn't using any GPU memory at all. Pass `devices` (from _gpu_devices())
    to avoid a second nvidia-smi call when the caller already has it,
    `rows_by_id` ({module id: container_rows() result}) to avoid a second
    `docker ps` subprocess per module when the caller already fetched it
    (see _raptor_host_mem_mib()'s own docstring -- same reasoning), and
    `per_container` (from _gpu_mem_by_container_and_device()) likewise.
    Split out from print_mem_usage() so it's testable without a terminal
    or real nvidia-smi.
    """
    if devices is None:
        devices = _gpu_devices()
    uuid_to_index = {d["uuid"]: d["index"] for d in devices if d["uuid"]}
    # On a single-GPU host there's no ambiguity about which device a
    # container's memory belongs to -- skip the uuid join entirely, so a
    # host where --query-compute-apps's gpu_uuid doesn't happen to match
    # --query-gpu's uuid (confirmed to diverge on at least one real
    # GB10/DGX Spark host) still gets correctly attributed.
    only_index = devices[0]["index"] if len(devices) == 1 else None
    if per_container is None:
        per_container = _gpu_mem_by_container_and_device()
    usage = []
    for m in modules:
        if not m.gpu:
            continue
        if m.health_images:
            # Image-only module (e.g. 08's shared base image) -- no
            # containers/docker-compose.yml of its own, see print_status().
            usage.append({"module": m, "containers": None, "module_mib": 0})
            continue
        containers = []
        module_mib = 0
        rows = rows_by_id.get(m.id, []) if rows_by_id is not None else m.container_rows()
        for service, state, status_text, cid in rows:
            per_device_uuid = next((v for k, v in per_container.items() if k.startswith(cid)), None)
            if per_device_uuid:
                mem = sum(per_device_uuid.values())
                if only_index is not None:
                    per_device = {only_index: mem}
                else:
                    per_device = {uuid_to_index.get(u, u): v for u, v in per_device_uuid.items()}
            else:
                per_device, mem = {}, None
            containers.append((service, state, status_text, mem, per_device))
            if mem:
                module_mib += mem
        usage.append({"module": m, "containers": containers, "module_mib": module_mib})
    return usage

def print_mem_usage(modules: list[Module] | None = None):
    """VRAM usage across every GPU module, plus host RAM (CPU memory) across
    every Raptor module, GPU or not. --status only shows per-container VRAM
    when narrowed to a single module via -m (otherwise it would be a wall of
    text for the full registry) -- that makes it tedious to eyeball
    total/aggregate memory usage across the whole stack, which is exactly
    what this is for. The two totals are printed first, since they're
    usually the numbers someone reaches for this flag to get; the VRAM
    device table and per-GPU-module breakdown that follow are VRAM-only
    (host RAM has no further per-module breakdown here).

    Only containers *actually using* GPU memory right now are listed in the
    VRAM breakdown -- most GPU modules also run plain orchestrator/cleanup/
    init containers that never touch the GPU at all, and listing every one
    of those next to the handful that matter just buries the number
    someone's here for.
    """
    target = modules or MODULES
    gpu_modules = [m for m in target if m.gpu]
    sep = f"  {C.DIM}{'─'*72}{C.RESET}"

    print(f"\n  {C.BOLD}{C.BLUE}Raptor {_RAPTOR_VERSION or '?'} — Memory Usage{C.RESET}")
    print()
    print(sep)

    # The 4 data sources below (docker ps, 2x nvidia-smi, docker stats) are
    # independent of each other -- fire them off concurrently instead of
    # waiting on each subprocess's own Docker/nvidia-smi daemon round trip
    # one after another. Confirmed live this is still the dominant cost
    # even after collapsing docker ps to one call total (see
    # _bulk_container_rows()): the wall-clock floor is now ~max(these 4
    # calls) instead of their sum.
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows_future = pool.submit(_bulk_container_rows, [m for m in target if not m.health_images])
        stats_future = pool.submit(_container_mem_mib)
        devices_future = pool.submit(_gpu_devices) if gpu_modules else None
        per_container_future = pool.submit(_gpu_mem_by_container_and_device) if gpu_modules else None

        rows_by_id = rows_future.result()
        stats_mem = stats_future.result()
        devices = devices_future.result() if devices_future else []
        per_container = per_container_future.result() if per_container_future else {}

    usage = _gpu_module_usage(gpu_modules, devices, rows_by_id, per_container) if gpu_modules else []
    grand_total = sum(row["module_mib"] for row in usage)

    device_raptor_mib = {d["index"]: 0 for d in devices}
    for row in usage:
        for _service, _state, _status_text, _mem, per_device in (row["containers"] or []):
            for idx, v in per_device.items():
                device_raptor_mib[idx] = device_raptor_mib.get(idx, 0) + v

    # VRAM and host/CPU RAM are reported independently -- on a host with a
    # real discrete GPU they're two separate physical pools, so one isn't a
    # subset of the other and neither should be gated on the other existing.
    if gpu_modules:
        print(f"  Total VRAM used by Raptor containers: {C.BOLD}{C.MAGENTA}{grand_total} MiB{C.RESET}")
        used_values = [d["used_mib"] for d in devices]
        if devices and all(v is not None for v in used_values):
            other = sum(used_values) - grand_total
            if other > 0:
                print(f"  {C.DIM}({other} MiB used on the GPU(s) by something other than a tracked Raptor container){C.RESET}")
    else:
        print(f"  {C.DIM}No GPU modules in the given selection.{C.RESET}")

    host_mib = _raptor_host_mem_mib(target, stats_mem=stats_mem, rows_by_id=rows_by_id)
    print(f"  Total host RAM (CPU memory) used by Raptor containers: {C.BOLD}{C.MAGENTA}{host_mib} MiB{C.RESET}")
    if any(d.get("mem_is_host_approx") for d in devices):
        # Unified-memory SoC: the VRAM figure above is itself only an
        # approximation read from this exact same /proc/meminfo-backed pool
        # (see _gpu_devices()), not a second, independent figure -- say so,
        # or the two numbers above read as additive when they're the same
        # physical bytes counted two different ways.
        print(f"  {C.DIM}(unified memory: VRAM and host RAM are the same pool){C.RESET}")
    print()
    print(sep)

    if not gpu_modules:
        return

    # Column widths are computed from the actual values about to be printed
    # (like Go's tabwriter, the way `docker stats` lays out its own table)
    # instead of a fixed guess -- a fixed width either wastes space on short
    # values or clips a long one.
    if devices:
        def _used_over_total(d: dict) -> str:
            if d["used_mib"] is None or d["total_mib"] is None:
                return "N/A"
            return f"{d['used_mib']}/{d['total_mib']} MiB"

        dev_rows = [(d["index"], d["name"], _used_over_total(d),
                     f"{device_raptor_mib.get(d['index'], 0)} MiB") for d in devices]
        idx_w = max(len("GPU"), max(len(r[0]) for r in dev_rows))
        name_w = max(len("Name"), max(len(r[1]) for r in dev_rows))
        ut_w = max(len("Used/Total"), max(len(r[2]) for r in dev_rows))
        raptor_w = max(len("Raptor"), max(len(r[3]) for r in dev_rows))

        print(f"  {C.BOLD}{C.DIM}{'GPU':<{idx_w}}  {'Name':<{name_w}}  {'Used/Total':>{ut_w}}  {'Raptor':>{raptor_w}}{C.RESET}")
        for d, (idx, name, used_total, raptor) in zip(devices, dev_rows):
            print(f"  {C.CYAN}{idx:<{idx_w}}{C.RESET}  {name:<{name_w}}  "
                  f"{used_total:>{ut_w}}  {C.MAGENTA}{raptor:>{raptor_w}}{C.RESET}")
            if d.get("mem_is_host_approx"):
                print(f"  {C.DIM}(unified memory: Used/Total above is host RAM){C.RESET}")
    else:
        print(f"  {C.YELLOW}nvidia-smi not available — showing module registry only, no live VRAM numbers.{C.RESET}")
    print()
    print(sep)

    mod_id_w = max([len("ID")] + [len(row["module"].id) for row in usage])
    # One shared Name column and one shared VRAM column across BOTH a
    # module's own total row and its container rows -- that's the actual
    # complaint a flat per-group width calc didn't fix: a module's total
    # (e.g. "4722 MiB") and its own containers' individual values need to
    # land in the very same column as each other, not just be internally
    # consistent within "all module rows" and "all container rows" as two
    # separate groups. No tree-branch glyphs here (plain "* " marker
    # instead) -- simpler indentation, not a hierarchy diagram.
    all_active = [c for row in usage for c in (row["containers"] or []) if c[3]]
    name_w = max(
        [len("Module")] + [len(row["module"].name) for row in usage]
        + [2 + len(c[0]) for c in all_active]  # "* " + service name
    )
    mem_w = max(
        [len("VRAM")] + [len(f"{row['module_mib']} MiB") for row in usage if row["containers"]]
        + [len(f"{c[3]} MiB") for c in all_active]
    )
    svc_w = name_w - 2  # reserve "* "/"x " for the marker + space

    print(f"  {'ID':>{mod_id_w}}  {'Module':<{name_w}}  {'VRAM':>{mem_w}}")
    print(sep)
    for row in usage:
        m = row["module"]
        id_str = f"{C.CYAN}{m.id:>{mod_id_w}}{C.RESET}"
        if row["containers"] is None:
            print(f"  {id_str}  {m.name:<{name_w}}  {C.DIM}(shared image — no containers of its own){C.RESET}")
            print()
            continue
        if not row["containers"]:
            print(f"  {id_str}  {m.name:<{name_w}}  {C.DIM}(not running){C.RESET}")
            print()
            continue
        total_str = f"{row['module_mib']} MiB"
        total_fmt = f"{C.MAGENTA}{total_str:>{mem_w}}{C.RESET}" if row["module_mib"] else f"{C.DIM}{total_str:>{mem_w}}{C.RESET}"
        print(f"  {id_str}  {m.name:<{name_w}}  {total_fmt}")
        active = [c for c in row["containers"] if c[3]]
        for service, state, status_text, mem, per_device in active:
            ok = state == "running" or (state == "exited" and "Exited (0)" in status_text)
            marker = f"{C.GREEN}*{C.RESET}" if ok else f"{C.RED}x{C.RESET}"
            # A device index is a short numeric string ("0", "1", ...) -- prefix it
            # with "GPU". A raw uuid fallback (uuid join failed, see
            # _gpu_module_usage()) already reads as a GPU identifier on its own;
            # prefixing it too would print "GPUGPU-...".
            gpu_label = "/".join(g if not g.isdigit() else f"GPU{g}" for g in sorted(per_device)) if per_device else "GPU?"
            mem_str = f"{mem} MiB"
            name_field = f"{service:<{svc_w}}"
            print(f"  {' ' * mod_id_w}  {marker} {name_field}  "
                  f"{C.MAGENTA}{mem_str:>{mem_w}}{C.RESET}  {C.MAGENTA}[{gpu_label}]{C.RESET}")
        hidden = len(row["containers"]) - len(active)
        if active and hidden:
            print(f"  {' ' * mod_id_w}    {C.DIM}(+{hidden} idle){C.RESET}")
        # A blank line between each module's block -- confirmed live that
        # packing every module directly against the next, with no gap,
        # makes it hard to tell at a glance where one module's own rows
        # end and the next module's begin, especially once a module has
        # several container lines of its own.
        print()

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY BANNER
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(results: dict[str, bool], log: Logger, elapsed: float):
    log.header("Build Summary")
    passed = [mid for mid, ok in results.items() if ok]
    failed = [mid for mid, ok in results.items() if not ok]

    for mid, ok in results.items():
        m = MODULE_BY_ID[mid]
        status = f"{C.GREEN}OK{C.RESET}" if ok else f"{C.RED}FAILED{C.RESET}"
        log.info(f"  [{mid}] {m.name:<36} {status}")

    log.info("")
    log.info(f"  Modules started : {C.GREEN}{len(passed)}{C.RESET}")
    if failed:
        log.info(f"  Modules failed  : {C.RED}{len(failed)}{C.RESET}  →  {', '.join(failed)}")
    log.info(f"  Elapsed         : {elapsed:.1f}s")
    log.info("")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Raptor {_RAPTOR_VERSION or '?'} platform build script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-a", "--all",
        action="store_true",
        default=False,
        help="Build all modules in dependency order (default when no flag given)",
    )
    group.add_argument(
        "-m", "--modules",
        metavar="ID[,ID...]",
        help="Comma-separated module IDs, e.g. 02  or  02,04,20",
    )
    parser.add_argument(
        "-e", "--exclude",
        metavar="ID[,ID...]",
        help="Comma-separated module IDs to skip (all other modules will run)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List modules and exit (combine with -m to filter)",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop modules (docker compose down), keep volumes",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Stop modules and remove volumes (docker compose down -v); for module 01 also cleans NFS directories",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Stop then start modules",
    )
    parser.add_argument(
        "-l", "--log",
        metavar="FILE",
        help="Save output to FILE (still prints to terminal)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip GPU modules — start only CPU-only modules",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Rebuild Docker images before starting (passes --build to docker compose up)",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Rebuild Docker images without starting containers (docker compose build, no up)",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip Docker network creation check",
    )
    parser.add_argument(
        "--env-file",
        metavar="FILE",
        default=None,
        help="Unified .env file to pass to all docker compose calls (default: auto-detect modules/.env)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show running/stopped status of all modules and exit",
    )
    parser.add_argument(
        "--mem-usage",
        action="store_true",
        help="Show VRAM usage per GPU module/physical GPU, plus host RAM used by all Raptor "
             "containers, then exit (combine with -m to filter)",
    )
    parser.add_argument(
        "--logs",
        nargs="?",
        const=True,
        default=False,
        metavar="SERVICE",
        help="Follow docker compose logs for the specified module(s) (requires -m). "
             "Optionally pass a service name (as in docker-compose.yml, e.g. redis1) "
             "to follow just that container within the module(s).",
    )
    parser.add_argument(
        "--check-ports",
        action="store_true",
        help="Check all HOST PORT SETTINGS in the env file for conflicts and suggest alternatives",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check the env file for unfilled <placeholder> values (e.g. <your_postgres_password>), "
             "double-scheme URLs (e.g. http://https://...), and a half-set-up TLS entry",
    )
    parser.add_argument(
        "--init-env",
        action="store_true",
        help="Bootstrap every missing .env from .env.example (root + each module), then "
             "interactively fill unfilled <placeholder> values (suggests generated secrets / this host's IP)",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore module(s) from their latest backup (or --restore-file), then start them. "
             "-m <id[,id...]> restores just those; omit -m to restore every module with backup/restore configured "
             "(same convention as --delete/--stop)",
    )
    parser.add_argument(
        "--restore-key",
        metavar="PATH",
        help="Age private key file to decrypt the backup (required if it's age-encrypted)",
    )
    parser.add_argument(
        "--restore-file",
        metavar="PATH",
        help="Restore from this specific backup archive instead of auto-picking the newest -- only valid together with -m <single id>, since it names one module's one backup file",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Trigger module(s)' backup now. -m <id[,id...]> backs up just those; omit -m to back up "
             "every module with a backup mechanism configured (same convention as --restore). "
             "Any of 01/03/24 in the target set holds a write-freeze window (module 13's maintenance "
             "mode) around all of them, so 03's/24's/01's-slice-of-04's backups land at the same "
             "moment; 02/06 alone need no freeze. No container is ever stopped.",
    )
    return parser.parse_args()


def resolve_modules(ids_str: str) -> list[Module]:
    raw = [x.strip() for x in ids_str.split(",") if x.strip()]
    modules: list[Module] = []
    errors: list[str] = []
    for raw_id in raw:
        norm = raw_id.zfill(2)
        m = MODULE_BY_ID.get(norm)
        if m is None:
            errors.append(raw_id)
        elif m not in modules:
            modules.append(m)
    if errors:
        print(f"{C.RED}Unknown module IDs: {', '.join(errors)}{C.RESET}")
        print("Run  python build.py --list  to see valid IDs.")
        sys.exit(1)
    ordered = [m for m in MODULES if m in modules]
    return ordered


def main():
    global UNIFIED_ENV
    args = parse_args()

    if args.build and args.build_only:
        print(f"{C.RED}--build and --build-only are mutually exclusive{C.RESET}")
        sys.exit(1)

    if args.env_file:
        p = Path(args.env_file)
        if not p.is_absolute():
            p = SCRIPT_DIR / p
        if not p.exists():
            print(f"{C.RED}--env-file: {p} not found{C.RESET}")
            sys.exit(1)
        UNIFIED_ENV = p

    _apply_internal_port_mode()
    _apply_keycloak_tls_mode()

    if args.init_env:
        init_env_cmd(Logger(None))
        sys.exit(0)

    if args.list:
        if args.modules:
            print_module_list(resolve_modules(args.modules))
        else:
            print_module_list()
        sys.exit(0)

    if args.status:
        if args.modules:
            print_status(resolve_modules(args.modules))
        else:
            print_status()
        sys.exit(0)

    if args.mem_usage:
        if args.modules:
            print_mem_usage(resolve_modules(args.modules))
        else:
            print_mem_usage()
        sys.exit(0)

    if args.check_ports:
        env_file = UNIFIED_ENV
        if env_file is None:
            # auto-detect: prefer .env, then .env.local
            for candidate in (".env", ".env.local"):
                p = SCRIPT_DIR / candidate
                if p.exists():
                    env_file = p
                    break
        if env_file is None or not env_file.exists():
            print(f"{C.RED}No env file found. Use --env-file to specify one.{C.RESET}")
            sys.exit(1)
        check_ports_cmd(env_file, Logger(None))
        sys.exit(0)

    if args.check_env:
        env_file = UNIFIED_ENV
        if env_file is None:
            # auto-detect: prefer .env, then .env.local
            for candidate in (".env", ".env.local"):
                p = SCRIPT_DIR / candidate
                if p.exists():
                    env_file = p
                    break
        if env_file is None or not env_file.exists():
            print(f"{C.RED}No env file found. Use --env-file to specify one.{C.RESET}")
            sys.exit(1)
        check_placeholders_cmd(env_file, Logger(None))
        sys.exit(0)

    if args.logs:
        if not args.modules:
            print(f"{C.RED}--logs requires -m <module_id>{C.RESET}")
            sys.exit(1)
        cli_service = args.logs if isinstance(args.logs, str) else None
        modules = resolve_modules(args.modules)
        for module in modules:
            compose_file = module.dir / "docker-compose.yml"
            if not compose_file.exists():
                print(f"{C.YELLOW}No docker-compose.yml for {module.name}{C.RESET}")
                continue
            env_args = _env_file_args(module.dir)
            # Services that actually have a container (running or stopped) --
            # not every service *defined* in the compose file, since profile-
            # gated ones (e.g. 02's "cluster" profile) may never have been
            # started. Picking one of those from the menu would silently show
            # nothing, looking exactly like a started-but-broken container.
            #
            # `docker compose ps` is unusable here: when COMPOSE_PROJECT_NAME
            # is shared across modules (as it is for this repo's unified-env
            # setup), it matches by project label alone and returns every
            # container in the *whole stack*, ignoring the -f file entirely
            # (confirmed live: returned all ~97 containers across all
            # modules). `com.docker.compose.project.config_files` is set per
            # container to the exact compose file that created it, so filter
            # on that instead -- immune to a shared project name.
            running_result = subprocess.run(
                ["docker", "ps", "-a",
                 "--filter", f"label=com.docker.compose.project.config_files={compose_file}",
                 "--format", '{{.Label "com.docker.compose.service"}}'],
                check=False, capture_output=True, text=True,
            )
            services = sorted({s for s in running_result.stdout.splitlines() if s.strip()})
            service = cli_service
            if service and service not in services:
                defined_result = subprocess.run(
                    ["docker", "compose"] + env_args + ["--profile", "*", "-f", str(compose_file), "config", "--services"],
                    cwd=str(module.dir), check=False, capture_output=True, text=True,
                )
                defined = [s for s in defined_result.stdout.splitlines() if s.strip()]
                if service not in defined:
                    print(f"{C.RED}No such service '{service}' in {module.name}. "
                          f"Defined: {', '.join(defined)}{C.RESET}")
                else:
                    print(f"{C.RED}'{service}' is defined in {module.name} but has no container "
                          f"(never started, or a profile that isn't active). "
                          f"Running: {', '.join(services) or '(none)'}{C.RESET}")
                continue
            if service is None and len(services) == 1:
                # Only one real choice -- no point prompting, and leaving
                # `service` unset here used to mean "no service positional
                # arg at all" below, which docker compose resolves against
                # the *whole* shared-COMPOSE_PROJECT_NAME stack on at least
                # one Compose/BuildKit version combination (confirmed live:
                # showed an unrelated module's container's logs, not this
                # module's, after a delay) -- not on every version, which is
                # exactly why it must never be left implicit.
                service = services[0]
            elif service is None and len(services) > 1 and sys.stdin.isatty():
                print(f"{C.BOLD}{module.name}{C.RESET} has {len(services)} containers:")
                print(f"  0) {C.BOLD}all{C.RESET}")
                for i, s in enumerate(services, 1):
                    print(f"  {i}) {s}")
                try:
                    choice = input("Pick one (Enter = all): ").strip()
                except (EOFError, KeyboardInterrupt):
                    choice = ""
                if choice and choice != "0":
                    try:
                        idx = int(choice)
                        if 1 <= idx <= len(services):
                            service = services[idx - 1]
                        else:
                            print(f"{C.YELLOW}Out of range, showing all{C.RESET}")
                    except ValueError:
                        print(f"{C.YELLOW}Invalid input, showing all{C.RESET}")
            if service:
                label = f"{module.name} / {service}"
            elif len(services) > 1:
                label = f"{module.name}  (services: {', '.join(services)})"
            else:
                label = module.name
            print(f"{C.BOLD}{C.BLUE}=== Logs: {label} ==={C.RESET}")
            # --profile "*": see stop_module()'s own copy of this fix -- a
            # profile-gated service's logs aren't in scope without it either.
            cmd = ["docker", "compose"] + env_args + ["--profile", "*", "-f", str(compose_file), "logs", "-f", "--tail=100"]
            if service:
                cmd.append(service)
            elif services:
                # "show all" (len(services) > 1, picked 0/Enter): name every
                # discovered service explicitly, same reasoning as above --
                # never leave compose to resolve an implicit, unscoped set.
                cmd.extend(services)
            try:
                subprocess.run(cmd, check=False, cwd=str(module.dir))
            except KeyboardInterrupt:
                pass
        sys.exit(0)

    if args.restore:
        key_path = Path(args.restore_key) if args.restore_key else None
        restore_file = Path(args.restore_file) if args.restore_file else None
        if args.modules:
            restore_targets = resolve_modules(args.modules)
        else:
            # Same convention as --delete/--stop: omitting -m means "every module", not an error.
            if restore_file:
                print(f"{C.RED}--restore-file needs -m <single_id> -- it names one module's one backup file, not every module's{C.RESET}")
                sys.exit(1)
            restore_targets = [m for m in MODULES if m.restore is not None]
        if not restore_targets:
            print(f"{C.RED}No module has backup/restore configured yet{C.RESET}")
            sys.exit(1)
        restore_log = Logger(Path(args.log) if args.log else None)
        results = {}
        for module in restore_targets:
            results[module.id] = restore_module(module, key_path, restore_file, restore_log)
        restore_log.close()
        sys.exit(0 if all(results.values()) else 1)

    if args.backup:
        if args.modules:
            backup_targets = resolve_modules(args.modules)
        else:
            # Same convention as --delete/--stop/--restore: omitting -m means "every module".
            backup_targets = [m for m in MODULES if m.id in _BACKUP_SIDECARS or m.id == "01"]
        unconfigured = [m.id for m in backup_targets if m.id not in _BACKUP_SIDECARS and m.id != "01"]
        if unconfigured:
            print(f"{C.RED}No backup mechanism configured for module(s): {', '.join(unconfigured)}{C.RESET}")
            sys.exit(1)
        if not backup_targets:
            print(f"{C.RED}No module has a backup mechanism configured yet{C.RESET}")
            sys.exit(1)
        backup_log = Logger(Path(args.log) if args.log else None)
        ok = backup_all(backup_targets, backup_log)
        backup_log.close()
        sys.exit(0 if ok else 1)

    log_path = Path(args.log) if args.log else None
    log = Logger(log_path)

    if args.modules:
        target_modules = resolve_modules(args.modules)
        strict_dep_check = not args.build_only
    else:
        # Deprecated modules (kept for rollback, not part of the live
        # pipeline) sit this one out -- still reachable via -m <id>, which
        # bypassed this branch entirely above.
        deprecated_ids = [m.id for m in MODULES if m.deprecated]
        target_modules = [m for m in MODULES if not m.deprecated]
        if deprecated_ids:
            log.warning(f"  Skipping deprecated modules [{', '.join(deprecated_ids)}] -- use -m to target one directly")
        strict_dep_check = False

    if args.exclude:
        excluded_ids = {m.id for m in resolve_modules(args.exclude)}
        target_modules = [m for m in target_modules if m.id not in excluded_ids]
        log.warning(f"  --exclude: skipping modules [{', '.join(sorted(excluded_ids))}]")

    if args.cpu_only:
        skipped = [m.id for m in target_modules if m.gpu]
        target_modules = [m for m in target_modules if not m.gpu]
        if skipped:
            log.warning(f"  --cpu-only: skipping GPU modules [{', '.join(skipped)}]")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.header(f"Raptor {_RAPTOR_VERSION or '?'} Platform Build  —  {ts}")
    if log_path:
        log.info(f"  Log file : {log_path}")
    log.info(f"  Modules  : {', '.join(m.id for m in target_modules)}")
    log.info(f"  Script   : {SCRIPT_DIR}")

    # ── STOP / DELETE ─────────────────────────────────────────────────
    if args.stop or args.delete or args.restart:
        remove_vols = args.delete
        stop_targets = list(reversed(target_modules))  # reverse order
        action = "Deleting" if remove_vols else "Stopping"
        log.header(f"{action} {len(stop_targets)} module(s)")
        t_start = time.monotonic()
        results: dict[str, bool] = {}
        for module in stop_targets:
            results[module.id] = stop_module(module, log, remove_volumes=remove_vols)
        elapsed = time.monotonic() - t_start
        print_summary(results, log, elapsed)
        if not args.restart:
            log.close()
            sys.exit(0 if all(results.values()) else 1)
        log.header("Restarting modules...")

    # ── START ─────────────────────────────────────────────────────────
    if not args.stop and not args.delete:
        if not args.no_network:
            log.header("Pre-flight: Docker network")
            ensure_raptor_network(log)

        results = {}
        t_start = time.monotonic()
        built_set: set = set()

        for module in target_modules:
            ok = run_module(module, log, check_deps_strict=strict_dep_check, build=args.build, build_only=args.build_only, built_set=built_set)
            results[module.id] = ok

            if not ok and not strict_dep_check:
                log.error(f"\nFull build aborted after module [{module.id}] failure.")
                log.warning("Fix the issue above and re-run, or use -m to restart from a specific module.")
                break

        elapsed = time.monotonic() - t_start
        print_summary(results, log, elapsed)

    log.close()
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
