#!/usr/bin/env python3
"""
Raptor 0.3 — Platform Build Script
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
import os
import shlex
import socket
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent  # deployment/modules/
UNIFIED_ENV: Optional[Path] = None  # set by --env-file CLI arg

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
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"

def strip_ansi(text: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)

# ──────────────────────────────────────────────────────────────────────────────
# LOGGER
# ──────────────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, log_path: Optional[Path] = None):
        self._log_path = log_path
        self._fh = open(log_path, "w", encoding="utf-8") if log_path else None
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

    def warn(self, msg: str):
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
    cmd: List[str]
    cwd: Optional[Path] = None
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
    except Exception:
        return False

def _image_exists(name: str) -> bool:
    try:
        subprocess.check_output(
            ["docker", "image", "inspect", name],
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
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
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────────────────────
# MODULE
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Module:
    id: str
    name: str
    description: str
    steps: List[Step]
    deps: List[str]
    health_containers: List[str] = field(default_factory=list)
    health_images: List[str] = field(default_factory=list)
    gpu: bool = False
    nfs_init_script: Optional[str] = None  # path relative to module dir, called with --clean on --delete
    delete_warn: Optional[str] = None      # warning message printed on --delete

    @property
    def dir(self) -> Path:
        return SCRIPT_DIR / self.name

    def is_healthy(self) -> bool:
        if self.health_images:
            return all(_image_exists(img) for img in self.health_images)
        if not self.health_containers:
            return True
        return any(_container_running(c) for c in self.health_containers)

# ──────────────────────────────────────────────────────────────────────────────
# STOP MODULE
# ──────────────────────────────────────────────────────────────────────────────
def stop_module(module: Module, log: Logger, remove_volumes: bool = False) -> bool:
    log.header(f"{'Delete' if remove_volumes else 'Stop'} [{module.id}]  {module.name}")
    compose_file = module.dir / "docker-compose.yml"
    if not compose_file.exists():
        log.warn(f"  No docker-compose.yml found — skipping")
        return True

    cmd = ["docker", "compose"] + _env_file_args(module.dir) + ["-f", str(compose_file), "down"]
    if remove_volumes:
        cmd.append("-v")
    log.step(f"  → docker compose down{' -v' if remove_volumes else ''}")
    result = subprocess.run(cmd, cwd=str(module.dir))
    if result.returncode != 0:
        log.error(f"Module [{module.id}] {module.name} — {'delete' if remove_volumes else 'stop'} FAILED")
        return False

    if remove_volumes and module.delete_warn:
        log.warn(f"  ⚠  {module.delete_warn}")

    if remove_volumes and module.nfs_init_script:
        script = module.dir / module.nfs_init_script
        log.step(f"  → Cleaning NFS directories ({module.nfs_init_script} --clean)")
        result = subprocess.run(["sudo", "bash", str(script), "--clean"], cwd=str(module.dir))
        if result.returncode != 0:
            log.warn(f"  NFS cleanup returned {result.returncode} — continuing")

    log.ok(f"Module [{module.id}] {module.name} — {'deleted' if remove_volumes else 'stopped'}")
    return True

# ──────────────────────────────────────────────────────────────────────────────
# PORT CHECKER
# ──────────────────────────────────────────────────────────────────────────────

def _parse_host_ports(env_file: Path) -> list[tuple[str, int]]:
    """Extract PORT variables from the HOST PORT SETTINGS section of an env file."""
    result: list[tuple[str, int]] = []
    in_section = False
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if "HOST PORT SETTINGS" in stripped:
                in_section = True
                continue
            if in_section and ("全域共用設定" in stripped or "GPU ALLOCATION SETTINGS" in stripped):
                break
            if not in_section or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, val = stripped.partition("=")
            val = val.split("#")[0].strip()   # strip inline comments
            try:
                port = int(val)
                if 1 <= port <= 65535:
                    result.append((key.strip(), port))
            except ValueError:
                pass
    return result


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
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["lsof", "-i", f":{port}", "-sTCP:LISTEN", "-n", "-P"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        for line in out.splitlines():
            if str(port) in line:
                return line.strip()
    except Exception:
        pass
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
    import re
    from collections import defaultdict

    log.header(f"Port Conflict Check  —  {env_file.name}")

    ports = _parse_host_ports(env_file)
    if not ports:
        log.warn("No HOST PORT SETTINGS section found or no port variables detected.")
        log.warn(f"Ensure {env_file} has a 'HOST PORT SETTINGS' section.")
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

def _compose_up(compose_file: str, cwd: Optional[Path] = None) -> Step:
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
MODULES: List[Module] = [

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="01",
        name="01-nfs-server",
        description="NFS Server — shared storage backing SeaweedFS volumes",
        deps=[],
        health_containers=["raptor-nfs-server"],
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
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="02",
        name="02-redis-cluster",
        description="Redis 7-node cluster + standalone instance",
        deps=[],
        health_containers=["raptor-redis1", "raptor-redis-standalone"],
        steps=[
            _compose_up("02-redis-cluster/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="03",
        name="03-database",
        description="PostgreSQL (shared) + Qdrant vector DB",
        deps=[],
        health_containers=["raptor-postgres", "raptor-qdrant"],
        steps=[
            _compose_up("03-database/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="04",
        name="04-object-storage",
        description="SeaweedFS distributed object storage + LakeFS + Asset Management API",
        deps=["01", "03"],
        health_containers=["raptor-seaweedfs-master1", "raptor-seaweedfs-s3"],
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
        health_containers=["raptor-kafka-broker1", "raptor-kafka-broker2", "raptor-kafka-broker3"],
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
        health_containers=["raptor-keycloak"],
        steps=[
            _compose_up("06-authentication/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="07",
        name="07-ai-ml-services",
        description="MLflow tracking server + AI Lifecycle API",
        deps=["01", "03", "04"],
        health_containers=["raptor-mlflow", "raptor-ai-lifecycle-api"],
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
        health_images=["raptor/media-worker:0.3"],
        gpu=True,
        steps=[
            Step(
                label="Build raptor/media-worker:0.3  (build context = deployment/modules/)",
                cmd=[
                    "docker", "build",
                    "-t", "raptor/media-worker:0.3",
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
        health_containers=["raptor-audio-recognizer"],
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
        health_containers=["raptor-image-processing"],
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
        health_containers=["raptor-video-frame-description"],
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
        health_containers=["raptor-document-analysis"],
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
        health_containers=["raptor-api-gateway"],
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
        health_containers=["raptor-prometheus", "raptor-grafana"],
        steps=[
            _compose_up("14-monitoring/docker-compose.yml"),
        ],
    ),

    # ──────────────────────────────────────────────────────────────────
    Module(
        id="15",
        name="15-chat-service",
        description="Chat service — LangGraph RAG pipeline with hybrid search and Redis memory",
        deps=["02", "17"],
        health_containers=["raptor-chat-service"],
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
        health_containers=["raptor-training-service"],
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
        health_containers=["raptor-opensearch-node1", "raptor-hybridsearch-api"],
        gpu=True,
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
        deps=["03", "04", "13", "17"],
        health_containers=["raptor-query-orchestrator"],
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
        health_containers=["raptor-neo4j"],
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
        health_containers=["raptor-graph-service"],
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
        health_containers=["raptor-agent-protocol"],
        steps=[
            _compose_up("21-agent-protocol/docker-compose.yml"),
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
    cmd = (["sudo"] + step.cmd) if step.sudo else step.cmd
    log.step(f"  → {step.label}")

    env = os.environ.copy()
    env.update(step.env_extra)

    try:
        if step.interactive:
            # Inherit terminal stdin/stdout/stderr so the script can prompt the user
            proc = subprocess.run(
                cmd if not step.shell else " ".join(shlex.quote(c) for c in cmd),
                cwd=str(step.cwd) if step.cwd else None,
                env=env,
                shell=step.shell,
            )
            rc = proc.returncode
        else:
            proc = subprocess.Popen(
                cmd if not step.shell else " ".join(shlex.quote(c) for c in cmd),
                cwd=str(step.cwd) if step.cwd else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=step.shell,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.raw("     " + line.rstrip())
            proc.wait()
            rc = proc.returncode

        if rc != 0:
            if step.optional:
                log.warn(f"     (optional step exited {rc} — continuing)")
                return True
            log.error(f"     Step failed (exit {rc})")
            return False
        return True
    except FileNotFoundError as exc:
        if step.optional:
            log.warn(f"     (optional step not found: {exc} — continuing)")
            return True
        log.error(f"     Command not found: {exc}")
        return False
    except Exception as exc:
        if step.optional:
            log.warn(f"     (optional step error: {exc} — continuing)")
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
            log.warn(f"  Unknown dependency module id '{dep_id}' — skipping check")
            continue
        if dep.is_healthy():
            log.ok(f"  Dep [{dep_id}] {dep.name} — running")
        else:
            msg = f"  Dep [{dep_id}] {dep.name} — NOT running"
            if strict:
                log.error(msg)
                all_ok = False
            else:
                log.warn(msg + " (will be started earlier in this run)")
    return all_ok

# ──────────────────────────────────────────────────────────────────────────────
# ENSURE DOCKER NETWORK
# ──────────────────────────────────────────────────────────────────────────────
def ensure_raptor_network(log: Logger):
    log.step("Checking Docker network 'raptor'...")
    if _network_exists("raptor"):
        log.ok("Network 'raptor' already exists")
        return
    log.step("Creating Docker network 'raptor' (172.30.0.0/16)...")
    result = subprocess.run(
        ["docker", "network", "create",
         "raptor", "--subnet=172.30.0.0/16"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        log.ok("Network 'raptor' created")
    else:
        log.warn(f"Could not create network: {result.stderr.strip()}")

# ──────────────────────────────────────────────────────────────────────────────
# MODULE RUNNER
# ──────────────────────────────────────────────────────────────────────────────
def run_module(
    module: Module,
    log: Logger,
    check_deps_strict: bool = False,
    build: bool = False,
    built_set: Optional[set] = None,
) -> bool:
    log.header(f"Module [{module.id}]  {module.name}")
    log.info(f"  {module.description}")

    if module.deps:
        log.step("Checking dependencies...")
        if not check_deps(module, log, strict=check_deps_strict):
            log.error(f"  Module [{module.id}] aborted — unmet dependencies")
            return False

    if not module.dir.exists():
        log.warn(f"  Module directory not found: {module.dir}  — skipping")
        return False

    success = True
    for step in module.steps:
        if build and "up" in step.cmd and "compose" in step.cmd:
            if "08" in module.deps:
                # Modules 09-12 share raptor/media-worker:0.3 built by module 08.
                # --build should rebuild that image, not re-run docker compose --build.
                if built_set is None or "08" not in built_set:
                    m08 = MODULE_BY_ID.get("08")
                    if m08:
                        log.step("  → --build: rebuilding shared image via module 08 first...")
                        for build_step in m08.steps:
                            if not run_step(build_step, log):
                                log.error("  Shared image (module 08) build FAILED")
                                return False
                    if built_set is not None:
                        built_set.add("08")
                # compose up without --build — image was just rebuilt above
            else:
                step = Step(
                    label=step.label + " --build",
                    cmd=step.cmd + ["--build"],
                    cwd=step.cwd,
                    env_extra=step.env_extra,
                    optional=step.optional,
                    sudo=step.sudo,
                    shell=step.shell,
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
        log.ok(f"Module [{module.id}] {module.name} — started")
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


def print_module_list(modules: Optional[List[Module]] = None):
    target = modules or MODULES
    cpu = [m for m in target if not m.gpu]
    gpu = [m for m in target if m.gpu]

    print(f"\n{C.BOLD}{C.BLUE}Raptor 0.3 — Module Registry{C.RESET}")

    print(f"\n{C.BOLD}{C.GREEN}── CPU-only ({len(cpu)} modules) {'─'*40}{C.RESET}")
    _print_module_rows(cpu)

    print(f"\n{C.BOLD}{C.YELLOW}── GPU required ({len(gpu)} modules) {'─'*37}{C.RESET}")
    _print_module_rows(gpu)
    print()

# ──────────────────────────────────────────────────────────────────────────────
# STATUS
# ──────────────────────────────────────────────────────────────────────────────
def print_status(modules: Optional[List[Module]] = None):
    target = modules or MODULES
    running = [m for m in target if m.is_healthy()]
    stopped = [m for m in target if not m.is_healthy()]

    print(f"\n{C.BOLD}{C.BLUE}Raptor 0.3 — Module Status{C.RESET}\n")
    print(f"  {'ID':>2}  {'Module Name':<30}  Status")
    print(f"  {C.DIM}{'─'*50}{C.RESET}")
    for m in target:
        if m.health_images:
            if m.is_healthy():
                status = f"{C.GREEN}● built{C.RESET}    {C.DIM}[image]{C.RESET}"
            else:
                status = f"{C.YELLOW}○ not built{C.RESET} {C.DIM}[image]{C.RESET}"
        elif m.is_healthy():
            status = f"{C.GREEN}● running{C.RESET}"
        else:
            status = f"{C.DIM}○ stopped{C.RESET}"
        print(f"  {C.CYAN}{m.id:>2}{C.RESET}  {m.name:<30}  {status}")

    print(f"\n  Running: {C.GREEN}{len(running)}{C.RESET}  /  Stopped: {C.DIM}{len(stopped)}{C.RESET}\n")

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
        description="Raptor 0.3 platform build script",
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
        "--logs",
        action="store_true",
        help="Follow docker compose logs for the specified module(s) (requires -m)",
    )
    parser.add_argument(
        "--check-ports",
        action="store_true",
        help="Check all HOST PORT SETTINGS in the env file for conflicts and suggest alternatives",
    )
    return parser.parse_args()


def resolve_modules(ids_str: str) -> List[Module]:
    raw = [x.strip() for x in ids_str.split(",") if x.strip()]
    modules: List[Module] = []
    errors: List[str] = []
    for raw_id in raw:
        norm = raw_id.zfill(2)
        m = MODULE_BY_ID.get(norm)
        if m is None:
            errors.append(raw_id)
        elif m not in modules:
            modules.append(m)
    if errors:
        print(f"{C.RED}Unknown module IDs: {', '.join(errors)}{C.RESET}")
        print(f"Run  python build.py --list  to see valid IDs.")
        sys.exit(1)
    ordered = [m for m in MODULES if m in modules]
    return ordered


def main():
    global UNIFIED_ENV
    args = parse_args()

    if args.env_file:
        p = Path(args.env_file)
        if not p.is_absolute():
            p = SCRIPT_DIR / p
        if not p.exists():
            print(f"{C.RED}--env-file: {p} not found{C.RESET}")
            sys.exit(1)
        UNIFIED_ENV = p

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

    if args.logs:
        if not args.modules:
            print(f"{C.RED}--logs requires -m <module_id>{C.RESET}")
            sys.exit(1)
        modules = resolve_modules(args.modules)
        for module in modules:
            compose_file = module.dir / "docker-compose.yml"
            if not compose_file.exists():
                print(f"{C.YELLOW}No docker-compose.yml for {module.name}{C.RESET}")
                continue
            print(f"{C.BOLD}{C.BLUE}=== Logs: {module.name} ==={C.RESET}")
            try:
                subprocess.run(
                    ["docker", "compose", "-f", str(compose_file), "logs", "-f", "--tail=100"],
                    cwd=str(module.dir),
                )
            except KeyboardInterrupt:
                pass
        sys.exit(0)

    log_path = Path(args.log) if args.log else None
    log = Logger(log_path)

    if args.modules:
        target_modules = resolve_modules(args.modules)
        strict_dep_check = True
    else:
        target_modules = list(MODULES)
        strict_dep_check = False

    if args.exclude:
        excluded_ids = {m.id for m in resolve_modules(args.exclude)}
        target_modules = [m for m in target_modules if m.id not in excluded_ids]
        log.warn(f"  --exclude: skipping modules [{', '.join(sorted(excluded_ids))}]")

    if args.cpu_only:
        skipped = [m.id for m in target_modules if m.gpu]
        target_modules = [m for m in target_modules if not m.gpu]
        if skipped:
            log.warn(f"  --cpu-only: skipping GPU modules [{', '.join(skipped)}]")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.header(f"Raptor 0.3 Platform Build  —  {ts}")
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
            ok = run_module(module, log, check_deps_strict=strict_dep_check, build=args.build, built_set=built_set)
            results[module.id] = ok

            if not ok and not strict_dep_check:
                log.error(f"\nFull build aborted after module [{module.id}] failure.")
                log.warn("Fix the issue above and re-run, or use -m to restart from a specific module.")
                break

        elapsed = time.monotonic() - t_start
        print_summary(results, log, elapsed)

    log.close()
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
