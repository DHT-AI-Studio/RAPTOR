"""Guards against the failure modes that unit tests of handlers can't see:
the image not building, or config the container is handed not matching what
Settings actually reads. All of this runs without Docker."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

MODULE_DIR = Path(__file__).resolve().parents[2]
MODULES_DIR = MODULE_DIR.parent

# modules/ is the Docker build context (docker-compose.yml: `context: ..`)
_COPY_RE = re.compile(r"^COPY\s+(?P<args>.+)$", re.MULTILINE)


def _copy_sources() -> list[str]:
    sources = []
    for match in _COPY_RE.finditer((MODULE_DIR / "Dockerfile").read_text()):
        *srcs, _dest = match.group("args").split()
        sources.extend(s for s in srcs if not s.startswith("--"))
    return sources


def test_dockerfile_copy_sources_exist_in_build_context():
    missing = [s for s in _copy_sources() if not (MODULES_DIR / s).exists()]
    assert not missing, f"Dockerfile COPYs paths absent from the build context: {missing}"


def test_dockerfile_copies_every_top_level_runtime_package():
    sources = " ".join(_copy_sources())
    for needed in ("36-grpc-gateway/server.py", "36-grpc-gateway/interceptors/", "36-grpc-gateway/services/"):
        assert needed in sources


def test_compose_build_points_at_this_modules_dockerfile():
    compose = yaml.safe_load((MODULE_DIR / "docker-compose.yml").read_text())
    build = compose["services"]["grpc-gateway"]["build"]
    assert (MODULE_DIR / build["context"] / build["dockerfile"]).resolve() == (MODULE_DIR / "Dockerfile").resolve()


def test_compose_names_follow_raptor_version_convention():
    # build.py's health check looks for `<name>-${RAPTOR_VERSION}`.
    svc = yaml.safe_load((MODULE_DIR / "docker-compose.yml").read_text())["services"]["grpc-gateway"]
    assert svc["container_name"].startswith("raptor-grpc-gateway-${RAPTOR_VERSION")
    assert "${RAPTOR_VERSION" in svc["image"]


def test_env_example_only_sets_variables_settings_reads():
    from config import Settings

    prefix = Settings.model_config["env_prefix"]
    fields = {prefix + name for name in Settings.model_fields}
    declared = {
        line.split("=", 1)[0]
        for line in (MODULE_DIR / ".env.example").read_text().splitlines()
        if line and not line.startswith("#") and "=" in line
    }
    assert declared, ".env.example sets nothing"
    assert declared <= {f.upper() for f in fields}, f"unknown to Settings: {declared - {f.upper() for f in fields}}"


def test_root_env_example_wiring_targets_real_settings_fields():
    from config import Settings

    prefix = Settings.model_config["env_prefix"]
    fields = {(prefix + name).upper() for name in Settings.model_fields}
    wired = {
        line.split("=", 1)[0]
        for line in (MODULES_DIR / ".env.example").read_text().splitlines()
        if line.startswith(prefix.upper())
    }
    assert wired, "root .env.example has no GRPC_GATEWAY_* lines"
    assert wired <= fields, f"root .env.example wires unknown settings: {wired - fields}"


def test_default_downstream_urls_use_container_ports_not_host_ports():
    # Containers on the raptor network reach each other on the *container*
    # port; module 25 publishes 8025 on the host but listens on 8000.
    from config import Settings

    s = Settings(_env_file=None)
    assert s.personal_db_url.endswith(":8000")
    assert s.asset_management_url.endswith(":8000")


def test_default_ports_match_compose_mapping():
    from config import Settings

    s = Settings(_env_file=None)
    compose = yaml.safe_load((MODULE_DIR / "docker-compose.yml").read_text())
    published = "".join(compose["services"]["grpc-gateway"]["ports"])
    assert f":{s.grpc_port}" in published and f":{s.metrics_port}" in published
