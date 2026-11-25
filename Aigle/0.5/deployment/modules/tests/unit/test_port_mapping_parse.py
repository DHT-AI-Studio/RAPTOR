"""CF-3 (VIE01-233): internal services publish "${INTERNAL_BIND_IP:-127.0.0.1}:${PORT_X}:N".
build.py's PORT_* comment check and scripts/check_catalog.py's host-port parser
must still read those three-part mappings (they only understood "host:container")."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_MODULES = Path(__file__).resolve().parents[2]
_REPO = _MODULES.parents[1]


@pytest.fixture
def builder(monkeypatch, tmp_path):
    path = tmp_path / "build.py"
    path.write_text((_MODULES / "build.py").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text("RAPTOR_VERSION=unit-test\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("raptor_test_builder_ports", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def catalog():
    pytest.importorskip("yaml")
    spec = importlib.util.spec_from_file_location("raptor_test_check_catalog", _REPO / "scripts" / "check_catalog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose(tmp_path, *entries):
    body = "services:\n  svc:\n    ports:\n" + "".join(f'      - "{e}"\n' for e in entries)
    body += "    networks:\n      raptor:\n"
    path = tmp_path / "docker-compose.yml"
    path.write_text(body, encoding="utf-8")
    return path


def test_build_reads_plain_host_container_mapping(builder, tmp_path):
    compose = _compose(tmp_path, "${PORT_GUARDRAIL_SERVICE:-8023}:8026")
    assert builder._container_ports_from_compose(compose) == {"PORT_GUARDRAIL_SERVICE": "8026"}


def test_build_drops_a_bind_address_variable(builder, tmp_path):
    compose = _compose(
        tmp_path,
        "${INTERNAL_BIND_IP:-127.0.0.1}:${PORT_GUARDRAIL_SERVICE:-8023}:8026",
        "${INTERNAL_BIND_IP:-127.0.0.1}:${PORT_MEMORY_SERVICE}:8026",
    )
    assert builder._container_ports_from_compose(compose) == {
        "PORT_GUARDRAIL_SERVICE": "8026",
        "PORT_MEMORY_SERVICE": "8026",
    }


def test_build_drops_a_literal_bind_address(builder, tmp_path):
    compose = _compose(tmp_path, "127.0.0.1:${PORT_KEYCLOAK:-8080}:8080")
    assert builder._container_ports_from_compose(compose) == {"PORT_KEYCLOAK": "8080"}


def test_build_real_internal_modules_keep_their_port_comments_checkable(builder):
    """The four CF-3 internal modules must stay visible to verify_container_port_comments."""
    expected = {
        "23-guardrail-service": ("PORT_GUARDRAIL_SERVICE", "8026"),
        "25-personal-db-service": ("PORT_PERSONAL_DB", "8000"),
        "26-memory-service": ("PORT_MEMORY_SERVICE", "8026"),
        "33-privacy-compliance-service": ("PORT_PRIVACY_SERVICE", "8033"),
    }
    for module, (var, port) in expected.items():
        found = builder._container_ports_from_compose(_MODULES / module / "docker-compose.yml")
        assert found.get(var) == port, module


@pytest.mark.parametrize(
    ("entry", "host_port"),
    [
        ("8023:8026", 8023),
        ("${PORT_X:-8023}:8026", 8023),
        ("127.0.0.1:8080:8080", 8080),
        ("${INTERNAL_BIND_IP:-127.0.0.1}:${PORT_GUARDRAIL_SERVICE:-8023}:8026", 8023),
        ("${INTERNAL_BIND_IP:-127.0.0.1}:${PORT_MEMORY_SERVICE}:8026", None),  # no default, as before
        ("addr=nfs,nfsvers=4,port=2049", None),
    ],
)
def test_catalog_parse_host_port(catalog, entry, host_port):
    assert catalog.parse_host_port(entry) == host_port


# ── EXPOSE_INTERNAL_PORTS: developer switch back to the pre-CF-3 binding ────

def _set_flag(builder, value):
    env = Path(builder.SCRIPT_DIR) / ".env"
    line = "" if value is None else f"EXPOSE_INTERNAL_PORTS={value}  # comment\n"
    env.write_text(f"RAPTOR_VERSION=unit-test\n{line}INTERNAL_BIND_IP=127.0.0.1\n", encoding="utf-8")


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("EXPOSE_INTERNAL_PORTS", raising=False)
    monkeypatch.delenv("INTERNAL_BIND_IP", raising=False)


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes"])
def test_expose_internal_ports_true_binds_all_interfaces(builder, clean_env, value):
    _set_flag(builder, value)
    assert builder._apply_internal_port_mode() is True
    assert os.environ["INTERNAL_BIND_IP"] == "0.0.0.0"


@pytest.mark.parametrize("value", [None, "false", "0", "no", "maybe"])
def test_expose_internal_ports_false_unset_or_garbage_keeps_cf3_binding(builder, clean_env, value, capsys):
    _set_flag(builder, value)
    assert builder._apply_internal_port_mode() is False
    assert "INTERNAL_BIND_IP" not in os.environ
    assert ("not a boolean" in capsys.readouterr().out) == (value == "maybe")


def test_expose_internal_ports_shell_env_wins_over_env_file(builder, clean_env, monkeypatch):
    _set_flag(builder, "false")
    monkeypatch.setenv("EXPOSE_INTERNAL_PORTS", "true")
    assert builder._apply_internal_port_mode() is True
    assert os.environ["INTERNAL_BIND_IP"] == "0.0.0.0"


def test_env_example_defaults_to_cf3_mode():
    text = (_MODULES / ".env.example").read_text(encoding="utf-8")
    assert "\nEXPOSE_INTERNAL_PORTS=false\n" in text
    assert "\nINTERNAL_BIND_IP=127.0.0.1\n" in text
