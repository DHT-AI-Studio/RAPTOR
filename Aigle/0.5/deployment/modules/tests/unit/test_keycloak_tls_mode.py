"""06-authentication's TLS entry (CF-3a): KEYCLOAK_TLS_ENABLED=true activates its "tls" Compose profile.
Compose interpolation has no conditionals, so build.py translates the boolean into COMPOSE_PROFILES
itself -- same mechanism as EXPOSE_INTERNAL_PORTS/INTERNAL_BIND_IP (see test_port_mapping_parse.py)."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_MODULES = Path(__file__).resolve().parents[2]


@pytest.fixture
def builder(monkeypatch, tmp_path):
    path = tmp_path / "build.py"
    path.write_text((_MODULES / "build.py").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text("RAPTOR_VERSION=unit-test\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("raptor_test_builder_tls", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _set_env(builder, lines: dict):
    env = Path(builder.SCRIPT_DIR) / ".env"
    text = "RAPTOR_VERSION=unit-test\n" + "".join(f"{k}={v}\n" for k, v in lines.items())
    env.write_text(text, encoding="utf-8")


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("KEYCLOAK_TLS_ENABLED", raising=False)
    monkeypatch.delenv("COMPOSE_PROFILES", raising=False)


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes"])
def test_keycloak_tls_enabled_true_sets_tls_profile(builder, clean_env, value):
    _set_env(builder, {"KEYCLOAK_TLS_ENABLED": value})
    assert builder._apply_keycloak_tls_mode() is True
    assert os.environ["COMPOSE_PROFILES"] == "tls"


@pytest.mark.parametrize("value", [None, "false", "0", "no", "maybe"])
def test_keycloak_tls_enabled_false_unset_or_garbage_stays_off(builder, clean_env, value, capsys):
    _set_env(builder, {} if value is None else {"KEYCLOAK_TLS_ENABLED": value})
    assert builder._apply_keycloak_tls_mode() is False
    assert "COMPOSE_PROFILES" not in os.environ
    assert ("not a boolean" in capsys.readouterr().out) == (value == "maybe")


def test_keycloak_tls_enabled_combines_with_an_existing_compose_profile(builder, clean_env):
    """02's own "cluster" profile (already in the .env file) must survive --
    this appends "tls" to it, not replace it."""
    _set_env(builder, {"KEYCLOAK_TLS_ENABLED": "true", "COMPOSE_PROFILES": "cluster"})
    assert builder._apply_keycloak_tls_mode() is True
    assert os.environ["COMPOSE_PROFILES"] == "cluster,tls"


def test_keycloak_tls_enabled_is_idempotent_if_tls_already_present(builder, clean_env):
    _set_env(builder, {"KEYCLOAK_TLS_ENABLED": "true", "COMPOSE_PROFILES": "tls"})
    assert builder._apply_keycloak_tls_mode() is True
    assert os.environ["COMPOSE_PROFILES"] == "tls"


def test_keycloak_tls_enabled_shell_env_wins_over_env_file(builder, clean_env, monkeypatch):
    _set_env(builder, {"KEYCLOAK_TLS_ENABLED": "false"})
    monkeypatch.setenv("KEYCLOAK_TLS_ENABLED", "true")
    assert builder._apply_keycloak_tls_mode() is True
    assert os.environ["COMPOSE_PROFILES"] == "tls"


def test_env_example_defaults_to_tls_off():
    text = (_MODULES / ".env.example").read_text(encoding="utf-8")
    assert "\nKEYCLOAK_TLS_ENABLED=false\n" in text
