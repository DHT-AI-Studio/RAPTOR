"""M07-FIX-04/06: version / alias resolution and registry error classification."""
from types import SimpleNamespace

import pytest
from src.inference.exceptions import (
    ConflictingParametersError,
    InvalidParameterError,
    ModelAliasNotFoundError,
    ModelNotFoundError,
    ModelVersionNotFoundError,
    RegistryUnavailableError,
)
from src.inference.registry import resolve_model_version


class _MlflowErr(Exception):
    def __init__(self, msg, error_code):
        super().__init__(msg)
        self.error_code = error_code


class FakeClient:
    """v1 production (alias), v2 staging, v3 not promoted."""

    def __init__(self):
        self.models = {
            "m": {
                "aliases": {"production": "1"},
                "versions": {
                    "1": {"stage": "production", "physical_path": "p1"},
                    "2": {"stage": "staging", "physical_path": "p2"},
                    "3": {"stage": "none", "physical_path": "p3"},
                },
            }
        }
        self.down = False
        self.calls = []

    def _check(self, name):
        self.calls.append(name)
        if self.down:
            raise _MlflowErr("API request to http://mlflow failed: Connection refused", "INTERNAL_ERROR")
        if ":" in name or "/" in name:  # MLflow 3.16 validates names on lookup
            raise _MlflowErr(f"Invalid model name '{name}'. Names cannot contain '/' or ':'.", "INVALID_PARAMETER_VALUE")
        if name not in self.models:
            raise _MlflowErr(f"Registered Model with name={name} not found", "RESOURCE_DOES_NOT_EXIST")

    def get_registered_model(self, name):
        self._check(name)
        return SimpleNamespace(name=name, aliases=dict(self.models[name]["aliases"]))

    def search_model_versions(self, filter_string):
        name = filter_string.split("'")[1]
        self._check(name)
        return [SimpleNamespace(version=v) for v in self.models[name]["versions"]]

    def get_model_version(self, name, version):
        self._check(name)
        tags = self.models[name]["versions"].get(str(version))
        if tags is None:
            raise _MlflowErr(f"Model Version (name={name}, version={version}) not found", "RESOURCE_DOES_NOT_EXIST")
        aliases = [a for a, v in self.models[name]["aliases"].items() if v == str(version)]
        return SimpleNamespace(version=str(version), tags=dict(tags), aliases=aliases)


@pytest.fixture
def client():
    return FakeClient()


def test_default_keeps_highest_version_policy(client):
    # promoted version is 1, but without a selector the legacy "latest number" policy stays
    assert resolve_model_version(client, "m")["version"] == "3"


def test_explicit_version(client):
    info = resolve_model_version(client, "m", model_version="2")
    assert info["version"] == "2" and info["tags"]["stage"] == "staging"


def test_alias_resolves_to_fixed_version_and_follows_moves(client):
    info = resolve_model_version(client, "m", model_alias="production")
    assert info["version"] == "1" and info["aliases"] == ["production"]
    client.models["m"]["aliases"]["production"] = "2"  # promote v2
    assert resolve_model_version(client, "m", model_alias="production")["version"] == "2"


def test_missing_alias_does_not_fall_back(client):
    # only production creates an alias today -- a staging alias must not be assumed
    with pytest.raises(ModelAliasNotFoundError) as ei:
        resolve_model_version(client, "m", model_alias="staging")
    assert ei.value.code == "model_alias_not_found" and ei.value.details["model_alias"] == "staging"


def test_missing_version_does_not_fall_back(client):
    with pytest.raises(ModelVersionNotFoundError) as ei:
        resolve_model_version(client, "m", model_version="9")
    assert ei.value.code == "model_version_not_found"


def test_missing_model(client):
    with pytest.raises(ModelNotFoundError) as ei:
        resolve_model_version(client, "nope")
    assert type(ei.value) is ModelNotFoundError and ei.value.code == "model_not_found"


def test_name_rejected_by_newer_mlflow_is_model_not_found(client):
    # e.g. a raw Ollama tag used with engine="ollama" must still reach the fallback
    with pytest.raises(ModelNotFoundError) as ei:
        resolve_model_version(client, "gemma4:12b")
    assert type(ei.value) is ModelNotFoundError


def test_registry_down_is_not_model_not_found(client):
    client.down = True
    with pytest.raises(RegistryUnavailableError) as ei:
        resolve_model_version(client, "m")
    assert ei.value.code == "registry_unavailable" and ei.value.http_status == 503
    assert not isinstance(ei.value, ModelNotFoundError)


def test_selector_validation(client):
    with pytest.raises(ConflictingParametersError):
        resolve_model_version(client, "m", model_version="1", model_alias="production")
    with pytest.raises(InvalidParameterError):
        resolve_model_version(client, "m", model_version="v1")
    assert client.calls == []  # rejected before touching the registry
