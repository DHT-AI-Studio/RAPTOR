"""The gateway reuses module 13's service classes (see the Dockerfile) and hands them its
own `Settings` object, relying on it having the same attribute names. Nothing checked that,
and `StorageService.upload_file` reading `settings.upload_timeout` made every UploadAsset
fail with AttributeError on a real stack (the upload tests use a fake storage service).

This reads the real source of every reused class and fails when it reads a setting the
gateway's `Settings` does not define, so a new setting added to module 13 fails here first.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from config import Settings

MODULE_DIR = Path(__file__).resolve().parents[2]
MODULES_DIR = MODULE_DIR.parent


def _reused_module13_files() -> list[Path]:
    dockerfile = (MODULE_DIR / "Dockerfile").read_text()
    paths = re.findall(r"^COPY\s+(13-api-services/app/services/\S+\.py)\s", dockerfile, flags=re.MULTILINE)
    return [MODULES_DIR / p for p in paths if not p.endswith("__init__.py")]


def _settings_attributes_read(path: Path) -> set[str]:
    """Names read as `settings.X` or `self.settings.X` (or `self._settings.X`)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        if (isinstance(base, ast.Name) and base.id in {"settings", "_settings"}) or (
            isinstance(base, ast.Attribute) and base.attr in {"settings", "_settings"}
        ):
            found.add(node.attr)
    return found


def test_the_scan_finds_the_reused_files_and_some_settings():
    files = _reused_module13_files()
    assert {f.name for f in files} >= {"storage_service.py", "personal_db_service.py", "kafka_service.py"}
    assert "upload_timeout" in _settings_attributes_read(MODULES_DIR / "13-api-services/app/services/storage_service.py"), (
        "the scan no longer sees StorageService reading settings; update it"
    )


@pytest.mark.parametrize("path", _reused_module13_files(), ids=lambda p: p.name)
def test_gateway_settings_defines_every_setting_a_reused_module13_class_reads(path):
    settings = Settings(_env_file=None)
    missing = sorted(a for a in _settings_attributes_read(path) if not hasattr(settings, a))
    assert not missing, (
        f"{path.name} reads settings {missing} that the gateway's Settings (config.py) does not define; "
        "any RPC that reaches that code raises AttributeError (INTERNAL)"
    )
