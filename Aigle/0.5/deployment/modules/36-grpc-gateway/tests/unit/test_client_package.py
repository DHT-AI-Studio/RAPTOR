"""Guards for the raptor-grpc-client package and the buf contract config (VIE01-239).

None of this needs buf or network: it generates the stubs the same way
scripts/build_client.sh does and checks the declared requirements against what
the generated code enforces at import time.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

MODULE_DIR = Path(__file__).resolve().parents[2]
PYPROJECT = (MODULE_DIR / "client" / "pyproject.toml").read_text()


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in text.split("."))


def _declared_minimum(package: str) -> tuple[int, ...]:
    match = re.search(rf'"{package}>=([0-9.]+)"', PYPROJECT)
    assert match, f"client/pyproject.toml must declare a minimum for {package}"
    return _version(match.group(1))


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("client_src")
    subprocess.run(
        [
            sys.executable, "-m", "grpc_tools.protoc",
            "-I", str(MODULE_DIR / "proto"),
            f"--python_out={out}", f"--grpc_python_out={out}",
            str(MODULE_DIR / "proto" / "raptor" / "v1" / "raptor.proto"),
        ],
        check=True,
    )
    return out


def test_generated_layout_matches_what_the_package_ships(generated):
    assert (generated / "raptor" / "v1" / "raptor_pb2.py").is_file()
    assert (generated / "raptor" / "v1" / "raptor_pb2_grpc.py").is_file()
    # protoc must not create raptor/__init__.py: `raptor` is a namespace package so that
    # a sibling such as raptor.audio.v1 (a separate AudioService on :8044) can coexist.
    assert not (generated / "raptor" / "__init__.py").exists()


def test_declared_protobuf_minimum_covers_the_generated_code(generated):
    # e.g. ValidateProtobufRuntimeVersion(PUBLIC, 5, 28, 1, ...): older runtimes raise on import.
    code = (generated / "raptor" / "v1" / "raptor_pb2.py").read_text()
    match = re.search(r"ValidateProtobufRuntimeVersion\(\s*[\w.]+,\s*(\d+),\s*(\d+),\s*(\d+)", code)
    assert match, "could not find the generated protobuf runtime check"
    required = tuple(int(g) for g in match.groups())
    assert _declared_minimum("protobuf") >= required, (
        f"client/pyproject.toml allows protobuf older than the {required} the generated code needs"
    )


def test_declared_grpcio_minimum_covers_the_generated_code(generated):
    code = (generated / "raptor" / "v1" / "raptor_pb2_grpc.py").read_text()
    match = re.search(r"GRPC_GENERATED_VERSION = '([0-9.]+)'", code)
    assert match, "could not find the generated grpcio version check"
    assert _declared_minimum("grpcio") >= _version(match.group(1))


def test_client_package_is_a_namespace_package_named_raptor_grpc_client():
    assert 'name = "raptor-grpc-client"' in PYPROJECT
    assert "namespaces = true" in PYPROJECT


def test_build_script_keeps_raptor_a_namespace_package():
    script = (MODULE_DIR / "scripts" / "build_client.sh").read_text()
    touched = re.findall(r'touch\s+"\$SRC_DIR/([^"\s]+)"', script)
    assert touched == ["raptor/v1/__init__.py"], touched


def test_buf_config_enforces_standard_lint_and_file_level_breaking_checks():
    cfg = yaml.safe_load((MODULE_DIR / "proto" / "buf.yaml").read_text())
    assert cfg["version"] == "v2"
    assert cfg["lint"]["use"] == ["STANDARD"]
    assert cfg["breaking"]["use"] == ["FILE"]


def test_buf_lint_exceptions_are_exactly_the_three_documented_naming_rules():
    # Growing this list silently would defeat the lint; each addition should be a reviewed decision.
    cfg = yaml.safe_load((MODULE_DIR / "proto" / "buf.yaml").read_text())
    assert sorted(cfg["lint"]["except"]) == [
        "RPC_REQUEST_RESPONSE_UNIQUE",
        "RPC_REQUEST_STANDARD_NAME",
        "RPC_RESPONSE_STANDARD_NAME",
    ]
