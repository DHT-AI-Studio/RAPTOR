"""Generate the Python stubs for proto/raptor/audio/v1/audio.proto (RA-6).

    python scripts/gen_proto.py [OUT_DIR]        # default: generated/

Writes raptor/audio/v1/audio_pb2.py and audio_pb2_grpc.py under OUT_DIR (``raptor`` and
``raptor.audio`` stay namespace packages, so the Module 36 client package can sit beside them).
Run by the Dockerfile and by tests/conftest.py; needs grpcio-tools.
"""
from __future__ import annotations

import os
import sys

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTO_DIR = os.path.join(MODULE_DIR, "proto")
PROTO = os.path.join("raptor", "audio", "v1", "audio.proto")


def generate(out_dir: str) -> str:
    from grpc_tools import protoc

    pkg = os.path.join(out_dir, "raptor", "audio", "v1")
    os.makedirs(pkg, exist_ok=True)
    open(os.path.join(pkg, "__init__.py"), "a").close()
    code = protoc.main(["grpc_tools.protoc", f"-I{PROTO_DIR}", f"--python_out={out_dir}",
                        f"--grpc_python_out={out_dir}", os.path.join(PROTO_DIR, PROTO)])
    if code != 0:
        raise RuntimeError(f"protoc failed with exit code {code}")
    return out_dir


if __name__ == "__main__":
    print(generate(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(MODULE_DIR, "generated"))))
