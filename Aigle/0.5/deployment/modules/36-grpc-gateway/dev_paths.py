"""Import-path setup for running gateway code outside Docker (tests, scripts/demo_local.py).

In the image, the Dockerfile copies module 13's shared service classes and
vendor/app/services/video_search_service.py into one `app/` package. Locally
that package is module 13's real `app/` (a sibling directory), with vendor/'s
file added to `app.services`'s search path so `app.services.video_search_service`
resolves the same way it does in the image.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
MODULES_DIR = MODULE_DIR.parent
GENERATED_DIR = MODULE_DIR / "generated"


def setup() -> None:
    for p in (MODULE_DIR, GENERATED_DIR, MODULES_DIR / "13-api-services"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import app.services as app_services  # module 13's package

    vendored = str(MODULE_DIR / "vendor" / "app" / "services")
    if vendored not in app_services.__path__:
        app_services.__path__.append(vendored)

    # generated/ is gitignored; CI's unit-test job only installs requirements.txt,
    # so produce the stubs on demand.
    if not (GENERATED_DIR / "raptor" / "v1" / "raptor_pb2.py").exists():
        subprocess.run(["bash", str(MODULE_DIR / "scripts" / "gen_proto.sh")], check=True)
