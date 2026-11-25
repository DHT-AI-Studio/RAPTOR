"""Module 07 test setup.

- Puts the module root on sys.path so tests can `import src.inference...`.
- Stubs `src.core.config` before anything imports it: the real one resolves
  env vars from base.yaml and mkdirs every *_root path at import time, which
  unit tests must not depend on. Engine configs are empty -> adapters use
  their built-in defaults, and nothing talks to MLflow / Ollama / a GPU.
"""
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# model_manager's module-level singleton constructs a real MlflowClient at
# import time even with a file:// tracking_uri (see below); current mlflow
# refuses the filesystem backend outright unless this is set -- confirmed
# live against mlflow 3.17.1 (requirements.txt pins no version).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

_TMP = tempfile.mkdtemp(prefix="m07-tests-")
_VALUES = {
    ("inference", "idle_reaper"): {"enabled": False},
    # model_manager builds a ModelManager at import time; point it at local
    # throwaway paths (no MLflow / lakeFS connection is made in __init__)
    ("mlflow", "tracking_uri"): f"file://{_TMP}/mlruns",
    ("storage", "models_tmp_root"): f"{_TMP}/models",
    ("storage", "models_from_lakefs_root"): f"{_TMP}/lakefs-models",
    ("lakefs", "endpoint"): "http://lakefs.invalid:8000",
    ("ollama", "api_base"): "http://ollama.invalid:11434",
}


class _StubConfig:
    def get_config(self, *keys):
        return _VALUES.get(keys)


_config_mod = types.ModuleType("src.core.config")
_config_mod.config = _StubConfig()
sys.modules.setdefault("src.core.config", _config_mod)
