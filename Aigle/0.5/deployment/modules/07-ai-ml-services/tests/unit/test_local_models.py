"""Local model library (MODELS_TMP_ROOT, e.g. a NAS mount): listing and local-path registration guards."""
import os

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("lakefs")

from src.core import model_manager as mm
from src.inference.exceptions import ValidationError


def _model(path, marker="config.json"):
    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, marker), "w").close()


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "models"
    _model(root / "google_gemma-3-270m-it")                       # flat download layout
    _model(root / "google" / "gemma-4-12B")                        # NAS <org>/<model> layout
    _model(root / "google" / "gemma-4-e2b", "model.safetensors")
    (root / "google" / "notes").mkdir()                             # non-model subdir: skipped
    _model(root / "unsloth" / "gguf-model", "q4.gguf")
    (root / "legacy_partial").mkdir()                             # old behaviour: unknown dirs still listed
    (root / ".cache").mkdir()                                     # hidden: skipped
    manager = object.__new__(mm.ModelManager)
    manager.models_tmp_root = str(root)
    return manager, root


def test_scan_lists_flat_and_nested_models(library):
    manager, root = library
    names = [m["name"] for m in manager._scan_local_models(str(root))]
    assert names == ["google/gemma-4-12B", "google/gemma-4-e2b", "google_gemma-3-270m-it",
                     "legacy_partial", "unsloth/gguf-model"]
    paths = {m["name"]: m["path"] for m in manager._scan_local_models(str(root))}
    assert paths["google/gemma-4-12B"] == str(root / "google" / "gemma-4-12B")


def test_resolve_local_model_path(library):
    manager, root = library
    assert manager.resolve_local_model_path("google/gemma-4-12B") == os.path.realpath(root / "google" / "gemma-4-12B")
    with pytest.raises(ValidationError, match="escapes"):
        manager.resolve_local_model_path("../../etc")
    with pytest.raises(ValidationError, match="not found"):
        manager.resolve_local_model_path("google/missing")
    with pytest.raises(ValidationError, match="does not look like a model"):
        manager.resolve_local_model_path("google/notes")


def test_register_local_rejects_bad_input_before_touching_mlflow(library):
    manager, _ = library
    assert "only supports HF" in manager.register_local_to_mlflow("x", "google/gemma-4-12B", "text-generation",
                                                                   engine="ollama")["error"]
    assert "not found" in manager.register_local_to_mlflow("x", "nonexistent/nope", "text-generation")["error"]
    assert "positive integer" in manager.register_local_to_mlflow("x", "google/gemma-4-12B", "text-generation",
                                                                   max_output_tokens=0)["error"]
    assert "unknown task" in manager.register_local_to_mlflow("x", "google/gemma-4-12B", "nonsense")["error"]
