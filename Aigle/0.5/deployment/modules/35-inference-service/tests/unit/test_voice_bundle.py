"""NAS export requires complete snapshots and preserves relocatable config."""
import shutil
import sys

import pytest
import yaml
from scripts.prefetch_voice import _CHATTERBOX_FILES, prefetch


def _write_nas_fixture(nas, arm64: bool = False):
    whisper_file = "openai/whisper-large-v3/model.safetensors" if arm64 \
        else "Systran/faster-whisper-large-v3/model.bin"
    files = {
        "pyannote/speaker-diarization-3.1/config.yaml": "pipeline:\n  params:\n    segmentation: original\n    embedding: original\n",
        "pyannote/segmentation-3.0/pytorch_model.bin": "segmentation",
        "pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin": "embedding",
        whisper_file: "whisper",
        "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn/config.json": "{}",
        "coqui/XTTS-v2/model.pth": "xtts", "coqui/XTTS-v2/config.json": "{}",
        "coqui/XTTS-v2/vocab.json": "{}",
        **{f"ResembleAI/chatterbox/{name}": name for name in _CHATTERBOX_FILES},
    }
    for name, content in files.items():
        path = nas / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def test_nas_bundle_is_offline_and_relocatable(tmp_path, monkeypatch):
    nas, bundle = tmp_path / "nas", tmp_path / "bundle"
    _write_nas_fixture(nas)
    vad = tmp_path / "vad.bin"
    vad.write_bytes(b"vad")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # Any network API import would fail.
    prefetch(bundle, nas, vad_checkpoint=vad, smoke_load=False,
             backends=frozenset({"xtts-v2", "chatterbox-multilingual-v3"}))
    moved = tmp_path / "relocated"
    shutil.move(bundle, moved)
    params = yaml.safe_load((moved / "diarize.yaml").read_text())["pipeline"]["params"]
    for key in ("segmentation", "embedding"):
        assert (moved / params[key]).read_text() == key
    assert (moved / "whisper/model.bin").read_text() == "whisper"
    assert (moved / "xtts/model.pth").read_text() == "xtts"
    for name in _CHATTERBOX_FILES:
        assert (moved / "chatterbox" / name).read_text() == name


def test_backends_parameter_gates_which_weights_are_fetched(tmp_path, monkeypatch):
    """A build that only bakes in Chatterbox must not require (or fetch) XTTS's
    files, and vice versa -- proves `backends=` is load-bearing, not cosmetic."""
    nas, bundle = tmp_path / "nas", tmp_path / "bundle"
    _write_nas_fixture(nas)
    (nas / "coqui/XTTS-v2/model.pth").unlink()  # XTTS deliberately incomplete
    vad = tmp_path / "vad.bin"
    vad.write_bytes(b"vad")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    prefetch(bundle, nas, vad_checkpoint=vad, smoke_load=False,
             backends=frozenset({"chatterbox-multilingual-v3"}))
    assert not (bundle / "xtts").exists()
    assert (bundle / "chatterbox" / "t3_mtl23ls_v3.safetensors").read_text() == "t3_mtl23ls_v3.safetensors"


def test_arm64_bundles_the_hf_format_whisper_checkpoint_instead_of_ctranslate2(tmp_path, monkeypatch):
    """ctranslate2 (Systran/faster-whisper-large-v3's own format) has no CUDA-enabled
    linux_aarch64 wheel -- arm64 builds must get the HF transformers (safetensors)
    snapshot into root/"whisper" instead (see diarize.py's _IS_ARM64), not the CT2 one."""
    nas, bundle = tmp_path / "nas", tmp_path / "bundle"
    _write_nas_fixture(nas, arm64=True)
    vad = tmp_path / "vad.bin"
    vad.write_bytes(b"vad")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    prefetch(bundle, nas, vad_checkpoint=vad, smoke_load=False,
             backends=frozenset({"xtts-v2", "chatterbox-multilingual-v3"}), arch="arm64")
    assert (bundle / "whisper/model.safetensors").read_text() == "whisper"
    assert not (bundle / "whisper/model.bin").exists()


def test_missing_nas_models_fail_before_export(tmp_path):
    target = tmp_path / "bundle"
    with pytest.raises(FileNotFoundError, match="NAS model files missing"):
        prefetch(target, tmp_path / "empty-nas")
    assert not target.exists()
