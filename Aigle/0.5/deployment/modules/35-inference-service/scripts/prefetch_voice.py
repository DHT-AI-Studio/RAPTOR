"""Export a relocatable, credential-free model bundle at image build time."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import tempfile
from pathlib import Path

import yaml

# Pinned to each repo's main HEAD as of 2026-10-15 (bandit B615) -- an
# unpinned download silently follows whatever the repo owner pushes next.
_REVISIONS = {
    "pyannote/speaker-diarization-3.1": "84fd25912480287da0247647c3d2b4853cb3ee5d",
    "pyannote/segmentation-3.0": "e66f3d3b9eb0873085418a7b813d3b369bf160bb",
    "pyannote/wespeaker-voxceleb-resnet34-LM": "837717ddb9ff5507820346191109dc79c958d614",
    "Systran/faster-whisper-large-v3": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    # arm64's diarize ASR step: ctranslate2 (Systran/faster-whisper-large-v3's own format)
    # ships no CUDA-enabled linux_aarch64 wheel, so arm64 builds bundle this HF-format
    # snapshot of the same model into root/"whisper" instead (see diarize.py's _IS_ARM64).
    "openai/whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
    "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn": "99ccb2737be22b8bb50dcfcc39ad4d567fb90cfd",
    "coqui/XTTS-v2": "6c2b0d75eae4b7047358e3b6bd9325f857d43f77",
    "ResembleAI/chatterbox": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
}

# ChatterboxMultilingualTTS.from_pretrained()'s own allow_patterns -- the full repo is
# 13.9 GB (duplicate .pt/.safetensors formats, v2 *and* v3 weights); this is ~3.2 GB.
_CHATTERBOX_FILES = ["ve.pt", "t3_mtl23ls_v3.safetensors", "s3gen.pt",
                     "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]

_ALL_BACKENDS = frozenset({"xtts-v2", "chatterbox-multilingual-v3"})


def prefetch(root: Path, nas_root: Path | None = None, *,
             vad_checkpoint: Path | None = None, smoke_load: bool = True,
             backends: frozenset[str] = _ALL_BACKENDS, arch: str | None = None) -> None:
    """Export complete local NAS snapshots or download with a build-time secret.

    backends: which clone backends to actually fetch weights for -- a subset of
    {"xtts-v2", "chatterbox-multilingual-v3"}, matching Dockerfile.voice's
    CLONE_BACKENDS_BUILD build arg. Diarization/STT assets are always fetched;
    they aren't backend-selectable.

    arch: "amd64" or "arm64" (Dockerfile.voice passes its own $TARGETARCH explicitly --
    platform.machine() would report the *build host's* arch under cross-compilation, not
    the target's). Decides which format of the diarize ASR checkpoint goes into
    root/"whisper": CTranslate2 (amd64) or HF transformers (arm64) -- see diarize.py's
    _IS_ARM64. Defaults to platform.machine() for a plain local/manual run.
    """
    is_arm64 = (arch or platform.machine()) in ("arm64", "aarch64")
    whisper_repo = "openai/whisper-large-v3" if is_arm64 else "Systran/faster-whisper-large-v3"
    whisper_marker = "model.safetensors" if is_arm64 else "model.bin"
    token = None
    if nas_root is None:
        secret = Path("/run/secrets/hf_token")
        if not secret.is_file():
            raise RuntimeError("HUGGINGFACE_TOKEN build secret is required")
        token = secret.read_text().strip()
        if not token or token.startswith("<"):
            raise RuntimeError("HUGGINGFACE_TOKEN build secret is required")
    else:
        required = [
            ("pyannote/speaker-diarization-3.1", "config.yaml"),
            ("pyannote/segmentation-3.0", "pytorch_model.bin"),
            ("pyannote/wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin"),
            (whisper_repo, whisper_marker),
            ("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", "config.json"),
        ]
        if "xtts-v2" in backends:
            required += [("coqui/XTTS-v2", "model.pth"), ("coqui/XTTS-v2", "config.json"),
                         ("coqui/XTTS-v2", "vocab.json")]
        if "chatterbox-multilingual-v3" in backends:
            required += [("ResembleAI/chatterbox", name) for name in _CHATTERBOX_FILES]
        missing = [str(nas_root / repo / name) for repo, name in required
                   if not (nas_root / repo / name).is_file()]
        if missing:
            raise FileNotFoundError("NAS model files missing: " + ", ".join(missing))
        if root.resolve() == nas_root.resolve() or any(
            root.resolve().is_relative_to((nas_root / repo).resolve()) for repo, _ in required
        ):
            raise ValueError("bundle destination must be separate from source snapshots")
    def asset(repo, name):
        if nas_root is not None:
            return nas_root / repo / name
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo, name, revision=_REVISIONS[repo], token=token))
    def snapshot(repo, destination):
        if nas_root is not None:
            shutil.copytree(nas_root / repo, destination, dirs_exist_ok=True)
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo, revision=_REVISIONS[repo], local_dir=destination, token=token)
    def chatterbox_snapshot(destination):
        """Only the ~3.2 GB this service actually loads (see _CHATTERBOX_FILES),
        not snapshot()'s whole-repo copy -- the chatterbox repo is 13.9 GB."""
        destination.mkdir(parents=True, exist_ok=True)
        if nas_root is not None:
            for name in _CHATTERBOX_FILES:
                shutil.copyfile(nas_root / "ResembleAI/chatterbox" / name, destination / name)
        else:
            from huggingface_hub import snapshot_download
            snapshot_download("ResembleAI/chatterbox", revision=_REVISIONS["ResembleAI/chatterbox"],
                              local_dir=destination, token=token, allow_patterns=_CHATTERBOX_FILES)
    root.mkdir(parents=True, exist_ok=True)
    config_path = asset("pyannote/speaker-diarization-3.1", "config.yaml")
    config = yaml.safe_load(Path(config_path).read_text())
    for key, repo in [("segmentation", "pyannote/segmentation-3.0"),
                      ("embedding", "pyannote/wespeaker-voxceleb-resnet34-LM")]:
        source = asset(repo, "pytorch_model.bin")
        target = root / f"{key}.bin"
        shutil.copyfile(source, target)
        config["pipeline"]["params"][key] = target.name
    (root / "diarize.yaml").write_text(yaml.safe_dump(config))
    snapshot(whisper_repo, root / "whisper")
    snapshot("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", root / "align-zh")
    if "xtts-v2" in backends:
        snapshot("coqui/XTTS-v2", root / "xtts")
    if "chatterbox-multilingual-v3" in backends:
        chatterbox_snapshot(root / "chatterbox")
    # WhisperX 3.3.1 ships its VAD checkpoint in the wheel. Export and smoke-load it.
    if vad_checkpoint is None:
        import whisperx
        vad_checkpoint = Path(whisperx.__file__).parent / "assets/pytorch_model.bin"
    shutil.copyfile(vad_checkpoint, root / "whisperx-vad.bin")
    if not smoke_load:
        return
    from whisperx.vad import load_vad_model
    load_vad_model("cpu", model_fp=str(root / "whisperx-vad.bin"))
    # Smoke-load local configuration; build fails if any referenced checkpoint is absent.
    from pyannote.audio import Pipeline
    for key in ("segmentation", "embedding"):
        config["pipeline"]["params"][key] = str((root / f"{key}.bin").resolve())
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as local:
        yaml.safe_dump(config, local)
        local.flush()
        if Pipeline.from_pretrained(local.name) is None:
            raise RuntimeError("local diarization pipeline failed to load")
    if "chatterbox-multilingual-v3" in backends:
        # Load-only (no synthesis -- no reference audio exists at build time);
        # confirms the exported files are complete and loadable.
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        chatterbox_model = ChatterboxMultilingualTTS.from_local(root / "chatterbox", "cpu", t3_model="v3")
        if chatterbox_model.sr != 24000:
            raise RuntimeError(f"local chatterbox checkpoint has unexpected sample rate {chatterbox_model.sr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("VOICE_MODEL_DIR", "/opt/voice-models")))
    parser.add_argument("--nas-root", type=Path, help="use local repo snapshots without a token or network")
    parser.add_argument("--backends", default="xtts-v2,chatterbox-multilingual-v3",
                        help="comma list of clone backends to fetch weights for, "
                             "matching Dockerfile.voice's CLONE_BACKENDS_BUILD")
    parser.add_argument("--arch", help="amd64 or arm64 -- pass Dockerfile.voice's own "
                                       "$TARGETARCH; defaults to platform.machine() otherwise")
    arguments = parser.parse_args()
    requested_backends = frozenset(n.strip() for n in arguments.backends.split(",") if n.strip())
    prefetch(arguments.root, arguments.nas_root, backends=requested_backends, arch=arguments.arch)
