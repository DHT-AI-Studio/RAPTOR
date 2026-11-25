"""Bake hexgrad/Kokoro-82M-v1.1-zh's weights into the image at Docker build time.

Public repo, no token needed -- unlike module 35's prefetch_voice.py. Pinned to a
specific revision (bandit B615): an unpinned download silently follows whatever the
repo owner pushes next.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
_REVISION = "01e7505bd6a7a2ac4975463114c3a7650a9f7218"


def prefetch(root: Path) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(
        _REPO_ID, revision=_REVISION, local_dir=root,
        allow_patterns=["config.json", "*.pth", "voices/*.pt"],
    )
    if not (root / "config.json").is_file() or not list(root.glob("*.pth")):
        raise RuntimeError(f"Kokoro weights (config.json, *.pth) missing after download in {root}")
    if not list((root / "voices").glob("*.pt")):
        raise RuntimeError(f"Kokoro voices missing after download in {root / 'voices'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("RT_TTS_KOKORO_DIR", "/models/tts/kokoro")))
    prefetch(parser.parse_args().root)
