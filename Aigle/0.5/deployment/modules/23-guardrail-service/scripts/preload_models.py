#!/usr/bin/env python3
"""Download image-guard model weights to the local HF cache before starting Docker.

Usage:
    # Uses HF_CACHE_PATH and HF_TOKEN from .env (or current environment):
    python3 scripts/preload_models.py

    # Or set variables inline:
    HF_TOKEN=hf_xxx HF_CACHE_PATH=/data/hf python3 scripts/preload_models.py

Both models are gated — you must request access on huggingface.co before running.
"""
from __future__ import annotations

import logging
import os
import pathlib
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Pinned revisions — keep in sync with the service modules.
_MODELS = [
    ("meta-llama/Llama-Guard-3-11B-Vision", "62d4275543ec7503de66c486de1c0c2103e365ac"),
    ("google/shieldgemma-2-4b-it",          "eaf60452b5fc41a911338a022e628b0c15283897"),
]


def _load_dotenv() -> None:
    """Best-effort: load .env from the same directory as this script's parent."""
    env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def _snapshot_cached(cache_root: pathlib.Path, repo_id: str, revision: str) -> bool:
    safe_repo = repo_id.replace("/", "--")
    snapshot_dir = cache_root / "hub" / f"models--{safe_repo}" / "snapshots" / revision
    return snapshot_dir.is_dir() and any(snapshot_dir.iterdir())


def main() -> None:
    _load_dotenv()

    raw_cache_path = os.environ.get("HF_CACHE_PATH", "~/.cache/huggingface")
    cache_root = pathlib.Path(raw_cache_path).expanduser().resolve()

    if not cache_root.exists():
        log.info("HF_CACHE_PATH '%s' does not exist — creating it now.", cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        log.info("Created '%s'.", cache_root)
    else:
        log.info("HF_CACHE_PATH: %s", cache_root)

    hf_token = os.environ.get("HF_TOKEN") or None
    if not hf_token:
        log.warning(
            "HF_TOKEN is not set. Both models are gated — download will fail "
            "unless you have already authenticated via `huggingface-cli login`."
        )

    try:
        import huggingface_hub
    except ImportError:
        log.error(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )
        sys.exit(1)

    hub_cache = cache_root / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)

    for repo_id, revision in _MODELS:
        if _snapshot_cached(cache_root, repo_id, revision):
            log.info("cache hit  — %s @ %s", repo_id, revision[:8])
            continue

        log.info(
            "cache miss — %s @ %s  "
            "(downloading — Llama Guard 11B may take 10–30 min on first run)",
            repo_id, revision[:8],
        )
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=hf_token,
            cache_dir=str(hub_cache),
        )
        log.info("download complete — %s", repo_id)

    log.info("All models ready. You can now run: docker compose up")


if __name__ == "__main__":
    main()
