# services/video_face_detection_service/model_files.py
# 啟動時確保兩個 ONNX 模型在 FACE_MODEL_DIR（預設在共用的 /hf_cache volume 底下），
# 和其他模組的 Hugging Face 模型一樣：第一次啟動時下載，之後重啟直接讀 volume 裡的檔案。
#
# 每個檔案都以 SHA-256 驗證：已存在且相符就不下載；不存在或不相符就重新下載。
# 下載先寫到同目錄的暫存檔，驗證通過才改名，中斷或下載到錯的檔案都不會留下壞檔。
# 下載失敗或驗證不符時丟出例外，服務不會用來源不明的模型啟動。

import hashlib
import logging
import os
import urllib.request
from collections.abc import Iterable

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: str, sha256: str, timeout: float) -> None:
    tmp = f"{dest}.part-{os.getpid()}"
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:  # nosec B310 -- URL from config, not user input
            for block in iter(lambda: resp.read(_CHUNK), b""):
                digest.update(block)
                out.write(block)
        if digest.hexdigest() != sha256:
            raise RuntimeError(f"{url} has SHA-256 {digest.hexdigest()}, expected {sha256}")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def ensure_models(files: Iterable[tuple[str, str, str]], timeout: float = 600.0) -> None:
    """files 為 (本機路徑, 下載網址, SHA-256)。缺少或不相符的檔案下載到本機路徑。"""
    for path, url, sha256 in files:
        if os.path.exists(path):
            if sha256_of(path) == sha256:
                logger.info(f"Face model present: {path}")
                continue
            logger.warning(f"Face model {path} does not match its SHA-256; downloading it again")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        logger.info(f"Downloading face model {url} -> {path}")
        _download(url, path, sha256, timeout)
        logger.info(f"Face model downloaded and verified: {path}")
