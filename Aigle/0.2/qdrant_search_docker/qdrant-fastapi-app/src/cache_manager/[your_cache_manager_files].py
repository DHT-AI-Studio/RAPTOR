import os
import pickle
from typing import Any, Optional

class CacheManager:
    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_file_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def _is_cache_valid(self, cache_file: str) -> bool:
        if not os.path.exists(cache_file):
            return False
        if (os.path.getmtime(cache_file) + self.ttl) < os.path.getmtime(__file__):
            return False
        return True

    def get(self, key: str) -> Optional[Any]:
        cache_file = self._get_cache_file_path(key)
        if self._is_cache_valid(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def set(self, key: str, value: Any) -> None:
        cache_file = self._get_cache_file_path(key)
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)

    def clear(self, key: str) -> None:
        cache_file = self._get_cache_file_path(key)
        if os.path.exists(cache_file):
            os.remove(cache_file)

    def clear_all(self) -> None:
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)