from .cache_manager import CacheManager
from .cache_manager_dist_lock import CacheManager as CacheManagerDistLock

# global_cache_manager = None

# def get_cache_manager():
#     global global_cache_manager
#     if global_cache_manager is None:
#         global_cache_manager = CacheManager()
#     return global_cache_manager