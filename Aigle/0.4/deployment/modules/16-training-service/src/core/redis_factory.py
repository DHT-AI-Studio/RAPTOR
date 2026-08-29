"""Redis client factory with automatic standalone/cluster detection.

Given a redis URL, probe the target node's ``cluster_enabled`` flag and return
the matching client — standalone ``Redis``/``AsyncRedis`` or
``RedisCluster``/``AsyncRedisCluster``. This lets the same code run against a
standalone redis (e.g. local dev) or a production cluster with **no config
change** and without hard-wiring a client type.

Reusable across modules: copy this file in and call ``make_redis_client`` /
``make_async_redis_client``.

NOTE: auto-detection only picks the right *client*. The calling code's redis
*usage* must still be cluster-safe (single-key operations, or multi-key
operations pinned to one hash slot via ``{tag}`` hash tags) to actually run on
a cluster — cross-slot MGET / pipelines / transactions fail under cluster mode.
"""
from __future__ import annotations

import logging
from typing import Optional

from redis import Redis
from redis.cluster import RedisCluster
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio import RedisCluster as AsyncRedisCluster

logger = logging.getLogger(__name__)


def probe_cluster_enabled(redis_url: str, password: Optional[str] = None) -> bool:
    """Return True if the redis node at ``redis_url`` runs with cluster mode on.

    Uses a short-lived standalone connection to read ``INFO`` — works against
    both standalone and cluster nodes (a cluster node reports
    ``cluster_enabled:1``). Kept synchronous so it can run in a constructor
    without an event loop.
    """
    probe = Redis.from_url(redis_url, password=password, decode_responses=True)
    try:
        return str(probe.info("cluster").get("cluster_enabled", 0)) == "1"
    finally:
        probe.close()


def make_redis_client(redis_url: str, password: Optional[str] = None,
                      cluster: Optional[bool] = None):
    """Return a sync redis client, auto-selecting standalone vs cluster.

    Pass ``cluster`` explicitly to skip the probe (e.g. when it was detected
    once already).
    """
    if cluster is None:
        cluster = probe_cluster_enabled(redis_url, password)
    if cluster:
        logger.info("Redis at %s → cluster mode (RedisCluster).", redis_url)
        return RedisCluster.from_url(redis_url, password=password, decode_responses=True)
    return Redis.from_url(redis_url, password=password, decode_responses=True)


def make_async_redis_client(redis_url: str, password: Optional[str] = None,
                            cluster: Optional[bool] = None, **kwargs):
    """Return an async redis client, auto-selecting standalone vs cluster.

    This is a plain (non-awaiting) constructor: ``from_url`` builds the client
    without connecting, so assigning the result is instant and cannot race with
    early requests. Pass ``cluster`` explicitly to skip the probe.
    """
    if cluster is None:
        cluster = probe_cluster_enabled(redis_url, password)
    if cluster:
        logger.info("Redis at %s → cluster mode (AsyncRedisCluster).", redis_url)
    Client = AsyncRedisCluster if cluster else AsyncRedis
    return Client.from_url(redis_url, password=password, decode_responses=True, **kwargs)
