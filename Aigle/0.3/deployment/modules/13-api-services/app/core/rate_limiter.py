# app/core/rate_limiter.py
"""Redis sliding-window rate limiter for API key calls."""

from __future__ import annotations

import math
import time
import logging
from typing import Optional

from fastapi import HTTPException, status
from redis.asyncio import Redis

_logger = logging.getLogger(__name__)

# Key pattern:  rate_limit:{api_key}:{minute_bucket}
_KEY_TPL = "rate_limit:{key}:{bucket}"


async def check_rate_limit(
    redis: Redis,
    api_key: str,
    limit_per_minute: int,
) -> None:
    """
    Sliding-window (1-minute) rate limiter.

    Increments a counter keyed by (api_key, current-minute-bucket).
    Raises HTTP 429 if the counter exceeds `limit_per_minute`.
    """
    bucket = math.floor(time.time() / 60)
    key = _KEY_TPL.format(key=api_key, bucket=bucket)

    try:
        pipe = redis.pipeline()
        await pipe.incr(key)
        await pipe.expire(key, 90)  # keep alive for 1.5 minutes
        results = await pipe.execute()
        count = results[0]
    except Exception as exc:
        _logger.warning(f"Rate limiter Redis error: {exc} — allowing request")
        return  # fail open

    if count > limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "limit_per_minute": limit_per_minute,
                "current_count": count,
            },
            headers={"Retry-After": "60"},
        )
