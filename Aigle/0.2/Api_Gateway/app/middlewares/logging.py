"""Logging middleware that emits structured JSON logs for every request."""
from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

_logger = logging.getLogger("gateway.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs request/response metadata in structured form."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        start_time = time.perf_counter()
        
        # 獲取或生成 correlation_id
        correlation_id = request.headers.get("X-Request-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
            # 將生成的 ID 存入 request.state，便於下游使用
            request.state.correlation_id = correlation_id
        
        log_context = {
            "request_method": request.method,
            "request_path": request.url.path,
            "correlation_id": correlation_id,
            "client_host": request.client.host if request.client else None,
        }

        try:
            response = await call_next(request)
            # 將 correlation_id 添加到回應 header
            response.headers["X-Request-ID"] = correlation_id
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            _logger.exception(
                "request_failed",
                extra={
                    **log_context,
                    "response_time_ms": round(duration_ms, 4),
                },
            )
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000
        return response

