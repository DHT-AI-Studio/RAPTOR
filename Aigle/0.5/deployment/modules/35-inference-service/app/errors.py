"""Error envelope shared by every Module 35 route.

Bodies are flat, e.g. ``{"error": "model_not_found", "model": "...", "available": [...]}``,
because that is the shape the INF-3/INF-7 acceptance criteria specify.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError


class ServiceError(Exception):
    def __init__(
        self,
        status_code: int,
        error: str,
        message: str = "",
        headers: dict[str, str] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(message or error)
        self.status_code = status_code
        self.error = error
        self.message = message
        self.headers = headers
        self.extra = extra

    def body(self) -> dict[str, Any]:
        out: dict[str, Any] = {"error": self.error}
        if self.message:
            out["message"] = self.message
        out.update(self.extra)
        return out


def _validation_body(exc: RequestValidationError) -> dict[str, Any]:
    details = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", ()) if p != "body"]
        details.append({"field": ".".join(loc), "message": err.get("msg", "")})
    return {"error": "invalid_request", "message": "request validation failed", "details": details}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RedisError)
    async def _cache_error(_: Request, exc: RedisError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "redis_unavailable"})

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.body(), headers=exc.headers)

    # Missing / malformed request fields are 400 across Module 35 (INF-5, INF-8),
    # leaving 422 for "valid request, unknown model / limit exceeded".
    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content=_validation_body(exc))
