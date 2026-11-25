"""Flat error envelope ``{"error": code, "message": ...}`` for Module 34's HTTP routes."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ServiceError(Exception):
    def __init__(self, status_code: int, error: str, message: str = "",
                 headers: dict[str, str] | None = None, **extra: Any) -> None:
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

    def response(self) -> JSONResponse:
        return JSONResponse(status_code=self.status_code, content=self.body(), headers=self.headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return exc.response()
